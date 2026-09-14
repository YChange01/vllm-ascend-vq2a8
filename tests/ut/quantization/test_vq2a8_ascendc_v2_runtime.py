# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_ascend.quantization import vq2a8_ascendc_v2 as v2
from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_execution import CachedVQ2TP1MoE, packed_cache_plan
from vllm_ascend.quantization.vq2a8_optimization import OptimizationOptions

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def bridge():
    spec = importlib.util.spec_from_file_location(
        "_vq2_expert_runtime_test_bridge", REPO / "csrc/vq2a8_expert_reference/scripts/vq2_bridge.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def payload(k=2048, *, columns=None):
    rng = np.random.default_rng(1701)
    n = 4096
    codes = rng.integers(0, 256, size=(n // 2, k // 8, 4), dtype=np.uint8)
    words = np.ascontiguousarray(codes).view(np.int32).reshape(n // 2, k // 8)
    book_bytes = rng.integers(0, 256, size=(k // 256, n // 32, 16, 2), dtype=np.uint8)
    book_bytes[(book_bytes & 127) == 127] = 128  # Retain signed zero; reject only NaNs.
    ids = np.repeat(np.arange(k // 256, dtype=np.uint8), 256)
    rng.shuffle(ids)
    weights = {
        "packed_indices": torch.from_numpy(words),
        "codebooks": torch.from_numpy(book_bytes).view(torch.float8_e4m3fn),
        "codebook_tile_ids": torch.from_numpy(ids),
        "weight_scale": torch.linspace(0.1, 1, k, dtype=torch.float32),
        "weight_bias": torch.linspace(-0.2, 0.3, k, dtype=torch.float32),
        "rht_sign": torch.where(torch.arange(k) % 2 == 0, 1, -1).to(torch.int8),
    }
    return weights, NS(columns=k, rht_true_columns=columns or k, rht_block_size=128)


@pytest.mark.parametrize("k", [2048, 4096])
def test_v2_runtime_conversion_matches_independent_byte_bridge(bridge, k):
    weights, spec = payload(k)
    before = {name: value.view(torch.uint8).clone() for name, value in weights.items()}
    expected = bridge.bridge_direct_weights(
        weights["packed_indices"].numpy(),
        weights["codebooks"].view(torch.uint8).numpy(),
        weights["codebook_tile_ids"].numpy(),
        k_order="codebook",
    )
    converted = v2.convert_expert_payload(weights, spec)
    np.testing.assert_array_equal(converted["packed_zn"].numpy(), expected.packed_zn)
    np.testing.assert_array_equal(converted["pair_lut"].numpy(), expected.fixed_k256_lut)
    np.testing.assert_array_equal(converted["activation_order"].numpy(), expected.activation_gather)
    assert set(converted) == {"packed_zn", "pair_lut", "activation_order", "weight_scale", "weight_bias", "rht_sign"}
    assert converted["packed_zn"].numel() == weights["packed_indices"].numel() * 4
    assert torch.unique(converted["pair_lut"][0, 0]).numel() > 4  # No scalar-W2 level fitting.
    assert (converted["pair_lut"] == 128).any()  # Preserve negative zero bytes.
    for name, value in weights.items():
        assert torch.equal(value.view(torch.uint8), before[name])
    for name in ("weight_scale", "weight_bias", "rht_sign"):
        assert converted[name] is weights[name]  # Original K order, no transformations here.


@pytest.mark.parametrize(
    "bad",
    ["field", "dtype", "shape", "nan", "tile_range", "tile_population", "sign", "scale", "spec", "unsupported"],
)
def test_invalid_weight_contract_is_rejected_without_lossy_fallback(bad):
    weights, spec = payload()
    if bad == "field":
        weights["dense_weight"] = torch.zeros(1)
    elif bad == "dtype":
        weights["packed_indices"] = weights["packed_indices"].to(torch.int64)
    elif bad == "shape":
        weights["codebooks"] = weights["codebooks"][:, :-1]
    elif bad == "nan":
        weights["codebooks"].view(torch.uint8)[0, 0, 0, 0] = 127
    elif bad == "tile_range":
        weights["codebook_tile_ids"][0] = 255
    elif bad == "tile_population":
        ids = weights["codebook_tile_ids"]
        ids[0] = (int(ids[0]) + 1) % 8
    elif bad == "sign":
        weights["rht_sign"][0] = 0
    elif bad == "scale":
        weights["weight_scale"][0] = float("nan")
    elif bad == "spec":
        spec.columns += 1
    elif bad == "unsupported":
        weights["packed_indices"] = weights["packed_indices"][:512].contiguous()
    with pytest.raises(ValueError):
        v2.convert_expert_payload(weights, spec)


def header(count=256):
    gate, gate_spec = payload(4096)
    down, down_spec = payload(2048)
    return NS(
        layer_index=3,
        expert_ids=tuple(range(count)),
        specs={"gate_up": gate_spec, "down": down_spec},
        tensor_shapes={
            f"{kind}_{name}": (count, *value.shape)
            for kind, weights in (("gate_up", gate), ("down", down))
            for name, value in weights.items()
        },
    )


def test_cache_plan_counts_replacement_layout_and_activation_order():
    layer = header()
    original = packed_cache_plan([layer], 1 << 40)
    planned = v2.ascendc_v2_cache_plan([layer], 1 << 40)
    # Same packed and LUT bytes here; uint8[K] tile IDs replaced by int64[K].
    assert planned["planned_bytes"] - original["planned_bytes"] == 256 * (4096 + 2048) * 7
    one = 0
    for k in (4096, 2048):
        weights, spec = payload(k)
        converted = v2.convert_expert_payload(weights, spec)
        one += sum(((v.numel() * v.element_size() + 511) // 512) * 512 for v in converted.values())
    assert v2.expert_cached_bytes(layer) == one
    assert v2.ascendc_v2_cache_plan([layer], one * 3)["layer_limits"] == {3: 3}
    with pytest.raises(ValueError, match="cannot retain"):
        v2.ascendc_v2_cache_plan([layer], one - 1)


def test_cache_header_rejects_unsupported_geometry_before_loading():
    layer = header()
    layer.tensor_shapes["down_packed_indices"] = (256, 512, 256)
    with pytest.raises(ValueError, match="supports only"):
        v2.ascendc_v2_cache_plan([layer], 1 << 40)


def cache_runtime(limit=1):
    runtime = v2.AscendCV2VQ2TP1MoE.__new__(v2.AscendCV2VQ2TP1MoE)
    runtime.device = torch.device("cpu")
    runtime.layer_index = 3
    runtime.layer = NS(expert_ids=(0, 1))
    runtime._cache = OrderedDict()
    runtime.cache_experts = limit
    runtime.cache_hits = runtime.cache_loads = runtime.cache_peak_bytes = 0
    runtime._resident_bytes = runtime.evictions = 0
    runtime.progress = runtime.verbose_experts = runtime.measurement_mode = False
    runtime.prepare_batches = runtime.projection_rows = 0
    runtime.timing = dict.fromkeys(
        ("host_load_validate_s", "host_read_s", "host_validate_s", "h2d_s", "prepare_s", "packed_projection_s"), 0.0
    )
    runtime.reset_native_trace()
    return runtime


def test_cpu_conversion_occurs_once_per_cache_miss_not_per_projection(monkeypatch):
    runtime = cache_runtime()
    weights, spec = payload()
    reads, conversions = [], []
    original = v2.convert_expert_payload

    def load(layer, index, kind, **kwargs):
        assert kwargs["device"] == "cpu" and "timings" in kwargs
        reads.append((index, kind))
        return weights, spec

    def convert(values, shape):
        assert all(value.device.type == "cpu" for value in values.values())
        conversions.append(1)
        return original(values, shape)

    monkeypatch.setattr(v2, "convert_expert_payload", convert)
    runtime.artifact = NS(load_expert=load)
    first = runtime._get_expert(0)
    assert runtime._get_expert(0) is first
    assert len(reads) == len(conversions) == 2  # Two matrices on one v2 miss.
    expected = sum(t.numel() * t.element_size() for p, _ in first.values() for t in p.values())
    assert runtime._resident_bytes == expected
    assert all("packed_indices" not in p and "codebooks" not in p for p, _ in first.values())
    del first
    runtime._get_expert(1)
    assert len(conversions) == 4 and runtime.evictions == 1
    runtime._get_expert(0)
    assert len(conversions) == 6 and runtime.evictions == 2
    assert runtime.cache_hits == 1 and runtime.cache_loads == 3


def test_failed_conversion_never_publishes_a_partial_device_cache():
    runtime = cache_runtime()
    weights, spec = payload()
    weights["codebook_tile_ids"][0] = 255
    runtime.artifact = NS(load_expert=lambda *args, **kwargs: (weights, spec))
    with pytest.raises(ValueError, match="out of range"):
        runtime._get_expert(0)
    assert not runtime._cache and runtime.cache_loads == runtime._resident_bytes == 0


def test_original_backend_host_hook_remains_identity():
    runtime = CachedVQ2TP1MoE.__new__(CachedVQ2TP1MoE)
    host = {"gate_up": (object(), object())}
    assert runtime._prepare_host_expert(host) is host


@pytest.mark.parametrize("rows,columns", [(1, 2048), (3, 2048), (2, 2000)])
def test_native_dispatch_gathers_only_after_original_fp8_quantization(monkeypatch, rows, columns):
    runtime = cache_runtime()
    runtime.device = NS(type="npu")  # CPU math + capturing ABI stand-in, not an NPU test.
    runtime._timing_sync = lambda: None
    weights, spec = payload(columns=columns)
    converted = v2.convert_expert_payload(weights, spec)
    hidden = (torch.arange(rows * columns).reshape(rows, columns) % 19 - 9).to(torch.bfloat16) / 16
    reference = RowwiseVQ2A8Preparation().many([(hidden, weights, spec)])[0]
    calls = []

    def capture(inputs):
        calls.append(inputs)
        assert len(inputs) == 1
        q, scale, bias, packed, lut = inputs[0]
        expected_q = reference[0].view(torch.uint8).index_select(1, converted["activation_order"])
        assert torch.equal(q.view(torch.uint8), expected_q)
        torch.testing.assert_close(scale, reference[1], atol=0, rtol=0)
        torch.testing.assert_close(bias, reference[2], atol=0, rtol=0)
        assert packed is converted["packed_zn"] and lut is converted["pair_lut"]
        return [torch.zeros(rows, 4096, dtype=torch.bfloat16)]

    monkeypatch.setattr(v2, "grouped_projection", capture)
    output = runtime._projections_many([(hidden, converted, spec)])
    assert output[0].shape == (rows, 4096)
    assert len(calls) == runtime.native_launches == runtime.native_calls == 1
    assert runtime.native_rows == runtime.projection_rows == runtime.prepare_batches == rows


@pytest.mark.parametrize("preset", ["pipeline", "fwht", "prepare_graph"])
def test_v2_rejects_other_kernel_and_quantization_presets_before_dispatch(preset):
    runtime = cache_runtime()
    runtime.device = NS(type="npu")
    runtime._optimization = NS(options=OptimizationOptions.preset(preset))
    with pytest.raises(ValueError, match="only rowwise"):
        runtime._projections_many([])


def test_v2_never_executes_cpu_fallback():
    runtime = cache_runtime()
    with pytest.raises(ValueError, match="no CPU/CUDA fallback"):
        runtime._projections_many([])


@pytest.mark.parametrize("dtype,shape", [(torch.float32, (1, 2048)), (torch.float8_e4m3fn, (2048,))])
def test_invalid_activation_gather_metadata(dtype, shape):
    with pytest.raises(ValueError, match="gather requires"):
        v2.gather_prepared_activation(torch.zeros(shape, dtype=dtype), torch.arange(2048))


def manifest_files(tmp_path, monkeypatch):
    monkeypatch.setattr(v2, "__file__", str(tmp_path / "package/quantization/vq2a8_ascendc_v2.py"))
    source_root = tmp_path
    hashes = {}
    for relative in v2.ASCENDC_V2_REQUIRED_SOURCES:
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        hashes[relative] = hashlib.sha256(relative.encode()).hexdigest()
    library = tmp_path / "build/libvq2a8_ascendc_v2.so"
    library.parent.mkdir()
    library.write_bytes(b"test-only-not-executable")
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    manifest = {
        "status": "built",
        "implementation": "ascendc_v2",
        "abi_version": 1,
        "library": str(library),
        "library_sha256": digest,
        "source_sha256": hashes,
    }
    (library.parent / "build-manifest.json").write_text(json.dumps(manifest))
    return library, digest, manifest, source_root


def test_hash_pinned_manifest_validation_does_not_load_device(tmp_path, monkeypatch):
    library, digest, _, _ = manifest_files(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.ops, "load_library", lambda path: pytest.fail("validation must not load library"))
    assert v2.validate_build_manifest(library, digest) == {
        "path": str(library.resolve()),
        "sha256": digest,
        "abi_version": 1,
    }


@pytest.mark.parametrize("bad", ["binary", "abi", "implementation", "status", "missing_source", "source", "traversal"])
def test_manifest_rejects_stale_or_mismatched_binary_and_sources(tmp_path, monkeypatch, bad):
    library, digest, manifest, source_root = manifest_files(tmp_path, monkeypatch)
    if bad == "binary":
        library.write_bytes(b"changed")
    elif bad == "abi":
        manifest["abi_version"] = 2
    elif bad == "implementation":
        manifest["implementation"] = "ascendc"
    elif bad == "status":
        manifest["status"] = "failed"
    elif bad == "missing_source":
        del manifest["source_sha256"]["csrc/vq2a8_ascendc_v2/kernel.cpp"]
    elif bad == "source":
        (source_root / "csrc/vq2a8_ascendc_v2/kernel.cpp").write_bytes(b"changed")
    elif bad == "traversal":
        manifest["source_sha256"]["../outside.cpp"] = "0" * 64
    (library.parent / "build-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        v2.validate_build_manifest(library, digest)


def test_distinct_operator_namespace_and_five_tensor_abi(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.ops, "vq2a8_ascendc_v2", NS(grouped_projection=lambda *args: calls.append(args), abi_version=lambda: 1)
    )
    monkeypatch.setattr(
        torch.ops, "vq2a8_ascendc", NS(grouped_projection=lambda *args: pytest.fail("old kernel called"))
    )
    values = [tuple(object() for _ in range(5)) for _ in range(3)]
    v2.grouped_projection(values)
    assert calls == [tuple(list(items) for items in zip(*values))]
    with pytest.raises(ValueError, match="five-tensor"):
        v2.grouped_projection([tuple(object() for _ in range(6))])


@pytest.mark.parametrize("abi", [0, 2, True])
def test_v2_rejects_incompatible_native_abi_before_launch(monkeypatch, abi):
    monkeypatch.setattr(
        torch.ops,
        "vq2a8_ascendc_v2",
        NS(grouped_projection=lambda *args: pytest.fail("must not launch incompatible ABI"), abi_version=lambda: abi),
    )
    with pytest.raises(RuntimeError, match="incompatible tensor ABI"):
        v2.grouped_projection([tuple(object() for _ in range(5))])
