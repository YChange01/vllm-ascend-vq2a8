# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only layout/ownership/routing tests, not CANN or speed acceptance."""

import hashlib
from collections import OrderedDict
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tests.ut.quantization.test_vq2a8_v4_device_route import no_host_tensor_reads
from tests.ut.quantization.test_vq2a8_v4_graph import FakeBackend, ObservedCompute
from vllm_ascend.quantization import vq2a8_execution as execution
from vllm_ascend.quantization import vq2a8_v4_device_route as route
from vllm_ascend.quantization import vq2a8_v4_v2 as v2
from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_optimization import configure_runtime
from vllm_ascend.quantization.vq2a8_repack import VQ2_DIRECT_TP1_FORMAT


def source_payload(k=2048):
    n = 4096
    generator = np.random.default_rng(19)
    words = generator.integers(0, 2**32, (n // 2, k // 8), dtype=np.uint32)
    ids = np.tile(np.arange(k // 256, dtype=np.uint8), 256)
    # Any finite FP8 pair is supported; these are not four scalar levels.
    books = generator.integers(0, 127, (k // 256, n // 32, 16, 2), dtype=np.uint8)
    payload = {
        "packed_indices": torch.from_numpy(words.view(np.int32)),
        "codebooks": torch.from_numpy(books).view(torch.float8_e4m3fn),
        "codebook_tile_ids": torch.from_numpy(ids),
        "weight_scale": torch.linspace(0.5, 1.5, k),
        "weight_bias": torch.linspace(-0.1, 0.1, k),
        "rht_sign": ((torch.arange(k) % 2) * 2 - 1).to(torch.int8),
    }
    spec = NS(rows=n, columns=k, rht_true_columns=k, rht_block_size=128)
    return payload, spec


def layer_header(payload, spec, *, experts=(0, 2), index=0):
    return NS(
        layer_index=index,
        expert_ids=experts,
        specs={"gate_up": spec, "down": spec},
        tensor_shapes={
            f"{kind}_{field}": (len(experts), *tensor.shape)
            for kind in ("gate_up", "down")
            for field, tensor in payload.items()
        },
    )


@pytest.mark.parametrize("k", [2048, 4096])
def test_conversion_preserves_arbitrary_pair_bytes_and_metadata(k):
    payload, spec = source_payload(k)
    original = {key: value.view(torch.uint8).clone() for key, value in payload.items()}
    converted = v2.convert_expert_payload(payload, spec)
    order = converted["activation_order"].numpy()
    expected_order = np.argsort(payload["codebook_tile_ids"].numpy(), kind="stable")
    np.testing.assert_array_equal(order, expected_order)
    words = payload["packed_indices"].numpy().view(np.uint32)
    old_codes = ((words[..., None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15).reshape(2048, k)
    packed = converted["packed_zn"].numpy()
    unpacked = np.stack((packed & 15, packed >> 4), axis=-1).reshape(128, k // 16, 16, 16)
    new_codes = unpacked.transpose(0, 3, 1, 2).reshape(2048, k)
    np.testing.assert_array_equal(new_codes, old_codes[:, order])
    lut = converted["pair_lut"].numpy().reshape(k // 256, 128, 16, 2)
    original_books = payload["codebooks"].view(torch.uint8).numpy()
    original_ids = payload["codebook_tile_ids"].numpy()
    # Decode selected rows independently from each representation, not by
    # merely repeating the packing routine's indexing expression.
    columns = np.arange(k)
    for row in (0, 1, 31, 32, 2037, 4095):
        expected = original_books[original_ids[order], row // 32, old_codes[row // 2, order], row % 2]
        actual = lut[columns // 256, row // 32, new_codes[row // 2], row % 2]
        np.testing.assert_array_equal(actual, expected)
    for field in ("weight_scale", "weight_bias", "rht_sign"):
        assert converted[field] is payload[field]  # original-K preparation
    for key, value in payload.items():
        assert torch.equal(value.view(torch.uint8), original[key])
    assert set(converted) == set(v2.V4_V2_FIELDS)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("id", "out of range"),
        ("population", "homogeneous"),
        ("nan", "NaN"),
        ("sign", "metadata"),
        ("geometry", "geometry"),
        ("field", "six"),
    ],
)
def test_conversion_rejects_unsupported_or_corrupt_payload(mutation, match):
    payload, spec = source_payload()
    if mutation == "id":
        payload["codebook_tile_ids"][0] = 255
    elif mutation == "population":
        payload["codebook_tile_ids"][0] = 1
    elif mutation == "nan":
        payload["codebooks"].view(torch.uint8)[0, 0, 0, 0] = 127
    elif mutation == "sign":
        payload["rht_sign"][0] = 0
    elif mutation == "geometry":
        spec.rows = 2048
    else:
        payload["extra"] = payload["rht_sign"]
    with pytest.raises(ValueError, match=match):
        v2.convert_expert_payload(payload, spec)


def test_plan_matches_only_converted_payload_and_both_metadata_banks():
    payload, spec = source_payload()
    layer = layer_header(payload, spec)
    converted = v2.convert_expert_payload(payload, spec)
    one_payload = sum(t.numel() * t.element_size() for t in converted.values())
    rounded_payload = sum(v2._rounded(t.numel() * t.element_size()) for t in converted.values())
    metadata = 2 * (v2._rounded(256 * 8) + v2._rounded(2 * 8) + 2 * v2._rounded(2 * 64))
    expected = 2 * 2 * rounded_payload + metadata
    plan = v2.v4_v2_resident_plan([layer], expected)
    assert plan["full_packed_bytes"] == plan["planned_bytes"] == expected
    assert plan["layer_plans"][0]["payload_bytes"] == 2 * 2 * one_payload
    assert plan["layer_plans"][0]["metadata_reserve_bytes"] == metadata
    assert plan["layout"] == "v2_zn_pair_lut"
    with pytest.raises(ValueError, match="no fallback"):
        v2.v4_v2_resident_plan([layer], expected - 1)


@pytest.mark.parametrize("ids", [(), (0, 0), (-1,), (256,), (True,)])
def test_plan_rejects_bad_inventory(ids):
    payload, spec = source_payload()
    with pytest.raises(ValueError, match="expert IDs"):
        v2.v4_v2_resident_plan([layer_header(payload, spec, experts=ids)], 10**10)


def test_preload_converts_before_single_h2d_and_has_no_original_payload(monkeypatch):
    payload, spec = source_payload()
    layer = layer_header(payload, spec, experts=(0,))
    reads = []

    def load(index, expert, kind, **kwargs):
        assert kwargs["device"] == "cpu"
        reads.append((index, expert, kind))
        return payload, spec

    artifact = NS(manifest={"format": VQ2_DIRECT_TP1_FORMAT}, load_expert=load)

    def init(self, artifact, index, device, **kwargs):
        self.artifact, self.layer, self.layer_index, self.device = artifact, layer, index, torch.device(device)
        self._cache = OrderedDict()
        self.cache_experts = 1
        self.cache_loads = self.cache_hits = self.cache_peak_bytes = 0
        self._resident_bytes = self.evictions = self.h2d_bytes = 0
        self.progress = self.verbose_experts = self.measurement_mode = False
        self.timing = dict.fromkeys(("host_load_validate_s", "host_read_s", "host_validate_s", "h2d_s"), 0.0)

    monkeypatch.setattr(execution.AscendCVQ2TP1MoE, "__init__", init)
    runtime = v2.AscendCV4V2VQ2TP1MoE(artifact, 0, "cpu")
    built = []

    def bank_factory():
        built.append(tuple(runtime._cache))
        return {"metadata_bytes": 32}

    monkeypatch.setattr(runtime, "_create_device_route_banks", bank_factory)
    plan = v2.v4_v2_resident_plan([layer], 10**10)
    report = runtime.initialize_resident(budget_bytes=plan["planned_bytes"])
    assert reads == [(0, 0, "gate_up"), (0, 0, "down")]
    assert built == [(0,)]
    assert report["preload_loads"] == 1
    assert report["post_init_loads"] == report["post_init_h2d_bytes"] == 0
    assert report["dual_payload_residency"] is False
    for cached, _ in runtime._cache[0].values():
        assert set(cached) == set(v2.V4_V2_FIELDS)
        assert "packed_indices" not in cached
    runtime.check_resident_integrity()
    runtime._cache[0]["gate_up"][0]["activation_order"] = torch.arange(spec.columns)
    with pytest.raises(RuntimeError, match="storage changed"):
        runtime.check_resident_integrity()


class CpuBank:
    """Arithmetic oracle only. Native gathers AFTER the original preparation."""

    def __init__(self, count, n, k, generator):
        self.scale = torch.rand(count, k, generator=generator) + 0.1
        self.bias = torch.randn(count, k, generator=generator) * 0.01
        self.sign = (torch.randint(0, 2, (count, k), generator=generator) * 2 - 1).to(torch.int8)
        self.weight = torch.randn(count, n, k, generator=generator).to(torch.float8_e4m3fn).float()
        self.calls = []

    def select(self, slots):
        valid = (slots >= 0) & (slots < self.scale.shape[0])
        safe = slots.clamp(0, self.scale.shape[0] - 1)
        return [self.scale[safe], self.bias[safe], self.sign[safe], valid.int()]

    def project(self, q, scale, bias, slots):
        self.calls.append((q.shape, scale.shape, bias.shape, slots.clone()))
        valid = (slots >= 0) & (slots < self.scale.shape[0])
        weight = self.weight[slots.clamp(0, self.scale.shape[0] - 1)]
        if q.ndim == 2:
            output = torch.bmm(q.float().unsqueeze(1), weight.transpose(1, 2)).squeeze(1)
        else:
            output = torch.bmm(q.float(), weight.transpose(1, 2))
        output = (output * scale[..., None] + bias[..., None]).bfloat16()
        mask = valid.reshape((slots.numel(),) + (1,) * (output.ndim - 1))
        return [torch.where(mask, output, float("nan")), valid.int()]


def fake_runtime():
    runtime = object.__new__(v2.AscendCV4V2VQ2TP1MoE)
    count, width = 6, 16
    generator = torch.Generator().manual_seed(7)
    runtime.device = torch.device("cpu")
    runtime._resident_ready, runtime._resident_failed = True, False
    runtime.config = NS(
        top_k=6,
        hidden_size=width,
        num_experts=count,
        renormalize=True,
        num_shared=0,
        swiglu_limit=7.0,
        routed_scale=1.0,
    )
    runtime.root = {
        "gate.weight": torch.randn(count, width, generator=generator),
        "gate.tid2eid": torch.tensor([[1, 1, 5, 0, 2, 3], [3, 0, 4, 1, 5, 5]]),
    }
    runtime.layer = NS(expert_ids=tuple(range(count)))
    runtime.cache_experts, runtime.token_chunk = count, 2
    runtime._cache = {}
    runtime._v2_payload_locations = {}
    runtime._device_route_banks = {"lookup": torch.arange(count), "slots": torch.arange(count)}
    for kind in ("gate_up", "down"):
        bank = CpuBank(count, width * 2 if kind == "gate_up" else width, width, generator)
        spec = NS(
            rows=width * 2 if kind == "gate_up" else width, columns=width, rht_true_columns=width, rht_block_size=8
        )
        runtime._device_route_banks[kind] = (bank, spec)
        for slot in range(count):
            payload = {"weight_scale": bank.scale[slot], "weight_bias": bank.bias[slot], "rht_sign": bank.sign[slot]}
            runtime._cache.setdefault(slot, {})[kind] = (payload, spec)
            runtime._v2_payload_locations[id(payload)] = (kind, slot)
    runtime._get_expert = lambda slot: runtime._cache[slot]
    runtime._row_preparation = RowwiseVQ2A8Preparation(compact=True)
    runtime._optimization = None
    runtime._timing_sync = lambda: None
    runtime.timing = {"prepare_s": 0.0, "packed_projection_s": 0.0}
    runtime.native_calls = runtime.native_launches = runtime.native_rows = runtime.native_experts = 0
    runtime.projection_rows = runtime.prepare_batches = 0
    return runtime


def test_prefill_mixed_rows_uses_same_bank_and_preserves_real_preparation(monkeypatch):
    runtime = fake_runtime()
    bank, _ = runtime._device_route_banks["gate_up"]
    requests = [
        (torch.randn(rows, 16).bfloat16(), *runtime._cache[slot]["gate_up"]) for rows, slot in ((1, 4), (3, 1), (2, 4))
    ]
    preparation = RowwiseVQ2A8Preparation(compact=True)
    expected = []
    for values, slot in zip(preparation.many(requests), (4, 1, 4)):
        q, scale, bias = values
        expected.append(bank.project(q[None], scale[None], bias[None], torch.tensor([slot]))[0][0])
    actual = runtime._projections_many(requests)
    assert [tuple(t.shape) for t in actual] == [(1, 32), (3, 32), (2, 32)]
    for left, right in zip(actual, expected):
        assert torch.equal(left, right)
    assert bank.calls[-1][:3] == (torch.Size([3, 3, 16]), torch.Size([3, 3]), torch.Size([3, 3]))
    assert torch.equal(bank.calls[-1][3], torch.tensor([4, 1, 4]))
    assert runtime.native_launches == 1 and runtime.native_rows == 6


def test_prefill_and_singleton_graph_compute_share_v2_banks(monkeypatch):
    runtime = fake_runtime()
    configure_runtime(runtime, "device_route_decode")
    compute = route.DeviceRouteGraphCompute(runtime)
    hidden = torch.randn(1, 16).bfloat16()
    for token in (0, 1, 0):
        tokens = torch.tensor([token])
        with no_host_tensor_reads(monkeypatch):
            eager = runtime._optimization.forward(runtime, hidden, tokens)
            captured_math, valid = compute(hidden, tokens)
        assert torch.equal(eager, captured_math)
        assert bool(valid)
    assert compute.signature["compute_backend"] == "v2"
    assert compute.banks["gate_up"][0] is runtime._device_route_banks["gate_up"][0]
    # Multi-token prefill reaches the candidate override, not the old grouped op.
    result = runtime._optimization.forward(runtime, hidden.expand(2, -1), torch.tensor([0, 1]))
    assert result.shape == (2, 16)
    assert any(len(call[0]) == 3 for call in runtime._device_route_banks["gate_up"][0].calls)
    runtime.v4_compute_backend = "v1"
    with pytest.raises(RuntimeError, match="signature changed"):
        compute.check_runtime_contract(runtime)


def test_bank_factory_hook_preserves_current_v4_protocol():
    expected = {"metadata_bytes": 123}
    runtime = NS(_require_ready=lambda: None, _create_device_route_banks=lambda: expected)
    assert route.create_device_route_banks(runtime) is expected


@pytest.mark.parametrize("policy", ["owner", "caller"])
def test_candidate_graph_protocol_keeps_banks_and_replays_changed_routes(monkeypatch, policy):
    runtime = fake_runtime()
    configure_runtime(runtime, "device_route_decode")
    state, backend = runtime._optimization, FakeBackend()
    eager_banks = runtime._device_route_banks
    graph_banks = {**eager_banks, "metadata_bytes": 1024}
    real_compute = route.DeviceRouteGraphCompute

    def wrapped_compute(current, *, banks=None):
        compute = real_compute(current, banks=banks)
        observed = ObservedCompute(backend, compute)
        observed.signature = compute.signature
        observed.check_runtime_contract = compute.check_runtime_contract
        return observed

    monkeypatch.setattr(route, "DeviceRouteGraphCompute", wrapped_compute)
    monkeypatch.setattr(runtime, "_create_device_route_banks", lambda: graph_banks)
    state.prepare_graph(runtime, backend=backend, replay_stream_policy=policy)
    before = runtime.native_calls
    hidden = torch.randn(1, 16).bfloat16()
    expected_compute = real_compute(runtime)
    retained_outputs = []
    for token in (0, 1, 0):
        tokens = torch.tensor([token])
        expected, _ = expected_compute(hidden, tokens)
        with no_host_tensor_reads(monkeypatch):
            output = state.forward_graph(runtime, hidden, tokens)
        assert torch.equal(output, expected)
        assert bool(state.valid)
        retained_outputs.append((output, output.clone()))
    assert runtime.native_calls == before
    assert state.graph_snapshot()["replays"] == 3
    assert runtime._device_route_banks is eager_banks
    assert state._graph_banks is graph_banks
    assert state.graph_snapshot()["graph_payload_copy_bytes"] == 0
    assert all(torch.equal(output, snapshot) for output, snapshot in retained_outputs)
    state.close_graph()


def test_loader_rejects_other_backend_filename_before_loading(tmp_path):
    path = tmp_path / "libvq2a8_ascendc.so"
    path.write_bytes(b"not an extension")
    with pytest.raises(ValueError, match="not the V1/v2/v3"):
        v2.load_v4_v2_library(path, hashlib.sha256(path.read_bytes()).hexdigest())
