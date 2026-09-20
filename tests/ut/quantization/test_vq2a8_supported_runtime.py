# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the supported path; these do not claim NPU acceptance."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.quantization.vq2a8_activation_packed import PackedRowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_config import SUPPORTED_MODES, resolve_runtime_options
from vllm_ascend.quantization.vq2a8_offline import validate_offline_config
from vllm_ascend.quantization.vq2a8_runtime_guard import PlannedRuntimeGuard
from vllm_ascend.quantization.vq2a8_v4_v2 import (
    installed_v4_v2_library_path,
    load_v4_v2_library,
    require_v4_v2_features,
)


def test_defaults_are_explicit_accepted_configuration():
    original = {"enabled": True, "artifact": "/checkpoint/experts"}
    resolved = resolve_runtime_options(original)
    assert resolved.items() >= SUPPORTED_MODES.items()
    assert original == {"enabled": True, "artifact": "/checkpoint/experts"}


def valid_engine_config():
    return SimpleNamespace(
        additional_config={"vq2a8_offline": {"enabled": True, "artifact": "/experts"}},
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        model_config=SimpleNamespace(enforce_eager=True, quantization=None, dtype=torch.bfloat16, max_model_len=16),
        quant_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=16, enable_chunked_prefill=False),
        compilation_config=SimpleNamespace(mode=0, cudagraph_mode=0),
        cache_config=SimpleNamespace(block_size=128, kv_cache_memory_bytes=1024**3),
        load_config=SimpleNamespace(load_format="safetensors"),
    )


def test_standard_engine_config_resolves_accepted_defaults():
    result = validate_offline_config(valid_engine_config())
    assert result["v4_decoder_metadata_mode"] == "position_template"
    assert result["v4_runtime_guard"] == "planned"


@pytest.mark.parametrize(
    "section, key, value",
    [
        ("parallel_config", "tensor_parallel_size", 2),
        ("cache_config", "block_size", 64),
        ("cache_config", "kv_cache_memory_bytes", None),
        ("scheduler_config", "enable_chunked_prefill", True),
        ("scheduler_config", "max_num_batched_tokens", 32),
        ("scheduler_config", "max_num_batched_tokens", 0),
        ("scheduler_config", "max_num_batched_tokens", 8),
        ("model_config", "max_model_len", 17),
        ("model_config", "max_model_len", True),
        ("additional_config", "mix_placement", True),
        ("additional_config", "enable_flashcomm1", True),
        ("additional_config", "multistream_dsv4_dsa_overlap", True),
        ("additional_config", "finegrained_tp_config", {"oproj_tensor_parallel_size": 2}),
    ],
)
def test_unsupported_standard_engine_settings_rejected(section, key, value):
    config = valid_engine_config()
    target = getattr(config, section)
    if isinstance(target, dict):
        target[key] = value
    else:
        setattr(target, key, value)
    with pytest.raises(ValueError):
        validate_offline_config(config)


@pytest.mark.parametrize(
    "key, value",
    [
        ("execution_policy", "ascendc"),
        ("execution_policy", "ascendc_v3"),
        ("root_linear_mode", "online_fp8_sm90"),
        ("v4_compute_backend", "v1"),
        ("v4_activation_reorder", "row_reuse"),
        ("v4_activation_reorder", "chunk_reuse2"),
        ("v4_activation_reorder", "chunk_reuse4"),
        ("v4_activation_tail", "fused_reorder"),
        ("v4_b1_schedule", "tile_major"),
        ("v4_swiglu_mode", "fused_select_sign"),
        ("v4_runtime_guard", "native"),
        ("v4_runtime_guard", "signature"),
        ("v4_decoder_input_mode", "b1_packed"),
        ("v4_select_sign", "separate"),
        ("v4_decode_graph", "moe"),
        ("v4_graph_replay_stream", "owner"),
        ("v4_validity_mode", "fused"),
        ("v4_route_mapping", "torch"),
        ("v4_serving", 1),
        ("v4_device_route_decode", False),
        ("v4_host_profile", True),
    ],
)
def test_removed_modes_fail_instead_of_falling_back(key, value):
    with pytest.raises(ValueError, match=key):
        resolve_runtime_options({"enabled": True, "artifact": "/experts", key: value})


@pytest.mark.parametrize("key", ["v3_preparation", "ascendc_v2_library", "v3_startup_trace", "typo"])
def test_unknown_options_are_rejected(key):
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_runtime_options({"enabled": True, "artifact": "/experts", key: "unused"})


def test_installed_library_lookup_is_package_relative(tmp_path):
    library = tmp_path / "libvq2a8_ascendc_v4_v2.so"
    with pytest.raises(RuntimeError, match="not installed"):
        installed_v4_v2_library_path(tmp_path)
    library.touch()
    assert installed_v4_v2_library_path(tmp_path) == library.resolve()


def test_explicit_library_must_be_pinned(tmp_path):
    library = tmp_path / "libvq2a8_ascendc_v4_v2.so"
    library.touch()
    with pytest.raises(ValueError, match="SHA256"):
        load_v4_v2_library(library)
    with pytest.raises(ValueError, match="does not match"):
        load_v4_v2_library(library, "0" * 64)


def test_feature_gate_requires_only_retained_abis():
    native = SimpleNamespace(
        activation_reorder_version=lambda: 1,
        layer_validity_vectorized_version=lambda: 1,
        route_mapping_version=lambda: 1,
        select_sign_version=lambda: 1,
        route_mapping=lambda ids, lookup: None,
    )
    require_v4_v2_features(native_ops=native)
    native.select_sign_version = lambda: 2
    with pytest.raises(RuntimeError, match="select_sign_version"):
        require_v4_v2_features(native_ops=native)


def test_planned_guard_checks_current_metadata_without_caching_pass():
    config = SimpleNamespace(
        top_k=6, hidden_size=4096, num_shared=1, renormalize=True, routed_scale=1.0, swiglu_limit=10.0
    )
    spec = SimpleNamespace(rows=4096, columns=2048, rht_true_columns=2048, rht_block_size=128)
    runtime = SimpleNamespace(
        **SUPPORTED_MODES,
        config=config,
        root={"gate.weight": torch.zeros(2, 2)},
        _device_route_banks={
            "gate_up": (object(), spec),
            "down": (object(), spec),
            "lookup": torch.zeros(256, dtype=torch.int64),
        },
    )
    guard = PlannedRuntimeGuard(runtime)
    guard.check(runtime)
    runtime.v4_runtime_guard = "native"
    with pytest.raises(RuntimeError, match="v4_runtime_guard"):
        guard.check(runtime)
    runtime.v4_runtime_guard = "planned"
    guard.check(runtime)
    runtime.root["gate.weight"] = runtime.root["gate.weight"].clone()
    with pytest.raises(RuntimeError, match="root.gate.weight"):
        guard.check(runtime)


@pytest.mark.parametrize("width", [2048, 4096])
def test_rowwise_torch_tail_retains_exact_operation_order(width):
    preparation = PackedRowwiseVQ2A8Preparation(native_ops=SimpleNamespace(select_sign_version=lambda: 1))
    spec = SimpleNamespace(rht_block_size=128)
    preparation.prepare_for_graph(torch.device("cpu"), 128)
    generator = torch.Generator().manual_seed(17)
    signed = torch.randn(6, width, generator=generator)
    weights = torch.rand(6, width, generator=generator) + 0.5
    weight_bias = torch.randn(6, width, generator=generator)
    actual_q, actual_scale, actual_bias = preparation._from_signed(signed, weights, weight_bias, spec)
    rotated = torch.empty_like(signed)
    expected_bias = torch.empty(6)
    for row in range(6):
        destination = rotated[row : row + 1]
        torch.matmul(
            signed.reshape(6, width // 128, 128)[row : row + 1],
            preparation._hadamard,
            out=destination.reshape(1, width // 128, 128),
        )
        torch.matmul(destination, weight_bias[row], out=expected_bias[row : row + 1])
    transformed = rotated * weights
    maximum = torch.finfo(torch.float8_e4m3fn).max
    from vllm_ascend.quantization.vq2a8_reference import VQ2_FP8_MIN_SCALE

    expected_scale = torch.clamp(transformed.abs().amax(-1) / maximum, min=VQ2_FP8_MIN_SCALE)
    normalized = transformed / expected_scale.unsqueeze(-1)
    expected_q = torch.clamp(normalized, -maximum, maximum).to(torch.float8_e4m3fn)
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(actual_scale.view(torch.int32), expected_scale.view(torch.int32))
    assert torch.equal(actual_bias.view(torch.int32), expected_bias.view(torch.int32))
