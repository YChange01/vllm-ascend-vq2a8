# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from tools.validate_vq2a8_tp1_acceptance import format_compact_summary, summarize_log
from vllm_ascend.quantization import vq2a8_execution as execution
from vllm_ascend.quantization.vq2a8_offline import OfflineMoEOwner, offline_engine_options
from vllm_ascend.quantization.vq2a8_reference import prepare_repacked_vq2a8_activation_reference
from vllm_ascend.quantization.vq2a8_runtime import VQ2TP1Layer


def layer_header(index, count):
    return VQ2TP1Layer(
        layer_index=index,
        expert_ids=tuple(range(count)),
        tensor_path=Path("/unused"),
        metadata_path=Path("/unused"),
        tensor_sha256="",
        metadata_sha256="",
        specs={},
        tensor_shapes={
            f"{kind}_{field}": (count, 4) for kind in ("gate_up", "down") for field in execution.VQ2_TP1_TORCH_DTYPES
        },
    )


def test_plan_rounds_each_tensor_and_accounts_single_expert_hash_layers():
    unit = 12 * 512
    layers = [layer_header(0, 1), layer_header(3, 256)]
    plan = execution.packed_cache_plan(layers, unit * 18)
    assert plan["layer_limits"] == {0: 1, 3: 17}
    assert plan["planned_bytes"] == unit * 18
    assert plan["full_packed_bytes"] == unit * 257
    assert not plan["all_experts_fit"]
    full = execution.packed_cache_plan(layers, unit * 300)
    assert full["all_experts_fit"] and full["planned_bytes"] == unit * 257
    limited = execution.packed_cache_plan(layers, unit * 300, expert_limit=8)
    assert limited["layer_limits"] == {0: 1, 3: 8}


@pytest.mark.parametrize("budget", [0, -1, 1.5, True, 12 * 512 - 1])
def test_invalid_or_too_small_budget_fails_before_expert_allocation(budget):
    with pytest.raises(ValueError):
        execution.packed_cache_plan([layer_header(0, 1)], budget)


def test_budget_reserves_headroom_and_reuses_allocator_cache_only_once(monkeypatch):
    gib = execution.GIB
    monkeypatch.setattr(execution, "synchronize_execution", lambda device: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (60 * gib, 100 * gib))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 20 * gib)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 30 * gib)
    plan = execution.device_cache_budget("cuda:0")
    assert plan["budget_bytes"] == 54 * gib  # min(60+10,90-20) -16
    assert execution.device_cache_budget("cuda:0", budget_gib=30)["budget_bytes"] == 30 * gib
    with pytest.raises(ValueError, match="exceeds"):
        execution.device_cache_budget("cuda:0", budget_gib=55)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reserve_gib": 0},
        {"budget_gib": -1},
        {"budget_gib": float("nan")},
        {"memory_fraction": 2},
        {"reserve_gib": True},
    ],
)
def test_invalid_device_budget_options_rejected_without_device_access(kwargs):
    with pytest.raises(ValueError):
        execution.device_cache_budget("cuda:0", **kwargs)


def cache_only_runtime(limit=3):
    runtime = execution.CachedVQ2TP1MoE.__new__(execution.CachedVQ2TP1MoE)
    runtime.device = torch.device("cpu")
    runtime.layer_index = 0
    runtime.layer = NS(expert_ids=(0, 1, 2))
    runtime._cache = OrderedDict()
    runtime.cache_experts = limit
    runtime.cache_hits = runtime.cache_loads = runtime.cache_peak_bytes = 0
    runtime._resident_bytes = runtime.evictions = 0
    runtime.progress = False
    runtime.timing = dict.fromkeys(("host_load_validate_s", "h2d_s", "prepare_s", "packed_projection_s"), 0.0)
    runtime.prepare_batches = runtime.projection_rows = 0
    return runtime


def test_cached_weights_validate_only_on_misses_and_survive_repeated_passes():
    runtime = cache_only_runtime()
    calls = []

    def load(layer, expert, kind, **kwargs):
        assert kwargs == {"device": "cpu"}  # Never disable payload validation.
        calls.append((expert, kind))
        return {"packed": torch.ones(4, dtype=torch.int32)}, None

    runtime.artifact = NS(load_expert=load)
    for _ in range(2):
        for expert in (0, 1, 2):
            runtime._get_expert(expert)
    assert len(calls) == 6  # two projections once per expert, not per pass
    assert runtime.cache_stats() == dict(
        resident_experts=3, resident_bytes=96, peak_packed_bytes=96, loads=3, hits=3, evictions=0
    )
    runtime.clear_cache()
    assert runtime.cache_stats()["resident_bytes"] == 0
    assert runtime.cache_stats()["resident_experts"] == 0


def test_bounded_lru_and_failed_load_never_publish_partial_expert():
    runtime = cache_only_runtime(limit=2)
    runtime.artifact = NS(load_expert=lambda *args, **kwargs: ({"packed": torch.ones(4)}, None))
    for index in (0, 1, 0, 2):
        runtime._get_expert(index)
    assert list(runtime._cache) == [0, 2]
    assert runtime.cache_stats()["resident_bytes"] == 64
    assert runtime.evictions == 1

    def bad(*args, **kwargs):
        raise ValueError("Invalid payload")

    runtime.artifact.load_expert = bad
    with pytest.raises(ValueError, match="Invalid payload"):
        runtime._get_expert(1)
    assert runtime.cache_loads == 3 and 1 not in runtime._cache
    assert runtime._resident_bytes == 32
    with pytest.raises(ValueError, match="no stored"):
        runtime._get_expert(999)


@pytest.mark.parametrize("count", [1, 3, 10, 32])
@pytest.mark.parametrize("case", ["normal", "zero", "impulse", "small"])
def test_cached_policy_preserves_per_row_preparation_and_aligned_scalars(monkeypatch, count, case):
    runtime = cache_only_runtime()
    # Execute host math and a capturing kernel stand-in; no device claims.
    runtime.device = torch.device("cuda:0")
    monkeypatch.setattr(execution, "synchronize_execution", lambda device: None)
    calls = []

    def capture(q, scale, bias, *weights):
        assert q.shape == (1, 512)
        assert scale.shape == bias.shape == (1,)
        assert scale.storage_offset() == bias.storage_offset() == 0
        calls.append((q.clone(), scale.clone(), bias.clone()))
        return (q.float() * scale + bias).to(torch.bfloat16)

    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization.vq2a8_triton", NS(vq2a8_tp1_m1_packed_gemm=capture))
    x = ((torch.arange(count * 512).reshape(count, 512) % 17) - 8).to(torch.bfloat16) / 16
    if case == "zero":
        x.zero_()
    elif case == "impulse":
        x.zero_()
        x[:, 0] = 1
    elif case == "small":
        x *= 0.0001
    columns = torch.arange(512)
    payload = {
        "weight_scale": torch.ones(512),
        "weight_bias": torch.zeros(512),
        "rht_sign": torch.where(columns % 2 == 0, 1, -1).to(torch.int8),
        "packed_indices": None,
        "codebooks": None,
        "codebook_tile_ids": None,
    }
    actual = runtime._projection(x, payload, NS(columns=512, rht_true_columns=512, rht_block_size=128))
    assert actual.shape == (count, 512)
    assert len(calls) == count and runtime.prepare_batches == count and runtime.projection_rows == count
    for row, (q, scale, bias) in zip(x.split(1), calls):
        ref_q, ref_scale, ref_bias = prepare_repacked_vq2a8_activation_reference(
            row, payload["weight_scale"], payload["weight_bias"], payload["rht_sign"], 128
        )
        assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
        torch.testing.assert_close(scale, ref_scale, rtol=0, atol=0)
        torch.testing.assert_close(bias, ref_bias, rtol=0, atol=0)


def test_owner_configures_actual_layer_limits_before_any_expert_is_loaded(monkeypatch):
    owner = OfflineMoEOwner.__new__(OfflineMoEOwner)
    owner.options = {"execution_policy": "cached", "cache_experts": 256}
    owner.device = torch.device("cuda:0")
    owner.layers = {
        index: NS(layer=layer_header(index, count), cache_experts=1, cache_stats=lambda: {"resident_experts": 0})
        for index, count in ((0, 1), (3, 256))
    }
    monkeypatch.setattr(
        "vllm_ascend.quantization.vq2a8_offline.device_cache_budget",
        lambda *args, **kwargs: {"budget_bytes": 18 * 12 * 512},
    )
    owner.configure_cache(0.9)
    assert owner.layers[0].cache_experts == 1 and owner.layers[3].cache_experts == 17
    owner.layers[3].cache_stats = lambda: {"resident_experts": 1}
    with pytest.raises(ValueError, match="before"):
        owner.configure_cache(0.9)


def test_offline_policy_preserves_the_old_plan_as_explicit_baseline():
    baseline = offline_engine_options(Path("/m"), Path("/a"), execution_policy="baseline")
    cached = offline_engine_options(Path("/m"), Path("/a"))
    assert baseline["additional_config"]["vq2a8_offline"]["cache_experts"] == 2
    assert baseline["additional_config"]["vq2a8_offline"]["token_chunk"] == 2
    assert cached["additional_config"]["vq2a8_offline"]["cache_experts"] == 256
    assert cached["additional_config"]["vq2a8_offline"]["token_chunk"] == 2
    assert cached["kv_cache_memory_bytes"] == baseline["kv_cache_memory_bytes"]


def test_timeout_summary_reports_unknown_and_counts_only_finished_forwards(tmp_path):
    log = tmp_path / "child.log"
    log.write_text(
        'MODEL_FORWARD_TIMING {"phase":"profile","elapsed_s":10}\n'
        "MODEL stage=forward_start phase=prefill step=0 tokens=10\n"
        "MODEL layer=38 stage=expert_start expert=4 tokens=1\n"
    )
    result = summarize_log(log, None, timed_out=True)
    result["probe"] = "full_model"
    text = format_compact_summary({"stage": "model", "status": "failed", "probes": ["full_model"], "results": [result]})
    assert "finite=unknown repeat_exact=unknown" in text
    assert "FORWARD=profile completed=1 total_s=10.000" in text
    assert "FORWARD=prefill completed=0" in text
    assert "FORWARD=decode completed=0" in text
    assert "timeout=True" in text and "layer=38" in text


@pytest.mark.parametrize(
    "comparison,expected",
    [
        ({"allclose": True, "max_abs_error": 0}, "PASS"),
        ({"allclose": True, "max_abs_error": 0.01}, "UNKNOWN_OR_FAIL"),
        (None, "UNKNOWN_OR_FAIL"),
    ],
)
def test_cached_short_report_distinguishes_exact_agreement_from_tolerance(comparison, expected):
    text = format_compact_summary(
        {
            "stage": "moe",
            "probes": ["layer0"],
            "status": "running",
            "results": [
                {
                    "probe": "layer0",
                    "records": [
                        {
                            "type": "MOE_RESULT",
                            "data": {
                                "execution_policy": "cached",
                                "accepted_policy_comparison": comparison,
                            },
                        }
                    ],
                }
            ],
        }
    )
    assert f"baseline_exact={expected}" in text
