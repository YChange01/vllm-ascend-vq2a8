# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only cache/KV budget contracts; no device allocation/performance claim."""

import json
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_ascend.quantization import vq2a8_execution as execution
from vllm_ascend.quantization import vq2a8_offline as offline


def options(tmp_path, policy="cached", **kwargs):
    if policy in ("ascendc", "ascendc_v2", "ascendc_v3"):
        kwargs[f"{policy}_library"] = tmp_path / f"lib{policy}.so"
        kwargs[f"{policy}_sha256"] = "a" * 64
    return offline.offline_engine_options(tmp_path / "model", tmp_path / "artifact", execution_policy=policy, **kwargs)


def config(value):
    return NS(
        additional_config=value["additional_config"],
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        model_config=NS(enforce_eager=True, quantization=None, dtype=torch.bfloat16, max_model_len=32),
        quant_config=None,
        scheduler_config=NS(max_num_seqs=1, max_num_batched_tokens=32),
        compilation_config=NS(mode=0, cudagraph_mode=0),
        cache_config=NS(gpu_memory_utilization=0.25, kv_cache_memory_bytes=value["kv_cache_memory_bytes"]),
        load_config=NS(load_format="safetensors"),
    )


def owner(**overrides):
    value = offline.OfflineMoEOwner.__new__(offline.OfflineMoEOwner)
    value.options = {"execution_policy": "cached", "cache_experts": 256, "cache_reserve_gib": 5.0, **overrides}
    value.device = torch.device("cuda:0")
    value.layers = {3: NS(layer=object(), cache_experts=256, cache_stats=lambda: {"resident_experts": 0})}
    value.cache_plan = None
    return value


def fake_plan(headers, budget, **kwargs):
    return {"budget_bytes": budget, "planned_bytes": budget, "layer_limits": {3: 6}}


def budget_log(capsys):
    lines = capsys.readouterr().out.splitlines()
    return json.loads(next(line.split(" ", 1)[1] for line in lines if line.startswith("MODEL_CACHE_BUDGET ")))


@pytest.mark.parametrize("policy", ["baseline", "cached", "ascendc", "ascendc_v2", "ascendc_v3"])
def test_cache_fraction_none_is_omitted_and_preserves_old_engine_defaults(tmp_path, policy):
    original = options(tmp_path, policy)
    explicit_none = options(tmp_path, policy, cache_memory_fraction=None)
    assert original == explicit_none
    assert "cache_memory_fraction" not in original["additional_config"]["vq2a8_offline"]
    assert original["gpu_memory_utilization"] == (0.35 if policy == "baseline" else 0.9)
    assert original["kv_cache_memory_bytes"] == 1024**3


@pytest.mark.parametrize("policy", offline.CACHE_EXECUTION_POLICIES)
@pytest.mark.parametrize("fraction", [0.0001, 0.95, 1])
def test_cache_fraction_override_is_explicit_for_cached_policies_and_keeps_engine_fraction(tmp_path, policy, fraction):
    value = options(tmp_path, policy, cache_memory_fraction=fraction)
    assert value["gpu_memory_utilization"] == 0.9
    cfg = config(value)
    validated = offline.validate_offline_config(cfg)
    assert validated["cache_memory_fraction"] == fraction
    assert cfg.cache_config.gpu_memory_utilization == 0.25


@pytest.mark.parametrize("fraction", [0, -1, 1.01, True, False, float("nan"), float("inf"), ".95"])
def test_cache_fraction_rejects_invalid_numbers_in_builder_and_config(tmp_path, fraction):
    with pytest.raises(ValueError, match="cache_memory_fraction"):
        options(tmp_path, cache_memory_fraction=fraction)
    cfg = config(options(tmp_path))
    cfg.additional_config["vq2a8_offline"]["cache_memory_fraction"] = fraction
    with pytest.raises(ValueError, match="cache_memory_fraction"):
        offline.validate_offline_config(cfg)


def test_cache_fraction_explicit_json_null_is_not_a_valid_override(tmp_path):
    cfg = config(options(tmp_path))
    cfg.additional_config["vq2a8_offline"]["cache_memory_fraction"] = None
    with pytest.raises(ValueError, match="cache_memory_fraction"):
        offline.validate_offline_config(cfg)


@pytest.mark.parametrize("policy", ["baseline", "unknown"])
def test_cache_fraction_override_rejects_noncache_policy_before_model_allocation(tmp_path, policy):
    with pytest.raises(ValueError, match="cache_memory_fraction"):
        options(tmp_path, policy, cache_memory_fraction=0.95)


@pytest.mark.parametrize("kv_bytes", [None, 0, -1, True, 1.0, "1073741824", "missing"])
def test_cache_fraction_requires_fixed_positive_integer_kv_budget(tmp_path, kv_bytes):
    cfg = config(options(tmp_path, cache_memory_fraction=0.95))
    if kv_bytes == "missing":
        del cfg.cache_config.kv_cache_memory_bytes
    else:
        cfg.cache_config.kv_cache_memory_bytes = kv_bytes
    with pytest.raises(ValueError, match="explicit positive integer kv_cache_memory_bytes"):
        offline.validate_offline_config(cfg)


def test_cache_fraction_requires_explicit_kv_to_fit_inside_reserve(tmp_path):
    cfg = config(options(tmp_path, cache_memory_fraction=0.95, cache_reserve_gib=1))
    assert offline.validate_offline_config(cfg)["cache_reserve_gib"] == 1
    cfg.cache_config.kv_cache_memory_bytes += 1
    with pytest.raises(ValueError, match="fit inside cache_reserve_gib"):
        offline.validate_offline_config(cfg)


def test_cache_fraction_absent_does_not_add_a_new_kv_contract(tmp_path):
    cfg = config(options(tmp_path))
    del cfg.cache_config.kv_cache_memory_bytes
    assert offline.validate_offline_config(cfg)
    cfg.cache_config.kv_cache_memory_bytes = 100 * 1024**3
    assert offline.validate_offline_config(cfg)


@pytest.mark.parametrize("policy", offline.CACHE_EXECUTION_POLICIES)
def test_cache_fraction_owner_selects_override_for_each_cached_backend(monkeypatch, policy):
    from vllm_ascend.quantization import vq2a8_ascendc_v2 as v2

    value = owner(execution_policy=policy, cache_memory_fraction=0.95)
    calls = []

    def sample(device, **kwargs):
        calls.append(kwargs)
        return {"budget_bytes": 1234}

    monkeypatch.setattr(offline, "device_cache_budget", sample)
    monkeypatch.setattr(offline, "packed_cache_plan", fake_plan)
    monkeypatch.setattr(v2, "ascendc_v2_cache_plan", fake_plan)
    value._configure_v3_residency = lambda budget: setattr(value, "cache_plan", budget)
    value.configure_cache(0.25)
    assert len(calls) == 1 and calls[0]["memory_fraction"] == 0.95
    assert calls[0]["budget_gib"] == 0.0
    assert value.cache_plan["budget_bytes"] == 1234


@pytest.mark.parametrize("override", ["missing", None])
def test_cache_fraction_owner_omission_and_manual_none_retain_engine_fraction(monkeypatch, override):
    value = owner(**({} if override == "missing" else {"cache_memory_fraction": None}))
    calls = []
    monkeypatch.setattr(offline, "device_cache_budget", lambda device, **kw: calls.append(kw) or {"budget_bytes": 1234})
    monkeypatch.setattr(offline, "packed_cache_plan", fake_plan)
    value.configure_cache(0.25)
    assert calls[0]["memory_fraction"] == 0.25


@pytest.mark.parametrize("bad", [True, False, -1, float("nan"), float("inf"), "1", None])
def test_cache_fraction_safe_sampling_does_not_bypass_explicit_budget_validation(monkeypatch, bad):
    value = owner(cache_budget_gib=bad)
    monkeypatch.setattr(
        offline, "device_cache_budget", lambda *args, **kwargs: pytest.fail("invalid cap must fail first")
    )
    with pytest.raises(ValueError, match="cache_budget_gib"):
        value.configure_cache(0.9)


@pytest.mark.parametrize("requested_gib,expected_gib", [(0, 65), (12, 12)])
def test_cache_fraction_reuses_original_safe_budget_formula_with_one_sample(
    monkeypatch, capsys, requested_gib, expected_gib
):
    gib = offline.GIB
    syncs = []
    monkeypatch.setattr(execution, "synchronize_execution", lambda device: syncs.append(device))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (60 * gib, 100 * gib))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 20 * gib)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 30 * gib)
    monkeypatch.setattr(offline, "packed_cache_plan", fake_plan)
    value = owner(cache_memory_fraction=0.95, cache_budget_gib=requested_gib)
    value.configure_cache(0.25)
    assert len(syncs) == 1
    assert value.cache_plan["budget_bytes"] == expected_gib * gib
    assert value.cache_plan["memory_fraction"] == 0.95
    assert value.cache_plan["engine_memory_fraction"] == 0.25
    assert value.cache_plan["cache_memory_fraction"] == 0.95
    log = budget_log(capsys)
    assert log["engine_memory_fraction"] == 0.25 and log["cache_memory_fraction"] == 0.95
    assert log["free_gib"] == 60 and log["reusable_gib"] == 70
    assert log["reserve_gib"] == 5 and log["available_gib"] == 65
    assert log["fits"] and "not_model_residency" in log["scope"]


def test_cache_fraction_oversized_cap_logs_before_rejection_and_never_plans(monkeypatch, capsys):
    value = owner(cache_memory_fraction=0.95, cache_budget_gib=13)
    calls = []
    monkeypatch.setattr(
        offline,
        "device_cache_budget",
        lambda *args, **kwargs: calls.append(kwargs) or {"budget_bytes": 12 * offline.GIB},
    )
    monkeypatch.setattr(offline, "packed_cache_plan", lambda *args, **kwargs: pytest.fail("unsafe cap must not plan"))
    with pytest.raises(ValueError, match="exceeds safe current budget"):
        value.configure_cache(0.25)
    log = budget_log(capsys)
    assert len(calls) == 1 and not log["fits"]
    assert log["requested_gib"] == 13 and log["available_gib"] == 12


def test_cache_fraction_v3_shortfall_is_logged_before_any_bank_initialization(monkeypatch, capsys):
    value = owner(execution_policy="ascendc_v3", cache_memory_fraction=0.95)
    shapes = {}
    specs = {}
    for kind, k in (("gate_up", 4096), ("down", 2048)):
        n = 4096
        specs[kind] = NS(rows=n, columns=k, rht_true_columns=k, rht_block_size=128)
        shapes.update(
            {
                f"{kind}_packed_indices": (1, n // 2, k // 8),
                f"{kind}_codebooks": (1, k // 256, n // 32, 16, 2),
                **{
                    f"{kind}_{field}": (1, k)
                    for field in ("codebook_tile_ids", "weight_scale", "weight_bias", "rht_sign")
                },
            }
        )
    value.layers[3].layer = NS(
        layer_index=3,
        expert_ids=(0,),
        tensor_shapes=shapes,
        specs=specs,
    )
    value.layers[3].initialize_resident = lambda **kwargs: pytest.fail("shortfall must not allocate weights")
    monkeypatch.setattr(offline, "device_cache_budget", lambda *args, **kwargs: {"budget_bytes": 0})
    with pytest.raises(ValueError, match="full residency requires"):
        value.configure_cache(0.25)
    text = capsys.readouterr().out
    assert text.index("MODEL_CACHE_BUDGET ") < text.index("V3_RESIDENCY_BUDGET ")
    assert '"fits": false' in text
    assert "v3_resident_load_start" not in text
