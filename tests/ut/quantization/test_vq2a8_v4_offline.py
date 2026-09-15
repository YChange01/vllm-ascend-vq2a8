# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only V4 integration contracts; no NPU or performance certification."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_ascend.quantization import vq2a8_ascendc as native_v1
from vllm_ascend.quantization import vq2a8_offline as offline


def engine_options(tmp_path, **overrides):
    kwargs = {
        "execution_policy": "ascendc_v4",
        "ascendc_library": tmp_path / "libvq2a8_ascendc.so",
        "ascendc_sha256": "a" * 64,
    }
    kwargs.update(overrides)
    return offline.offline_engine_options(tmp_path / "model", tmp_path / "artifact", **kwargs)


def config(tmp_path):
    return NS(
        additional_config=engine_options(tmp_path)["additional_config"],
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        model_config=NS(enforce_eager=True, quantization=None, dtype=torch.bfloat16, max_model_len=32),
        quant_config=None,
        scheduler_config=NS(max_num_seqs=1, max_num_batched_tokens=32),
        compilation_config=NS(mode=0, cudagraph_mode=0),
        cache_config=NS(gpu_memory_utilization=0.9),
        load_config=NS(load_format="safetensors"),
    )


def test_v4_is_opt_in_and_uses_the_v1_library_with_unchanged_startup_geometry(tmp_path):
    plan = engine_options(tmp_path)
    value = offline.validate_offline_config(config(tmp_path))
    assert "ascendc_v4" in offline.CACHE_EXECUTION_POLICIES
    assert value["execution_policy"] == "ascendc_v4"
    assert value["ascendc_library"] == str(tmp_path / "libvq2a8_ascendc.so")
    assert value["ascendc_sha256"] == "a" * 64
    assert not any(key.startswith(("ascendc_v2_", "ascendc_v3_", "v3_")) for key in value)
    assert value["cache_experts"] == 256 and value["token_chunk"] == 2
    assert value["root_linear_mode"] == "bf16" and value["cache_reserve_gib"] == 16.0
    assert plan["enforce_eager"] and plan["compilation_config"] == {"mode": 0, "cudagraph_mode": "NONE"}
    assert plan["tensor_parallel_size"] == 1 and plan["distributed_executor_backend"] == "uni"
    assert plan["gpu_memory_utilization"] == 0.9
    original = offline.offline_engine_options(tmp_path / "model", tmp_path / "artifact")
    assert original["additional_config"]["vq2a8_offline"]["execution_policy"] == "cached"
    legacy = engine_options(tmp_path, execution_policy="ascendc")
    assert legacy["additional_config"]["vq2a8_offline"]["execution_policy"] == "ascendc"


def test_moe_graph_options_are_explicit_and_keep_engine_eager(tmp_path):
    plan = engine_options(tmp_path, v4_device_route_decode=True, v4_decode_graph="moe")
    assert plan["enforce_eager"] is True
    assert plan["compilation_config"] == {"mode": 0, "cudagraph_mode": "NONE"}
    cfg = config(tmp_path)
    cfg.additional_config = plan["additional_config"]
    cfg.cache_config.kv_cache_memory_bytes = plan["kv_cache_memory_bytes"]
    assert offline.validate_offline_config(cfg)["v4_decode_graph"] == "moe"
    assert "v4_decode_graph" not in engine_options(tmp_path)["additional_config"]["vq2a8_offline"]


@pytest.mark.parametrize("policy", ["owner", "caller"])
def test_graph_replay_stream_round_trips_with_unchanged_memory_and_topology(tmp_path, policy):
    plan = engine_options(tmp_path, v4_device_route_decode=True, v4_decode_graph="moe", v4_graph_replay_stream=policy)
    cfg = config(tmp_path)
    cfg.additional_config = plan["additional_config"]
    cfg.cache_config.kv_cache_memory_bytes = plan["kv_cache_memory_bytes"]
    assert offline.validate_offline_config(cfg)["v4_graph_replay_stream"] == policy
    assert plan["enforce_eager"] and plan["compilation_config"] == {"mode": 0, "cudagraph_mode": "NONE"}
    assert plan["kv_cache_memory_bytes"] == 1024**3
    assert "v4_graph_replay_stream" not in engine_options(tmp_path)["additional_config"]["vq2a8_offline"]


@pytest.mark.parametrize("policy", [None, True, False, 1, "auto", "default"])
def test_bad_graph_replay_stream_rejected_at_both_config_entrypoints(tmp_path, policy):
    with pytest.raises(ValueError, match="v4_graph_replay_stream"):
        engine_options(tmp_path, v4_device_route_decode=True, v4_decode_graph="moe", v4_graph_replay_stream=policy)
    cfg = config(tmp_path)
    cfg.additional_config["vq2a8_offline"]["v4_graph_replay_stream"] = policy
    with pytest.raises(ValueError, match="v4_graph_replay_stream"):
        offline.validate_offline_config(cfg)


def test_caller_replay_requires_moe_graph(tmp_path):
    with pytest.raises(ValueError, match="caller requires"):
        engine_options(tmp_path, v4_graph_replay_stream="caller")
    cfg = config(tmp_path)
    cfg.additional_config["vq2a8_offline"]["v4_graph_replay_stream"] = "caller"
    with pytest.raises(ValueError, match="caller requires"):
        offline.validate_offline_config(cfg)


@pytest.mark.parametrize("kv", [None, True, 0, -1, 17 * 1024**3])
def test_graph_requires_explicit_kv_covered_by_existing_reserve(tmp_path, kv):
    cfg = config(tmp_path)
    cfg.additional_config["vq2a8_offline"].update(v4_device_route_decode=True, v4_decode_graph="moe")
    cfg.cache_config.kv_cache_memory_bytes = kv
    with pytest.raises(ValueError, match="kv_cache_memory_bytes|KV cache"):
        offline.validate_offline_config(cfg)


@pytest.mark.parametrize("mode", [True, False, None, "full", "v3", 1])
def test_v4_unknown_graph_modes_fail_before_loading(tmp_path, mode):
    with pytest.raises(ValueError, match="v4_decode_graph"):
        engine_options(tmp_path, v4_device_route_decode=True, v4_decode_graph=mode)
    cfg = config(tmp_path)
    cfg.additional_config["vq2a8_offline"].update(v4_device_route_decode=True, v4_decode_graph=mode)
    with pytest.raises(ValueError, match="v4_decode_graph"):
        offline.validate_offline_config(cfg)


def test_graph_requires_device_route_even_when_config_is_supplied_directly(tmp_path):
    with pytest.raises(ValueError, match="device-route"):
        engine_options(tmp_path, v4_decode_graph="moe")
    cfg = config(tmp_path)
    cfg.additional_config["vq2a8_offline"]["v4_decode_graph"] = "moe"
    with pytest.raises(ValueError, match="device_route"):
        offline.validate_offline_config(cfg)


@pytest.mark.parametrize(
    "key,value",
    [
        ("ascendc_library", "relative.so"),
        ("ascendc_sha256", "bad-hash"),
        ("ascendc_sha256", None),
        ("root_linear_mode", "online_fp8_sm90"),
        ("cache_experts", 171),
        ("cache_experts", True),
        ("ascendc_v2_library", "/wrong-v2.so"),
        ("ascendc_v3_library", "/wrong-v3.so"),
        ("v3_preparation", "fused"),
        ("v3_decode_graph", "moe"),
    ],
)
def test_v4_config_rejects_changed_library_or_residency_contract(tmp_path, key, value):
    cfg = config(tmp_path)
    cfg.additional_config["vq2a8_offline"][key] = value
    with pytest.raises(ValueError):
        offline.validate_offline_config(cfg)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("parallel_config", "tensor_parallel_size", 2),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "enable_expert_parallel", True),
        ("model_config", "enforce_eager", False),
        ("model_config", "dtype", torch.float16),
        ("model_config", "enable_sleep_mode", True),
        ("compilation_config", "cudagraph_mode", 1),
        ("cache_config", "cpu_offload_gb", 1),
        ("scheduler_config", "async_scheduling", True),
    ],
)
def test_v4_config_keeps_single_device_eager_ownership_limits(tmp_path, section, key, value):
    cfg = config(tmp_path)
    setattr(getattr(cfg, section), key, value)
    with pytest.raises(ValueError):
        offline.validate_offline_config(cfg)


@pytest.mark.parametrize(
    "option",
    [
        {"tensor_parallel_size": 2},
        {"v3_preparation": "fused"},
        {"v3_decode_graph": "moe"},
        {"ascendc_v2_library": "/v2.so"},
        {"ascendc_v3_library": "/v3.so"},
    ],
)
def test_v4_builder_does_not_silently_drop_other_backend_options(tmp_path, option):
    with pytest.raises(ValueError):
        engine_options(tmp_path, **option)


def test_v4_owner_loads_pinned_v1_library_and_only_direct_tp1_artifact(monkeypatch, tmp_path):
    options = engine_options(tmp_path)["additional_config"]["vq2a8_offline"]
    loaded = []
    artifact = object()
    monkeypatch.setattr(
        native_v1, "load_pinned_library", lambda path, sha: loaded.append((path, sha)) or {"sha256": sha}
    )
    monkeypatch.setattr(offline, "artifact_format", lambda path: offline.VQ2_DIRECT_TP1_FORMAT)
    monkeypatch.setattr(offline, "audit_offline_root", lambda path: {})

    def open_direct(path, model_config, **kwargs):
        assert path == tmp_path / "artifact" and model_config == tmp_path / "model/config.json"
        assert kwargs == {"require_complete": True, "require_reference_identity": True}
        return artifact

    monkeypatch.setattr(offline, "open_vq2a8_tp1_artifact", open_direct)
    owner = offline.OfflineMoEOwner(tmp_path / "model", options, NS(type="npu"))
    assert loaded == [(str(tmp_path / "libvq2a8_ascendc.so"), "a" * 64)]
    assert owner.artifact is artifact and owner.native_library == {"sha256": "a" * 64}
    assert owner.layers == {} and owner.cache_plan is None


def test_device_route_missing_native_abi_fails_before_artifact_or_weight_load(monkeypatch, tmp_path):
    from vllm_ascend.quantization import vq2a8_v4_device_route

    options = engine_options(tmp_path, v4_device_route_decode=True)["additional_config"]["vq2a8_offline"]
    monkeypatch.setattr(native_v1, "load_pinned_library", lambda *args: {})
    monkeypatch.setattr(offline, "artifact_format", lambda *args: pytest.fail("weights must not be visited"))

    def missing():
        raise RuntimeError("Rebuild ResidentBank")

    monkeypatch.setattr(vq2a8_v4_device_route, "require_device_route_library", missing)
    with pytest.raises(RuntimeError, match="ResidentBank"):
        offline.OfflineMoEOwner(tmp_path / "model", options, NS(type="npu"))


@pytest.mark.parametrize("format_name", [offline.VQ2_TP1_ZN_FORMAT, "vq2a8_tp2_zn", "unknown"])
def test_v4_owner_rejects_zn_or_unknown_artifact_without_format_fallback(monkeypatch, tmp_path, format_name):
    options = engine_options(tmp_path)["additional_config"]["vq2a8_offline"]
    monkeypatch.setattr(native_v1, "load_pinned_library", lambda *args: {})
    monkeypatch.setattr(offline, "artifact_format", lambda path: format_name)
    monkeypatch.setattr(
        offline, "open_vq2a8_tp1_artifact", lambda *args, **kwargs: pytest.fail("unexpected direct fallback")
    )
    monkeypatch.setattr(
        offline, "open_vq2a8_tp1_zn_artifact", lambda *args, **kwargs: pytest.fail("V1 native cannot read ZN")
    )
    with pytest.raises(ValueError):
        offline.OfflineMoEOwner(tmp_path / "model", options, NS(type="npu"))


def test_v4_configure_cache_selects_residency_not_a_reduced_lazy_cap(monkeypatch):
    owner = offline.OfflineMoEOwner.__new__(offline.OfflineMoEOwner)
    owner.options = {"execution_policy": "ascendc_v4", "cache_experts": 256}
    owner.device = NS(type="npu")
    owner.layers = {0: NS(cache_stats=lambda: {"resident_experts": 0}, cache_experts=256)}
    calls = []
    monkeypatch.setattr(offline, "device_cache_budget", lambda *args, **kwargs: {"budget_bytes": 1234})
    monkeypatch.setattr(offline, "packed_cache_plan", lambda *args, **kwargs: pytest.fail("lazy planner called"))
    owner._configure_v4_residency = lambda budget: calls.append(budget)
    owner.configure_cache(0.9)
    assert len(calls) == 1 and calls[0]["budget_bytes"] == 1234
    assert calls[0]["engine_memory_fraction"] == 0.9
    assert owner.layers[0].cache_experts == 256


def test_v4_configure_cache_does_not_load_if_requested_budget_is_unsafe(monkeypatch):
    owner = offline.OfflineMoEOwner.__new__(offline.OfflineMoEOwner)
    owner.options = {"execution_policy": "ascendc_v4", "cache_budget_gib": 1}
    owner.device = NS(type="npu")
    owner.layers = {0: NS(cache_stats=lambda: {"resident_experts": 0})}
    monkeypatch.setattr(offline, "device_cache_budget", lambda *args, **kwargs: {"budget_bytes": 1234})
    owner._configure_v4_residency = lambda budget: pytest.fail("loading before budget rejection")
    with pytest.raises(ValueError, match="exceeds safe current budget"):
        owner.configure_cache(0.9)


def test_v4_configure_cache_requires_fresh_runtime_before_any_loading(monkeypatch):
    owner = offline.OfflineMoEOwner.__new__(offline.OfflineMoEOwner)
    owner.options = {"execution_policy": "ascendc_v4"}
    owner.layers = {0: NS(cache_stats=lambda: {"resident_experts": 1})}
    monkeypatch.setattr(offline, "device_cache_budget", lambda *args, **kwargs: pytest.fail("already populated"))
    with pytest.raises(ValueError, match="before the first expert call"):
        owner.configure_cache(0.9)


def resident_owner(calls):
    owner = offline.OfflineMoEOwner.__new__(offline.OfflineMoEOwner)
    owner.options = {"execution_policy": "ascendc_v4"}
    owner.artifact = NS(layers={0: object(), 1: object()})
    owner.layers = {
        index: NS(
            layer=NS(layer_index=index, expert_ids=(0, 1)),
            initialize_resident=lambda *, budget_bytes, i=index: calls.append(("load", i, budget_bytes)),
            check_resident_integrity=lambda i=index: calls.append(("integrity", i)),
            abort_residency=lambda i=index: calls.append(("abort", i)),
        )
        for index in (0, 1)
    }
    return owner


def test_v4_plans_every_layer_before_loading_and_marks_ready_only_after_integrity(monkeypatch):
    from vllm_ascend.quantization import vq2a8_execution_v4 as v4

    calls = []
    owner = resident_owner(calls)

    def plan(headers, budget):
        assert not calls and [header.layer_index for header in headers] == [0, 1]
        assert budget == 30
        calls.append(("plan",))
        return {"planned_bytes": 30, "layer_plans": {0: {"planned_bytes": 10}, 1: {"planned_bytes": 20}}}

    def last_integrity():
        assert owner.cache_plan["preload_complete"] is False
        calls.append(("integrity", 1))

    owner.layers[1].check_resident_integrity = last_integrity
    monkeypatch.setattr(v4, "packed_resident_plan", plan)
    owner._configure_v4_residency({"budget_bytes": 30})
    assert calls == [("plan",), ("load", 0, 10), ("load", 1, 20), ("integrity", 0), ("integrity", 1)]
    assert owner.cache_plan["preload_complete"] is True
    assert owner.cache_plan["planned_bytes"] == 30 and owner.cache_plan["preload_elapsed_s"] >= 0


def test_v4_all_layer_budget_failure_never_allocates_or_aborts_partial_layers(monkeypatch):
    from vllm_ascend.quantization import vq2a8_execution_v4 as v4

    calls = []
    owner = resident_owner(calls)

    def insufficient(headers, budget):
        raise ValueError("full residency exceeds budget")

    monkeypatch.setattr(v4, "packed_resident_plan", insufficient)
    with pytest.raises(ValueError, match="full residency"):
        owner._configure_v4_residency({"budget_bytes": 1})
    assert calls == []


def test_v4_rejects_missing_model_layers_before_planning_or_loading(monkeypatch):
    from vllm_ascend.quantization import vq2a8_execution_v4 as v4

    calls = []
    owner = resident_owner(calls)
    owner.layers.pop(1)
    monkeypatch.setattr(v4, "packed_resident_plan", lambda *args: pytest.fail("incomplete inventory"))
    with pytest.raises(ValueError, match="all artifact layers"):
        owner._configure_v4_residency({"budget_bytes": 30})
    assert calls == []


@pytest.mark.parametrize("failure", ["load", "integrity"])
def test_v4_partial_load_or_integrity_failure_aborts_every_layer_without_ready(monkeypatch, failure):
    from vllm_ascend.quantization import vq2a8_execution_v4 as v4

    calls = []
    owner = resident_owner(calls)
    monkeypatch.setattr(
        v4,
        "packed_resident_plan",
        lambda *args: {"planned_bytes": 30, "layer_plans": {0: {"planned_bytes": 10}, 1: {"planned_bytes": 20}}},
    )

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic initialization failure")

    if failure == "load":
        owner.layers[1].initialize_resident = fail
    else:
        owner.layers[1].check_resident_integrity = fail
    with pytest.raises(RuntimeError, match="synthetic initialization failure"):
        owner._configure_v4_residency({"budget_bytes": 30})
    assert owner.cache_plan["preload_complete"] is False
    assert calls[-2:] == [("abort", 0), ("abort", 1)]


def test_v4_cleanup_error_does_not_mask_load_failure_or_skip_other_layer_cleanup(monkeypatch, capsys):
    from vllm_ascend.quantization import vq2a8_execution_v4 as v4

    calls = []
    owner = resident_owner(calls)
    monkeypatch.setattr(
        v4,
        "packed_resident_plan",
        lambda *args: {"planned_bytes": 30, "layer_plans": {0: {"planned_bytes": 10}, 1: {"planned_bytes": 20}}},
    )

    def fail_load(**kwargs):
        raise RuntimeError("original load error")

    def fail_cleanup():
        raise RuntimeError("secondary cleanup error")

    owner.layers[1].initialize_resident = fail_load
    owner.layers[0].abort_residency = fail_cleanup
    with pytest.raises(RuntimeError, match="original load error"):
        owner._configure_v4_residency({"budget_bytes": 30})
    assert calls[-1] == ("abort", 1)
    assert "MODEL_V4_CLEANUP_ERROR secondary cleanup error" in capsys.readouterr().out
    assert owner.cache_plan["preload_complete"] is False


def model_method(name):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/worker/vq2a8_offline_model.py"
    tree = ast.parse(path.read_text("utf-8"))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_v4_probe_switches_only_explicit_batched_arithmetic_and_retains_resident_payloads(monkeypatch):
    configure = model_method("configure_performance_probe")
    layer = offline.AscendCVQ2TP1MoE.__new__(offline.AscendCVQ2TP1MoE)
    layer.execution_policy, layer.token_chunk, layer.root = "ascendc_v4", 2, {}
    payloads = layer._cache = {0: object()}
    calls = []
    layer.check_resident_integrity = lambda: calls.append("integrity")
    monkeypatch.setattr(torch, "npu", NS(synchronize=lambda: calls.append("sync")), raising=False)
    model = NS(_offline_loaded=True, _offline_root_mode="bf16", model=NS(offline_owner=NS(layers={0: layer})))

    configure(model, measurement=False, compact=False, optimization=None)
    assert getattr(layer, "_optimization", None) is None
    baseline = layer._row_preparation
    configure(model, measurement=True, compact=True, optimization="batched")
    state = layer._optimization
    assert state.options.preparation == "rowwise"
    assert state.options.window == 128 and state.options.shared_batch is False and state.options.pipeline is False
    assert layer._cache is payloads and layer.token_chunk == 2
    configure(model, measurement=False, compact=False, optimization=None)
    assert layer._optimization is None and layer._row_preparation is baseline
    assert layer._cache is payloads and layer.token_chunk == 2
    assert calls == ["sync", "integrity"] * 3


@pytest.mark.parametrize("preset", ["fwht", "pipeline", "prepare_graph", "v3"])
def test_v4_probe_rejects_other_math_or_graph_paths_before_changing_state(monkeypatch, preset):
    configure = model_method("configure_performance_probe")
    layer = offline.AscendCVQ2TP1MoE.__new__(offline.AscendCVQ2TP1MoE)
    layer.execution_policy = "ascendc_v4"
    model = NS(_offline_loaded=True, _offline_root_mode="bf16", model=NS(offline_owner=NS(layers={0: layer})))
    monkeypatch.setattr(torch, "npu", NS(synchronize=lambda: pytest.fail("invalid preset")), raising=False)
    with pytest.raises(ValueError, match="V4 preserves V1 arithmetic"):
        configure(model, measurement=True, compact=True, optimization=preset)
    assert not hasattr(layer, "_optimization")


def test_v4_offline_evidence_requires_integrity_before_reporting_generation(monkeypatch):
    collect = model_method("offline_evidence")

    def corrupted():
        raise RuntimeError("resident payload changed")

    layer = NS(execution_policy="ascendc_v4", check_resident_integrity=corrupted)
    model = NS(_offline_logits=[torch.zeros(4, 8)], model=NS(offline_owner=NS(layers={0: layer})))
    with pytest.raises(RuntimeError, match="resident payload changed"):
        collect(model)


def test_v4_cache_report_includes_each_layer_residency_without_replacing_cache_statistics():
    calls = []
    owner = offline.OfflineMoEOwner.__new__(offline.OfflineMoEOwner)
    owner.options, owner.calls, owner.cache_plan = (
        {"execution_policy": "ascendc_v4"},
        {0: 4, 1: 4},
        {"planned_bytes": 30},
    )
    owner.layers = {
        index: NS(
            execution_policy="ascendc_v4",
            cache_experts=2,
            cache_stats=lambda: {"resident_bytes": 10, "resident_experts": 2, "loads": 2, "hits": 3, "evictions": 0},
            v4_report=lambda i=index: calls.append(i) or {"layer": i, "ready": True},
        )
        for index in (0, 1)
    }
    report = owner.cache_report()
    assert report["v4"] == {"0": {"layer": 0, "ready": True}, "1": {"layer": 1, "ready": True}}
    assert calls == [0, 1]
    assert report["resident_packed_bytes"] == 20 and report["resident_experts"] == 4
    assert report["loads"] == 4 and report["hits"] == 6 and report["evictions"] == 0
    assert report["plan"] is owner.cache_plan


def residency_evidence():
    plans = {
        0: {"experts": 2, "payload_bytes": 20, "planned_bytes": 32},
        1: {"experts": 1, "payload_bytes": 40, "planned_bytes": 64},
    }
    reports = {
        str(index): {
            "ready": True,
            "failed": False,
            "fallback_enabled": False,
            "execution_policy": "ascendc_v4",
            "layout": "v1_packed",
            "layer_index": index,
            "expected_experts": entry["experts"],
            "resident_experts": entry["experts"],
            "payload_bytes": entry["payload_bytes"],
            "planned_bytes": entry["planned_bytes"],
            "preload_loads": entry["experts"],
            "preload_h2d_bytes": entry["payload_bytes"],
            "preload_evictions": 0,
            "preload_elapsed_s": 0.25,
            "post_init_loads": 0,
            "post_init_h2d_bytes": 0,
            "post_init_evictions": 0,
            "cleanup_error": None,
        }
        for index, entry in plans.items()
    }
    cache = {
        "resident_experts": 3,
        "per_layer_cache_limit": 2,
        "resident_packed_bytes": 60,
        "loads": 3,
        "hits": 6,
        "evictions": 0,
        "layer_calls": {0: 4, 1: 4},
        "plan": {
            "allocation": "eager_packed_only",
            "layout": "v1_packed",
            "preload_complete": True,
            "all_experts_fit": True,
            "budget_bytes": 128,
            "planned_bytes": 96,
            "layer_plans": plans,
        },
        "v4": reports,
    }
    return cache, reports


def test_v4_residency_evidence_accepts_actual_variable_expert_counts_and_allocation_rounding():
    cache, reports = residency_evidence()
    result = offline.validate_v4_residency_evidence(cache, reports, 2)
    assert result["experts"] == 3 and result["payload_bytes"] == 60 and result["planned_bytes"] == 96
    assert result["preload_complete"] is True
    assert result["runtime_expert_payload_h2d_bytes"] == 0 and result["runtime_expert_evictions"] == 0
    assert "not_all_device_transfers_or_latency" in result["scope"]
    # JSON serialization turns integer layer keys into strings; both are accepted.
    cache["plan"]["layer_plans"] = {str(key): value for key, value in cache["plan"]["layer_plans"].items()}
    assert offline.validate_v4_residency_evidence(cache, reports, 2) == result


@pytest.mark.parametrize(
    "key,value",
    [
        ("ready", False),
        ("failed", True),
        ("fallback_enabled", True),
        ("execution_policy", "ascendc"),
        ("layout", "zn_pair_lut_k256"),
        ("layer_index", 1),
        ("layer_index", False),
        ("expected_experts", 1),
        ("resident_experts", 1),
        ("payload_bytes", 19),
        ("planned_bytes", 20),
        ("preload_loads", 1),
        ("preload_h2d_bytes", 0),
        ("preload_h2d_bytes", 32),
        ("preload_evictions", 1),
        ("post_init_loads", 1),
        ("post_init_h2d_bytes", 1),
        ("post_init_evictions", 1),
        ("post_init_loads", False),
        ("post_init_h2d_bytes", 0.0),
    ],
)
def test_v4_residency_evidence_rejects_partial_or_runtime_loaded_weights(key, value):
    cache, reports = residency_evidence()
    reports["0"][key] = value
    with pytest.raises(ValueError):
        offline.validate_v4_residency_evidence(cache, reports, 2)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("plan", "preload_complete", False),
        ("plan", "all_experts_fit", False),
        ("plan", "allocation", "lazy_packed_only"),
        ("plan", "layout", "zn_pair_lut_k256"),
        ("plan", "planned_bytes", 95),
        ("plan", "budget_bytes", 95),
        ("plan", "budget_bytes", True),
        ("cache", "resident_experts", 2),
        ("cache", "loads", 4),
        ("cache", "evictions", 1),
        ("cache", "resident_packed_bytes", 59),
    ],
)
def test_v4_residency_evidence_requires_completed_whole_model_budget_and_counters(section, key, value):
    cache, reports = residency_evidence()
    (cache["plan"] if section == "plan" else cache)[key] = value
    with pytest.raises(ValueError):
        offline.validate_v4_residency_evidence(cache, reports, 2)


@pytest.mark.parametrize(
    "bad", ["missing_report", "missing_plan", "duplicate_report", "duplicate_plan", "missing_counter"]
)
def test_v4_residency_evidence_rejects_incomplete_or_duplicate_layer_identity(bad):
    cache, reports = residency_evidence()
    if bad == "missing_report":
        del reports["1"]
    elif bad == "missing_plan":
        del cache["plan"]["layer_plans"][1]
    elif bad == "duplicate_report":
        reports[0] = reports["0"]
    elif bad == "duplicate_plan":
        cache["plan"]["layer_plans"]["0"] = cache["plan"]["layer_plans"][0]
    else:
        del reports["0"]["post_init_loads"]
    with pytest.raises(ValueError):
        offline.validate_v4_residency_evidence(cache, reports, 2)


def generation_evidence():
    cache, reports = residency_evidence()
    logits = torch.zeros(4, 8)
    logits[:, 7] = 2
    return {
        "steps": [{"tokens": 3, "positions": [0, 1, 2]}]
        + [{"tokens": 1, "positions": [position]} for position in (3, 4, 5)],
        "load": {"moe_layers": 2, "registered_parameters_loaded": 4},
        "cache": cache,
        "v4": reports,
        "logits": logits,
        "peak_allocated_bytes": 1024,
        "peak_reserved_bytes": 2048,
        "root_fp8": {"mode": "bf16"},
        "expert_backend": {
            "policy": "ascendc_v4",
            "library": {"sha256": "a" * 64},
            "fallback_enabled": False,
            "layers": [
                {
                    "layer": index,
                    "steps": [
                        {
                            "tokens": rows,
                            "projection_calls": 2,
                            "projection_rows": rows * 2,
                            "expert_calls": 1,
                            "kernel_launches": 2,
                        }
                        for rows in (3, 1, 1, 1)
                    ],
                }
                for index in range(2)
            ],
        },
    }


@pytest.mark.parametrize(
    "bad", [None, "missing_residency", "runtime_reload", "wrong_library", "no_native_launch", "wrong_logits"]
)
def test_v4_generation_gate_keeps_residency_native_coverage_and_greedy_checks(bad):
    data = generation_evidence()
    if bad == "missing_residency":
        del data["v4"]
    elif bad == "runtime_reload":
        data["v4"]["0"]["post_init_loads"] = 1
    elif bad == "wrong_library":
        data["expert_backend"]["library"]["sha256"] = "b" * 64
    elif bad == "no_native_launch":
        data["expert_backend"]["layers"][0]["steps"][0]["kernel_launches"] = 0
    elif bad == "wrong_logits":
        data["logits"][0, 6] = 3
    kwargs = {"execution_policy": "ascendc_v4", "ascendc_sha256": "a" * 64}
    if bad is None:
        report = offline.validate_offline_evidence(data, [0, 1, 2], [7] * 4, 2, 8, **kwargs)
        assert report["layers_executed"] == 2 and report["finite_logits"] and report["greedy_logits_agree"]
        assert report["expert_backend"]["policy"] == "ascendc_v4"
    else:
        with pytest.raises(ValueError):
            offline.validate_offline_evidence(data, [0, 1, 2], [7] * 4, 2, 8, **kwargs)
