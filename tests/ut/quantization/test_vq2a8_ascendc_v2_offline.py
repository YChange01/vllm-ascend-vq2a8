# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from tools import validate_vq2a8_tp1_offline as child
from vllm_ascend.quantization import vq2a8_ascendc_v2 as v2
from vllm_ascend.quantization import vq2a8_offline as offline


def config(tmp_path):
    plan = offline.offline_engine_options(
        tmp_path / "model",
        tmp_path / "artifact",
        execution_policy="ascendc_v2",
        ascendc_v2_library=tmp_path / "libvq2a8_ascendc_v2.so",
        ascendc_v2_sha256="a" * 64,
    )
    return NS(
        additional_config=plan["additional_config"],
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        model_config=NS(enforce_eager=True, quantization=None, dtype=torch.bfloat16, max_model_len=32),
        quant_config=None,
        scheduler_config=NS(max_num_seqs=1, max_num_batched_tokens=32),
        compilation_config=NS(mode=0, cudagraph_mode=0),
        cache_config=NS(gpu_memory_utilization=0.9),
        load_config=NS(load_format="safetensors"),
    )


def test_explicit_v2_options_do_not_change_default_or_publish_old_library_keys(tmp_path):
    cfg = config(tmp_path)
    options = offline.validate_offline_config(cfg)
    assert options["execution_policy"] == "ascendc_v2"
    assert options["cache_experts"] == 256 and options["token_chunk"] == 2
    assert not {"ascendc_library", "ascendc_sha256"} & set(options)
    original = offline.offline_engine_options(tmp_path / "model", tmp_path / "artifact")
    assert original["additional_config"]["vq2a8_offline"]["execution_policy"] == "cached"


@pytest.mark.parametrize(
    "bad",
    ["old_library", "old_sha", "old_policy", "baseline", "missing_sha", "relative", "root_fp8", "sha_type"],
)
def test_v2_config_rejects_missing_identity_and_old_new_backend_mix(tmp_path, bad):
    cfg = config(tmp_path)
    options = cfg.additional_config["vq2a8_offline"]
    if bad == "old_library":
        options["ascendc_library"] = str(tmp_path / "libvq2a8_ascendc.so")
    elif bad == "old_sha":
        options["ascendc_sha256"] = "b" * 64
    elif bad == "old_policy":
        options["execution_policy"] = "ascendc"
    elif bad == "baseline":
        options["execution_policy"] = "cached"
    elif bad == "missing_sha":
        del options["ascendc_v2_sha256"]
    elif bad == "relative":
        options["ascendc_v2_library"] = "relative.so"
    elif bad == "root_fp8":
        options["root_linear_mode"] = "online_fp8_sm90"
    elif bad == "sha_type":
        options["ascendc_v2_sha256"] = True
    with pytest.raises(ValueError):
        offline.validate_offline_config(cfg)


@pytest.mark.parametrize("policy", ["cached", "ascendc", "ascendc_v2"])
def test_option_builder_rejects_wrong_backend_library_arguments_before_discarding_them(tmp_path, policy):
    kwargs = dict(execution_policy=policy)
    if policy == "ascendc_v2":
        kwargs.update(ascendc_library=tmp_path / "old.so", ascendc_sha256="a" * 64)
    else:
        kwargs.update(ascendc_v2_library=tmp_path / "v2.so", ascendc_v2_sha256="b" * 64)
    with pytest.raises(ValueError):
        offline.offline_engine_options(tmp_path / "model", tmp_path / "artifact", **kwargs)


def test_owner_creates_v2_subclass_without_calling_the_old_runtime_constructor(monkeypatch):
    calls = []

    class V2Stub(offline.CachedVQ2TP1MoE):
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(v2, "AscendCV2VQ2TP1MoE", V2Stub)
    owner = offline.OfflineMoEOwner.__new__(offline.OfflineMoEOwner)
    owner.options = {"execution_policy": "ascendc_v2", "cache_experts": 7, "token_chunk": 2}
    owner.artifact, owner.device = object(), NS(type="npu")
    owner.layers, owner.calls = {}, {}
    layer = owner.create_layer(3)
    assert isinstance(layer, V2Stub)
    assert calls[0][0] == (owner.artifact, 3, owner.device)
    assert calls[0][1] == {"cache_experts": 7, "token_chunk": 2, "progress": True, "verbose_experts": False}
    assert owner.calls == {3: 0}


def test_owner_selects_v2_cache_accounting_not_old_layout(monkeypatch):
    owner = offline.OfflineMoEOwner.__new__(offline.OfflineMoEOwner)
    owner.options = {"execution_policy": "ascendc_v2", "cache_experts": 256}
    owner.device = NS(type="npu")
    layer = NS(layer=object(), cache_stats=lambda: {"resident_experts": 0}, cache_experts=256)
    owner.layers = {3: layer}
    calls = []
    monkeypatch.setattr(offline, "device_cache_budget", lambda *args, **kwargs: {"budget_bytes": 1234})
    monkeypatch.setattr(offline, "packed_cache_plan", lambda *args, **kwargs: pytest.fail("old cache plan called"))

    def planner(headers, budget, **kwargs):
        calls.append((headers, budget, kwargs))
        return {"planned_bytes": 1234, "layer_limits": {3: 9}}

    monkeypatch.setattr(v2, "ascendc_v2_cache_plan", planner)
    owner.configure_cache(0.9)
    assert calls == [([layer.layer], 1234, {"expert_limit": 256})]
    assert layer.cache_experts == 9


def evidence():
    logits = torch.zeros(4, 8)
    logits[:, 7] = 2
    return {
        "steps": [{"tokens": 3, "positions": [0, 1, 2]}]
        + [{"tokens": 1, "positions": [position]} for position in (3, 4, 5)],
        "load": {"moe_layers": 2, "registered_parameters_loaded": 4},
        "cache": {"layer_calls": {0: 4, 1: 4}, "resident_experts": 4, "per_layer_cache_limit": 2},
        "logits": logits,
        "peak_allocated_bytes": 1024,
        "peak_reserved_bytes": 2048,
        "root_fp8": {"mode": "bf16"},
        "expert_backend": {
            "policy": "ascendc_v2",
            "library": {"sha256": "a" * 64, "abi_version": 1},
            "fallback_enabled": False,
            "layers": [
                {
                    "layer": index,
                    "steps": [
                        {
                            "tokens": m,
                            "projection_calls": 2,
                            "projection_rows": m * 2,
                            "expert_calls": 1,
                            "kernel_launches": 2,
                        }
                        for m in (3, 1, 1, 1)
                    ],
                }
                for index in range(2)
            ],
        },
    }


@pytest.mark.parametrize("bad", [None, "policy", "hash", "layers", "profile", "launch", "fallback", "greedy"])
def test_v2_model_evidence_requires_native_generation_not_profile_or_old_backend(bad):
    data = evidence()
    backend = data["expert_backend"]
    if bad == "policy":
        backend["policy"] = "ascendc"
    elif bad == "hash":
        backend["library"]["sha256"] = "b" * 64
    elif bad == "layers":
        backend["layers"].pop()
    elif bad == "profile":
        backend["layers"][0]["steps"].insert(0, copy.deepcopy(backend["layers"][0]["steps"][0]))
    elif bad == "launch":
        backend["layers"][0]["steps"][0]["kernel_launches"] = 0
    elif bad == "fallback":
        backend["fallback_enabled"] = True
    elif bad == "greedy":
        data["logits"][0, 6] = 3
    kwargs = dict(execution_policy="ascendc_v2", ascendc_v2_sha256="a" * 64)
    if bad:
        with pytest.raises(ValueError):
            offline.validate_offline_evidence(data, [0, 1, 2], [7] * 4, 2, 8, **kwargs)
    else:
        result = offline.validate_offline_evidence(data, [0, 1, 2], [7] * 4, 2, 8, **kwargs)
        assert result["layers_executed"] == 2 and result["expert_backend"]["policy"] == "ascendc_v2"


def test_configure_v2_worker_preserves_acceptance_trace_and_requests_preset():
    calls = []
    model = NS(configure_performance_probe=lambda **kwargs: calls.append(kwargs) or {"ok": True})
    assert child.configure_v2_worker(NS(get_model=lambda: model), "batched") == {"ok": True}
    assert calls == [{"measurement": False, "compact": False, "optimization": "batched"}]


@pytest.mark.parametrize("valid", [None, False])
def test_model_evidence_consumes_deferred_intermediate_validity_before_accepting_logits(valid):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/worker/vq2a8_offline_model.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "offline_evidence"
    )
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    owner = NS(layers={3: NS(_optimization=NS(valid=valid))})
    model = NS(_offline_logits=[torch.zeros(4, 8)], model=NS(offline_owner=owner))
    with pytest.raises(ValueError, match="intermediate validity"):
        namespace["offline_evidence"](model)
