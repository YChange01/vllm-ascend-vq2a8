# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU integration contracts; these tests never certify an NPU kernel."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_ascend.quantization import vq2a8_offline as offline


def config(options):
    return NS(
        additional_config=options["additional_config"],
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        model_config=NS(enforce_eager=True, quantization=None, dtype=torch.bfloat16, max_model_len=32),
        quant_config=None,
        scheduler_config=NS(max_num_seqs=1, max_num_batched_tokens=32),
        compilation_config=NS(mode=0, cudagraph_mode=0),
        cache_config=NS(gpu_memory_utilization=0.9),
        load_config=NS(load_format="safetensors"),
    )


def v3_options(tmp_path):
    return offline.offline_engine_options(
        tmp_path / "model",
        tmp_path / "artifact",
        execution_policy="ascendc_v3",
        ascendc_v3_library=tmp_path / "libvq2a8_ascendc_v3.so",
        ascendc_v3_sha256="b" * 64,
    )


def test_v3_integration_is_explicit_and_old_defaults_are_unchanged(tmp_path):
    before = offline.offline_engine_options(Path("/model"), Path("/artifact"))
    assert before["additional_config"]["vq2a8_offline"]["execution_policy"] == "cached"
    assert not any("v3" in key for key in before["additional_config"]["vq2a8_offline"])
    options = v3_options(tmp_path)
    value = offline.validate_offline_config(config(options))
    assert value["execution_policy"] == "ascendc_v3"
    assert value["cache_experts"] == 256 and value["root_linear_mode"] == "bf16"
    assert value["cache_reserve_gib"] == 16.0, "Do not silently consume the user's reserve"
    assert options["enforce_eager"] and options["compilation_config"]["cudagraph_mode"] == "NONE"


@pytest.mark.parametrize(
    "key,value",
    [
        ("ascendc_v3_library", "relative.so"),
        ("ascendc_v3_sha256", "not-a-hash"),
        ("root_linear_mode", "online_fp8_sm90"),
        ("cache_experts", 171),
        ("execution_policy", "ascendc"),
        ("ascendc_v2_library", "/wrong/version.so"),
    ],
)
def test_v3_integration_fails_closed_on_changed_contract(tmp_path, key, value):
    cfg = config(v3_options(tmp_path))
    cfg.additional_config["vq2a8_offline"][key] = value
    with pytest.raises(ValueError):
        offline.validate_offline_config(cfg)


@pytest.mark.parametrize("policy", ["cached", "ascendc", "ascendc_v2"])
def test_v3_integration_library_never_changes_another_backend(tmp_path, policy):
    with pytest.raises(ValueError, match="execution_policy=ascendc_v3"):
        offline.offline_engine_options(
            tmp_path, tmp_path, execution_policy=policy, ascendc_v3_library=tmp_path / "lib.so"
        )


def test_v3_integration_plans_all_layers_before_initializing(monkeypatch):
    from vllm_ascend.quantization import vq2a8_execution_v3 as v3

    owner = offline.OfflineMoEOwner.__new__(offline.OfflineMoEOwner)
    calls = []
    owner.layers = {
        index: NS(layer=index, initialize_resident=lambda *, budget_bytes, i=index: calls.append((i, budget_bytes)))
        for index in (0, 1)
    }

    def plan(layers, budget):
        assert layers == [0, 1] and not calls
        assert budget == 30
        return {"planned_bytes": 30, "layer_plans": {0: {"planned_bytes": 10}, 1: {"planned_bytes": 20}}}

    monkeypatch.setattr(v3, "resident_plan", plan)
    owner._configure_v3_residency({"budget_bytes": 30})
    assert calls == [(0, 10), (1, 20)]
    assert owner.cache_plan["planned_bytes"] == 30


def test_v3_integration_budget_failure_never_begins_partial_loading(monkeypatch):
    from vllm_ascend.quantization import vq2a8_execution_v3 as v3

    owner = offline.OfflineMoEOwner.__new__(offline.OfflineMoEOwner)
    owner.layers = {0: NS(layer=0, initialize_resident=lambda **kwargs: pytest.fail("must plan first"))}

    def insufficient(*args):
        raise ValueError("full residency exceeds budget")

    monkeypatch.setattr(v3, "resident_plan", insufficient)
    with pytest.raises(ValueError, match="full residency"):
        owner._configure_v3_residency({"budget_bytes": 1})


def test_v3_integration_resident_out_op_has_no_host_upload_or_output_allocation():
    source = (Path(__file__).resolve().parents[3] / "csrc/vq2a8_ascendc_v3/torch_binding.cpp").read_text("utf-8")
    body = source.split("void GroupedProjectionOut(", 1)[1].split("}  // namespace", 1)[0]
    assert "at::empty" not in body and "at::zeros" not in body and ".to(" not in body
    assert "LaunchGroupedV3(" in body and "[stream, blocks, descriptors, constants, owners" in body
    assert "Tensor(a!)[] owners" in source


def test_v3_integration_model_probe_retains_its_runtime_and_reports_null_timers(monkeypatch):
    from vllm_ascend.quantization.vq2a8_execution import AscendCVQ2TP1MoE

    source = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/worker/vq2a8_offline_model.py"
    tree = ast.parse(source.read_text("utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VQ2A8TP1OfflineForCausalLM")
    cls.bases = []
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in ("configure_performance_probe", "performance_snapshot")
    ]
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), "exec"), namespace)
    layer = AscendCVQ2TP1MoE.__new__(AscendCVQ2TP1MoE)
    layer.execution_policy = "ascendc_v3"
    calls = []
    layer.configure_v3_probe = lambda **options: calls.append(options)
    layer.check_resident_integrity = lambda: None
    layer.v3_validity = lambda: torch.tensor(False)
    layer.v3_report = lambda: {"ready": True, "full_model_graph_verified": False}
    layer.native_calls = layer.native_launches = layer.h2d_bytes = 0
    monkeypatch.setattr(
        torch,
        "npu",
        NS(
            synchronize=lambda: None,
            memory_allocated=lambda: 1,
            memory_reserved=lambda: 2,
            max_memory_allocated=lambda: 3,
            max_memory_reserved=lambda: 4,
            mem_get_info=lambda: (100, 200),
        ),
        raising=False,
    )
    model = namespace[cls.name]()
    model._offline_loaded, model._offline_root_mode = True, "bf16"
    model.model = NS(offline_owner=NS(layers={0: layer}, cache_report=lambda: {}))
    model.configure_performance_probe(measurement=True, compact=True, optimization="v3")
    assert calls == [{"measurement": True, "compact": True, "optimization": "v3", "profile": False}]
    assert not hasattr(layer, "_baseline_preparation"), "Legacy configure_runtime must not replace v3"
    report = model.performance_snapshot()
    assert report["finite"] is False and report["host_observed_timing"] is None
    assert report["v3"] == {"0": {"ready": True, "full_model_graph_verified": False}}
    with pytest.raises(ValueError, match="fixed resident"):
        model.configure_performance_probe(measurement=True, compact=True, optimization="fwht")
