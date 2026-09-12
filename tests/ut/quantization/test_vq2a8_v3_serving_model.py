# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU execution of isolated serving methods; no vLLM/NPU startup emulation."""

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import regex as re
import torch

REPO = Path(__file__).resolve().parents[3]


def _offline_functions():
    source = REPO / "vllm_ascend/quantization/vq2a8_offline.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    methods = {"offline_engine_options", "validate_offline_config", "_validate_cache_memory_fraction"}
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign) or isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    scope = {"Path": Path, "torch": torch, "math": math, "re": re, "GIB": 1024**3}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), scope)
    return NS(**{name: scope[name] for name in methods})


def _config(policy="ascendc_v3", *, serving=True):
    options = {
        "enabled": True,
        "artifact": "/artifact",
        "execution_policy": policy,
        "cache_experts": 256,
        "root_linear_mode": "bf16",
        "v3_serving": serving,
    }
    library_fields = {
        "ascendc": ("ascendc_library", "ascendc_sha256"),
        "ascendc_v2": ("ascendc_v2_library", "ascendc_v2_sha256"),
        "ascendc_v3": ("ascendc_v3_library", "ascendc_v3_sha256"),
    }
    if policy in library_fields:
        path, sha = library_fields[policy]
        options.update({path: "/native.so", sha: "a" * 64})
    return NS(
        additional_config={"vq2a8_offline": options},
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        model_config=NS(enforce_eager=True, quantization=None, dtype=torch.bfloat16, max_model_len=32),
        quant_config=None,
        scheduler_config=NS(max_num_seqs=1, max_num_batched_tokens=32),
        compilation_config=NS(mode=0, cudagraph_mode=0),
        cache_config=NS(gpu_memory_utilization=0.9),
        load_config=NS(load_format="safetensors"),
    )


def _model(*, serving=True):
    source = REPO / "vllm_ascend/patch/worker/vq2a8_offline_model.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VQ2A8TP1OfflineForCausalLM"
    )
    methods = {"__init__", "load_weights", "_configure_v3_serving", "_retain_finite_flag", "forward", "compute_logits"}
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    cls.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    layers = {
        index: NS(
            measurement_mode=False,
            trace_native=True,
            native_steps=["existing state"],
            resident=object(),
        )
        for index in (0, 1)
    }
    owner = NS(
        measurement_mode=False,
        layers=layers,
        calls={0: 7, 1: 11},
        load_root=Mock(return_value=({"root.weight"}, {"strict": True})),
        configure_cache=Mock(),
    )
    context = NS(attn_metadata={"real": True})

    class Parent:
        def __init__(self, **kwargs):
            self.model = NS(offline_owner=owner)
            self.parent_init = kwargs
            self.root_weight = torch.ones(1)
            self.forward_calls = []
            self.logits_calls = []
            self.forward_result = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
            self.logits_result = torch.tensor([[2.0, 3.0]])

        def named_parameters(self):
            return iter((("root.weight", self.root_weight),))

        def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
            self.forward_calls.append((input_ids, positions, intermediate_tensors, inputs_embeds))
            return self.forward_result

        def compute_logits(self, hidden_states):
            self.logits_calls.append(hidden_states)
            return self.logits_result

    loader = object()
    scope = {
        "Parent": Parent,
        "torch": torch,
        "json": json,
        "ROOT_FP8_POLICY": "online_fp8_sm90",
        "default_weight_loader": loader,
        "get_ascend_config": lambda: NS(),
        "get_forward_context": lambda: context,
        "validate_offline_config": _offline_functions().validate_offline_config,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), "exec"), scope)
    config = _config(serving=serving)
    model = scope[cls.name](vllm_config=config, prefix="serving")
    model.configure_performance_probe = Mock(side_effect=AssertionError("serving must not invoke offline probes"))
    return model, owner, context, loader


def _tokens():
    return torch.tensor([3], dtype=torch.int64), torch.tensor([0], dtype=torch.int64)


def test_serving_flag_is_saved_without_enabling_quiet_before_strict_load():
    for serving in (True, False):
        model, owner, _, _ = _model(serving=serving)
        assert model._v3_serving is serving
        assert not model._offline_loaded and not model._offline_trace
        assert not owner.measurement_mode
        assert all(not layer.measurement_mode for layer in owner.layers.values())


def test_strict_load_enables_serving_quiet_without_probe_or_resident_reset(capsys):
    model, owner, _, loader = _model()
    resident = {index: layer.resident for index, layer in owner.layers.items()}
    weights = iter((("root.weight", torch.ones(1)),))
    loaded = model.load_weights(weights)
    assert loaded == {"root.weight"}
    owner.load_root.assert_called_once_with({"root.weight": model.root_weight}, weights, loader)
    owner.configure_cache.assert_called_once_with(0.9)
    model.configure_performance_probe.assert_not_called()
    assert model._offline_loaded and owner.measurement_mode and not model._offline_trace
    assert model._measurement_valid is None and model._measurement_forwards == 0
    assert owner.calls == {0: 7, 1: 11}
    for index, layer in owner.layers.items():
        assert layer.measurement_mode and not layer.trace_native
        assert layer.resident is resident[index]
        assert layer.native_steps == ["existing state"]
    assert capsys.readouterr().out.strip() == 'MODEL_LOAD_RESULT {"strict": true}'


def test_nonserving_strict_load_keeps_the_existing_diagnostic_mode():
    model, owner, _, _ = _model(serving=False)
    model.load_weights(iter(()))
    assert model._offline_loaded and not owner.measurement_mode
    assert all(not layer.measurement_mode and layer.trace_native for layer in owner.layers.values())
    assert not hasattr(model, "_measurement_forwards")
    model.configure_performance_probe.assert_not_called()


def test_failed_root_load_does_not_enable_serving_or_mark_weights_loaded():
    model, owner, _, _ = _model()
    owner.load_root.side_effect = ValueError("strict root mismatch")
    with pytest.raises(ValueError, match="strict root mismatch"):
        model.load_weights(iter(()))
    assert not model._offline_loaded and not owner.measurement_mode
    owner.configure_cache.assert_not_called()
    model.configure_performance_probe.assert_not_called()


def test_serving_startup_dummy_runs_parent_without_real_forward_or_finite_evidence(capsys):
    model, _, context, _ = _model()
    model.load_weights(iter(()))
    capsys.readouterr()
    model.forward_result.fill_(float("nan"))
    for metadata in (None, {}):
        context.attn_metadata = metadata
        result = model.forward(*_tokens())
        assert result is model.forward_result
        assert model._measurement_forwards == 0 and model._measurement_valid is None
    assert len(model.forward_calls) == 2
    assert not model._offline_steps and not model._offline_logits
    assert capsys.readouterr().out == ""


def test_real_serving_forward_retains_deferred_finite_flag_and_counts_only_real_calls(capsys):
    model, _, context, _ = _model()
    model.load_weights(iter(()))
    capsys.readouterr()
    model.forward(*_tokens())
    assert model._measurement_forwards == 1 and model._measurement_valid.item() is True
    model.forward_result.fill_(float("nan"))
    model.forward(*_tokens())
    assert model._measurement_forwards == 2 and model._measurement_valid.item() is False
    context.attn_metadata = None
    model.forward_result.fill_(1)
    model.forward(*_tokens())
    assert model._measurement_forwards == 2 and model._measurement_valid.item() is False
    assert capsys.readouterr().out == ""


def test_ordinary_offline_measurement_still_rejects_dummy_attention():
    model, owner, context, _ = _model(serving=False)
    model.load_weights(iter(()))
    owner.measurement_mode = True
    model._measurement_forwards = 0
    model._measurement_valid = None
    context.attn_metadata = None
    with pytest.raises(ValueError, match="require real attention metadata"):
        model.forward(*_tokens())
    assert not model.forward_calls and model._measurement_forwards == 0


def test_quiet_logits_keep_device_finite_evidence_without_logs_trace_or_tensor_bool(monkeypatch, capsys):
    model, _, _, _ = _model()
    model.load_weights(iter(()))
    capsys.readouterr()
    model.logits_result[0, 0] = float("nan")
    hidden = torch.ones((1, 2), dtype=torch.bfloat16)

    def forbidden_bool(tensor):
        raise AssertionError("quiet serving must defer tensor-to-host truth checks")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__bool__", forbidden_bool)
        output = model.compute_logits(hidden)
    assert output is model.logits_result and model.logits_calls == [hidden]
    assert model._measurement_valid.item() is False
    assert not model._offline_logits and not model._offline_steps
    assert capsys.readouterr().out == ""


def test_quiet_serving_rejects_missing_logits():
    model, _, _, _ = _model()
    model.load_weights(iter(()))
    model.logits_result = None
    with pytest.raises(ValueError, match="Missing model logits"):
        model.compute_logits(torch.ones((1, 2)))


def test_serving_does_not_bypass_loaded_weights_and_token_input_requirements():
    model, _, _, _ = _model()
    with pytest.raises(ValueError, match="loaded weights and real token IDs"):
        model.forward(*_tokens())
    model.load_weights(iter(()))
    with pytest.raises(ValueError, match="loaded weights and real token IDs"):
        model.forward(None, torch.tensor([0]))
    with pytest.raises(ValueError, match="inputs_embeds are unsupported"):
        model.forward(*_tokens(), inputs_embeds=torch.ones((1, 2)))
    assert not model.forward_calls


def test_serving_options_round_trip_only_real_booleans():
    functions = _offline_functions()
    for value in (True, False):
        plan = functions.offline_engine_options(
            Path("/model"),
            Path("/artifact"),
            execution_policy="ascendc_v3",
            ascendc_v3_library="/native.so",
            ascendc_v3_sha256="a" * 64,
            v3_serving=value,
        )
        config = _config()
        config.additional_config = plan["additional_config"]
        assert functions.validate_offline_config(config).get("v3_serving", False) is value
    for value in (None, 0, 1, "true"):
        with pytest.raises(ValueError, match="v3_serving must be boolean"):
            functions.offline_engine_options(Path("/model"), Path("/artifact"), v3_serving=value)
        with pytest.raises(ValueError, match="v3_serving must be boolean"):
            functions.validate_offline_config(_config(serving=value))


def test_serving_flag_cannot_be_used_with_other_backends():
    functions = _offline_functions()
    for policy in ("baseline", "cached", "ascendc", "ascendc_v2"):
        with pytest.raises(ValueError, match="v3_serving.*execution_policy=ascendc_v3"):
            functions.offline_engine_options(
                Path("/model"), Path("/artifact"), execution_policy=policy, v3_serving=True
            )
        for value in (True, False):
            with pytest.raises(ValueError, match="v3_serving.*execution_policy=ascendc_v3"):
                functions.validate_offline_config(_config(policy, serving=value))
