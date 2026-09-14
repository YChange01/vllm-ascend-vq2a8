# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated CPU serving-method contracts, not a simulated vLLM/NPU server."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from tools import serve_vq2a8_v4
from vllm_ascend.quantization import vq2a8_offline as offline
from vllm_ascend.quantization.vq2a8_execution_v4 import AscendCV4VQ2TP1MoE
from vllm_ascend.quantization.vq2a8_optimization import OptimizationOptions

REPO = Path(__file__).resolve().parents[3]


def config(*, serving=True, policy="ascendc_v4"):
    options = {
        "enabled": True,
        "artifact": str(REPO / "unused-artifact"),
        "execution_policy": policy,
        "cache_experts": 256,
        "root_linear_mode": "bf16",
        "token_chunk": 2,
        "v4_serving": serving,
    }
    library_fields = {
        "ascendc": ("ascendc_library", "ascendc_sha256"),
        "ascendc_v2": ("ascendc_v2_library", "ascendc_v2_sha256"),
        "ascendc_v3": ("ascendc_v3_library", "ascendc_v3_sha256"),
        "ascendc_v4": ("ascendc_library", "ascendc_sha256"),
    }
    if policy in library_fields:
        path, sha = library_fields[policy]
        options.update({path: str(REPO / "native.so"), sha: "a" * 64})
    return NS(
        additional_config={"vq2a8_offline": options},
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        model_config=NS(enforce_eager=True, quantization=None, dtype=torch.bfloat16, max_model_len=128),
        quant_config=None,
        scheduler_config=NS(max_num_seqs=1, max_num_batched_tokens=128),
        compilation_config=NS(mode=0, cudagraph_mode=0),
        cache_config=NS(gpu_memory_utilization=0.9),
        load_config=NS(load_format="safetensors"),
    )


def isolated_model(*, serving=True):
    source = REPO / "vllm_ascend/patch/worker/vq2a8_offline_model.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VQ2A8TP1OfflineForCausalLM"
    )
    methods = {
        "__init__",
        "load_weights",
        "_configure_v3_serving",
        "_configure_v4_serving",
        "_enable_v4_serving_batched",
        "_retain_finite_flag",
        "forward",
        "compute_logits",
    }
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    assert {node.name for node in cls.body} == methods
    cls.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    layers = {}
    for index in (0, 1):
        layer = AscendCV4VQ2TP1MoE.__new__(AscendCV4VQ2TP1MoE)
        layer.root = {"gate.bias": torch.ones(2)}
        layer.token_chunk = 2
        layer.measurement_mode = False
        layer.trace_native = True
        layer.native_steps = ["existing trace"]
        layer._cache = {0: object()}
        layer.cache_loads, layer.h2d_bytes, layer.evictions = 256, 123456, 0
        layer.check_resident_integrity = Mock(return_value={"ready": True})
        layer.clear_cache = Mock(side_effect=AssertionError("serving must retain resident experts"))
        layers[index] = layer
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
            self.root_weight = torch.ones(1)
            self.parent_init = kwargs
            self.forward_calls = []
            self.logits_calls = []
            self.forward_result = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
            self.logits_result = torch.tensor([[2.0, 3.0]])
            self.before_parent_forward = lambda: None

        def named_parameters(self):
            return iter((("root.weight", self.root_weight),))

        def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
            self.before_parent_forward()
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
        "validate_offline_config": offline.validate_offline_config,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), "exec"), scope)
    model = scope[cls.name](vllm_config=config(serving=serving), prefix="serving")
    model.configure_performance_probe = Mock(side_effect=AssertionError("serving cannot invoke offline probes"))
    return model, owner, context, loader


def tokens():
    return torch.tensor([3], dtype=torch.int64), torch.tensor([0], dtype=torch.int64)


@pytest.fixture
def device_fence(monkeypatch):
    fence = Mock()
    monkeypatch.setattr(torch, "npu", NS(synchronize=fence), raising=False)
    return fence


@pytest.mark.parametrize("serving", [True, False])
def test_v4_serving_option_round_trips_without_changing_original_startup_geometry(serving):
    kwargs = {
        "execution_policy": "ascendc_v4",
        "ascendc_library": REPO / "native.so",
        "ascendc_sha256": "a" * 64,
    }
    plain = offline.offline_engine_options(REPO / "model", REPO / "artifact", **kwargs)
    explicit = offline.offline_engine_options(REPO / "model", REPO / "artifact", v4_serving=serving, **kwargs)
    before = plain["additional_config"]["vq2a8_offline"]
    after = explicit["additional_config"]["vq2a8_offline"]
    assert "v4_serving" not in before
    assert after.get("v4_serving", False) is serving
    assert {key: value for key, value in after.items() if key != "v4_serving"} == before
    assert after["token_chunk"] == 2 and after["cache_experts"] == 256
    cfg = config()
    cfg.additional_config = explicit["additional_config"]
    assert offline.validate_offline_config(cfg).get("v4_serving", False) is serving


def test_real_serving_command_additional_config_passes_model_validator(tmp_path):
    model = tmp_path / "model"
    (model / "experts_vq_ascend_v2").mkdir(parents=True)
    library = tmp_path / "libvq2a8_ascendc.so"
    library.write_bytes(b"CPU command contract only, never loaded")
    args = serve_vq2a8_v4.parse_args(["--model", str(model), "--library", str(library)])
    command = serve_vq2a8_v4.build_command(args)

    def argument(name):
        return command[command.index(name) + 1]

    cfg = config()
    cfg.additional_config = json.loads(argument("--additional-config"))
    cfg.model_config.max_model_len = int(argument("--max-model-len"))
    cfg.scheduler_config.max_num_batched_tokens = int(argument("--max-num-batched-tokens"))
    cfg.cache_config.kv_cache_memory_bytes = int(argument("--kv-cache-memory-bytes"))
    cfg.cache_config.gpu_memory_utilization = float(argument("--gpu-memory-utilization"))
    checked = offline.validate_offline_config(cfg)
    assert checked["v4_serving"] is True and checked["execution_policy"] == "ascendc_v4"
    assert checked["artifact"] == str((model / "experts_vq_ascend_v2").resolve())
    assert checked["cache_memory_fraction"] == cfg.cache_config.gpu_memory_utilization == 0.9
    assert checked["cache_reserve_gib"] == 8.0 and checked["token_chunk"] == 2
    assert cfg.model_config.max_model_len == cfg.scheduler_config.max_num_batched_tokens == 128
    assert cfg.cache_config.kv_cache_memory_bytes == 1024**3
    assert "serve" in command and "--enforce-eager" in command


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_v4_serving_requires_real_boolean_in_builder_and_config(value):
    with pytest.raises(ValueError, match="v4_serving.*boolean"):
        offline.offline_engine_options(REPO / "model", REPO / "artifact", v4_serving=value)
    with pytest.raises(ValueError, match="v4_serving.*boolean"):
        offline.validate_offline_config(config(serving=value))


@pytest.mark.parametrize("policy", ["baseline", "cached", "ascendc", "ascendc_v2", "ascendc_v3"])
def test_v4_serving_cannot_enable_other_backend_or_silently_cross_to_v3(policy):
    with pytest.raises(ValueError, match="v4_serving.*ascendc_v4"):
        offline.offline_engine_options(REPO / "model", REPO / "artifact", execution_policy=policy, v4_serving=True)
    for value in (True, False):
        with pytest.raises(ValueError, match="v4_serving.*ascendc_v4"):
            offline.validate_offline_config(config(serving=value, policy=policy))


def test_model_starts_unloaded_and_does_not_enable_quiet_or_batched_in_constructor():
    for serving in (True, False):
        model, owner, _, _ = isolated_model(serving=serving)
        assert model._v4_serving is serving
        assert model._v4_serving_batched_ready is False
        assert not model._offline_loaded and not owner.measurement_mode
        for layer in owner.layers.values():
            assert not layer.measurement_mode and layer.trace_native
            assert not hasattr(layer, "_optimization")
            layer.check_resident_integrity.assert_not_called()


def test_loaded_residency_enables_quiet_without_batched_or_counter_resets(capsys, device_fence):
    model, owner, _, loader = isolated_model()
    observed = []
    roots = iter((("root.weight", torch.ones(1)),))
    owner.load_root.side_effect = lambda *args: observed.append("roots") or ({"root.weight"}, {"strict": True})
    owner.configure_cache.side_effect = lambda *args: observed.append("resident")
    caches = {index: layer._cache for index, layer in owner.layers.items()}
    for index, layer in owner.layers.items():
        layer.check_resident_integrity.side_effect = lambda i=index: observed.append(("integrity", i))
    assert model.load_weights(roots) == {"root.weight"}
    assert observed == ["roots", "resident", ("integrity", 0), ("integrity", 1)]
    owner.load_root.assert_called_once_with({"root.weight": model.root_weight}, roots, loader)
    owner.configure_cache.assert_called_once_with(0.9)
    assert model._offline_loaded and owner.measurement_mode
    assert not model._v4_serving_batched_ready and not model._offline_trace
    assert model._measurement_valid is None and model._measurement_forwards == 0
    assert owner.calls == {0: 7, 1: 11}
    for index, layer in owner.layers.items():
        assert layer.measurement_mode and not layer.trace_native
        assert not hasattr(layer, "_optimization") and layer.token_chunk == 2
        assert layer._cache is caches[index] and layer.native_steps == ["existing trace"]
        assert (layer.cache_loads, layer.h2d_bytes, layer.evictions) == (256, 123456, 0)
        layer.clear_cache.assert_not_called()
    model.configure_performance_probe.assert_not_called()
    device_fence.assert_not_called()
    assert 'MODEL_LOAD_RESULT {"strict": true}' in capsys.readouterr().out


def test_v4_offline_default_keeps_existing_diagnostic_behavior():
    model, owner, _, _ = isolated_model(serving=False)
    model.load_weights(iter(()))
    assert model._offline_loaded and not owner.measurement_mode
    assert not model._v4_serving_batched_ready
    assert not hasattr(model, "_measurement_forwards")
    for layer in owner.layers.values():
        assert not layer.measurement_mode and layer.trace_native
        assert not hasattr(layer, "_optimization")
        layer.check_resident_integrity.assert_not_called()


@pytest.mark.parametrize("failure", ["roots", "resident"])
def test_failed_loading_never_enables_serving_or_marks_loaded(failure):
    model, owner, _, _ = isolated_model()
    if failure == "roots":
        owner.load_root.side_effect = RuntimeError("strict root failure")
    else:
        owner.configure_cache.side_effect = RuntimeError("full residency failure")
    with pytest.raises(RuntimeError, match="failure"):
        model.load_weights(iter(()))
    assert not model._offline_loaded and not owner.measurement_mode
    assert not model._v4_serving_batched_ready
    for layer in owner.layers.values():
        layer.check_resident_integrity.assert_not_called()


@pytest.mark.parametrize("invalid", ["unloaded", "fp8", "empty", "nonv4", "integrity"])
def test_quiet_configuration_rejects_incomplete_or_wrong_residency_before_enable(invalid):
    model, owner, _, _ = isolated_model()
    model._offline_loaded = True
    if invalid == "unloaded":
        model._offline_loaded = False
    elif invalid == "fp8":
        model._offline_root_mode = "online_fp8_sm90"
    elif invalid == "empty":
        owner.layers = {}
    elif invalid == "nonv4":
        owner.layers[1].execution_policy = "ascendc"
    else:
        owner.layers[1].check_resident_integrity.side_effect = RuntimeError("bad residency")
    with pytest.raises((RuntimeError, ValueError)):
        model._configure_v4_serving()
    assert owner.measurement_mode is False
    assert model._v4_serving_batched_ready is False


def test_dummy_profiles_keep_v1_geometry_and_do_not_trigger_one_time_batched(device_fence, capsys):
    model, owner, context, _ = isolated_model()
    model.load_weights(iter(()))
    capsys.readouterr()
    model.forward_result.fill_(float("nan"))
    for metadata in (None, {}, None):
        context.attn_metadata = metadata
        assert model.forward(*tokens()) is model.forward_result
    assert model._measurement_valid is None and model._measurement_forwards == 0
    assert len(model.forward_calls) == 3 and not model._v4_serving_batched_ready
    for layer in owner.layers.values():
        assert not hasattr(layer, "_optimization") and layer.token_chunk == 2
        assert layer.check_resident_integrity.call_count == 1
    device_fence.assert_not_called()
    assert capsys.readouterr().out == ""


def test_first_real_forward_configures_original_batched_once_before_parent_call(device_fence, capsys):
    model, owner, _, _ = isolated_model()
    model.load_weights(iter(()))
    capsys.readouterr()

    def before_parent():
        assert model._v4_serving_batched_ready
        for layer in owner.layers.values():
            assert layer._optimization.options == OptimizationOptions.preset("batched")
            assert layer._optimization.profile is False

    model.before_parent_forward = before_parent
    assert model.forward(*tokens()) is model.forward_result
    states = {index: layer._optimization for index, layer in owner.layers.items()}
    caches = {index: layer._cache for index, layer in owner.layers.items()}
    assert "MODEL_V4_SERVING_READY preset=batched" in capsys.readouterr().out
    for _ in range(3):
        model.forward(*tokens())
    device_fence.assert_called_once_with()
    for index, layer in owner.layers.items():
        assert layer._optimization is states[index] and layer._cache is caches[index]
        assert layer.token_chunk == 2
        assert layer.check_resident_integrity.call_count == 2  # Load boundary and first-real switch only.
        assert (layer.cache_loads, layer.h2d_bytes, layer.evictions) == (256, 123456, 0)
        layer.clear_cache.assert_not_called()
    assert model._measurement_forwards == 4 and model._measurement_valid.item() is True
    model.configure_performance_probe.assert_not_called()
    assert capsys.readouterr().out == ""


def test_first_real_switch_failure_is_not_marked_ready_or_executed(device_fence):
    model, owner, _, _ = isolated_model()
    model.load_weights(iter(()))
    owner.layers[1].check_resident_integrity.side_effect = RuntimeError("resident expert missing")
    with pytest.raises(RuntimeError, match="resident expert missing"):
        model.forward(*tokens())
    assert not model._v4_serving_batched_ready and not model.forward_calls
    assert model._measurement_forwards == 0


def test_hot_serving_has_no_tensor_bool_or_host_copies_and_no_trace_limit(monkeypatch, device_fence, capsys):
    model, owner, _, _ = isolated_model()
    model.load_weights(iter(()))
    # The one-time V1 FastMoEState construction validates constant router bias.
    # The following assertion concerns steady serving, not that startup fence.
    model.forward(*tokens())
    capsys.readouterr()
    model.forward_result[0, 0] = float("nan")
    model.logits_result[0, 0] = float("nan")
    ids, positions = tokens()
    hidden = torch.ones(1, 2)

    def forbidden(*args, **kwargs):
        raise AssertionError("steady serving must not synchronize tensor values to host")

    with monkeypatch.context() as patch:
        for name in ("__bool__", "item", "cpu", "tolist"):
            patch.setattr(torch.Tensor, name, forbidden)
        for _ in range(140):
            assert model.forward(ids, positions) is model.forward_result
            assert model.compute_logits(hidden) is model.logits_result
    assert model._measurement_forwards == 141 and model._measurement_valid.item() is False
    assert model._offline_steps == [] and model._offline_logits == []
    assert not model._offline_trace
    for layer in owner.layers.values():
        assert layer.check_resident_integrity.call_count == 2
        assert layer.native_steps == ["existing trace"]
    device_fence.assert_called_once_with()
    assert capsys.readouterr().out == ""


def test_serving_does_not_bypass_real_token_and_missing_logits_requirements(device_fence):
    model, _, _, _ = isolated_model()
    with pytest.raises(ValueError, match="loaded weights and real token IDs"):
        model.forward(*tokens())
    model.load_weights(iter(()))
    with pytest.raises(ValueError, match="loaded weights and real token IDs"):
        model.forward(None, torch.tensor([0]))
    with pytest.raises(ValueError, match="inputs_embeds are unsupported"):
        model.forward(*tokens(), inputs_embeds=torch.ones(1, 2))
    model.logits_result = None
    with pytest.raises(ValueError, match="Missing model logits"):
        model.compute_logits(torch.ones(1, 2))
    device_fence.assert_not_called()


def test_nonserving_offline_measurement_still_rejects_dummy_attention(device_fence):
    model, owner, context, _ = isolated_model(serving=False)
    model.load_weights(iter(()))
    owner.measurement_mode = True
    model._measurement_forwards = 0
    model._measurement_valid = None
    context.attn_metadata = None
    with pytest.raises(ValueError, match="require real attention metadata"):
        model.forward(*tokens())
    assert not model.forward_calls and not model._v4_serving_batched_ready
    device_fence.assert_not_called()
