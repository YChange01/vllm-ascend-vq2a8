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


def isolated_model(*, serving=True, graph=False):
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
        "prepare_v4_graphs",
        "_check_v4_graph_memory",
        "set_v4_graph_enabled",
        "v4_graph_report",
        "performance_snapshot",
        "_forward_without_v4_graph_phase",
        "forward",
        "compute_logits",
    }
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    assert {node.name for node in cls.body} == methods
    cls.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    layers = {}
    for index in (0, 1):
        layer = AscendCV4VQ2TP1MoE.__new__(AscendCV4VQ2TP1MoE)
        layer.layer_index = index
        layer._v4_graph_enabled = False
        layer._v4_graph_is_decode = False
        layer._v4_graph_state = None
        layer._v4_graph_bypasses = {}
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
    if graph:
        model._v4_decode_graph = "moe"
        model._v4_device_route_decode = True
        model._v4_graph_kv_cache_bytes = 256 * 1024**2
        for layer in layers.values():
            layer.prepare_v4_graph = Mock()
            layer.v4_graph_report = Mock(return_value={"captures": 1, "entries": 1, "replays": 0})
            layer.set_v4_graph_enabled = Mock(
                side_effect=lambda value, target=layer: setattr(target, "_v4_graph_enabled", value)
            )
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


def test_device_route_serving_is_explicit_and_switches_once(monkeypatch, device_fence, capsys):
    from vllm_ascend.quantization import vq2a8_optimization

    model, owner, _, _ = isolated_model()
    model._v4_device_route_decode = True
    selected = []
    monkeypatch.setattr(vq2a8_optimization, "configure_runtime", lambda layer, preset, **kw: selected.append(preset))
    model.load_weights(iter(()))
    model.forward(*tokens())
    model.forward(*tokens())
    assert selected == ["device_route_decode"] * len(owner.layers)
    assert "MODEL_V4_SERVING_READY preset=device_route_decode" in capsys.readouterr().out
    device_fence.assert_called_once_with()


@pytest.mark.parametrize("valid", [True, False])
def test_device_route_validation_is_one_model_boundary_read_not_one_per_layer(monkeypatch, device_fence, valid):
    model, owner, _, _ = isolated_model()
    model.load_weights(iter(()))
    model._v4_device_route_decode = True
    for index, layer in owner.layers.items():
        layer._optimization = NS(valid=torch.tensor(valid if index == 1 else True))
    reads = []
    original_bool = torch.Tensor.__bool__
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__bool__", lambda tensor: reads.append(tensor) or original_bool(tensor))
        if valid:
            assert model.compute_logits(torch.ones(1, 2)) is model.logits_result
        else:
            with pytest.raises(ValueError, match="no output tokens are accepted"):
                model.compute_logits(torch.ones(1, 2))
            assert len(model.logits_calls) == 1
    assert len(reads) == 1


@pytest.mark.parametrize("source", ["final_hidden", "logits"])
def test_device_route_boundary_rejects_nonfinite_final_output(monkeypatch, device_fence, source):
    model, owner, _, _ = isolated_model()
    model.load_weights(iter(()))
    model._v4_device_route_decode = True
    for layer in owner.layers.values():
        layer._optimization = NS(valid=torch.tensor(True))
    if source == "final_hidden":
        model._retain_finite_flag(torch.tensor([float("nan")]))
    else:
        model.logits_result = torch.tensor([[float("inf"), 0.0]])
    reads = []
    original_bool = torch.Tensor.__bool__
    monkeypatch.setattr(torch.Tensor, "__bool__", lambda tensor: reads.append(tensor) or original_bool(tensor))
    with pytest.raises(ValueError, match="no output tokens are accepted"):
        model.compute_logits(torch.ones(1, 2))
    assert len(reads) == 1


@pytest.mark.parametrize("flag", [True, False])
def test_device_route_option_round_trip(flag):
    plan = offline.offline_engine_options(
        REPO / "model",
        REPO / "artifact",
        execution_policy="ascendc_v4",
        ascendc_library=REPO / "native.so",
        ascendc_sha256="a" * 64,
        v4_device_route_decode=flag,
    )
    cfg = config()
    cfg.additional_config = plan["additional_config"]
    assert offline.validate_offline_config(cfg).get("v4_device_route_decode", False) is flag
    assert ("v4_device_route_decode" in cfg.additional_config["vq2a8_offline"]) is flag


@pytest.mark.parametrize("flag", [None, 0, 1, "true"])
def test_device_route_option_rejects_nonbooleans(flag):
    cfg = config()
    cfg.additional_config["vq2a8_offline"]["v4_device_route_decode"] = flag
    with pytest.raises(ValueError, match="v4_device_route_decode"):
        offline.validate_offline_config(cfg)


def test_device_route_cannot_be_enabled_on_v1():
    with pytest.raises(ValueError, match="v4_device_route_decode"):
        offline.offline_engine_options(
            REPO / "model", REPO / "artifact", execution_policy="cached", v4_device_route_decode=True
        )


@pytest.fixture
def graph_model(monkeypatch, device_fence):
    from vllm_ascend.quantization import vq2a8_optimization

    model, owner, context, _ = isolated_model(graph=True)
    monkeypatch.setattr(torch.npu, "mem_get_info", Mock(return_value=(32 * 1024**3, 64 * 1024**3)), raising=False)
    monkeypatch.setattr(torch.npu, "memory_allocated", Mock(return_value=1024**3), raising=False)
    monkeypatch.setattr(torch.npu, "memory_reserved", Mock(return_value=2 * 1024**3), raising=False)
    selected = []

    def configure(layer, preset, **kwargs):
        selected.append((layer.layer_index, preset))
        layer._optimization = NS(valid=torch.tensor(True))

    monkeypatch.setattr(vq2a8_optimization, "configure_runtime", configure)
    model.load_weights(iter(()))
    return model, owner, context, selected


def test_graph_preparation_is_explicit_scratch_only_and_ready_after_all_layers(graph_model, capsys):
    model, owner, context, selected = graph_model
    calls_before = dict(owner.calls)
    caches_before = {index: layer._cache for index, layer in owner.layers.items()}
    context.attn_metadata = {"dummy": True}
    for layer in owner.layers.values():
        layer.prepare_v4_graph.side_effect = lambda: (
            pytest.fail("graph ready before all captures") if model._v4_graphs_ready else None
        )
    report = model.prepare_v4_graphs()
    assert report["ready"] and report["effective_graph_mode"] == "moe"
    assert report["full_model_graph_verified"] is False
    assert set(report["per_layer"]) == {"0", "1"}
    assert selected == [(0, "device_route_decode"), (1, "device_route_decode")]
    assert owner.calls == calls_before and model._measurement_forwards == 0
    assert model._measurement_valid is None and model.forward_calls == []
    assert model._offline_steps == [] and model._offline_logits == []
    for index, layer in owner.layers.items():
        layer.prepare_v4_graph.assert_called_once_with()
        assert layer._cache is caches_before[index] and not layer._v4_graph_is_decode
        assert layer._v4_graph_enabled
    output = capsys.readouterr().out
    assert "MODEL_V4_RUNTIME_PREPARED" in output and "MODEL_V4_SERVING_READY" not in output
    assert output.index('"layer": 1, "stage": "done"') < output.index("MODEL_V4_GRAPH_READY")
    model.prepare_v4_graphs()
    assert len(selected) == 2
    for layer in owner.layers.values():
        layer.prepare_v4_graph.assert_called_once_with()


def test_graph_dummy_nonempty_metadata_never_prepares_or_marks_real_forward(graph_model):
    model, owner, context, selected = graph_model
    context.attn_metadata = {"capture_dummy": NS(num_decodes=1, num_prefills=0)}
    model.forward_result.fill_(float("nan"))
    assert model.forward(*tokens()) is model.forward_result
    assert selected == [] and not model._v4_graphs_ready
    assert model._measurement_forwards == 0 and model._measurement_valid is None
    for layer in owner.layers.values():
        layer.prepare_v4_graph.assert_not_called()
        assert not layer._v4_graph_is_decode


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_graph_real_request_before_preparation_fails_without_lazy_capture(graph_model, phase):
    model, owner, context, selected = graph_model
    context.vq2a8_request_phase = phase
    with pytest.raises(RuntimeError, match="lazy capture is disabled"):
        model.forward(*tokens())
    assert selected == [] and model.forward_calls == []
    for layer in owner.layers.values():
        layer.prepare_v4_graph.assert_not_called()


def test_graph_prefill_decode_and_dummy_use_scheduler_phase_not_shape_or_metadata(graph_model, monkeypatch):
    model, owner, context, _ = graph_model
    model.prepare_v4_graphs()
    observed = []
    model.before_parent_forward = lambda: observed.append(
        [layer._v4_graph_is_decode for layer in owner.layers.values()]
    )
    # All four calls use the identical B1 tensors and nonempty metadata.
    for phase in ("prefill", "decode", "decode", "profile"):
        context.vq2a8_request_phase = phase
        with monkeypatch.context() as patch:
            for name in ("__bool__", "item", "cpu", "tolist"):
                patch.setattr(
                    torch.Tensor, name, lambda *a, **kw: pytest.fail("hot graph phase must not read device values")
                )
            model.forward(*tokens())
        assert not model._v4_graph_forward_active
        assert all(not layer._v4_graph_is_decode for layer in owner.layers.values())
    assert observed == [[False, False], [True, True], [True, True], [False, False]]
    assert model._measurement_forwards == 3
    context.in_profile_run = True
    context.vq2a8_request_phase = "decode"
    model.forward(*tokens())
    assert observed[-1] == [False, False] and model._measurement_forwards == 3


def test_graph_ab_switch_retains_prepared_graphs_and_runtime(graph_model):
    model, owner, context, selected = graph_model
    model.prepare_v4_graphs()
    states = {index: layer._optimization for index, layer in owner.layers.items()}
    for enabled in (False, True, False, True):
        report = model.set_v4_graph_enabled(enabled)
        assert report["effective_graph_mode"] == ("moe" if enabled else "none")
        for index, layer in owner.layers.items():
            assert layer._optimization is states[index] and layer._v4_graph_enabled is enabled
            layer.prepare_v4_graph.assert_called_once_with()
    assert len(selected) == 2


def test_graph_probe_configuration_keeps_prepared_runtime_and_rejects_preset_replacement(graph_model):
    model, owner, _, selected = graph_model
    model.prepare_v4_graphs()
    source = REPO / "vllm_ascend/patch/worker/vq2a8_offline_model.py"
    tree = ast.parse(source.read_text("utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "configure_performance_probe"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    configure = namespace["configure_performance_probe"]
    states = {index: layer._optimization for index, layer in owner.layers.items()}
    for measurement in (False, True, False):
        report = configure(model, measurement=measurement, compact=True, optimization="device_route_decode")
        assert report["graph_status"]["ready"]
        assert all(layer._optimization is states[index] for index, layer in owner.layers.items())
    assert len(selected) == 2
    for preset, profile in (("batched", False), (None, False), ("device_route_decode", True)):
        with pytest.raises(ValueError, match="graph A/B keeps device_route_decode"):
            configure(model, measurement=True, compact=True, optimization=preset, profile=profile)
    assert len(selected) == 2


def test_graph_preparation_failure_is_sticky_and_never_announces_ready(graph_model, capsys):
    model, owner, _, _ = graph_model
    owner.layers[1].prepare_v4_graph.side_effect = RuntimeError("capture failed")
    with pytest.raises(RuntimeError, match="capture failed"):
        model.prepare_v4_graphs()
    assert model._v4_graphs_failed and not model._v4_graphs_ready
    assert "MODEL_V4_GRAPH_READY" not in capsys.readouterr().out
    for enabled in (False, True):
        with pytest.raises(RuntimeError, match="healthy idle"):
            model.set_v4_graph_enabled(enabled)
    with pytest.raises(RuntimeError, match="healthy idle"):
        model.prepare_v4_graphs()
    assert model.forward_calls == []


@pytest.mark.parametrize("stage", ["before", "after_first_layer"])
def test_graph_memory_guard_retains_reserve_less_already_allocated_kv(graph_model, stage):
    model, owner, _, selected = graph_model
    model._v4_graph_reserve_bytes = 8 * 1024**3
    expected = model._v4_graph_reserve_bytes - model._v4_graph_kv_cache_bytes
    low = (expected - 1, 64 * 1024**3)
    enough = (expected, 64 * 1024**3)
    torch.npu.mem_get_info.side_effect = [low] if stage == "before" else [enough, low]
    with pytest.raises(RuntimeError, match="memory guard failed"):
        model.prepare_v4_graphs()
    assert model._v4_graphs_failed and not model._v4_graphs_ready
    assert model._v4_graph_kv_cache_bytes == 256 * 1024**2
    assert model._v4_graph_reserve_bytes == 8 * 1024**3
    assert model.v4_graph_report()["lowest_free_bytes"] == expected - 1
    owner.layers[1].prepare_v4_graph.assert_not_called()
    if stage == "before":
        assert not selected
        owner.layers[0].prepare_v4_graph.assert_not_called()
    else:
        owner.layers[0].prepare_v4_graph.assert_called_once_with()


def test_graph_memory_guard_accepts_exact_headroom_without_double_charging_kv(graph_model):
    model, _, _, _ = graph_model
    expected = model._v4_graph_reserve_bytes - model._v4_graph_kv_cache_bytes
    torch.npu.mem_get_info.return_value = (expected, 64 * 1024**3)
    report = model.prepare_v4_graphs()
    assert report["ready"] and report["lowest_free_bytes"] == expected
    assert all(sample["required_free_bytes"] == expected for sample in report["preparation_memory"].values())


def test_graph_forward_failure_clears_phase_without_eager_fallback(graph_model):
    model, owner, context, _ = graph_model
    model.prepare_v4_graphs()
    context.vq2a8_request_phase = "decode"
    model.before_parent_forward = Mock(side_effect=RuntimeError("replay failed"))
    with pytest.raises(RuntimeError, match="replay failed"):
        model.forward(*tokens())
    assert model._v4_graphs_failed and not model._v4_graph_forward_active
    assert all(not layer._v4_graph_is_decode for layer in owner.layers.values())
    with pytest.raises(RuntimeError, match="no eager fallback"):
        model.forward(*tokens())
    model.before_parent_forward.assert_called_once_with()


def test_graph_current_replay_invalidity_is_checked_at_existing_model_boundary(graph_model):
    model, owner, context, _ = graph_model
    model.prepare_v4_graphs()
    context.vq2a8_request_phase = "decode"
    model.forward(*tokens())
    model.compute_logits(model.forward_result)
    # A persistent validity buffer changes after capture and a prior good token.
    owner.layers[1]._optimization.valid.fill_(False)
    model.forward(*tokens())
    with pytest.raises(ValueError, match="no output tokens are accepted"):
        model.compute_logits(model.forward_result)
    assert model._v4_graphs_failed
    with pytest.raises(RuntimeError, match="healthy idle"):
        model.set_v4_graph_enabled(False)


def test_graph_snapshot_reports_real_per_layer_evidence_not_native_replay_counts(graph_model, monkeypatch):
    model, owner, _, _ = graph_model
    model.prepare_v4_graphs()
    owner.cache_report = Mock(return_value={"resident": True})
    for index, layer in owner.layers.items():
        layer._optimization.report = lambda: {"counter_scope": "eager_python"}
        layer._row_preparation = NS()
        layer.v4_report = Mock(return_value={"ready": True})
        layer.v4_graph_report.return_value = {"captures": 1, "entries": 1, "replays": 7 + index}
        layer.native_calls = layer.native_launches = 3
        layer.timing = dict.fromkeys(
            (
                "host_load_validate_s",
                "host_read_s",
                "host_validate_s",
                "h2d_s",
                "prepare_s",
                "packed_projection_s",
            ),
            0.0,
        )
    monkeypatch.setattr(torch.npu, "max_memory_allocated", Mock(return_value=1024**3), raising=False)
    monkeypatch.setattr(torch.npu, "max_memory_reserved", Mock(return_value=2 * 1024**3), raising=False)
    snapshot = model.performance_snapshot()
    assert snapshot["graph"] == {
        "0": {"captures": 1, "entries": 1, "replays": 7},
        "1": {"captures": 1, "entries": 1, "replays": 8},
    }
    assert snapshot["graph_status"]["effective_graph_mode"] == "moe"
    assert snapshot["native_calls"] == snapshot["native_launches"] == 6
    assert "eager_python_submissions_only" in snapshot["native_counter_scope"]


@pytest.mark.parametrize(
    "num_reqs,scheduled,computed,prompt,expected",
    [
        (1, 1, 0, 1, "prefill"),
        (1, 1, 9, 10, "prefill"),
        (1, 1, 10, 10, "decode"),
        (1, 1, 11, 10, "decode"),
        (1, 10, 0, 10, "prefill"),
        (2, 2, 10, 10, "prefill"),
    ],
)
def test_runner_marker_uses_host_request_progress(num_reqs, scheduled, computed, prompt, expected):
    source = REPO / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(source.read_text("utf-8"))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "execute_model")
    hook = next(node for node in ast.walk(method) if isinstance(node, ast.If) and "is_v4_decode =" in ast.unparse(node))
    # Select the precise small hook, not an enclosing execute_model branch.
    hooks = [node for node in ast.walk(hook) if isinstance(node, ast.If) and "is_v4_decode =" in ast.unparse(node)]
    hook = min(hooks, key=lambda node: len(ast.unparse(node)))
    context = NS()
    runner = NS(
        vllm_config=NS(additional_config={"vq2a8_offline": {"v4_decode_graph": "moe"}}),
        input_batch=NS(num_reqs=num_reqs, num_computed_tokens_cpu=[computed], num_prompt_tokens=[prompt]),
    )
    namespace = {
        "self": runner,
        "scheduler_output": NS(total_num_scheduled_tokens=scheduled),
        "get_forward_context": lambda: context,
    }
    exec(compile(ast.Module(body=[hook], type_ignores=[]), str(source), "exec"), namespace)
    assert context.vq2a8_request_phase == expected
    del context.vq2a8_request_phase
    runner.vllm_config.additional_config = {}
    exec(compile(ast.Module(body=[hook], type_ignores=[]), str(source), "exec"), namespace)
    assert not hasattr(context, "vq2a8_request_phase")
    dummy = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_dummy_run")
    assert "vq2a8_request_phase" not in ast.unparse(dummy)


def test_worker_prepares_graphs_after_warmup_before_ready_and_propagates_failure():
    source = REPO / "vllm_ascend/worker/worker.py"
    tree = ast.parse(source.read_text("utf-8"))
    method = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "compile_or_warm_up_model"
    )
    hook = next(node for node in method.body if isinstance(node, ast.If) and "prepare_v4_graphs" in ast.unparse(node))
    text = ast.unparse(method)
    assert text.index("_warm_up_atb()") < text.index("prepare_v4_graphs()") < text.index("set_random_seed(")
    prepare = Mock()
    worker = NS(
        vllm_config=NS(additional_config={"vq2a8_offline": {"v4_decode_graph": "moe"}}),
        model_runner=NS(get_model=Mock(return_value=NS(prepare_v4_graphs=prepare))),
    )
    code = compile(ast.Module(body=[hook], type_ignores=[]), str(source), "exec")
    exec(code, {"self": worker})
    prepare.assert_called_once_with()
    prepare.side_effect = RuntimeError("graph unavailable")
    with pytest.raises(RuntimeError, match="graph unavailable"):
        exec(code, {"self": worker})
    worker.vllm_config.additional_config = {}
    exec(code, {"self": worker})
    assert prepare.call_count == 2
