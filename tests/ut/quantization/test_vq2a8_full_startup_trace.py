# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only full-model trace contracts; no weights, vLLM engine, or NPU runs."""

import ast
import builtins
import importlib.util
import json
import math
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from tools import serve_vq2a8_v3 as server

REPO = Path(__file__).resolve().parents[3]
TRACE_PATH = REPO / "vllm_ascend/quantization/vq2a8_startup_trace.py"


@pytest.fixture
def trace_module(monkeypatch):
    spec = importlib.util.spec_from_file_location("_vq2_full_startup_trace_cpu", TRACE_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def server_argv(tmp_path):
    model = tmp_path / "model with spaces"
    (model / "experts_vq_ascend_v2").mkdir(parents=True)
    (model / "experts_vq_tp2_zn").mkdir()
    library = tmp_path / "candidate.so"
    library.write_bytes(b"CPU argument fixture only")
    return ["--model", str(model), "--library", str(library)]


def argument(command, name):
    return command[command.index(name) + 1]


def validation_function():
    # Reuse the isolated-source technique of test_vq2a8_v3_serving_model,
    # exercising the production config validator without importing vLLM.
    source = REPO / "vllm_ascend/quantization/vq2a8_offline.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"validate_offline_config", "_validate_cache_memory_fraction"}
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign) or isinstance(node, ast.FunctionDef) and node.name in names
    ]
    scope = {"Path": Path, "torch": torch, "math": math, "re": re, "GIB": 1024**3}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), scope)
    return scope["validate_offline_config"]


def config(mode="async"):
    return NS(
        additional_config={
            "vq2a8_offline": {
                "enabled": True,
                "artifact": "/artifact",
                "execution_policy": "ascendc_v3",
                "ascendc_v3_library": str(REPO / "native.so"),
                "ascendc_v3_sha256": "a" * 64,
                "cache_experts": 256,
                "root_linear_mode": "bf16",
                "v3_serving": True,
                "v3_decode_graph": "none",
                "v3_startup_trace": mode,
            }
        },
        parallel_config=NS(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            distributed_executor_backend="uni",
        ),
        model_config=NS(enforce_eager=True, quantization=None, dtype=torch.bfloat16, max_model_len=128),
        quant_config=None,
        scheduler_config=NS(max_num_seqs=1, max_num_batched_tokens=128, async_scheduling=False),
        compilation_config=NS(mode=0, cudagraph_mode=0),
        cache_config=NS(gpu_memory_utilization=0.98),
        load_config=NS(load_format="safetensors"),
    )


def test_full_startup_trace_cli_default_off_leaves_command_unchanged(server_argv):
    default = server.parse_args(server_argv)
    explicit = server.parse_args([*server_argv, "--startup-trace", "off"])
    assert default.startup_trace == explicit.startup_trace == "off"
    assert server.build_command(default) == server.build_command(explicit)
    options = json.loads(argument(server.build_command(default), "--additional-config"))["vq2a8_offline"]
    assert "v3_startup_trace" not in options


@pytest.mark.parametrize("mode", ["async", "sync"])
def test_full_startup_trace_cli_passes_opt_in_without_global_launch_blocking(server_argv, monkeypatch, mode):
    monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "1")
    args = server.parse_args([*server_argv, "--startup-trace", mode, "--physical-npu", "1"])
    command = server.build_command(args)
    additional = json.loads(argument(command, "--additional-config"))
    options = additional["vq2a8_offline"]
    assert options["v3_startup_trace"] == mode and options["v3_serving"] is True
    assert options["v3_decode_graph"] == "none"
    assert options["execution_policy"] == "ascendc_v3"
    environment = server.server_environment(args)
    assert environment["ASCEND_LAUNCH_BLOCKING"] == "0"
    assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "1"
    cfg = config(mode)
    cfg.additional_config = additional
    cfg.cache_config.kv_cache_memory_bytes = int(argument(command, "--kv-cache-memory-bytes"))
    assert validation_function()(cfg) is options


@pytest.mark.parametrize(
    "extra",
    [
        ["--startup-trace", "yes"],
        ["--startup-trace", "async", "--tensor-parallel-size", "2", "--physical-npus", "0,1"],
        ["--startup-trace", "sync", "--tensor-parallel-size", "2", "--physical-npus", "0,1"],
        ["--startup-trace", "async", "--decode-graph", "moe"],
        ["--startup-trace", "sync", "--decode-graph", "moe"],
    ],
)
def test_full_startup_trace_cli_rejects_unsupported_combinations(server_argv, extra):
    with pytest.raises(SystemExit):
        server.parse_args([*server_argv, *extra])


@pytest.mark.parametrize("mode", ["off", "async", "sync"])
def test_full_startup_trace_valid_config_preserves_options(mode):
    cfg = config(mode)
    options = cfg.additional_config["vq2a8_offline"]
    before = dict(options)
    assert validation_function()(cfg) is options
    assert options == before


@pytest.mark.parametrize("mode", [None, True, False, 0, 1, "yes", "", "ASYNC"])
def test_full_startup_trace_config_requires_explicit_named_mode(mode):
    with pytest.raises(ValueError):
        validation_function()(config(mode))


@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize("unsupported", ["tp2", "graph", "nonserving", "nonv3"])
def test_full_startup_trace_config_rejects_out_of_scope_usage(mode, unsupported):
    cfg = config(mode)
    options = cfg.additional_config["vq2a8_offline"]
    if unsupported == "tp2":
        cfg.parallel_config.tensor_parallel_size = 2
        cfg.parallel_config.distributed_executor_backend = "mp"
    elif unsupported == "graph":
        options["v3_decode_graph"] = "moe"
    elif unsupported == "nonserving":
        options["v3_serving"] = False
    else:
        options["execution_policy"] = "cached"
        for key in ("ascendc_v3_library", "ascendc_v3_sha256", "v3_serving", "v3_decode_graph"):
            options.pop(key)
    with pytest.raises(ValueError):
        validation_function()(cfg)


def read_trace(tracer):
    return [json.loads(line) for line in tracer.events_path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("mode", ["async", "sync"])
def test_full_startup_span_submission_and_completion_semantics(trace_module, tmp_path, mode):
    synced = []
    tracer = trace_module.StartupTrace(
        mode, directory=tmp_path / "trace", synchronize=lambda: synced.append(True), stack_timer=False, heartbeat_s=0
    )
    try:
        with tracer.span("unit.operation", layer=0):
            assert [e["event"] for e in read_trace(tracer)] == ["BEGIN"]
        events = read_trace(tracer)
        assert [e["event"] for e in events] == ["BEGIN", "SUBMITTED", "PASS"]
        assert synced == ([True] if mode == "sync" else [])
        assert all(e["mode"] == mode and e["stage"] == "unit.operation" for e in events)
        assert len({e["span_id"] for e in events}) == 1
        assert events[-1]["device_completion"] == ("synchronized" if mode == "sync" else "not_verified")
        assert all(e["device_completion"] == "not_verified" for e in events[:-1])
    finally:
        tracer.close()
    assert tracer.closed


def test_full_startup_nested_spans_preserve_parent_and_monotonic_sequence(trace_module, tmp_path):
    tracer = trace_module.StartupTrace("async", directory=tmp_path / "trace", stack_timer=False, heartbeat_s=0)
    try:
        with tracer.span("outer"), tracer.span("inner"):
            pass
        events = read_trace(tracer)
        assert [e["seq"] for e in events] == sorted({e["seq"] for e in events})
        begins = [e for e in events if e["event"] == "BEGIN"]
        assert len(begins) == 2 and begins[0]["span_id"] != begins[1]["span_id"]
        assert begins[0]["parent_id"] is None
        assert begins[1]["parent_id"] == begins[0]["span_id"]
        assert all(e["pid"] > 0 and e["time"] for e in events)
    finally:
        tracer.close()


@pytest.mark.parametrize("where", ["body", "sync"])
def test_full_startup_span_failure_preserves_exception_and_never_marks_pass(trace_module, tmp_path, where):
    error = RuntimeError("injected trace operation failure")

    def sync():
        if where == "sync":
            raise error

    tracer = trace_module.StartupTrace(
        "sync", directory=tmp_path / "trace", synchronize=sync, stack_timer=False, heartbeat_s=0
    )
    try:
        with pytest.raises(RuntimeError) as raised, tracer.span("failing"):
            if where == "body":
                raise error
        assert raised.value is error
        expected = ["BEGIN", "FAIL"] if where == "body" else ["BEGIN", "SUBMITTED", "FAIL"]
        assert [e["event"] for e in read_trace(tracer)] == expected
        assert read_trace(tracer)[-1]["device_completion"] == "not_verified"
    finally:
        tracer.close()


def test_full_startup_wrap_is_instance_only_and_restores_original_descriptor(trace_module, tmp_path):
    calls, sentinel = [], object()

    class Target:
        measurement_mode = True

        def operation(self, value, *, scale):
            calls.append((self, value, scale))
            return sentinel

    target, other = Target(), Target()
    original = Target.operation
    tracer = trace_module.StartupTrace("async", directory=tmp_path / "trace", stack_timer=False, heartbeat_s=0)
    try:
        tracer.wrap(target, "operation", "unit.operation")
        assert target.operation(7, scale=3) is sentinel
        assert Target.operation is original and other.operation.__func__ is original
        assert target.measurement_mode is True and other.measurement_mode is True
        assert calls == [(target, 7, 3)]
        assert "operation" in vars(target)
    finally:
        tracer.close()
    tracer.close()
    assert "operation" not in vars(target)
    assert target.operation.__func__ is original
    assert target.operation(9, scale=4) is sentinel
    assert calls[-1] == (target, 9, 4)


def test_full_startup_wrap_restores_existing_instance_override(trace_module, tmp_path):
    original = lambda value: value
    target = NS(operation=original, measurement_mode=True)
    tracer = trace_module.StartupTrace("async", directory=tmp_path / "trace", stack_timer=False, heartbeat_s=0)
    try:
        tracer.wrap(target, "operation", "instance.override", metadata=lambda value: {"value": value})
        assert target.operation(17) == 17
    finally:
        tracer.close()
    assert target.operation is original and target.measurement_mode is True


def isolated_load_weights(mode, calls, *, fail=False):
    source = REPO / "vllm_ascend/patch/worker/vq2a8_offline_model.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VQ2A8TP1OfflineForCausalLM")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load_weights")
    scope = {"json": json, "default_weight_loader": object()}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)

    def load_root(parameters, weights, loader):
        calls.append("load_root")
        if fail:
            raise ValueError("injected strict loader failure")
        assert loader is scope["default_weight_loader"]
        assert parameters == {"root.weight": "parameter-sentinel"}
        assert list(weights) == [("root.weight", "input-sentinel")]
        return {"root.weight"}, {"strict_load": True}

    def configure_cache(fraction):
        assert fraction == 0.98
        calls.append("configure_cache")

    class Model:
        load_weights = scope["load_weights"]

        def __init__(self):
            self.model = NS(offline_owner=NS(load_root=load_root, configure_cache=configure_cache))
            self._offline_root_mode = "bf16"
            self._offline_memory_fraction = 0.98
            self._offline_loaded = False
            self._v3_serving = True
            self._startup_trace_mode = mode
            self.measurement_mode = False

        def named_parameters(self):
            return iter((("root.weight", "parameter-sentinel"),))

        def _configure_v3_serving(self):
            calls.append("configure_serving")
            self.measurement_mode = True

    return Model()


def test_full_startup_trace_off_never_imports_or_installs_tracer(monkeypatch):
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name == "vllm_ascend.quantization.vq2a8_startup_trace":
            pytest.fail("Default-off loading must not import a tracer")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    calls = []
    model = isolated_load_weights("off", calls)
    loaded = model.load_weights(iter((("root.weight", "input-sentinel"),)))
    assert loaded == {"root.weight"}
    assert calls == ["load_root", "configure_cache", "configure_serving"]
    assert model._offline_loaded and model.measurement_mode
    assert not hasattr(model, "_vq2a8_startup_trace")


@pytest.mark.parametrize("mode", ["async", "sync"])
def test_full_startup_trace_installed_only_after_strict_residency_and_load_report(monkeypatch, capsys, mode):
    calls, sentinel = [], object()
    model = isolated_load_weights(mode, calls)

    def install(actual, *, mode):
        assert actual is model and mode == model._startup_trace_mode
        assert calls == ["load_root", "configure_cache", "configure_serving"]
        assert actual._offline_loaded and actual.measurement_mode
        assert 'MODEL_LOAD_RESULT {"strict_load": true}' in capsys.readouterr().out
        calls.append("install_trace")
        return sentinel

    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization.vq2a8_startup_trace", NS(install_startup_trace=install))
    assert model.load_weights(iter((("root.weight", "input-sentinel"),))) == {"root.weight"}
    assert calls[-1] == "install_trace"
    assert model._vq2a8_startup_trace is sentinel


def test_full_startup_trace_strict_load_failure_never_installs_tracer(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("No tracing setup after strict loading fails")

    monkeypatch.setitem(
        sys.modules, "vllm_ascend.quantization.vq2a8_startup_trace", NS(install_startup_trace=unexpected)
    )
    calls = []
    model = isolated_load_weights("async", calls, fail=True)
    with pytest.raises(ValueError, match="strict loader"):
        model.load_weights(iter((("root.weight", "input-sentinel"),)))
    assert calls == ["load_root"] and not model._offline_loaded
    assert not hasattr(model, "_vq2a8_startup_trace")


def toy_model(calls):
    class Echo(torch.nn.Module):
        def forward(self, hidden):
            return hidden

    class Preparation:
        def many(self, requests):
            calls.append("prepare")
            return requests

    class State:
        @contextmanager
        def scope(self, name):
            calls.append(("scope_enter", name))
            yield "original-scope-value"
            calls.append(("scope_exit", name))

    class Runtime:
        measurement_mode = True

        def __init__(self):
            self._row_preparation = Preparation()
            self._v3_prefill_state = State()

        def _bind_stream(self, stream):
            calls.append("bind_stream")

        def _launch_resident(self, inputs):
            calls.append("launch")
            return inputs

        def shared(self, hidden):
            calls.append("shared")
            return hidden

        def forward(self, hidden, input_ids=None):
            self._bind_stream("stream-sentinel")
            with self._v3_prefill_state.scope("gate_up") as value:
                assert value == "original-scope-value"
                prepared = self._row_preparation.many([(hidden, {}, NS(columns=hidden.shape[-1]))])
                self._launch_resident(prepared)
            return self.shared(hidden)

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.runtime = Runtime()

        def forward(self, hidden, input_ids=None):
            return self.runtime.forward(hidden, input_ids)

    class Decoder(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.layer_idx = index
            self.hc_attn_fn, self.hc_ffn_fn = object(), object()
            self.self_attn, self.input_layernorm, self.post_attention_layernorm = Echo(), Echo(), Echo()
            self.mlp = Adapter()

        def hc_pre(self, hidden, hc_fn):
            calls.append(("hc_pre", self.layer_idx, hc_fn))
            return hidden, "post", "comb"

        def hc_post(self, hidden, residual, post, comb):
            calls.append(("hc_post", self.layer_idx))
            return hidden

        def forward(self, hidden, input_ids=None):
            value, post, comb = self.hc_pre(hidden, self.hc_attn_fn)
            value = self.self_attn(self.input_layernorm(value))
            value = self.hc_post(value, hidden, post, comb)
            value, post, comb = self.hc_pre(value, hc_fn=self.hc_ffn_fn)
            value = self.mlp(self.post_attention_layernorm(value), input_ids=input_ids)
            return self.hc_post(value, hidden, post, comb)

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Decoder(0), Decoder(4)])
            self.offline_owner = NS(
                measurement_mode=True, layers={layer.layer_idx: layer.mlp.runtime for layer in self.layers}
            )

        def forward(self, hidden, input_ids=None):
            for layer in self.layers:
                hidden = layer(hidden, input_ids=input_ids)
            return hidden

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

        def forward(self, hidden, input_ids=None):
            return self.model(hidden, input_ids=input_ids)

        def compute_logits(self, hidden):
            return hidden

    return Model()


def test_full_startup_install_traces_original_nested_calls_and_restores_instances(trace_module, tmp_path, monkeypatch):
    calls = []
    model = toy_model(calls)
    original_model_forward = type(model).forward
    runtime = model.model.layers[0].mlp.runtime
    original_runtime_forward = type(runtime).forward
    tracer = trace_module.StartupTrace("async", directory=tmp_path / "trace", stack_timer=False, heartbeat_s=0)
    monkeypatch.setattr(trace_module, "StartupTrace", lambda mode: tracer)
    hidden, input_ids = torch.ones(2, 4), torch.tensor([0, 0])
    try:
        assert trace_module.install_startup_trace(model, mode="async") is tracer
        assert trace_module.install_startup_trace(model, mode="async") is tracer
        with pytest.raises(ValueError, match="changing mode"):
            trace_module.install_startup_trace(model, mode="sync")
        assert model(hidden, input_ids=input_ids) is hidden
        assert model.compute_logits(hidden) is hidden
        assert model.model.offline_owner.measurement_mode
        assert all(layer.mlp.runtime.measurement_mode for layer in model.model.layers)
        assert type(model).forward is original_model_forward
        assert type(runtime).forward is original_runtime_forward
        events = read_trace(tracer)
        begins = [e for e in events if e["event"] == "BEGIN"]
        stages = [e["stage"] for e in begins]
        assert stages.count("model.forward") == 1 and stages.count("decoder_model.forward") == 1
        for index in (0, 4):
            assert stages.count(f"moe.{index}.forward") == 1  # owner/runtime discovery must not double-wrap
            for stage in (
                "forward",
                "hc_pre.attention",
                "hc_post.attention",
                "hc_pre.ffn",
                "hc_post.ffn",
                "attention",
                "input_layernorm",
                "post_attention_layernorm",
                "mlp",
            ):
                assert stages.count(f"decoder.{index}.{stage}") == 1
            for stage in ("shared", "bind_stream", "launch_resident", "prepare_many", "scope.gate_up"):
                assert stages.count(f"moe.{index}.{stage}") == 1
        assert stages.count("model.compute_logits") == 1
        prepare = next(e for e in begins if e["stage"] == "moe.0.prepare_many")
        assert prepare["metadata"]["jobs"] == 1
        assert prepare["metadata"]["rows"] == [2] and prepare["metadata"]["widths"] == [4]
    finally:
        tracer.close()
    assert "forward" not in vars(model) and "forward" not in vars(runtime)
    for layer in model.model.layers:
        assert "hc_pre" not in vars(layer) and "hc_post" not in vars(layer)
        assert "many" not in vars(layer.mlp.runtime._row_preparation)
        assert "scope" not in vars(layer.mlp.runtime._v3_prefill_state)
    before = len(read_trace(tracer))
    assert model(hidden, input_ids=input_ids) is hidden
    assert len(read_trace(tracer)) == before


@pytest.mark.parametrize("mode", ["async", "sync"])
def test_full_startup_ready_is_persisted_before_first_forward_without_device_work(
    trace_module, tmp_path, monkeypatch, mode
):
    def forbidden():
        pytest.fail("Installing or closing startup tracing may not synchronize the device")

    calls = []
    model = toy_model(calls)
    tracer = trace_module.StartupTrace(
        mode, directory=tmp_path / "trace", synchronize=forbidden, stack_timer=False, heartbeat_s=0
    )
    monkeypatch.setattr(trace_module, "StartupTrace", lambda mode: tracer)
    try:
        assert trace_module.install_startup_trace(model, mode=mode) is tracer
        assert model._vq2a8_startup_trace is tracer
        assert calls == []
        events = read_trace(tracer)
        assert len(events) == 1
        ready = events[0]
        assert ready["event"] == "READY" and ready["stage"] == "startup.awaiting_worker_profile"
        assert ready["span_id"] is None and ready["parent_id"] is None
        assert ready["main_thread_id"] == trace_module.threading.main_thread().ident
        assert ready["thread_id"] == trace_module.threading.get_ident()
        assert ready["metadata"]["hooks_installed"] > 0 and not tracer._active
        assert ready["device_synchronized"] is False and ready["device_completion"] == "not_verified"
        assert ready["timing_valid"] is False
        assert trace_module.install_startup_trace(model, mode=mode) is tracer
        assert read_trace(tracer) == events
    finally:
        tracer.close()
    assert tracer.closed and calls == []
    assert read_trace(tracer) == events


def test_full_startup_tensor_metadata_does_not_read_values_or_add_sync(trace_module, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Async metadata collection may not inspect tensor contents or synchronize")

    tracer = trace_module.StartupTrace(
        "async", directory=tmp_path / "trace", synchronize=forbidden, stack_timer=False, heartbeat_s=0
    )
    hidden = torch.tensor([[float("nan"), 3.0]])
    target = NS(operation=lambda value: value)
    try:
        tracer.wrap(target, "operation", "tensor.metadata")
        with monkeypatch.context() as patch:
            for method in ("__bool__", "item", "tolist", "cpu", "numpy"):
                patch.setattr(torch.Tensor, method, forbidden)
            assert target.operation(hidden) is hidden
        metadata = read_trace(tracer)[0]["metadata"]["tensors"]["arg0"]
        assert metadata == {"shape": [1, 2], "dtype": "torch.float32", "device": "cpu"}
    finally:
        tracer.close()


def test_full_startup_stack_timer_uses_own_file_and_cancels_once(trace_module, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        trace_module.faulthandler, "dump_traceback_later", lambda *args, **kwargs: calls.append(("arm", args, kwargs))
    )
    monkeypatch.setattr(trace_module.faulthandler, "cancel_dump_traceback_later", lambda: calls.append(("cancel",)))
    tracer = trace_module.StartupTrace("async", directory=tmp_path / "trace", heartbeat_s=0)
    assert calls[0][0] == "arm" and calls[0][1] == (30,)
    assert calls[0][2]["repeat"] is True
    stack_file = calls[0][2]["file"]
    assert Path(stack_file.name) == tracer.stacks_path
    assert tracer.stacks_path.exists() and not stack_file.closed
    with pytest.raises(RuntimeError, match="Another StartupTrace"):
        trace_module.StartupTrace("async", directory=tmp_path / "another", heartbeat_s=0)
    assert len(calls) == 1  # failure must not cancel somebody else's timer
    tracer.close()
    tracer.close()
    assert calls[-1] == ("cancel",) and len(calls) == 2 and stack_file.closed
