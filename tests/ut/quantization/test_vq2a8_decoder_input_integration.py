# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU protocol tests for G installation and its real-model acceptance gate.

The production startup method is extracted with AST to avoid importing vLLM or
initializing NPU. Fake generation proves the validation protocol, not hardware.
"""

import ast
import copy
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools import validate_vq2a8_v4_decoder_graph as probe


@pytest.fixture
def startup(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/worker/vq2a8_offline_model.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "prepare_v4_graphs"
    )
    namespace = {"json": json}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    events = []
    recorder = object()
    fail = SimpleNamespace(error=None)

    def attach(runner):
        events.append(("attach", runner))
        runner._prepare_inputs = SimpleNamespace(profiled=True, original=runner._prepare_inputs)
        return recorder

    def wrap(_recorder, name, method):
        assert _recorder is recorder
        events.append(("wrap", name))
        return SimpleNamespace(original=method)

    def install(runner, mode):
        events.append(("install", runner))
        if fail.error is not None:
            raise fail.error
        adapter = SimpleNamespace(runner=runner, original_prepare=runner._prepare_inputs, mode=mode)
        runner._prepare_inputs = SimpleNamespace(input_plan=True, original=runner._prepare_inputs)
        return adapter

    profiling = ModuleType("vllm_ascend.quantization.vq2a8_host_profile")
    profiling.attach_host_profile, profiling.wrap_host_call = attach, wrap
    inputs = ModuleType("vllm_ascend.quantization.vq2a8_decoder_input_plan")
    inputs.install_decoder_input_plan = install
    monkeypatch.setitem(sys.modules, profiling.__name__, profiling)
    monkeypatch.setitem(sys.modules, inputs.__name__, inputs)

    def build(host_profile=True, input_mode="b1_packed"):
        bank = SimpleNamespace(decoder_input_adapter=None, failed=False, ready=True, host_profiler=None)
        model = SimpleNamespace(
            _v4_decode_graph="decoder",
            _v4_host_profile=host_profile,
            _v4_host_recorder=None,
            _v4_decoder_input_mode=input_mode,
            _v4_decoder_graph=bank,
            _v4_graphs_ready=True,
            _v4_graphs_failed=False,
            compute_logits=object(),
        )
        model._prepare_v4_decoder_graphs = lambda runner: events.append(("capture", runner))
        model.v4_graph_report = lambda: {"ready": model._v4_graphs_ready}
        return model

    return SimpleNamespace(
        call=namespace["prepare_v4_graphs"], build=build, events=events, recorder=recorder, fail=fail
    )


def test_decoder_input_integration_profiles_before_install_and_does_not_reinstall(startup):
    model, runner = startup.build(), SimpleNamespace(_prepare_inputs=object())
    original = runner._prepare_inputs
    assert startup.call(model, runner) == {"ready": True}
    adapter = model._v4_decoder_graph.decoder_input_adapter
    assert [name for name, _ in startup.events] == ["capture", "attach", "wrap", "install"]
    assert adapter.original_prepare.profiled and adapter.original_prepare.original is original
    assert model._v4_decoder_graph.host_profiler is startup.recorder
    installed = runner._prepare_inputs
    startup.call(model, runner)
    assert model._v4_decoder_graph.decoder_input_adapter is adapter
    assert runner._prepare_inputs is installed
    assert sum(name == "install" for name, _ in startup.events) == 1
    assert sum(name == "attach" for name, _ in startup.events) == 1


@pytest.mark.parametrize("cause", ["install", "different_runner", "missing_runner"])
def test_decoder_input_integration_install_failure_marks_graph_unusable(startup, cause):
    model, runner = startup.build(host_profile=False), SimpleNamespace(_prepare_inputs=object())
    if cause == "install":
        startup.fail.error = ValueError("bad G geometry")
    elif cause == "different_runner":
        startup.call(model, runner)
        runner = SimpleNamespace(_prepare_inputs=object())
    else:
        runner = None
    with pytest.raises(ValueError):
        startup.call(model, runner)
    assert model._v4_decoder_graph.failed is True
    assert model._v4_decoder_graph.ready is False
    assert model._v4_graphs_ready is False
    assert model._v4_graphs_failed is True


def test_decoder_input_integration_profile_rebind_keeps_original_rejection_semantics(startup):
    model = startup.build()
    startup.call(model, SimpleNamespace(_prepare_inputs=object()))
    with pytest.raises(ValueError, match="Host profiling cannot be rebound"):
        startup.call(model, SimpleNamespace(_prepare_inputs=object()))
    assert sum(name == "install" for name, _ in startup.events) == 1


def test_decoder_input_integration_general_mode_never_installs(startup):
    model = startup.build(input_mode="general")
    startup.call(model, SimpleNamespace(_prepare_inputs=object()))
    assert all(name != "install" for name, _ in startup.events)
    assert model._v4_decoder_graph.decoder_input_adapter is None


def output():
    return SimpleNamespace(
        token_ids=[11, 12, 13, 14],
        logprobs=[{token: SimpleNamespace(logprob=-0.125 * token) for token in range(5)} for _ in range(4)],
    )


class FakeLLM:
    def __init__(self, reference, defect=None):
        self.reference, self.defect = reference, defect
        self.enabled = self.template = True
        self.replays = self.skips = 0
        self.counters = dict.fromkeys(("fastpath_calls", "packed_uploads", "grouped_slot_calls"), 0)
        self.rpc, self.requests = [], []
        self.graph_identity = object()
        self.worker = SimpleNamespace(get_model=lambda: self)

    def collective_rpc(self, function, args=()):
        self.rpc.append((function.__name__, args))
        return [function(self.worker, *args)]

    def set_v4_decoder_input_enabled(self, enabled):
        self.enabled = enabled

    def set_v4_position_template_verification(self, enabled):
        if self.defect != "shadow_enabled" or enabled:
            self.template = enabled

    def v4_graph_report(self):
        return {
            "decoder": {
                "replays": self.replays,
                "decoder_input": dict(self.counters),
                "position_template": {
                    "original_builder_skips": self.skips,
                    "reference_verification_enabled": self.template,
                },
            }
        }

    def generate(self, prompts, params, use_tqdm):
        assert prompts == [{"prompt_token_ids": [8]}] and params is self.params and use_tqdm is False
        index = len(self.requests)
        self.requests.append((self.graph_identity, self.enabled, self.template))
        if self.defect == "generation_error" and index == 1:
            raise RuntimeError("generation failed")
        delta = len(self.reference.token_ids) - 1
        self.replays += delta - int(self.defect == "replay_count")
        if not self.template:
            self.skips += delta
        for key in self.counters:
            hit = self.enabled and self.defect != "no_hit"
            if self.defect == key:
                hit = False
            if self.defect == "general_leak" and not self.enabled:
                hit = True
            if self.defect == "general_after_leak" and index == 2:
                hit = True
            self.counters[key] += delta if hit else 0
        candidate = copy.deepcopy(self.reference)
        if self.defect == "wrong_tokens" and index == 1:
            candidate.token_ids[-1] = 99
        if self.defect == "wrong_logprobs" and index == 1:
            candidate.logprobs[-1][0].logprob += 0.01
        return [SimpleNamespace(outputs=[candidate])]


@pytest.mark.parametrize("template", [False, True])
def test_decoder_input_integration_serving_same_graph_general_packed_general(template):
    reference = output()
    llm = FakeLLM(reference)
    llm.params = object()
    error, rows = probe.verify_input_serving_path(llm, [8], llm.params, reference, lambda x: x[0], template)
    assert error == 0.0
    assert [row["mode"] for row in rows] == ["general_before", "packed", "general_after"]
    assert [row["fastpath_calls"] for row in rows] == [0, 3, 0]
    assert [enabled for _, enabled, _ in llm.requests] == [False, True, False]
    assert all(graph is llm.graph_identity for graph, _, _ in llm.requests)
    assert not any(name == "graph_switch" for name, _ in llm.rpc)
    if template:
        assert all(shadow is False for _, _, shadow in llm.requests)
    assert llm.enabled and llm.template


@pytest.mark.parametrize(
    "defect",
    [
        "no_hit",
        "fastpath_calls",
        "packed_uploads",
        "grouped_slot_calls",
        "general_leak",
        "general_after_leak",
        "shadow_enabled",
        "replay_count",
        "wrong_tokens",
        "wrong_logprobs",
        "generation_error",
    ],
)
def test_decoder_input_integration_serving_failures_restore_both_toggles(defect):
    reference = output()
    llm = FakeLLM(reference, defect)
    llm.params = object()
    with pytest.raises(RuntimeError if defect == "generation_error" else AssertionError):
        probe.verify_input_serving_path(llm, [8], llm.params, reference, lambda x: x[0], template=True)
    assert llm.enabled is True and llm.template is True
    assert llm.rpc[-2:] == [("input_switch", (True,)), ("set_template_verification", (True,))]


def receipt_fixture(template=False):
    args = SimpleNamespace(
        route_mapping="fused",
        runtime_guard="planned",
        select_sign="fused",
        activation_tail="torch",
        validity_mode="fused_vectorized",
        activation_reorder="vectorized",
        decoder_metadata_mode="position_template" if template else "planned",
        decoder_input_mode="b1_packed",
    )
    options = {name: getattr(args, name) for name in vars(args)}
    count = probe.REUSE_ROUNDS * sum(n - 1 for _, n in probe.CASES)
    cases = []
    for round_id in range(probe.REUSE_ROUNDS):
        for prompt, tokens in probe.CASES:
            comparisons = []
            for name, enabled in (("general_before", False), ("packed", True), ("general_after", False)):
                comparisons.append(
                    {
                        "mode": name,
                        "replays": tokens - 1,
                        "template_shadow_enabled": False,
                        **dict.fromkeys(
                            ("fastpath_calls", "packed_uploads", "grouped_slot_calls"), tokens - 1 if enabled else 0
                        ),
                    }
                )
            cases.append(
                {
                    "round": round_id,
                    "prompt": prompt,
                    "output": tokens,
                    "max_logprob_error": 0.0,
                    "input_fastpath_calls": tokens - 1,
                    "input_serving_comparison": comparisons,
                }
            )
    decoder = {
        "replays": count * (5 if template else 4),
        "decoder_input": {
            "mode": "b1_packed",
            "enabled": True,
            "scope": "block_table_upload_and_grouped_slot_mapping",
            "general_prepare_inputs_preserved": True,
            "dynamic_rows_cached": False,
            "fastpath_calls": count * 2,
            "packed_uploads": count * 2,
            "grouped_slot_calls": count * 2,
        },
    }
    if template:
        decoder["position_template"] = {
            "reference_verification_enabled": True,
            "reference_checks": count,
            "reference_positions": sorted({p + step for p, n in probe.CASES for step in range(n - 1)}),
            "original_builder_skips": count * 4,
        }
    receipt = {
        "status": "PASS",
        "hardware_execution_verified": True,
        "cases": cases,
        "graph": {**options, "decoder": decoder},
        **options,
    }
    return args, receipt


@pytest.mark.parametrize("template", [False, True])
def test_decoder_input_integration_complete_receipt(template):
    probe.validate_receipt(*receipt_fixture(template))


@pytest.mark.parametrize("key", ["input_serving_comparison", "input_fastpath_calls"])
@pytest.mark.parametrize("index", range(probe.REUSE_ROUNDS * len(probe.CASES)))
def test_decoder_input_integration_each_case_requires_g_evidence(key, index):
    args, receipt = receipt_fixture()
    del receipt["cases"][index][key]
    with pytest.raises(ValueError):
        probe.validate_receipt(args, receipt)


@pytest.mark.parametrize("mode_index", range(3))
@pytest.mark.parametrize(
    "field", ["mode", "replays", "template_shadow_enabled", "fastpath_calls", "packed_uploads", "grouped_slot_calls"]
)
def test_decoder_input_integration_rejects_corrupt_comparison(mode_index, field):
    args, receipt = receipt_fixture()
    evidence = receipt["cases"][-1]["input_serving_comparison"][mode_index]
    evidence[field] = "wrong" if field == "mode" else True if field == "template_shadow_enabled" else 12345
    with pytest.raises(ValueError):
        probe.validate_receipt(args, receipt)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "round", "prompt", "output", "order"])
def test_decoder_input_integration_requires_exact_case_matrix(mutation):
    args, receipt = receipt_fixture()
    cases = receipt["cases"]
    if mutation == "missing":
        cases.pop()
    elif mutation == "duplicate":
        cases[-1] = copy.deepcopy(cases[0])
    elif mutation == "order":
        cases.reverse()
    else:
        cases[0][mutation] = 999
    with pytest.raises(ValueError):
        probe.validate_receipt(args, receipt)


@pytest.mark.parametrize(
    "field",
    [
        "mode",
        "enabled",
        "scope",
        "general_prepare_inputs_preserved",
        "dynamic_rows_cached",
        "packed_uploads",
        "grouped_slot_calls",
    ],
)
def test_decoder_input_integration_requires_final_adapter_evidence(field):
    args, receipt = receipt_fixture()
    del receipt["graph"]["decoder"]["decoder_input"][field]
    with pytest.raises(ValueError):
        probe.validate_receipt(args, receipt)
