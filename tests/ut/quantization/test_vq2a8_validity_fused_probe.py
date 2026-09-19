# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acceptance orchestration regressions; CPU mocks are not hardware evidence."""

import copy
import json
import weakref
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from tools import validate_vq2a8_validity_fused as probe


def stage(_name):
    return nullcontext()


def passing_result(queue=False, reduction="scalar"):
    results = {
        "numeric": probe.numerical_case_names(),
        "invalid": {"invalid_cases": 71, "recovery_after_each": True},
        "boundaries": {
            "invalid_cases": probe.boundary_case_count(),
            "recovery_after_each": True,
            "input_immutability_checked": True,
        },
        "nonfinite_patterns": {"patterns": 256, "raw_bits_verified": True, "recovery_after_each": True},
        "native_contract": 11,
        "graph": [f"graph_g{groups}_{case}" for groups in (1, 6) for case in probe.GRAPH_CASES],
    }
    if queue:
        results["queue_lifetime"] = {
            "iterations": probe.QUEUE_ITERATIONS,
            "all_outputs_checked": True,
            "owners_dropped_before_fence": True,
            "explicit_per_iteration_synchronize": False,
            "allocation_pressure_bytes_per_iteration": probe.PRESSURE_BYTES,
            "task_queue_enable": "1",
            "input_upload_before_loop": True,
            "input_templates": probe.QUEUE_TEMPLATES,
            "fresh_device_clones": True,
        }
    return {
        "events": [
            {
                "case": probe.CASE,
                "event": "CASE_PASS",
                "native_abi": 1,
                "reduction": reduction,
                "device_execution_verified": True,
                "graph_verified": True,
                "results": results,
            }
        ]
    }


@pytest.mark.parametrize("reduction", ["scalar", "vectorized"])
def test_plan_does_not_import_npu_or_claim_execution(tmp_path, capsys, reduction):
    output = tmp_path / "unused"
    assert probe.main(["--plan-only", "--queue-lifetime", "--reduction", reduction, "--report-dir", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PLANNED" and report["native_abi"] == 1
    assert report["queue_lifetime_requested"]
    assert report["reduction"] == reduction
    for key in ("device_execution_verified", "graph_verified", "model_integration_verified", "performance_verified"):
        assert report[key] is False
    assert not output.exists()
    command = report["command"]
    child = probe.parse_args(command[command.index("--child") :])
    assert child.child and child.queue_lifetime and child.physical_npu == 1
    assert child.reduction == reduction


def test_environment_defaults_only_unset_queue_and_isolates_physical_card():
    args = probe.parse_args([])
    assert probe.probe_environment(args, {})["TASK_QUEUE_ENABLE"] == "1"
    for mode in ("0", "1", "2"):
        env = probe.probe_environment(args, {"TASK_QUEUE_ENABLE": mode, "ASCEND_RT_VISIBLE_DEVICES": "5"})
        assert env["TASK_QUEUE_ENABLE"] == mode
        assert env["ASCEND_RT_VISIBLE_DEVICES"] == "1"


@pytest.mark.parametrize("queue", [False, True])
@pytest.mark.parametrize("reduction", ["scalar", "vectorized"])
def test_complete_evidence_is_required(queue, reduction):
    good = passing_result(queue, reduction)
    probe.validate_child_evidence(good, queue_lifetime=queue, reduction=reduction)
    for key in good["events"][0]["results"]:
        bad = copy.deepcopy(good)
        bad["events"][0]["results"].pop(key)
        with pytest.raises(ValueError, match="Incomplete"):
            probe.validate_child_evidence(bad, queue_lifetime=queue, reduction=reduction)
    for key in ("case", "native_abi", "reduction", "device_execution_verified", "graph_verified"):
        bad = copy.deepcopy(good)
        bad["events"][0][key] = None
        with pytest.raises(ValueError, match="Incomplete"):
            probe.validate_child_evidence(bad, queue_lifetime=queue, reduction=reduction)


@pytest.mark.parametrize(
    "key,value",
    [
        ("iterations", 1),
        ("all_outputs_checked", False),
        ("owners_dropped_before_fence", False),
        ("explicit_per_iteration_synchronize", True),
        ("allocation_pressure_bytes_per_iteration", 0),
        ("task_queue_enable", "0"),
        ("input_upload_before_loop", False),
        ("input_templates", 0),
        ("fresh_device_clones", False),
    ],
)
def test_queue_shortcut_cannot_pass(key, value):
    bad = passing_result(True)
    bad["events"][0]["results"]["queue_lifetime"][key] = value
    with pytest.raises(ValueError, match="Incomplete"):
        probe.validate_child_evidence(bad, queue_lifetime=True)


def test_numeric_matrix_and_all_invalid_predicates_with_cpu_oracle():
    covered = set()

    def checker(statuses, outputs, flags):
        for output in outputs:
            covered.update(int(value) & 0xFFFF for value in torch.unique(output.view(torch.int16)).tolist())
        return probe.reference(statuses, outputs, flags)

    names = probe.run_numeric_checks("cpu", checker, stage)
    assert names == probe.numerical_case_names() and len(names) == 48
    assert covered == set(range(0x7F80)) | set(range(0x8000, 0xFF80))
    assert probe.run_invalid_checks("cpu", probe.reference, stage) == {"invalid_cases": 71, "recovery_after_each": True}


def test_always_true_checker_cannot_pass_invalid_gate():
    with pytest.raises(AssertionError, match="predicate mismatch"):
        probe.run_invalid_checks("cpu", lambda *args: torch.tensor(True), stage)


def test_reference_and_byte_snapshots_precede_native_to_reject_input_mutation():
    def corrupt(statuses, outputs, flags):
        outputs[0].fill_(0.25)
        return probe.reference(statuses, outputs, flags)

    with pytest.raises(AssertionError, match="mutated an input"):
        probe.assert_checked(corrupt, probe.fixture("cpu"), "mutation")


def test_vector_mask_word_and_each_row_boundary_matrix():
    covered = set()

    def checker(statuses, outputs, flags):
        for index, output in enumerate(outputs):
            indices = (~torch.isfinite(output)).reshape(-1).nonzero().flatten().tolist()
            covered.update((output.shape[-1], index, position) for position in indices)
        return probe.reference(statuses, outputs, flags)

    result = probe.run_boundary_checks("cpu", checker, stage)
    assert result == passing_result()["events"][0]["results"]["boundaries"]
    assert len(covered) == probe.boundary_case_count()
    for width in (2048, 4096):
        for index, rows in enumerate((6, 6, 1)):
            assert {position for w, i, position in covered if (w, i) == (width, index)} == set(
                probe.boundary_positions(rows, width)
            )


def test_vectorized_request_cannot_accept_scalar_receipt():
    with pytest.raises(ValueError, match="Incomplete"):
        probe.validate_child_evidence(passing_result(), queue_lifetime=False, reduction="vectorized")


def test_every_nonfinite_raw_pattern_and_recovery_with_cpu_oracle():
    covered = set()
    outcomes = []

    def checker(statuses, outputs, flags):
        bits = [value.view(torch.int16).reshape(-1) for value in outputs]
        nonfinite = torch.cat([part[~torch.isfinite(part.view(torch.bfloat16))] for part in bits])
        assert nonfinite.numel() in (0, 1)
        covered.update(int(value) & 0xFFFF for value in nonfinite.tolist())
        result = probe.reference(statuses, outputs, flags)
        outcomes.append(bool(result))
        return result

    assert probe.run_nonfinite_pattern_checks("cpu", checker, stage) == {
        "patterns": 256,
        "raw_bits_verified": True,
        "recovery_after_each": True,
    }
    assert covered == set(range(0x7F80, 0x8000)) | set(range(0xFF80, 0x10000))
    assert outcomes == [False, True] * 256


@pytest.mark.parametrize("value", [False, True])
def test_constant_checker_cannot_pass_nonfinite_pattern_gate(value):
    with pytest.raises(AssertionError, match="predicate mismatch"):
        probe.run_nonfinite_pattern_checks("cpu", lambda *args: torch.tensor(value), stage)


@pytest.mark.parametrize("key,value", [("patterns", 255), ("raw_bits_verified", False), ("recovery_after_each", False)])
def test_partial_nonfinite_evidence_cannot_pass(key, value):
    bad = passing_result()
    bad["events"][0]["results"]["nonfinite_patterns"][key] = value
    with pytest.raises(ValueError, match="Incomplete"):
        probe.validate_child_evidence(bad, queue_lifetime=False)


def test_boolean_native_abi_evidence_cannot_pass():
    bad = passing_result()
    bad["events"][0]["native_abi"] = True
    with pytest.raises(ValueError, match="Incomplete"):
        probe.validate_child_evidence(bad, queue_lifetime=False)


def test_graph_replay_reads_changed_values_and_recovers(monkeypatch):
    state = {"capture": None, "resets": 0}

    class Graph:
        def replay(self):
            self.output.copy_(probe.reference(*self.inputs))

        def reset(self):
            state["resets"] += 1

    @contextmanager
    def capture(graph):
        state["capture"] = graph
        try:
            yield
        finally:
            state["capture"] = None

    def checker(*values):
        result = probe.reference(*values)
        if state["capture"] is not None:
            state["capture"].inputs = values
            state["capture"].output = result
        return result

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(NPUGraph=Graph, graph=capture, synchronize=lambda: None), raising=False
    )
    assert probe.run_graph_checks("cpu", checker, stage) == passing_result()["events"][0]["results"]["graph"]
    assert state["resets"] == 2


def test_queue_drops_all_input_and_view_owners_before_fence(monkeypatch):
    owners = []
    syncs = []
    monkeypatch.setattr(probe, "QUEUE_ITERATIONS", 17)

    def checker(*values):
        for tensor in (tensor for part in values for tensor in part):
            owners.append(weakref.ref(tensor))
            if tensor._base is not None:
                owners.append(weakref.ref(tensor._base))
        return probe.reference(*values)

    def synchronize():
        assert all(owner() is None for owner in owners)
        syncs.append(True)

    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=synchronize), raising=False)
    result = probe.run_queue_checks("cpu", checker, stage)
    assert syncs == [True, True]
    assert result["iterations"] == 17 and result["all_outputs_checked"]
    assert result["owners_dropped_before_fence"]
    assert not result["explicit_per_iteration_synchronize"]
    assert result["input_upload_before_loop"] and result["fresh_device_clones"]
    assert result["input_templates"] == probe.QUEUE_TEMPLATES


def test_queue_iteration_body_has_no_h2d_fixture_or_host_fence():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(probe.run_queue_checks))
    loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "range(QUEUE_ITERATIONS)"
    )
    text = ast.unparse(loop)
    for forbidden in ("fixture(", "torch.tensor(", ".cpu(", ".to(", ".synchronize(", "reference("):
        assert forbidden not in text
    assert "tensor.clone()" in text
