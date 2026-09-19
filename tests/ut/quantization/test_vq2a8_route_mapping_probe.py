# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU probe/oracle contracts; these tests are not native NPU evidence."""

import copy
import inspect
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tools import validate_vq2a8_route_mapping as probe


@pytest.mark.parametrize("groups", range(1, 7))
@pytest.mark.parametrize("size", probe.LOOKUP_SIZES)
@pytest.mark.parametrize("pattern", probe.PATTERNS)
@pytest.mark.parametrize("offset", [0, 1])
def test_safe_clamp_oracle_matches_literal_integer_mapping(groups, size, pattern, offset):
    values, owners = probe.fixture("cpu", groups, size, pattern, offset=offset)
    actual = probe.reference(*values)
    wanted_slots, wanted_valid = probe.literal_reference(*probe.case_values(groups, size, pattern))
    expected = (torch.tensor(wanted_slots, dtype=torch.int64), torch.tensor(wanted_valid))
    probe.assert_equal(actual, expected, "cpu_oracle")
    for view, owner in zip(values, owners):
        assert view.is_contiguous() and view.storage_offset() == offset
        assert view.untyped_storage().data_ptr() == owner.untyped_storage().data_ptr()
        assert owner[-1] == 91
        if offset:
            assert owner[0] == 73


@pytest.mark.parametrize("bad_id", [probe.INT64_MIN, -1, 2, probe.INT64_MAX])
def test_out_of_range_ids_never_select_clamped_lookup_value(bad_id):
    slots, valid = probe.reference(torch.tensor([bad_id]), torch.tensor([99, 123]))
    assert slots.tolist() == [-1]
    assert valid.shape == () and valid.dtype == torch.bool and not valid


def test_lookup_slots_not_expert_ids_and_no_new_upper_bound_policy():
    ids = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    lookup = torch.tensor([256, probe.INT64_MAX], dtype=torch.int64)
    slots, valid = probe.reference(ids, lookup)
    assert slots.tolist() == [256, probe.INT64_MAX, 256, probe.INT64_MAX]
    assert bool(valid)  # The resident bank, not this mapping, checks upper bounds.
    lookup[1] = -7
    slots, valid = probe.reference(ids, lookup)
    assert slots.tolist() == [256, -7, 256, -7] and not valid


@pytest.mark.parametrize("failure", ["count", "dtype", "rank", "slots", "valid"])
def test_assert_equal_rejects_wrong_output_contract(failure):
    expected = probe.reference(torch.tensor([0, 1]), torch.tensor([5, 7]))
    actual = list(expected)
    if failure == "count":
        actual.pop()
    elif failure == "dtype":
        actual[0] = actual[0].int()
    elif failure == "rank":
        actual[1] = actual[1].reshape(1)
    elif failure == "slots":
        actual[0] = actual[0] + 1
    elif failure == "valid":
        actual[1] = ~actual[1]
    with pytest.raises(AssertionError):
        probe.assert_equal(actual, expected, failure)


def test_numeric_probe_checks_every_case_and_input_storage():
    calls = []

    def checker(*values):
        calls.append(tuple(value.numel() for value in values))
        return probe.reference(*values)

    names = probe.run_numeric_checks("cpu", checker, lambda _: nullcontext())
    assert names == probe.numerical_case_names()
    assert len(names) == len(set(names)) == len(calls) == 360


def test_numeric_probe_rejects_input_storage_mutation():
    def checker(ids, lookup):
        result = probe.reference(ids, lookup)
        lookup[-1] += 1
        return result

    with pytest.raises(AssertionError, match="input storage/sentinels modified"):
        probe.run_numeric_checks("cpu", checker, lambda _: nullcontext())


def install_fake_graphs(monkeypatch, *, fail_replay=False, mutate_input=False):
    state = SimpleNamespace(active=None, created=[], replays=0, synchronizations=0)

    class FakeGraph:
        def __init__(self):
            self.captured = None
            self.resets = 0
            state.created.append(self)

        def replay(self):
            state.replays += 1
            if fail_replay:
                raise RuntimeError("synthetic driver replay failure")
            values, outputs = self.captured
            for target, source in zip(outputs, probe.reference(*values)):
                target.copy_(source)
            if mutate_input and values[1].numel() > 1:
                # The first G1/N256 case does not select this entry. A reference
                # recomputed from the changed inputs and sentinel-only checks
                # would both incorrectly accept the mutation.
                values[1][-1] = -77

        def reset(self):
            self.resets += 1

    class Capture:
        def __init__(self, graph):
            self.graph = graph

        def __enter__(self):
            state.active = self.graph

        def __exit__(self, *args):
            state.active = None

    def synchronize():
        state.synchronizations += 1

    def checker(*values):
        outputs = probe.reference(*values)
        if state.active is not None:
            state.active.captured = (values, outputs)
        return outputs

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(NPUGraph=FakeGraph, graph=Capture, synchronize=synchronize), raising=False
    )
    return state, checker


def test_graph_probe_changes_live_inputs_and_recovers_all_geometries(monkeypatch):
    state, checker = install_fake_graphs(monkeypatch)
    assert probe.run_graph_checks("cpu", checker, lambda _: nullcontext()) == probe.graph_case_names()
    assert state.replays == 32
    assert len(state.created) == 4 and all(graph.resets == 1 for graph in state.created)


def test_failed_graph_replay_does_not_reset_in_finally(monkeypatch):
    state, checker = install_fake_graphs(monkeypatch, fail_replay=True)
    with pytest.raises(RuntimeError, match="synthetic driver replay failure"):
        probe.run_graph_checks("cpu", checker, lambda _: nullcontext())
    assert state.replays == 1
    assert len(state.created) == 1 and state.created[0].resets == 0


def test_graph_probe_rejects_unselected_internal_lookup_mutation(monkeypatch):
    state, checker = install_fake_graphs(monkeypatch, mutate_input=True)
    with pytest.raises(AssertionError, match="input storage/sentinels modified"):
        probe.run_graph_checks("cpu", checker, lambda _: nullcontext())
    assert len(state.created) == 2
    assert state.created[-1].resets == 0


def test_queue_probe_checks_all_variable_shape_outputs_without_loop_fences(monkeypatch):
    state, _ = install_fake_graphs(monkeypatch)
    monkeypatch.setattr(probe, "QUEUE_ITERATIONS", 23)
    monkeypatch.setattr(probe, "PRESSURE_BYTES", 1024)
    calls = []
    fixture_calls = []
    template_storages = set()
    original_fixture = probe.fixture
    original_tensor = torch.tensor

    def guarded_fixture(*args, **kwargs):
        assert state.synchronizations == 0
        fixture_calls.append(True)
        result = original_fixture(*args, **kwargs)
        template_storages.update(owner.untyped_storage().data_ptr() for owner in result[1])
        return result

    def guarded_tensor(*args, **kwargs):
        # Python-value Tensor construction is permitted before the preparation
        # fence and after the final fence, never within the stress loop.
        assert state.synchronizations != 1
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(probe, "fixture", guarded_fixture)
    monkeypatch.setattr(torch, "tensor", guarded_tensor)

    def checker(*values):
        assert state.synchronizations == 1
        assert all(value.untyped_storage().data_ptr() not in template_storages for value in values)
        assert all(value.storage_offset() == len(calls) % 2 for value in values)
        calls.append(True)
        return probe.reference(*values)

    result = probe.run_queue_checks("cpu", checker, lambda _: nullcontext())
    assert result == probe.queue_evidence()
    assert len(calls) == 23 and state.synchronizations == 2
    assert len(fixture_calls) == probe.QUEUE_TEMPLATE_COUNT == 30


def test_queue_templates_repeat_original_schedule_without_losing_cases():
    templates = probe.prepare_queue_templates("cpu")
    assert len(templates) == 30
    for iteration in range(513):
        template = templates[iteration % len(templates)]
        groups = iteration % 6 + 1
        size = probe.LOOKUP_SIZES[iteration % len(probe.LOOKUP_SIZES)]
        pattern = probe.PATTERNS[iteration % len(probe.PATTERNS)]
        ids, lookup = probe.case_values(groups, size, pattern)
        wanted_slots, wanted_valid = probe.literal_reference(ids, lookup)
        assert template["offset"] == iteration % 2
        assert template["expected_slots"] == wanted_slots
        assert template["expected_valid"] == wanted_valid
        for owner, values in zip(template["owners"], (ids, lookup)):
            assert owner[template["offset"] : -1].tolist() == values


def test_queue_probe_does_not_only_check_final_output(monkeypatch):
    install_fake_graphs(monkeypatch)
    monkeypatch.setattr(probe, "QUEUE_ITERATIONS", 23)
    monkeypatch.setattr(probe, "PRESSURE_BYTES", 1024)
    calls = 0

    def checker(*values):
        nonlocal calls
        slots, valid = probe.reference(*values)
        if calls == 0:
            slots = slots + 1
        calls += 1
        return slots, valid

    with pytest.raises(AssertionError, match="owner-release slots mismatch"):
        probe.run_queue_checks("cpu", checker, lambda _: nullcontext())
    assert calls == 23


def complete_evidence(queue_lifetime=False):
    values = {"numeric": probe.numerical_case_names(), "native_contract": 12, "graph": probe.graph_case_names()}
    if queue_lifetime:
        values["queue_lifetime"] = {**probe.queue_evidence(), "task_queue_enable": "1"}
    return {
        "status": "PASS",
        "exit_code": 0,
        "reaped": True,
        "events": [
            {"case": probe.CASE, "event": "PASS", "stage": "final_sync"},
            {
                "case": probe.CASE,
                "event": "CASE_PASS",
                "native_abi": 1,
                "results": values,
                "library": {"path": "/tmp/" + probe.LIBRARY_NAME, "sha256": "a" * 64},
                "device_execution_verified": True,
                "graph_verified": True,
                "model_integration_verified": False,
                "performance_verified": False,
            },
        ],
    }


@pytest.mark.parametrize("queue_lifetime", [False, True])
def test_complete_child_evidence_is_accepted(queue_lifetime):
    probe.validate_child_evidence(complete_evidence(queue_lifetime), queue_lifetime=queue_lifetime)


@pytest.mark.parametrize(
    "failure",
    [
        "empty",
        "exit_code",
        "reaped",
        "status",
        "earlier_failure",
        "case",
        "abi_bool",
        "device",
        "graph_flag",
        "model_claim",
        "performance_claim",
        "library",
        "digest",
        "numeric",
        "graph_cases",
        "contract",
        "sync",
        "extra",
    ],
)
def test_partial_or_overclaimed_evidence_is_rejected(failure):
    result = complete_evidence()
    final = result["events"][-1]
    if failure == "empty":
        result = {}
    elif failure in ("exit_code", "reaped", "status"):
        result[failure] = {"exit_code": 1, "reaped": False, "status": "TIMEOUT"}[failure]
    elif failure == "earlier_failure":
        result["events"].insert(0, {"event": "CASE_FAIL"})
    elif failure == "case":
        final["case"] = "different"
    elif failure == "abi_bool":
        final["native_abi"] = True
    elif failure == "device":
        final["device_execution_verified"] = False
    elif failure == "graph_flag":
        final["graph_verified"] = False
    elif failure == "model_claim":
        final["model_integration_verified"] = True
    elif failure == "performance_claim":
        final["performance_verified"] = True
    elif failure == "library":
        final["library"]["path"] = "/tmp/wrong.so"
    elif failure == "digest":
        final["library"]["sha256"] = "x" * 64
    elif failure == "numeric":
        final["results"]["numeric"].pop()
    elif failure == "graph_cases":
        final["results"]["graph"].pop()
    elif failure == "contract":
        final["results"]["native_contract"] = 11
    elif failure == "sync":
        result["events"].pop(0)
    elif failure == "extra":
        final["results"]["unknown"] = True
    with pytest.raises(ValueError, match="child evidence"):
        probe.validate_child_evidence(result, queue_lifetime=False)


@pytest.mark.parametrize(
    "field,value",
    [
        ("iterations", 512),
        ("all_outputs_checked", False),
        ("ordinary_outputs_checked", False),
        ("owners_dropped_before_fence", False),
        ("input_upload_before_loop", False),
        ("input_templates", 0),
        ("fresh_device_clone_owners_each_iteration", False),
        ("explicit_per_iteration_synchronize", True),
        ("allocation_pressure_bytes_per_iteration", 0),
        ("task_queue_enable", "0"),
    ],
)
def test_incomplete_queue_evidence_is_rejected(field, value):
    result = complete_evidence(True)
    result["events"][-1]["results"]["queue_lifetime"][field] = value
    with pytest.raises(ValueError, match="queue evidence"):
        probe.validate_child_evidence(result, queue_lifetime=True)


def test_queue_request_cannot_accept_non_queue_child_evidence():
    with pytest.raises(ValueError, match="child evidence"):
        probe.validate_child_evidence(complete_evidence(False), queue_lifetime=True)


def test_plan_only_has_no_device_or_model_claims(capsys):
    assert probe.main(["--plan-only", "--physical-npu", "1", "--queue-lifetime"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["scope"] == "integer_route_mapping_only" and plan["status"] == "PLANNED"
    assert plan["queue_lifetime_requested"] is True
    assert all(
        plan[key] is False
        for key in ("device_execution_verified", "graph_verified", "model_integration_verified", "performance_verified")
    )
    assert "--queue-lifetime" in plan["command"]
    assert plan["command"][plan["command"].index("--physical-npu") + 1] == "1"


@pytest.mark.parametrize("arguments", [["--physical-npu", "-1"], ["--timeout-s", "0"], ["--child", "--plan-only"]])
def test_invalid_cli_rejected(arguments):
    with pytest.raises(SystemExit):
        probe.parse_args(arguments)


def test_probe_environment_maps_requested_card_and_preserves_explicit_queue_mode():
    args = probe.parse_args(["--physical-npu", "1"])
    environment = probe.probe_environment(args, {})
    assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "1"
    assert environment["TASK_QUEUE_ENABLE"] == "1"
    assert environment["ASCEND_LAUNCH_BLOCKING"] == "0"
    assert probe.probe_environment(args, {"TASK_QUEUE_ENABLE": "2"})["TASK_QUEUE_ENABLE"] == "2"


def test_child_rejects_unsupported_queue_mode_before_imports(monkeypatch):
    args = probe.parse_args(["--child", "--physical-npu", "1", "--queue-lifetime"])
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "2")
    for name in ("enable", "dump_traceback_later", "cancel_dump_traceback_later"):
        monkeypatch.setattr(probe.faulthandler, name, lambda *args, **kwargs: None)
    events = []
    monkeypatch.setattr(probe, "emit", lambda *args, **kwargs: events.append((args, kwargs)))
    assert probe.run_case_child(args) == 1
    assert events[-1][0] == (probe.CASE, "CASE_FAIL")
    assert "TASK_QUEUE_ENABLE" in events[-1][1]["error"]


def test_parent_uses_bounded_supervisor_busy_gate_and_not_shell():
    source = inspect.getsource(probe.main)
    assert 'subprocess.run(["npu-smi", "info"]' in source and "timeout=20" in source
    assert 'state == "unknown" or (state == "busy" and not args.allow_busy)' in source
    assert "run_child(" in source and "args.timeout_s" in source
    assert "validate_child_evidence(" in source
    assert "shell=True" not in source


def test_evidence_validation_does_not_mutate_receipt():
    result = complete_evidence(True)
    before = copy.deepcopy(result)
    probe.validate_child_evidence(result, queue_lifetime=True)
    assert result == before
