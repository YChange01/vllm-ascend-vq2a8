# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU mock/oracle tests; these do not execute AscendC or certify hardware."""

import ast
import copy
import inspect
import json
import weakref
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from tools import validate_vq2a8_tail_reorder as probe


def stage(_name):
    return nullcontext()


class FakeBank:
    def __init__(self, fixture):
        # Do not create fixture -> bank -> fixture cycles: owner release must
        # be observable without an unrelated Python garbage-collection pass.
        self.fixture = SimpleNamespace(**vars(fixture))

    def prepare_vectorized(self, q, scale, bias, ids):
        result = torch.empty_like(q)
        valid = torch.tensor([int(0 <= slot < probe.EXPERTS) for slot in ids.tolist()], dtype=torch.int32)
        for row, slot in enumerate(ids.tolist()):
            if valid[row]:
                result[row].view(torch.uint8).copy_(q[row].view(torch.uint8)[self.fixture.orders[slot]])
        return result, valid

    def prepare_tail(self, x, scale, bias, ids):
        q = x.clamp(-448, 448).to(torch.float8_e4m3fn)
        reordered, valid = self.prepare_vectorized(q, scale, bias, ids)
        # Deliberately retain NaNs in valid output: the probe must not demand
        # valid prepare output values, which native prepare has not initialized.
        output = torch.full((x.shape[0], probe.OUTPUT_WIDTH), float("nan"), dtype=torch.bfloat16)
        for row in range(x.shape[0]):
            if not valid[row]:
                output[row].view(torch.int16).fill_(0x7FC0)
        descriptors = torch.empty((x.shape[0], 9), dtype=torch.int64)
        prepared = (reordered, valid, descriptors, output)
        descriptors.copy_(
            torch.tensor(probe.descriptor_reference(self.fixture, (x, scale, bias, ids), ids.tolist(), prepared))
        )
        return prepared

    def project_vectorized(self, q, scale, bias, ids):
        _reordered, valid = self.prepare_vectorized(q, scale, bias, ids)
        weights = torch.arange(q.shape[1]) % 3 - 1
        result = ((q.float() * weights).sum(-1) * scale + bias).bfloat16()[:, None].repeat(1, probe.OUTPUT_WIDTH)
        for row in range(q.shape[0]):
            if not valid[row]:
                result[row].view(torch.int16).fill_(0x7FC0)
        return result, valid

    def project_tail(self, x, scale, bias, ids):
        return self.project_vectorized(x.clamp(-448, 448).to(torch.float8_e4m3fn), scale, bias, ids)


def fixture(k=2048):
    result = SimpleNamespace(
        k=k,
        device="cpu",
        orders=[torch.arange(k).roll(index + 1) for index in range(3)],
        payloads={name: [torch.zeros(32, dtype=torch.uint8) for _ in range(3)] for name in ("packed_zn", "pair_lut")},
    )
    result.bank = FakeBank(result)
    return result


def passing_result(queue=False):
    values = {
        "numeric": probe.numeric_names(),
        "invalid": probe.invalid_names(),
        "native_contract": probe.CONTRACT_CASES,
        "graph": probe.graph_names(),
    }
    if queue:
        values["queue_lifetime"] = {**probe.queue_evidence(), "task_queue_enable": "1"}
    return {
        "status": "PASS",
        "exit_code": 0,
        "reaped": True,
        "events": [
            {"event": "PASS", "stage": "final_sync"},
            {
                "event": "CASE_PASS",
                "case": probe.CASE,
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


def test_plan_roundtrip_and_no_execution_claim(tmp_path, capsys):
    destination = tmp_path / "unused"
    assert probe.main(["--plan-only", "--queue-lifetime", "--report-dir", str(destination)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "PLANNED" and not destination.exists()
    assert all(
        value[name] is False
        for name in (
            "device_execution_verified",
            "graph_verified",
            "performance_verified",
            "model_integration_verified",
        )
    )
    args = probe.parse_args(value["command"][value["command"].index("--child") :])
    assert args.child and args.queue_lifetime and args.physical_npu == 1 and args.timeout_s == 600
    tree = ast.parse(inspect.getsource(probe))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert "torch" not in ast.unparse(node)


@pytest.mark.parametrize("version", [True, False, "1", None, 0, 2])
def test_wrong_abi_fails_without_fallback(version):
    with pytest.raises(RuntimeError, match="independent ABI"):
        probe.require_abi(SimpleNamespace(activation_tail_reorder_version=lambda: version))


def test_abi_and_environment():
    probe.require_abi(SimpleNamespace(activation_tail_reorder_version=lambda: 1))
    with pytest.raises(RuntimeError, match="missing"):
        probe.require_abi(SimpleNamespace())
    for queue in ("0", "1", "2"):
        result = probe.probe_environment(
            probe.parse_args([]), {"TASK_QUEUE_ENABLE": queue, "ASCEND_RT_VISIBLE_DEVICES": "7"}
        )
        assert result["TASK_QUEUE_ENABLE"] == queue and result["ASCEND_RT_VISIBLE_DEVICES"] == "1"


@pytest.mark.parametrize("queue", [False, True])
def test_strict_receipt_and_each_coverage_family(queue):
    good = passing_result(queue)
    probe.validate_child_evidence(good, queue_lifetime=queue)
    for key in good["events"][-1]["results"]:
        bad = copy.deepcopy(good)
        del bad["events"][-1]["results"][key]
        with pytest.raises(ValueError, match="Incomplete"):
            probe.validate_child_evidence(bad, queue_lifetime=queue)
    for key in (
        "case",
        "native_abi",
        "library",
        "device_execution_verified",
        "graph_verified",
        "model_integration_verified",
        "performance_verified",
    ):
        bad = copy.deepcopy(good)
        bad["events"][-1][key] = None if key != "library" else {}
        with pytest.raises(ValueError, match="Incomplete"):
            probe.validate_child_evidence(bad, queue_lifetime=queue)


@pytest.mark.parametrize("field,value", [("status", "FAIL"), ("exit_code", 1), ("reaped", False)])
def test_failed_supervisor_cannot_pass(field, value):
    result = passing_result()
    result[field] = value
    with pytest.raises(ValueError, match="Incomplete"):
        probe.validate_child_evidence(result, queue_lifetime=False)


def test_queue_shortcuts_fail():
    good = passing_result(True)
    for key in probe.queue_evidence():
        bad = copy.deepcopy(good)
        bad["events"][-1]["results"]["queue_lifetime"].pop(key)
        with pytest.raises(ValueError, match="Incomplete"):
            probe.validate_child_evidence(bad, queue_lifetime=True)


@pytest.mark.parametrize(
    "field",
    [
        "independent_queue_banks",
        "bank_payload_upload_before_loop",
        "input_owners_dropped_before_fence",
        "bank_owners_dropped_before_fence",
        "payload_python_owners_dropped_before_fence",
    ],
)
def test_queue_owner_receipt_cannot_claim_retained_owners(field):
    bad = passing_result(True)
    bad["events"][-1]["results"]["queue_lifetime"][field] = False
    with pytest.raises(ValueError, match="Incomplete"):
        probe.validate_child_evidence(bad, queue_lifetime=True)


def test_complete_numeric_and_invalid_matrices_with_cpu_mock():
    fixtures = {k: fixture(k) for k in probe.WIDTHS}
    assert probe.run_numeric_checks(fixtures, stage) == probe.numeric_names()
    assert len(probe.numeric_names()) == 96
    assert probe.run_invalid_checks(fixtures, stage) == probe.invalid_names()


def test_special_patterns_retain_signed_zero_nan_payload_and_fp8_rounding_edges():
    value = probe.normalized_pattern(2048, 1, "special")
    bits = {item & 0xFFFFFFFF for item in value.view(torch.int32).flatten().tolist()}
    assert {
        0,
        0x80000000,
        1,
        0x80000001,
        0x7F800000,
        0xFF800000,
        0x7F800001,
        0xFF800001,
        0x7FC12345,
        0xFFC12345,
    } <= bits
    rounding = probe.normalized_pattern(2048, 1, "rounding")
    assert (rounding == 2**-10).any() and (rounding == -(2**-10)).any()
    assert torch.signbit(rounding).any()


@pytest.mark.parametrize("defect", ["bytes", "descriptor", "status", "poison", "projection", "mutation"])
def test_fault_injection_is_detected(defect):
    state = fixture()
    inputs, ids, owner = probe.inputs_fixture(state, 6, "finite", ids=[0, 1, 2, 0, 1, -1])
    original = state.bank.prepare_tail
    project = state.bank.project_tail

    def broken(*values):
        result = original(*values)
        if defect == "bytes":
            result[0][0].view(torch.uint8)[0] ^= 1
        elif defect == "descriptor":
            result[2][0, 0] += 32
        elif defect == "status":
            result[1][-1] = 1
        elif defect == "poison":
            result[3][-1, -1] = 0
        elif defect == "mutation":
            values[0][0, 0] = 3
        return result

    def broken_project(*values):
        result = project(*values)
        if defect == "projection":
            result[0][0, 0] = 12345
        return result

    state.bank.prepare_tail = broken
    state.bank.project_tail = broken_project
    with pytest.raises(AssertionError, match="differ"):
        probe.checked_case(state, inputs, ids, owner, "fault")


def test_queue_fresh_owner_release_no_per_iteration_upload_or_fence(monkeypatch):
    unrelated = fixture()
    unrelated_payloads = [tensor.clone() for tensors in unrelated.payloads.values() for tensor in tensors]
    owners, bank_owners, payload_owners, template_owners, syncs, builds, pressure = [], [], [], [], [], [], []
    monkeypatch.setattr(probe, "QUEUE_ITERATIONS", 29)
    original_prepare = FakeBank.prepare_tail
    original_inputs = probe.inputs_fixture
    original_empty = torch.empty
    factory = object()

    def make(k, device, requested_factory):
        assert not syncs and device == "cpu" and requested_factory is factory
        result = fixture(k)
        builds.append(k)
        bank_owners.append(weakref.ref(result.bank))
        payload_owners.extend(weakref.ref(tensor) for tensors in result.payloads.values() for tensor in tensors)
        return result

    def tracked_inputs(*args, **kwargs):
        assert not syncs
        inputs, ids, owner = original_inputs(*args, **kwargs)
        template_owners.extend(weakref.ref(tensor) for tensor in (*inputs, owner))
        return inputs, ids, owner

    def tracked_prepare(self, *inputs):
        assert syncs == [True]
        owners.extend(weakref.ref(tensor) for tensor in inputs)
        return original_prepare(self, *inputs)

    def tracked_empty(*args, **kwargs):
        if args == (probe.PRESSURE_BYTES,):
            pressure.append(all(owner() is None for owner in bank_owners + payload_owners))
        return original_empty(*args, **kwargs)

    def synchronize():
        assert all(owner() is None for owner in owners)
        if not syncs:
            assert all(owner() is not None for owner in bank_owners + payload_owners)
            assert not pressure
        else:
            assert all(owner() is None for owner in bank_owners + payload_owners + template_owners)
            assert pressure == [False] * probe.QUEUE_ITERATIONS + [True] * probe.POST_RELEASE_PRESSURE_CHUNKS
        syncs.append(True)

    monkeypatch.setattr(probe, "make_fixture", make)
    monkeypatch.setattr(probe, "inputs_fixture", tracked_inputs)
    monkeypatch.setattr(FakeBank, "prepare_tail", tracked_prepare)
    monkeypatch.setattr(torch, "empty", tracked_empty)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=synchronize), raising=False)
    assert probe.run_queue_checks("cpu", factory, stage) == probe.queue_evidence()
    assert syncs == [True, True]
    assert builds == list(probe.WIDTHS)
    assert unrelated.bank is not None
    for got, want in zip([tensor for tensors in unrelated.payloads.values() for tensor in tensors], unrelated_payloads):
        assert torch.equal(got, want)
    tree = ast.parse(inspect.getsource(probe.run_queue_checks))
    loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "range(QUEUE_ITERATIONS)"
    )
    body = ast.unparse(loop)
    for forbidden in (
        "make_fixture(",
        "inputs_fixture(",
        "torch.tensor(",
        ".cpu(",
        ".to(",
        ".synchronize(",
        "baseline(",
    ):
        assert forbidden not in body
    assert "pending.append((geometry," in body
    assert "SimpleNamespace(k=fixture.k, device=fixture.device)" in body
    source = inspect.getsource(probe.run_queue_checks)
    for release in ("templates.clear()", "queue_fixtures.clear()", "del fixture, template"):
        assert (
            source.index(release)
            < source.index("post_release_pressure = []")
            < source.rindex("torch.npu.synchronize()")
        )


@pytest.mark.parametrize("driver_failure", [False, True])
def test_live_graph_replays_mutations_and_recovers_without_stale_outputs(monkeypatch, driver_failure):
    state = {"capture": None, "resets": 0}

    class Stream:
        def synchronize(self):
            pass

    class Graph:
        def __init__(self):
            self.jobs = []

        def replay(self):
            if driver_failure:
                raise RuntimeError("driver replay failed")
            for method, args, outputs in self.jobs:
                updated = method(*args)
                for target, source in zip(outputs, updated):
                    target.copy_(source)
                if len(outputs) == 4:
                    # Real graph outputs keep their captured addresses.
                    f = method.__self__.fixture
                    outputs[2].copy_(torch.tensor(probe.descriptor_reference(f, args, args[-1].tolist(), outputs)))

        def reset(self):
            state["resets"] += 1

    @contextmanager
    def capture(graph, stream):
        state["capture"] = graph
        try:
            yield
        finally:
            state["capture"] = None

    def make(k, device, factory):
        result = fixture(k)
        for name in ("prepare_tail", "project_tail"):
            method = getattr(result.bank, name)

            def record(*args, method=method):
                outputs = method(*args)
                if state["capture"] is not None:
                    state["capture"].jobs.append((method, args, outputs))
                return outputs

            setattr(result.bank, name, record)
        return result

    monkeypatch.setattr(probe, "make_fixture", make)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            Stream=Stream, stream=lambda _: nullcontext(), NPUGraph=Graph, graph=capture, synchronize=lambda: None
        ),
        raising=False,
    )
    if driver_failure:
        with pytest.raises(RuntimeError, match="driver replay failed"):
            probe.run_graph_checks("cpu", None, stage)
        assert state["resets"] == 0
    else:
        assert probe.run_graph_checks("cpu", None, stage) == probe.graph_names()
        assert state["resets"] == 4


def test_missing_projection_output_cannot_pass():
    state = fixture()
    inputs, ids, owner = probe.inputs_fixture(state, 1, "finite")
    original = state.bank.project_tail
    state.bank.project_tail = lambda *values: original(*values)[:1]
    with pytest.raises(AssertionError, match="two outputs"):
        probe.checked_case(state, inputs, ids, owner, "incomplete")
