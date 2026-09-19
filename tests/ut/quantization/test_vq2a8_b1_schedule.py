# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU scheduling/validator tests only; never emulate native acceptance."""

import copy
import inspect
import json
import weakref
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import validate_vq2a8_b1_schedule as probe


def stage(_):
    return nullcontext()


class FakeBank:
    def __init__(self, state):
        self.state = SimpleNamespace(**vars(state))

    def project_vectorized(self, q, scale, bias, ids):
        valid = torch.tensor([int(0 <= slot < probe.EXPERTS) for slot in ids.tolist()], dtype=torch.int32)
        values = ((q.float() * (torch.arange(q.shape[-1]) % 3 - 1)).sum(-1) * scale + bias).bfloat16()
        output = values[..., None].repeat(*([1] * values.ndim), probe.common.OUTPUT_WIDTH)
        for row, value in enumerate(valid):
            if not value:
                output[row].view(torch.int16).fill_(0x7FC0)
        return output, valid

    def project_candidate(self, q, scale, bias, ids, chunks, schedule):
        assert chunks in (0, 2, 4) and schedule == 1
        return self.project_vectorized(q, scale, bias, ids)


def fixture(k=2048):
    result = SimpleNamespace(
        k=k,
        device="cpu",
        payloads={name: [torch.zeros(32, dtype=torch.uint8) for _ in range(3)] for name in ("packed_zn", "pair_lut")},
    )
    result.bank = FakeBank(result)
    return result


def passing_result(queue=False, chunks=0):
    values = {
        "numeric": probe.numeric_names(),
        "invalid": probe.invalid_names(),
        "native_contract": probe.CONTRACT_CASES,
        "graph": probe.graph_names(),
    }
    if queue:
        values["queue_lifetime"] = {**probe.queue_evidence(chunks), "task_queue_enable": "1"}
    events = [{"event": "PASS", "stage": "final_sync"}]
    if queue:
        events.append({"event": "PASS", "stage": "queue_lifetime_verify"})
    events.append(
        {
            "event": "CASE_PASS",
            "case": probe.CASE,
            "native_abi": 1,
            "results": values,
            "dispatch_scope": probe.DISPATCH_SCOPE,
            "reorder_chunks": chunks,
            "library": {"path": "/tmp/" + probe.LIBRARY_NAME, "sha256": "b" * 64},
            "device_execution_verified": True,
            "graph_verified": True,
            "model_integration_verified": False,
            "performance_verified": False,
        }
    )
    return {"status": "PASS", "exit_code": 0, "reaped": True, "events": events}


@pytest.mark.parametrize("jobs", range(1, 7))
@pytest.mark.parametrize("cores", (1, 2, 4, 7, 16, 20, 24, 25, 32, 48, 64))
def test_b1_schedule_bijection_and_identical_peer_skip(jobs, cores):
    tiles = 32
    mapped = lambda work: (work % jobs) * tiles + work // jobs
    scheduled = [[mapped(work) for work in range(core, jobs * tiles, cores)] for core in range(cores)]
    flat = [work for lane in scheduled for work in lane]
    assert sorted(flat) == list(range(jobs * tiles))
    assert len(flat) == len(set(flat))
    assert [mapped(work) for work in range(jobs)] == [route * tiles for route in range(jobs)]
    for invalid_mask in range(1 << jobs):
        for core in range(cores):
            cube = [work for work in scheduled[core] if not invalid_mask & (1 << (work // tiles))]
            for peer in (core * 2, core * 2 + 1):
                vector = [
                    mapped(work)
                    for work in range(peer // 2, jobs * tiles, cores)
                    if not invalid_mask & (1 << (mapped(work) // tiles))
                ]
                assert cube == vector


def test_b1_schedule_native_source_only_changes_work_mapping():
    root = Path(__file__).resolve().parents[3]
    kernel = (root / "csrc/vq2a8_ascendc_v4_v2/kernel.cpp").read_text()
    header = (root / "csrc/vq2a8_ascendc_v4_v2/b1_schedule.h").read_text()
    assert "return (work % jobs) * nTiles + work / jobs;" in header
    assert "B1TileMajor ? B1TileMajorWork(work, jobs, nTiles) : work" in kernel
    assert "const uint32_t base = descriptorWork / nTiles * kJobWords;" in kernel
    assert "const uint32_t nBegin = (descriptorWork % nTiles) * kN;" in kernel
    process = kernel[kernel.index("void Process(") : kernel.index("private:", kernel.index("void Process("))]
    assert process.index("if (m_ == 0) continue;") < process.index("x_.SetGlobalBuffer")
    assert "core /= 2;" in process
    assert "AscendCV2Projection<> kernel" in kernel
    assert "AscendCV2Projection<true> kernel" in kernel
    assert kernel.count("Mmad(cL0_.Get<float>()") == 1
    assert kernel.count("Cast(out, result, RoundMode::CAST_RINT") == 1
    assert "madOffset += kMadK" in kernel and "p.disableGemv = true;" in kernel
    old_launch = kernel[kernel.index("void LaunchGrouped(") : kernel.index("void LaunchGroupedB1(")]
    assert "vq2a8_ascendc_v4_v2_grouped<<<blocks, nullptr, stream>>>" in old_launch


@pytest.mark.parametrize("chunks", (0, 2, 4))
def test_b1_schedule_numeric_invalid_cpu_validator_matrices(chunks):
    fixtures = {k: fixture(k) for k in probe.WIDTHS}
    assert len(probe.numeric_names()) == 48
    assert len(probe.graph_names()) == 24
    assert probe.run_numeric_checks(fixtures, stage, chunks) == probe.numeric_names()
    assert probe.run_invalid_checks(fixtures, stage, chunks) == probe.invalid_names()


@pytest.mark.parametrize("defect", ("projection", "status", "poison", "input", "owner_sentinel"))
def test_b1_schedule_fault_injection_is_not_hidden(defect):
    state = fixture()
    inputs, ids, owner = probe.inputs_fixture(state, 6, ids=[0, 1, 2, 0, 1, -1])
    method = state.bank.project_candidate

    def broken(*args):
        result = method(*args)
        if defect == "projection":
            result[0][0].view(torch.int16)[0] ^= 1
        elif defect == "status":
            result[1][-1] = 1
        elif defect == "poison":
            result[0][-1].view(torch.int16)[-1] = 0
        elif defect == "input":
            args[0][0].view(torch.uint8)[0] ^= 1
        elif defect == "owner_sentinel":
            owner[0] ^= 1
        return result

    state.bank.project_candidate = broken
    with pytest.raises(AssertionError, match="differ"):
        probe.checked_case(state, inputs, ids, owner, "injected")


@pytest.mark.parametrize("version", (None, True, False, "1", 0, 2))
def test_b1_schedule_requires_typed_independent_abi(version):
    with pytest.raises(RuntimeError):
        probe.require_abi(SimpleNamespace(b1_schedule_version=lambda: version))


def test_b1_schedule_missing_abi_and_missing_method_do_not_fallback():
    with pytest.raises(RuntimeError, match="missing"):
        probe.require_abi(SimpleNamespace())
    probe.require_abi(SimpleNamespace(b1_schedule_version=lambda: 1))
    with pytest.raises(AttributeError):
        probe.project_candidate(SimpleNamespace(project_vectorized=lambda *_: pytest.fail("no fallback")), (1, 2, 3, 4))


@pytest.mark.parametrize("chunks", (2, 4))
def test_b1_schedule_combination_requires_chunk_abi_and_exact_receipt(chunks):
    native = SimpleNamespace(b1_schedule_version=lambda: 1)
    with pytest.raises(RuntimeError, match="chunk-reuse ABI"):
        probe.require_abi(native, chunks)
    native.activation_reorder_chunk_reuse_version = lambda: True
    with pytest.raises(RuntimeError, match="chunk-reuse ABI"):
        probe.require_abi(native, chunks)
    native.activation_reorder_chunk_reuse_version = lambda: 1
    probe.require_abi(native, chunks)
    good = passing_result(True, chunks)
    probe.validate_child_evidence(good, queue_lifetime=True, reorder_chunks=chunks)
    with pytest.raises(ValueError, match="Incomplete"):
        probe.validate_child_evidence(good, queue_lifetime=True)


@pytest.mark.parametrize("queue", (False, True))
def test_b1_schedule_receipt_requires_complete_exact_coverage(queue):
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
        "dispatch_scope",
        "library",
        "device_execution_verified",
        "graph_verified",
        "model_integration_verified",
        "performance_verified",
    ):
        bad = copy.deepcopy(good)
        bad["events"][-1][key] = {} if key == "library" else None
        with pytest.raises(ValueError, match="Incomplete"):
            probe.validate_child_evidence(bad, queue_lifetime=queue)


def test_b1_schedule_queue_receipt_requires_final_verify_and_lifetime_evidence():
    good = passing_result(True)
    for key in probe.queue_evidence():
        bad = copy.deepcopy(good)
        del bad["events"][-1]["results"]["queue_lifetime"][key]
        with pytest.raises(ValueError, match="Incomplete"):
            probe.validate_child_evidence(bad, queue_lifetime=True)
    for rejected_stage in ("final_sync", "queue_lifetime_verify"):
        bad = copy.deepcopy(good)
        bad["events"] = [event for event in bad["events"] if event.get("stage") != rejected_stage]
        with pytest.raises(ValueError, match="Incomplete"):
            probe.validate_child_evidence(bad, queue_lifetime=True)


@pytest.mark.parametrize("field,value", (("status", "FAIL"), ("exit_code", 1), ("reaped", False)))
def test_b1_schedule_failed_supervisor_cannot_pass(field, value):
    result = passing_result()
    result[field] = value
    with pytest.raises(ValueError, match="Incomplete"):
        probe.validate_child_evidence(result, queue_lifetime=False)


def test_b1_schedule_plan_makes_no_hardware_claim(tmp_path, capsys):
    destination = tmp_path / "unused"
    assert probe.main(["--plan-only", "--queue-lifetime", "--report-dir", str(destination)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["dispatch_scope"] == probe.DISPATCH_SCOPE and not destination.exists()
    for field in ("device_execution_verified", "graph_verified", "model_integration_verified", "performance_verified"):
        assert value[field] is False
    args = probe.parse_args(value["command"][value["command"].index("--child") :])
    assert args.child and args.queue_lifetime and args.physical_npu == 1


@pytest.mark.parametrize("chunks", (0, 2, 4))
def test_b1_schedule_queue_releases_banks_and_payload_owners_before_fence(monkeypatch, chunks):
    references, fences, phases = [], [], []

    def create(k, *_):
        state = fixture(k)
        references.append(weakref.ref(state.bank))
        references.extend(weakref.ref(tensor) for items in state.payloads.values() for tensor in items)
        return state

    def synchronize():
        fences.append(phases[-1])
        if phases[-1] == "queue_owner_release_and_allocation_pressure":
            assert all(reference() is None for reference in references)

    @contextmanager
    def tracked_stage(name):
        phases.append(name)
        yield

    monkeypatch.setattr(probe.common, "make_fixture", create)
    monkeypatch.setattr(probe.common, "PRESSURE_BYTES", 64)
    monkeypatch.setattr(probe, "QUEUE_ITERATIONS", 25)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=synchronize), raising=False)
    assert probe.run_queue_checks("cpu", None, tracked_stage, chunks) == probe.queue_evidence(chunks)
    assert fences == ["queue_preupload_and_oracles", "queue_owner_release_and_allocation_pressure"]
    assert phases[-1] == "queue_lifetime_verify"
    source = inspect.getsource(probe.run_queue_checks)
    loop = source[source.index("for iteration in") : source.index("templates.clear()")]
    assert ".cpu(" not in loop and ".to(" not in loop and "synchronize(" not in loop


@pytest.mark.parametrize("failure", (None, "driver", "stale"))
@pytest.mark.parametrize("chunks", (0, 2, 4))
def test_b1_schedule_graph_replays_live_inputs_and_recovers(monkeypatch, failure, chunks):
    state = {"capture": None, "resets": 0}

    class Stream:
        def synchronize(self):
            pass

    class Graph:
        def __init__(self):
            self.jobs = []

        def replay(self):
            if failure == "driver":
                raise RuntimeError("driver replay failed")
            if failure == "stale":
                return
            for method, args, outputs in self.jobs:
                updated = method(*args)
                for target, source in zip(outputs, updated):
                    target.view(torch.uint8).copy_(source.view(torch.uint8))

        def reset(self):
            state["resets"] += 1

    @contextmanager
    def capture(graph, stream):
        state["capture"] = graph
        try:
            yield
        finally:
            state["capture"] = None

    def make(k, *_):
        result = fixture(k)
        method = result.bank.project_candidate

        def record(*args):
            outputs = method(*args)
            if state["capture"] is not None:
                state["capture"].jobs.append((method, args, outputs))
            return outputs

        result.bank.project_candidate = record
        return result

    monkeypatch.setattr(probe.common, "make_fixture", make)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            Stream=Stream, stream=lambda _: nullcontext(), NPUGraph=Graph, graph=capture, synchronize=lambda: None
        ),
        raising=False,
    )
    if failure is None:
        assert probe.run_graph_checks("cpu", None, stage, chunks) == probe.graph_names()
        assert state["resets"] == 4
    else:
        with pytest.raises(RuntimeError if failure == "driver" else AssertionError):
            probe.run_graph_checks("cpu", None, stage, chunks)
        assert state["resets"] == 0


@pytest.mark.parametrize("defect", ("projection", "status", "input"))
def test_b1_schedule_queue_faults_cannot_pass(monkeypatch, defect):
    phases = []

    @contextmanager
    def tracked_stage(name):
        phases.append(name)
        yield

    def make(k, *_):
        state = fixture(k)
        method = state.bank.project_candidate

        def broken(*inputs):
            result = method(*inputs)
            if defect == "projection":
                result[0][0].view(torch.int16)[0] ^= 1
            elif defect == "status":
                result[1][0] = 0
            elif defect == "input":
                inputs[0][0].view(torch.uint8)[0] ^= 1
            return result

        state.bank.project_candidate = broken
        return state

    monkeypatch.setattr(probe.common, "make_fixture", make)
    monkeypatch.setattr(probe.common, "PRESSURE_BYTES", 64)
    monkeypatch.setattr(probe, "QUEUE_ITERATIONS", 2)
    monkeypatch.setattr(probe, "QUEUE_TEMPLATES", 4)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)
    with pytest.raises(AssertionError, match="differ"):
        probe.run_queue_checks("cpu", None, tracked_stage)
    assert phases[-1] == "queue_lifetime_verify"
