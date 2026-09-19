# SPDX-License-Identifier: Apache-2.0
"""Candidate J CPU contracts/probe fault injection; not native numerical acceptance."""

import copy
import inspect
import json
import weakref
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import validate_vq2a8_reorder_chunk_reuse as probe


def stage(_name):
    return nullcontext()


class FakeBank:
    def __init__(self, fixture):
        self.fixture = SimpleNamespace(**vars(fixture))
        self.calls = []

    def prepare_vectorized(self, q, scale, bias, ids):
        result = torch.empty_like(q)
        valid = torch.tensor([int(0 <= slot < probe.EXPERTS) for slot in ids.tolist()], dtype=torch.int32)
        for row, slot in enumerate(ids.tolist()):
            if valid[row]:
                result[row].reshape(-1).view(torch.uint8).copy_(
                    q[row].reshape(-1).view(torch.uint8)[self.fixture.orders[slot]]
                )
        return result, valid

    def prepare_chunk_reuse(self, q, scale, bias, ids, chunks):
        assert chunks in (2, 4)
        self.calls.append(("prepare", chunks))
        reordered, valid = self.prepare_vectorized(q, scale, bias, ids)
        output = torch.empty((*q.shape[:-1], probe.common.OUTPUT_WIDTH), dtype=torch.bfloat16)
        output.view(torch.int16).fill_(0x7FC0)
        records = torch.empty((len(ids), 9), dtype=torch.int64)
        prepared = (reordered, valid, records, output)
        records.copy_(
            torch.tensor(
                probe.common.descriptor_reference(self.fixture, (q, scale, bias, ids), ids.tolist(), prepared),
                dtype=torch.int64,
            )
        )
        return prepared

    def project_vectorized(self, q, scale, bias, ids):
        _, valid = self.prepare_vectorized(q, scale, bias, ids)
        values = ((q.float() * (torch.arange(q.shape[-1]) % 3 - 1)).sum(-1) * scale + bias).bfloat16()
        output = values[..., None].repeat(*([1] * values.ndim), probe.common.OUTPUT_WIDTH)
        for row, value in enumerate(valid):
            if not value:
                output[row].view(torch.int16).fill_(0x7FC0)
        return output, valid

    def project_chunk_reuse(self, q, scale, bias, ids, chunks):
        assert chunks in (2, 4)
        self.calls.append(("direct", chunks))
        return self.project_vectorized(q, scale, bias, ids)

    def project_candidate(self, q, scale, bias, ids, chunks, schedule):
        assert chunks in (2, 4) and schedule == 0
        self.calls.append(("candidate", chunks, schedule))
        return self.project_vectorized(q, scale, bias, ids)


def fixture(k=2048):
    result = SimpleNamespace(
        k=k,
        device="cpu",
        orders=[torch.arange(k).roll(i + 1) for i in range(3)],
        payloads={name: [torch.zeros(32, dtype=torch.uint8) for _ in range(3)] for name in ("packed_zn", "pair_lut")},
    )
    result.bank = FakeBank(result)
    return result


def passing_result(chunks, queue=False):
    values = {
        "numeric": probe.numeric_names(),
        "invalid": probe.invalid_names(),
        "native_contract": probe.CONTRACT_CASES,
        "graph": probe.graph_names(),
    }
    if queue:
        values["queue_lifetime"] = {**probe.queue_evidence(), "task_queue_enable": "1"}
    events = [{"event": "PASS", "stage": "final_sync"}]
    if queue:
        events.append({"event": "PASS", "stage": "queue_lifetime_verify"})
    events.append(
        {
            "event": "CASE_PASS",
            "case": probe.CASE,
            "native_abi": 1,
            "chunks": chunks,
            "results": values,
            "dispatch_scope": probe.DISPATCH_SCOPE,
            "library": {"path": "/tmp/" + probe.LIBRARY_NAME, "sha256": "b" * 64},
            "device_execution_verified": True,
            "graph_verified": True,
            "model_integration_verified": False,
            "performance_verified": False,
        }
    )
    return {"status": "PASS", "exit_code": 0, "reaped": True, "events": events}


@pytest.mark.parametrize("chunks", (2, 4))
def test_chunk_reuse_numeric_and_invalid_matrix(chunks):
    fixtures = {k: fixture(k) for k in probe.WIDTHS}
    assert len(probe.numeric_names()) == 96
    assert probe.run_numeric_checks(fixtures, stage, chunks) == probe.numeric_names()
    assert probe.run_invalid_checks(fixtures, stage, chunks) == probe.invalid_names()
    for state in fixtures.values():
        assert {call[0] for call in state.bank.calls} == {"prepare", "candidate", "direct"}
        assert {call[1] for call in state.bank.calls} == {chunks}


@pytest.mark.parametrize("chunks", (2, 4))
@pytest.mark.parametrize("defect", ("byte", "nan", "zero", "status", "descriptor", "poison", "projection", "input"))
def test_chunk_reuse_faults_rejected(chunks, defect):
    state = fixture()
    pattern = "raw_bytes" if defect in ("nan", "zero") else "finite"
    inputs, ids, owner = probe.inputs_fixture(state, 6, pattern, ids=[0, 1, 2, 0, 1, -1])
    prepare, project = state.bank.prepare_chunk_reuse, state.bank.project_candidate

    def broken_prepare(*values):
        outputs = prepare(*values)
        if defect == "byte":
            outputs[0][0].view(torch.uint8)[0] ^= 1
        elif defect in ("nan", "zero"):
            row = outputs[0][0].view(torch.uint8)
            index = (row == (0x7F if defect == "nan" else 0x80)).nonzero()[0, 0]
            row[index] ^= 0x80
        elif defect == "status":
            outputs[1][-1] = 1
        elif defect == "descriptor":
            outputs[2][0, 8] = 1
        elif defect == "poison":
            outputs[3][-1].view(torch.int16)[-1] = 0
        elif defect == "input":
            owner[0] ^= 1
        return outputs

    def broken_project(*values):
        outputs = project(*values)
        if defect == "projection":
            outputs[0][0].view(torch.int16)[0] ^= 1
        return outputs

    state.bank.prepare_chunk_reuse, state.bank.project_candidate = broken_prepare, broken_project
    with pytest.raises(AssertionError, match="differ"):
        probe.checked_case(state, inputs, ids, owner, "fault", chunks=chunks, projection=pattern == "finite")


@pytest.mark.parametrize("version", (None, True, False, "1", 0, 2))
def test_chunk_reuse_abi_strict(version):
    with pytest.raises(RuntimeError):
        probe.require_abi(SimpleNamespace(activation_reorder_chunk_reuse_version=lambda: version))


@pytest.mark.parametrize("chunks", (2, 4))
@pytest.mark.parametrize("queue", (False, True))
def test_chunk_reuse_receipt_exact_identity_and_coverage(chunks, queue):
    result = passing_result(chunks, queue)
    probe.validate_child_evidence(result, queue_lifetime=queue, chunks=chunks)
    for key in result["events"][-1]:
        bad = copy.deepcopy(result)
        del bad["events"][-1][key]
        with pytest.raises(ValueError, match="Incomplete"):
            probe.validate_child_evidence(bad, queue_lifetime=queue, chunks=chunks)
    for other in (1, 3, 6 - chunks, True):
        bad = copy.deepcopy(result)
        bad["events"][-1]["chunks"] = other
        with pytest.raises(ValueError, match="Incomplete"):
            probe.validate_child_evidence(bad, queue_lifetime=queue, chunks=chunks)
    for key in result["events"][-1]["results"]:
        bad = copy.deepcopy(result)
        del bad["events"][-1]["results"][key]
        with pytest.raises(ValueError, match="Incomplete"):
            probe.validate_child_evidence(bad, queue_lifetime=queue, chunks=chunks)
    for field, value in (("status", "FAIL"), ("exit_code", 1), ("reaped", False)):
        bad = copy.deepcopy(result)
        bad[field] = value
        with pytest.raises(ValueError, match="Incomplete"):
            probe.validate_child_evidence(bad, queue_lifetime=queue, chunks=chunks)


@pytest.mark.parametrize("chunks", (2, 4))
def test_chunk_reuse_plan_only_no_hardware_claim(chunks, tmp_path, capsys):
    target = tmp_path / "unused"
    assert probe.main(["--chunks", str(chunks), "--plan-only", "--queue-lifetime", "--report-dir", str(target)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["chunks"] == chunks and not target.exists()
    args = probe.parse_args(result["command"][result["command"].index("--child") :])
    assert args.chunks == chunks and args.child and args.queue_lifetime
    for field in ("device_execution_verified", "graph_verified", "model_integration_verified", "performance_verified"):
        assert result[field] is False


@pytest.mark.parametrize("chunks", (2, 4))
@pytest.mark.parametrize("failure", (None, "driver", "stale"))
def test_chunk_reuse_graph_recovery_and_failure(chunks, failure, monkeypatch):
    capture_state = {"graph": None, "resets": 0}

    class Stream:
        def synchronize(self):
            pass

    class Graph:
        def __init__(self):
            self.jobs = []

        def replay(self):
            if failure == "driver":
                raise RuntimeError("driver failed")
            if failure == "stale":
                return
            for method, args, outputs in self.jobs:
                updated = method(*args)
                for dst, src in zip(outputs, updated):
                    dst.view(torch.uint8).copy_(src.view(torch.uint8))
                if len(outputs) == 4:
                    outputs[2].copy_(
                        torch.tensor(
                            probe.common.descriptor_reference(
                                method.__self__.fixture, args[:4], args[3].tolist(), outputs
                            ),
                            dtype=torch.int64,
                        )
                    )

        def reset(self):
            capture_state["resets"] += 1

    @contextmanager
    def capture(graph, stream):
        capture_state["graph"] = graph
        try:
            yield
        finally:
            capture_state["graph"] = None

    def make(k, *_):
        result = fixture(k)
        for name in ("prepare_chunk_reuse", "project_candidate"):
            method = getattr(result.bank, name)

            def record(*args, method=method):
                outputs = method(*args)
                if capture_state["graph"] is not None:
                    capture_state["graph"].jobs.append((method, args, outputs))
                return outputs

            setattr(result.bank, name, record)
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
        assert capture_state["resets"] == 4
    else:
        with pytest.raises(RuntimeError if failure == "driver" else AssertionError):
            probe.run_graph_checks("cpu", None, stage, chunks)


@pytest.mark.parametrize("chunks", (2, 4))
def test_chunk_reuse_queue_releases_owners_before_fence(chunks, monkeypatch):
    references, phases, fences = [], [], []

    def make(k, *_):
        result = fixture(k)
        references.append(weakref.ref(result.bank))
        references.extend(weakref.ref(t) for values in result.payloads.values() for t in values)
        return result

    @contextmanager
    def tracked_stage(name):
        phases.append(name)
        yield

    def synchronize():
        fences.append(phases[-1])
        if phases[-1] == "queue_owner_release_and_allocation_pressure":
            assert all(reference() is None for reference in references)

    monkeypatch.setattr(probe.common, "make_fixture", make)
    monkeypatch.setattr(probe.common, "PRESSURE_BYTES", 64)
    monkeypatch.setattr(probe, "QUEUE_ITERATIONS", 25)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=synchronize), raising=False)
    assert probe.run_queue_checks("cpu", None, tracked_stage, chunks) == probe.queue_evidence()
    assert phases[-1] == "queue_lifetime_verify"
    assert fences == ["queue_preupload_and_oracles", "queue_owner_release_and_allocation_pressure"]
    source = inspect.getsource(probe.run_queue_checks)
    loop = source[source.index("for iteration in") : source.index("templates.clear()")]
    assert ".cpu(" not in loop and ".to(" not in loop and "synchronize(" not in loop


@pytest.mark.parametrize("k", probe.WIDTHS)
@pytest.mark.parametrize("chunks", (2, 4))
@pytest.mark.parametrize("routes", range(1, 7))
def test_chunk_reuse_work_coverage_no_duplicate_writes(k, chunks, routes):
    groups = 4096 // (256 * chunks)
    writes, descriptor_routes = [], []
    for work in range(routes * groups):
        route, group = divmod(work, groups)
        columns = range(group * chunks * 256, (group + 1) * chunks * 256, 256)
        writes.extend((route, col) for col in columns if col < k)
        if group == 0:
            descriptor_routes.append(route)
    assert len(writes) == len(set(writes)) == routes * k // 256
    assert descriptor_routes == list(range(routes))


def test_chunk_reuse_native_source_has_distinct_abi_and_unchanged_baseline():
    root = Path(__file__).resolve().parents[3]
    kernel = (root / "csrc/vq2a8_ascendc_v4_v2/resident_prepare.cpp").read_text()
    binding = (root / "csrc/vq2a8_ascendc_v4_v2/torch_binding.cpp").read_text()
    section = kernel[kernel.index("void ProcessChunkReuse()") : kernel.index("// Candidate F")]
    assert section.count("DataCopy(inputUb_.Get<uint8_t>()") == 1
    assert section.index("ValidResidentSlot") < section.index("static_cast<uint32_t>(expert)")
    assert "chunk < ChunkReuse" in section and "route = work / chunkGroups" in section
    assert "if constexpr (ChunkReuse == 0)" in kernel
    assert "if constexpr (!RowReuse) DataCopy(input, x_[rowBase], k_);" in kernel
    for chunks in (2, 4):
        assert f"ResidentPrepareKernel<true, false, false, {chunks}>" in kernel
    assert "activation_reorder_chunk_reuse_version() -> int" in binding
    assert "TORCH_CHECK(chunks == 2 || chunks == 4" in binding
    assert "TORCH_CHECK(schedule == 0 || schedule == 1" in binding
    assert "RecordInputs({x, scale, bias, ids}" in binding
    assert "[state, x, scale, bias, ids, reordered, descriptors, output, valid" in binding
