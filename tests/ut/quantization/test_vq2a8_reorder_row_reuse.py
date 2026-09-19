# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU oracle/dispatch tests only, not AscendC numerical or speed acceptance."""

import copy
import inspect
import json
import weakref
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import validate_vq2a8_reorder_row_reuse as probe
from vllm_ascend.quantization.vq2a8_v4_v2 import AscendCV4V2VQ2TP1MoE, require_v4_v2_features


def stage(_name):
    return nullcontext()


class FakeBank:
    def __init__(self, fixture):
        self.fixture = SimpleNamespace(**vars(fixture))

    def prepare_vectorized(self, q, scale, bias, ids):
        result = torch.empty_like(q)
        valid = torch.tensor([int(0 <= slot < probe.EXPERTS) for slot in ids.tolist()], dtype=torch.int32)
        for row, slot in enumerate(ids.tolist()):
            if valid[row]:
                result[row].reshape(-1).view(torch.uint8).copy_(
                    q[row].reshape(-1).view(torch.uint8)[self.fixture.orders[slot]]
                )
        return result, valid

    def prepare_row_reuse(self, q, scale, bias, ids):
        reordered, valid = self.prepare_vectorized(q, scale, bias, ids)
        shape = (*q.shape[:-1], probe.common.OUTPUT_WIDTH)
        # Valid prepare output is unspecified: intentionally poison it too.
        output = torch.full(shape, float("nan"), dtype=torch.bfloat16)
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

    def project_row_reuse(self, *inputs):
        return self.project_vectorized(*inputs)


def fixture(k=2048):
    result = SimpleNamespace(
        k=k,
        device="cpu",
        orders=[torch.arange(k).roll(i + 1) for i in range(3)],
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
            "library": {"path": "/tmp/" + probe.LIBRARY_NAME, "sha256": "b" * 64},
            "device_execution_verified": True,
            "graph_verified": True,
            "model_integration_verified": False,
            "performance_verified": False,
        }
    )
    return {"status": "PASS", "exit_code": 0, "reaped": True, "events": events}


@pytest.mark.parametrize("rank3", [False, True])
@pytest.mark.parametrize("k", probe.WIDTHS)
def test_reorder_row_reuse_preserves_all_raw_codes_and_offset_sentinels(k, rank3):
    state = fixture(k)
    values, _, owner = probe.inputs_fixture(state, 6, "raw_bytes", rank3=rank3)
    assert set(values[0].view(torch.uint8).flatten().tolist()) == set(range(256))
    assert values[0].data_ptr() == owner.data_ptr() + 32
    assert torch.equal(owner[:32], torch.full((32,), 0x5A, dtype=torch.uint8))
    assert torch.equal(owner[-32:], owner[:32])
    assert values[0].dtype == torch.float8_e4m3fn


def test_reorder_row_reuse_numeric_invalid_cpu_oracle_matrices():
    fixtures = {k: fixture(k) for k in probe.WIDTHS}
    assert len(probe.numeric_names()) == 96
    assert probe.run_numeric_checks(fixtures, stage) == probe.numeric_names()
    assert probe.run_invalid_checks(fixtures, stage) == probe.invalid_names()
    assert len(probe.graph_names()) == 24


@pytest.mark.parametrize(
    "defect",
    ["bytes", "nan_byte", "signed_zero", "status", "descriptor", "poison", "projection", "input", "owner_sentinel"],
)
def test_reorder_row_reuse_fault_injection_fails(defect):
    state = fixture()
    pattern = "raw_bytes" if defect in ("nan_byte", "signed_zero") else "finite"
    inputs, ids, owner = probe.inputs_fixture(state, 6, pattern, ids=[0, 1, 2, 0, 1, -1])
    prepare, project = state.bank.prepare_row_reuse, state.bank.project_row_reuse

    def broken(*values):
        result = prepare(*values)
        if defect == "bytes":
            result[0][0].view(torch.uint8)[0] ^= 1
        elif defect in ("nan_byte", "signed_zero"):
            row = result[0][0].view(torch.uint8)
            index = (row == (0x7F if defect == "nan_byte" else 0x80)).nonzero()[0, 0]
            row[index] ^= 0x80
        elif defect == "status":
            result[1][-1] = 1
        elif defect == "descriptor":
            result[2][0, 6] = 2
        elif defect == "poison":
            result[3][-1].view(torch.int16)[-1] = 0
        elif defect == "input":
            values[0][0].view(torch.uint8)[0] ^= 1
        elif defect == "owner_sentinel":
            owner[0] ^= 1
        return result

    def broken_project(*values):
        result = project(*values)
        if defect == "projection":
            result[0][0].view(torch.int16)[0] ^= 1
        return result

    state.bank.prepare_row_reuse, state.bank.project_row_reuse = broken, broken_project
    with pytest.raises(AssertionError, match="differ"):
        probe.checked_case(state, inputs, ids, owner, "fault", projection=pattern == "finite")


@pytest.mark.parametrize(
    "rank,rows,expected", [(2, 1, "row_reuse"), (3, 1, "row_reuse"), (3, 2, "vectorized"), (3, 32, "vectorized")]
)
def test_reorder_row_reuse_m1_only_dispatch(rank, rows, expected):
    calls = []

    def method(name):
        def project(*args):
            calls.append((name, args))
            return "output", "valid"

        return project

    bank = SimpleNamespace(
        project_row_reuse=method("row_reuse"), project_vectorized=method("vectorized"), project=method("scalar")
    )
    model = SimpleNamespace(v4_activation_reorder="row_reuse")
    q = torch.empty((6, 2048) if rank == 2 else (6, rows, 2048), dtype=torch.float8_e4m3fn)
    scale, bias, ids = object(), object(), object()
    result = AscendCV4V2VQ2TP1MoE.project_v4_prepared(model, bank, q, scale, bias, ids)
    assert result == ("output", "valid")
    assert len(calls) == 1 and calls[0][0] == expected
    assert all(got is want for got, want in zip(calls[0][1], (q, scale, bias, ids)))


def test_reorder_row_reuse_missing_native_method_does_not_fallback():
    bank = SimpleNamespace(project_vectorized=lambda *_: pytest.fail("no M1 fallback"))
    with pytest.raises(AttributeError):
        AscendCV4V2VQ2TP1MoE.project_v4_prepared(
            SimpleNamespace(v4_activation_reorder="row_reuse"), bank, torch.empty(1, 2048), None, None, None
        )


@pytest.mark.parametrize("version", [None, True, False, "1", 0, 2])
def test_reorder_row_reuse_abi_strict(version):
    native = SimpleNamespace(activation_reorder_version=lambda: 1, activation_reorder_row_reuse_version=lambda: version)
    with pytest.raises(RuntimeError):
        probe.require_abi(native)
    with pytest.raises(RuntimeError):
        require_v4_v2_features("row_reuse", native_ops=native)


def test_reorder_row_reuse_independent_abi_and_default_gates():
    calls = []
    native = SimpleNamespace(
        activation_reorder_version=lambda: calls.append("vectorized") or 1,
        activation_reorder_row_reuse_version=lambda: calls.append("row_reuse") or 1,
    )
    require_v4_v2_features(native_ops=native)
    assert calls == []
    require_v4_v2_features("vectorized", native_ops=native)
    assert calls == ["vectorized"]
    calls.clear()
    require_v4_v2_features("row_reuse", native_ops=native)
    assert calls == ["vectorized", "row_reuse"]
    probe.require_abi(native)
    with pytest.raises(RuntimeError, match="missing"):
        probe.require_abi(SimpleNamespace())
    with pytest.raises(ValueError, match="Fused tail"):
        require_v4_v2_features("row_reuse", "sign_fused_direct", activation_tail="fused_reorder", native_ops=native)


@pytest.mark.parametrize("queue", [False, True])
def test_reorder_row_reuse_receipt_requires_exact_scope_and_coverage(queue):
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


def test_reorder_row_reuse_queue_receipt_requires_final_verify_and_no_shortcuts():
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


@pytest.mark.parametrize("field,value", [("status", "FAIL"), ("exit_code", 1), ("reaped", False)])
def test_reorder_row_reuse_failed_supervisor_cannot_pass(field, value):
    result = passing_result()
    result[field] = value
    with pytest.raises(ValueError, match="Incomplete"):
        probe.validate_child_evidence(result, queue_lifetime=False)


def test_reorder_row_reuse_plan_is_import_only_no_execution_claim(tmp_path, capsys):
    destination = tmp_path / "unused"
    assert probe.main(["--plan-only", "--queue-lifetime", "--report-dir", str(destination)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["dispatch_scope"] == probe.DISPATCH_SCOPE and not destination.exists()
    for field in ("device_execution_verified", "graph_verified", "model_integration_verified", "performance_verified"):
        assert value[field] is False
    args = probe.parse_args(value["command"][value["command"].index("--child") :])
    assert args.child and args.queue_lifetime and args.physical_npu == 1


def test_reorder_row_reuse_queue_releases_banks_before_fence(monkeypatch):
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
    result = probe.run_queue_checks("cpu", None, tracked_stage)
    assert result == probe.queue_evidence()
    assert fences == ["queue_preupload_and_oracles", "queue_owner_release_and_allocation_pressure"]
    assert phases[-1] == "queue_lifetime_verify"
    source = inspect.getsource(probe.run_queue_checks)
    loop = source[source.index("for iteration in") : source.index("templates.clear()")]
    assert ".cpu(" not in loop and ".to(" not in loop and "synchronize(" not in loop


def test_reorder_row_reuse_native_source_contract():
    root = Path(__file__).resolve().parents[3]
    kernel = (root / "csrc/vq2a8_ascendc_v4_v2/resident_prepare.cpp").read_text()
    binding = (root / "csrc/vq2a8_ascendc_v4_v2/torch_binding.cpp").read_text()
    section = kernel[kernel.index("void ProcessRowReuse()") : kernel.index("// Candidate D starts")]
    assert section.count("DataCopy(inputUb_.Get<uint8_t>()") == 1
    assert section.index("ValidResidentSlot") < section.index("static_cast<uint32_t>(expert)")
    assert "for (uint32_t column = 0; column < k_; column += kSelectColumns)" in section
    assert "GatherVectorized(order, rowBase, column)" in section
    assert "if constexpr (!RowReuse) DataCopy(input, x_[rowBase], k_);" in kernel
    assert 'if constexpr (RowReuse) TORCH_CHECK(m == 1, "Row-reuse reorder requires M=1")' in binding
    assert "[state, x, scale, bias, ids, reordered, descriptors, output, valid" in binding
    assert "RecordInputs({x, scale, bias, ids}" in binding


@pytest.mark.parametrize("failure", [None, "driver", "stale"])
def test_reorder_row_reuse_live_graph_recovery_and_failures(monkeypatch, failure):
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
                    # Byte copy, including the FP8 NaN sign bits.
                    target.view(torch.uint8).copy_(source.view(torch.uint8))
                if len(outputs) == 4:
                    outputs[2].copy_(
                        torch.tensor(
                            probe.common.descriptor_reference(
                                method.__self__.fixture, args, args[-1].tolist(), outputs
                            ),
                            dtype=torch.int64,
                        )
                    )

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
        for name in ("prepare_row_reuse", "project_row_reuse"):
            method = getattr(result.bank, name)

            def record(*args, method=method):
                outputs = method(*args)
                if state["capture"] is not None:
                    state["capture"].jobs.append((method, args, outputs))
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
        assert probe.run_graph_checks("cpu", None, stage) == probe.graph_names()
        assert state["resets"] == 4
    else:
        with pytest.raises(RuntimeError if failure == "driver" else AssertionError):
            probe.run_graph_checks("cpu", None, stage)
        assert state["resets"] == 0


@pytest.mark.parametrize("defect", ["bytes", "status", "descriptor", "projection", "input"])
def test_reorder_row_reuse_queue_faults_cannot_pass(monkeypatch, defect):
    phase = []

    @contextmanager
    def tracked_stage(name):
        phase.append(name)
        yield

    def make(k, *_):
        state = fixture(k)
        prepare, project = state.bank.prepare_row_reuse, state.bank.project_row_reuse

        def broken(*inputs):
            outputs = prepare(*inputs)
            if defect == "bytes":
                outputs[0][0].view(torch.uint8)[0] ^= 1
            elif defect == "status":
                outputs[1][0] = 0
            elif defect == "descriptor":
                outputs[2][0, 8] = 0
            elif defect == "input":
                inputs[0][0].view(torch.uint8)[0] ^= 1
            return outputs

        def broken_project(*inputs):
            outputs = project(*inputs)
            if defect == "projection":
                outputs[0][0].view(torch.int16)[0] ^= 1
            return outputs

        state.bank.prepare_row_reuse, state.bank.project_row_reuse = broken, broken_project
        return state

    monkeypatch.setattr(probe.common, "make_fixture", make)
    monkeypatch.setattr(probe.common, "PRESSURE_BYTES", 64)
    monkeypatch.setattr(probe, "QUEUE_ITERATIONS", 2)
    monkeypatch.setattr(probe, "QUEUE_TEMPLATES", 4)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)
    with pytest.raises(AssertionError, match="differ"):
        probe.run_queue_checks("cpu", None, tracked_stage)
    assert phase[-1] == "queue_lifetime_verify"
