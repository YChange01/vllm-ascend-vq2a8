# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts only; not native compilation, bitwise NPU or performance proof."""

import copy
import inspect
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import validate_vq2a8_select_sign as probe
from vllm_ascend.quantization.vq2a8_select_sign import FusedSelectSign


@pytest.mark.parametrize("version", [True, None, 0, 2, "1"])
def test_strict_abi(version):
    with pytest.raises(RuntimeError, match="ABI"):
        FusedSelectSign(SimpleNamespace(select_sign_version=lambda: version))


def test_missing_abi_and_member_fail_closed():
    with pytest.raises(RuntimeError, match="no implicit fallback"):
        FusedSelectSign(SimpleNamespace())
    checker = FusedSelectSign(SimpleNamespace(select_sign_version=lambda: 1))
    with pytest.raises(RuntimeError, match="no implicit fallback"):
        checker(SimpleNamespace(), torch.zeros(1, 2048), torch.zeros(1, dtype=torch.int64))


@pytest.mark.parametrize("width", probe.WIDTHS)
@pytest.mark.parametrize("groups", range(1, 7))
@pytest.mark.parametrize("dtype", probe.DTYPES)
@pytest.mark.parametrize("layout", probe.LAYOUTS)
def test_views_forward_without_materialization(width, groups, dtype, layout, monkeypatch):
    hidden, owner = probe.make_hidden("cpu", width, groups, dtype, layout)
    ids = torch.tensor([73, *range(groups), 91], dtype=torch.int64)[1:-1]
    result = object()
    calls = []

    def operation(value, slots):
        calls.append((value is hidden, slots is ids))
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("Host tensor access or materialization in dispatch")

    checker = FusedSelectSign(SimpleNamespace(select_sign_version=lambda: 1))
    for name in ("item", "cpu", "numpy", "tolist", "contiguous", "clone"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    assert checker(SimpleNamespace(select_sign=operation), hidden, ids) is result
    assert calls == [(True, True)]


@pytest.mark.parametrize("bad", range(13))
def test_invalid_metadata_rejected_before_native(bad):
    hidden = torch.zeros(2, 2048)
    ids = torch.zeros(2, dtype=torch.int64)
    cases = [
        (None, ids),
        (hidden, None),
        (hidden.half(), ids),
        (hidden.flatten(), ids),
        (hidden[:, :1024].contiguous(), ids),
        (hidden[:0], ids[:0]),
        (torch.zeros(7, 2048), torch.zeros(7, dtype=torch.int64)),
        (torch.zeros(2, 4096)[:, ::2], ids),
        (hidden.as_strided((2, 2048), (1, 1)), ids),
        (torch.zeros(2, 2049)[:, :2048], ids),
        (torch.zeros(4097)[1:].reshape(2, 2048), ids),
        (hidden, ids.int()),
        (hidden, ids[:1]),
    ]
    checker = FusedSelectSign(SimpleNamespace(select_sign_version=lambda: 1))
    with pytest.raises(ValueError, match="Select/sign"):
        checker(SimpleNamespace(select_sign=lambda *_: pytest.fail("native called")), *cases[bad])


@pytest.mark.parametrize("width", probe.WIDTHS)
@pytest.mark.parametrize("layout", probe.LAYOUTS)
def test_queue_view_reconstruction_preserves_geometry(width, layout):
    hidden, owner = probe.make_hidden("cpu", width, 6, "bfloat16", layout)
    copied_owner = owner.clone()
    view = probe.hidden_view(copied_owner, 6, width, layout)
    assert view.shape == hidden.shape
    assert view.stride() == hidden.stride()
    assert view.storage_offset() == hidden.storage_offset()
    assert torch.equal(hidden, view)


def valid_receipt(queue):
    results = {
        "numeric": probe.numeric_names(),
        "native_contract": probe.NATIVE_CONTRACT_CASES,
        "graph": probe.graph_names(),
    }
    if queue:
        results["queue_lifetime"] = probe.queue_evidence()
    return {
        "status": "PASS",
        "exit_code": 0,
        "reaped": True,
        "events": [
            *([{"event": "PASS", "stage": "queue_lifetime_verify"}] if queue else []),
            {"event": "PASS", "stage": "final_sync"},
            {
                "event": "CASE_PASS",
                "case": probe.CASE,
                "native_abi": 1,
                "results": results,
                "library": {"path": "/tmp/" + probe.LIBRARY_NAME, "sha256": "a" * 64},
                "device_execution_verified": True,
                "graph_verified": True,
                "model_integration_verified": False,
                "performance_verified": False,
            },
        ],
    }


@pytest.mark.parametrize("queue", [False, True])
def test_receipt_requires_complete_exact_cases(queue):
    result = valid_receipt(queue)
    probe.validate_child_evidence(result, queue)
    for key in ("numeric", "graph"):
        bad = copy.deepcopy(result)
        bad["events"][-1]["results"][key].pop()
        with pytest.raises(ValueError, match="Incomplete"):
            probe.validate_child_evidence(bad, queue)


@pytest.mark.parametrize("field,value", [("exit_code", 1), ("reaped", False), ("status", "FAIL")])
def test_process_failure_cannot_be_upgraded(field, value):
    result = valid_receipt(True)
    result[field] = value
    with pytest.raises(ValueError, match="Incomplete"):
        probe.validate_child_evidence(result, True)


@pytest.mark.parametrize("missing", ["verify_event", "ordinary_reference"])
def test_queue_receipt_requires_verified_same_device_ordinary_reference(missing):
    result = valid_receipt(True)
    if missing == "verify_event":
        result["events"] = [event for event in result["events"] if event.get("stage") != "queue_lifetime_verify"]
    else:
        del result["events"][-1]["results"]["queue_lifetime"]["ordinary_reference"]
    with pytest.raises(ValueError, match="Incomplete"):
        probe.validate_child_evidence(result, True)


def queue_cpu_harness(monkeypatch, *, corruption=None):
    """Run real queue control flow; only native ops/device add are CPU oracles.

    The synthetic device canonicalizes NaNs to a different payload than the
    CPU add. This reproduces the invalid-slot oracle bug without pretending
    CPU execution proves NPU queue lifetime, allocation or stream behavior.
    """
    observations = {"adds": [], "stages": [], "fused_calls": 0, "inside_queue": False}

    class DeviceAddTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, value, source, add_corruption=None):
            tensor = torch.Tensor._make_subclass(cls, value, require_grad=False)
            tensor.source = source
            tensor.add_corruption = add_corruption
            return tensor

        def cpu(self, *args, **kwargs):
            assert not observations["inside_queue"], "Unexpected CPU copy inside queue loop"
            return self.as_subclass(torch.Tensor)

        def __add__(self, value):
            plain = self.as_subclass(torch.Tensor)
            cpu_expected = plain + value
            result = cpu_expected.clone()
            result.view(torch.int32)[torch.isnan(result)] = 0x7FFFFFFF
            if self.add_corruption == "ordinary_finite":
                result.view(torch.int32).reshape(-1)[0] ^= 1
            elif self.add_corruption == "ordinary_nan":
                # Remains NaN: an equal_nan comparison would hide this error.
                result.view(torch.int32)[-1, 0] ^= 1
            observations["adds"].append((self.source, cpu_expected, result.clone()))
            return DeviceAddTensor(result, self.source)

    def values(hidden, ids, *, source, iteration=None):
        selected = ((ids >= 0) & (ids < 3)).int()
        scale = torch.arange(hidden.numel(), dtype=torch.float32).reshape(hidden.shape) / 128
        scale = torch.where(selected[:, None] != 0, scale, float("nan"))
        bias = scale.clone()
        signed = hidden.float().clone()
        statuses = selected, selected.clone()
        fields = [signed, scale, bias, *statuses]
        if source == "candidate" and iteration == 0 and corruption in tuple(f"field_{i}" for i in range(5)):
            fields[int(corruption[-1])].view(torch.uint8).reshape(-1)[0] ^= 1
        if source == "candidate" and iteration == 1 and corruption == "field_nan_payload":
            fields[1].view(torch.int32)[-1, 0] ^= 1
        add_corruption = None
        if source == "candidate" and (
            (iteration == 0 and corruption == "ordinary_finite") or (iteration == 1 and corruption == "ordinary_nan")
        ):
            add_corruption = corruption
        fields[1] = DeviceAddTensor(scale, source, add_corruption)
        return tuple(fields)

    def unfused_reference(bank, hidden, ids, native):
        return values(hidden, ids, source="reference")

    def fused(bank, hidden, ids):
        iteration = observations["fused_calls"]
        observations["fused_calls"] += 1
        return values(hidden, ids, source="candidate", iteration=iteration)

    def synchronize():
        assert not observations["inside_queue"], "Unexpected synchronize inside queue loop"

    @contextmanager
    def stage(name):
        observations["stages"].append((name, "BEGIN"))
        observations["inside_queue"] = name == "queue_lifetime"
        try:
            yield
        except BaseException:
            observations["stages"].append((name, "FAIL"))
            raise
        else:
            observations["stages"].append((name, "PASS"))
        finally:
            observations["inside_queue"] = False

    monkeypatch.setattr(probe, "WIDTHS", (8,))
    monkeypatch.setattr(probe, "DTYPES", ("float32",))
    monkeypatch.setattr(probe, "LAYOUTS", ("contiguous",))
    monkeypatch.setattr(probe, "QUEUE_ITERATIONS", 4)
    monkeypatch.setattr(probe, "PRESSURE_BYTES", 64)
    monkeypatch.setattr(probe, "make_bank", lambda *args: (object(), ()))
    monkeypatch.setattr(probe, "reference", unfused_reference)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=synchronize), raising=False)
    return lambda: probe.run_queue("cpu", None, None, fused, stage), observations


def test_queue_same_device_oracle_accepts_different_cpu_nan_payload(monkeypatch):
    run, observations = queue_cpu_harness(monkeypatch)
    evidence = run()
    assert evidence["ordinary_reference"] == "same_device_unfused_select_then_add"
    reference_adds = [record for record in observations["adds"] if record[0] == "reference"]
    candidate_adds = [record for record in observations["adds"] if record[0] == "candidate"]
    assert len(reference_adds) == 3 and len(candidate_adds) == observations["fused_calls"] == 4
    _, old_cpu_reference, candidate = candidate_adds[1]
    assert old_cpu_reference.view(torch.int32)[-1, 0] == 0x7FC00000
    assert candidate.view(torch.int32)[-1, 0] == 0x7FFFFFFF
    # The original run_queue assertion fails for exactly the second/min-ID
    # template, even though its five native outputs and device add are correct.
    with pytest.raises(AssertionError, match="ordinary operation changed"):
        probe.assert_ordinary_bits(candidate, old_cpu_reference, "queue_1")
    probe.assert_ordinary_bits(candidate, reference_adds[1][2], "queue_1")
    assert observations["stages"] == [
        ("queue_lifetime", "BEGIN"),
        ("queue_lifetime", "PASS"),
        ("queue_lifetime_verify", "BEGIN"),
        ("queue_lifetime_verify", "PASS"),
    ]


@pytest.mark.parametrize("corruption,iteration", [("ordinary_finite", 0), ("ordinary_nan", 1)])
def test_queue_ordinary_finite_or_nan_payload_corruption_still_fails(monkeypatch, corruption, iteration):
    run, observations = queue_cpu_harness(monkeypatch, corruption=corruption)
    with pytest.raises(AssertionError, match=rf"queue_{iteration}: ordinary operation changed.*actual_bits=0x"):
        run()
    assert observations["stages"][-1] == ("queue_lifetime_verify", "FAIL")


@pytest.mark.parametrize("field", ["signed", "scale", "bias", "select_status", "input_status"])
def test_queue_each_fused_output_single_bit_corruption_still_fails(monkeypatch, field):
    index = ("signed", "scale", "bias", "select_status", "input_status").index(field)
    run, observations = queue_cpu_harness(monkeypatch, corruption=f"field_{index}")
    with pytest.raises(AssertionError, match=rf"queue_0/{field}: unequal bytes"):
        run()
    assert observations["stages"][-1] == ("queue_lifetime_verify", "FAIL")


def test_queue_native_nan_payload_is_still_bit_exact(monkeypatch):
    run, observations = queue_cpu_harness(monkeypatch, corruption="field_nan_payload")
    with pytest.raises(AssertionError, match=r"queue_1/scale: unequal bytes"):
        run()
    assert observations["stages"][-1] == ("queue_lifetime_verify", "FAIL")


@pytest.mark.parametrize("mutation", ["actual_dtype", "expected_dtype", "shape"])
def test_ordinary_comparison_rejects_dtype_or_shape_changes(mutation):
    actual, expected = torch.zeros(2, 4), torch.zeros(2, 4)
    if mutation == "actual_dtype":
        actual = actual.double()
    elif mutation == "expected_dtype":
        expected = expected.double()
    else:
        actual = actual.reshape(4, 2)
    with pytest.raises(AssertionError, match="ordinary operation dtype/shape changed"):
        probe.assert_ordinary_bits(actual, expected, "queue_0")


def test_invalid_recovery_and_no_loop_upload_or_fence():
    source = inspect.getsource(probe.run_queue)
    loop = source.split("for iteration in range(QUEUE_ITERATIONS):", 1)[1].split("banks.clear()", 1)[0]
    assert "clone()" in loop and "del hidden, owner, ids, values" in loop
    assert all(token not in loop for token in (".cpu(", ".to(", "synchronize(", ".item(", "torch.tensor("))
    assert "invalid_slot" in probe.GRAPH_PHASES and "recovered_slot" in probe.GRAPH_PHASES
    assert "invalid_sign" in probe.GRAPH_PHASES and "recovered_sign" in probe.GRAPH_PHASES
    assert probe.QUEUE_ITERATIONS == 513
    assert len(probe.numeric_names()) == len(set(probe.numeric_names())) == 454
    assert len(probe.graph_names()) == 84


def test_plan_only_has_no_hardware_claim(capsys):
    assert probe.main(["--library", "example.so", "--plan-only", "--queue-lifetime"]) == 0
    result = __import__("json").loads(capsys.readouterr().out)
    assert result["status"] == "PLANNED"
    assert not result["device_execution_verified"] and not result["performance_verified"]


def test_native_bounds_owners_and_exact_outputs_are_explicit():
    native = Path(__file__).resolve().parents[3] / "csrc/vq2a8_ascendc_v4_v2"
    kernel = (native / "select_sign_kernel.cpp").read_text()
    binding = (native / "select_sign_binding.cpp").read_text()
    assert kernel.index("ValidResidentSlot(slot, experts_)") < kernel.index("static_cast<uint32_t>(slot)")
    assert "int32_t(0x7fc00000)" in kernel and "int16_t(0)" in kernel
    assert "return {signedOutput, selectedScale, selectedBias, selectStatus, inputStatus}" in binding
    callback = binding.split('RunOpApi("Vq2a8V4V2ResidentSelectSign"', 1)[1]
    assert "scaleOwners, biasOwners, signOwners" in callback
    assert "stream.stream()" not in callback
    assert "Record(owner, stream)" in binding
