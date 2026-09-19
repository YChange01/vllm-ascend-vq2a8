# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts only; not native compilation, bitwise NPU or performance proof."""

import copy
import inspect
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
