# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""I control-flow/acceptance contracts only, not native compilation or NPU proof."""

import copy
import inspect
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import validate_vq2a8_swiglu_select_sign as probe
from vllm_ascend.quantization.vq2a8_reference import deepseek_v4_swiglu_reference


@pytest.mark.parametrize("version", [True, None, 0, 2, "1", 1.0])
def test_swiglu_select_sign_abi_fail_closed(version):
    with pytest.raises(RuntimeError, match="ABI 1"):
        probe.require_abi(SimpleNamespace(swiglu_select_sign_version=lambda: version))


def test_swiglu_select_sign_missing_abi_or_member_cannot_fallback():
    with pytest.raises(RuntimeError, match="no fallback"):
        probe.require_abi(SimpleNamespace())
    with pytest.raises(RuntimeError, match="no fallback"):
        probe.fused(SimpleNamespace(), None, None, 7.0)
    probe.require_abi(SimpleNamespace(swiglu_select_sign_version=lambda: 1))


@pytest.mark.parametrize("width", probe.WIDTHS)
@pytest.mark.parametrize("groups", range(1, 7))
@pytest.mark.parametrize("layout", probe.LAYOUTS)
@pytest.mark.parametrize("pattern", probe.PATTERNS)
def test_swiglu_select_sign_gate_up_fixture_and_queue_view(width, groups, layout, pattern):
    value, owner = probe.make_gate_up("cpu", width, groups, layout, pattern)
    reconstructed = probe.gate_up_view(owner.clone(), groups, width, layout)
    assert value.shape == (groups, 2 * width)
    assert value.dtype == torch.bfloat16
    assert value.data_ptr() % 32 == 0
    assert value.stride(1) == 1
    assert value.stride(0) == 0 or value.stride(0) >= 2 * width
    assert value.stride() == reconstructed.stride()
    assert value.storage_offset() == reconstructed.storage_offset()
    assert torch.equal(value.contiguous().view(torch.int16), reconstructed.contiguous().view(torch.int16))


def test_swiglu_select_sign_dispatch_does_not_read_or_copy(monkeypatch):
    x = torch.zeros(2, 4096, dtype=torch.bfloat16)
    ids = torch.zeros(2, dtype=torch.int64)
    sentinel = object()
    calls = []

    def operation(value, slots, limit):
        calls.append((value is x, slots is ids, limit))
        return sentinel

    def forbidden(*args, **kwargs):
        raise AssertionError("Host read or tensor materialization in dispatch")

    for name in ("item", "cpu", "numpy", "tolist", "contiguous", "clone"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    assert probe.fused(SimpleNamespace(swiglu_select_sign=operation), x, ids, 7.1) is sentinel
    assert calls == [(True, True, 7.1)]


def test_swiglu_select_sign_reference_has_two_bf16_boundaries():
    values = torch.tensor([[-0.36328125, 0.546875, 0.203125, -1.765625]], dtype=torch.bfloat16)
    expected = torch.nn.functional.silu(values[:, :2]) * values[:, 2:]
    actual = deepseek_v4_swiglu_reference(values, 0.0)
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert "deepseek_v4_swiglu_reference" in inspect.getsource(probe.reference)
    assert "activation_sign_strided" in inspect.getsource(probe.reference)


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
                "model_dispatch_enabled": False,
                "model_integration_verified": False,
                "performance_verified": False,
            },
        ],
    }


@pytest.mark.parametrize("queue", [False, True])
def test_swiglu_select_sign_receipt_complete_exact_matrix(queue):
    receipt = valid_receipt(queue)
    probe.validate_child_evidence(receipt, queue)
    for key in ("numeric", "graph"):
        changed = copy.deepcopy(receipt)
        changed["events"][-1]["results"][key].pop()
        with pytest.raises(ValueError, match="Incomplete"):
            probe.validate_child_evidence(changed, queue)
    assert len(probe.numeric_names()) == len(set(probe.numeric_names())) == 704
    assert len(probe.graph_names()) == len(set(probe.graph_names())) == 84


@pytest.mark.parametrize("field,value", [("status", "FAIL"), ("exit_code", 1), ("reaped", False)])
def test_swiglu_select_sign_failed_process_never_upgraded(field, value):
    receipt = valid_receipt(True)
    receipt[field] = value
    with pytest.raises(ValueError, match="Incomplete"):
        probe.validate_child_evidence(receipt, True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("native_abi", True),
        ("native_abi", 2),
        ("device_execution_verified", False),
        ("graph_verified", False),
        ("model_integration_verified", True),
        ("performance_verified", True),
        ("model_dispatch_enabled", True),
    ],
)
def test_swiglu_select_sign_receipt_does_not_certify_model(field, value):
    receipt = valid_receipt(True)
    receipt["events"][-1][field] = value
    with pytest.raises(ValueError, match="Incomplete"):
        probe.validate_child_evidence(receipt, True)


@pytest.mark.parametrize("missing", ["numeric", "graph", "verify", "sync", "queue_evidence", "no_dispatch"])
def test_swiglu_select_sign_receipt_cannot_omit_stage(missing):
    receipt = valid_receipt(True)
    final = receipt["events"][-1]
    if missing in ("numeric", "graph"):
        final["results"].pop(missing)
    elif missing == "queue_evidence":
        final["results"]["queue_lifetime"].pop("ordinary_reference")
    elif missing == "no_dispatch":
        final.pop("model_dispatch_enabled")
    else:
        stage = "queue_lifetime_verify" if missing == "verify" else "final_sync"
        receipt["events"] = [event for event in receipt["events"] if event.get("stage") != stage]
    with pytest.raises(ValueError, match="Incomplete"):
        probe.validate_child_evidence(receipt, True)


def test_swiglu_select_sign_one_bit_including_nan_payload_is_failure():
    value = torch.tensor([[float("nan"), -0.0]], dtype=torch.float32)
    expected = (
        value,
        value.clone(),
        value.clone(),
        torch.ones(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    )
    actual = tuple(x.clone() for x in expected)
    actual[0].view(torch.int32)[0, 0] ^= 1
    with pytest.raises(AssertionError, match="unequal bytes"):
        probe.assert_bits(actual, expected, "nan")


class CpuBank:
    """Independent CPU oracle to exercise probe control flow, not the candidate."""

    def __init__(self, packed, books, order, scale, bias, sign):
        self.scale, self.bias, self.sign = scale, bias, sign

    def select(self, ids):
        selected = ((ids >= 0) & (ids < len(self.scale))).to(torch.int32)
        safe_ids = ids.clamp(0, len(self.scale) - 1)
        scale = torch.stack(self.scale)[safe_ids]
        bias = torch.stack(self.bias)[safe_ids]
        sign = torch.stack(self.sign)[safe_ids]
        return (
            torch.where(selected[:, None] != 0, scale, float("nan")),
            torch.where(selected[:, None] != 0, bias, float("nan")),
            torch.where(selected[:, None] != 0, sign, 0),
            selected,
        )

    def swiglu_select_sign(self, value, ids, limit):
        return probe.reference(self, value, ids, limit, SimpleNamespace(activation_sign_strided=cpu_sign))


def cpu_sign(activated, scale, bias, sign):
    valid = (torch.isfinite(activated) & torch.isfinite(scale) & torch.isfinite(bias) & (sign.abs() == 1)).all(1).int()
    return activated.float() * sign.float(), valid


@contextmanager
def cpu_stage(name):
    yield


def test_swiglu_select_sign_real_numeric_control_flow_matches_evidence(monkeypatch):
    monkeypatch.setattr(probe, "WIDTHS", (2048,))
    monkeypatch.setattr(probe, "LIMITS", (7.0,))
    monkeypatch.setattr(probe, "LAYOUTS", ("padded",))
    monkeypatch.setattr(probe, "PATTERNS", ("random",))
    names = probe.run_numeric("cpu", CpuBank, SimpleNamespace(activation_sign_strided=cpu_sign), cpu_stage)
    assert names == probe.numeric_names()


def test_swiglu_select_sign_real_queue_flow_has_no_cpu_reads_inside_loop(monkeypatch):
    monkeypatch.setattr(probe, "WIDTHS", (2048,))
    monkeypatch.setattr(probe, "LAYOUTS", ("padded",))
    monkeypatch.setattr(probe, "QUEUE_ITERATIONS", 7)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)
    active = []
    original_cpu = torch.Tensor.cpu

    def cpu(value, *args, **kwargs):
        assert "queue_lifetime" not in active
        return original_cpu(value, *args, **kwargs)

    @contextmanager
    def stage(name):
        active.append(name)
        yield
        active.remove(name)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    result = probe.run_queue("cpu", CpuBank, SimpleNamespace(activation_sign_strided=cpu_sign), stage)
    assert result == probe.queue_evidence()
    assert result["native_stream_check_may_drain_host_queue"] is True
    assert result["runtime_queue_slots_measured"] is False


def test_swiglu_select_sign_native_guards_and_bf16_fences_are_present():
    root = Path(__file__).resolve().parents[3]
    source = (root / "csrc/vq2a8_ascendc_v4_v2/swiglu_select_sign_binding.cpp").read_text()
    kernel = (root / "csrc/vq2a8_ascendc_v4_v2/swiglu_select_sign_kernel.cpp").read_text()
    assert "hidden.size(1) == 2 * width" in source
    assert "hidden.scalar_type() == at::kBFloat16" in source
    assert "std::isfinite(limit) && limit >= 0" in source
    assert "rowStride <= (available - 2 * width)" in source
    assert "launchStream == bankStream" in source
    assert "scaleOwners, biasOwners, signOwners" in source
    assert "NPUCachingAllocator::recordStream" in source
    assert source.index("stream.stream()") < source.index("OpCommand::RunOpApi")
    assert kernel.count("Cast(inputUb_.Get<InputT>(), x, RoundMode::CAST_RINT") == 3
    assert "Mins(scratch, x, clampLimit_" in kernel
    assert "Maxs(x, x" not in kernel  # gate only clamps its positive side
    assert "Maxs(scratch, scratch, -clampLimit_" in kernel
    assert "Select(x, mask, scratch, x," in kernel
    assert "Select(up, mask, scratch, up," in kernel
    assert "ValidResidentSlot(slot, experts_)" in kernel
    assert "Fence<HardEvent::MTE3_V>()" in kernel


def test_swiglu_select_sign_plan_only_does_not_claim_device_or_model(tmp_path, capsys):
    assert probe.main(["--library", str(tmp_path / probe.LIBRARY_NAME), "--queue-lifetime", "--plan-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PLANNED"
    assert report["scope"] == "swiglu_resident_select_sign_only"
    assert report["queue_lifetime_requested"] is True
    for key in (
        "device_execution_verified",
        "graph_verified",
        "model_dispatch_enabled",
        "model_integration_verified",
        "performance_verified",
    ):
        assert report[key] is False
    assert "--queue-lifetime" in report["command"]


@pytest.mark.parametrize(
    "args", [["--physical-npu", "-1"], ["--timeout-s", "0"], ["--timeout-s", "7201"], ["--child", "--plan-only"]]
)
def test_swiglu_select_sign_invalid_cli_rejected(args):
    with pytest.raises(SystemExit):
        probe.parse_args(["--library", probe.LIBRARY_NAME, *args])
