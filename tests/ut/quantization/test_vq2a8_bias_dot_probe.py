# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe harness tests only; CPU results never certify NPU arithmetic."""

import copy
import json
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from tools import validate_vq2a8_bias_dot_probe as probe


@contextmanager
def stage(_):
    yield


@pytest.mark.parametrize("width,rows,pattern,seed", probe.cases())
def test_fixture_contract_and_reproducibility(width, rows, pattern, seed):
    x, weight = probe.fixture(width, rows, pattern, seed)
    assert probe.validate_inputs({"rotated": x, "weight_bias": weight}) == (x, weight)
    other = probe.fixture(width, rows, pattern, seed)
    assert torch.equal(x.view(torch.int32), other[0].view(torch.int32))
    assert torch.equal(weight.view(torch.int32), other[1].view(torch.int32))


def test_reference_preserves_production_rowwise_geometry(monkeypatch):
    x, weight = probe.fixture(2048, 6, "random", 17)
    original, calls = torch.matmul, []

    def matmul(a, b, *, out):
        calls.append((tuple(a.shape), tuple(b.shape), tuple(out.shape)))
        return original(a, b, out=out)

    monkeypatch.setattr(torch, "matmul", matmul)
    result = probe.reference(x, weight)
    assert result.shape == (6,)
    assert calls == [((1, 2048), (2048,), (1,))] * 6


@pytest.mark.parametrize("bits", (0, -2147483648, 1065353216, 2143289345))
def test_bit_gate_rejects_one_changed_bit_including_nan_zero(bits):
    expected = torch.tensor([bits], dtype=torch.int32).view(torch.float32)
    actual = torch.tensor([bits ^ 1], dtype=torch.int32).view(torch.float32)
    probe.assert_bits(expected.clone(), expected, "same")
    with pytest.raises(AssertionError, match="differing FP32"):
        probe.assert_bits(actual, expected, "different")


@pytest.mark.parametrize("bad", (None, {}, {"rotated": 1, "weight_bias": 1}, {"x": torch.zeros(1)}))
def test_bad_real_inputs_fail(bad):
    with pytest.raises(ValueError):
        probe.validate_inputs(bad)


def test_numeric_harness_cannot_pass_corrupted_candidate(monkeypatch):
    monkeypatch.setattr(probe, "cases", lambda: [(2048, 1, "random", 17)])
    assert probe.run_numeric("cpu", probe.reference, stage) == probe.numeric_names()

    def corrupt(x, weight):
        output = probe.reference(x, weight)
        output.view(torch.int32)[0] ^= 1
        return output

    with pytest.raises(AssertionError, match="NOT model-compatible"):
        probe.run_numeric("cpu", corrupt, stage)


def test_queue_ordinary_reference_is_same_device_and_errors_fail(monkeypatch):
    monkeypatch.setattr(probe, "QUEUE_ITERATIONS", 4)
    monkeypatch.setattr(probe, "PRESSURE_BYTES", 64)
    assert probe.run_queue("cpu", probe.reference, stage) == probe.queue_evidence()

    def corrupt(x, weight):
        output = probe.reference(x, weight)
        output[0] += 1
        return output

    with pytest.raises(AssertionError, match="queue_0"):
        probe.run_queue("cpu", corrupt, stage)


def receipt(queue=True, real=False):
    results = {
        "numeric": probe.numeric_names() + (["real_inputs"] if real else []),
        "native_contract": probe.CONTRACT_CASES,
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
            *([{"event": "PASS", "stage": "queue_lifetime_verify"}] if queue else []),
            {
                "event": "CASE_PASS",
                "case": probe.CASE,
                "native_abi": 1,
                "results": results,
                "library": {"path": "/test/" + probe.LIBRARY_NAME, "sha256": "a" * 64},
                "device_execution_verified": True,
                "graph_verified": True,
                "model_integration_verified": False,
                "performance_verified": False,
                "model_dispatch_enabled": False,
                "real_inputs": {"path": "/test/input.pt", "sha256": "b" * 64} if real else None,
            },
        ],
    }


@pytest.mark.parametrize("queue", (False, True))
@pytest.mark.parametrize("real", (False, True))
def test_strict_receipt(queue, real):
    probe.validate_child_evidence(receipt(queue, real), queue, real)


@pytest.mark.parametrize(
    "field,value",
    (
        ("native_abi", True),
        ("graph_verified", False),
        ("model_dispatch_enabled", True),
        ("model_integration_verified", True),
        ("performance_verified", True),
    ),
)
def test_probe_does_not_certify_missing_execution_or_model(field, value):
    value_receipt = receipt()
    value_receipt["events"][-1][field] = value
    with pytest.raises(ValueError, match="Incomplete"):
        probe.validate_child_evidence(value_receipt, True)


def test_missing_cases_and_false_stage_pass_are_rejected():
    value = receipt()
    value["events"][-1]["results"]["numeric"].pop()
    with pytest.raises(ValueError):
        probe.validate_child_evidence(value, True)
    value = receipt()
    value["events"] = [event for event in value["events"] if event.get("stage") != "queue_lifetime_verify"]
    with pytest.raises(ValueError):
        probe.validate_child_evidence(value, True)
    value = receipt()
    value["events"].insert(0, {"event": "CASE_FAIL"})
    with pytest.raises(ValueError):
        probe.validate_child_evidence(value, True)


def test_real_inputs_identity_required():
    value = receipt(real=True)
    value["events"][-1]["real_inputs"] = {}
    with pytest.raises(ValueError, match="identity"):
        probe.validate_child_evidence(value, True, True)


def test_plan_is_not_device_evidence(capsys):
    assert probe.main(["--library", "missing.so", "--plan-only", "--queue-lifetime"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "PLANNED"
    assert not value["device_execution_verified"]
    assert not value["model_dispatch_enabled"]


def test_dot_probe_has_no_production_dispatch():
    root = Path(__file__).resolve().parents[3]
    production = root / "vllm_ascend"
    assert not [path for path in production.rglob("*.py") if "bias_dot_rows_probe" in path.read_text(encoding="utf-8")]
    binding = (root / "csrc/vq2a8_ascendc_v4_v2/bias_dot_probe_binding.cpp").read_text()
    assert "recordStream" in binding
    assert "[launchStream, blocks, rotated, weightBias, output, rows, width]" in binding
    assert "false);" in binding


def test_receipt_fixture_is_not_mutated_by_validation():
    value = receipt()
    expected = copy.deepcopy(value)
    probe.validate_child_evidence(value, True)
    assert value == expected
