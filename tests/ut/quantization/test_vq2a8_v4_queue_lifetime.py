# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU orchestration contracts, not proof of NPU task-queue deadlock freedom."""

import json
import subprocess
import sys
import weakref
from pathlib import Path

import pytest
import torch

from tools import validate_vq2a8_v4_device_route as tool

REPO = Path(__file__).resolve().parents[3]


def test_queue_lifetime_opt_in_keeps_default_short_probe_unchanged():
    args = tool.parse_args([])
    assert not args.queue_lifetime
    assert args.queue_iterations == 2049
    assert "--queue-lifetime" not in tool.child_command(args)
    assert tool.queue_lifetime_plan(args)["iterations"] == 0
    assert args.timeout_s == 180 and not args.allow_busy


def test_queue_lifetime_command_propagates_bounded_async_condition():
    args = tool.parse_args(["--queue-lifetime", "--queue-iterations", "4097", "--physical-npu", "1"])
    command = tool.child_command(args)
    assert "--queue-lifetime" in command
    assert command[command.index("--queue-iterations") + 1] == "4097"
    assert command[command.index("--launch-blocking") + 1] == "0"
    plan = tool.queue_lifetime_plan(args)
    assert plan["minimum_operator_calls"] == 4097 * 3 > 4096
    assert plan["runtime_queue_enabled_or_slots_measured"] is False
    assert plan["requires_runtime_task_queue_enabled"] is True
    assert plan["explicit_per_iteration_synchronize"] is False
    assert plan["native_stream_check_may_drain_host_queue"] is True
    assert plan["model_weights_loaded"] is plan["performance_verified"] is False


@pytest.mark.parametrize(
    "argv",
    [
        ["--queue-lifetime", "--launch-blocking", "1"],
        ["--queue-lifetime", "--queue-iterations", "1365"],
        ["--queue-lifetime", "--queue-iterations", "8193"],
        ["--queue-lifetime", "--queue-iterations", "-1"],
        ["--queue-lifetime", "--queue-iterations", "2.5"],
        ["--queue-iterations", "4097"],
    ],
)
def test_queue_lifetime_rejects_conditions_that_cannot_exercise_bounded_async_reuse(argv):
    with pytest.raises(SystemExit) as error:
        tool.parse_args(argv)
    assert error.value.code == 2


@pytest.mark.parametrize("iterations", [1366, 8192])
def test_queue_iteration_bounds_exceed_reference_ring(iterations):
    args = tool.parse_args(["--queue-lifetime", "--queue-iterations", str(iterations)])
    assert tool.queue_lifetime_plan(args)["minimum_operator_calls"] > tool.QUEUE_CAPACITY_REFERENCE


def test_queue_lifetime_plan_imports_no_runtime_and_creates_no_report(tmp_path):
    script = REPO / "tools/validate_vq2a8_v4_device_route.py"
    directory = tmp_path / "not-created"
    code = (
        "import builtins,runpy,sys\n"
        "original=builtins.__import__\n"
        "def guard(name,*args,**kwargs):\n"
        "    if name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend'):\n"
        "        raise AssertionError('plan imported runtime: '+name)\n"
        "    return original(name,*args,**kwargs)\n"
        "builtins.__import__=guard\n"
        f"sys.argv=[{str(script)!r},'--plan-only','--queue-lifetime','--report-dir',{str(directory)!r}]\n"
        f"runpy.run_path({str(script)!r},run_name='__main__')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["device_execution_verified"] is False
    assert plan["queue_lifetime"]["enabled"] is True
    assert plan["queue_lifetime"]["minimum_operator_calls"] == 6147
    assert not directory.exists()


def orchestration_bank(corruption=None):
    """An intentionally simple CPU stub, not a replacement native math oracle."""
    instances = []

    def projection(x, scale, bias, *_):
        return ((x.float().sum(dim=1) * scale + bias)[:, None].expand(-1, tool.OUTPUT_COLUMNS)).bfloat16().contiguous()

    class Bank:
        def __init__(self, *fields):
            self.payloads = [dict(zip(tool.PAYLOAD_FIELDS, values)) for values in zip(*fields)]
            self.references = ()
            self.calls = 0
            instances.append(self)

        def select(self, ids):
            metadata = [
                torch.stack([self.payloads[expert][name] for expert in ids.tolist()])
                for name in tool.PAYLOAD_FIELDS[3:]
            ]
            if corruption == "metadata":
                metadata[0] = metadata[0] + 1
            return [*metadata, torch.ones_like(ids, dtype=torch.int32)]

        def project(self, x, scale, bias, ids):
            self.references = tuple(weakref.ref(value) for value in (x, scale, bias, ids))
            self.calls += 1
            output = projection(x, scale, bias)
            if corruption == "projection":
                output[0, 0] = float("nan")
            valid = torch.ones_like(ids, dtype=torch.int32)
            if corruption == "validity":
                valid.zero_()
            return output, valid

    return Bank, projection, instances


def test_queue_lifetime_cpu_orchestration_releases_inputs_before_matmul_and_fences_only_at_boundaries(monkeypatch):
    bank, projection, instances = orchestration_bank()
    original_matmul = torch.matmul
    matmul_count = 0
    sync_counts = []
    progress = []

    def matmul(left, right):
        nonlocal matmul_count
        assert all(reference() is None for reference in instances[0].references), "Python retained temporary inputs"
        matmul_count += 1
        return original_matmul(left, right)

    monkeypatch.setattr(torch, "matmul", matmul)
    iterations = tool.QUEUE_MIN_ITERATIONS
    report = tool.run_queue_lifetime_checks(
        torch.device("cpu"),
        bank_factory=bank,
        projection=projection,
        synchronize=lambda: sync_counts.append(matmul_count),
        iterations=iterations,
        progress=progress.append,
    )
    assert instances[0].calls == matmul_count == iterations
    assert sync_counts == [0, iterations]
    assert progress == [256, 512, 768, 1024, 1280, iterations]
    assert report["minimum_operator_calls"] == iterations * 3 > 4096
    assert report["retained_output_samples"] == 2
    assert report["temporary_inputs_dropped_before_matmul"] is True
    assert report["all_iteration_checks_passed"] is True
    assert report["preserved_outputs_bitwise_exact"] is True
    assert report["runtime_queue_enabled_or_slots_measured"] is False
    assert "device_execution_verified" not in report
    assert report["full_model_verified"] is report["performance_verified"] is False


@pytest.mark.parametrize("corruption", ["metadata", "projection", "validity", "matmul"])
def test_queue_lifetime_device_checks_cannot_be_skipped_by_deferred_validation(monkeypatch, corruption):
    bank, projection, _ = orchestration_bank(corruption)
    if corruption == "matmul":
        original = torch.matmul
        monkeypatch.setattr(torch, "matmul", lambda left, right: original(left, right) + 1)
    with pytest.raises(AssertionError, match="Queue lifetime"):
        tool.run_queue_lifetime_checks(
            torch.device("cpu"),
            bank_factory=bank,
            projection=projection,
            synchronize=lambda: None,
            iterations=tool.QUEUE_MIN_ITERATIONS,
            progress=lambda _: None,
        )


@pytest.mark.parametrize("iterations", [False, 0, 1365, 8193])
def test_queue_lifetime_helper_rejects_unbounded_or_insufficient_work_before_bank_creation(iterations):
    with pytest.raises(ValueError, match="Queue lifetime iterations"):
        tool.run_queue_lifetime_checks(
            torch.device("cpu"),
            bank_factory=None,
            projection=None,
            synchronize=None,
            iterations=iterations,
        )
