# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU orchestration contracts, not proof of NPU task-queue deadlock freedom."""

import inspect
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
    assert plan["minimum_operator_calls"] == 4097 * 9 > 4096
    assert plan["phases"][0]["minimum_operator_calls"] == 4097 * 3
    assert plan["phases"][0]["operator_paths"] == ["resident_select", "resident_project", "matmul"]
    assert plan["phases"][0]["grouped_descriptor_copy_is_blocking"] is False
    assert plan["phases"][1]["grouped_descriptor_copy_is_blocking"] is True
    assert plan["short_checks_and_pressure_share_process"] is True
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
    assert plan["queue_lifetime"]["minimum_operator_calls"] == 18441
    assert plan["queue_lifetime"]["phases"][0]["minimum_operator_calls"] == 6147
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
            self.selected_references = ()
            self.calls = 0
            instances.append(self)

        def select(self, ids):
            metadata = [
                torch.stack([self.payloads[expert][name] for expert in ids.tolist()])
                for name in tool.PAYLOAD_FIELDS[3:]
            ]
            if corruption == "metadata":
                metadata[0] = metadata[0] + 1
            result = [*metadata, torch.ones_like(ids, dtype=torch.int32)]
            self.selected_references = tuple(weakref.ref(value) for value in result)
            return result

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


class MixedOperators:
    """CPU orchestration stand-ins; never used by the physical-NPU worker."""

    def __init__(self, reference, corruption=None, failure=None):
        self.reference = reference
        self.corruption = corruption
        self.failure = failure
        self.calls = {"projection": 0, "grouped_projection": 0, "grouped_projection_pipeline": 0}
        self.inputs = ()
        self.outputs = ()

    def _run(self, name, jobs):
        self.calls[name] += 1
        if self.failure is not None and name == self.failure[0]:
            raise self.failure[1]
        self.inputs = tuple(weakref.ref(value) for job in jobs for value in job[:3])
        outputs = [self.reference(*job) for job in jobs]
        # Keep setup's original oracle intact; corrupt only the stressed call.
        in_pressure = name != "projection" or self.calls[name] > tool.EXPERTS
        if in_pressure and self.corruption == name:
            outputs[0] = outputs[0] + 1
        self.outputs = tuple(weakref.ref(value) for value in outputs)
        return outputs

    def projection(self, *job):
        return self._run("projection", [job])[0]

    def grouped(self, jobs):
        return self._run("grouped_projection", jobs)

    def pipeline(self, jobs):
        return self._run("grouped_projection_pipeline", jobs)


def test_queue_lifetime_cpu_orchestration_releases_inputs_before_matmul_and_fences_only_at_boundaries(monkeypatch):
    bank, projection, instances = orchestration_bank()
    operators = MixedOperators(projection)
    original_matmul = torch.matmul
    matmul_count = 0
    sync_counts = []
    progress = []
    mixed_progress = []
    events = []
    matmul_references = ()

    def emit(case, event, **fields):
        events.append((case, event, fields))
        if event == "QUEUE_STEP" and fields["phase"] == "RETURN":
            step = fields["step"]
            if step == "matmul_release_inputs":
                assert all(reference() is None for reference in matmul_references)
            elif step == "resident_release_outputs":
                assert all(reference() is None for reference in instances[0].selected_references)
            elif any(step == f"{name}_release_inputs" for name in operators.calls):
                assert all(reference() is None for reference in operators.inputs)
            elif any(step == f"{name}_release_outputs" for name in operators.calls):
                assert all(reference() is None for reference in operators.outputs)

    def matmul(left, right):
        nonlocal matmul_count, matmul_references
        assert all(reference() is None for reference in instances[0].references), "Python retained temporary inputs"
        matmul_references = (weakref.ref(left), weakref.ref(right))
        matmul_count += 1
        return original_matmul(left, right)

    monkeypatch.setattr(torch, "matmul", matmul)
    monkeypatch.setattr(tool, "emit", emit)
    iterations = tool.QUEUE_MIN_ITERATIONS
    report = tool.run_queue_lifetime_checks(
        torch.device("cpu"),
        bank_factory=bank,
        projection=operators.projection,
        grouped_projection=operators.grouped,
        grouped_projection_pipeline=operators.pipeline,
        synchronize=lambda: sync_counts.append((matmul_count, dict(operators.calls))),
        iterations=iterations,
        progress=progress.append,
        mixed_progress=mixed_progress.append,
    )
    assert instances[0].calls == matmul_count == iterations * 2
    # The entire original resident phase precedes ALL grouped descriptor copies.
    setup_calls = {"projection": tool.EXPERTS, "grouped_projection": 0, "grouped_projection_pipeline": 0}
    assert sync_counts[:2] == [(0, setup_calls), (iterations, setup_calls)]
    assert sync_counts[2] == (
        iterations * 2,
        {
            "projection": tool.EXPERTS + iterations,
            "grouped_projection": iterations,
            "grouped_projection_pipeline": iterations,
        },
    )
    assert len(sync_counts) == 3
    assert progress == mixed_progress == [*range(1, 9), 256, 512, 768, 1024, 1280, iterations]
    assert report["minimum_operator_calls"] == iterations * 9 > 4096
    assert report["phases"][0]["minimum_operator_calls"] == iterations * 3 > 4096
    assert report["mixed_grouped_jobs"] == 2
    assert report["explicit_synchronization_stages"] == [
        "queue_lifetime_setup",
        "queue_lifetime_wrap",
        "queue_lifetime_mixed",
    ]
    returns = [fields for _, event, fields in events if event == "QUEUE_STEP" and fields["phase"] == "RETURN"]
    assert {record["iteration"] for record in returns} == {*range(1, 9), 256, 512, 768, 1024, 1280}
    for name in ("left_clone", "right_clone", "matmul", *operators.calls):
        assert any(record["step"] == name for record in returns)
    assert report["retained_output_samples"] == 2
    assert report["temporary_inputs_dropped_before_matmul"] is True
    assert report["all_iteration_checks_passed"] is True
    assert report["preserved_outputs_bitwise_exact"] is True
    assert report["runtime_queue_enabled_or_slots_measured"] is False
    assert "device_execution_verified" not in report
    assert report["full_model_verified"] is report["performance_verified"] is False


@pytest.mark.parametrize(
    "corruption",
    ["metadata", "projection", "validity", "matmul", "single", "grouped_projection", "grouped_projection_pipeline"],
)
def test_queue_lifetime_device_checks_cannot_be_skipped_by_deferred_validation(monkeypatch, corruption):
    bank, projection, _ = orchestration_bank(corruption)
    operators = MixedOperators(projection, "projection" if corruption == "single" else corruption)
    monkeypatch.setattr(tool, "emit", lambda *_, **__: None)
    if corruption == "matmul":
        original = torch.matmul
        monkeypatch.setattr(torch, "matmul", lambda left, right: original(left, right) + 1)
    with pytest.raises(AssertionError, match="Queue lifetime"):
        tool.run_queue_lifetime_checks(
            torch.device("cpu"),
            bank_factory=bank,
            projection=operators.projection,
            grouped_projection=operators.grouped,
            grouped_projection_pipeline=operators.pipeline,
            synchronize=lambda: None,
            iterations=tool.QUEUE_MIN_ITERATIONS,
            progress=lambda _: None,
            mixed_progress=lambda _: None,
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


def test_queue_lifetime_requires_both_real_grouped_entries_before_allocating():
    with pytest.raises(ValueError, match="no fallback"):
        tool.run_queue_lifetime_checks(torch.device("cpu"), bank_factory=None, projection=None, synchronize=None)


@pytest.mark.parametrize("name", ["projection", "grouped_projection", "grouped_projection_pipeline"])
def test_mixed_projection_exception_identity_and_precise_failed_step_are_preserved(monkeypatch, name):
    _, reference, _ = orchestration_bank()
    failure = RuntimeError(f"injected {name} failure")
    operators = MixedOperators(reference, failure=(name, failure))
    prepared = (torch.ones(1, 512).to(torch.float8_e4m3fn), torch.ones(1), torch.zeros(1))
    template = {"prepared": prepared, "projection_payload": (None, None, None), "expected": reference(*prepared)}
    events = []
    monkeypatch.setattr(tool, "emit", lambda case, event, **fields: events.append((event, fields)))
    with pytest.raises(RuntimeError) as error:
        tool._queue_mixed_projection_checks(
            [template, template],
            operators.projection,
            operators.grouped,
            operators.pipeline,
            tool._queue_step_recorder("queue_lifetime_mixed", 1),
        )
    assert error.value is failure
    assert events[-1][0] == "QUEUE_STEP"
    assert events[-1][1] == {"stage": "queue_lifetime_mixed", "iteration": 1, "step": name, "phase": "FAIL"}
    assert not any(fields["step"] == name and fields["phase"] == "RETURN" for _, fields in events)


def test_mixed_queue_runs_after_short_checks_in_same_worker_without_per_step_tensor_reads_or_fences():
    child = inspect.getsource(tool.run_case_child)
    assert child.index("checks = run_synthetic_checks(") < child.index("native_contracts = run_native_contract_checks(")
    assert child.index("native_contracts = run_native_contract_checks(") < child.index("**run_queue_lifetime_checks(")
    assert "grouped_projection=grouped_projection," in child
    assert "grouped_projection_pipeline=grouped_projection_pipeline," in child
    for helper in (tool._queue_lifetime_iteration, tool._queue_mixed_projection_checks, tool._queue_step_recorder):
        source = inspect.getsource(helper)
        assert not any(token in source for token in (".synchronize(", ".cpu(", ".item(", "empty_cache("))
