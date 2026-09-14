# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4 orchestration/report contracts on CPU; never launch a model or NPU."""

import ast
import copy
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import accept_vq2a8_v4 as accept
from tools import benchmark_vq2a8_v4 as benchmark
from tools.vq2a8_perf_report import token_metrics


def arguments(tmp_path, *extra):
    return accept.parse_args(["--model", str(tmp_path / "model"), "--output-dir", str(tmp_path / "report"), *extra])


def test_default_plan_loads_only_v4_and_reuses_v1_kernel(tmp_path):
    args = arguments(tmp_path)
    steps = accept.commands(args, tmp_path / "report")
    assert [name for name, _ in steps] == ["environment", "preflight", "v4"]
    assert args.cache_reserve_gib == 8.0
    assert args.physical_npu == 1 and args.cases == "10:4"
    assert args.library.name == "libvq2a8_ascendc.so"
    assert args.library.parent.name == "vq2a8-ascendc-v023-v1"
    assert args.artifact == args.model / "experts_vq_ascend_v2"
    assert sum("benchmark_vq2a8_v4.py" in " ".join(command) for _, command in steps) == 1
    assert "--reference-only" not in steps[-1][1]
    assert all("--audit-consistency" not in command for _, command in steps)
    assert not any("v3" in item or "build_vq2" in item for _, command in steps for item in command)


def test_comparison_plan_uses_distinct_sequential_reference_worker(tmp_path):
    steps = accept.commands(arguments(tmp_path, "--compare-v1"), tmp_path / "report")
    assert [name for name, _ in steps] == ["environment", "preflight", "v1_reference", "v4"]
    assert "--reference-only" in steps[2][1] and "--reference-only" not in steps[3][1]
    assert "--reference-report" in steps[3][1]
    assert steps[3][1][-1] == str(tmp_path / "report/v1_reference/summary.json")


@pytest.mark.parametrize(
    "extra",
    [
        ["--physical-npu", "-1"],
        ["--cache-reserve-gib", "nan"],
        ["--cache-reserve-gib", "0"],
        ["--cache-budget-gib", "inf"],
        ["--warmups", "1"],
        ["--repeats", "4"],
        ["--cases", "127:4"],
        ["--tensor-parallel-size", "2"],
        ["--preparation", "fused"],
        ["--artifact", "/nonstandard/artifact"],
    ],
)
def test_unsupported_options_rejected_before_execution(tmp_path, extra):
    with pytest.raises(SystemExit) as error:
        arguments(tmp_path, *extra)
    assert error.value.code == 2


def test_plan_only_does_not_create_report_or_launch(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(accept, "supervise", lambda *a: pytest.fail("no children for plan-only"))
    monkeypatch.setattr(accept, "require_idle_device", lambda *a: pytest.fail("no device check for plan-only"))
    assert accept.run(arguments(tmp_path, "--plan-only")) == 0
    assert not (tmp_path / "report").exists()
    assert "plan_only_no_device_execution" in capsys.readouterr().out


@pytest.mark.parametrize("failed_stage", ["environment", "preflight", "v1_reference", "v4"])
@pytest.mark.parametrize("timeout", [False, True])
def test_child_failure_stops_all_following_stages(monkeypatch, tmp_path, failed_stage, timeout):
    monkeypatch.setattr(accept.platform, "system", lambda: "Linux")
    monkeypatch.setattr(accept, "require_idle_device", lambda *a: {"state": "idle"})
    calls = []

    def supervise(command, log, env, seconds):
        name = log.stem
        calls.append(name)
        assert env["ASCEND_RT_VISIBLE_DEVICES"] == "1"
        assert env["ASCEND_LAUNCH_BLOCKING"] == "0"
        assert seconds > 0
        failing = name == failed_stage
        log.write_text("No raw output should be forwarded", encoding="utf-8")
        return {"exit": int(failing and not timeout), "timeout": failing and timeout, "log": str(log)}

    monkeypatch.setattr(accept, "supervise", supervise)
    assert accept.run(arguments(tmp_path, "--compare-v1", "--physical-npu", "1")) == 1
    order = ["environment", "preflight", "v1_reference", "v4"]
    assert calls == order[: order.index(failed_stage) + 1]
    report = json.loads((tmp_path / "report/run.json").read_text())
    assert report["status"] == "FAIL" and not report["performance_measurement_verified"]


@pytest.mark.parametrize("compare", [False, True])
def test_supervisor_requires_complete_matching_v4_receipt(monkeypatch, tmp_path, compare):
    monkeypatch.setattr(accept.platform, "system", lambda: "Linux")
    monkeypatch.setattr(accept, "require_idle_device", lambda *a: {"state": "idle"})

    def supervise(command, log, env, seconds):
        if log.stem == "v4":
            directory = log.parent / "v4"
            directory.mkdir()
            (directory / "summary.json").write_text(
                json.dumps({"status": "PASS", "execution_policy": "ascendc", "performance_measurement_verified": True}),
                encoding="utf-8",
            )
        return {"exit": 0, "timeout": False, "log": str(log)}

    monkeypatch.setattr(accept, "supervise", supervise)
    assert accept.run(arguments(tmp_path, *(["--compare-v1"] if compare else []))) == 1


@pytest.mark.parametrize("selected", [0, 1, 7])
def test_idle_guard_only_checks_selected_card(monkeypatch, tmp_path, selected):
    text = "| NPU ID | Process id | Process name |\n"
    text += "".join(
        f"| No running processes found in NPU {index} |\n" if index == selected else f"| {index} | 1234 | busy |\n"
        for index in range(8)
    )

    def query(command, **kwargs):
        assert command == ["npu-smi", "info"] and kwargs["check"] and kwargs["timeout"] == 20
        return SimpleNamespace(stdout=text, stderr="")

    monkeypatch.setattr(benchmark.subprocess, "run", query)
    result = benchmark.require_idle_device(selected, tmp_path / "snapshot.log")
    assert result["state"] == "idle" and not result["exclusive_reservation"]


@pytest.mark.parametrize(
    "text",
    [
        "| NPU ID | Process id |\n| 1 | 1234 | busy |",
        "unknown format",
        "| NPU ID | Process id |\n| No running processes found in NPU 0 |",
        "| NPU ID | Process id |\n| No running processes found in NPU 1 |\n| 1 | 1234 | busy |",
    ],
)
def test_idle_guard_rejects_busy_unknown_missing_contradictory(monkeypatch, tmp_path, text):
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=text, stderr=""))
    with pytest.raises(RuntimeError, match="no device work started"):
        benchmark.require_idle_device(1, tmp_path / "snapshot.log")


@pytest.mark.parametrize("error", [OSError("missing"), subprocess.TimeoutExpired("npu-smi", 20)])
def test_failed_npu_query_fails_closed(monkeypatch, tmp_path, error):
    def query(*args, **kwargs):
        raise error

    monkeypatch.setattr(benchmark.subprocess, "run", query)
    with pytest.raises(RuntimeError, match="unknown"):
        benchmark.require_idle_device(1, tmp_path / "snapshot.log")


@pytest.mark.parametrize("block", ["preflight", "v1_reference", "v4"])
def test_every_device_stage_checks_idle_before_launch(monkeypatch, tmp_path, block):
    monkeypatch.setattr(accept.platform, "system", lambda: "Linux")
    calls = []

    def check(card, log):
        calls.append(("idle", log.stem.replace("-npu", ""), card))
        if log.stem == block + "-npu":
            raise RuntimeError("busy")
        return {"state": "idle"}

    def supervise(command, log, env, seconds):
        calls.append(("launch", log.stem, 1))
        return {"exit": 0, "timeout": False, "log": str(log)}

    monkeypatch.setattr(accept, "require_idle_device", check)
    monkeypatch.setattr(accept, "supervise", supervise)
    assert accept.run(arguments(tmp_path, "--compare-v1")) == 1
    assert calls[0] == ("launch", "environment", 1)
    assert calls[-1] == ("idle", block, 1)
    assert ("launch", block, 1) not in calls


def test_v4_log_console_keeps_progress_bounded_and_raw_noise_hidden(tmp_path):
    console = io.StringIO()
    relay = accept.V4SummaryChildLog(tmp_path / "child.log", "v4", console=console)
    relay._emit("irrelevant private raw log\nV4_ENGINE_")
    relay._emit("READY {}\nMODEL layer=2 stage=v4_resident_payload loaded=16\n")
    relay._emit("MODEL_V4_RESIDENT_READY {}\nV4_SAMPLE " + "x" * 5000 + "\nV4_ERROR=actual cause\n")
    output = console.getvalue()
    assert "private" not in output
    assert "V4_ENGINE_READY {}" in output and "v4_resident_payload" in output
    assert "MODEL_V4_RESIDENT_READY" in output and "V4_ERROR=actual cause" in output
    assert max(map(len, output.splitlines())) <= 2000


def test_supervisor_interrupt_is_recorded_not_left_running(monkeypatch, tmp_path):
    monkeypatch.setattr(accept.platform, "system", lambda: "Linux")

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(accept, "supervise", interrupt)
    with pytest.raises(KeyboardInterrupt):
        accept.run(arguments(tmp_path))
    report = json.loads((tmp_path / "report/run.json").read_text())
    assert report["status"] == "INTERRUPTED" and not report["performance_measurement_verified"]


def test_v4_supervise_runs_only_given_cpu_child_and_keeps_raw_log(tmp_path, capsys):
    log = tmp_path / "cpu-child.log"
    result = accept.supervise(
        [accept.sys.executable, "-c", 'print("raw framework detail"); print("V4_SAMPLE synthetic_cpu_test")'],
        log,
        dict(accept.os.environ),
        10,
    )
    assert result["exit"] == 0 and not result["timeout"] and result["reaped"]
    assert "raw framework detail" in log.read_text()
    output = capsys.readouterr().out
    assert "V4_SAMPLE synthetic_cpu_test" in output and "raw framework detail" not in output


def test_v4_supervise_timeout_cleans_only_owned_child(monkeypatch, tmp_path):
    class Child:
        returncode = None
        pid = 43210

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("only-child", timeout)

    child = Child()
    launched, cleaned = [], []

    def launch(command, **kwargs):
        launched.append((command, kwargs))
        assert kwargs["start_new_session"] is True
        return child

    monkeypatch.setattr(accept.subprocess, "Popen", launch)
    monkeypatch.setattr(accept, "terminate_child", lambda process: cleaned.append(process) or False)
    result = accept.supervise(["only-child"], tmp_path / "timeout.log", {}, 5)
    assert len(launched) == 1 and cleaned == [child]
    assert result["timeout"] and not result["reaped"] and result["exit"] is None


def test_supervisor_prints_actual_case_medians_and_transfer_evidence(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(accept.platform, "system", lambda: "Linux")
    monkeypatch.setattr(accept, "require_idle_device", lambda *a: {"state": "idle"})

    def supervise(command, log, env, seconds):
        if log.stem == "v4":
            directory = log.parent / "v4"
            directory.mkdir()
            receipt = {
                "status": "PASS",
                "execution_policy": "ascendc_v4",
                "performance_measurement_verified": True,
                "v1_comparison": "NOT_RUN",
                "cases": {
                    "p10-o4": {
                        "metrics": {
                            key: {"median": value} for key, value in [("ttft_s", 2), ("tpot_s", 0.25), ("e2e_s", 3)]
                        }
                    }
                },
                "samples": [{"case": "p10-o4", "kind": "measured", "expert_payload_h2d_bytes": 0}] * 5,
            }
            (directory / "summary.json").write_text(json.dumps(receipt), encoding="utf-8")
        return {"exit": 0, "timeout": False, "log": str(log)}

    monkeypatch.setattr(accept, "supervise", supervise)
    assert accept.run(arguments(tmp_path)) == 0
    output = capsys.readouterr().out
    assert '"tpot_median_s": 0.25' in output and '"expert_payload_h2d_bytes_per_request": 0' in output


def snapshots():
    state = {"cache": {"loads": 2, "evictions": 0, "resident_packed_bytes": 512}, "h2d_bytes": 512}
    return state, copy.deepcopy(state)


def test_no_expert_transfer_check_accepts_unchanged_startup_counters():
    benchmark.check_no_payload_transfer(*snapshots())


@pytest.mark.parametrize("field", ["loads", "evictions", "resident_packed_bytes", "h2d_bytes"])
def test_no_expert_transfer_check_rejects_any_runtime_change(field):
    before, after = snapshots()
    target = after if field == "h2d_bytes" else after["cache"]
    target[field] += 1
    with pytest.raises(ValueError):
        benchmark.check_no_payload_transfer(before, after)


@pytest.mark.parametrize("value", [True, None, -1, "0"])
def test_invalid_transfer_counters_cannot_certify_residency(value):
    before, after = snapshots()
    before["h2d_bytes"] = after["h2d_bytes"] = value
    with pytest.raises(ValueError):
        benchmark.check_no_payload_transfer(before, after)


def sample():
    return {
        **token_metrics([0.1, 0.2, 0.3, 0.4], 0.5, 4),
        "token_ready_s": [0.1, 0.2, 0.3, 0.4],
        "tokens": [1, 2, 3, 4],
        "finite": True,
        "forwards": 4,
        "native_calls": 86,
        "native_launches": 4,
        "device_span_ms": 490.0,
        "cache_delta": {"loads": 0, "hits": 10, "evictions": 0},
        "expert_payload_h2d_bytes": 0,
    }


def test_valid_sample_is_backed_by_timestamps_and_real_counts():
    benchmark.validate_sample(sample(), [1, 2, 3, 4], 4, resident=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("finite", False),
        ("forwards", 3),
        ("native_calls", 0),
        ("native_launches", 0),
        ("tpot_s", 0.00001),
        ("device_span_ms", float("nan")),
        ("expert_payload_h2d_bytes", 16),
        ("expert_payload_h2d_bytes", False),
    ],
)
def test_bad_sample_cannot_pass(field, value):
    record = sample()
    record[field] = value
    with pytest.raises(ValueError):
        benchmark.validate_sample(record, [1, 2, 3, 4], 4, resident=True)


def test_v1_reference_can_measure_real_misses_without_calling_it_v4():
    record = sample()
    record["cache_delta"]["loads"] = 1
    record["expert_payload_h2d_bytes"] = 512
    benchmark.validate_sample(record, [1, 2, 3, 4], 4, resident=False)
    with pytest.raises(ValueError):
        benchmark.validate_sample(record, [1, 2, 3, 4], 4, resident=True)


def test_reference_report_requires_same_identity_and_measured_v1(tmp_path):
    path = tmp_path / "summary.json"
    report = {
        "status": "PASS",
        "execution_policy": "ascendc",
        "optimization": "batched",
        "performance_measurement_verified": True,
        "identity": {"model": "same"},
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    assert benchmark.comparison_reference(path, {"model": "same"}) == report
    with pytest.raises(ValueError):
        benchmark.comparison_reference(path, {"model": "other"})
    for key, value in (
        ("execution_policy", "ascendc_v4"),
        ("optimization", "baseline"),
        ("performance_measurement_verified", False),
        ("status", "RUNNING"),
    ):
        bad = {**report, key: value}
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            benchmark.comparison_reference(path, {"model": "same"})


def test_numerical_gate_does_not_claim_changed_logits_or_tokens_exact():
    left = torch.tensor([[1.0, 2.0]])
    assert benchmark.numerical_gate([1], left, [1], left.clone())["accepted"]
    assert not benchmark.numerical_gate([1], left, [1], left + 0.001)["accepted"]
    assert not benchmark.numerical_gate([1], left, [0], left)["accepted"]


def test_default_run_preserves_v1_startup_then_configures_batched_and_no_audit():
    source = Path(benchmark.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert "check_runtime_environment" in calls
    assert "require_v023_stack" not in calls and "check_python_environment" not in calls
    assert "validate_v4_residency_evidence" in calls
    assert source.index("llm = LLM(**options)") < source.index('report["startup_snapshot"] = snapshot(llm)')
    assert "execution_policy=policy" in source
    assert 'optimization="batched"' in source
    assert "_configure_v3" not in source
    assert 'v1_comparison="NOT_RUN"' in source
    assert source.index('report["device_snapshot"] = require_idle_device(') < source.index(
        'report["device"] = _initialize_device('
    )
