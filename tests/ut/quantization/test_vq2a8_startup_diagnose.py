# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only startup probe contracts, not proof of Ascend execution or speed."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import diagnose_vq2a8_tp1_startup as tool
from tools import vq2a8_startup_ops as ops_probe
from tools import vq2a8_startup_projection as projection_probe

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "tools" / "diagnose_vq2a8_tp1_startup.py"
PROCESS_HEADER = "| NPU ID | Process id | Process name | Process memory(MB) |\n"
IDLE_ONE = "| No running processes found in NPU 1 |\n"
BUSY_ONE = "| 1 | 112345 | python3.11 | 73456 |\n"


def events_in(text):
    return [json.loads(line.split("STARTUP_PROBE ", 1)[1]) for line in text.splitlines() if "STARTUP_PROBE " in line]


def test_startup_probe_defaults_are_weightless_single_card():
    args = tool.parse_args([])
    assert args.physical_npu == 1
    assert args.timeout_s == 120
    assert args.launch_blocking == "1"
    assert not args.allow_busy and not args.plan_only
    assert tuple(args.cases) == tool.CASES
    assert tool.CASES == (
        "basic",
        "hc_pre_m2",
        "hc_pre_m128",
        "hc_post_m128",
        "prepare_m32",
        "prepare_group6",
        "projection_m1",
        "projection_m32",
        "projection_group6",
    )
    assert args.library.as_posix().endswith("build/vq2a8-ascendc-v3/libvq2a8_ascendc_v3.so")
    assert not hasattr(args, "model") and not hasattr(args, "artifact")


@pytest.mark.parametrize(
    "flags",
    [
        ["--physical-npu", "-1"],
        ["--physical-npu", "0,1"],
        ["--timeout-s", "0"],
        ["--timeout-s", "-1"],
        ["--timeout-s", "nan"],
        ["--timeout-s", "inf"],
        ["--cases", ""],
        ["--cases", "basic,basic"],
        ["--cases", "basic,unknown"],
        ["--cases", "basic,"],
        ["--launch-blocking", "true"],
        ["--child", "unknown"],
        ["--plan-only", "--child", "basic"],
        ["--model", "/not-a-model"],
        ["--artifact", "/not-an-artifact"],
    ],
)
def test_startup_probe_rejects_invalid_cli(flags):
    with pytest.raises(SystemExit):
        tool.parse_args(flags)


def test_startup_probe_explicit_case_order_and_paths(tmp_path):
    args = tool.parse_args(
        [
            "--physical-npu",
            "7",
            "--cases",
            "projection_m32,basic",
            "--timeout-s",
            "23",
            "--launch-blocking",
            "0",
            "--library",
            str(tmp_path / "lib.so"),
            "--report-dir",
            str(tmp_path / "report"),
            "--allow-busy",
            "--plan-only",
        ]
    )
    assert tuple(args.cases) == ("projection_m32", "basic")
    assert args.physical_npu == 7 and args.timeout_s == 23
    assert args.launch_blocking == "0" and args.allow_busy and args.plan_only
    assert args.library == tmp_path / "lib.so"
    assert args.report_dir == tmp_path / "report"


@pytest.mark.parametrize("blocking", ["0", "1"])
def test_startup_probe_child_environment_isolated_and_nonmutating(blocking):
    stale_names = (
        "ASCEND_VISIBLE_DEVICES",
        "NPU_VISIBLE_DEVICES",
        "ASCEND_DEVICE_ID",
        "DEVICE_ID",
        "RANK_ID",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    )
    original = {
        **dict.fromkeys(stale_names, "stale-value"),
        "ASCEND_RT_VISIBLE_DEVICES": "0,2,3",
        "ASCEND_LAUNCH_BLOCKING": "inherited",
        "PYTHONPATH": "/existing/python/path",
        "PYTHONUNBUFFERED": "0",
        "LD_LIBRARY_PATH": "/existing/cann/lib",
        "PATH": "/existing/bin",
    }
    before = dict(original)
    args = tool.parse_args(["--physical-npu", "5", "--launch-blocking", blocking])
    child = tool.child_environment(args, environ=original)
    assert original == before
    assert all(name not in child for name in stale_names)
    assert child["ASCEND_RT_VISIBLE_DEVICES"] == "5"
    assert child["ASCEND_LAUNCH_BLOCKING"] == blocking
    assert child["PYTHONUNBUFFERED"] == "1"
    assert Path(child["PYTHONPATH"].split(os.pathsep)[0]) == REPO_ROOT
    assert "/existing/python/path" in child["PYTHONPATH"]
    assert child["LD_LIBRARY_PATH"] == original["LD_LIBRARY_PATH"]
    assert child["PATH"] == original["PATH"]


@pytest.mark.parametrize("case", tool.CASES)
def test_startup_probe_child_command_carries_only_explicit_probe_options(case, tmp_path):
    args = tool.parse_args(["--physical-npu", "3", "--library", str(tmp_path / "native.so"), "--launch-blocking", "0"])
    command = tool.child_command(args, case)
    assert command[:2] == [sys.executable, "-u"]
    assert Path(command[2]).resolve() == SCRIPT
    assert command[command.index("--child") + 1] == case
    assert command[command.index("--physical-npu") + 1] == "3"
    assert Path(command[command.index("--library") + 1]) == args.library
    assert command[command.index("--launch-blocking") + 1] == "0"
    assert not {"--model", "--artifact", "--tensor-parallel-size", "--plan-only"}.intersection(command)


@pytest.mark.parametrize(
    "snapshot,physical_npu,expected",
    [
        (PROCESS_HEADER + IDLE_ONE, 1, "idle"),
        (PROCESS_HEADER + BUSY_ONE, 1, "busy"),
        (PROCESS_HEADER + "| 1 | 112345 | | 123 |\n", 1, "busy"),
        (PROCESS_HEADER + IDLE_ONE + "| 0 | 887 | other | 64000 |\n", 1, "idle"),
        (PROCESS_HEADER + IDLE_ONE + "| 7 | 886 | other | 64000 |\n", 7, "busy"),
        (PROCESS_HEADER + IDLE_ONE, 0, "unknown"),
        (PROCESS_HEADER + "| No running processes found in NPU 10 |\n", 1, "unknown"),
        (PROCESS_HEADER + IDLE_ONE + BUSY_ONE, 1, "unknown"),
        (IDLE_ONE, 1, "unknown"),
        (IDLE_ONE + PROCESS_HEADER, 1, "unknown"),
        (PROCESS_HEADER, 1, "unknown"),
        ("", 1, "unknown"),
        ("npu-smi: device query failed", 1, "unknown"),
        ("| NPU ID | Name | Health |\n| 1 | Ascend950DT | OK |\n", 1, "unknown"),
    ],
)
def test_startup_probe_snapshot_is_per_card_and_fails_closed(snapshot, physical_npu, expected):
    assert tool.parse_snapshot(snapshot, physical_npu) == expected


def test_startup_probe_stage_records_submission_before_device_sync(capsys):
    observed = []

    def sync():
        observed.extend(events_in(capsys.readouterr().out))
        assert [event["event"] for event in observed] == ["BEGIN", "SUBMITTED"]

    stage = tool.stage_recorder("basic", sync)
    with stage("allocation"):
        pass
    observed.extend(events_in(capsys.readouterr().out))
    assert [event["event"] for event in observed] == ["BEGIN", "SUBMITTED", "PASS"]
    assert all(event["case"] == "basic" and event["stage"] == "allocation" for event in observed)


def test_startup_probe_sync_failure_is_never_pass(capsys):
    def sync():
        raise RuntimeError("injected device synchronization failure")

    with (
        pytest.raises(RuntimeError, match="injected device"),
        tool.stage_recorder("projection_m32", sync)("projection"),
    ):
        pass
    events = events_in(capsys.readouterr().out)
    assert [event["event"] for event in events] == ["BEGIN", "SUBMITTED", "FAIL"]
    assert all(event["case"] == "projection_m32" and event["stage"] == "projection" for event in events)


def test_startup_probe_python_failure_is_neither_submitted_nor_pass(capsys):
    synced = []
    with (
        pytest.raises(ValueError, match="injected launch"),
        tool.stage_recorder("basic", lambda: synced.append(True))("allocation"),
    ):
        raise ValueError("injected launch failure")
    assert not synced
    assert [event["event"] for event in events_in(capsys.readouterr().out)] == ["BEGIN", "FAIL"]


def test_startup_probe_reads_only_structured_events_in_order(tmp_path):
    log = tmp_path / "child.log"
    expected = [
        {"event": "BEGIN", "case": "basic", "stage": "allocation"},
        {"event": "SUBMITTED", "case": "basic", "stage": "allocation"},
    ]
    log.write_text(
        "ordinary output\n" + "\n".join("STARTUP_PROBE " + json.dumps(event) for event in expected) + "\n",
        encoding="utf-8",
    )
    assert tool.read_events(log) == expected


def test_startup_probe_plan_only_does_not_import_accelerator_or_touch_reports(tmp_path):
    report = tmp_path / "must-not-be-created"
    program = """
import builtins, runpy, subprocess, sys
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torch_npu', 'vllm', 'vllm_ascend'}:
        raise AssertionError('plan-only attempted accelerator/framework import: ' + name)
    return original_import(name, *args, **kwargs)
def no_process(*args, **kwargs):
    raise AssertionError('plan-only must not launch npu-smi or a probe child')
builtins.__import__ = checked_import
subprocess.Popen = no_process
script, report = sys.argv[1:]
sys.argv = [script, '--plan-only', '--physical-npu', '5', '--report-dir', report,
            '--library', '/nonexistent/native.so', '--cases', 'basic,projection_m32']
runpy.run_path(script, run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-u", "-c", program, str(SCRIPT), str(report)],
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "basic" in result.stdout and "projection_m32" in result.stdout
    assert not report.exists()


@pytest.mark.parametrize("case_pass", [False, True])
def test_startup_probe_watchdog_reaps_own_child_even_after_case_pass(tmp_path, case_pass):
    event = {"event": "CASE_PASS" if case_pass else "SUBMITTED", "case": "basic", "stage": "test"}
    command = [
        sys.executable,
        "-u",
        "-c",
        "import time; print(" + repr("STARTUP_PROBE " + json.dumps(event)) + ", flush=True); time.sleep(60)",
    ]
    result = tool.run_child(command, dict(os.environ), tmp_path / "timeout.log", timeout_s=1)
    assert result["status"] == "TIMEOUT"
    assert result["reaped"] is True
    assert result["exit_code"] is not None
    assert result["last_event"] == event
    assert result["elapsed_s"] < 12


@pytest.mark.parametrize(
    "events,exit_code,expected",
    [
        ([{"event": "CASE_PASS", "case": "basic"}], 0, "PASS"),
        ([], 0, "FAIL"),
        ([{"event": "PASS", "case": "basic", "stage": "allocation"}], 0, "FAIL"),
        ([{"event": "CASE_PASS", "case": "basic"}], 3, "FAIL"),
        ([{"event": "FAIL", "case": "basic", "stage": "allocation"}], 1, "FAIL"),
    ],
)
def test_startup_probe_child_pass_requires_successful_exit_and_case_sentinel(tmp_path, events, exit_code, expected):
    output = "\n".join("STARTUP_PROBE " + json.dumps(event) for event in events)
    command = [sys.executable, "-u", "-c", f"import sys; print({output!r}, flush=True); sys.exit({exit_code})"]
    result = tool.run_child(command, dict(os.environ), tmp_path / "child.log", timeout_s=10)
    assert result["status"] == expected
    assert result["reaped"] is True
    assert result["exit_code"] == exit_code


def test_startup_probe_event_reader_ignores_partial_json_and_nonrecords(tmp_path):
    log = tmp_path / "partial.log"
    log.write_text(
        'STARTUP_PROBE {"event":\n'
        "STARTUP_PROBE []\n"
        "STARTUP_PROBE null\n"
        'ordinary log STARTUP_PROBE {"event":"CASE_PASS"}\n'
        'STARTUP_PROBE {"event":"SUBMITTED","stage":"projection"}\n',
        encoding="utf-8",
    )
    assert tool.read_events(log) == [{"event": "SUBMITTED", "stage": "projection"}]


def mock_main_probes(monkeypatch, snapshot, child_status="PASS", *, reaped=True):
    calls = []
    monkeypatch.setattr(tool, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(tool, "child_environment", lambda args: {"ASCEND_RT_VISIBLE_DEVICES": str(args.physical_npu)})

    def query(command, **kwargs):
        assert command == ["npu-smi", "info"]
        assert kwargs["check"] and kwargs["timeout"] > 0
        calls.append("snapshot")
        if isinstance(snapshot, Exception):
            raise snapshot
        return SimpleNamespace(stdout=snapshot, stderr="")

    def child(command, environment, log, timeout_s):
        calls.append(command[command.index("--child") + 1])
        assert environment == {"ASCEND_RT_VISIBLE_DEVICES": "1"}
        return {
            "status": child_status,
            "exit_code": 0 if child_status == "PASS" else 1,
            "last_event": None,
            "elapsed_s": 0,
            "reaped": reaped,
            "log": str(log),
        }

    monkeypatch.setattr(tool.subprocess, "run", query)
    monkeypatch.setattr(tool, "run_child", child)
    return calls


@pytest.mark.parametrize(
    "snapshot,allow_busy,expected",
    [
        (PROCESS_HEADER + IDLE_ONE, False, "PASS"),
        (PROCESS_HEADER + BUSY_ONE, False, "BLOCKED"),
        (PROCESS_HEADER + BUSY_ONE, True, "PASS"),
        (PROCESS_HEADER, False, "BLOCKED"),
        (PROCESS_HEADER, True, "BLOCKED"),
        (OSError("npu-smi unavailable"), True, "BLOCKED"),
        (subprocess.CalledProcessError(1, ["npu-smi"]), True, "BLOCKED"),
        (subprocess.TimeoutExpired(["npu-smi"], 20), True, "BLOCKED"),
    ],
)
def test_startup_probe_main_busy_override_never_overrides_unknown(
    tmp_path, monkeypatch, snapshot, allow_busy, expected
):
    calls = mock_main_probes(monkeypatch, snapshot)
    report_dir = tmp_path / "report"
    args = ["--cases", "basic", "--report-dir", str(report_dir)] + (["--allow-busy"] if allow_busy else [])
    result = tool.main(args)
    assert result == (0 if expected == "PASS" else 1)
    assert calls == (["snapshot", "basic"] if expected == "PASS" else ["snapshot"])
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == expected
    assert summary["physical_npu"] == 1


@pytest.mark.parametrize("status,reaped", [("FAIL", True), ("TIMEOUT", True), ("TIMEOUT", False)])
def test_startup_probe_suite_stops_after_failed_or_unreaped_child(tmp_path, monkeypatch, status, reaped):
    calls = mock_main_probes(monkeypatch, PROCESS_HEADER + IDLE_ONE, status, reaped=reaped)
    report_dir = tmp_path / "report"
    assert tool.main(["--cases", "basic,hc_pre_m2", "--report-dir", str(report_dir)]) == 1
    assert calls == ["snapshot", "basic"]
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == status
    assert len(summary["results"]) == 1
    assert summary["results"][0]["reaped"] is reaped


def test_startup_probe_refuses_to_overwrite_report_directory(tmp_path, monkeypatch):
    calls = mock_main_probes(monkeypatch, PROCESS_HEADER + IDLE_ONE)
    report_dir = tmp_path / "existing-report"
    report_dir.mkdir()
    old_report = report_dir / "summary.json"
    old_report.write_text("user-owned report", encoding="utf-8")
    with pytest.raises(FileExistsError):
        tool.main(["--cases", "basic", "--report-dir", str(report_dir)])
    assert not calls
    assert old_report.read_text(encoding="utf-8") == "user-owned report"


@pytest.mark.parametrize("survives_kill", [False, True])
def test_startup_probe_posix_cleanup_signals_only_owned_session_and_is_bounded(monkeypatch, survives_kill):
    signals, waits = [], []

    class Child:
        pid = 987654
        exited = False

        def poll(self):
            return -9 if self.exited else None

        def wait(self, timeout):
            waits.append(timeout)
            if len(waits) == 1 or survives_kill:
                raise subprocess.TimeoutExpired(["own-child"], timeout)
            self.exited = True
            return -9

    monkeypatch.setattr(tool, "os", SimpleNamespace(name="posix", killpg=lambda pid, sig: signals.append((pid, sig))))
    monkeypatch.setattr(tool, "signal", SimpleNamespace(SIGTERM=15, SIGKILL=9))
    assert tool.terminate_child(Child()) is (not survives_kill)
    assert signals == [(987654, tool.signal.SIGTERM), (987654, tool.signal.SIGKILL)]
    assert waits == [5, 3]


def test_startup_probe_cleanup_does_not_signal_completed_child(monkeypatch):
    def unexpected(*args):
        pytest.fail("Completed children must not be signaled")

    monkeypatch.setattr(tool, "os", SimpleNamespace(name="posix", killpg=unexpected))
    child = SimpleNamespace(pid=987654, poll=lambda: 0, wait=unexpected)
    assert tool.terminate_child(child) is True


@pytest.mark.parametrize(
    "case,expected",
    [("projection_m1", (1, 2048, 1)), ("projection_m32", (32, 4096, 1)), ("projection_group6", (32, 4096, 6))],
)
def test_startup_projection_geometry_includes_first_prefill_kernel(case, expected):
    assert projection_probe._case_geometry(case) == expected


def test_startup_projection_geometry_rejects_nonprojection_case():
    with pytest.raises(ValueError, match="Unknown"):
        projection_probe._case_geometry("basic")


def test_startup_projection_missing_manifest_is_unknown_not_compatibility_pass(tmp_path):
    library = tmp_path / "probe.so"
    library.write_bytes(b"synthetic-library-identity-only")
    identity = projection_probe._library_identity(library, "Ascend950DT_9582")
    assert identity["sha256"] == hashlib.sha256(library.read_bytes()).hexdigest()
    assert identity["runtime_soc"] == "Ascend950DT_9582"
    assert identity["manifest_present"] is False
    assert identity["build_soc"] is None
    assert identity["soc_check"] == identity["manifest_hash_check"] == "UNKNOWN"
    assert identity["compatibility_verified"] is False


@pytest.mark.parametrize(
    "build_soc,hash_matches,expected_soc,expected_hash",
    [
        ("Ascend950DT_9582", True, "MATCH", "MATCH"),
        ("Ascend950DT_9574", True, "MISMATCH", "MATCH"),
        ("Ascend950DT_9582", False, "MATCH", "MISMATCH"),
        (None, None, "UNKNOWN", "UNKNOWN"),
    ],
)
def test_startup_projection_manifest_identity_reports_mismatch_without_claiming_execution(
    tmp_path, build_soc, hash_matches, expected_soc, expected_hash
):
    library = tmp_path / "probe.so"
    library.write_bytes(b"synthetic-library-identity-only")
    manifest = {}
    if build_soc is not None:
        manifest["soc"] = build_soc
    if hash_matches is not None:
        manifest["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest() if hash_matches else "0" * 64
    (tmp_path / "build-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    identity = projection_probe._library_identity(library, "Ascend950DT_9582")
    assert identity["soc_check"] == expected_soc
    assert identity["manifest_hash_check"] == expected_hash
    assert identity["manifest_present"] is True
    assert identity["compatibility_verified"] is False


@pytest.mark.parametrize("manifest", [[], "bad", {"soc": 123}, {"soc": ""}])
def test_startup_projection_rejects_malformed_build_manifest(tmp_path, manifest):
    library = tmp_path / "probe.so"
    library.write_bytes(b"synthetic-library-identity-only")
    (tmp_path / "build-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        projection_probe._library_identity(library, "Ascend950DT_9582")


def test_startup_projection_requires_an_existing_shared_library(tmp_path):
    with pytest.raises(FileNotFoundError):
        projection_probe._library_identity(tmp_path / "absent.so", "Ascend950DT_9582")
    wrong_suffix = tmp_path / "not-a-shared-library.txt"
    wrong_suffix.write_text("not-a-library", encoding="utf-8")
    with pytest.raises(ValueError, match=".so"):
        projection_probe._library_identity(wrong_suffix, "Ascend950DT_9582")


@pytest.mark.parametrize("mismatch", ["soc", "hash"])
def test_startup_projection_identity_mismatch_rejects_before_loading_or_launching(tmp_path, monkeypatch, mismatch):
    import torch

    def forbidden(*args, **kwargs):
        pytest.fail("Mismatched build must not load a native library or construct projection tensors")

    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda index: "Ascend950DT_9582"), raising=False)
    monkeypatch.setitem(
        sys.modules,
        "tools.validate_vq2a8_ascendc_v2",
        SimpleNamespace(_convert_inputs=forbidden, _synthetic=forbidden),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.validate_vq2a8_phase4_kernel",
        SimpleNamespace(bitwise_equal=forbidden, same_fp8_oracle=forbidden, synthetic_dense_oracle=forbidden),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.quantization.vq2a8_ascendc_v3",
        SimpleNamespace(grouped_projection_resident=forbidden, load_pinned_library=forbidden),
    )
    library = tmp_path / "probe.so"
    library.write_bytes(b"synthetic-library-identity-only")
    manifest = {
        "soc": "Ascend950DT_9574" if mismatch == "soc" else "Ascend950DT_9582",
        "library_sha256": "0" * 64 if mismatch == "hash" else hashlib.sha256(library.read_bytes()).hexdigest(),
    }
    (tmp_path / "build-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    names, stage = cpu_stage_names()
    with pytest.raises(ValueError, match="built for" if mismatch == "soc" else "SHA256"):
        projection_probe.run_case("projection_m32", stage, library)
    assert names == ["projection_imports", "projection_library"]


def cpu_stage_names():
    names = []

    @contextmanager
    def stage(name):
        names.append(name)
        yield

    return names, stage


def test_startup_basic_probe_cpu_arithmetic_contract():
    import torch

    names, stage = cpu_stage_names()
    metrics = ops_probe._basic(torch, torch.device("cpu"), stage)
    assert names == ["basic.setup_cpu", "basic.upload", "basic.add", "basic.matmul", "basic.download", "basic.verify"]
    assert metrics["verification"] == "shape_dtype_finite_and_constant_result"


@pytest.mark.parametrize("jobs", [1, 6])
def test_startup_preparation_cpu_runs_real_rowwise_code_at_both_widths(monkeypatch, jobs):
    import torch

    # Load just the three pure tensor/metadata files. Do not initialize the
    # vLLM plugin merely to exercise a CPU arithmetic contract.
    for leaf in ("vq2a8_artifact", "vq2a8_reference", "vq2a8_activation"):
        name = "vllm_ascend.quantization." + leaf
        spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "vllm_ascend/quantization" / f"{leaf}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    names, stage = cpu_stage_names()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        metrics = ops_probe._prepare(torch, torch.device("cpu"), jobs, stage)
    finally:
        torch.set_num_threads(previous_threads)
    assert metrics["geometry"] == [
        {"width": 4096, "jobs": jobs, "rows_per_job": 32},
        {"width": 2048, "jobs": jobs, "rows_per_job": 32},
    ]
    assert metrics["assignment_rows_per_width"] == jobs * 32
    assert "prepare.k4096.call_many" in names and "prepare.k2048.call_many" in names
    assert "prepare.k4096.verify" in names and "prepare.k2048.verify" in names


def test_startup_helpers_reject_unknown_case_before_import_or_device_init():
    with pytest.raises(ValueError, match="Unknown"):
        ops_probe.run_case("not-a-probe", lambda name: pytest.fail("must reject before entering a stage"))
    with pytest.raises(ValueError, match="Unknown"):
        projection_probe.run_case("not-a-probe", lambda name: pytest.fail("must reject before entering a stage"), None)
