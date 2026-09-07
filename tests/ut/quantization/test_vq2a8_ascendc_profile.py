# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import csv
import inspect
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import profile_vq2a8_ascendc as profile


@pytest.mark.parametrize("level", [2, 3])
def test_private_config_preserves_global_and_other_settings(tmp_path, level):
    source = tmp_path / "global.json"
    value = {"log": {"flush_level": level, "other": 42}, "nested": [{"flush_level": 3}], "trace": True}
    source.write_text(json.dumps(value))
    before = source.read_bytes()
    evidence = profile.prepare_config(source, tmp_path / "private")
    assert source.read_bytes() == before
    expected = {**value, "log": {"flush_level": 2, "other": 42}, "nested": [{"flush_level": 2}]}
    assert json.loads(Path(evidence["private"]).read_text()) == expected
    assert evidence["source_sha256"] == profile.digest(source)
    assert not Path(evidence["private"]).is_symlink()


@pytest.mark.parametrize("value", [{}, {"flush_level": True}, {"flush_level": "3"}, {"flush_level": 4}])
def test_private_config_rejects_unknown_contract(tmp_path, value):
    source = tmp_path / "config.json"
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        profile.prepare_config(source, tmp_path / "private")
    assert not (tmp_path / "private").exists()


def test_config_never_reuses_an_existing_private_directory(tmp_path):
    source = tmp_path / "config.json"
    source.write_text('{"flush_level": 3}')
    with pytest.raises(FileExistsError):
        profile.prepare_config(source, tmp_path)
    assert source.read_text() == '{"flush_level": 3}'


@pytest.mark.parametrize(
    "status,digest,rows",
    [
        ("failed", "abc", [{"status": "passed"}]),
        ("completed_review_pending", "wrong", [{"status": "passed"}]),
        ("completed_review_pending", "abc", []),
        ("completed_review_pending", "abc", [{"status": "failed"}]),
    ],
)
def test_suite_gate_requires_same_completed_library(tmp_path, status, digest, rows):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"status": status, "library": {"sha256": digest}, "results": rows}))
    with pytest.raises(ValueError):
        profile.checked_suite(path, {"sha256": "abc"})


def test_suite_gate_preserves_receipt(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {"status": "completed_review_pending", "library": {"sha256": "abc"}, "results": [{"status": "passed"}]}
        )
    )
    before = path.read_bytes()
    assert profile.checked_suite(path, {"sha256": "abc"})["library_sha256"] == "abc"
    assert path.read_bytes() == before


def test_command_selects_one_fused_and_keeps_sync_default(tmp_path):
    args = SimpleNamespace(soc="Ascend950PR_957d", timeout_minutes=5, library=tmp_path / "lib.so")
    command = profile.profiler_command("/bin/msprof", args, tmp_path, "abc")
    assert command[:3] == ["/bin/msprof", "op", "simulator"]
    assert "--launch-count=1" in command and "--timeout=5" in command
    assert "--kernel-name=vq2a8_ascendc_fused" in command
    assert not any(s.startswith("--aic-metrics") for s in command)  # do not disable sync tracing
    assert not any("validate_vq2a8_ascendc" in s for s in command)


@pytest.mark.parametrize(
    "maps,expected",
    [
        ("1-2 r-x 0 00:00 1 /cann/libruntime.so\n", []),
        ("1-2 r-x 0 00:00 1 /cann/libruntime_camodel.so\n", ["/cann/libruntime_camodel.so"]),
    ],
)
def test_runtime_guard_requires_simulator_mapping(maps, expected):
    assert profile.simulator_runtime_paths(maps) == expected


def test_bare_application_is_rejected_before_torch_import(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: "hardware runtime only")
    with pytest.raises(RuntimeError, match="No mapped"):
        profile.run_application(SimpleNamespace(application_report=tmp_path / "application.json"))


@pytest.mark.parametrize("mode", ["missing_env", "outside_report", "wrong_level", "missing_level", "changed_library"])
def test_application_preflight_fails_closed_before_npu(mode, tmp_path, monkeypatch):
    report = tmp_path / "report"
    report.mkdir()
    config_dir = (tmp_path if mode == "outside_report" else report) / "config"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps({} if mode == "missing_level" else {"flush_level": 3 if mode == "wrong_level" else 2})
    )
    read_text = Path.read_text

    def read(path, *args, **kwargs):
        if str(path) == "/proc/self/maps":
            return "1-2 r-x 0 00:00 1 /sim/lib/libruntime_camodel.so\n"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(profile, "library_evidence", lambda p: {"sha256": "changed"})
    if mode == "missing_env":
        monkeypatch.delenv("CAMODEL_CONFIG_PATH", raising=False)
    else:
        monkeypatch.setenv("CAMODEL_CONFIG_PATH", str(config_dir))
    args = SimpleNamespace(
        application_report=report / "application.json", library=tmp_path / "lib.so", expected_library_sha256="original"
    )
    with pytest.raises((RuntimeError, ValueError)):
        profile.run_application(args)
    assert not args.application_report.exists()


def test_application_has_one_projection_and_no_numerical_campaign():
    tree = ast.parse(inspect.getsource(profile.run_application))
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert calls.count("vq2a8_ascendc") == 1
    assert not {"check_projection", "cube_control", "benchmark", "accepted_rows", "compare"} & set(calls)
    assert "output.cpu" not in calls


def make_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["instr", "addr", "pipe", "call_count", "detail"])
        writer.writeheader()
        writer.writerows(rows)


def test_instruction_summary_is_bounded_and_preserves_operands(tmp_path):
    path = tmp_path / "core0.cubecore0/core0.cubecore0_instr_exe.csv"
    rows = [
        {"instr": "mock_mmad", "addr": str(i), "pipe": "CUBE", "call_count": 4, "detail": f"synthetic_operand={i}"}
        for i in range(100)
    ]
    make_csv(path, rows)
    evidence = profile.collect_instruction_csv(tmp_path)[0]
    assert evidence["status"] == "instruction_rows_collected" and evidence["rows"] == 100
    assert evidence["histogram"][0]["rows"] == 100  # not 400 call_count
    assert len(evidence["samples"]) == 1  # repeated PC variants cannot crowd out later transfers
    assert evidence["samples"][0]["detail"] == "synthetic_operand=0"
    assert "native_instruction_verified" not in evidence


def test_csv_samples_keep_sync_and_later_transfer_kinds(tmp_path):
    path = tmp_path / "core0.veccore0_instr_exe.csv"
    rows = [{"instr": "mock_load", "pipe": "MTE2", "detail": str(i)} for i in range(20)]
    rows += [
        {"instr": "mock_copy_ubuf_l1", "pipe": "MTE3", "detail": "UB to L1"},
        {"instr": "mock_copy_ubuf_gm", "pipe": "MTE3", "detail": "output only"},
        {"instr": "mock_set_flag", "pipe": "MTE3", "detail": "handoff"},
    ]
    make_csv(path, rows)
    samples = profile.collect_instruction_csv(tmp_path)[0]["samples"]
    assert len(samples) == 4
    assert samples[-1]["kind"] == "sync"
    assert samples[1]["detail"] == "UB to L1"


def test_scalar_madd_is_not_cube_matrix_evidence(tmp_path):
    path = tmp_path / "core0.cubecore0_instr_exe.csv"
    make_csv(
        path,
        [
            {"instr": "MADD", "pipe": "SCALAR", "detail": "dtype:S64"},
            {"instr": "MMAD", "pipe": "CUBE", "detail": "dtype:E4M3E4M3"},
            {"instr": "SET_FLAG", "pipe": "CUBE", "detail": "PIPE:CUBE,TRIGGERPIPE:MTE1,FLAGID:0"},
        ],
    )
    samples = profile.collect_instruction_csv(tmp_path)[0]["samples"]
    assert [row["kind"] for row in samples] == ["other", "matrix", "sync"]
    assert samples[1]["detail"] == "dtype:E4M3E4M3"


def test_profiler_log_keeps_bounded_errors_despite_successful_parsing(tmp_path):
    path = tmp_path / "profiler.log"
    path.write_text(
        "[ERROR] pem_ccu.cc:2270 execute_set_flag already has same set_flag! pc:0x10d0f488.\n" * 20
        + "[INFO] The timeout has reached and the application will be forcibly killed.\n"
        + "[INFO] Profiling running finished. All task success.\n"
    )
    result = profile.inspect_profiler_log(path)
    assert result["application_timeout_reported"] is True
    assert result["runtime_error_count"] == 20
    assert len(result["errors"]) == 12
    assert result["errors"][0]["line"] == 1
    assert "0x10d0f488" in result["errors"][0]["text"]


@pytest.mark.parametrize("marker", ["[error]", "[ERROR]", "[Error]"])
def test_shutdown_errors_are_counted_without_claiming_their_cause(tmp_path, marker):
    path = tmp_path / "profiler.log"
    path.write_text(
        marker
        + " an earlier error\n"
        + "Model SignalHandler: SigIntHandler received signal: 2\n"
        + marker
        + " [core0.veccore0] [su_ccu_illegal_instr_t0] ZEROEXT\n"
        + "[INFO] The timeout has reached and the application will be forcibly killed.\n"
    )
    review = profile.inspect_profiler_log(path)
    assert review["runtime_error_count"] == 2
    assert review["application_timeout_reported"] is True
    assert review["shutdown_notice_seen"] is True
    assert [row["after_shutdown_notice"] for row in review["errors"]] == [False, True]


def test_instruction_progress_reads_only_bounded_instruction_tails(tmp_path):
    dump = tmp_path / "profile/OPPROF_mock/dump"
    dump.mkdir(parents=True)
    for core in ("cubecore0", "veccore0", "veccore1"):
        (dump / f"core0.{core}.instr_log.dump").write_text("old\n" * 2000 + "PC: last1\nPC: last2\n")
    (dump / "core0.cubecore0.ccu_log.dump").write_text("not instruction progress\n")
    rows = profile.instruction_progress(tmp_path)
    assert len(rows) == 3
    assert all(row["bytes"] > profile.PROGRESS_TAIL_BYTES for row in rows)
    assert all(row["tail"] == ["PC: last1", "PC: last2"] for row in rows)


def test_instruction_progress_handles_missing_and_empty_logs(tmp_path):
    assert profile.instruction_progress(tmp_path) == []
    dump = tmp_path / "profile/OPPROF_mock/dump"
    dump.mkdir(parents=True)
    (dump / "core0.cubecore0.instr_log.dump").touch()
    assert profile.instruction_progress(tmp_path) == [{"core": "cubecore0", "bytes": 0, "tail": []}]


@pytest.mark.parametrize("minutes", [0, 5, 30, 45, 60, 61])
def test_explicit_simulator_deadline_is_bounded(monkeypatch, minutes):
    monkeypatch.setattr(profile.sys, "argv", ["profile", "--diagnostic-build", "--timeout-minutes", str(minutes)])
    received = []
    monkeypatch.setattr(profile, "run", lambda args: received.append(args.timeout_minutes) or 0)
    if 1 <= minutes <= 60:
        assert profile.main() == 0
        assert received == [minutes]
    else:
        with pytest.raises(SystemExit):
            profile.main()
        assert received == []


@pytest.mark.parametrize("mode", ["empty", "unknown_schema", "large"])
def test_csv_missing_or_unreadable_instruction_contract(tmp_path, mode, monkeypatch):
    path = tmp_path / "core0_instr_exe.csv"
    make_csv(path, [])
    if mode == "unknown_schema":
        path.write_text("a,b\n1,2\n")
    if mode == "large":
        monkeypatch.setattr(profile, "MAX_CSV_BYTES", 1)
    assert (
        profile.collect_instruction_csv(tmp_path)[0]["status"]
        == {"empty": "empty", "unknown_schema": "unrecognized_columns", "large": "too_large_for_summary"}[mode]
    )


@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "diagnostic",
        "profiler_failure",
        "missing_app",
        "wrong_library",
        "missing_vector",
        "inner_timeout",
        "runtime_error",
        "shutdown_error",
    ],
)
def test_profiler_supervision_preserves_gate_flags(tmp_path, monkeypatch, mode):
    cann = tmp_path / "cann"
    config = cann / "tools/simulator/Ascend950PR_957d/lib/config.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"flush_level": 3}')
    suite = tmp_path / "suite"
    suite.mkdir()
    library = tmp_path / "lib.so"
    library.write_bytes(b"library")
    lib_sha = profile.digest(library)
    (suite / "summary.json").write_text(
        json.dumps(
            {"status": "completed_review_pending", "library": {"sha256": lib_sha}, "results": [{"status": "passed"}]}
        )
    )
    directory = tmp_path / "report"
    directory.mkdir()
    monkeypatch.setattr(profile.sys, "platform", "linux")
    monkeypatch.setattr(profile.tempfile, "mkdtemp", lambda **kw: str(directory))
    monkeypatch.setattr(profile.shutil, "which", lambda name: "/bin/msprof")
    monkeypatch.setattr(profile, "library_evidence", lambda p: {"path": str(p), "sha256": lib_sha})
    monkeypatch.setattr(
        profile.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="v1", stderr="")
    )

    def collect(command, output, env, timeout_seconds):
        assert json.loads((Path(env["CAMODEL_CONFIG_PATH"]) / "config.json").read_text())["flush_level"] == 2
        assert config.read_text() == '{"flush_level": 3}'
        assert timeout_seconds == 420
        (output / "profiler.log").write_text(
            "[INFO] The timeout has reached and the application will be forcibly killed.\n"
            if mode == "inner_timeout"
            else "[ERROR] pem_ccu.cc:2270 execute_set_flag already has same set_flag! pc:0x10d0f488.\n"
            if mode == "runtime_error"
            else "Model SignalHandler: SigIntHandler received signal: 2\n[error] illegal_instr\n"
            if mode == "shutdown_error"
            else "[INFO] Profiling running finished. All task success.\n"
        )
        if mode != "missing_app":
            profile.write_json(
                output / "application.json",
                {
                    "status": "completed",
                    "projection_calls": 1,
                    "library_sha256": "wrong" if mode == "wrong_library" else lib_sha,
                    "library_unchanged": True,
                },
            )
        for core in ("cubecore0", "veccore0", "veccore1"):
            if mode == "missing_vector" and core == "veccore1":
                continue
            make_csv(
                output / f"profile/core0.{core}/core0.{core}_instr_exe.csv",
                [{"instr": "mock", "pipe": "CUBE", "detail": "mock detail"}],
            )
        return {"exit": 1 if mode == "profiler_failure" else 0, "timeout": False}

    monkeypatch.setattr(profile, "run_profiler", collect)
    code = profile.run(
        SimpleNamespace(
            library=library,
            suite_report=None if mode == "diagnostic" else suite,
            diagnostic_build=mode == "diagnostic",
            cann=cann,
            soc="Ascend950PR_957d",
            timeout_minutes=5,
        )
    )
    report = json.loads((directory / "summary.json").read_text())
    assert (code == 0) is (mode in ("success", "diagnostic"))
    assert report["diagnostic_build"] is (mode == "diagnostic")
    assert report["matching_standalone_suite_supplied"] is (mode != "diagnostic")
    assert (report["suite"] is None) is (mode == "diagnostic")
    assert report["global_config_unchanged"] is True
    for key in (
        "native_instruction_verified",
        "on_chip_decode_verified",
        "performance_verified",
        "model_integration_verified",
    ):
        assert report[key] is False
    assert report["default_model_backend"] == "unchanged"


def test_profiler_timeout_kills_only_owned_process_group(tmp_path, monkeypatch):
    class Child:
        pid = 123
        returncode = -9

        def wait(self, timeout=None):
            if timeout:
                raise subprocess.TimeoutExpired("msprof", timeout)

    killed = []

    def spawn(command, **kwargs):
        assert kwargs["start_new_session"] is True
        return Child()

    monkeypatch.setattr(profile.subprocess, "Popen", spawn)
    monkeypatch.setattr(profile.os, "killpg", lambda pid, sig: killed.append(pid))
    ticks = iter([0, 0, 0.5, 1])
    monkeypatch.setattr(profile, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    result = profile.run_profiler(["msprof"], tmp_path, {}, 1)
    assert result == {"exit": -9, "timeout": True}
    assert killed == [123]


def test_progress_poll_does_not_reset_deadline_or_kill_running_child(tmp_path, monkeypatch, capsys):
    timeouts = []

    class Child:
        def wait(self, timeout):
            timeouts.append(timeout)
            if len(timeouts) == 1:
                raise subprocess.TimeoutExpired("msprof", timeout)
            return 0

    ticks = iter([0, 0, 30, 30])
    monkeypatch.setattr(profile, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    monkeypatch.setattr(profile.subprocess, "Popen", lambda *a, **kw: Child())
    monkeypatch.setattr(profile.os, "killpg", lambda *a: pytest.fail("Progress poll is not the overall deadline"))
    result = profile.run_profiler(["msprof"], tmp_path, {}, 45)
    assert result == {"exit": 0, "timeout": False}
    assert timeouts == [30, 15]
    assert "ASCENDC_SIM_PROGRESS" in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["console", "dump"])
def test_progress_io_failure_does_not_abandon_owned_child(tmp_path, monkeypatch, failure):
    waits = []

    class Child:
        def wait(self, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                raise subprocess.TimeoutExpired("msprof", timeout)
            return 0

    def unavailable(*args, **kwargs):
        raise OSError("Progress channel unavailable")

    ticks = iter([0, 0, 30, 30])
    monkeypatch.setattr(profile, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    monkeypatch.setattr(profile.subprocess, "Popen", lambda *a, **kw: Child())
    monkeypatch.setattr(
        profile, "print" if failure == "console" else "instruction_progress", unavailable, raising=False
    )
    assert profile.run_profiler(["msprof"], tmp_path, {}, 45) == {"exit": 0, "timeout": False}
    assert len(waits) == 2
