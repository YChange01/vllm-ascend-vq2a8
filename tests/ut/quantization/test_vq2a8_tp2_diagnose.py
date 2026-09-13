# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only diagnostic guard tests; these do not certify NPU/HCCL execution."""

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "tools" / "vq2_tp2_diagnose.sh"
PROCESS_HEADER = "| NPU ID | Process id | Process name | Process memory(MB) |\n"
IDLE_ZERO = "| No running processes found in NPU 0 |\n"
IDLE_ONE = "| No running processes found in NPU 1 |\n"
IDLE_TEN = "| No running processes found in NPU 10 |\n"
BUSY_ZERO = "| 0 | 999 | python | 2 |\n"
BUSY_ONE = "| 1 | 998 | python | 2 |\n"
BUSY_OTHER = "| 4 | 997 | python | 2 |\n"
BUSY_EXIT = 10


@pytest.fixture
def script_source():
    return SCRIPT_PATH.read_text(encoding="utf-8")


def embedded_python(script_source, section):
    if section == "idle":
        return script_source.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    return script_source.split("--no-python python -u -c '\n", 1)[1].split("\n'\n", 1)[0]


@pytest.mark.parametrize(
    "process_log,expected_exit",
    [
        pytest.param(PROCESS_HEADER + IDLE_ZERO + IDLE_ONE, 0, id="both-idle"),
        pytest.param(PROCESS_HEADER + BUSY_ZERO + IDLE_ONE, BUSY_EXIT, id="zero-busy"),
        pytest.param(PROCESS_HEADER + IDLE_ZERO + BUSY_ONE, BUSY_EXIT, id="one-busy"),
        pytest.param(PROCESS_HEADER + BUSY_ZERO + BUSY_ONE, BUSY_EXIT, id="both-busy"),
        pytest.param(PROCESS_HEADER + IDLE_ZERO, 2, id="one-missing"),
        pytest.param(PROCESS_HEADER + IDLE_ONE, 2, id="zero-missing"),
        pytest.param(PROCESS_HEADER, 2, id="both-missing"),
        pytest.param(IDLE_ZERO + IDLE_ONE, 2, id="header-missing"),
        pytest.param("", 2, id="empty-output"),
        pytest.param(PROCESS_HEADER + IDLE_ZERO + IDLE_ONE + BUSY_OTHER, 0, id="other-card-busy"),
        pytest.param(PROCESS_HEADER + IDLE_ZERO + IDLE_TEN, 2, id="ten-not-one"),
        pytest.param(PROCESS_HEADER + IDLE_ONE + IDLE_TEN, 2, id="ten-not-zero"),
        pytest.param(PROCESS_HEADER + IDLE_ZERO + IDLE_ONE + BUSY_ZERO, 2, id="contradictory-zero"),
        pytest.param(PROCESS_HEADER + IDLE_ZERO + IDLE_ONE + BUSY_ONE, 2, id="contradictory-one"),
        pytest.param(IDLE_ZERO + IDLE_ONE + PROCESS_HEADER + BUSY_ZERO, 2, id="ignore-pre-header-idle"),
        pytest.param(PROCESS_HEADER + BUSY_ZERO, 2, id="busy-zero-one-unknown"),
        pytest.param(PROCESS_HEADER + BUSY_ONE, 2, id="busy-one-zero-unknown"),
    ],
)
def test_tp2_diagnose_idle_guard_fails_closed(script_source, process_log, expected_exit, monkeypatch, capsys):
    code = compile(embedded_python(script_source, "idle"), str(SCRIPT_PATH) + ":idle", "exec")
    monkeypatch.setattr(Path, "read_text", lambda self, **kwargs: process_log)
    monkeypatch.setattr(sys, "argv", ["idle", "device-snapshot.log"])
    with pytest.raises(SystemExit) as raised:
        exec(code, {})
    assert raised.value.code == expected_exit
    output = capsys.readouterr().out
    if expected_exit == 0:
        assert "0,1 clear; not an exclusive reservation" in output
    elif expected_exit == BUSY_EXIT:
        assert "occupied physical NPUs=" in output
    else:
        assert "skip test" in output or "cannot recognize NPU process table" in output


@pytest.mark.parametrize("section", ["idle", "raw"])
def test_tp2_diagnose_embedded_python_compiles_without_importing_npu(script_source, section):
    compile(embedded_python(script_source, section), str(SCRIPT_PATH) + ":" + section, "exec")


def test_tp2_diagnose_shell_uses_lf_line_endings():
    assert b"\r" not in SCRIPT_PATH.read_bytes()


def test_tp2_diagnose_unreadable_snapshot_is_unknown(script_source, monkeypatch, capsys):
    code = compile(embedded_python(script_source, "idle"), str(SCRIPT_PATH) + ":idle", "exec")

    def unreadable(self, **kwargs):
        raise OSError("mock snapshot unreadable")

    monkeypatch.setattr(Path, "read_text", unreadable)
    monkeypatch.setattr(sys, "argv", ["idle", "device-snapshot.log"])
    with pytest.raises(SystemExit) as raised:
        exec(code, {})
    assert raised.value.code == 2
    assert "cannot read NPU snapshot" in capsys.readouterr().out


@pytest.fixture
def bash_path():
    windows_bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(windows_bash) if windows_bash.is_file() else shutil.which("bash")
    if not executable:
        pytest.skip("Bash is required for shell control-flow tests")
    return executable


def run_bash(bash_path, source, *args, **environment):
    env = dict(os.environ)
    env.pop("BASH_ENV", None)
    env.update({key: str(value) for key, value in environment.items()})
    return subprocess.run(
        [bash_path, "--noprofile", "--norc", "-s", "--", *args],
        input=source,
        text=True,
        capture_output=True,
        env=env,
        timeout=15,
        check=False,
    )


def shell_functions(script_source):
    """Keep actual function bodies, without running the diagnostic entry point."""
    return "cleanup() {" + script_source.split("cleanup() {", 1)[1].split("\n{\ntrap cleanup", 1)[0]


@pytest.mark.parametrize(
    "idle_exit,allow_busy,should_run,isolation",
    [
        (0, "0", True, "IDLE_SNAPSHOT"),
        (0, "1", True, "IDLE_SNAPSHOT"),
        (BUSY_EXIT, "0", False, None),
        (BUSY_EXIT, "1", True, "SHARED_DEVICE"),
        (2, "0", False, None),
        (2, "1", False, None),
        (1, "1", False, None),
        (127, "1", False, None),
    ],
)
def test_tp2_diagnose_busy_override_only_allows_known_busy(
    bash_path, script_source, tmp_path, idle_exit, allow_busy, should_run, isolation
):
    events = tmp_path / "events.log"
    source = (
        shell_functions(script_source)
        + r"""
idle() { return "$VQ2_TEST_IDLE_EXIT"; }
timeout() {
    printf 'ARG=%s\n' "$@" >> "$VQ2_TEST_EVENT_LOG"
    printf 'RAW_PASS=allreduce rank=0\n'
}
sleep() { command sleep 0.01; }
run_test mock_hccl mock_test --communication-only
"""
    )
    result = run_bash(
        bash_path,
        source,
        VQ2_DIAG_DIR=tmp_path.as_posix(),
        VQ2_TEST_EVENT_LOG=events.as_posix(),
        VQ2_TEST_IDLE_EXIT=idle_exit,
        VQ2_ALLOW_BUSY=allow_busy,
    )
    assert not result.stderr, result.stderr
    if not should_run:
        assert "mock_hccl=SKIPPED_BUSY_OR_UNKNOWN" in result.stdout
        assert not events.exists()
        assert not (tmp_path / "mock_hccl.log").exists()
        return
    assert "mock_hccl EXIT=0" in result.stdout
    assert f"TEST_ISOLATION={isolation}" in result.stdout
    assert "TIMING_VALID=False" in result.stdout
    if isolation == "SHARED_DEVICE":
        assert "WARN" in result.stdout
    assert events.read_text(encoding="utf-8").splitlines() == [
        "ARG=-k",
        "ARG=10s",
        "ARG=180s",
        "ARG=env",
        "ARG=ASCEND_RT_VISIBLE_DEVICES=0,1",
        "ARG=ASCEND_LAUNCH_BLOCKING=1",
        "ARG=ASCEND_SLOG_PRINT_TO_STDOUT=1",
        "ARG=ASCEND_GLOBAL_LOG_LEVEL=1",
        "ARG=mock_test",
        "ARG=--communication-only",
    ]


@pytest.mark.parametrize("query_exit", [1, 124, 137])
def test_tp2_diagnose_smi_failure_is_unknown_not_busy(bash_path, script_source, tmp_path, query_exit):
    source = (
        shell_functions(script_source)
        + r"""
timeout() { return "$VQ2_TEST_QUERY_EXIT"; }
python() { printf 'UNEXPECTED_PYTHON\n'; return 99; }
idle query_failure
exit $?
"""
    )
    result = run_bash(
        bash_path,
        source,
        VQ2_DIAG_DIR=tmp_path.as_posix(),
        VQ2_TEST_QUERY_EXIT=query_exit,
        VQ2_ALLOW_BUSY="1",
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "UNEXPECTED_PYTHON" not in result.stdout


@pytest.mark.parametrize(
    "process_log,expected_exit",
    [
        (PROCESS_HEADER + IDLE_ZERO + IDLE_ONE, 0),
        (PROCESS_HEADER + BUSY_ZERO + IDLE_ONE, BUSY_EXIT),
        (PROCESS_HEADER + BUSY_ZERO, 2),
    ],
)
def test_tp2_diagnose_shell_idle_propagates_parser_status(
    bash_path, script_source, tmp_path, process_log, expected_exit
):
    python_path = shlex.quote(Path(sys.executable).as_posix())
    source = (
        shell_functions(script_source)
        + f'\npython() {{ {python_path} "$@"; }}\n'
        + r"""
timeout() { shift 3; "$@"; }
npu-smi() { printf '%s' "$VQ2_TEST_PROCESS_LOG"; }
idle parser_status
exit $?
"""
    )
    result = run_bash(
        bash_path,
        source,
        VQ2_DIAG_DIR=tmp_path.as_posix(),
        VQ2_TEST_PROCESS_LOG=process_log,
    )
    assert result.returncode == expected_exit, result.stdout + result.stderr


@pytest.mark.parametrize(
    "args,expected_exit",
    [
        (["--help"], 0),
        (["-h"], 0),
        (["--allow-busy", "--help"], 0),
        (["--unknown"], 2),
        (["--allow-busy", "--unknown"], 2),
    ],
)
def test_tp2_diagnose_cli_help_and_errors_before_probes(bash_path, script_source, args, expected_exit):
    # Even a regression cannot probe real hardware or create a report in this test.
    mocks = r"""
mktemp() { printf 'UNEXPECTED_SIDE_EFFECT=mktemp\n' >&2; return 99; }
timeout() { printf 'UNEXPECTED_SIDE_EFFECT=timeout\n' >&2; return 99; }
python() { printf 'UNEXPECTED_SIDE_EFFECT=python\n' >&2; return 99; }
npu-smi() { printf 'UNEXPECTED_SIDE_EFFECT=npu-smi\n' >&2; return 99; }
uname() { printf 'UNEXPECTED_SIDE_EFFECT=uname\n' >&2; return 99; }
tar() { printf 'UNEXPECTED_SIDE_EFFECT=tar\n' >&2; return 99; }
"""
    result = run_bash(bash_path, mocks + script_source, *args)
    output = result.stdout + result.stderr
    assert result.returncode == expected_exit, output
    assert "UNEXPECTED_SIDE_EFFECT" not in output
    assert "REPORT=" not in output
    if expected_exit == 0:
        assert "--allow-busy" in output


@pytest.mark.parametrize("args,expected", [([], "0"), (["--allow-busy"], "1")])
def test_tp2_diagnose_override_requires_cli_not_inherited_env(bash_path, script_source, args, expected):
    prolog = script_source.split("\ncd --", 1)[0]
    source = prolog + '\nprintf "ALLOW_BUSY=%s\\n" "$VQ2_ALLOW_BUSY"\n'
    result = run_bash(bash_path, source, *args, VQ2_ALLOW_BUSY="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == f"ALLOW_BUSY={expected}\n"


@pytest.mark.parametrize("own_pid", ["", "12345"])
def test_tp2_diagnose_cleanup_only_signals_own_test(bash_path, script_source, own_pid):
    source = (
        shell_functions(script_source)
        + r"""
kill() { printf 'MOCK_KILL=%s\n' "$*"; }
wait() { printf 'MOCK_WAIT=%s\n' "$*"; }
cleanup
printf 'REMAINING_PID=%s\n' "$VQ2_TEST_PID"
"""
    )
    result = run_bash(bash_path, source, VQ2_TEST_PID=own_pid)
    assert result.returncode == 0, result.stdout + result.stderr
    if own_pid:
        assert "MOCK_KILL=-TERM 12345" in result.stdout
        assert "MOCK_WAIT=12345" in result.stdout
    else:
        assert "MOCK_KILL" not in result.stdout
        assert "MOCK_WAIT" not in result.stdout
    assert "REMAINING_PID=\n" in result.stdout


def test_tp2_diagnose_remains_bounded_communication_only(script_source):
    assert "ASCEND_RT_VISIBLE_DEVICES=0,1" in script_source
    assert "timeout -k 10s 180s" in script_source
    assert "tools/validate_vq2a8_tp2_collective.py --communication-only --timeout-s 120" in script_source
    raw = embedded_python(script_source, "raw")
    assert "torch.full((1, 8)" in raw
    assert "timedelta(seconds=120)" in raw
    assert "from vllm" not in raw
    assert "--model" not in script_source
    assert not re.search(r"\b(?:pkill|killall)\b|npu-smi\s+set|docker\s+(?:stop|restart|kill)", script_source)
