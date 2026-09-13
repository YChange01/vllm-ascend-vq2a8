# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only diagnostic guard tests; these do not certify NPU/HCCL execution."""

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
        pytest.param(PROCESS_HEADER + BUSY_ZERO + IDLE_ONE, 1, id="zero-busy"),
        pytest.param(PROCESS_HEADER + IDLE_ZERO + BUSY_ONE, 1, id="one-busy"),
        pytest.param(PROCESS_HEADER + BUSY_ZERO + BUSY_ONE, 1, id="both-busy"),
        pytest.param(PROCESS_HEADER + IDLE_ZERO, 1, id="one-missing"),
        pytest.param(PROCESS_HEADER + IDLE_ONE, 1, id="zero-missing"),
        pytest.param(PROCESS_HEADER, 1, id="both-missing"),
        pytest.param(IDLE_ZERO + IDLE_ONE, 1, id="header-missing"),
        pytest.param("", 1, id="empty-output"),
        pytest.param(PROCESS_HEADER + IDLE_ZERO + IDLE_ONE + BUSY_OTHER, 0, id="other-card-busy"),
        pytest.param(PROCESS_HEADER + IDLE_ZERO + IDLE_TEN, 1, id="ten-not-one"),
        pytest.param(PROCESS_HEADER + IDLE_ONE + IDLE_TEN, 1, id="ten-not-zero"),
        pytest.param(PROCESS_HEADER + IDLE_ZERO + IDLE_ONE + BUSY_ZERO, 1, id="contradictory-zero"),
        pytest.param(PROCESS_HEADER + IDLE_ZERO + IDLE_ONE + BUSY_ONE, 1, id="contradictory-one"),
        pytest.param(IDLE_ZERO + IDLE_ONE + PROCESS_HEADER + BUSY_ZERO, 1, id="ignore-pre-header-idle"),
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
    else:
        assert "skip test" in output or "cannot recognize NPU process table" in output


@pytest.mark.parametrize("section", ["idle", "raw"])
def test_tp2_diagnose_embedded_python_compiles_without_importing_npu(script_source, section):
    compile(embedded_python(script_source, section), str(SCRIPT_PATH) + ":" + section, "exec")


def test_tp2_diagnose_shell_uses_lf_line_endings():
    assert b"\r" not in SCRIPT_PATH.read_bytes()
