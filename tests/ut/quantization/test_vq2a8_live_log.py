# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from tools.validate_vq2a8_tp1_acceptance import summarize_log
from tools.vq2a8_live_log import LiveChildLog


def test_child_output_visible_before_exit_and_stderr_retained(tmp_path):
    class Console(io.StringIO):
        def __init__(self):
            super().__init__()
            self.ready = threading.Event()

        def write(self, text):
            result = super().write(text)
            if "MODEL stage=waiting_for_input\n" in self.getvalue():
                self.ready.set()
            return result

    path, console = tmp_path / "child.log", Console()
    code = (
        "import sys; print('MODEL stage=waiting_for_input', flush=True); "
        "sys.stdin.readline(); print('stderr tail', file=sys.stderr, flush=True)"
    )
    with path.open("wb") as log, LiveChildLog(path, "test", console=console, poll=0.01):
        child = subprocess.Popen(
            [sys.executable, "-u", "-c", code], stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            assert console.ready.wait(10), "Output was not mirrored while the child was alive."
            assert child.poll() is None
            child.communicate(input=b"continue\n", timeout=10)
            assert child.returncode == 0
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()
    assert console.getvalue() == path.read_text()
    assert "stderr tail" in console.getvalue()


def test_partial_utf8_and_large_output_are_drained_without_changing_disk(tmp_path):
    path, console = tmp_path / "child.log", io.StringIO()
    payload = "start\n" + "数据" * 50000 + "\nend without newline"
    data = payload.encode()
    with path.open("wb") as log, LiveChildLog(path, "test", console=console, poll=0.01):
        log.write(data[:7])  # Split a multibyte character across writes/reads.
        log.flush()
        log.write(data[7:])
        log.flush()
    assert path.read_bytes() == data
    assert console.getvalue() == payload + "\n"


def test_silent_child_heartbeat_has_last_stage_and_is_not_saved_as_child_output(tmp_path):
    class Console(io.StringIO):
        def __init__(self):
            super().__init__()
            self.waiting = threading.Event()

        def write(self, text):
            result = super().write(text)
            if "PROBE_WAIT=" in text:
                self.waiting.set()
            return result

    path, console = tmp_path / "child.log", Console()
    with path.open("w") as log, LiveChildLog(path, "full_model", console=console, heartbeat=0.02, poll=0.005):
        log.write("MODEL layer=3 stage=decoder_start\n")
        log.flush()
        assert console.waiting.wait(10)
    assert "PROBE_WAIT=full_model" in console.getvalue()
    assert "last_stage=MODEL layer=3 stage=decoder_start" in console.getvalue()
    assert "PROBE_WAIT=" not in path.read_text()


@pytest.mark.parametrize("returncode", [0, 1, 134])
def test_relay_does_not_turn_failure_into_pass(tmp_path, returncode):
    path, console = tmp_path / "child.log", io.StringIO()
    code = f"print('VQ2A8_TP1_OFFLINE_EXECUTION_GATE=PASS {{}}', flush=True); raise SystemExit({returncode})"
    with path.open("wb") as log, LiveChildLog(path, "test", console=console):
        child = subprocess.run([sys.executable, "-u", "-c", code], stdout=log, stderr=subprocess.STDOUT, check=False)
    assert summarize_log(path, child.returncode)["passed"] is (returncode == 0)
    assert console.getvalue() == path.read_text()


def test_timeout_keeps_and_displays_partial_log(tmp_path):
    path, console = tmp_path / "child.log", io.StringIO()
    code = "import time; print('MODEL stage=slow', flush=True); time.sleep(60)"
    with (
        path.open("wb") as log,
        LiveChildLog(path, "test", console=console),
        pytest.raises(subprocess.TimeoutExpired),
    ):
        subprocess.run([sys.executable, "-u", "-c", code], stdout=log, stderr=subprocess.STDOUT, timeout=1)
    assert "MODEL stage=slow" in console.getvalue()
    assert console.getvalue() == path.read_text()
    assert not summarize_log(path, None, timed_out=True)["passed"]


def test_closed_terminal_does_not_lose_disk_log(tmp_path):
    class BrokenConsole:
        def write(self, _):
            raise BrokenPipeError("closed pipe")

    path = tmp_path / "child.log"
    with path.open("w") as log, LiveChildLog(path, "test", console=BrokenConsole()):
        log.write("MODEL stage=still_logged\n")
        log.flush()
    assert path.read_text() == "MODEL stage=still_logged\n"


def test_shutdown_between_empty_read_and_stop_check_still_drains_tail():
    class RacingSource(io.BytesIO):
        def __init__(self):
            super().__init__(b"last flushed bytes\n")
            self.first = True

        def read(self, size):
            if self.first:
                self.first = False
                # Simulate the child flushing its last bytes and exiting just
                # after a reader saw EOF. A second read must drain those bytes.
                relay._stop.set()
                return b""
            return super().read(size)

    console = io.StringIO()
    relay = LiveChildLog(SimpleNamespace(open=lambda _: RacingSource()), "test", console=console)
    relay._relay()
    assert console.getvalue() == "last flushed bytes\n"
