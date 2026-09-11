# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only logging tests; no devices or inference timing claims."""

import ast
import inspect
import io
import threading
import time
from textwrap import dedent
from types import SimpleNamespace as NS

import pytest

from tools import vq2a8_v3_progress as progress
from vllm_ascend.quantization.vq2a8_execution_v3 import AscendCV3VQ2TP1MoE


def test_v3_progress_update_is_only_a_scalar_snapshot_assignment(monkeypatch):
    value = progress.RequestProgress("request", 4, 0)
    body = ast.parse(dedent(inspect.getsource(value.update))).body[0].body
    assert len(body) == 1 and isinstance(body[0], ast.Assign)
    assert not any(isinstance(node, ast.Call) for node in ast.walk(body[0]))
    monkeypatch.setattr(progress, "time", NS(monotonic=lambda: pytest.fail("update must not read a clock")))
    value.update(2, 1.25)
    assert value._snapshot == (2, 1.25)


def test_v3_progress_interval_zero_never_starts_a_writer(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(progress.sys, "stdout", stream)
    monkeypatch.setattr(
        progress.threading, "Thread", lambda *args, **kwargs: pytest.fail("disabled progress started thread")
    )
    with (
        progress.ProgressReporter() as reporter,
        progress.RequestProgress("disabled", 4, 0, reporter=reporter) as request,
    ):
        request.update(4, 1.0)
    assert stream.getvalue() == "" and reporter._thread is None


def test_v3_progress_heartbeat_is_rate_limited_and_reports_stall_age(monkeypatch):
    clock = NS(now=0.0)
    monkeypatch.setattr(progress, "time", NS(monotonic=lambda: clock.now))
    reporter = progress.ProgressReporter()
    reporter._stream = io.StringIO()
    request = progress.RequestProgress("heartbeat", 4, 5, reporter=reporter)
    request._started_at = 0.0
    reporter._active = request
    waits = []

    def wait(timeout):
        waits.append(timeout)
        clock.now += timeout
        request.update(1, 1.5)
        if len(waits) == 3:
            reporter._stop.set()

    reporter._wake = NS(clear=lambda: None, wait=wait)
    reporter._run()
    lines = reporter._stream.getvalue().splitlines()
    assert waits == [5.0, 5.0, 5.0]
    assert len(lines) == 3
    assert "tokens=0/4 elapsed_s=0.000 last_token_s=none phase=prefill" in lines[0]
    assert "tokens=1/4 elapsed_s=5.000 last_token_s=3.500 phase=decode" in lines[1]
    assert "tokens=1/4 elapsed_s=10.000 last_token_s=8.500 phase=decode" in lines[2]


def test_v3_progress_all_requests_share_one_writer_and_never_print_on_main_thread(monkeypatch):
    writers = []

    class Stream(io.StringIO):
        def write(self, value):
            writers.append(threading.get_ident())
            return super().write(value)

    stream = Stream()
    monkeypatch.setattr(progress.sys, "stdout", stream)
    reporter = progress.ProgressReporter()
    worker = None
    for index in range(20):
        with progress.RequestProgress(f"request-{index}", 4, reporter=reporter) as request:
            request.update(4, 0.25)
        worker = worker or reporter._thread
        assert reporter._thread is worker
    reporter.close()
    assert worker is not None and worker.daemon and not worker.is_alive()
    assert writers and threading.get_ident() not in writers and len(set(writers)) == 1
    assert 'PERF_V3_PROGRESS="request-19" tokens=4/4' in stream.getvalue()
    assert "phase=completed" in stream.getvalue()


def test_v3_progress_blocked_stdout_keeps_one_thread_bounded_state_and_bounded_shutdown(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    class BlockedStream:
        def write(self, value):
            entered.set()
            release.wait(2.0)

        def flush(self):
            pass

    monkeypatch.setattr(progress.sys, "stdout", BlockedStream())
    reporter = progress.ProgressReporter()
    try:
        with progress.RequestProgress("blocked-0", 4, reporter=reporter) as request:
            assert entered.wait(1.0)
            request.update(4, 0.1)
        worker = reporter._thread
        for index in range(1, 30):
            with progress.RequestProgress(f"blocked-{index}", 4, reporter=reporter) as request:
                request.update(4, 0.2)
            assert reporter._thread is worker
            assert len(reporter._pending) <= progress.MAX_PENDING_TRANSITIONS
        started = time.perf_counter()
        reporter.close()
        assert time.perf_counter() - started < 0.5
        assert worker.is_alive() and worker.daemon
    finally:
        release.set()
        reporter.close()
        if reporter._thread is not None:
            reporter._thread.join(timeout=1.0)
    assert not reporter._thread.is_alive()


@pytest.mark.parametrize("error", [BrokenPipeError, ValueError, RuntimeError])
def test_v3_progress_stdout_errors_do_not_change_request_result(monkeypatch, error):
    class BrokenStream:
        def write(self, value):
            raise error("closed stdout")

        def flush(self):
            raise error("closed stdout")

    monkeypatch.setattr(progress.sys, "stdout", BrokenStream())
    with progress.RequestProgress("broken", 4) as request:
        request.update(4, 1.0)
        result = "inference result"
    assert result == "inference result" and request._reporter._failed


def test_v3_progress_thread_start_failure_is_best_effort(monkeypatch):
    class FailedThread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise RuntimeError("thread unavailable")

    monkeypatch.setattr(progress.threading, "Thread", FailedThread)
    with progress.RequestProgress("unavailable", 4) as request:
        request.update(4, 1.0)
    assert request._reporter._failed and request._reporter._thread is None


def test_v3_progress_preserves_inference_exception_and_never_reports_it_completed(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(progress.sys, "stdout", stream)
    with pytest.raises(RuntimeError, match="inference failed"), progress.RequestProgress("failed", 4) as request:
        request.update(1, 0.1)
        raise RuntimeError("inference failed")
    assert "status=interrupted" in stream.getvalue()
    assert "phase=completed" not in stream.getvalue()


def test_v3_progress_request_id_is_single_line_escaped_and_early_eos_can_complete(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(progress.sys, "stdout", stream)
    with progress.RequestProgress("one\ntwo", 8) as request:
        request.update(2, 0.5)
    text = stream.getvalue()
    assert 'PERF_V3_PROGRESS="one\\ntwo"' in text
    assert "tokens=2/8" in text and "phase=completed" in text


@pytest.mark.parametrize("interval", [-1, True, float("nan"), float("inf"), "5"])
def test_v3_progress_rejects_invalid_interval_before_timing(interval):
    with pytest.raises(ValueError, match="interval"):
        progress.RequestProgress("id", 4, interval)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_v3_progress_rejects_invalid_token_target_before_timing(count):
    with pytest.raises(ValueError, match="total_tokens"):
        progress.RequestProgress("id", count)


@pytest.mark.parametrize(
    "loaded,total,now,last,expected",
    [
        (1, 256, 1.0, 0.0, False),
        (32, 256, 1.0, 0.0, True),
        (256, 256, 1.0, 0.0, True),
        (3, 256, 5.0, 0.0, True),
        (4, 256, 6.0, 5.0, False),
        (1, 1, 0.1, 0.0, True),
    ],
)
def test_v3_resident_load_progress_uses_count_or_host_time_limit(loaded, total, now, last, expected):
    assert AscendCV3VQ2TP1MoE._resident_load_progress_due(loaded, total, now, last) is expected


def test_v3_resident_load_progress_distinguishes_loaded_payload_from_planned_residency(capsys):
    value = AscendCV3VQ2TP1MoE.__new__(AscendCV3VQ2TP1MoE)
    value.layer_index = 3
    value._resident_load_progress(32, 256, 12.5, 2 * 1024**3, 17 * 1024**3)
    line = capsys.readouterr().out
    assert "layer=3 stage=v3_resident_payload loaded=32 total=256 elapsed_s=12.500" in line
    assert "loaded_gib=2.000000 planned_gib=17.000000" in line
    assert "bytes_scope=loaded_payload_vs_planned_resident" in line


def test_v3_resident_progress_adds_no_device_synchronization_or_decode_logging():
    cls = ast.parse(dedent(inspect.getsource(AscendCV3VQ2TP1MoE)))
    methods = {node.name: node for node in cls.body[0].body if isinstance(node, ast.FunctionDef)}
    for name in ("_forward", "_resident_load_progress", "_resident_load_progress_due"):
        calls = [node for node in ast.walk(methods[name]) if isinstance(node, ast.Call)]
        assert not any(isinstance(node.func, ast.Attribute) and node.func.attr == "synchronize" for node in calls)
        assert not any(isinstance(node.func, ast.Name) and node.func.id == "synchronize_execution" for node in calls)
    assert "_resident_load_progress" not in ast.unparse(methods["_forward"])
    initializer = ast.unparse(methods["initialize_resident"])
    assert "if self.progress:" in initializer
