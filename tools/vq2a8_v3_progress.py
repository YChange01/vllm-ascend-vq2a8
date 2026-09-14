# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Best-effort host progress, separate from measured request/token timestamps.

One ProgressReporter belongs to an entire benchmark run. Reusing it bounds the
run to one daemon writer even if stdout blocks indefinitely. Request update()
only replaces a tuple of Python scalars: no clock, locks, I/O or device calls.
Start/finish and reporter shutdown must be outside the request's timed region.
"""

import json
import math
import sys
import threading
import time
from collections import deque
from contextlib import suppress

WRITER_JOIN_TIMEOUT_S = 0.1
MAX_PENDING_TRANSITIONS = 2
MAX_WRITER_WAIT_S = 60.0


class ProgressReporter:
    """Explicit run-scoped owner; no shared mutable module/class singleton."""

    def __init__(self):
        self._stream = sys.stdout
        self._thread = None
        self._active = None
        self._pending = deque(maxlen=MAX_PENDING_TRANSITIONS)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._closed = False
        self._failed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def _activate(self, request):
        if self._closed or self._failed or request.interval_s == 0:
            return
        self._active = request
        self._pending.append((request, request._record(request._started_at, phase="prefill")))
        if self._thread is None:
            try:
                self._thread = threading.Thread(target=self._run, name="vq2a8-v3-progress", daemon=True)
                self._thread.start()
            except Exception:
                self._failed = True
                self._thread = None
                self._active = None
                self._pending.clear()
                return
        self._wake.set()

    def _finish(self, request, *, successful):
        if self._closed or self._failed or request.interval_s == 0:
            return
        if self._active is request:
            self._active = None
        phase = "completed" if successful else None
        self._pending.append((request, request._record(time.monotonic(), phase=phase, interrupted=not successful)))
        self._wake.set()

    def _write(self, record):
        request_id, completed, total, elapsed, last_token_age, phase, interrupted = record
        last_token = "none" if last_token_age is None else f"{last_token_age:.3f}"
        line = (
            f"PERF_V3_PROGRESS={request_id} tokens={completed}/{total} "
            f"elapsed_s={elapsed:.3f} last_token_s={last_token} phase={phase}"
        )
        if interrupted:
            line += " status=interrupted"
        self._stream.write(line + "\n")
        self._stream.flush()

    def _run(self):
        last_request, last_emit = None, 0.0
        try:
            while True:
                # Clear before inspecting state so a concurrent transition
                # cannot be lost between inspection and Event.wait().
                self._wake.clear()
                while self._pending:
                    request, record = self._pending.popleft()
                    self._write(record)
                    last_request, last_emit = request, time.monotonic()
                if self._stop.is_set():
                    return
                request = self._active
                now = time.monotonic()
                if request is not None:
                    if request is not last_request or now - last_emit >= request.interval_s:
                        self._write(request._record(now))
                        last_request, last_emit = request, time.monotonic()
                    timeout = min(MAX_WRITER_WAIT_S, max(0.0, request.interval_s - (time.monotonic() - last_emit)))
                else:
                    timeout = None
                self._wake.wait(timeout)
        except Exception:
            # Closed/broken stdout and background failures are telemetry loss,
            # never inference failures. Do not attempt another blocking sink.
            self._failed = True
        finally:
            self._active = None
            self._pending.clear()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._active = None
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            with suppress(Exception):
                self._thread.join(timeout=WRITER_JOIN_TIMEOUT_S)
        # A stuck stdout writer remains a single daemon. It owns only scalar
        # progress records, not the model, tensors, request outputs or files.


class RequestProgress:
    """Use reporter=shared_owner for every request in a multi-request run.

    elapsed_s passed to update is the caller's existing request-relative token
    timestamp. Printed elapsed/last_token_s are approximate liveness indicators,
    not TTFT/TPOT measurements; last_token_s is age since the latest token.
    """

    def __init__(self, request_id, total_tokens, interval_s=5.0, *, reporter=None):
        if type(request_id) is not str or not request_id:
            raise ValueError("Progress requires a nonempty string request ID.")
        if type(total_tokens) is not int or total_tokens <= 0:
            raise ValueError("Progress total_tokens must be a positive integer.")
        if type(interval_s) not in (int, float) or not math.isfinite(interval_s) or interval_s < 0:
            raise ValueError("Progress interval_s must be finite and non-negative; zero disables it.")
        if reporter is not None and not isinstance(reporter, ProgressReporter):
            raise TypeError("reporter must be the run's ProgressReporter instance.")
        self.request_id = json.dumps(request_id, ensure_ascii=True)
        self.total_tokens, self.interval_s = total_tokens, float(interval_s)
        self._snapshot = (0, 0.0)
        self._reporter = reporter
        self._owns_reporter = reporter is None
        self._started_at = 0.0
        self._entered = self._finished = False

    def __enter__(self):
        if self._entered:
            raise RuntimeError("RequestProgress contexts cannot be reused.")
        self._entered = True
        self._started_at = time.monotonic()
        if self._reporter is None:
            self._reporter = ProgressReporter()
        self._reporter._activate(self)
        return self

    def update(self, completed_tokens, elapsed_s):
        self._snapshot = (completed_tokens, elapsed_s)

    def _record(self, now, *, phase=None, interrupted=False):
        completed, token_elapsed = self._snapshot
        # Validate only in telemetry, never convert a tensor on the update path.
        if type(completed) is not int or type(token_elapsed) not in (int, float):
            raise TypeError("Progress snapshots must contain only Python scalars.")
        elapsed = max(0.0, now - self._started_at, token_elapsed)
        age = max(0.0, elapsed - token_elapsed) if completed else None
        phase = phase or ("decode" if completed else "prefill")
        return self.request_id, completed, self.total_tokens, elapsed, age, phase, interrupted

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._finished:
            self._finished = True
            try:
                self._reporter._finish(self, successful=exc_type is None)
            except Exception:
                pass
            finally:
                if self._owns_reporter:
                    self._reporter.close()
        return False
