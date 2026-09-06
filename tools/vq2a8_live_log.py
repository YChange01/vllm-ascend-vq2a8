# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mirror a child's on-disk log without piping/blocking its stdout or stderr."""

from __future__ import annotations

import codecs
import sys
import threading
import time
from pathlib import Path
from typing import TextIO


class LiveChildLog:
    """The child writes directly to disk; a bounded reader tees it to the terminal.

    File redirection preserves subprocess.run's timeout/abort behavior and lets
    the child continue logging even if the terminal's pipe closes. No NPU
    imports, process-wide patches or unbounded output queue are needed.
    """

    def __init__(
        self, path: Path, probe: str, *, console: TextIO | None = None, heartbeat: float = 15.0, poll: float = 0.1
    ):
        self.path, self.probe = path, probe
        self.console = console if console is not None else sys.stdout
        self.heartbeat, self.poll = heartbeat, poll
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._relay, name="vq2a8-live-log", daemon=True)
        self._console_open = True
        self._pending_line = ""
        self._last_stage = "child_start"
        self._line_start = True

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        # Caller has closed/flushed the writer; drain the remaining bytes once.
        self._stop.set()
        self._thread.join()

    def _emit(self, text: str):
        if not text or not self._console_open:
            return
        try:
            self.console.write(text)
            self.console.flush()
            self._line_start = text.endswith("\n")
        except (OSError, UnicodeError, ValueError):
            # A disconnected/closed terminal must not discard the disk log or
            # change the child's acceptance result.
            self._console_open = False

    def _record_stage(self, text: str):
        lines = (self._pending_line + text).split("\n")
        for line in lines[:-1]:
            if line.startswith(("MODEL ", "MOE ", "KERNEL ", "CHAIN ", "ROOT_FP8 ")):
                self._last_stage = line.strip()[:240]
        self._pending_line = lines[-1][-1024:]

    def _relay(self):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        started = last_output = time.monotonic()
        with self.path.open("rb") as source:
            while True:
                # Observe shutdown BEFORE reading. If the writer flushes and
                # stops between an empty read and this check, read again on
                # the next iteration instead of dropping its final bytes.
                draining = self._stop.is_set()
                chunk = source.read(65536)
                if chunk:
                    text = decoder.decode(chunk)
                    self._record_stage(text)
                    self._emit(text)
                    last_output = time.monotonic()
                    continue
                if draining:
                    self._emit(decoder.decode(b"", final=True))
                    if not self._line_start:
                        self._emit("\n")
                    return
                now = time.monotonic()
                if now - last_output >= self.heartbeat:
                    prefix = "" if self._line_start else "\n"
                    self._emit(
                        f"{prefix}PROBE_WAIT={self.probe} elapsed_s={now - started:.1f} "
                        f"last_stage={self._last_stage} LOG={self.path}\n"
                    )
                    last_output = now
                self._stop.wait(self.poll)
