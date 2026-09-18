# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, instance-local host spans; never synchronize or inspect tensors.

Wall time includes host waits and submission. Thread CPU time measures only
the calling thread, not NPU execution, all worker threads, or CPU utilization.
Nested spans are inclusive and must not be added together as exclusive time.
Only install wrappers when requested; the disabled path needs no recorder.
"""

from contextlib import contextmanager
from functools import wraps
from threading import Lock
from time import perf_counter, thread_time

from torch.autograd.profiler import record_function

HOST_RANGE_PREFIX = "vq2a8::host::"
MAX_HOST_PHASES = 64
OVERFLOW_PHASE = "__other__"
RUNNER_HOST_METHODS = (
    "_update_states",
    "_prepare_inputs",
    "_preprocess",
    "_build_attention_metadata",
    "_sample",
    "_bookkeeping_sync",
    "execute_model",
    "sample_tokens",
)


def _empty_totals():
    return {
        "calls": 0,
        "total_wall_s": 0.0,
        "max_wall_s": 0.0,
        "total_thread_cpu_s": 0.0,
        "max_thread_cpu_s": 0.0,
    }


class HostProfileRecorder:
    """Bounded cumulative host counters plus current-profiler CPU annotations.

    ``reset`` starts a new completed-span accounting window. An active span
    finishing after reset belongs to that new window. Names stay registered
    across resets so concurrent/nested completion cannot exceed the bound.
    """

    def __init__(
        self, *, wall_clock=perf_counter, cpu_clock=thread_time, range_factory=None, max_phases=MAX_HOST_PHASES
    ):
        if type(max_phases) is not int or max_phases < 1:
            raise ValueError("max_phases must be a positive integer")
        self._wall_clock = wall_clock
        self._cpu_clock = cpu_clock
        self._range_factory = record_function if range_factory is None else range_factory
        self._max_phases = max_phases
        self._totals = {}
        self._lock = Lock()

    def _name(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError("Host phase name must be a nonempty string")
        with self._lock:
            if name not in self._totals and len(self._totals) >= self._max_phases - 1:
                name = OVERFLOW_PHASE
            self._totals.setdefault(name, _empty_totals())
        return name

    @contextmanager
    def phase(self, name):
        """Measure a synchronous host scope, preserving its result/exception."""
        name = self._name(name)
        with self._range_factory(HOST_RANGE_PREFIX + name):
            wall_start, cpu_start = self._wall_clock(), self._cpu_clock()
            try:
                yield
            finally:
                wall = self._wall_clock() - wall_start
                cpu = self._cpu_clock() - cpu_start
                with self._lock:
                    totals = self._totals[name]
                    totals["calls"] += 1
                    totals["total_wall_s"] += wall
                    totals["max_wall_s"] = max(totals["max_wall_s"], wall)
                    totals["total_thread_cpu_s"] += cpu
                    totals["max_thread_cpu_s"] = max(totals["max_thread_cpu_s"], cpu)

    def report(self, *, reset=False):
        """Return a detached JSON-compatible snapshot; do not print or sync."""
        with self._lock:
            phases = {name: dict(value) for name, value in sorted(self._totals.items()) if value["calls"]}
            if reset:
                for name in self._totals:
                    self._totals[name] = _empty_totals()
        return {
            "scope": "host",
            "timing": "inclusive",
            "device_synchronized": False,
            "thread_cpu_scope": "calling_thread_only",
            "max_phases": self._max_phases,
            "phases": phases,
        }

    def reset(self):
        """Reset cumulative counters without changing wrappers or phase names."""
        with self._lock:
            for name in self._totals:
                self._totals[name] = _empty_totals()


def instrumentation_range(recorder, name):
    """Decorate a synchronous callable with one inclusive host span."""

    def decorate(function):
        return wrap_host_call(recorder, name, function)

    return decorate


def wrap_host_call(recorder, name, function):
    """Wrap one callable (including a bound method), preserving its signature."""

    @wraps(function)
    def profiled(*args, **kwargs):
        with recorder.phase(name):
            return function(*args, **kwargs)

    return profiled


def attach_host_profile(runner, recorder=None):
    """Wrap available known methods on this runner only; repeated calls are safe.

    Nothing is patched on its class or on ``nn.Module``. Missing methods are
    normal across runner versions. An existing recorder cannot be replaced:
    doing so would retain old closures or count each call twice.
    """
    existing = getattr(runner, "_vq2a8_host_profile_recorder", None)
    if existing is not None:
        if recorder is not None and recorder is not existing:
            raise ValueError("This runner already has a different host profile recorder")
        return existing
    if recorder is None:
        recorder = HostProfileRecorder()
    for name in RUNNER_HOST_METHODS:
        method = getattr(runner, name, None)
        if callable(method):
            setattr(runner, name, wrap_host_call(recorder, name, method))
    runner._vq2a8_host_profile_recorder = recorder
    return recorder
