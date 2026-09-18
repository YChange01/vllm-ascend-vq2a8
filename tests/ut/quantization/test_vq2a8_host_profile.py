# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only timing/wrapper contracts, without NPU activity or synchronization."""

import inspect
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.quantization.vq2a8_host_profile import (
    HOST_RANGE_PREFIX,
    OVERFLOW_PHASE,
    RUNNER_HOST_METHODS,
    HostProfileRecorder,
    attach_host_profile,
    instrumentation_range,
    wrap_host_call,
)


class Clock:
    def __init__(self):
        self.wall = 0.0
        self.cpu = 0.0

    def advance(self, wall, cpu):
        self.wall += wall
        self.cpu += cpu


@pytest.fixture
def probe():
    clock = Clock()
    scopes = []

    @contextmanager
    def scope(name):
        scopes.append(("enter", name))
        try:
            yield
        finally:
            scopes.append(("exit", name))

    recorder = HostProfileRecorder(wall_clock=lambda: clock.wall, cpu_clock=lambda: clock.cpu, range_factory=scope)
    return recorder, clock, scopes


def test_host_profile_separates_inclusive_wall_and_thread_cpu(probe, capsys):
    recorder, clock, scopes = probe
    with recorder.phase("execute_model"):
        clock.advance(2.0, 0.2)
        with recorder.phase("metadata"):
            clock.advance(3.0, 0.3)
        clock.advance(5.0, 0.5)
    report = recorder.report()
    assert report["scope"] == "host" and report["timing"] == "inclusive"
    assert not report["device_synchronized"]
    assert report["thread_cpu_scope"] == "calling_thread_only"
    assert report["phases"]["execute_model"] == {
        "calls": 1,
        "total_wall_s": 10.0,
        "max_wall_s": 10.0,
        "total_thread_cpu_s": 1.0,
        "max_thread_cpu_s": 1.0,
    }
    assert report["phases"]["metadata"]["total_wall_s"] == 3.0
    assert report["phases"]["metadata"]["total_thread_cpu_s"] == pytest.approx(0.3)
    assert scopes == [
        ("enter", HOST_RANGE_PREFIX + "execute_model"),
        ("enter", HOST_RANGE_PREFIX + "metadata"),
        ("exit", HOST_RANGE_PREFIX + "metadata"),
        ("exit", HOST_RANGE_PREFIX + "execute_model"),
    ]
    json.dumps(report)
    assert not capsys.readouterr().out


def test_host_profile_accumulates_counts_totals_max_and_returns_detached_snapshot(probe):
    recorder, clock, _ = probe
    for duration in (1.0, 4.0, 2.0):
        with recorder.phase("copies"):
            clock.advance(duration, duration / 2)
    result = recorder.report()
    assert result["phases"]["copies"] == {
        "calls": 3,
        "total_wall_s": 7.0,
        "max_wall_s": 4.0,
        "total_thread_cpu_s": 3.5,
        "max_thread_cpu_s": 2.0,
    }
    result["phases"]["copies"]["calls"] = -1
    assert recorder.report()["phases"]["copies"]["calls"] == 3
    assert recorder.report(reset=True)["phases"]["copies"]["calls"] == 3
    assert recorder.report()["phases"] == {}
    with recorder.phase("copies"):
        clock.advance(2.0, 1.0)
    assert recorder.report()["phases"]["copies"]["calls"] == 1
    recorder.reset()
    assert recorder.report()["phases"] == {}


def test_host_profile_preserves_original_exception_and_closes_scope(probe):
    recorder, clock, scopes = probe
    error = RuntimeError("original execution error")
    with pytest.raises(RuntimeError) as raised, recorder.phase("replay"):
        clock.advance(3.0, 0.25)
        raise error
    assert raised.value is error
    assert recorder.report()["phases"]["replay"]["calls"] == 1
    assert scopes[-1] == ("exit", HOST_RANGE_PREFIX + "replay")


def test_host_profile_bounds_names_without_retaining_per_call_records():
    recorder = HostProfileRecorder(max_phases=3)
    for index in range(100):
        with recorder.phase(f"dynamic_{index}"):
            pass
    phases = recorder.report()["phases"]
    assert set(phases) == {"dynamic_0", "dynamic_1", OVERFLOW_PHASE}
    assert phases[OVERFLOW_PHASE]["calls"] == 98
    recorder.reset()
    with recorder.phase("new_after_reset"):
        pass
    assert set(recorder.report()["phases"]) == {OVERFLOW_PHASE}


@pytest.mark.parametrize("maximum", [0, -1, True, 2.5])
def test_host_profile_rejects_invalid_phase_bound(maximum):
    with pytest.raises(ValueError, match="positive integer"):
        HostProfileRecorder(max_phases=maximum)


@pytest.mark.parametrize("name", [None, "", 42])
def test_host_profile_rejects_invalid_phase_name(probe, name):
    recorder, _, _ = probe
    with pytest.raises(ValueError, match="nonempty string"), recorder.phase(name):
        pass


def test_host_profile_decorator_retains_signature_result_and_metadata(probe):
    recorder, clock, _ = probe

    def operation(a, /, b=2, *, extra=3):
        """Original docstring."""
        clock.advance(2.0, 0.4)
        return a + b + extra

    wrapped = instrumentation_range(recorder, "metadata")(operation)
    assert inspect.signature(wrapped) == inspect.signature(operation)
    assert wrapped.__name__ == operation.__name__ and wrapped.__doc__ == operation.__doc__
    assert wrapped.__wrapped__ is operation
    assert wrapped(1, b=4, extra=5) == 10
    assert recorder.report()["phases"]["metadata"]["calls"] == 1


def test_host_profile_attach_is_instance_local_and_idempotent(probe):
    recorder, clock, _ = probe

    class Runner:
        def execute_model(self, value, *, add=1):
            return self._prepare_inputs(value) + add

        def _prepare_inputs(self, value):
            clock.advance(2.0, 0.5)
            return value * 2

        def unrelated(self):
            return "untouched"

    original = Runner.execute_model
    runner, other = Runner(), Runner()
    assert attach_host_profile(runner, recorder) is recorder
    wrapper = runner.execute_model
    assert attach_host_profile(runner) is recorder
    assert attach_host_profile(runner, recorder) is recorder
    assert runner.execute_model is wrapper
    assert Runner.execute_model is original and other.execute_model.__func__ is original
    assert runner.unrelated.__func__ is Runner.unrelated
    assert inspect.signature(runner.execute_model) == inspect.signature(other.execute_model)
    assert runner.execute_model(3, add=4) == 10
    assert other.execute_model(3, add=4) == 10
    phases = recorder.report()["phases"]
    assert phases["execute_model"]["calls"] == phases["_prepare_inputs"]["calls"] == 1
    with pytest.raises(ValueError, match="different host profile recorder"):
        attach_host_profile(runner, HostProfileRecorder())


def test_host_profile_attach_skips_absent_or_noncallable_methods():
    runner = SimpleNamespace(_sample=None)
    recorder = attach_host_profile(runner)
    assert isinstance(recorder, HostProfileRecorder)
    assert runner._sample is None and recorder.report()["phases"] == {}
    assert RUNNER_HOST_METHODS == (
        "_update_states",
        "_prepare_inputs",
        "_preprocess",
        "_build_attention_metadata",
        "_sample",
        "_bookkeeping_sync",
        "execute_model",
        "sample_tokens",
    )


def test_host_profile_never_synchronizes_or_reads_tensor_values(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Host diagnostics must not inspect/synchronize tensor values")

    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=forbidden), raising=False)
    recorder = HostProfileRecorder()
    value = object()
    assert wrap_host_call(recorder, "submit", lambda: value)() is value
    assert recorder.report()["phases"]["submit"]["calls"] == 1


def test_host_profile_annotations_are_visible_to_current_cpu_profiler():
    recorder = HostProfileRecorder()
    with (
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], acc_events=True) as profile,
        recorder.phase("runtime_contract"),
    ):
        pass
    assert HOST_RANGE_PREFIX + "runtime_contract" in {event.key for event in profile.key_averages()}


def test_host_profile_reset_inside_active_scope_preserves_later_completion(probe):
    recorder, clock, _ = probe
    with recorder.phase("execute_model"):
        clock.advance(1.0, 0.25)
        recorder.reset()
        clock.advance(2.0, 0.5)
    assert recorder.report()["phases"]["execute_model"]["total_wall_s"] == 3.0
