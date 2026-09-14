# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, instance-local tracing of the real V3 startup execution path.

Async mode never adds a device fence: PASS only means the Python call returned.
Sync mode fences after SUBMITTED and is diagnostic, never a performance run.
Neither mode changes measurement mode, arithmetic, routing, or tensor values.
"""

from __future__ import annotations

import atexit
import faulthandler
import functools
import json
import os
import tempfile
import threading
import time
import weakref
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path


def _npu_synchronize():
    import torch

    torch.npu.synchronize()


def _call_metadata(args, kwargs):
    """Only host-side tensor descriptors; never inspect tensor data."""
    tensors = {}
    for name, value in [*((f"arg{index}", value) for index, value in enumerate(args)), *kwargs.items()]:
        shape = getattr(value, "shape", None)
        if shape is not None and hasattr(value, "dtype"):
            tensors[name] = {
                "shape": [int(dim) for dim in shape],
                "dtype": str(value.dtype),
                "device": str(getattr(value, "device", "unknown")),
            }
    return {"tensors": tensors} if tensors else {}


class StartupTrace:
    # faulthandler has one process-wide delayed timer. Do not let two tracers
    # replace/cancel one another's timer. Timers outside this helper cannot be
    # queried by Python; the engine must reserve this facility for diagnostics.
    _timer_lock = threading.Lock()
    _timer_owner = None

    def __init__(self, mode, *, directory=None, synchronize=None, stack_timer=True, heartbeat_s=30):
        if mode not in ("async", "sync"):
            raise ValueError("Startup trace mode must be async or sync")
        if heartbeat_s < 0:
            raise ValueError("heartbeat_s must be nonnegative")
        self.mode = mode
        self.directory = Path(directory if directory is not None else tempfile.mkdtemp(prefix="vq2-full-startup-"))
        self.directory = self.directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events_path = self.directory / "events.jsonl"
        self.stacks_path = self.directory / "stacks.log"
        self._events = self.events_path.open("x", encoding="utf-8", buffering=1)
        try:
            self._stacks = self.stacks_path.open("x", encoding="utf-8", buffering=1)
        except BaseException:
            self._events.close()
            raise
        self._synchronize = synchronize if synchronize is not None else _npu_synchronize
        self._started = time.monotonic()
        self._lock = threading.RLock()
        self._local = threading.local()
        self._seq = 0
        self._span_seq = 0
        self._active = {}
        self._last = None
        self._patches = []
        self._wrapped = set()
        self._timer_armed = False
        self._stop = threading.Event()
        self._heartbeat = None
        self.closed = False
        try:
            if stack_timer:
                with self._timer_lock:
                    owner = self._timer_owner() if self._timer_owner is not None else None
                    if owner is not None and not owner.closed:
                        raise RuntimeError("Another StartupTrace owns the process-wide stack timer")
                    faulthandler.dump_traceback_later(30, repeat=True, file=self._stacks)
                    type(self)._timer_owner = weakref.ref(self)
                    self._timer_armed = True
            if heartbeat_s:
                self._heartbeat = threading.Thread(
                    target=self._heartbeat_loop, args=(heartbeat_s,), name="vq2-startup-heartbeat", daemon=True
                )
                self._heartbeat.start()
            atexit.register(self.close)
        except BaseException:
            self.close()
            raise
        print(
            "STARTUP_TRACE_ARMED="
            + json.dumps(
                {
                    "directory": str(self.directory),
                    "events": str(self.events_path),
                    "stacks": str(self.stacks_path),
                    "mode": self.mode,
                    "pid": os.getpid(),
                    "main_thread_id": threading.main_thread().ident,
                    "device_synchronized": False,
                    "synchronize_after_submission": self.mode == "sync",
                    "async_pass_is_device_completion": False,
                    "timing_valid": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    @staticmethod
    def _boundary(stage):
        return stage in ("model.forward", "model.compute_logits", "decoder_model.forward") or (
            stage.startswith("decoder.") and stage.endswith(".forward")
        )

    def _emit(self, event, span_id, parent_id, stage, started, metadata, **extra):
        with self._lock:
            if self.closed:
                return
            now = time.monotonic()
            self._seq += 1
            record = {
                "event": event,
                "seq": self._seq,
                "span_id": span_id,
                "parent_id": parent_id,
                "stage": stage,
                "pid": os.getpid(),
                "thread_id": threading.get_ident(),
                "main_thread_id": threading.main_thread().ident,
                "time": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": round(now - self._started, 6),
                "span_elapsed_s": round(now - started, 6),
                "mode": self.mode,
                "timing_valid": False,
                "device_synchronized": event == "PASS" and self.mode == "sync",
                "device_completion": "synchronized" if event == "PASS" and self.mode == "sync" else "not_verified",
                "metadata": metadata,
                **extra,
            }
            self._events.write(json.dumps(record, sort_keys=True) + "\n")
            self._events.flush()
            self._last = record
            if span_id is not None:
                self._active[span_id] = record
            if event == "FAIL" or self._boundary(stage):
                print("STARTUP_TRACE_EVENT=" + json.dumps(record, sort_keys=True), flush=True)

    def ready(self):
        """Persist host/thread identity even if worker profiling never starts."""
        self._emit(
            "READY",
            None,
            None,
            "startup.awaiting_worker_profile",
            self._started,
            {"hooks_installed": len(self._patches)},
        )

    @contextmanager
    def span(self, stage, **metadata):
        if self.closed:
            yield
            return
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = self._local.stack = []
        with self._lock:
            self._span_seq += 1
            span_id = self._span_seq
        parent_id = stack[-1] if stack else None
        started = time.monotonic()
        self._emit("BEGIN", span_id, parent_id, stage, started, metadata)
        stack.append(span_id)
        try:
            yield
            self._emit("SUBMITTED", span_id, parent_id, stage, started, metadata)
            if self.mode == "sync":
                self._synchronize()
            self._emit("PASS", span_id, parent_id, stage, started, metadata)
        except BaseException as exc:
            # A logging failure must not replace the operation's original
            # exception. No extra synchronization on this failure path.
            with suppress(Exception):
                self._emit(
                    "FAIL", span_id, parent_id, stage, started, metadata, error_type=type(exc).__name__, error=str(exc)
                )
            raise
        finally:
            stack.pop()
            with self._lock:
                self._active.pop(span_id, None)

    def _replace(self, obj, name, replacement):
        key = (id(obj), name)
        if key in self._wrapped:
            return False
        own = getattr(obj, "__dict__", {})
        present = name in own
        previous = own.get(name)
        setattr(obj, name, replacement)
        self._patches.append((obj, name, present, previous, replacement))
        self._wrapped.add(key)
        return True

    def wrap(self, obj, name, stage, *, metadata=None):
        """Wrap one instance method, preserving the original returned object.

        ``stage`` may be a string or callable accepting the original arguments.
        Optional ``metadata`` has the same calling convention and returns dict.
        """
        original = getattr(obj, name, None)
        if not callable(original) or (id(obj), name) in self._wrapped:
            return False

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            label = stage(*args, **kwargs) if callable(stage) else stage
            details = _call_metadata(args, kwargs)
            if metadata is not None:
                details.update(metadata(*args, **kwargs))
            with self.span(label, **details):
                return original(*args, **kwargs)

        return self._replace(obj, name, wrapped)

    def wrap_scope(self, state, prefix):
        original = getattr(state, "scope", None)
        if not callable(original) or (id(state), "scope") in self._wrapped:
            return False

        @functools.wraps(original)
        @contextmanager
        def scope(name):
            with self.span(f"{prefix}.scope.{name}"), original(name) as value:
                yield value

        return self._replace(state, "scope", scope)

    def _heartbeat_loop(self, interval):
        while not self._stop.wait(interval):
            with self._lock:
                if self.closed:
                    return
                active = bool(self._active)
                last = next(reversed(self._active.values())) if active else self._last
                snapshot = {
                    "pid": os.getpid(),
                    "elapsed_s": round(time.monotonic() - self._started, 3),
                    "active": active,
                    "last": last,
                }
            print("STARTUP_TRACE_WAIT=" + json.dumps(snapshot, sort_keys=True), flush=True)

    def close(self):
        """Restore only this tracer's wrappers and stop its own resources."""
        with self._lock:
            if self.closed:
                return
            self.closed = True
        self._stop.set()
        if self._heartbeat is not None and self._heartbeat is not threading.current_thread():
            self._heartbeat.join(timeout=1)
        for obj, name, present, previous, replacement in reversed(self._patches):
            if getattr(obj, name, None) is replacement:
                if present:
                    setattr(obj, name, previous)
                else:
                    delattr(obj, name)
        self._patches.clear()
        self._wrapped.clear()
        if self._timer_armed:
            with self._timer_lock:
                owner = self._timer_owner() if self._timer_owner is not None else None
                if owner is self:
                    faulthandler.cancel_dump_traceback_later()
                    type(self)._timer_owner = None
            self._timer_armed = False
        self._events.close()
        self._stacks.close()
        atexit.unregister(self.close)


def _request_metadata(requests):
    return {
        "jobs": len(requests),
        "rows": [int(hidden.shape[0]) for hidden, _, _ in requests],
        "widths": [int(spec.columns) for _, _, spec in requests],
    }


def install_startup_trace(model, *, mode):
    """Install once after resident loading; the caller validates TP1/eager V3.

    No class-wide monkey patches and no changes to any measurement/profile
    flags. The model owns the returned tracer until explicit close or exit.
    """
    previous = getattr(model, "_vq2a8_startup_trace", None)
    if previous is not None and not previous.closed:
        if previous.mode != mode:
            raise ValueError("Close the existing startup trace before changing mode")
        return previous
    trace = StartupTrace(mode)
    try:
        trace.wrap(model, "forward", "model.forward")
        trace.wrap(model, "compute_logits", "model.compute_logits")
        inner = getattr(model, "model", None)
        if inner is not None and inner is not model:
            trace.wrap(inner, "forward", "decoder_model.forward")
        runtimes = {}
        for index, decoder in enumerate(getattr(inner, "layers", ())):
            layer = getattr(decoder, "layer_idx", index)
            prefix = f"decoder.{layer}"
            phase = {"name": "attention"}

            def hc_pre_stage(*args, decoder=decoder, prefix=prefix, phase=phase, **kwargs):
                fn = args[1] if len(args) > 1 else kwargs.get("hc_fn")
                phase["name"] = "attention" if fn is getattr(decoder, "hc_attn_fn", None) else "ffn"
                return f"{prefix}.hc_pre.{phase['name']}"

            def hc_post_stage(*args, prefix=prefix, phase=phase, **kwargs):
                return f"{prefix}.hc_post.{phase['name']}"

            trace.wrap(decoder, "forward", f"{prefix}.forward")
            trace.wrap(decoder, "hc_pre", hc_pre_stage)
            trace.wrap(decoder, "hc_post", hc_post_stage)
            for attr, name in (
                ("self_attn", "attention"),
                ("input_layernorm", "input_layernorm"),
                ("post_attention_layernorm", "post_attention_layernorm"),
                ("mlp", "mlp"),
            ):
                module = getattr(decoder, attr, None)
                if module is not None:
                    trace.wrap(module, "forward", f"{prefix}.{name}")
            runtime = getattr(getattr(decoder, "mlp", None), "runtime", None)
            if runtime is not None:
                runtimes[id(runtime)] = (layer, runtime)
        owner = getattr(inner, "offline_owner", None)
        for layer, runtime in getattr(owner, "layers", {}).items():
            runtimes.setdefault(id(runtime), (layer, runtime))
        for layer, runtime in runtimes.values():
            prefix = f"moe.{layer}"
            for method, stage in (("forward", "forward"), ("shared", "shared"), ("_bind_stream", "bind_stream")):
                trace.wrap(runtime, method, f"{prefix}.{stage}")
            trace.wrap(runtime, "_launch_resident", f"{prefix}.launch_resident")
            preparation = getattr(runtime, "_row_preparation", None)
            if preparation is not None:
                trace.wrap(preparation, "many", f"{prefix}.prepare_many", metadata=_request_metadata)
            state = getattr(runtime, "_v3_prefill_state", None)
            if state is not None:
                trace.wrap_scope(state, prefix)
        trace.ready()
        model._vq2a8_startup_trace = trace
        return trace
    except BaseException:
        trace.close()
        raise
