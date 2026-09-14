# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicitly prepared, single-owner-stream V4 MoE decode graph.

This is runtime capture, not torch.compile or a full-model graph. The caller
selects genuine decode steps; B1 shape alone must never select this path.
CPU backend injection tests the protocol only, not NPU capture support.
"""

from contextlib import contextmanager
from threading import Lock
from time import perf_counter

import torch

GRAPH_WARMUPS = 2


class V4MoEDecodeGraph:
    """One graph and private pool; no lazy capture, shared pool, or fallback.

    The explicit stream/NPUGraph/replay/reset interfaces follow torch-npu
    v2.10.0 ``torch_npu/npu/graphs.py``. Actual wheel/CANN/native-op support
    must pass the physical-device probe. In particular, an unsupported owner
    stream is an error, never a reason to silently switch the bank's stream.
    """

    def __init__(self, device, *, backend=None, owners=(), signature=()):
        self.device = torch.device(device)
        if backend is None and self.device.type != "npu":
            raise ValueError("V4 MoE decode graph requires an NPU.")
        self.backend = torch.npu if backend is None else backend
        self.owners = tuple(owners)
        self.contract = signature
        self.graph = self.hidden = self.input_ids = self.output = self.valid = None
        self.stream = self.stream_id = self.signature = None
        self.prepared = self.failed = self.closing = self.closed = False
        self.failure = self.cleanup_error = None
        self.warmups = self.captures = self.replays = 0
        self.capture_attempts = self.replay_attempts = 0
        self.capture_s = 0.0
        self.allocated_before = self.allocated_after = None
        self.reserved_before = self.reserved_after = None
        self._lock = Lock()

    @contextmanager
    def _exclusive(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("V4 MoE graph cannot be used concurrently or recursively.")
        try:
            yield
        finally:
            self._lock.release()

    def _require_open(self):
        if self.closed or self.closing:
            raise RuntimeError("V4 MoE graph is closed or closing; no further execution is allowed.")
        if self.failed:
            raise RuntimeError(f"V4 MoE graph previously failed; discard the runtime: {self.failure}")

    def _check_inputs(self, hidden, input_ids):
        if (
            hidden.ndim != 2
            or hidden.shape[0] != 1
            or hidden.shape[1] <= 0
            or hidden.dtype != torch.bfloat16
            or hidden.device != self.device
            or input_ids is None
            or input_ids.shape != (1,)
            or input_ids.dtype not in (torch.int32, torch.int64)
            or input_ids.device != self.device
        ):
            raise ValueError("V4 MoE graph requires B1 BF16 hidden and one device int32/int64 token ID.")
        # Token values and incoming integer width are not a graph-cache key.
        # copy_ converts either supported width into the one int64 buffer.
        signature = (tuple(hidden.shape), hidden.dtype, hidden.device)
        if self.signature is not None and signature != self.signature:
            raise ValueError("V4 MoE graph input signature changed; no implicit recapture.")
        return signature

    def _current_stream(self):
        current = self.backend.current_stream(self.device)
        if self.stream_id is not None and current.npu_stream != self.stream_id:
            raise RuntimeError("V4 MoE graph requires its single owner stream.")
        capturing = getattr(self.backend, "is_current_stream_capturing", None)
        if capturing is not None and capturing():
            raise RuntimeError("V4 MoE graph does not support nested capture or replay inside another graph.")
        return current

    def _memory(self, name):
        method = getattr(self.backend, name, None)
        return int(method(self.device)) if method is not None else None

    def _latch(self, error):
        self.failed = True
        if self.failure is None:
            # Do not retain a traceback and its unbounded temporary Tensor owners.
            self.failure = f"{type(error).__name__}: {error}"

    def _check_outputs(self, outputs):
        if not isinstance(outputs, tuple) or len(outputs) != 2:
            raise ValueError("V4 MoE compute must return (output, device validity).")
        output, valid = outputs
        if (
            output.shape != self.hidden.shape
            or output.dtype != torch.bfloat16
            or output.device != self.device
            or valid.shape != ()
            or valid.dtype != torch.bool
            or valid.device != self.device
        ):
            raise ValueError("V4 MoE compute requires BF16 output matching hidden and scalar device bool validity.")
        return output, valid

    @torch.inference_mode()
    def prepare(self, compute, hidden, input_ids):
        """Warm up/capture once at startup, before any serving/timing window."""
        with self._exclusive():
            self._require_open()
            if self.prepared or self.graph is not None:
                raise RuntimeError("V4 MoE graph preparation is single-shot; no recapture.")
            signature = self._check_inputs(hidden, input_ids)
            current = self._current_stream()
            self.stream, self.stream_id, self.signature = current, current.npu_stream, signature
            self.owners += (compute,)
            started = perf_counter()
            try:
                self.allocated_before = self._memory("memory_allocated")
                self.reserved_before = self._memory("memory_reserved")
                self.hidden = hidden.clone(memory_format=torch.contiguous_format)
                self.input_ids = torch.empty((1,), device=self.device, dtype=torch.int64)
                self.input_ids.copy_(input_ids)
                for _ in range(GRAPH_WARMUPS):
                    self._check_outputs(compute(self.hidden, self.input_ids))
                    self.warmups += 1
                current.synchronize()
                # Keep even a partially captured graph alive until safe close.
                self.graph = self.backend.NPUGraph()
                self.capture_attempts += 1
                # No pool argument: each layer has its own private graph pool.
                # torch.npu.graph enters synchronization/allocator housekeeping;
                # this context is intentionally never entered by replay().
                with self.backend.graph(self.graph, stream=current):
                    self.output, self.valid = self._check_outputs(compute(self.hidden, self.input_ids))
                current.synchronize()
                self.captures += 1
                self.allocated_after = self._memory("memory_allocated")
                self.reserved_after = self._memory("memory_reserved")
                self.prepared = True
            except BaseException as error:
                self._latch(error)
                if hasattr(error, "add_note"):
                    error.add_note(
                        f"V4 MoE capture on owner stream {self.stream_id} failed; "
                        "no stream change or eager fallback was attempted."
                    )
                raise
            finally:
                self.capture_s = perf_counter() - started
        return self.snapshot()

    @torch.inference_mode()
    def replay(self, hidden, input_ids):
        with self._exclusive():
            self._require_open()
            if not self.prepared:
                raise RuntimeError("Prepare V4 MoE graph explicitly before decode; lazy capture is disabled.")
            self._check_inputs(hidden, input_ids)
            self._current_stream()
            try:
                self.hidden.copy_(hidden)
                self.input_ids.copy_(input_ids)
                self.replay_attempts += 1
                self.graph.replay()
                # Both results escape the private pool. Keeping an earlier
                # output/flag must not observe a later replay overwriting it.
                outputs = self.output.clone(), self.valid.clone()
                self.replays += 1
                return outputs
            except BaseException as error:
                self._latch(error)
                raise

    def close(self):
        """Fence before reset/release; failed fences retain every graph owner."""
        with self._exclusive():
            if self.closed:
                return
            self.closing = True
            try:
                if self.stream is not None:
                    self.stream.synchronize()
                if self.graph is not None:
                    self.graph.reset()
            except BaseException as error:
                self._latch(error)
                self.cleanup_error = f"{type(error).__name__}: {error}"
                raise
            self.graph = None
            self.output = self.valid = self.hidden = self.input_ids = None
            self.owners = ()
            self.prepared = False
            self.closed = True
            self.cleanup_error = None

    def snapshot(self):
        tensors = (self.hidden, self.input_ids, self.output, self.valid)
        return {
            "scope": "moe_decode",
            "captures": self.captures,
            "replays": self.replays,
            "warmups": self.warmups,
            "capture_attempts": self.capture_attempts,
            "replay_attempts": self.replay_attempts,
            "entries": int(self.graph is not None),
            "pool_count": int(self.graph is not None),
            "prepared": self.prepared,
            "failed": self.failed,
            "closed": self.closed,
            "closing": self.closing,
            "failure": self.failure,
            "cleanup_error": self.cleanup_error,
            "owner_stream": self.stream_id,
            "signature": {
                "hidden_shape": list(self.signature[0]) if self.signature is not None else None,
                "hidden_dtype": "torch.bfloat16",
                "device": str(self.device),
                "token_buffer_dtype": "torch.int64",
                "contract": self.contract,
            },
            "capture_s": self.capture_s,
            "allocated_before": self.allocated_before,
            "allocated_after": self.allocated_after,
            "reserved_before": self.reserved_before,
            "reserved_after": self.reserved_after,
            # Do not reset process-global peak accounting or label a cumulative
            # allocator peak as this graph's private capture peak.
            "capture_peak_reserved_bytes": None,
            "static_buffer_bytes": sum(t.numel() * t.element_size() for t in tensors if t is not None),
            "graph_functional_verified": False,
            "graph_performance_target_met": None,
            "full_model_graph_verified": False,
        }
