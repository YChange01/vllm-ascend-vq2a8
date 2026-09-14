# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture a complete B1 MoE layer, including device routing and preparation.

Attention, KV updates, root linears outside MoE and sampling remain eager.
This is deliberately reported as a MoE decode graph, never a full-model graph.
Capture/warmup happens on the owner stream and must finish before timing.
"""

from threading import Lock

import torch

GRAPH_WARMUPS = 2


class MoEDecodeGraph:
    def __init__(self, device, *, backend=None):
        self.device = torch.device(device)
        if backend is None and self.device.type != "npu":
            raise ValueError("MoE decode graph requires an NPU.")
        self.backend = torch.npu if backend is None else backend
        self.graph = self.hidden = self.input_ids = self.output = None
        self.stream_id = self.signature = None
        self.captures = self.replays = 0
        self.failed = False
        self._lock = Lock()

    def run(self, compute, hidden, input_ids):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("MoE graph cannot be used concurrently or recursively.")
        try:
            return self._run(compute, hidden, input_ids)
        finally:
            self._lock.release()

    def _run(self, compute, hidden, input_ids):
        if self.failed:
            raise RuntimeError("MoE graph previously failed; discard the runtime.")
        if (
            hidden.ndim != 2
            or hidden.shape[0] != 1
            or hidden.dtype != torch.bfloat16
            or input_ids is None
            or input_ids.shape != (1,)
            or input_ids.dtype not in (torch.int32, torch.int64)
            or hidden.device != self.device
            or input_ids.device != self.device
        ):
            raise ValueError("MoE graph requires B1 BF16 hidden and one device integer token ID.")
        current = self.backend.current_stream(self.device)
        signature = (tuple(hidden.shape), hidden.dtype, input_ids.dtype)
        if self.stream_id is not None and current.npu_stream != self.stream_id:
            raise RuntimeError("MoE graph requires its single owner stream.")
        if self.signature is not None and signature != self.signature:
            raise ValueError("MoE graph input signature changed; no implicit recapture.")
        try:
            if self.graph is None:
                self.stream_id, self.signature = current.npu_stream, signature
                self.hidden, self.input_ids = hidden.clone(), input_ids.clone()
                # Warm up on the SAME stream: resident banks reject a side-stream
                # first use. This layer has no KV writes or RNG-dependent state.
                for _ in range(GRAPH_WARMUPS):
                    compute(self.hidden, self.input_ids)
                current.synchronize()
                graph = self.backend.NPUGraph()
                with self.backend.graph(graph, stream=current):
                    self.output = compute(self.hidden, self.input_ids)
                self.graph = graph
                self.captures += 1
            # Dynamic hash token IDs, non-hash routing and expert descriptors are
            # all read inside the graph; captured pointers never freeze a route.
            self.hidden.copy_(hidden)
            self.input_ids.copy_(input_ids)
            self.graph.replay()
            self.replays += 1
            # Subsequent layer/token replays must not overwrite a returned value.
            return self.output.clone()
        except Exception:
            self.failed = True
            raise

    def report(self):
        return {
            "scope": "moe_decode",
            "captures": self.captures,
            "replays": self.replays,
            "entries": int(self.graph is not None),
            "failed": self.failed,
            "full_model_graph_verified": False,
        }
