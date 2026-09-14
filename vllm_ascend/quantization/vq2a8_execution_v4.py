# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V1 arithmetic with an eagerly loaded, non-evictable direct-TP1 payload.

Only the expert lifetime changes. Row-wise preparation, routing, native V1
projection and explicitly selected V1 performance profiles remain inherited.
This module never imports the V3 preparation, layout or graph implementations.
"""

from __future__ import annotations

import math
import time

from vllm_ascend.quantization.vq2a8_execution import (
    AscendCVQ2TP1MoE,
    CachedVQ2TP1MoE,
    packed_cache_plan,
    synchronize_execution,
)
from vllm_ascend.quantization.vq2a8_repack import VQ2_DIRECT_TP1_FORMAT
from vllm_ascend.quantization.vq2a8_runtime import VQ2_TP1_TORCH_DTYPES

PRELOAD_PROGRESS_EXPERTS = 32
PRELOAD_PROGRESS_SECONDS = 5.0


def packed_resident_plan(layers, budget_bytes: int) -> dict:
    """Require every direct-format expert to fit before any device allocation.

    The existing V1 planner rounds each tensor separately to allocator units.
    Actual payload bytes are also reported, but do not replace that rounded
    budget. Roots, KV, temporary activations and headroom remain outside it.
    """
    layers = tuple(layers)
    for layer in layers:
        ids = layer.expert_ids
        if not ids or any(type(value) is not int or value < 0 for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("V4 residency requires a nonempty, unique integer expert inventory.")
    plan = packed_cache_plan(layers, budget_bytes)
    if not plan["all_experts_fit"]:
        raise ValueError(
            f"V4 full residency requires {plan['full_packed_bytes']} bytes, exceeds budget "
            f"{budget_bytes} bytes; no cache fallback."
        )
    layer_plans = {}
    for layer in layers:
        layer_plan = packed_cache_plan([layer], budget_bytes)
        # element_size is available without allocating NPU tensors. The base
        # planner already validated every shape and its expert axis.
        payload_bytes = sum(
            math.prod(layer.tensor_shapes[f"{kind}_{field}"]) * dtype.itemsize
            for kind in ("gate_up", "down")
            for field, dtype in VQ2_TP1_TORCH_DTYPES.items()
        )
        layer_plans[layer.layer_index] = {
            "experts": len(layer.expert_ids),
            "payload_bytes": payload_bytes,
            "planned_bytes": layer_plan["full_packed_bytes"],
        }
    return {
        **plan,
        "allocation": "eager_packed_only",
        "layout": "v1_packed",
        "layer_plans": layer_plans,
    }


def _tensor_identity(tensor) -> tuple:
    """Host metadata only: no tensor values, copies, device fences or kernels."""
    return (
        id(tensor),
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


class AscendCV4VQ2TP1MoE(AscendCVQ2TP1MoE):
    """Preload V1 packed experts once; all subsequent expert accesses are hits."""

    execution_policy = "ascendc_v4"

    def __init__(self, *args, **kwargs):
        # In particular, do not configure a batched profile here. vLLM startup
        # profiling must retain the existing V1 default arithmetic/row grouping.
        super().__init__(*args, **kwargs)
        self._resident_ready = False
        self._resident_failed = False
        self._resident_started = False
        self._resident_plan = None
        self._resident_expected_ids = frozenset(self.layer.expert_ids)
        self._resident_fingerprints = {}
        self._preload_loads = 0
        self._preload_h2d_bytes = 0
        self._preload_evictions = 0
        self._preload_elapsed_s = 0.0
        self._resident_cleanup_error = None

    def _require_ready(self):
        if not self._resident_ready or self._resident_failed:
            raise RuntimeError(f"V4 layer {self.layer_index} residency is not ready; no cache fallback.")

    def _fingerprint(self, expert_id):
        expert = self._cache[expert_id]
        if set(expert) != {"gate_up", "down"}:
            raise RuntimeError("V4 resident expert must contain both V1 projections.")
        result = [id(expert)]
        for kind in ("gate_up", "down"):
            payload, spec = expert[kind]
            if set(payload) != set(VQ2_TP1_TORCH_DTYPES):
                raise RuntimeError("V4 resident payload must retain the six direct-TP1 fields.")
            result.append((kind, id(payload), id(spec)))
            for field, dtype in VQ2_TP1_TORCH_DTYPES.items():
                tensor = payload[field]
                shape = tuple(self.layer.tensor_shapes[f"{kind}_{field}"][1:])
                if tensor.dtype != dtype or tuple(tensor.shape) != shape or not tensor.is_contiguous():
                    raise RuntimeError(f"V4 resident {kind}.{field} dtype, shape or layout changed.")
                if tensor.device != self.device:
                    raise RuntimeError(f"V4 resident {kind}.{field} is on the wrong device.")
                result.append((kind, field, _tensor_identity(tensor)))
        return tuple(result)

    def initialize_resident(self, *, budget_bytes: int):
        """Load this complete layer after the owner has planned the full model.

        The strict V1 CPU reader and its synchronous H2D path are reused,
        explicitly bypassing our hit-only lookup during this startup phase.
        Initialization is single-shot, including after a failed attempt.
        """
        if self._resident_started or self._resident_failed or self._cache:
            raise RuntimeError("V4 residency must initialize once from an empty cache.")
        if self.artifact.manifest.get("format") != VQ2_DIRECT_TP1_FORMAT:
            raise ValueError("V4 requires the V1 direct-TP1 artifact; packed-zN is not supported.")
        plan = packed_resident_plan([self.layer], budget_bytes)
        self._resident_plan = plan["layer_plans"][self.layer_index]
        self._resident_started = True
        self.cache_experts = len(self.layer.expert_ids)
        start = last_progress = time.perf_counter()
        try:
            if self.cache_loads or self.h2d_bytes or self.evictions:
                raise RuntimeError("V4 preload requires untouched startup load/H2D/eviction counters.")
            for position, expert_id in enumerate(self.layer.expert_ids, 1):
                CachedVQ2TP1MoE._get_expert(self, expert_id)
                self._resident_fingerprints[expert_id] = self._fingerprint(expert_id)
                now = time.perf_counter()
                if self.progress and (
                    position % PRELOAD_PROGRESS_EXPERTS == 0
                    or position == len(self.layer.expert_ids)
                    or now - last_progress >= PRELOAD_PROGRESS_SECONDS
                ):
                    print(
                        f"MODEL layer={self.layer_index} stage=v4_resident_payload "
                        f"loaded={position} total={len(self.layer.expert_ids)} "
                        f"elapsed_s={now - start:.3f} loaded_bytes={self._resident_bytes} "
                        f"planned_bytes={self._resident_plan['planned_bytes']}",
                        flush=True,
                    )
                    last_progress = now
            # Each base load already fences. This last fence also establishes
            # the layer-wide ready boundary before any native consumer runs.
            synchronize_execution(self.device)
            self._preload_loads = self.cache_loads
            self._preload_h2d_bytes = self.h2d_bytes
            self._preload_evictions = self.evictions
            self._preload_elapsed_s = time.perf_counter() - start
            self._resident_ready = True
            return self.check_resident_integrity()
        except BaseException:
            self._preload_loads = self.cache_loads
            self._preload_h2d_bytes = self.h2d_bytes
            self._preload_evictions = self.evictions
            self._preload_elapsed_s = time.perf_counter() - start
            try:
                self.abort_residency()
            except BaseException as cleanup_error:
                # Preserve the original load failure and the references held
                # by this runtime if device completion cannot be established.
                self._resident_cleanup_error = str(cleanup_error)
            raise

    def _get_expert(self, expert_id):
        self._require_ready()
        if type(expert_id) is not int or expert_id not in self._resident_expected_ids:
            raise ValueError(f"Layer {self.layer_index} has no stored expert {expert_id}.")
        expert = self._cache.get(expert_id)
        fingerprint = self._resident_fingerprints.get(expert_id)
        if expert is None or fingerprint is None or id(expert) != fingerprint[0]:
            raise RuntimeError(f"V4 resident expert {expert_id} is missing or replaced; no cache fallback.")
        self.cache_hits += 1
        return expert

    def forward(self, hidden, input_ids=None):
        self._require_ready()
        return super().forward(hidden, input_ids)

    def clear_cache(self):
        raise RuntimeError("V4 resident weights cannot be evicted; stop the runtime or abort initialization.")

    def abort_residency(self):
        """Permanently invalidate and release only after device work completes.

        Used for model-wide startup rollback, including layers that had already
        finished preloading. A failed fence retains references and propagates.
        """
        self._resident_failed = True
        self._resident_ready = False
        try:
            synchronize_execution(self.device)
        except BaseException as error:
            self._resident_cleanup_error = str(error)
            raise
        CachedVQ2TP1MoE.clear_cache(self)
        self._resident_fingerprints.clear()
        self._resident_cleanup_error = None

    def check_resident_integrity(self) -> dict:
        """Explicit request-boundary audit, deliberately not a forward-loop scan."""
        self._require_ready()
        expected = self._resident_expected_ids
        if set(self._cache) != expected or set(self._resident_fingerprints) != expected:
            raise RuntimeError("V4 resident expert inventory changed.")
        if self.cache_experts != len(expected):
            raise RuntimeError("V4 resident capacity changed.")
        if self.cache_loads != self._preload_loads or self.h2d_bytes != self._preload_h2d_bytes:
            raise RuntimeError("V4 expert loads or H2D occurred after initialization.")
        if self.evictions or self._preload_evictions:
            raise RuntimeError("V4 residency cannot contain evictions.")
        if self._preload_loads != len(expected):
            raise RuntimeError("V4 preload did not load every expert exactly once.")
        payload_bytes = sum(self._payload_bytes(expert) for expert in self._cache.values())
        if payload_bytes != self._resident_bytes or payload_bytes != self._resident_plan["payload_bytes"]:
            raise RuntimeError("V4 resident payload byte count changed.")
        if self.device.type != "cpu" and self._preload_h2d_bytes != payload_bytes:
            raise RuntimeError("V4 preload H2D byte count does not cover the resident payload.")
        for expert_id, fingerprint in self._resident_fingerprints.items():
            if self._fingerprint(expert_id) != fingerprint:
                raise RuntimeError(f"V4 resident expert {expert_id} storage changed.")
        return self.v4_report()

    def v4_report(self) -> dict:
        """Separate startup costs from weight-cache activity after initialization."""
        return {
            "execution_policy": self.execution_policy,
            "layer_index": self.layer_index,
            "layout": "v1_packed",
            "fallback_enabled": False,
            "ready": self._resident_ready and not self._resident_failed,
            "failed": self._resident_failed,
            "expected_experts": len(self._resident_expected_ids),
            "resident_experts": len(self._cache),
            "planned_bytes": self._resident_plan["planned_bytes"] if self._resident_plan else 0,
            "payload_bytes": self._resident_bytes,
            "preload_loads": self._preload_loads,
            "preload_h2d_bytes": self._preload_h2d_bytes,
            "preload_evictions": self._preload_evictions,
            "preload_elapsed_s": self._preload_elapsed_s,
            "post_init_loads": self.cache_loads - self._preload_loads,
            "post_init_h2d_bytes": self.h2d_bytes - self._preload_h2d_bytes,
            "post_init_evictions": self.evictions - self._preload_evictions,
            "cleanup_error": self._resident_cleanup_error,
        }
