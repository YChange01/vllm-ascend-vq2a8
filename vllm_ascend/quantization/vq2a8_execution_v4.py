# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eagerly loaded, non-evictable expert ownership for the V4/v2 runtime."""

from __future__ import annotations

import time

from vllm_ascend.quantization.vq2a8_execution import (
    AscendCVQ2TP1MoE,
    CachedVQ2TP1MoE,
    synchronize_execution,
)
from vllm_ascend.quantization.vq2a8_repack import VQ2_DIRECT_TP1_FORMAT

PRELOAD_PROGRESS_EXPERTS = 32
PRELOAD_PROGRESS_SECONDS = 5.0


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
    """Preload packed experts once; all subsequent expert accesses are hits."""

    execution_policy = "ascendc_v4"

    @staticmethod
    def residency_plan(layers, budget_bytes):
        raise NotImplementedError("The concrete resident layout must provide its own planner.")

    def __init__(self, *args, **kwargs):
        # Do not configure a batched profile here. Startup profiling retains
        # the accepted initial eager arithmetic and row grouping.
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
        self._device_route_banks = None
        self._v4_decoder_compute = None
        self._v4_decoder_capture = False
        self._v4_decoder_valid = None
        self._v4_decoder_graph_owner = None

    def _require_ready(self):
        if not self._resident_ready or self._resident_failed:
            raise RuntimeError(f"V4 layer {self.layer_index} residency is not ready; no cache fallback.")

    def _fingerprint(self, expert_id):
        raise NotImplementedError("The concrete resident layout must provide its own storage guard.")

    def _validate_resident_artifact(self):
        if self.artifact.manifest.get("format") != VQ2_DIRECT_TP1_FORMAT:
            raise ValueError("V4 requires the V1 direct-TP1 artifact; packed-zN is not supported.")

    def initialize_resident(self, *, budget_bytes: int):
        """Load this complete layer after the owner has planned the full model.

        The strict artifact CPU reader and synchronous H2D path are reused,
        explicitly bypassing our hit-only lookup during this startup phase.
        Initialization is single-shot, including after a failed attempt.
        """
        if self._resident_started or self._resident_failed or self._cache:
            raise RuntimeError("V4 residency must initialize once from an empty cache.")
        self._validate_resident_artifact()
        plan = self.residency_plan([self.layer], budget_bytes)
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
        if self._v4_decoder_capture:
            if self._v4_decoder_compute is None:
                raise RuntimeError("Decoder capture requires pure MoE compute without nested MoE graphs.")
            output, self._v4_decoder_valid = self._v4_decoder_compute(hidden, input_ids)
            return output
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
        # The optional native banks strongly own the same payload storage.
        # Release them only after the fence above, never while queued work runs.
        if self._v4_decoder_graph_owner is not None:
            self._v4_decoder_graph_owner.close()
            self._v4_decoder_graph_owner = None
        self._device_route_banks = None
        self._v4_decoder_compute = None
        self._v4_decoder_valid = None
        self._optimization_states = {}
        self._optimization = None
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
