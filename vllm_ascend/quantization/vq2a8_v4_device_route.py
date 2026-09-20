# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device routing and captured MoE compute for resident V4/v2 experts.

Multi-token prefills retain bounded host route scheduling. Singleton routing
stays on device; invalid values are consumed at the model output boundary.
Native banks own their immutable tensor storage for the graph lifetime.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm_ascend.quantization.vq2a8_moe import route_vq2a8
from vllm_ascend.quantization.vq2a8_optimization import FastMoEState, OptimizationOptions
from vllm_ascend.quantization.vq2a8_reference import deepseek_v4_swiglu_reference

DEVICE_ROUTE_PRESET = "device_route_decode"
MAX_DEVICE_ROUTE_SLOTS = 6


def initialize_device_route_banks(runtime):
    """Create small device pointer tables once, without copying expert payloads."""
    runtime._require_ready()
    runtime.check_resident_integrity()
    if getattr(runtime, "_device_route_banks", None) is not None:
        return
    runtime._device_route_banks = create_device_route_banks(runtime)


def create_device_route_banks(runtime):
    """Build metadata on the current stream, sharing every resident payload.

    Unlike initialize_device_route_banks, this does not publish or replace the
    eager banks. Graph capture owns a separate metadata-only bank on its own
    managed NPU stream; native construction-stream guards remain unchanged.
    Payload producer readiness must be established by the caller first.
    """
    runtime._require_ready()
    return runtime._create_device_route_banks()


def _make_layer_validity(runtime):
    from vllm_ascend.quantization.vq2a8_validity_fused import FusedLayerValidity

    if runtime.v4_validity_mode != "fused_vectorized":
        raise ValueError("Only fused_vectorized validity is supported.")
    return FusedLayerValidity()


def _resident_projection(runtime, bank, spec, preparation, hidden, slots, retain, raw_statuses):
    """Retain fresh select/input/project statuses in the original order."""
    activation, scale, bias = preparation.packed_resident(
        bank, hidden, slots, spec, validity=retain, raw_statuses=raw_statuses
    )
    output, valid = runtime.project_v4_prepared(bank, activation, scale, bias, slots)
    raw_statuses.append(valid)
    return output


def _make_route_mapping(runtime):
    from vllm_ascend.quantization.vq2a8_route_mapping import FusedRouteMapping

    if runtime.v4_route_mapping != "fused":
        raise ValueError("Only fused route mapping is supported.")
    return FusedRouteMapping()


class DeviceRouteDecodeState(FastMoEState):
    """Keep the top-k slots on device, preserving duplicates and reduction order."""

    def __init__(self, runtime, *, profile=False):
        if getattr(runtime, "execution_policy", None) != "ascendc_v4":
            raise ValueError("Device-route decode is only supported by the V4 resident runtime.")
        runtime._require_ready()
        if getattr(runtime, "_device_route_banks", None) is None:
            raise ValueError("Initialize V4 device-route banks outside the forward/timing window first.")
        super().__init__(runtime, OptimizationOptions(), profile=profile)
        self._layer_validity = _make_layer_validity(runtime)
        self._route_mapping = _make_route_mapping(runtime)
        self.stats.update(singleton_forwards=0, batched_prefill_forwards=0, device_select_calls=0)

    def report(self):
        return {
            **super().report(),
            "preset": DEVICE_ROUTE_PRESET,
            "singleton_route_host_reads": 0,
            "singleton_descriptor_h2d_bytes": 0,
            "scope": "M=1 device routing; multi-token prefill retains batched V4",
            "device_execution_verified": False,  # counters alone are not NPU acceptance
        }

    def _project(self, runtime, hidden, slots, kind, raw_statuses):
        bank, spec = runtime._device_route_banks[kind]
        with self.scope("resident_projection"):
            output = _resident_projection(
                runtime, bank, spec, runtime._row_preparation, hidden, slots, self.retain, raw_statuses
            )
        self.stats["preparation_calls"] += 1
        self.stats["fused_select_sign_calls"] = self.stats.get("fused_select_sign_calls", 0) + 1
        runtime.native_calls += slots.numel()
        runtime.native_rows += slots.numel()
        runtime.native_launches += 1
        runtime.projection_rows += slots.numel()
        runtime.prepare_batches += slots.numel()
        return output

    def forward(self, runtime, hidden, input_ids):
        if hidden.ndim != 2 or hidden.shape[0] != 1:
            self.stats["batched_prefill_forwards"] += 1
            return super().forward(runtime, hidden, input_ids)
        if (
            hidden.shape[1] != runtime.config.hidden_size
            or hidden.dtype != torch.bfloat16
            or hidden.device != runtime.device
        ):
            raise ValueError("Device-route decode requires BF16 [1,hidden_size] on the resident device.")
        raw_statuses = []
        route_flags = []
        retain_route = route_flags.append
        with self.scope("route"):
            logits = F.linear(hidden.float(), runtime.root["gate.weight"])
            weights, ids = route_vq2a8(
                logits,
                runtime.config.top_k,
                renormalize=runtime.config.renormalize,
                correction_bias=runtime.root.get("gate.bias"),
                hash_table=runtime.root.get("gate.tid2eid"),
                input_ids=input_ids,
                validity=retain_route,
                device_only=True,
            )
            lookup = runtime._device_route_banks["lookup"]
            ids = ids.reshape(-1)
            slots, mapped_valid = self._route_mapping(ids, lookup)
            retain_route(mapped_valid)
        with self.scope("gate_up"):
            gate = self._project(runtime, hidden.expand(slots.numel(), -1), slots, "gate_up", raw_statuses)
        with self.scope("swiglu"):
            activation = deepseek_v4_swiglu_reference(gate, runtime.config.swiglu_limit)
        with self.scope("down"):
            values = self._project(runtime, activation, slots, "down", raw_statuses)
        with self.scope("mix_shared"):
            result = (values.reshape(1, slots.numel(), hidden.shape[1]).float() * weights.unsqueeze(-1)).sum(1)
            result *= runtime.config.routed_scale
            if runtime.config.num_shared:
                result += runtime.shared(hidden).float()
            result = result.to(hidden.dtype)
            self.retain(self._layer_validity(raw_statuses, [gate, values, result], route_flags))
        runtime.native_experts += slots.numel()
        self.stats["singleton_forwards"] += 1
        self.stats["jobs"] += slots.numel()
        self.stats["rows"] += slots.numel()
        self.stats["windows"] += 1
        self.row_histogram[1] = self.row_histogram.get(1, 0) + slots.numel()
        return result


class DeviceRouteGraphCompute:
    """Fixed-structure MoE with one fresh device validity output.

    Construction is startup-only. Each projection owns a prebuilt RHT constant
    even when gate/up and down use different blocks. __call__ does not mutate
    the eager optimization state's validity, counters, or preparation cache.
    All bank/root owners remain reachable for the graph's entire lifetime.
    """

    def __init__(self, runtime, *, banks=None):
        # Startup-only import keeps the optional plan out of eager routing.
        from vllm_ascend.quantization.vq2a8_runtime_guard import (
            RUNTIME_GUARD_MODES,
            PlannedRuntimeGuard,
        )

        if getattr(runtime, "execution_policy", None) != "ascendc_v4":
            raise ValueError("V4 graph compute requires the V4 resident runtime.")
        runtime._require_ready()
        if not isinstance(getattr(runtime, "_optimization", None), DeviceRouteDecodeState):
            raise ValueError("Configure device_route_decode before constructing V4 graph compute.")
        if getattr(runtime, "_device_route_banks", None) is None:
            raise ValueError("Initialize device-route banks before constructing V4 graph compute.")
        self.runtime = runtime
        self.root = dict(runtime.root)
        self.banks = dict(runtime._device_route_banks if banks is None else banks)
        self.config = runtime.config
        self._swiglu_mode = runtime.v4_swiglu_mode
        if self._swiglu_mode != "torch":
            raise ValueError("Only the accepted Torch SwiGLU is supported.")
        self._runtime_guard_mode = runtime.v4_runtime_guard
        if self._runtime_guard_mode not in RUNTIME_GUARD_MODES:
            raise ValueError("Only the planned runtime guard is supported.")
        self._layer_validity = _make_layer_validity(runtime)
        self._route_mapping = _make_route_mapping(runtime)
        self.preparations = {}
        geometries = {}
        for kind in ("gate_up", "down"):
            _, spec = self.banks[kind]
            preparation = runtime.make_v4_preparation(compact=True)
            preparation.prepare_for_graph(runtime.device, spec.rht_block_size)
            self.preparations[kind] = preparation
            geometries[kind] = [spec.rows, spec.columns, spec.rht_true_columns, spec.rht_block_size]
        self.signature = {
            "layer": getattr(runtime, "layer_index", None),
            "compute_backend": getattr(runtime, "v4_compute_backend", "v1"),
            "activation_preparation": getattr(runtime, "v4_activation_preparation", "rowwise"),
            "activation_reorder": getattr(runtime, "v4_activation_reorder", "scalar"),
            "b1_schedule": getattr(runtime, "v4_b1_schedule", "baseline"),
            "swiglu_mode": self._swiglu_mode,
            "validity_mode": getattr(runtime, "v4_validity_mode", "torch"),
            "route_mapping": getattr(runtime, "v4_route_mapping", "torch"),
            "select_sign": getattr(runtime, "v4_select_sign", "separate"),
            "activation_tail": getattr(runtime, "v4_activation_tail", "torch"),
            "runtime_guard": self._runtime_guard_mode,
            "top_k": self.config.top_k,
            "hidden_size": self.config.hidden_size,
            "hash_route": self.root.get("gate.tid2eid") is not None,
            "num_shared": self.config.num_shared,
            "renormalize": self.config.renormalize,
            "routed_scale": self.config.routed_scale,
            "swiglu_limit": self.config.swiglu_limit,
            "projection_geometry": geometries,
        }
        self._runtime_guard_plan = PlannedRuntimeGuard(runtime)

    def check_runtime_contract(self, runtime):
        try:
            self._runtime_guard_plan.check(runtime)
        except (AttributeError, KeyError, TypeError) as error:
            raise RuntimeError(
                "V4 MoE graph runtime/root/geometry signature changed; no implicit recapture."
            ) from error

    def _project(self, hidden, slots, kind, retain, raw_statuses):
        bank, spec = self.banks[kind]
        return _resident_projection(
            self.runtime, bank, spec, self.preparations[kind], hidden, slots, retain, raw_statuses
        )

    @torch.inference_mode()
    def __call__(self, hidden, input_ids):
        if (
            hidden.shape != (1, self.config.hidden_size)
            or hidden.dtype != torch.bfloat16
            or hidden.device != self.runtime.device
            or input_ids is None
            or input_ids.shape != (1,)
            or input_ids.dtype not in (torch.int32, torch.int64)
            or input_ids.device != hidden.device
        ):
            raise ValueError("V4 graph compute requires BF16 B1 hidden and one device integer token ID.")
        # This bounded list exists only during compute/capture; no prior flag
        # is an input, so a later valid replay cannot replay stale validity.
        flags = []
        raw_statuses = []
        logits = F.linear(hidden.float(), self.root["gate.weight"])
        weights, ids = route_vq2a8(
            logits,
            self.config.top_k,
            renormalize=self.config.renormalize,
            correction_bias=self.root.get("gate.bias"),
            hash_table=self.root.get("gate.tid2eid"),
            input_ids=input_ids,
            validity=flags.append,
            device_only=True,
        )
        lookup = self.banks["lookup"]
        ids = ids.reshape(-1)
        slots, mapped_valid = self._route_mapping(ids, lookup)
        flags.append(mapped_valid)
        gate = self._project(hidden.expand(slots.numel(), -1), slots, "gate_up", flags.append, raw_statuses)
        activation = deepseek_v4_swiglu_reference(gate, self.config.swiglu_limit)
        values = self._project(activation, slots, "down", flags.append, raw_statuses)
        result = (values.reshape(1, slots.numel(), hidden.shape[1]).float() * weights.unsqueeze(-1)).sum(1)
        result *= self.config.routed_scale
        if self.config.num_shared:
            result += self.runtime.shared(hidden).float()
        result = result.to(hidden.dtype)
        return result, self._layer_validity(raw_statuses, [gate, values, result], flags)
