# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in M=1 device routing with unchanged V1 packed projection arithmetic.

Multiple-token prefills retain V4's batched path. No graph, FWHT, expanded
expert weights, host route IDs or per-step pointer-table uploads are used by
the singleton path. Invalid device values are checked once at the model output
boundary, not once per layer. Native banks own their immutable tensor storage.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm_ascend.quantization.vq2a8_moe import route_vq2a8
from vllm_ascend.quantization.vq2a8_optimization import FastMoEState, OptimizationOptions
from vllm_ascend.quantization.vq2a8_reference import deepseek_v4_swiglu_reference
from vllm_ascend.quantization.vq2a8_runtime import VQ2_TP1_FIELDS

DEVICE_ROUTE_PRESET = "device_route_decode"
MAX_DEVICE_ROUTE_SLOTS = 6


def require_device_route_library():
    """Check the new ABI before loading model-sized weights; old libraries work
    unchanged when the explicit device-route option is absent.
    """
    try:
        return torch.classes.vq2a8_ascendc.ResidentBank
    except (RuntimeError, AttributeError) as error:
        raise RuntimeError(
            "V4 device routing needs ResidentBank. Rebuild tools/build_vq2a8_ascendc.py "
            "--soc Ascend950DT_9582 --build-dir build/vq2a8-ascendc-v4-device-route "
            "and select that libvq2a8_ascendc.so; the original V4 library is unchanged."
        ) from error


def initialize_device_route_banks(runtime):
    """Create small device pointer tables once, without copying expert payloads."""
    runtime._require_ready()
    runtime.check_resident_integrity()
    if getattr(runtime, "_device_route_banks", None) is not None:
        return
    if runtime.device.type != "npu":
        raise ValueError("V4 device routing requires NPU; no CPU fallback.")
    bank_type = require_device_route_library()
    count = runtime.config.num_experts
    expert_ids = tuple(runtime.layer.expert_ids)
    if not 1 <= runtime.config.top_k <= MAX_DEVICE_ROUTE_SLOTS or any(i >= count for i in expert_ids):
        raise ValueError("V4 device routing requires 1..6 routes and an in-range expert inventory.")
    # An artifact may contain a sparse subset (e.g. a hash layer). Raw router
    # IDs map to dense bank slots; absent experts stay invalid, never expert 0.
    with torch.device("cpu"):
        mapping = torch.full((count,), -1, dtype=torch.int64, device="cpu")
        for slot, expert_id in enumerate(expert_ids):
            mapping[expert_id] = slot
    banks = {"lookup": mapping.to(runtime.device), "metadata_bytes": mapping.numel() * mapping.element_size()}
    for kind in ("gate_up", "down"):
        entries = [runtime._cache[expert_id][kind] for expert_id in expert_ids]
        spec = entries[0][1]
        geometry = (spec.rows, spec.columns, spec.rht_true_columns, spec.rht_block_size)
        if any((s.rows, s.columns, s.rht_true_columns, s.rht_block_size) != geometry for _, s in entries):
            raise ValueError("Resident bank experts must share V1 projection/preparation geometry.")
        bank = bank_type(*[[p[field] for p, _ in entries] for field in VQ2_TP1_FIELDS])
        banks[kind] = (bank, spec)
        banks["metadata_bytes"] += bank.metadata()[3]
    # Publish only after both banks have been constructed. Owners survive
    # baseline/candidate switching, so A/B does not reload or duplicate weights.
    runtime._device_route_banks = banks


class DeviceRouteDecodeState(FastMoEState):
    """Keep the top-k slots on device, preserving duplicates and reduction order."""

    def __init__(self, runtime, *, profile=False):
        if getattr(runtime, "execution_policy", None) != "ascendc_v4":
            raise ValueError("Device-route decode is only supported by the V4 resident runtime.")
        runtime._require_ready()
        if getattr(runtime, "_device_route_banks", None) is None:
            raise ValueError("Initialize V4 device-route banks outside the forward/timing window first.")
        super().__init__(runtime, OptimizationOptions.preset("batched"), profile=profile)
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

    def _project(self, runtime, hidden, slots, kind):
        bank, spec = runtime._device_route_banks[kind]
        with self.scope("device_select"):
            weight_scale, weight_bias, signs, valid = bank.select(slots)
            self.retain((valid != 0).all())
            self.stats["device_select_calls"] += 1
        # Slot count is static host metadata, not device route data. Each row
        # still performs the original one-row RHT/bias GEMV and FP8 conversion.
        requests = [
            (
                hidden[i : i + 1],
                {"weight_scale": weight_scale[i], "weight_bias": weight_bias[i], "rht_sign": signs[i]},
                spec,
            )
            for i in range(slots.numel())
        ]
        with self.scope("preparation"):
            prepared = runtime._row_preparation.many(requests)
            self.stats["preparation_calls"] += 1
        quantized, scale, bias = (torch.cat(values).contiguous() for values in zip(*prepared))
        with self.scope("native_projection"):
            output, valid = bank.project(quantized, scale, bias, slots)
            self.retain((valid != 0).all() & torch.isfinite(output).all())
        runtime.native_calls += slots.numel()
        runtime.native_rows += slots.numel()
        runtime.native_launches += 1  # projection only, same coverage definition as V1
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
        with self.scope("route"):
            logits = F.linear(hidden.float(), runtime.root["gate.weight"])
            weights, ids = route_vq2a8(
                logits,
                runtime.config.top_k,
                renormalize=runtime.config.renormalize,
                correction_bias=runtime.root.get("gate.bias"),
                hash_table=runtime.root.get("gate.tid2eid"),
                input_ids=input_ids,
                validity=self.retain,
                device_only=True,
            )
            lookup = runtime._device_route_banks["lookup"]
            ids = ids.reshape(-1)
            in_range = (ids >= 0) & (ids < lookup.numel())
            mapped = lookup.index_select(0, ids.clamp(0, lookup.numel() - 1))
            slots = torch.where(in_range, mapped, -1).contiguous()
            self.retain((slots >= 0).all())
        with self.scope("gate_up"):
            gate = self._project(runtime, hidden.expand(slots.numel(), -1), slots, "gate_up")
        with self.scope("swiglu"):
            activation = deepseek_v4_swiglu_reference(gate, runtime.config.swiglu_limit)
        with self.scope("down"):
            values = self._project(runtime, activation, slots, "down")
        with self.scope("mix_shared"):
            result = (values.reshape(1, slots.numel(), hidden.shape[1]).float() * weights.unsqueeze(-1)).sum(1)
            result *= runtime.config.routed_scale
            if runtime.config.num_shared:
                result += runtime.shared(hidden).float()
            result = result.to(hidden.dtype)
            self.retain(torch.isfinite(result).all())
        runtime.native_experts += slots.numel()
        self.stats["singleton_forwards"] += 1
        self.stats["jobs"] += slots.numel()
        self.stats["rows"] += slots.numel()
        self.stats["windows"] += 1
        self.row_histogram[1] = self.row_histogram.get(1, 0) + slots.numel()
        return result
