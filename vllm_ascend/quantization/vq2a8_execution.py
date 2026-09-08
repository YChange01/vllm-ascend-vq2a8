# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded cached execution with an explicit native AscendC projection policy.

The default cached policy retains the accepted M=1 kernel. Both policies keep
the same router and row-wise preparation; no dense expert matrix is stored.
"""

from __future__ import annotations

import json
import math
import time

import torch
import torch.nn.functional as F

from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_moe import VQ2TP1MoE, mix_vq2a8_routes
from vllm_ascend.quantization.vq2a8_reference import (
    deepseek_v4_swiglu_reference,
    prepare_repacked_vq2a8_activation_reference,
)
from vllm_ascend.quantization.vq2a8_runtime import VQ2_TP1_TORCH_DTYPES

GIB = 1024**3
ALLOCATION_GRANULARITY = 512
ASCENDC_MAX_ROWS = 32
ASCENDC_MAX_JOBS = 6


def synchronize_execution(device: torch.device) -> None:
    if device.type != "cpu":
        getattr(torch, device.type).synchronize(device)


def packed_cache_plan(layers, budget_bytes: int, *, expert_limit: int = 256) -> dict:
    """Header-only plan, with a common per-layer cap and rounded tensor sizes.

    Budget excludes roots, KV, workspaces and allocator headroom. Allocation is
    lazy. The reserve is still needed: this is not an allocator fragmentation
    bound or a guarantee against another process consuming HBM after planning.
    """
    if type(budget_bytes) is not int or budget_bytes < 0:
        raise ValueError("Packed cache budget must be a non-negative integer.")
    if type(expert_limit) is not int or not 1 <= expert_limit <= 256:
        raise ValueError("Expert cache limit must be in [1,256].")
    inventory = {}
    for layer in layers:
        size = 0
        for kind in ("gate_up", "down"):
            for field, dtype in VQ2_TP1_TORCH_DTYPES.items():
                shape = layer.tensor_shapes[f"{kind}_{field}"]
                if shape[0] != len(layer.expert_ids) or any(value <= 0 for value in shape):
                    raise ValueError("Invalid expert-axis shape in cache plan.")
                width = torch.empty((), dtype=dtype, device="cpu").element_size()
                size += math.ceil(math.prod(shape[1:]) * width / ALLOCATION_GRANULARITY) * ALLOCATION_GRANULARITY
        if layer.layer_index in inventory:
            raise ValueError("Duplicate layer in packed cache plan.")
        inventory[layer.layer_index] = (len(layer.expert_ids), size)
    if not inventory:
        raise ValueError("Packed cache plan requires at least one layer.")

    def required(cap):
        return sum(min(count, cap) * size for count, size in inventory.values())

    if required(1) > budget_bytes:
        raise ValueError(f"HBM budget cannot retain one packed expert per layer: need {required(1)} bytes.")
    cap = max(value for value in range(1, expert_limit + 1) if required(value) <= budget_bytes)
    return {
        "budget_bytes": budget_bytes,
        "planned_bytes": required(cap),
        "full_packed_bytes": sum(count * size for count, size in inventory.values()),
        "per_layer_cache_limit": cap,
        "layer_limits": {index: min(count, cap) for index, (count, _) in inventory.items()},
        "all_experts_fit": all(count <= cap for count, _ in inventory.values()),
        "allocation": "lazy_packed_only",
    }


def device_cache_budget(device, *, reserve_gib=16.0, budget_gib=0.0, memory_fraction=0.9) -> dict:
    """Sample after root load; never equate disk size with currently free HBM."""
    values = (reserve_gib, budget_gib, memory_fraction)
    if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
        raise ValueError("Cache budget values must be finite numbers.")
    if reserve_gib < 1 or budget_gib < 0 or not 0 < memory_fraction <= 1:
        raise ValueError("Invalid cache budget, reserve or memory fraction.")
    device = torch.device(device)
    if device.type not in ("cuda", "npu"):
        raise ValueError("Device cache budgeting requires CUDA or NPU.")
    backend = getattr(torch, device.type)
    synchronize_execution(device)
    free, total = backend.mem_get_info(device)
    allocated, reserved = backend.memory_allocated(device), backend.memory_reserved(device)
    # Idle allocator blocks are reusable by this process; count them once.
    reusable = free + max(0, reserved - allocated)
    available = max(0, min(reusable, int(total * memory_fraction) - allocated) - int(reserve_gib * GIB))
    requested = int(budget_gib * GIB)
    if requested > available:
        raise ValueError(f"Requested packed cache {requested} exceeds safe current budget {available} bytes.")
    return {
        "budget_bytes": requested or available,
        "free_bytes": free,
        "total_bytes": total,
        "allocated_bytes_at_plan": allocated,
        "reserved_bytes_at_plan": reserved,
        "reserve_bytes": int(reserve_gib * GIB),
        "memory_fraction": memory_fraction,
    }


class CachedVQ2TP1MoE(VQ2TP1MoE):
    """Reuse packed payloads without changing M=1 preparation or reduction."""

    measurement_mode = False

    def __init__(self, *args, progress=False, verbose_experts=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress = progress
        self.verbose_experts = verbose_experts
        self.evictions = 0
        self._resident_bytes = 0
        self.timing = dict.fromkeys(
            ("host_load_validate_s", "host_read_s", "host_validate_s", "h2d_s", "prepare_s", "packed_projection_s"), 0.0
        )
        self.projection_rows = 0
        self.prepare_batches = 0
        self.measurement_mode = False
        self.h2d_bytes = 0

    def _timing_sync(self):
        if not self.measurement_mode:
            synchronize_execution(self.device)

    def _emit(self, stage, **values):
        if self.progress and self.verbose_experts:
            fields = " ".join(f"{key}={value}" for key, value in values.items())
            print(f"MODEL layer={self.layer_index} stage={stage} {fields}", flush=True)

    @staticmethod
    def _payload_bytes(expert):
        return sum(t.numel() * t.element_size() for payload, _ in expert.values() for t in payload.values())

    def clear_cache(self):
        super().clear_cache()
        self._resident_bytes = 0

    def cache_stats(self):
        return {
            "resident_experts": len(self._cache),
            "resident_bytes": self._resident_bytes,
            "peak_packed_bytes": self.cache_peak_bytes,
            "loads": self.cache_loads,
            "hits": self.cache_hits,
            "evictions": self.evictions,
        }

    def _get_expert(self, expert_id):
        if expert_id not in self.layer.expert_ids:
            raise ValueError(f"Layer {self.layer_index} has no stored expert {expert_id}.")
        if expert_id in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(expert_id)
            return self._cache[expert_id]
        if len(self._cache) >= self.cache_experts:
            # Measurement removes timing fences, not payload lifetime fences.
            # Do not release weights until their last native consumer completes.
            if self.measurement_mode:
                synchronize_execution(self.device)
            _, evicted = self._cache.popitem(last=False)
            self._resident_bytes -= self._payload_bytes(evicted)
            del evicted
            self.evictions += 1
        self._emit("expert_load_start", expert=expert_id)
        start = time.perf_counter()
        host_timings = {}
        # Read/validate on CPU, once per cache miss, retaining strict checks.
        # Explicit CPU context is necessary under vLLM's NPU default device.
        with torch.device("cpu"):
            host = {
                kind: self.artifact.load_expert(self.layer_index, expert_id, kind, device="cpu", timings=host_timings)
                for kind in ("gate_up", "down")
            }
        host_s = time.perf_counter() - start
        start = time.perf_counter()
        expert = {
            kind: ({key: value.to(self.device, non_blocking=False) for key, value in payload.items()}, spec)
            for kind, (payload, spec) in host.items()
        }
        synchronize_execution(self.device)
        h2d_s = time.perf_counter() - start if self.device.type != "cpu" else 0.0
        self.timing["host_load_validate_s"] += host_s
        for key, value in host_timings.items():
            self.timing[key] += value
        self.timing["h2d_s"] += h2d_s
        self._cache[expert_id] = expert
        if self.device.type != "cpu":
            self.h2d_bytes = getattr(self, "h2d_bytes", 0) + self._payload_bytes(expert)
        self._resident_bytes += self._payload_bytes(expert)
        self.cache_peak_bytes = max(self.cache_peak_bytes, self._resident_bytes)
        self.cache_loads += 1
        if self.progress and self.verbose_experts:
            self._emit(
                "expert_load_done",
                expert=expert_id,
                host_s=f"{host_s:.3f}",
                read_s=f"{host_timings.get('host_read_s', 0.0):.3f}",
                validate_s=f"{host_timings.get('host_validate_s', 0.0):.3f}",
                h2d_s=f"{h2d_s:.3f}",
                resident=len(self._cache),
                limit=self.cache_experts,
                evictions=self.evictions,
            )
        return expert

    def _projection(self, hidden, payload, spec):
        if self.device.type == "cpu":
            return super()._projection(hidden, payload, spec)
        # Lazy import: CPU-only contracts must not import the Triton backend.
        from vllm_ascend.quantization.vq2a8_triton import vq2a8_tp1_m1_packed_gemm

        synchronize_execution(self.device)
        outputs = []
        # Preserve the accepted per-row GEMV/RHT geometry. Preparing a larger
        # batch can move values across FP8 rounding boundaries, so residency
        # optimization must not silently select that different numeric path.
        for row in hidden.split(1):
            start = time.perf_counter()
            if spec.columns != spec.rht_true_columns:
                row = F.pad(row, (0, spec.columns - spec.rht_true_columns))
            with torch.device("cpu"):
                quantized, scale, bias = prepare_repacked_vq2a8_activation_reference(
                    row, payload["weight_scale"], payload["weight_bias"], payload["rht_sign"], spec.rht_block_size
                )
            synchronize_execution(self.device)
            self.timing["prepare_s"] += time.perf_counter() - start
            self.prepare_batches += 1
            start = time.perf_counter()
            outputs.append(
                vq2a8_tp1_m1_packed_gemm(
                    quantized.contiguous(),
                    scale.contiguous(),
                    bias.contiguous(),
                    payload["packed_indices"],
                    payload["codebooks"],
                    payload["codebook_tile_ids"],
                )
            )
            synchronize_execution(self.device)
            self.timing["packed_projection_s"] += time.perf_counter() - start
            self.projection_rows += 1
        return torch.cat(outputs, dim=0)

    def expert(self, expert_id, hidden):
        if not (self.progress and self.verbose_experts):
            return super().expert(expert_id, hidden)
        self._emit("expert_start", expert=expert_id, tokens=hidden.shape[0], cached=expert_id in self._cache)
        start = time.perf_counter()
        result = super().expert(expert_id, hidden)
        self._emit("expert_done", expert=expert_id, elapsed_s=f"{time.perf_counter() - start:.3f}")
        return result

    @torch.inference_mode()
    def forward(self, hidden, input_ids=None):
        if self.measurement_mode:
            return self._forward(hidden, input_ids)
        synchronize_execution(self.device)
        start = time.perf_counter()
        before = dict(self.timing)
        cache_before = self.cache_stats()
        rows, batches = self.projection_rows, self.prepare_batches
        result = self._forward(hidden, input_ids)
        synchronize_execution(self.device)
        report = {
            "layer": self.layer_index,
            "tokens": hidden.shape[0],
            "elapsed_s": time.perf_counter() - start,
            **{name: value - before[name] for name, value in self.timing.items()},
            "projection_rows": self.projection_rows - rows,
            "prepare_batches": self.prepare_batches - batches,
            "cache_loads": self.cache_loads - cache_before["loads"],
            "cache_hits": self.cache_hits - cache_before["hits"],
            "evictions": self.evictions - cache_before["evictions"],
            "resident_bytes": self._resident_bytes,
            "execution_policy": getattr(self, "execution_policy", "cached"),
            "native_fp8_dot": False
            if self.device.type == "npu" and getattr(self, "execution_policy", "cached") != "ascendc"
            else None,  # native policy call coverage is not an ISA-verification claim
        }
        if self.progress:
            print("MODEL_MOE_TIMING " + json.dumps(report), flush=True)
        return result

    def _forward(self, hidden, input_ids):
        return super().forward(hidden, input_ids)


class AscendCVQ2TP1MoE(CachedVQ2TP1MoE):
    """Opt-in eager native projections; routing/cache/shared math is unchanged.

    Preparation retains the accepted one-row rounding geometry. Up to six
    independent expert projections share one native launch; their compressed
    weights remain separate. This is an eager path, not verified serving.
    """

    execution_policy = "ascendc"

    def __init__(self, artifact, layer_index, device, **kwargs):
        if torch.device(device).type != "npu":
            raise ValueError("AscendC expert execution requires NPU; no CPU/CUDA fallback.")
        super().__init__(artifact, layer_index, device, **kwargs)
        self.native_calls = 0
        self.native_rows = 0
        self.native_experts = 0
        self.native_launches = 0
        self.native_steps = []
        self.trace_native = False

    def reset_native_trace(self):
        self.native_calls = self.native_rows = self.native_experts = 0
        self.native_launches = 0
        self.native_steps = []
        self.trace_native = True

    def _projection(self, hidden, payload, spec):
        from vllm_ascend.quantization.vq2a8_ascendc import vq2a8_ascendc

        if self.device.type != "npu":
            raise ValueError("AscendC projections never fall back to CPU/CUDA.")
        # Lazily owned by this layer, not a process-global cache; no per-expert
        # transformed weight copies and no cached input-validity decisions.
        if not hasattr(self, "_row_preparation"):
            self._row_preparation = RowwiseVQ2A8Preparation()
        outputs = []
        for chunk in hidden.split(ASCENDC_MAX_ROWS):
            start = time.perf_counter()
            quantized, scale, bias = self._row_preparation.rows(chunk, payload, spec)
            self._timing_sync()
            self.timing["prepare_s"] += time.perf_counter() - start
            self.prepare_batches += chunk.shape[0]
            start = time.perf_counter()
            outputs.append(
                vq2a8_ascendc(
                    quantized,
                    scale,
                    bias,
                    payload["packed_indices"],
                    payload["codebooks"],
                    payload["codebook_tile_ids"],
                )
            )
            self._timing_sync()
            self.timing["packed_projection_s"] += time.perf_counter() - start
            self.projection_rows += chunk.shape[0]
            self.native_calls += 1
            self.native_launches += 1
            self.native_rows += chunk.shape[0]
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)

    def expert(self, expert_id, hidden):
        output = super().expert(expert_id, hidden)
        self.native_experts += 1  # both gate/up and down completed
        return output

    def _projections_many(self, requests):
        from vllm_ascend.quantization.vq2a8_ascendc import grouped_projection

        if self.device.type != "npu":
            raise ValueError("AscendC projections never fall back to CPU/CUDA.")
        if len(requests) == 1:
            return [self._projection(*requests[0])]
        if not hasattr(self, "_row_preparation"):
            self._row_preparation = RowwiseVQ2A8Preparation()
        start = time.perf_counter()
        with torch.device("cpu"):
            prepared = self._row_preparation.many(requests)
        self._timing_sync()
        rows = sum(hidden.shape[0] for hidden, _, _ in requests)
        self.timing["prepare_s"] += time.perf_counter() - start
        self.prepare_batches += rows
        inputs = [
            (*values, *(payload[name] for name in ("packed_indices", "codebooks", "codebook_tile_ids")))
            for values, (_, payload, _) in zip(prepared, requests)
        ]
        start = time.perf_counter()
        output = grouped_projection(inputs)
        self._timing_sync()
        self.timing["packed_projection_s"] += time.perf_counter() - start
        self.native_calls += len(requests)  # logical projections, not launches
        self.native_launches += 1
        self.native_rows += rows
        self.projection_rows += rows
        return output

    def _forward(self, hidden, input_ids):
        """Reuse the unchanged router, slot reduction and shared-expert math.

        The reference mixer requests experts in insertion order. A bounded
        callback computes up to six of those requests together and returns
        them to the original slot writer. No cross-token reduction is moved.
        """
        if self.token_chunk > ASCENDC_MAX_ROWS:
            # Do not silently change shared-expert GEMM geometry for callers
            # outside the bounded offline grouped path.
            return super()._forward(hidden, input_ids)
        weights, ids = self.route(hidden, input_ids)
        host_ids = ids.cpu().tolist()
        if not {index for row in host_ids for index in row}.issubset(self.layer.expert_ids):
            raise ValueError("Router selected experts missing from this artifact.")
        if not hidden.shape[0]:
            return torch.empty_like(hidden)
        outputs = []
        # Respect the original token chunk (including shared GEMM geometry).
        chunk_size = self.token_chunk
        for start in range(0, hidden.shape[0], chunk_size):
            chunk = hidden[start : start + chunk_size]
            route_ids = ids[start : start + chunk_size]
            plan = {}
            for token, token_ids in enumerate(host_ids[start : start + chunk_size]):
                for expert_id in token_ids:
                    plan.setdefault(expert_id, {}).setdefault(token, None)
            pending = list(plan)
            ready = {}

            def expert(expert_id, selected, *, ready=ready, pending=pending, chunk=chunk, plan=plan):
                if expert_id not in ready:
                    count = min(ASCENDC_MAX_JOBS, self.cache_experts, len(pending))
                    batch = pending[:count]
                    del pending[:count]
                    if not batch or batch[0] != expert_id:
                        raise RuntimeError("Grouped expert schedule disagrees with the reference mixer.")
                    verbose = self.progress and self.verbose_experts
                    if verbose:
                        started = time.perf_counter()
                        self._emit("expert_group_start", experts=",".join(map(str, batch)), jobs=len(batch))
                    # Holding <=cache_experts jobs cannot pin weights evicted
                    # by this batch. Payload owners are released before refill.
                    payloads = [self._get_expert(index) for index in batch]
                    rows = [selected] + [
                        chunk.index_select(0, torch.tensor(list(plan[index]), device=chunk.device, dtype=torch.int64))
                        for index in batch[1:]
                    ]
                    gates = self._projections_many([(row, *p["gate_up"]) for row, p in zip(rows, payloads)])
                    activated = [deepseek_v4_swiglu_reference(gate, self.config.swiglu_limit) for gate in gates]
                    values = self._projections_many([(row, *p["down"]) for row, p in zip(activated, payloads)])
                    ready.update(zip(batch, values))
                    self.native_experts += len(batch)  # both projections completed
                    if verbose:
                        self._emit(
                            "expert_group_done", jobs=len(batch), elapsed_s=f"{time.perf_counter() - started:.3f}"
                        )
                return ready.pop(expert_id)

            outputs.append(
                mix_vq2a8_routes(
                    chunk,
                    weights[start : start + chunk_size],
                    route_ids,
                    expert,
                    routed_scale=self.config.routed_scale,
                    shared=self.shared if self.config.num_shared else None,
                )
            )
            if pending or ready:
                raise RuntimeError("Grouped expert schedule did not consume every route.")
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)

    @torch.inference_mode()
    def forward(self, hidden, input_ids=None):
        before = self.native_calls, self.native_rows, self.native_experts, self.native_launches
        output = super().forward(hidden, input_ids)
        if self.trace_native:
            if len(self.native_steps) >= 128:
                raise ValueError("Native model trace exceeded the offline step bound.")
            self.native_steps.append(
                {
                    "tokens": hidden.shape[0],
                    "projection_calls": self.native_calls - before[0],
                    "projection_rows": self.native_rows - before[1],
                    "expert_calls": self.native_experts - before[2],
                    "kernel_launches": self.native_launches - before[3],
                }
            )
        return output
