# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in all-resident B1 device dispatch using V1 weights and rowwise math.

This is not a full-model graph implementation. Exact preparation still launches
PyTorch pointwise/GEMV operations and allocates intermediates. Prefill explicitly
uses the legacy eager grouped ABI in the separate V3 library. Initialization,
immutability checks and validity reporting belong outside measured decode.
"""

import math
from types import MappingProxyType

import torch
import torch.nn.functional as F

from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_ascendc_v3 import (
    CONSTANT_WORDS,
    JOB_WORDS,
    grouped_projection_out,
    grouped_projection_v3,
    make_constants,
)
from vllm_ascend.quantization.vq2a8_execution import ALLOCATION_GRANULARITY, AscendCVQ2TP1MoE, synchronize_execution
from vllm_ascend.quantization.vq2a8_moe import route_vq2a8
from vllm_ascend.quantization.vq2a8_optimization import FastMoEState, OptimizationOptions
from vllm_ascend.quantization.vq2a8_reference import VQ2_FP8_MIN_SCALE, deepseek_v4_swiglu_reference
from vllm_ascend.quantization.vq2a8_runtime import VQ2_TP1_TORCH_DTYPES

KINDS = ("gate_up", "down")
POINTER_FIELDS = ("packed_indices", "codebooks", "codebook_tile_ids")
TRANSFORM_FIELDS = ("weight_scale", "weight_bias", "rht_sign")
ELEMENT_BYTES = dict(packed_indices=4, codebooks=1, codebook_tile_ids=1, weight_scale=4, weight_bias=4, rht_sign=1)


def _rounded(size):
    return math.ceil(size / ALLOCATION_GRANULARITY) * ALLOCATION_GRANULARITY


def resident_plan(layers, budget_bytes, *, top_k=6):
    """Header-only device-storage plan; budget excludes roots, KV and scratch.

    Transform metadata is loaded directly into its final bank, never stacked
    from duplicate device copies. Payload estimates include allocator rounding;
    eager preparation/operator scratch, fragmentation and graph pools still
    require the caller's separate reserve. The plan never reduces expert count.
    """
    if type(budget_bytes) is not int or budget_bytes <= 0 or type(top_k) is not int or not 1 <= top_k <= 6:
        raise ValueError("V3 requires an explicit positive byte budget and top_k in [1,6].")
    plans = {}
    for layer in layers:
        if layer.layer_index in plans or not layer.expert_ids or len(set(layer.expert_ids)) != len(layer.expert_ids):
            raise ValueError("Invalid or duplicate resident layer/expert inventory.")
        experts = len(layer.expert_ids)
        jobs = 1 if experts == 1 else top_k
        payload_bytes = workspace_bytes = 0
        for kind in KINDS:
            spec = layer.specs[kind]
            n, k = spec.rows, spec.columns
            if not (
                0 < n <= 65536
                and n % 32 == 0
                and 0 < k <= 65536
                and k % 512 == 0
                and spec.rht_block_size == 128
                and 0 < spec.rht_true_columns <= k
            ):
                raise ValueError("V3 requires supported V1 N/K geometry and RHT128.")
            for field, element_size in ELEMENT_BYTES.items():
                shape = layer.tensor_shapes[f"{kind}_{field}"]
                if shape[0] != experts:
                    raise ValueError("Resident header expert axis mismatch.")
                row_bytes = math.prod(shape[1:]) * element_size
                payload_bytes += _rounded(experts * row_bytes)
            workspace_bytes += sum(
                _rounded(size)
                for size in (
                    experts * 3 * 8,  # pointer bank
                    jobs * 3 * 8,  # selected pointers
                    jobs * k * 2,  # padded BF16 input
                    jobs * k,  # FP8 input
                    jobs * 4,  # activation scale
                    jobs * 4,  # bias correction
                    jobs * n * 2,  # projection output
                    jobs * JOB_WORDS * 8,
                    jobs * k * 4,  # selected weight scale
                    jobs * k * 4,  # selected weight bias
                    jobs * k,  # selected sign
                )
            )
        # One constants table and one H128 FP32 matrix per runtime.
        # The ID lookup is bounded by the router expert domain (<=256).
        workspace_bytes += _rounded(CONSTANT_WORDS * 4) + _rounded(128 * 128 * 4) + _rounded(256 * 8)
        plans[layer.layer_index] = dict(
            experts=experts,
            jobs=jobs,
            payload_bytes=payload_bytes,
            workspace_bytes=workspace_bytes,
            planned_bytes=payload_bytes + workspace_bytes,
        )
    if not plans:
        raise ValueError("Resident planning requires at least one layer.")
    totals = {
        key: sum(plan[key] for plan in plans.values()) for key in ("payload_bytes", "workspace_bytes", "planned_bytes")
    }
    if totals["planned_bytes"] > budget_bytes:
        raise ValueError(
            f"V3 full residency requires {totals['planned_bytes']} bytes, "
            f"exceeds budget {budget_bytes}; no cache fallback."
        )
    return dict(
        **totals,
        budget_bytes=budget_bytes,
        layer_plans=plans,
        all_experts_fit=True,
        full_packed_bytes=totals["payload_bytes"],
        per_layer_cache_limit=max(plan["experts"] for plan in plans.values()),
        layer_limits={index: plan["experts"] for index, plan in plans.items()},
        scope="packed_payload_and_persistent_workspace_only_separate_scratch_reserve_required",
    )


class ResidentProjectionWorkspace:
    """One M1 bank/workspace, with borrowed output consumed on the owner stream."""

    def __init__(self, payloads, spec, jobs, constants, preparation, *, banks, launcher=grouped_projection_out):
        self.spec, self.jobs, self.constants = spec, jobs, constants
        self.preparation, self.launcher = preparation, launcher
        first = payloads[0]
        self.device = first["weight_scale"].device
        self.n, self.k = spec.rows, spec.columns
        self.tiles = first["codebooks"].shape[0]
        self.payloads = tuple(payloads)
        # These are views into the final banks allocated by initialize_resident.
        self.banks = banks
        if any(self.banks[field].shape != (len(payloads), self.k) for field in TRANSFORM_FIELDS):
            raise ValueError("Resident metadata must be views into owned final expert banks.")
        self.selected = {
            field: torch.empty((jobs, self.k), device=self.device, dtype=first[field].dtype)
            for field in TRANSFORM_FIELDS
        }
        self.hidden = torch.empty((jobs, self.k), device=self.device, dtype=torch.bfloat16)
        self.x = torch.empty((jobs, self.k), device=self.device, dtype=torch.float8_e4m3fn)
        self.scale = torch.empty(jobs, device=self.device, dtype=torch.float32)
        self.bias = torch.empty(jobs, device=self.device, dtype=torch.float32)
        self.output = torch.empty((jobs, self.n), device=self.device, dtype=torch.bfloat16)
        self.pointer_bank = torch.tensor(
            [[payload[field].data_ptr() for field in POINTER_FIELDS] for payload in payloads],
            dtype=torch.int64,
            device="cpu",
        ).to(self.device)
        self.selected_pointers = torch.empty((jobs, 3), device=self.device, dtype=torch.int64)
        records = []
        for job in range(jobs):
            records.append(
                [
                    self.x[job].data_ptr(),
                    self.scale[job:].data_ptr(),
                    self.bias[job:].data_ptr(),
                    0,
                    0,
                    0,
                    self.output[job].data_ptr(),
                    1,
                    self.n,
                    self.k,
                    self.tiles,
                    0,
                ]
            )
        self.descriptors = torch.tensor(records, dtype=torch.int64, device="cpu").to(self.device)
        self.owners = (
            (
                self.hidden,
                self.x,
                self.scale,
                self.bias,
                self.output,
                self.pointer_bank,
                self.selected_pointers,
            )
            + tuple(self.banks.values())
            + tuple(self.selected.values())
        )

    def project(self, hidden, slots, *, pipeline=False):
        if hidden.shape not in ((1, self.spec.rht_true_columns), (self.jobs, self.spec.rht_true_columns)):
            raise ValueError("Resident decode preparation requires one input row or one row per fixed job.")
        if (
            hidden.device != self.device
            or hidden.dtype != torch.bfloat16
            or slots.shape != (self.jobs,)
            or slots.dtype != torch.int64
            or slots.device != self.device
        ):
            raise ValueError("Resident decode input/slot metadata mismatch.")
        self.hidden[:, : self.spec.rht_true_columns].copy_(hidden)
        if self.spec.rht_true_columns != self.k:
            self.hidden[:, self.spec.rht_true_columns :].zero_()
        for field in TRANSFORM_FIELDS:
            torch.index_select(self.banks[field], 0, slots, out=self.selected[field])
        torch.index_select(self.pointer_bank, 0, slots, out=self.selected_pointers)
        self.descriptors[:, 3:6].copy_(self.selected_pointers)
        quantized, scale, bias = self._prepare_selected()
        self.x.copy_(quantized)
        self.scale.copy_(scale)
        self.bias.copy_(bias)
        self.launcher(
            self.descriptors,
            self.constants,
            self.owners,
            jobs=self.jobs,
            m=1,
            n=self.n,
            k=self.k,
            tiles=self.tiles,
            pipeline=pipeline,
        )
        return self.output

    def _prepare_selected(self):
        """V1 many() arithmetic with already-gathered, contiguous metadata."""
        x = self.hidden.float()
        weight_scale, weight_bias, sign = (self.selected[field] for field in TRANSFORM_FIELDS)
        valid = torch.isfinite(x).all() & torch.isfinite(weight_scale).all() & torch.isfinite(weight_bias).all()
        valid = valid & ((sign == -1) | (sign == 1)).all()
        self.preparation._validate(valid)
        block = self.spec.rht_block_size
        self.preparation._ensure_hadamard(self.device, block)
        signed = x.reshape(-1, self.k // block, block) * sign.float().reshape(-1, self.k // block, block)
        # Exactly the original one-row matmul and bias GEMV geometry/order.
        rotated = [(row @ self.preparation._hadamard).reshape(1, self.k) for row in signed.split(1)]
        bias = torch.cat([row @ weight_bias[index] for index, row in enumerate(rotated)])
        transformed = torch.cat(rotated) * weight_scale
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = torch.clamp(transformed.abs().amax(dim=-1) / fp8_max, min=VQ2_FP8_MIN_SCALE)
        quantized = torch.clamp(transformed / scale.unsqueeze(-1), -fp8_max, fp8_max).to(torch.float8_e4m3fn)
        return quantized, scale, bias


class AscendCV3VQ2TP1MoE(AscendCVQ2TP1MoE):
    execution_policy = "ascendc_v3"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resident_ready = False
        self._resident_failed = False
        self._resident_valid = None
        self._resident_stream = None
        self._resident_busy = False
        self._resident_workspaces = {}
        self._resident_plan = None
        self.decode_device_calls = self.prefill_legacy_calls = 0
        self._v3_prefill_state = FastMoEState(self, OptimizationOptions.preset("batched"))
        self._optimization = self._v3_prefill_state

    def _retain_valid(self, valid):
        self._resident_valid = valid if self._resident_valid is None else self._resident_valid & valid

    def initialize_resident(self, *, budget_bytes):
        """Initialize once after root loading/budget admission, never inside decode."""
        if self._resident_ready or self._resident_failed or self._cache:
            raise RuntimeError("V3 residency requires a fresh empty runtime; no refresh or partial-cache fallback.")
        plan = resident_plan([self.layer], budget_bytes, top_k=self.config.top_k)
        expert_ids = tuple(self.layer.expert_ids)
        domain = self.root["gate.weight"].shape[0]
        if not 1 <= domain <= 256 or any(type(index) is not int or not 0 <= index < domain for index in expert_ids):
            raise ValueError("V3 expert IDs must lie inside the bounded router domain.")
        table = self.root.get("gate.tid2eid")
        if table is None and set(expert_ids) != set(range(domain)):
            raise ValueError("Non-hash V3 router requires every expert resident.")
        if table is not None:
            # Startup-only validation: hash IDs cannot become native pointers unchecked.
            if table.ndim != 2 or table.shape[1] != self.config.top_k or table.shape[0] < 1:
                raise ValueError("Invalid resident hash table geometry.")
            if not set(table.cpu().unique().tolist()).issubset(expert_ids):
                raise ValueError("Hash table selects an unavailable resident expert.")
        self._resident_failed = True
        try:
            # Even when the caller is in inference_mode, these immutable banks
            # retain version counters for trial-boundary mutation detection.
            with torch.inference_mode(False):
                banks = {
                    kind: {
                        field: torch.empty(
                            self.layer.tensor_shapes[f"{kind}_{field}"],
                            dtype=VQ2_TP1_TORCH_DTYPES[field],
                            device=self.device,
                        )
                        for field in ELEMENT_BYTES
                    }
                    for kind in KINDS
                }
            resident = {}
            for row, expert in enumerate(expert_ids):
                resident[expert] = {}
                for kind in KINDS:
                    with torch.device("cpu"):
                        host, spec = self.artifact.load_expert(self.layer_index, expert, kind, device="cpu")
                    payload = {}
                    for field in ELEMENT_BYTES:
                        banks[kind][field][row].copy_(host[field])
                        payload[field] = banks[kind][field][row]
                    resident[expert][kind] = (MappingProxyType(payload), spec)
                    del host
                resident[expert] = MappingProxyType(resident[expert])
                if self.progress and ((row + 1) % 32 == 0 or row + 1 == len(expert_ids)):
                    print(
                        f"MODEL layer={self.layer_index} stage=v3_resident_payload "
                        f"loaded={row + 1} total={len(expert_ids)}",
                        flush=True,
                    )
            with torch.inference_mode(False):
                preparation = RowwiseVQ2A8Preparation(compact=True, validity=self._retain_valid)
                preparation._ensure_hadamard(self.device, 128)
                constants = make_constants(self.root["gate.weight"])
                self._resident_workspaces = {
                    kind: ResidentProjectionWorkspace(
                        [resident[expert][kind][0] for expert in expert_ids],
                        self.layer.specs[kind],
                        plan["layer_plans"][self.layer_index]["jobs"],
                        constants,
                        preparation,
                        banks=banks[kind],
                    )
                    for kind in KINDS
                }
                lookup = torch.full((domain,), -1, device="cpu", dtype=torch.int64)
                for row, expert in enumerate(expert_ids):
                    lookup[expert] = row
                self._resident_lookup = lookup.to(self.device)
            self._cache = MappingProxyType(resident)
            self.root = MappingProxyType(dict(self.root))
            self.cache_experts = len(expert_ids)
            self.cache_loads += len(expert_ids)
            self._resident_bytes = plan["payload_bytes"]
            self.cache_peak_bytes = self._resident_bytes
            self.h2d_bytes += sum(self._payload_bytes(expert) for expert in resident.values())
            self._row_preparation = preparation
            self._resident_plan = plan
            # Legacy roots may be inference tensors without version counters;
            # they still have pointer/shape guards and must not be mutated.
            self._resident_seals = tuple((value, self._seal(value)) for value in self._immutable_tensors())
            synchronize_execution(self.device)
            self._resident_ready, self._resident_failed = True, False
        except Exception:
            # Terminal failure: a partially initialized runtime can never decode.
            self._resident_ready = False
            raise
        return self.resident_report()

    @staticmethod
    def _seal(value):
        try:
            version = value._version
        except RuntimeError:
            version = None
        return value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype, value.device, version

    def _immutable_tensors(self):
        for workspace in self._resident_workspaces.values():
            yield from workspace.banks.values()
            yield workspace.pointer_bank
            yield workspace.constants
            yield workspace.preparation._hadamard
        yield from self.root.values()
        yield self._resident_lookup

    def _bind_stream(self, current):
        """Record all persistent allocation lifetimes once on the consumer stream."""
        stream = current.npu_stream
        if self._resident_stream is not None:
            if self._resident_stream != stream:
                raise RuntimeError("V3 workspace ownership is restricted to one stream.")
            return
        owners = list(self._immutable_tensors())
        for workspace in self._resident_workspaces.values():
            owners.extend(workspace.owners)
            owners.append(workspace.descriptors)
        seen = set()
        for tensor in owners:
            pointer = tensor.data_ptr()
            if pointer not in seen:
                tensor.record_stream(current)
                seen.add(pointer)
        self._resident_stream = stream

    def check_resident_immutable(self):
        """Host metadata audit at trial boundaries, not on every token."""
        if not self._resident_ready or self._resident_failed:
            raise RuntimeError("V3 resident initialization has not completed.")
        current = tuple(self._immutable_tensors())
        if len(current) != len(self._resident_seals) or any(
            value is not original or self._seal(value) != seal
            for value, (original, seal) in zip(current, self._resident_seals)
        ):
            self._resident_failed = True
            raise RuntimeError("Resident pointer/metadata changed; discard the V3 runtime.")

    def resident_report(self):
        self.check_resident_immutable()
        return dict(
            ready=True,
            plan=self._resident_plan,
            decode_device_calls=self.decode_device_calls,
            prefill_legacy_calls=self.prefill_legacy_calls,
            full_model_graph_verified=False,
            decode_route_host_reads=0,
            decode_descriptor_h2d=0,
            preparation="v1_rowwise_dense_rht_no_fwht",
            eager_intermediate_allocations=True,
            host_phase_timers_collected=False,
            decode_calls=self.decode_device_calls,
            descriptor_h2d_bytes=0,
            route_host_reads=0,
            counter_scope="decode_only_prefill_remains_eager_host_routed",
        )

    def v3_report(self):
        return self.resident_report()

    def check_resident_integrity(self):
        return self.check_resident_immutable()

    def v3_validity(self):
        return self.v3_valid

    @property
    def v3_valid(self):
        value = self._resident_valid
        prefill = self._v3_prefill_state.valid
        return prefill if value is None else value if prefill is None else value & prefill

    def configure_v3_probe(self, *, measurement, profile=False, optimization=None, compact=True):
        if any(type(value) is not bool for value in (measurement, profile, compact)):
            raise ValueError("V3 probe switches must be boolean.")
        if optimization not in (None, "v3", "batched"):
            raise ValueError("V3 probe supports only the fixed resident V1-math path; FWHT/pipeline are not implicit.")
        self.check_resident_immutable()
        self.measurement_mode = measurement
        self.trace_native = False
        self._resident_valid = None
        self._v3_prefill_state.valid = None
        self._v3_prefill_state.profile = profile
        # No caller can replace resident preparation with an optimization preset.
        self._row_preparation.compact = compact

    def clear_cache(self):
        raise RuntimeError("V3 immutable resident weights cannot be evicted; discard the owning runtime.")

    def _get_expert(self, expert_id):
        if not self._resident_ready or self._resident_failed or expert_id not in self._cache:
            raise RuntimeError("V3 requires a fully initialized resident expert; no lazy cache fallback.")
        self.cache_hits += 1
        return self._cache[expert_id]

    def _route_device(self, hidden, input_ids):
        logits = F.linear(hidden.float(), self.root["gate.weight"])
        table = self.root.get("gate.tid2eid")
        if table is None:
            return route_vq2a8(
                logits,
                self.config.top_k,
                renormalize=self.config.renormalize,
                correction_bias=self.root.get("gate.bias"),
                validity=self._retain_valid,
            )
        if (
            input_ids is None
            or input_ids.shape != (1,)
            or input_ids.device != hidden.device
            or input_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("V3 hash decode requires one integer token ID on the runtime device.")
        valid = (input_ids >= 0) & (input_ids < table.shape[0])
        self._retain_valid(valid.all() & torch.isfinite(logits).all())
        # Clamp before all gathers; invalid input is reported by the deferred
        # validity boundary and never produces an out-of-bounds memory access.
        ids = table.index_select(0, input_ids.clamp(0, table.shape[0] - 1).long())
        scores = F.softplus(logits.float()).sqrt()
        safe_ids = ids.clamp(0, logits.shape[1] - 1).long()
        self._retain_valid(((ids >= 0) & (ids < logits.shape[1])).all())
        weights = scores.gather(1, safe_ids)
        if self.config.renormalize:
            denominator = weights.sum(1, keepdim=True)
            self._retain_valid(torch.isfinite(denominator).all() & (denominator > 0).all())
            weights = weights / denominator
        return weights, ids.long()

    def _projections_many(self, requests):
        # This explicit legacy ABI is only the B>1 prefill path. It keeps the
        # original row preparation, namespace-isolated from the V1 library.
        prepared = self._row_preparation.many(requests)
        inputs = [
            (*values, *(payload[field] for field in POINTER_FIELDS))
            for values, (_, payload, _) in zip(prepared, requests)
        ]
        output = grouped_projection_v3(inputs)
        rows = sum(hidden.shape[0] for hidden, _, _ in requests)
        self.native_calls += len(requests)
        self.native_launches += 1
        self.native_rows += rows
        self.projection_rows += rows
        self.prepare_batches += rows
        return output

    def _projection(self, hidden, payload, spec):
        return self._projections_many([(hidden, payload, spec)])[0]

    def _forward(self, hidden, input_ids):
        if not self._resident_ready or self._resident_failed:
            raise RuntimeError("V3 decode requires completed all-resident initialization.")
        if self._resident_busy:
            raise RuntimeError("V3 workspaces cannot be entered concurrently or recursively.")
        if (
            hidden.ndim != 2
            or hidden.shape[1] != self.config.hidden_size
            or hidden.dtype != torch.bfloat16
            or hidden.device != self.device
        ):
            raise ValueError("V3 requires BF16 hidden states on the resident device.")
        self._bind_stream(torch.npu.current_stream(self.device))
        self._resident_busy = True
        try:
            if hidden.shape[0] != 1:
                self.prefill_legacy_calls += 1
                return self._v3_prefill_state.forward(self, hidden, input_ids)
            with self._v3_prefill_state.scope("v3_device_route"):
                weights, ids = self._route_device(hidden, input_ids)
            domain = self._resident_lookup.shape[0]
            in_range = (ids >= 0) & (ids < domain)
            slots = self._resident_lookup.index_select(0, ids.clamp(0, domain - 1).reshape(-1))
            self._retain_valid(in_range.all() & (slots >= 0).all())
            slots = slots.clamp_min(0)
            jobs = self._resident_workspaces["gate_up"].jobs
            if jobs == 1:
                slots = slots[:1]
            with self._v3_prefill_state.scope("v3_gate_up"):
                gates = self._resident_workspaces["gate_up"].project(hidden, slots)
            with self._v3_prefill_state.scope("v3_swiglu"):
                activation = deepseek_v4_swiglu_reference(gates, self.config.swiglu_limit)
            with self._v3_prefill_state.scope("v3_down"):
                values = self._resident_workspaces["down"].project(activation, slots)
            # No atomics, no changed reduction or BF16 rounding boundary. Even
            # singleton expert layers retain all six original top-k slots.
            expanded = values.expand(self.config.top_k, -1) if jobs == 1 else values
            result = (expanded.reshape(1, self.config.top_k, -1).float() * weights.unsqueeze(-1)).sum(1)
            result *= self.config.routed_scale
            if self.config.num_shared:
                result += self.shared(hidden).float()
            result = result.to(hidden.dtype)
            self._retain_valid(torch.isfinite(result).all())
            self.native_calls += jobs * 2
            self.native_rows += jobs * 2
            self.native_experts += jobs
            self.native_launches += 2
            self.projection_rows += jobs * 2
            self.prepare_batches += jobs * 2
            self.decode_device_calls += 1
            return result  # owned allocation, never aliases next-token workspaces
        finally:
            self._resident_busy = False
