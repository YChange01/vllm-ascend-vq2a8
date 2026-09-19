# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4 ownership/routing/graphs with an independent v2 compute candidate.

Only the compressed-layout conversion is adapted from the v2 source. No v2
cache, old queued Tensor-owning callback or v3 implementation is imported.
Preparation remains in original K order; the native bank gathers FP8 bytes
after quantization. Stable K regrouping changes accumulation order, requiring
new numerical and hardware acceptance rather than the V1 exact receipt.
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import regex as re
import torch
import torch.nn.functional as F

from vllm_ascend.quantization.vq2a8_abcd import validate_candidates
from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_execution import ALLOCATION_GRANULARITY, ASCENDC_MAX_JOBS, ASCENDC_MAX_ROWS
from vllm_ascend.quantization.vq2a8_execution_v4 import AscendCV4VQ2TP1MoE, _tensor_identity
from vllm_ascend.quantization.vq2a8_runtime import VQ2_TP1_TORCH_DTYPES
from vllm_ascend.quantization.vq2a8_v4_v2_layout import (
    V4_V2_ABI_VERSION as V4_V2_ABI_VERSION,
)
from vllm_ascend.quantization.vq2a8_v4_v2_layout import (
    V4_V2_BANK_COPIES as V4_V2_BANK_COPIES,
)
from vllm_ascend.quantization.vq2a8_v4_v2_layout import (
    V4_V2_BANK_WORDS as V4_V2_BANK_WORDS,
)
from vllm_ascend.quantization.vq2a8_v4_v2_layout import (
    V4_V2_DTYPES as V4_V2_DTYPES,
)
from vllm_ascend.quantization.vq2a8_v4_v2_layout import (
    V4_V2_FIELDS as V4_V2_FIELDS,
)
from vllm_ascend.quantization.vq2a8_v4_v2_layout import (
    V4_V2_MAX_EXPERTS as V4_V2_MAX_EXPERTS,
)
from vllm_ascend.quantization.vq2a8_v4_v2_layout import (
    V4_V2_SUPPORTED_K as V4_V2_SUPPORTED_K,
)
from vllm_ascend.quantization.vq2a8_v4_v2_layout import (
    V4_V2_SUPPORTED_N as V4_V2_SUPPORTED_N,
)
from vllm_ascend.quantization.vq2a8_v4_v2_layout import (
    _cpu_tensor as _cpu_tensor,
)
from vllm_ascend.quantization.vq2a8_v4_v2_layout import (
    _geometry as _geometry,
)
from vllm_ascend.quantization.vq2a8_v4_v2_layout import (
    convert_expert_payload as convert_expert_payload,
)


def require_v4_v2_features(
    reorder="scalar",
    preparation="rowwise",
    *,
    validity_mode="torch",
    route_mapping="torch",
    select_sign="separate",
    activation_tail="torch",
    runtime_guard="signature",
    decoder_input_mode="general",
    b1_schedule="baseline",
    native_ops=None,
):
    """Check only explicitly selected native extensions before weight loading."""
    if reorder not in ("scalar", "vectorized", "row_reuse", "chunk_reuse2", "chunk_reuse4") or preparation not in (
        "rowwise",
        "rowwise_packed",
        "sign_fused",
        "sign_fused_strided",
        "sign_fused_direct",
        "fused",
    ):
        raise ValueError("Invalid V4 v2 activation options.")
    if validity_mode not in ("torch", "fused", "fused_vectorized") or (
        validity_mode in ("fused", "fused_vectorized")
        and preparation not in ("sign_fused", "sign_fused_strided", "sign_fused_direct")
    ):
        raise ValueError("Fused validity requires native sign preparation.")
    if route_mapping not in ("torch", "fused"):
        raise ValueError("V4 v2 route mapping requires torch|fused.")
    if decoder_input_mode not in ("general", "b1_packed"):
        raise ValueError("Decoder input mode requires general|b1_packed.")
    validate_candidates(
        runtime_guard=runtime_guard,
        select_sign=select_sign,
        activation_tail=activation_tail,
        preparation=preparation,
        reorder=reorder,
        b1_schedule=b1_schedule,
    )
    native = torch.ops.vq2a8_ascendc_v4_v2 if native_ops is None else native_ops
    for selected, feature in (
        (reorder != "scalar", "activation_reorder_version"),
        (reorder == "row_reuse", "activation_reorder_row_reuse_version"),
        (reorder in ("chunk_reuse2", "chunk_reuse4"), "activation_reorder_chunk_reuse_version"),
        (b1_schedule == "tile_major", "b1_schedule_version"),
        (
            preparation in ("fused", "sign_fused", "sign_fused_strided", "sign_fused_direct"),
            "activation_preparation_version",
        ),
        (preparation in ("sign_fused_strided", "sign_fused_direct"), "activation_sign_strided_version"),
        (validity_mode == "fused", "layer_validity_version"),
        (validity_mode == "fused_vectorized", "layer_validity_vectorized_version"),
        (route_mapping == "fused", "route_mapping_version"),
        (select_sign == "fused", "select_sign_version"),
        (activation_tail == "fused_reorder", "activation_tail_reorder_version"),
        (runtime_guard == "native", "runtime_guard_version"),
        (decoder_input_mode == "b1_packed", "decoder_input_plan_version"),
    ):
        if selected:
            try:
                version = getattr(native, feature)()
            except (AttributeError, RuntimeError) as error:
                raise RuntimeError(f"Rebuild V4 v2 library for {feature}; no fallback.") from error
            if type(version) is not int or version != 1:
                raise RuntimeError(f"Unsupported V4 v2 {feature}={version!r}; require 1.")
    if route_mapping == "fused":
        # Reject partial/old binaries before loading the resident weights, not
        # only when the first layer constructs its mapper during graph setup.
        from vllm_ascend.quantization.vq2a8_route_mapping import FusedRouteMapping

        FusedRouteMapping(native_ops=native)


def require_v4_v2_library():
    try:
        abi = torch.ops.vq2a8_ascendc_v4_v2.abi_version()
        bank = torch.classes.vq2a8_ascendc_v4_v2.ResidentBank
    except (RuntimeError, AttributeError) as error:
        raise RuntimeError("Load libvq2a8_ascendc_v4_v2.so first; V4 v2 has no V1/v3 fallback.") from error
    if type(abi) is not int or abi != V4_V2_ABI_VERSION:
        raise RuntimeError("V4 v2 resident library ABI mismatch.")
    return bank


def load_v4_v2_library(path, expected_sha256):
    """Use an explicitly selected standalone binary, independent of V1 ABI."""
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("V4 v2 requires an explicit SHA256 library identity.")
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.name != "libvq2a8_ascendc_v4_v2.so":
        raise ValueError("V4 v2 requires libvq2a8_ascendc_v4_v2.so, not the V1/v2/v3 library.")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_sha256:
        raise ValueError("V4 v2 library SHA256 does not match the selected binary.")
    if str(path) not in torch.ops.loaded_libraries:
        if hasattr(torch.ops.vq2a8_ascendc_v4_v2, "abi_version"):
            raise RuntimeError("Another V4 v2 library is registered; use a fresh process.")
        import torch_npu  # noqa: F401 - register NPU before the native extension

        torch.ops.load_library(str(path))
    require_v4_v2_library()
    return {"path": str(path), "sha256": actual, "abi_version": V4_V2_ABI_VERSION}


def _projection_shapes(layer, kind):
    if getattr(layer, "v4_v2_prepacked", False):
        # The prepacked reader checks actual serialized headers and model
        # geometry. Do not interpret its six fields as legacy direct tensors.
        return layer.projection_shapes(kind)
    count = len(layer.expert_ids)
    source = {name: tuple(layer.tensor_shapes[f"{kind}_{name}"]) for name in VQ2_TP1_TORCH_DTYPES}
    packed = source["packed_indices"]
    if len(packed) != 3 or packed[0] != count:
        raise ValueError("Invalid V4 v2 expert packed tensor header.")
    n, k = packed[1] * 2, packed[2] * 8
    _geometry(n, k)
    books = source["codebooks"]
    if len(books) != 5 or books[0] != count or not 1 <= books[1] <= 256 or books[2:] != (n // 32, 16, 2):
        raise ValueError("Invalid V4 v2 expert codebook header.")
    if any(source[name] != (count, k) for name in ("codebook_tile_ids", "weight_scale", "weight_bias", "rht_sign")):
        raise ValueError("Invalid V4 v2 expert preparation/tile-ID header.")
    return ((n // 32, k // 16, 16, 8), (k // 256, n // 32, 32), (k,), (k,), (k,), (k,))


def _rounded(size):
    return (size + ALLOCATION_GRANULARITY - 1) // ALLOCATION_GRANULARITY * ALLOCATION_GRANULARITY


def v4_v2_resident_plan(layers, budget_bytes):
    """Plan the only payload layout plus both eager/graph metadata banks.

    Graph activations, kernel scratch, RHT constants, roots, KV and allocator
    headroom remain in the external reserve. Reserving graph metadata up front
    avoids unexpectedly consuming the full payload budget at graph startup.
    """
    if type(budget_bytes) is not int or budget_bytes < 0:
        raise ValueError("V4 v2 resident budget must be a nonnegative integer.")
    layer_plans = {}
    for layer in layers:
        ids = layer.expert_ids
        if (
            not ids
            or len(ids) > V4_V2_MAX_EXPERTS
            or any(type(i) is not int or not 0 <= i < V4_V2_MAX_EXPERTS for i in ids)
            or len(set(ids)) != len(ids)
            or layer.layer_index in layer_plans
        ):
            raise ValueError("V4 v2 requires unique layers and nonempty unique expert IDs in [0,256).")
        sizes = [
            math.prod(shape) * dtype.itemsize
            for kind in ("gate_up", "down")
            for shape, dtype in zip(_projection_shapes(layer, kind), V4_V2_DTYPES)
        ]
        count = len(ids)
        # Each bank set owns one lookup, one dense slot tensor, two tables.
        metadata = V4_V2_BANK_COPIES * (
            _rounded(V4_V2_MAX_EXPERTS * 8) + _rounded(count * 8) + 2 * _rounded(count * V4_V2_BANK_WORDS * 8)
        )
        layer_plans[layer.layer_index] = {
            "experts": count,
            "payload_bytes": count * sum(sizes),
            "metadata_reserve_bytes": metadata,
            "planned_bytes": count * sum(_rounded(size) for size in sizes) + metadata,
        }
    if not layer_plans:
        raise ValueError("V4 v2 planning requires at least one layer.")
    required = sum(plan["planned_bytes"] for plan in layer_plans.values())
    if required > budget_bytes:
        raise ValueError(f"V4 v2 full residency requires {required} bytes, exceeds budget {budget_bytes}; no fallback.")
    return {
        "budget_bytes": budget_bytes,
        "planned_bytes": required,
        "full_packed_bytes": required,
        "per_layer_cache_limit": max(plan["experts"] for plan in layer_plans.values()),
        "layer_limits": {index: plan["experts"] for index, plan in layer_plans.items()},
        "all_experts_fit": True,
        "allocation": "eager_v2_packed_only",
        "layout": "v2_zn_pair_lut",
        "compute_backend": "v2",
        "layer_plans": layer_plans,
    }


class AscendCV4V2VQ2TP1MoE(AscendCV4VQ2TP1MoE):
    """V4 baseline lifetime and graph protocol, with no old-payload fallback."""

    v4_compute_backend = "v2"
    v4_activation_reorder = "scalar"
    v4_activation_preparation = "rowwise"
    v4_validity_mode = "torch"
    v4_route_mapping = "torch"
    v4_runtime_guard = "signature"
    v4_select_sign = "separate"
    v4_activation_tail = "torch"
    v4_b1_schedule = "baseline"
    residency_plan = staticmethod(v4_v2_resident_plan)

    def __init__(
        self,
        *args,
        v4_activation_reorder="scalar",
        v4_activation_preparation="rowwise",
        v4_validity_mode="torch",
        v4_route_mapping="torch",
        v4_runtime_guard="signature",
        v4_select_sign="separate",
        v4_activation_tail="torch",
        v4_b1_schedule="baseline",
        **kwargs,
    ):
        if v4_activation_reorder not in ("scalar", "vectorized", "row_reuse", "chunk_reuse2", "chunk_reuse4"):
            raise ValueError("Invalid V4 v2 activation reorder.")
        if v4_activation_preparation not in (
            "rowwise",
            "rowwise_packed",
            "sign_fused",
            "sign_fused_strided",
            "sign_fused_direct",
            "fused",
        ):
            raise ValueError("Invalid V4 v2 activation preparation mode.")
        self.v4_activation_reorder = v4_activation_reorder
        self.v4_activation_preparation = v4_activation_preparation
        if v4_validity_mode not in ("torch", "fused", "fused_vectorized") or (
            v4_validity_mode in ("fused", "fused_vectorized")
            and v4_activation_preparation not in ("sign_fused", "sign_fused_strided", "sign_fused_direct")
        ):
            raise ValueError("Fused validity requires native sign preparation.")
        self.v4_validity_mode = v4_validity_mode
        if v4_route_mapping not in ("torch", "fused"):
            raise ValueError("V4 v2 route mapping requires torch|fused.")
        self.v4_route_mapping = v4_route_mapping
        validate_candidates(
            v4_runtime_guard,
            v4_select_sign,
            v4_activation_tail,
            preparation=v4_activation_preparation,
            reorder=v4_activation_reorder,
            b1_schedule=v4_b1_schedule,
        )
        self.v4_runtime_guard = v4_runtime_guard
        self.v4_select_sign = v4_select_sign
        self.v4_activation_tail = v4_activation_tail
        self.v4_b1_schedule = v4_b1_schedule
        self.v4_candidate_graph_build_calls = 0
        self.v4_candidate_reference_calls = 0
        super().__init__(*args, **kwargs)
        self._v2_payload_locations = {}

    def make_v4_preparation(self, *, compact=False, validity=None):
        if self.v4_activation_preparation in (
            "rowwise_packed",
            "sign_fused",
            "sign_fused_strided",
            "sign_fused_direct",
        ):
            from vllm_ascend.quantization.vq2a8_activation_packed import PackedRowwiseVQ2A8Preparation

            return PackedRowwiseVQ2A8Preparation(
                compact=compact,
                validity=validity,
                fuse_sign=self.v4_activation_preparation != "rowwise_packed",
                strided_sign=self.v4_activation_preparation in ("sign_fused_strided", "sign_fused_direct"),
                direct_output=self.v4_activation_preparation == "sign_fused_direct",
                fuse_select=self.v4_select_sign == "fused",
            )
        if self.v4_activation_preparation == "fused":
            from vllm_ascend.quantization.vq2a8_activation_fused import FusedV4V2Preparation

            return FusedV4V2Preparation(compact=compact, validity=validity)
        return RowwiseVQ2A8Preparation(compact=compact, validity=validity)

    def project_v4_prepared(self, bank, quantized, scale, bias, slots):
        """F changes M=1 only; M>1 explicitly retains vectorized prefill.

        Shape dispatch is host metadata, not route-ID reads or payload copies.
        A missing row-reuse ABI/method always fails; it never selects a fallback.
        """
        if (
            self.v4_activation_reorder in ("chunk_reuse2", "chunk_reuse4")
            or getattr(self, "v4_b1_schedule", "baseline") != "baseline"
        ):
            # New J/K never alter eager reference or prefill, even if M=1.
            self.v4_candidate_reference_calls = getattr(self, "v4_candidate_reference_calls", 0) + 1
            return bank.project_vectorized(quantized, scale, bias, slots)
        if self.v4_activation_reorder == "row_reuse":
            rows = 1 if quantized.ndim == 2 else quantized.shape[1]
            project = bank.project_row_reuse if rows == 1 else bank.project_vectorized
        else:
            project = bank.project_vectorized if self.v4_activation_reorder == "vectorized" else bank.project
        return project(quantized, scale, bias, slots)

    def project_v4_graph_prepared(self, bank, quantized, scale, bias, slots):
        """J/K only in graph construction; counters are not replay execution proof."""
        chunks = {"chunk_reuse2": 2, "chunk_reuse4": 4}.get(self.v4_activation_reorder, 0)
        schedule = int(getattr(self, "v4_b1_schedule", "baseline") == "tile_major")
        if chunks or schedule:
            rows = 1 if quantized.ndim == 2 else quantized.shape[1]
            if rows != 1:
                raise ValueError("J/K graph candidates require M=1; prefill must use the reference path.")
            self.v4_candidate_graph_build_calls = getattr(self, "v4_candidate_graph_build_calls", 0) + 1
            return bank.project_candidate(quantized, scale, bias, slots, chunks, schedule)
        return self.project_v4_prepared(bank, quantized, scale, bias, slots)

    def project_v4_normalized(self, bank, normalized, scale, bias, slots):
        """Candidate D only; callers must explicitly retain Torch RealDiv."""
        if self.v4_activation_tail != "fused_reorder":
            raise ValueError("Normalized projection requires fused_reorder activation tail.")
        return bank.project_tail(normalized, scale, bias, slots)

    def _prepare_host_expert(self, host):
        from vllm_ascend.quantization.vq2a8_v4_v2_prepacked import V4_V2_PREPACKED_FORMAT

        if self.artifact.manifest.get("format") == V4_V2_PREPACKED_FORMAT:
            # Reader has validated serialized layout and metadata. Never invoke
            # conversion, reopen original weights or create another layout.
            return host
        start = time.perf_counter()
        converted = {kind: (convert_expert_payload(payload, spec), spec) for kind, (payload, spec) in host.items()}
        self.timing["host_convert_s"] = self.timing.get("host_convert_s", 0.0) + time.perf_counter() - start
        return converted

    def _validate_resident_artifact(self):
        from vllm_ascend.quantization.vq2a8_v4_v2_prepacked import V4_V2_PREPACKED_FORMAT

        if self.artifact.manifest.get("format") == V4_V2_PREPACKED_FORMAT:
            if not getattr(self.layer, "v4_v2_prepacked", False):
                raise ValueError("V4 v2 prepacked artifacts require the dedicated validated reader.")
            return
        super()._validate_resident_artifact()

    def _fingerprint(self, expert_id):
        expert = self._cache[expert_id]
        if set(expert) != {"gate_up", "down"}:
            raise RuntimeError("V4 v2 resident expert must contain both projections.")
        result = [id(expert)]
        for kind in ("gate_up", "down"):
            payload, spec = expert[kind]
            if set(payload) != set(V4_V2_FIELDS):
                raise RuntimeError("V4 v2 resident payload field set changed.")
            result.append((kind, id(payload), id(spec)))
            for field, dtype, shape in zip(V4_V2_FIELDS, V4_V2_DTYPES, _projection_shapes(self.layer, kind)):
                tensor = payload[field]
                if (
                    tensor.dtype != dtype
                    or tuple(tensor.shape) != shape
                    or not tensor.is_contiguous()
                    or tensor.device != self.device
                ):
                    raise RuntimeError(f"V4 v2 resident {kind}.{field} geometry/device changed.")
                result.append((kind, field, _tensor_identity(tensor)))
        return tuple(result)

    def initialize_resident(self, *, budget_bytes):
        super().initialize_resident(budget_bytes=budget_bytes)
        try:
            self._v2_payload_locations = {
                id(self._cache[expert_id][kind][0]): (kind, slot)
                for slot, expert_id in enumerate(self.layer.expert_ids)
                for kind in ("gate_up", "down")
            }
            # Prefill and eager startup also use these banks. Never lazily build
            # metadata or upload another weight representation in forward().
            self._device_route_banks = self._create_device_route_banks()
            return self.check_resident_integrity()
        except BaseException:
            try:
                self.abort_residency()
            except BaseException as cleanup_error:
                self._resident_cleanup_error = str(cleanup_error)
            raise

    def _create_device_route_banks(self):
        self._require_ready()
        if self.device.type != "npu":
            raise ValueError("V4 v2 resident banks require NPU; no CPU fallback.")
        bank_type = require_v4_v2_library()
        ids = tuple(self.layer.expert_ids)
        if (
            not 1 <= self.config.top_k <= ASCENDC_MAX_JOBS
            or not 1 <= self.config.num_experts <= V4_V2_MAX_EXPERTS
            or any(i >= self.config.num_experts for i in ids)
        ):
            raise ValueError("V4 v2 requires 1..6 routes and an in-range expert inventory.")
        with torch.device("cpu"):
            lookup = torch.full((V4_V2_MAX_EXPERTS,), -1, dtype=torch.int64, device="cpu")
            for slot, expert_id in enumerate(ids):
                lookup[expert_id] = slot
            slots = torch.arange(len(ids), dtype=torch.int64, device="cpu")
        banks = {
            "lookup": lookup.to(self.device),
            "slots": slots.to(self.device),
            "metadata_bytes": lookup.numel() * lookup.element_size() + slots.numel() * slots.element_size(),
        }
        for kind in ("gate_up", "down"):
            entries = [self._cache[expert_id][kind] for expert_id in ids]
            spec = entries[0][1]
            geometry = (spec.rows, spec.columns, spec.rht_true_columns, spec.rht_block_size)
            if any((s.rows, s.columns, s.rht_true_columns, s.rht_block_size) != geometry for _, s in entries):
                raise ValueError("V4 v2 resident bank requires matching expert geometry.")
            bank = bank_type(*[[payload[field] for payload, _ in entries] for field in V4_V2_FIELDS])
            if self.v4_activation_reorder != "scalar" and not hasattr(bank, "project_vectorized"):
                raise RuntimeError("Rebuild the V4 v2 library for vectorized activation reorder; no scalar fallback.")
            if self.v4_activation_reorder == "row_reuse" and not hasattr(bank, "project_row_reuse"):
                raise RuntimeError("Rebuild the V4 v2 library for row-reuse activation reorder; no fallback.")
            if (
                self.v4_activation_reorder in ("chunk_reuse2", "chunk_reuse4") or self.v4_b1_schedule != "baseline"
            ) and not hasattr(bank, "project_candidate"):
                raise RuntimeError("Rebuild the V4 v2 library for J/K candidates; no fallback.")
            if bank.metadata()[3] != len(ids) * V4_V2_BANK_WORDS * 8:
                raise RuntimeError("V4 v2 native bank metadata differs from the residency plan.")
            banks[kind] = (bank, spec)
            banks["metadata_bytes"] += bank.metadata()[3]
        return banks

    def _projection(self, hidden, payload, spec):
        chunks = [self._projections_many([(chunk, payload, spec)])[0] for chunk in hidden.split(ASCENDC_MAX_ROWS)]
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

    def _projections_many(self, requests):
        """Batched prefill over the same bank; original preparation then pad.

        Prefill retains V4's host route scheduling. Slots concatenate views of
        startup device IDs, never a new CPU descriptor or expert payload. Mixed
        row counts pad prepared FP8 *bytes* only, preserving real-row GEMVs.
        """
        self._require_ready()
        if not 1 <= len(requests) <= ASCENDC_MAX_JOBS or self._device_route_banks is None:
            raise ValueError("V4 v2 grouped projection requires ready banks and 1..6 requests.")
        locations = []
        for hidden, payload, spec in requests:
            location = self._v2_payload_locations.get(id(payload))
            if location is None or hidden.ndim != 2 or not 1 <= hidden.shape[0] <= ASCENDC_MAX_ROWS:
                raise ValueError("V4 v2 projection requires a resident payload and 1..32 rows.")
            locations.append(location)
        kinds = {kind for kind, _ in locations}
        if len(kinds) != 1:
            raise ValueError("V4 v2 grouped projection cannot mix gate/up and down banks.")
        kind = locations[0][0]
        bank, bank_spec = self._device_route_banks[kind]
        if any(spec is not bank_spec for _, _, spec in requests):
            # Artifact specs may be distinct immutable objects with same shape.
            geometry = (bank_spec.rows, bank_spec.columns, bank_spec.rht_true_columns, bank_spec.rht_block_size)
            if any((s.rows, s.columns, s.rht_true_columns, s.rht_block_size) != geometry for _, _, s in requests):
                raise ValueError("V4 v2 projection geometry differs from its resident bank.")
        state = getattr(self, "_optimization", None)
        if state is not None and (state.options.preparation != "rowwise" or state.options.pipeline):
            raise ValueError("V4 v2 requires original rowwise preparation; V1 pipeline/FWHT presets are unsupported.")
        if not hasattr(self, "_row_preparation"):
            self._row_preparation = self.make_v4_preparation()
        start = time.perf_counter()
        prepared = self._row_preparation.many(requests)
        self._timing_sync()
        self.timing["prepare_s"] += time.perf_counter() - start
        counts = [hidden.shape[0] for hidden, _, _ in requests]
        rows, padded_rows = sum(counts), max(counts)
        if state is not None:
            state.stats["preparation_calls"] += 1
        q = (
            torch.stack(
                [
                    F.pad(values[0].view(torch.uint8), (0, 0, 0, padded_rows - count))
                    for values, count in zip(prepared, counts)
                ]
            )
            .contiguous()
            .view(torch.float8_e4m3fn)
        )
        scale, bias = (
            torch.stack(
                [F.pad(values[index], (0, padded_rows - count)) for values, count in zip(prepared, counts)]
            ).contiguous()
            for index in (1, 2)
        )
        slot_ids = self._device_route_banks["slots"]
        slots = torch.cat([slot_ids[slot : slot + 1] for _, slot in locations]).contiguous()
        start = time.perf_counter()
        output, valid = self.project_v4_prepared(bank, q, scale, bias, slots)
        valid = (valid != 0).all() & torch.isfinite(output).all()
        if state is not None:
            state.retain(valid)
        elif not bool(valid):
            raise ValueError("V4 v2 projection returned an invalid expert/activation/output.")
        self._timing_sync()
        self.timing["packed_projection_s"] += time.perf_counter() - start
        self.prepare_batches += rows
        self.native_calls += len(requests)
        self.native_launches += 1
        self.native_rows += rows
        self.projection_rows += rows
        return [output[index, :count] for index, count in enumerate(counts)]

    def abort_residency(self):
        super().abort_residency()
        self._v2_payload_locations.clear()

    def v4_report(self):
        from vllm_ascend.quantization.vq2a8_v4_v2_prepacked import V4_V2_PREPACKED_FORMAT

        report = super().v4_report()
        source_format = self.artifact.manifest.get("format")
        return {
            **report,
            "compute_backend": "v2",
            "source_format": source_format,
            "startup_conversion": source_format != V4_V2_PREPACKED_FORMAT,
            "preload_host_convert_s": self.timing.get("host_convert_s", 0.0),
            "preload_host_read_s": self.timing.get("host_read_s", 0.0),
            "preload_host_validate_s": self.timing.get("host_validate_s", 0.0),
            "preload_h2d_s": self.timing.get("h2d_s", 0.0),
            "activation_reorder": self.v4_activation_reorder,
            "b1_schedule": self.v4_b1_schedule,
            "candidate_graph_build_calls": self.v4_candidate_graph_build_calls,
            "candidate_reference_calls": self.v4_candidate_reference_calls,
            "activation_reorder_row_reuse_scope": "m1_only_m_gt1_vectorized"
            if self.v4_activation_reorder == "row_reuse"
            else None,
            "activation_preparation": self.v4_activation_preparation,
            "validity_mode": self.v4_validity_mode,
            "route_mapping": self.v4_route_mapping,
            "runtime_guard": self.v4_runtime_guard,
            "select_sign": self.v4_select_sign,
            "activation_tail": self.v4_activation_tail,
            "layout": "v2_zn_pair_lut",
            "arithmetic_contract": "v2_k_regrouped_requires_tolerance_validation",
            "metadata_bytes": self._device_route_banks["metadata_bytes"] if self._device_route_banks else 0,
            "metadata_reserve_bytes": self._resident_plan.get("metadata_reserve_bytes", 0)
            if self._resident_plan
            else 0,
            "prefill_payload_layout": "v2_zn_pair_lut",
            "dual_payload_residency": False,
            "native_projection_launch_counter_scope": "compute_only_excludes_one_device_prepare_kernel",
        }
