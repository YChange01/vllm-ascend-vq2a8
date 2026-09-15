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

import numpy as np
import regex as re
import torch
import torch.nn.functional as F

from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_execution import ALLOCATION_GRANULARITY, ASCENDC_MAX_JOBS, ASCENDC_MAX_ROWS
from vllm_ascend.quantization.vq2a8_execution_v4 import AscendCV4VQ2TP1MoE, _tensor_identity
from vllm_ascend.quantization.vq2a8_runtime import VQ2_TP1_TORCH_DTYPES

V4_V2_ABI_VERSION = 1
V4_V2_MAX_EXPERTS = 256
V4_V2_SUPPORTED_N = 4096
V4_V2_SUPPORTED_K = (2048, 4096)
V4_V2_FIELDS = ("packed_zn", "pair_lut", "activation_order", "weight_scale", "weight_bias", "rht_sign")
V4_V2_DTYPES = (torch.uint8, torch.uint8, torch.int64, torch.float32, torch.float32, torch.int8)
V4_V2_BANK_WORDS = 8
V4_V2_BANK_COPIES = 2  # eager plus optional graph; payload storage is shared


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


def _geometry(n, k):
    if n != V4_V2_SUPPORTED_N or k not in V4_V2_SUPPORTED_K:
        raise ValueError("V4 v2 supports N=4096 and K=2048/4096 only; no alternate-kernel fallback.")


def _cpu_tensor(tensor, name, dtype, shape=None):
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype or tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU {dtype} tensor before conversion.")
    if not tensor.is_contiguous() or (shape is not None and tuple(tensor.shape) != tuple(shape)):
        raise ValueError(f"Invalid contiguous geometry for {name}.")
    return tensor


def convert_expert_payload(payload, spec):
    """Convert one validated CPU expert, before its only payload H2D upload.

    Nibble packing and arbitrary 16-entry FP8 pair LUTs are lossless. This
    creates compressed zN data, not a dense FP8 expert or a second device bank.
    Source preparation metadata deliberately remains in original K order.
    """
    if set(payload) != set(VQ2_TP1_TORCH_DTYPES):
        raise ValueError("V4 v2 conversion requires exactly the six direct-TP1 fields.")
    words = _cpu_tensor(payload["packed_indices"], "packed_indices", torch.int32)
    if words.ndim != 2 or min(words.shape) <= 0:
        raise ValueError("packed_indices requires positive [N/2,K/8] geometry.")
    n, k = words.shape[0] * 2, words.shape[1] * 8
    _geometry(n, k)
    if spec.rows != n or spec.columns != k or not 0 < spec.rht_true_columns <= k:
        raise ValueError("V4 v2 conversion geometry disagrees with matrix spec.")
    books = _cpu_tensor(payload["codebooks"], "codebooks", torch.float8_e4m3fn)
    if books.ndim != 4 or not 1 <= books.shape[0] <= 256 or tuple(books.shape[1:]) != (n // 32, 16, 2):
        raise ValueError("codebooks requires [1..256,N/32,16,2] arbitrary FP8 pairs.")
    ids = _cpu_tensor(payload["codebook_tile_ids"], "codebook_tile_ids", torch.uint8, (k,)).numpy()
    if np.any(ids.astype(np.int64) >= books.shape[0]):
        raise ValueError("V4 v2 codebook tile ID is out of range.")
    byte_books = books.view(torch.uint8).numpy()
    if np.any((byte_books & np.uint8(127)) == np.uint8(127)):
        raise ValueError("V4 v2 codebooks contain E4M3FN NaN.")
    retained = {}
    for name in ("weight_scale", "weight_bias", "rht_sign"):
        tensor = _cpu_tensor(payload[name], name, VQ2_TP1_TORCH_DTYPES[name], (k,))
        valid = ((tensor == -1) | (tensor == 1)).all() if name == "rht_sign" else torch.isfinite(tensor).all()
        if not bool(valid):
            raise ValueError(f"Invalid expert preparation metadata: {name}.")
        retained[name] = tensor
    order = np.argsort(ids, kind="stable").astype(np.int64)
    blocks = ids[order].reshape(-1, 256)
    if not np.all(blocks == blocks[:, :1]):
        raise ValueError("V4 v2 tile populations cannot form homogeneous K256 blocks; no lossy fallback.")
    unsigned = words.numpy().view(np.uint32)
    shifts = np.arange(8, dtype=np.uint32) * np.uint32(4)
    indices = ((unsigned[..., None] >> shifts) & np.uint32(15)).astype(np.uint8).reshape(n // 2, k)
    indices = indices[:, order]
    zn = indices.reshape(n // 32, 16, k // 16, 16).transpose(0, 2, 3, 1)
    packed_zn = np.ascontiguousarray(zn[..., 0::2] | (zn[..., 1::2] << np.uint8(4)))
    pair_lut = np.ascontiguousarray(byte_books[blocks[:, 0]].reshape(k // 256, n // 32, 32))
    return {
        "packed_zn": torch.from_numpy(packed_zn),
        "pair_lut": torch.from_numpy(pair_lut),
        "activation_order": torch.from_numpy(order),
        **retained,
    }


def _projection_shapes(layer, kind):
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
    residency_plan = staticmethod(v4_v2_resident_plan)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._v2_payload_locations = {}

    def _prepare_host_expert(self, host):
        return {kind: (convert_expert_payload(payload, spec), spec) for kind, (payload, spec) in host.items()}

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
            self._row_preparation = RowwiseVQ2A8Preparation()
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
        output, valid = bank.project(q, scale, bias, slots)
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
        report = super().v4_report()
        return {
            **report,
            "compute_backend": "v2",
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
