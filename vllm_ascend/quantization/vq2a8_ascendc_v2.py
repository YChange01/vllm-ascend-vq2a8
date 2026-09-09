# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit experimental VQ2A8 AscendC v2 backend, never an automatic fallback.

Convert compressed CPU payloads on cache admission, not in the projection hot
path. Arbitrary sixteen-entry *pair* codebooks remain byte-exact; there is no
fitting to four scalar levels, MXFP8 requantization, or dense weight staging.
Stable K regrouping changes FP32 reduction order. This candidate therefore
requires new numerical/device/model acceptance, not the old bit-exact receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path, PurePosixPath

import numpy as np
import regex as re
import torch

from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_execution import (
    ALLOCATION_GRANULARITY,
    ASCENDC_MAX_JOBS,
    ASCENDC_MAX_ROWS,
    AscendCVQ2TP1MoE,
    packed_cache_plan,
)
from vllm_ascend.quantization.vq2a8_runtime import VQ2_TP1_TORCH_DTYPES

ASCENDC_V2_ABI_VERSION = 1
ASCENDC_V2_SUPPORTED_N = 4096
ASCENDC_V2_SUPPORTED_K = (2048, 4096)
ASCENDC_V2_LUT_K = 256
ASCENDC_V2_ZN_K0 = 16
ASCENDC_V2_ZN_N0 = 32
ASCENDC_V2_REQUIRED_SOURCES = (
    "csrc/vq2a8_ascendc_v2/CMakeLists.txt",
    "csrc/vq2a8_expert_reference/code.txt",
    "csrc/vq2a8_ascendc_v2/kernel.cpp",
    "csrc/vq2a8_ascendc_v2/torch_binding.cpp",
    "csrc/vq2a8_ascendc_v2/launch.h",
    "csrc/vq2a8_ascendc_v2/layout.h",
)


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _geometry(n, k):
    if n != ASCENDC_V2_SUPPORTED_N or k not in ASCENDC_V2_SUPPORTED_K:
        raise ValueError("VQ2A8 v2 ABI 1 supports only N=4096 and K=2048/4096; no fallback is enabled.")


def validate_build_manifest(path, expected_sha256):
    """Validate the chosen binary AND its current local build-source identity."""
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("VQ2A8 v2 backend requires an explicit SHA256 library identity.")
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.name != "libvq2a8_ascendc_v2.so":
        raise ValueError("Expected the separately built libvq2a8_ascendc_v2.so.")
    if _sha256(path) != expected_sha256:
        raise ValueError("VQ2A8 v2 library differs from the explicitly selected candidate.")
    manifest = json.loads((path.parent / "build-manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "built"
        or manifest.get("implementation") != "ascendc_v2"
        or type(manifest.get("abi_version")) is not int
        or manifest["abi_version"] != ASCENDC_V2_ABI_VERSION
        or manifest.get("library_sha256") != expected_sha256
        or Path(manifest.get("library", "")).resolve() != path
    ):
        raise ValueError("VQ2A8 v2 build manifest status/ABI/library identity does not match.")
    source_hashes = manifest.get("source_sha256")
    if not isinstance(source_hashes, dict) or not set(ASCENDC_V2_REQUIRED_SOURCES).issubset(source_hashes):
        raise ValueError("VQ2A8 v2 build manifest is missing required source hashes.")
    source_root = Path(__file__).resolve().parents[2]
    for relative, expected in source_hashes.items():
        if not isinstance(relative, str) or "\\" in relative or ":" in relative:
            raise ValueError("Invalid VQ2A8 v2 source hash path.")
        name = PurePosixPath(relative)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Invalid VQ2A8 v2 source hash path.")
        current = (source_root / relative).resolve(strict=True)
        if not current.is_relative_to(source_root.resolve()) or _sha256(current) != expected:
            raise ValueError(f"VQ2A8 v2 build source changed: {relative}; rebuild and revalidate.")
    return {"path": str(path), "sha256": expected_sha256, "abi_version": ASCENDC_V2_ABI_VERSION}


def _require_loaded():
    if not all(hasattr(torch.ops.vq2a8_ascendc_v2, name) for name in ("grouped_projection", "abi_version")):
        raise RuntimeError("Load the pinned libvq2a8_ascendc_v2.so first; there is no JIT or fallback.")
    abi = torch.ops.vq2a8_ascendc_v2.abi_version()
    if type(abi) is not int or abi != ASCENDC_V2_ABI_VERSION:
        raise RuntimeError("Loaded VQ2A8 v2 library has an incompatible tensor ABI.")


def load_pinned_library(path, expected_sha256):
    identity = validate_build_manifest(path, expected_sha256)
    path = identity["path"]
    if path in torch.ops.loaded_libraries:
        _require_loaded()
        return identity
    if hasattr(torch.ops.vq2a8_ascendc_v2, "grouped_projection"):
        raise RuntimeError("Another VQ2A8 v2 candidate is already registered; use a fresh process.")
    import torch_npu  # noqa: F401 - register NPU before native loading

    torch.ops.load_library(path)
    _require_loaded()
    return identity


def grouped_projection(inputs):
    """One MIX launch including FP32 scale/bias epilogue for 1..6 jobs."""
    _require_loaded()
    if not 1 <= len(inputs) <= ASCENDC_MAX_JOBS or any(len(values) != 5 for values in inputs):
        raise ValueError("VQ2A8 v2 grouped projection requires 1..6 five-tensor inputs.")
    return torch.ops.vq2a8_ascendc_v2.grouped_projection(*(list(values) for values in zip(*inputs)))


def _cpu_tensor(value, name, dtype, shape=None):
    if not isinstance(value, torch.Tensor) or value.dtype != dtype or value.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU {dtype} tensor before expert conversion.")
    if not value.is_contiguous() or (shape is not None and tuple(value.shape) != tuple(shape)):
        raise ValueError(f"{name} has invalid shape or is not contiguous.")
    return value


def convert_expert_payload(payload, spec):
    """CPU cache-miss conversion; original payload and source artifacts untouched.

    Replace indices/codebooks/tile IDs by zN indices, homogeneous K256 pair
    LUTs, and an activation gather permutation. Preparation metadata stays in
    original K order because RHT and FP8 quantization MUST precede gathering.
    """
    if set(payload) != set(VQ2_TP1_TORCH_DTYPES):
        raise ValueError("VQ2A8 v2 conversion requires the exact validated direct-TP1 payload fields.")
    words = _cpu_tensor(payload["packed_indices"], "packed_indices", torch.int32)
    if words.ndim != 2 or min(words.shape) <= 0:
        raise ValueError("packed_indices requires positive [N/2,K/8] geometry.")
    n, k = words.shape[0] * 2, words.shape[1] * 8
    _geometry(n, k)
    if spec.columns != k or not 0 < spec.rht_true_columns <= k:
        raise ValueError("VQ2A8 v2 conversion geometry disagrees with matrix spec.")
    books = _cpu_tensor(payload["codebooks"], "codebooks", torch.float8_e4m3fn)
    if books.ndim != 4 or not 1 <= books.shape[0] <= 256 or books.shape[1:] != (n // 32, 16, 2):
        raise ValueError("codebooks requires [1..256,N/32,16,2] arbitrary FP8 pairs.")
    ids = _cpu_tensor(payload["codebook_tile_ids"], "codebook_tile_ids", torch.uint8, (k,)).numpy()
    if np.any(ids.astype(np.int64) >= books.shape[0]):
        raise ValueError("VQ2A8 v2 codebook tile ID is out of range.")
    byte_books = books.view(torch.uint8).numpy()
    if np.any((byte_books & np.uint8(127)) == np.uint8(127)):
        raise ValueError("VQ2A8 v2 codebooks contain E4M3FN NaN.")
    retained = {}
    for name in ("weight_scale", "weight_bias", "rht_sign"):
        tensor = _cpu_tensor(payload[name], name, VQ2_TP1_TORCH_DTYPES[name], (k,))
        valid = ((tensor == -1) | (tensor == 1)).all() if name == "rht_sign" else torch.isfinite(tensor).all()
        if not bool(valid):
            raise ValueError(f"Invalid expert preparation metadata: {name}.")
        retained[name] = tensor
    order = np.argsort(ids, kind="stable").astype(np.int64)
    blocks = ids[order].reshape(-1, ASCENDC_V2_LUT_K)
    if not np.all(blocks == blocks[:, :1]):
        raise ValueError("VQ2A8 v2 tile populations cannot form homogeneous K256 blocks; no lossy fallback.")
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


def expert_cached_bytes(layer):
    """Header-only actual cache cost, including int64 activation-order tensors."""
    total = 0
    for kind in ("gate_up", "down"):
        count = len(layer.expert_ids)
        shapes = {name: layer.tensor_shapes[f"{kind}_{name}"] for name in VQ2_TP1_TORCH_DTYPES}
        packed = shapes["packed_indices"]
        if len(packed) != 3 or packed[0] != count:
            raise ValueError("Invalid expert packed tensor header.")
        n, k = packed[1] * 2, packed[2] * 8
        _geometry(n, k)
        books = shapes["codebooks"]
        if len(books) != 5 or books[0] != count or not 1 <= books[1] <= 256 or books[2:] != (n // 32, 16, 2):
            raise ValueError("Invalid expert codebook header.")
        if any(shapes[name] != (count, k) for name in ("codebook_tile_ids", "weight_scale", "weight_bias", "rht_sign")):
            raise ValueError("Invalid expert preparation/tile-ID header.")
        sizes = (n * k // 4, (k // 256) * (n // 32) * 32, k * 8, k * 4, k * 4, k)
        total += sum(math.ceil(size / ALLOCATION_GRANULARITY) * ALLOCATION_GRANULARITY for size in sizes)
    return total


def ascendc_v2_cache_plan(layers, budget_bytes, *, expert_limit=256):
    plan = packed_cache_plan(layers, budget_bytes, expert_limit=expert_limit, expert_size_bytes=expert_cached_bytes)
    return {**plan, "allocation": "lazy_ascendc_v2_zn_pair_lut_with_activation_order"}


def gather_prepared_activation(quantized, activation_order):
    """Byte-preserving gather AFTER original quantization; no FP8 arithmetic."""
    if (
        quantized.dtype != torch.float8_e4m3fn
        or quantized.ndim != 2
        or activation_order.dtype != torch.int64
        or activation_order.shape != (quantized.shape[1],)
        or activation_order.device != quantized.device
    ):
        raise ValueError("VQ2A8 v2 activation gather requires FP8 [M,K] and cached same-device int64[K].")
    # Some NPU index kernels do not accept float8. Moving the exact byte view
    # avoids conversion and preserves signs, subnormals and rounding boundaries.
    gathered = quantized.contiguous().view(torch.uint8).index_select(1, activation_order)
    return gathered.contiguous().view(torch.float8_e4m3fn)


class AscendCV2VQ2TP1MoE(AscendCVQ2TP1MoE):
    """Opt-in VQ2A8 v2 kernel using the existing strict cache/router/MoE owner."""

    execution_policy = "ascendc_v2"

    def __init__(self, artifact, layer_index, device, **kwargs):
        # Reject unsupported model geometry before root/device allocations.
        expert_cached_bytes(artifact.layers[layer_index])
        super().__init__(artifact, layer_index, device, **kwargs)

    def _prepare_host_expert(self, host):
        return {kind: (convert_expert_payload(payload, spec), spec) for kind, (payload, spec) in host.items()}

    def _projection(self, hidden, payload, spec):
        outputs = [self._projections_many([(chunk, payload, spec)])[0] for chunk in hidden.split(ASCENDC_MAX_ROWS)]
        if not outputs:
            raise ValueError("VQ2A8 v2 projection requires at least one row.")
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)

    def _projections_many(self, requests):
        if self.device.type != "npu":
            raise ValueError("VQ2A8 v2 projections require NPU; no CPU/CUDA fallback.")
        state = getattr(self, "_optimization", None)
        if state is not None and (
            state.options.pipeline or state.options.prepare_graph or state.options.preparation != "rowwise"
        ):
            raise ValueError(
                "VQ2A8 v2 backend supports only rowwise fast/batched presets, not old pipeline/FWHT graphs."
            )
        if not hasattr(self, "_row_preparation"):
            self._row_preparation = RowwiseVQ2A8Preparation()
        start = time.perf_counter()
        with torch.device("cpu"), state.scope("preparation") if state is not None else nullcontext():
            prepared = self._row_preparation.many(requests)
            inputs = [
                (
                    gather_prepared_activation(q, payload["activation_order"]),
                    scale.contiguous(),
                    bias.contiguous(),
                    payload["packed_zn"],
                    payload["pair_lut"],
                )
                for (q, scale, bias), (_, payload, _) in zip(prepared, requests)
            ]
        self._timing_sync()
        rows = sum(hidden.shape[0] for hidden, _, _ in requests)
        self.timing["prepare_s"] += time.perf_counter() - start
        self.prepare_batches += rows
        if state is not None:
            state.stats["preparation_calls"] += 1
        start = time.perf_counter()
        with state.scope("native_projection") if state is not None else nullcontext():
            output = grouped_projection(inputs)
        self._timing_sync()
        self.timing["packed_projection_s"] += time.perf_counter() - start
        self.native_calls += len(requests)
        self.native_launches += 1
        self.native_rows += rows
        self.projection_rows += rows
        return output
