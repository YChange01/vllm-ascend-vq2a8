# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless CPU packing of canonical expert weights for a future TP2 runtime.

Gate/up uses matching output-channel shards; down uses physical input-column
shards. The latter generally split each source codebook's population, so each
nonempty population is padded to K256 with *zero activation* columns. No dense
weight is materialized, no codeword is fitted, and no FP8 byte is requantized.

This is an artifact contract, not a claim of current native/runtime support.
In particular, down's rank-local dynamic A8 quantization differs from TP1's
full-K quantization, and regrouping K changes floating-point reduction order.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .vq2a8_artifact import VQ2MatrixSpec, validate_matrix_payload
from .vq2a8_repack import canonical_index_grid

VQ2_TP2_ZN_FORMAT = "vq2a8_zn_tp2_v1"
TP_SIZE = 2
LUT_K = 256
RHT_BLOCK_SIZE = 128
ZN_N0 = 32
ZN_K0 = 16
VECTOR_LENGTH = 2
RESIDENT_FIELDS = (
    "packed_zn",
    "pair_lut",
    "activation_order",
    "weight_scale",
    "weight_bias",
    "rht_sign",
)


def validate_tp2_spec(spec: VQ2MatrixSpec) -> None:
    """Validate shard geometry without reading or allocating weight tensors."""
    if not isinstance(spec, VQ2MatrixSpec):
        raise TypeError("TP2 packing requires a VQ2MatrixSpec.")
    for name in (
        "rows",
        "columns",
        "row_tiles",
        "column_tiles",
        "row_group_size",
        "group_size",
        "num_vectors",
        "num_elements",
        "rht_block_size",
        "rht_true_columns",
    ):
        value = getattr(spec, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{spec.name}: {name} must be a positive integer.")
    if spec.kind not in ("gate_up", "down"):
        raise ValueError(f"{spec.name}: unsupported TP2 matrix kind {spec.kind!r}.")
    if not all(value is True for value in (spec.enable_permutation, spec.enable_normalization, spec.enable_rht)):
        raise ValueError(f"{spec.name}: TP2 packing requires permutation, normalization, and RHT.")
    if not isinstance(spec.norm_dimension, int) or isinstance(spec.norm_dimension, bool) or spec.norm_dimension != 0:
        raise ValueError(f"{spec.name}: TP2 packing requires norm_dimension=0.")
    if spec.group_size != LUT_K or spec.row_group_size != ZN_N0 or spec.rht_block_size != RHT_BLOCK_SIZE:
        raise ValueError(f"{spec.name}: require group_size=256, row_group_size=32, and RHT block size=128.")
    if spec.rows % ZN_N0 or spec.columns % LUT_K:
        raise ValueError(f"{spec.name}: canonical N must align to 32 and K to 256.")
    if spec.rht_true_columns != spec.columns or spec.original_shape != (spec.rows, spec.columns):
        raise ValueError(f"{spec.name}: TP2 packing requires unpadded canonical true K equal to K.")
    if spec.row_tiles != spec.rows // ZN_N0 or spec.column_tiles != spec.columns // LUT_K:
        raise ValueError(f"{spec.name}: canonical tile counts disagree with N/K.")
    if spec.num_elements != spec.rows * spec.columns or spec.num_vectors != spec.rows * spec.columns // VECTOR_LENGTH:
        raise ValueError(f"{spec.name}: canonical element/vector counts disagree with N/K.")
    if spec.kind == "gate_up" and spec.rows % (VECTOR_LENGTH * TP_SIZE * ZN_N0):
        raise ValueError(f"{spec.name}: each TP2 gate and up slice must contain complete N32 output tiles.")
    if spec.kind == "down" and (spec.columns % TP_SIZE or (spec.columns // TP_SIZE) % RHT_BLOCK_SIZE):
        raise ValueError(f"{spec.name}: each TP2 down input slice must contain complete RHT128 blocks.")


def _shard_ranges(spec: VQ2MatrixSpec, rank: int) -> tuple[list[list[int]], list[int]]:
    if spec.kind == "gate_up":
        intermediate = spec.rows // VECTOR_LENGTH
        width = intermediate // TP_SIZE
        start = rank * width
        return [[start, start + width], [intermediate + start, intermediate + start + width]], [0, spec.columns]
    width = spec.columns // TP_SIZE
    return [[0, spec.rows]], [rank * width, (rank + 1) * width]


def _check_packed_mapping(
    packed: np.ndarray,
    pair_lut: np.ndarray,
    order: np.ndarray,
    canonical: np.ndarray,
    source_columns: np.ndarray,
    output_pairs: np.ndarray,
    row_tiles: np.ndarray,
    source_books: np.ndarray,
    tile_ids: np.ndarray,
    logical_k: int,
    raw_books: np.ndarray,
) -> None:
    """Independently unpack bytes and check their canonical coordinates."""
    packed_k = order.size
    if not np.array_equal(np.sort(order), np.arange(packed_k, dtype=np.int64)):
        raise RuntimeError("Internal TP2 activation order is not a bijection over padded K.")
    pairs = np.empty((*packed.shape[:-1], packed.shape[-1] * VECTOR_LENGTH), dtype=np.uint8)
    pairs[..., 0::2] = packed & np.uint8(15)
    pairs[..., 1::2] = packed >> np.uint8(4)
    restored = pairs.transpose(0, 3, 1, 2).reshape(output_pairs.size, packed_k)
    real = order < logical_k
    expected = canonical[np.ix_(output_pairs, source_columns[order[real]])]
    if not np.array_equal(restored[:, real], expected) or np.any(restored[:, ~real] != 0):
        raise RuntimeError("Internal TP2 packed-code/canonical-coordinate round trip failed.")
    if not np.all(tile_ids[order].reshape(-1, LUT_K) == source_books[:, None]):
        raise RuntimeError("Internal TP2 K256 block contains mixed source codebooks.")
    expected_lut = raw_books[source_books][:, row_tiles].reshape(pair_lut.shape)
    if not np.array_equal(pair_lut, expected_lut):
        raise RuntimeError("Internal TP2 pair LUT does not preserve canonical FP8 bytes.")


def repack_matrix_tp2(
    tensors: dict[str, torch.Tensor], spec: VQ2MatrixSpec, rank: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Pack one canonical matrix for rank 0/1, retaining all logical weights.

    ``activation_order[j]`` maps a packed K coordinate to a rank-local physical
    coordinate. Coordinates below ``logical_shape[1]`` are real; all remaining
    coordinates are appended zero inputs. RHT/normalization/A8 happen BEFORE
    this byte-preserving gather, never in regrouped codebook order.
    """
    validate_tp2_spec(spec)
    if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < TP_SIZE:
        raise ValueError("TP2 rank must be integer 0 or 1.")
    if not isinstance(tensors, dict):
        raise TypeError("Canonical TP2 payload must be a dictionary of CPU tensors.")
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError(f"{spec.name}.{name}: TP2 packing requires a contiguous CPU tensor.")
    validate_matrix_payload(tensors, spec)

    output_ranges, input_range = _shard_ranges(spec, rank)
    output_pairs = np.concatenate(
        [np.arange(start // VECTOR_LENGTH, end // VECTOR_LENGTH) for start, end in output_ranges]
    )
    row_tiles = np.concatenate([np.arange(start // ZN_N0, end // ZN_N0) for start, end in output_ranges])
    physical_columns = np.arange(input_range[0], input_range[1], dtype=np.int64)
    inverse_perm = np.argsort(tensors["perm"].numpy()).astype(np.int64)
    source_columns = inverse_perm[physical_columns]
    canonical = canonical_index_grid(tensors["packed_indices"], spec).numpy()
    local_codes = canonical[np.ix_(output_pairs, source_columns)]
    logical_n, logical_k = local_codes.shape[0] * VECTOR_LENGTH, local_codes.shape[1]
    local_tile_ids = source_columns // LUT_K
    source_books, valid_counts = np.unique(local_tile_ids, return_counts=True)
    if np.any(valid_counts > LUT_K):
        raise RuntimeError("Internal TP2 shard exceeds its canonical codebook population.")
    packed_k = int(source_books.size) * LUT_K
    padding_columns = packed_k - logical_k
    if padding_columns < 0 or packed_k % RHT_BLOCK_SIZE:
        raise RuntimeError("Internal TP2 padded K geometry is invalid.")

    # Metadata and real codes stay in LOCAL PHYSICAL order. Dummy coordinates
    # form whole appended RHT128 blocks, so they cannot mix with real inputs.
    extended_codes = np.zeros((output_pairs.size, packed_k), dtype=np.uint8)
    extended_codes[:, :logical_k] = local_codes
    extended_tile_ids = np.empty(packed_k, dtype=np.int64)
    extended_tile_ids[:logical_k] = local_tile_ids
    order = np.empty(packed_k, dtype=np.int64)
    sorted_real = np.argsort(local_tile_ids, kind="stable")
    consumed, dummy = 0, logical_k
    for block, (book, valid_count) in enumerate(zip(source_books, valid_counts)):
        count = int(valid_count)
        pad = LUT_K - count
        block_order = order[block * LUT_K : (block + 1) * LUT_K]
        block_order[:count] = sorted_real[consumed : consumed + count]
        block_order[count:] = np.arange(dummy, dummy + pad, dtype=np.int64)
        extended_tile_ids[dummy : dummy + pad] = book
        consumed += count
        dummy += pad
    if consumed != logical_k or dummy != packed_k:
        raise RuntimeError("Internal TP2 padding did not cover every local/dummy column.")

    ordered_codes = extended_codes[:, order]
    zn_pairs = ordered_codes.reshape(logical_n // ZN_N0, ZN_N0 // VECTOR_LENGTH, packed_k // ZN_K0, ZN_K0)
    zn_pairs = zn_pairs.transpose(0, 2, 3, 1)
    packed = np.ascontiguousarray(zn_pairs[..., 0::2] | (zn_pairs[..., 1::2] << np.uint8(4)))
    raw_books = tensors["codebooks"].view(torch.uint8).numpy()
    pair_lut = np.ascontiguousarray(raw_books[source_books][:, row_tiles].reshape(-1, logical_n // ZN_N0, 32))
    _check_packed_mapping(
        packed,
        pair_lut,
        order,
        canonical,
        source_columns,
        output_pairs,
        row_tiles,
        source_books,
        extended_tile_ids,
        logical_k,
        raw_books,
    )

    payload = {
        "packed_zn": torch.from_numpy(packed),
        "pair_lut": torch.from_numpy(pair_lut),
        "activation_order": torch.from_numpy(order),
    }
    for name in ("weight_scale", "weight_bias", "rht_sign"):
        values = (
            torch.ones(packed_k, dtype=torch.int8) if name == "rht_sign" else torch.zeros(packed_k, dtype=torch.float32)
        )
        values[:logical_k].copy_(tensors[name][input_range[0] : input_range[1]])
        payload[name] = values

    metadata: dict[str, Any] = {
        "format": VQ2_TP2_ZN_FORMAT,
        "tp_size": TP_SIZE,
        "tp_rank": rank,
        "kind": spec.kind,
        "canonical_shape": [spec.rows, spec.columns],
        "logical_shape": [logical_n, logical_k],
        "packed_shape": [logical_n, packed_k],
        "output_row_ranges": output_ranges,
        "input_column_range": input_range,
        "range_convention": "zero_based_half_open",
        "lut_source_tile_ids": source_books.tolist(),
        "tile_valid_counts": valid_counts.tolist(),
        "padding_columns": padding_columns,
        "tensor_shapes": {name: list(tensor.shape) for name, tensor in payload.items()},
        "rht_block_size": RHT_BLOCK_SIZE,
        "rht_true_columns": logical_k,
        "activation_semantics": {
            "physical_input": "rank-local logical input followed by padding_columns zeros",
            "dummy_metadata": {"weight_scale": 0.0, "weight_bias": 0.0, "rht_sign": 1},
            "preparation_order": [
                "physical_rht128",
                "physical_bias_gemv_and_weight_scale",
                "rank_local_dynamic_fp8",
                "byte_gather_activation_order",
            ],
            "quantization": "per-token per-expert TP-rank-local E4M3FN amax/448 with min_scale=1e-12",
            "bias_correction": "rank-local rotated-input dot weight_bias, added once to that rank projection",
            "down_aggregation": "sum rank-local down projection partials; never duplicate a full-K bias",
            "tp1_bitwise_equivalent": False,
            "floating_point_order": (
                "stable codebook regrouping changes K reduction order; TP2 changes A8 scale/reduction scope"
            ),
        },
        "validation": {"code_roundtrip_exact": True, "mapping_exact": True, "codebook_bytes_exact": True},
        "runtime_supported": False,
        "runtime_requirement": (
            "Requires a TP2 artifact loader, local-A8 preparation and TP collectives; "
            "current v3 runtime/native shape contract is TP1-only."
        ),
    }
    return payload, metadata
