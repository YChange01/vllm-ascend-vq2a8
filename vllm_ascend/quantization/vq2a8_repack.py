# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict CPU repacking helpers for the direct TP1 VQ2A8 layout."""

from __future__ import annotations

import math

import torch

from vllm_ascend.quantization.vq2a8_artifact import (
    VQ2_CODEBOOK_SIZE,
    VQ2_INDEX_BITS,
    VQ2_INDICES_PER_WORD,
    VQ2_VECTOR_LENGTH,
    VQ2MatrixSpec,
    validate_matrix_payload,
)

VQ2_DIRECT_TP1_FORMAT = "vq2a8_direct_tp1_v1"

_REPACKED_FIELDS = {
    "packed_indices",
    "codebooks",
    "codebook_tile_ids",
    "weight_scale",
    "weight_bias",
    "rht_sign",
}
_UINT32_MASK = (1 << 32) - 1


def canonical_index_grid(
    packed_indices: torch.Tensor,
    spec: VQ2MatrixSpec,
) -> torch.Tensor:
    """Return canonical codes as uint8 ``[output_pair, input_column]``.

    The canonical artifact serializes codes in column-tile, row-tile,
    column, output-pair order. This function only changes that serialization
    into a two-dimensional logical grid; it does not apply ``perm``.
    """
    _validate_spec_geometry(spec)
    _validate_packed_tensor(
        packed_indices,
        expected_shape=(spec.packed_word_count,),
        logical_columns=spec.num_vectors,
        name="packed_indices",
    )

    flat_indices = _unpack_packed_rows(packed_indices, spec.num_vectors).reshape(-1)

    grid = torch.empty(
        (spec.row_tiles, spec.vectors_per_row_group, spec.columns),
        dtype=torch.uint8,
    )
    position = 0
    for column_tile in range(spec.column_tiles):
        column_start = column_tile * spec.group_size
        column_end = min(column_start + spec.group_size, spec.columns)
        tile_width = column_end - column_start
        tile_vector_count = spec.row_tiles * tile_width * spec.vectors_per_row_group
        tile = flat_indices[position : position + tile_vector_count].reshape(
            spec.row_tiles,
            tile_width,
            spec.vectors_per_row_group,
        )
        grid[:, :, column_start:column_end] = tile.permute(0, 2, 1)
        position += tile_vector_count
    if position != spec.num_vectors:
        raise ValueError(f"{spec.name}: canonical traversal consumed {position} codes, expected {spec.num_vectors}.")
    return grid.reshape(spec.rows // VQ2_VECTOR_LENGTH, spec.columns).contiguous()


def pack_repacked_indices(indices: torch.Tensor) -> torch.Tensor:
    """Pack uint4 codes along K into int32 ``[output_pair, ceil(K / 8)]``."""
    _require_cpu_tensor(indices, "indices")
    if indices.dtype != torch.uint8 or indices.ndim != 2:
        raise ValueError(
            "indices must be a two-dimensional uint8 CPU tensor, got "
            f"dtype={indices.dtype}, shape={tuple(indices.shape)}."
        )
    output_pairs, columns = indices.shape
    if output_pairs <= 0 or columns <= 0:
        raise ValueError(f"indices dimensions must be positive, got shape={tuple(indices.shape)}.")
    if bool((indices >= VQ2_CODEBOOK_SIZE).any()):
        raise ValueError("indices contains a value outside the uint4 range [0, 15].")

    words_per_row = math.ceil(columns / VQ2_INDICES_PER_WORD)
    padded_columns = words_per_row * VQ2_INDICES_PER_WORD
    if padded_columns == columns:
        padded = indices
    else:
        padded = torch.zeros((output_pairs, padded_columns), dtype=torch.uint8)
        padded[:, :columns] = indices
    values = padded.reshape(output_pairs, words_per_row, VQ2_INDICES_PER_WORD)
    words = torch.zeros((output_pairs, words_per_row), dtype=torch.int64)
    for nibble in range(VQ2_INDICES_PER_WORD):
        words |= values[:, :, nibble].to(torch.int64) << (nibble * VQ2_INDEX_BITS)
    return words.to(torch.int32).contiguous()


def unpack_repacked_indices(
    packed_indices: torch.Tensor,
    columns: int,
) -> torch.Tensor:
    """Unpack a direct-layout tensor into uint8 ``[output_pair, K]`` codes."""
    if not isinstance(columns, int) or isinstance(columns, bool) or columns <= 0:
        raise ValueError(f"columns must be a positive integer, got {columns!r}.")
    if not isinstance(packed_indices, torch.Tensor):
        raise TypeError(f"packed_indices must be a torch.Tensor, got {type(packed_indices).__name__}.")
    if packed_indices.ndim != 2 or packed_indices.shape[0] <= 0:
        raise ValueError(f"packed_indices must have two positive dimensions, got shape={tuple(packed_indices.shape)}.")
    expected_shape = (
        packed_indices.shape[0],
        math.ceil(columns / VQ2_INDICES_PER_WORD),
    )
    _validate_packed_tensor(
        packed_indices,
        expected_shape=expected_shape,
        logical_columns=columns,
        name="packed_indices",
    )

    return _unpack_packed_rows(packed_indices, columns)


def repack_matrix_tp1(
    tensors: dict[str, torch.Tensor],
    spec: VQ2MatrixSpec,
) -> dict[str, torch.Tensor]:
    """Convert one canonical matrix to ``vq2a8_direct_tp1_v1``.

    TP1 preserves every output pair and input column. The canonical
    permutation is absorbed into the packed code grid. Codebooks retain their
    original tile order, so ``codebook_tile_ids`` records the source tile for
    every physical input column. Normalization and RHT tensors are already in
    physical-column order and are copied without reordering.
    """
    _validate_direct_tp1_spec(spec)
    _validate_canonical_cpu_tensors(tensors, spec)
    validate_matrix_payload(tensors, spec)

    canonical_grid = canonical_index_grid(tensors["packed_indices"], spec)
    source_columns = torch.argsort(tensors["perm"].to(torch.int64))
    repacked_grid = canonical_grid[:, source_columns].contiguous()
    codebook_tile_ids = torch.div(
        source_columns,
        spec.group_size,
        rounding_mode="floor",
    ).to(torch.uint8)

    repacked = {
        "packed_indices": pack_repacked_indices(repacked_grid),
        # clone() performs a byte-preserving copy, including for FP8 payloads.
        "codebooks": tensors["codebooks"].clone(memory_format=torch.contiguous_format),
        "codebook_tile_ids": codebook_tile_ids.contiguous(),
        # These tensors are defined in physical-column order in the producer
        # contract. Only the code indices (and their tile IDs) absorb perm.
        "weight_scale": tensors["weight_scale"].clone(memory_format=torch.contiguous_format),
        "weight_bias": tensors["weight_bias"].clone(memory_format=torch.contiguous_format),
        "rht_sign": tensors["rht_sign"].clone(memory_format=torch.contiguous_format),
    }
    validate_repacked_matrix(repacked, spec)
    if not torch.equal(unpack_repacked_indices(repacked["packed_indices"], spec.columns), repacked_grid):
        raise RuntimeError(f"{spec.name}: internal direct-layout pack round-trip failed.")
    return repacked


def validate_repacked_matrix(
    tensors: dict[str, torch.Tensor],
    spec: VQ2MatrixSpec,
) -> None:
    """Validate one CPU ``vq2a8_direct_tp1_v1`` matrix payload."""
    _validate_direct_tp1_spec(spec)
    if not isinstance(tensors, dict):
        raise TypeError(f"tensors must be a dict, got {type(tensors).__name__}.")
    actual_fields = set(tensors)
    if actual_fields != _REPACKED_FIELDS:
        missing = sorted(_REPACKED_FIELDS - actual_fields)
        extra = sorted(actual_fields - _REPACKED_FIELDS)
        raise ValueError(f"{spec.name}: repacked fields differ; missing={missing}, extra={extra}.")

    for name, tensor in tensors.items():
        _require_cpu_tensor(tensor, name)
        if not tensor.is_contiguous():
            raise ValueError(f"{spec.name}.{name}: tensor must be contiguous.")

    output_pairs = spec.rows // VQ2_VECTOR_LENGTH
    expected_packed_shape = (
        output_pairs,
        math.ceil(spec.columns / VQ2_INDICES_PER_WORD),
    )
    packed_indices = tensors["packed_indices"]
    _validate_packed_tensor(
        packed_indices,
        expected_shape=expected_packed_shape,
        logical_columns=spec.columns,
        name=f"{spec.name}.packed_indices",
    )
    # Unpacking also independently exercises the direct-layout padding check.
    unpacked = unpack_repacked_indices(packed_indices, spec.columns)
    if bool((unpacked >= VQ2_CODEBOOK_SIZE).any()):
        raise ValueError(f"{spec.name}.packed_indices contains an out-of-range code.")

    expected_codebook_shape = (
        spec.column_tiles,
        spec.row_tiles,
        VQ2_CODEBOOK_SIZE,
        VQ2_VECTOR_LENGTH,
    )
    codebooks = tensors["codebooks"]
    if codebooks.dtype != torch.float8_e4m3fn or tuple(codebooks.shape) != expected_codebook_shape:
        raise ValueError(
            f"{spec.name}.codebooks: dtype={codebooks.dtype}, "
            f"shape={tuple(codebooks.shape)}; expected dtype={torch.float8_e4m3fn}, "
            f"shape={expected_codebook_shape}."
        )
    if not bool(torch.isfinite(codebooks.float()).all()):
        raise ValueError(f"{spec.name}.codebooks contains non-finite values.")

    tile_ids = tensors["codebook_tile_ids"]
    if tile_ids.dtype != torch.uint8 or tuple(tile_ids.shape) != (spec.columns,):
        raise ValueError(
            f"{spec.name}.codebook_tile_ids: dtype={tile_ids.dtype}, "
            f"shape={tuple(tile_ids.shape)}; expected dtype={torch.uint8}, "
            f"shape=({spec.columns},)."
        )
    tile_ids_int64 = tile_ids.to(torch.int64)
    if bool((tile_ids_int64 >= spec.column_tiles).any()):
        raise ValueError(f"{spec.name}.codebook_tile_ids contains a value outside [0, {spec.column_tiles}).")
    actual_tile_counts = torch.bincount(tile_ids_int64, minlength=spec.column_tiles)
    expected_tile_counts = torch.full(
        (spec.column_tiles,),
        spec.group_size,
        dtype=torch.int64,
    )
    expected_tile_counts[-1] = spec.columns - (spec.column_tiles - 1) * spec.group_size
    if not torch.equal(actual_tile_counts, expected_tile_counts):
        raise ValueError(
            f"{spec.name}.codebook_tile_ids has counts {actual_tile_counts.tolist()}, "
            f"expected {expected_tile_counts.tolist()}."
        )

    for name in ("weight_scale", "weight_bias"):
        tensor = tensors[name]
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != (spec.columns,):
            raise ValueError(
                f"{spec.name}.{name}: dtype={tensor.dtype}, shape={tuple(tensor.shape)}; "
                f"expected dtype={torch.float32}, shape=({spec.columns},)."
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{spec.name}.{name} contains non-finite values.")

    rht_sign = tensors["rht_sign"]
    if rht_sign.dtype != torch.int8 or tuple(rht_sign.shape) != (spec.columns,):
        raise ValueError(
            f"{spec.name}.rht_sign: dtype={rht_sign.dtype}, shape={tuple(rht_sign.shape)}; "
            f"expected dtype={torch.int8}, shape=({spec.columns},)."
        )
    signs = rht_sign.to(torch.int16)
    if not bool(((signs == -1) | (signs == 1)).all()):
        raise ValueError(f"{spec.name}.rht_sign contains values other than -1 and 1.")


def _validate_spec_geometry(spec: VQ2MatrixSpec) -> None:
    if not isinstance(spec, VQ2MatrixSpec):
        raise TypeError(f"spec must be a VQ2MatrixSpec, got {type(spec).__name__}.")
    if spec.rows <= 0 or spec.columns <= 0:
        raise ValueError(f"{spec.name}: rows and columns must be positive.")
    if spec.row_tiles <= 0 or spec.row_group_size <= 0 or spec.group_size <= 0:
        raise ValueError(f"{spec.name}: tile and group dimensions must be positive.")
    if spec.rows != spec.row_tiles * spec.row_group_size:
        raise ValueError(f"{spec.name}: rows do not match row tile geometry.")
    if spec.row_group_size % VQ2_VECTOR_LENGTH:
        raise ValueError(f"{spec.name}: row_group_size is not divisible by vector length.")
    if spec.column_tiles != math.ceil(spec.columns / spec.group_size):
        raise ValueError(f"{spec.name}: columns do not match column tile geometry.")
    expected_vectors = spec.rows * spec.columns // VQ2_VECTOR_LENGTH
    if spec.num_elements != spec.rows * spec.columns or spec.num_vectors != expected_vectors:
        raise ValueError(f"{spec.name}: vector or element counts do not match matrix geometry.")
    if spec.original_shape != (spec.rows, spec.rht_true_columns):
        raise ValueError(f"{spec.name}: original_shape does not match rows and true columns.")


def _validate_direct_tp1_spec(spec: VQ2MatrixSpec) -> None:
    _validate_spec_geometry(spec)
    if not (spec.enable_permutation and spec.enable_normalization and spec.enable_rht):
        raise ValueError(f"{spec.name}: {VQ2_DIRECT_TP1_FORMAT} requires permutation, normalization, and RHT.")
    if spec.norm_dimension != 0:
        raise ValueError(f"{spec.name}: norm_dimension must be zero.")
    if spec.rht_block_size <= 0 or spec.rht_block_size & (spec.rht_block_size - 1):
        raise ValueError(f"{spec.name}: rht_block_size must be a positive power of two.")
    if spec.columns % spec.rht_block_size:
        raise ValueError(f"{spec.name}: rht_block_size must divide columns.")
    if not 0 < spec.rht_true_columns <= spec.columns:
        raise ValueError(f"{spec.name}: rht_true_columns must be in (0, columns].")
    if spec.column_tiles > torch.iinfo(torch.uint8).max + 1:
        raise ValueError(f"{spec.name}: {spec.column_tiles} codebook tiles cannot be represented by uint8 tile IDs.")


def _validate_canonical_cpu_tensors(
    tensors: dict[str, torch.Tensor],
    spec: VQ2MatrixSpec,
) -> None:
    if not isinstance(tensors, dict):
        raise TypeError(f"tensors must be a dict, got {type(tensors).__name__}.")
    for name, tensor in tensors.items():
        _require_cpu_tensor(tensor, name)
        if not tensor.is_contiguous():
            raise ValueError(f"{spec.name}.{name}: canonical tensor must be contiguous.")


def _validate_packed_tensor(
    tensor: torch.Tensor,
    *,
    expected_shape: tuple[int, ...],
    logical_columns: int,
    name: str,
) -> None:
    _require_cpu_tensor(tensor, name)
    if tensor.dtype != torch.int32 or tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name}: dtype={tensor.dtype}, shape={tuple(tensor.shape)}; "
            f"expected dtype={torch.int32}, shape={expected_shape}."
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name}: tensor must be contiguous.")
    remainder = logical_columns % VQ2_INDICES_PER_WORD
    if remainder == 0:
        return
    words = tensor.reshape(-1, tensor.shape[-1]).to(torch.int64) & _UINT32_MASK
    padding = words[:, -1] >> (remainder * VQ2_INDEX_BITS)
    if bool((padding != 0).any()):
        raise ValueError(f"{name}: unused high padding nibbles must be zero.")


def _require_cpu_tensor(tensor: object, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor, got device={tensor.device}.")


def _unpack_packed_rows(
    packed_indices: torch.Tensor,
    logical_columns: int,
) -> torch.Tensor:
    """Unpack validated words with bounded temporary memory."""
    words = packed_indices.reshape(-1, packed_indices.shape[-1]).to(torch.int64) & _UINT32_MASK
    unpacked = torch.empty(
        (words.shape[0], words.shape[1] * VQ2_INDICES_PER_WORD),
        dtype=torch.uint8,
    )
    for nibble in range(VQ2_INDICES_PER_WORD):
        unpacked[:, nibble::VQ2_INDICES_PER_WORD] = ((words >> (nibble * VQ2_INDEX_BITS)) & (VQ2_CODEBOOK_SIZE - 1)).to(
            torch.uint8
        )
    return unpacked[:, :logical_columns].contiguous()
