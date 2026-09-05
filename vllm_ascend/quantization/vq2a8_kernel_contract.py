# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor contract for the TP1, one-row packed VQ2A8 kernel."""

from __future__ import annotations

from dataclasses import dataclass

import torch

VQ2_CODEBOOK_SIZE = 16
VQ2_VECTOR_LENGTH = 2
VQ2_INDICES_PER_WORD = 8
VQ2_INDEX_BITS = 4
VQ2_ROW_GROUP_SIZE = 32
VQ2_BLOCK_M = 16
VQ2_BLOCK_N = 32
VQ2_BLOCK_K = 64


@dataclass(frozen=True)
class VQ2A8M1Shape:
    """Validated dimensions for the TP1, one-row packed GEMM kernel."""

    size_n: int
    size_k: int
    column_tiles: int
    row_tiles: int


def _require_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    dtype: torch.dtype,
    ndim: int,
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must use {dtype}, got {tensor.dtype}.")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape={tuple(tensor.shape)}.")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def validate_vq2a8_tp1_m1_inputs(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    bias_correction: torch.Tensor,
    packed_indices: torch.Tensor,
    codebooks: torch.Tensor,
    codebook_tile_ids: torch.Tensor,
) -> VQ2A8M1Shape:
    """Validate the exact tensor contract consumed by the bring-up kernel."""
    if not isinstance(activation, torch.Tensor):
        raise TypeError(f"activation must be a torch.Tensor, got {type(activation).__name__}.")
    device = activation.device
    _require_tensor(
        activation,
        "activation",
        dtype=torch.bfloat16,
        ndim=2,
        device=device,
    )
    _require_tensor(
        activation_scale,
        "activation_scale",
        dtype=torch.float32,
        ndim=1,
        device=device,
    )
    _require_tensor(
        bias_correction,
        "bias_correction",
        dtype=torch.float32,
        ndim=1,
        device=device,
    )
    _require_tensor(
        packed_indices,
        "packed_indices",
        dtype=torch.int32,
        ndim=2,
        device=device,
    )
    _require_tensor(
        codebooks,
        "codebooks",
        dtype=torch.bfloat16,
        ndim=4,
        device=device,
    )
    _require_tensor(
        codebook_tile_ids,
        "codebook_tile_ids",
        dtype=torch.uint8,
        ndim=1,
        device=device,
    )

    if tuple(activation.shape[:1]) != (1,):
        raise ValueError(f"The M=1 kernel requires activation shape [1, K], got {tuple(activation.shape)}.")
    if activation_scale.numel() != 1 or bias_correction.numel() != 1:
        raise ValueError("activation_scale and bias_correction must each contain one value for M=1.")

    size_k = activation.shape[1]
    size_n = packed_indices.shape[0] * VQ2_VECTOR_LENGTH
    if size_k <= 0 or size_k % VQ2_BLOCK_K:
        raise ValueError(f"VQ2A8 kernel K must be positive and divisible by {VQ2_BLOCK_K}, got {size_k}.")
    if size_n <= 0 or size_n % VQ2_BLOCK_N:
        raise ValueError(f"VQ2A8 kernel N must be positive and divisible by {VQ2_BLOCK_N}, got {size_n}.")
    if packed_indices.shape[1] * VQ2_INDICES_PER_WORD != size_k:
        raise ValueError(f"packed_indices must cover K exactly: shape={tuple(packed_indices.shape)}, K={size_k}.")
    if codebook_tile_ids.shape != (size_k,):
        raise ValueError(f"codebook_tile_ids must have shape ({size_k},), got {tuple(codebook_tile_ids.shape)}.")

    column_tiles, row_tiles, code_count, vector_length = codebooks.shape
    expected_row_tiles = size_n // VQ2_ROW_GROUP_SIZE
    if column_tiles <= 0:
        raise ValueError("codebooks must contain at least one column tile.")
    if row_tiles != expected_row_tiles:
        raise ValueError(
            f"codebooks row tile count must be N/{VQ2_ROW_GROUP_SIZE}={expected_row_tiles}, got {row_tiles}."
        )
    if code_count != VQ2_CODEBOOK_SIZE or vector_length != VQ2_VECTOR_LENGTH:
        raise ValueError(
            "codebooks must have trailing shape "
            f"({VQ2_CODEBOOK_SIZE}, {VQ2_VECTOR_LENGTH}), got {tuple(codebooks.shape[-2:])}."
        )
    return VQ2A8M1Shape(
        size_n=size_n,
        size_k=size_k,
        column_tiles=column_tiles,
        row_tiles=row_tiles,
    )
