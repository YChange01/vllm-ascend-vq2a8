# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native E4M3 Triton kernel for the frozen TP1 VQ2A8 artifact.

The kernel consumes one prepared activation row and one expert matrix.  It
performs packed-index lookup on the device and feeds decoded tiles directly
to ``tl.dot_scaled`` without materializing a dense weight.

The activation and codebooks remain E4M3 from artifact load through Cube
execution.  Packed words and complete 32-byte codebook rows are loaded at
aligned addresses; nibble and codebook selection happens on chip before the
raw E4M3 bytes enter ``tl.dot_scaled`` with FP32 accumulation.  This avoids the
unaligned dynamic one-byte gathers that Ascend's MTE cannot execute.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.quantization.vq2a8_kernel_contract import (
    VQ2_BLOCK_K,
    VQ2_BLOCK_M,
    VQ2_BLOCK_N,
    VQ2_CODEBOOK_SIZE,
    VQ2_INDEX_BITS,
    VQ2_INDICES_PER_WORD,
    VQ2_VECTOR_LENGTH,
    validate_vq2a8_tp1_m1_inputs,
)


@triton.jit
def _vq2a8_tp1_m1_packed_gemm_kernel(
    activation_ptr,
    activation_scale_ptr,
    bias_correction_ptr,
    packed_indices_ptr,
    codebooks_ptr,
    codebook_tile_ids_ptr,
    output_ptr,
    stride_ak,
    stride_pn,
    stride_pk,
    stride_cbt,
    stride_cbr,
    stride_on,
    SIZE_N: tl.constexpr,
    SIZE_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COLUMN_TILES: tl.constexpr,
    CODEBOOK_SIZE: tl.constexpr,
    VECTOR_LENGTH: tl.constexpr,
    INDICES_PER_WORD: tl.constexpr,
    INDEX_BITS: tl.constexpr,
):
    output_block = tl.program_id(0)
    offsets_n = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    pair_offsets = tl.arange(0, BLOCK_N // VECTOR_LENGTH)
    packed_word_offsets = tl.arange(0, BLOCK_K // INDICES_PER_WORD)
    nibble_offsets = tl.arange(0, INDICES_PER_WORD)
    table_offsets = tl.arange(0, CODEBOOK_SIZE * VECTOR_LENGTH)
    components = offsets_n % VECTOR_LENGTH

    for start_k in range(0, SIZE_K, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        activation_vector = tl.load(activation_ptr + offsets_k * stride_ak)
        activation_bytes = activation_vector.to(tl.uint8, bitcast=True)
        activation_e4m3 = tl.broadcast_to(
            activation_bytes[None, :],
            (BLOCK_M, BLOCK_K),
        )

        # Load one contiguous packed-word run for every output pair.  Unlike
        # expanding k//8 in the pointer expression, this is an affine 2-D
        # transfer whose tail is 256 bytes and whose row starts are aligned.
        packed_words = tl.load(
            packed_indices_ptr
            + (output_block * (BLOCK_N // VECTOR_LENGTH) + pair_offsets[:, None]) * stride_pn
            + (start_k // INDICES_PER_WORD + packed_word_offsets[None, :]) * stride_pk
        )
        pair_codes = (packed_words[:, :, None] >> (nibble_offsets[None, None, :] * INDEX_BITS)) & (CODEBOOK_SIZE - 1)
        pair_codes = tl.reshape(pair_codes, (BLOCK_N // VECTOR_LENGTH, BLOCK_K))
        codes = tl.broadcast_to(
            pair_codes[:, None, :],
            (BLOCK_N // VECTOR_LENGTH, VECTOR_LENGTH, BLOCK_K),
        )
        codes = tl.reshape(codes, (BLOCK_N, BLOCK_K))
        source_tiles = tl.load(codebook_tile_ids_ptr + offsets_k).to(tl.int32)

        # Each table load starts at a 32-byte boundary and transfers all
        # 16x2 E4M3 entries.  Selection happens in registers, so no MTE load
        # uses the unaligned address ``table + code * 2 + component``.
        weights = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.uint8)
        for source_tile in range(0, COLUMN_TILES):
            table = tl.load(codebooks_ptr + source_tile * stride_cbt + output_block * stride_cbr + table_offsets).to(
                tl.uint8, bitcast=True
            )
            for code in range(0, CODEBOOK_SIZE):
                even_value = tl.sum(
                    tl.where(table_offsets == code * VECTOR_LENGTH, table, 0),
                    axis=0,
                ).to(tl.uint8)
                odd_value = tl.sum(
                    tl.where(table_offsets == code * VECTOR_LENGTH + 1, table, 0),
                    axis=0,
                ).to(tl.uint8)
                codebook_value = tl.where(components == 0, even_value, odd_value)
                selected = (source_tiles[None, :] == source_tile) & (codes == code)
                weights = tl.where(selected, codebook_value[:, None], weights)
        # Both operands already contain their plain E4M3 values.  E8M0 value
        # 127 is exactly 1.0, so these per-32-element scales leave the dot
        # product unchanged; the model's dynamic activation scale is applied
        # once, after accumulation, by the Python wrapper below.
        lhs_unit_scale = tl.full((BLOCK_M, BLOCK_K // 32), 127, dtype=tl.uint8)
        rhs_unit_scale = tl.full((BLOCK_N, BLOCK_K // 32), 127, dtype=tl.uint8)
        accumulator = tl.dot_scaled(
            activation_e4m3,
            lhs_unit_scale,
            "e4m3",
            tl.trans(weights),
            rhs_unit_scale,
            "e4m3",
            acc=accumulator,
            out_dtype=tl.float32,
        )

    first_row_mask = tl.arange(0, BLOCK_M)[:, None] == 0
    first_row = tl.sum(accumulator * first_row_mask, axis=0)
    first_row = tl.reshape(first_row, (BLOCK_N,))
    scale = tl.load(activation_scale_ptr)
    bias = tl.load(bias_correction_ptr)
    output = first_row * scale + bias
    tl.store(output_ptr + offsets_n * stride_on, output)


def vq2a8_tp1_m1_packed_gemm(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    bias_correction: torch.Tensor,
    packed_indices: torch.Tensor,
    codebooks: torch.Tensor,
    codebook_tile_ids: torch.Tensor,
) -> torch.Tensor:
    """Run one TP1 expert projection directly from packed VQ2 tensors.

    ``activation`` and ``codebooks`` stay ``torch.float8_e4m3fn``.  No dense
    ``[N, K]`` weight is created on either host or device by this function.
    """
    shape = validate_vq2a8_tp1_m1_inputs(
        activation,
        activation_scale,
        bias_correction,
        packed_indices,
        codebooks,
        codebook_tile_ids,
    )
    if activation.device.type not in {"cuda", "npu"}:
        raise ValueError(f"The VQ2A8 packed kernel requires a CUDA or NPU tensor, got {activation.device}.")

    output = torch.empty(
        (1, shape.size_n),
        device=activation.device,
        dtype=torch.bfloat16,
    )
    grid = (triton.cdiv(shape.size_n, VQ2_BLOCK_N),)
    _vq2a8_tp1_m1_packed_gemm_kernel[grid](
        activation,
        activation_scale,
        bias_correction,
        packed_indices,
        codebooks,
        codebook_tile_ids,
        output,
        activation.stride(1),
        packed_indices.stride(0),
        packed_indices.stride(1),
        codebooks.stride(0),
        codebooks.stride(1),
        output.stride(1),
        SIZE_N=shape.size_n,
        SIZE_K=shape.size_k,
        BLOCK_M=VQ2_BLOCK_M,
        BLOCK_N=VQ2_BLOCK_N,
        BLOCK_K=VQ2_BLOCK_K,
        COLUMN_TILES=shape.column_tiles,
        CODEBOOK_SIZE=VQ2_CODEBOOK_SIZE,
        VECTOR_LENGTH=VQ2_VECTOR_LENGTH,
        INDICES_PER_WORD=VQ2_INDICES_PER_WORD,
        INDEX_BITS=VQ2_INDEX_BITS,
    )
    return output
