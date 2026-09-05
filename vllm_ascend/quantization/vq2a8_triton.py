# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton bring-up kernel for the frozen TP1 VQ2A8 artifact.

The kernel consumes one prepared activation row and one expert matrix.  It
performs packed-index lookup on the device and feeds decoded tiles directly
to ``tl.dot`` without materializing a dense weight.

The FP8 activation and codebook values are supplied as BF16 mirrors during
bring-up.  Every FP8 value is exactly representable in BF16, while this avoids
making the first packed-lookup gate depend on backend-specific FP8 ``tl.dot``
lowering.  A later optimized kernel can change the compute storage type after
this address-generation contract is proven on Ascend 950.
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
    VQ2_ROW_GROUP_SIZE,
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
    stride_cbi,
    stride_cbc,
    stride_on,
    SIZE_N: tl.constexpr,
    SIZE_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CODEBOOK_SIZE: tl.constexpr,
    VECTOR_LENGTH: tl.constexpr,
    INDICES_PER_WORD: tl.constexpr,
    INDEX_BITS: tl.constexpr,
    ROW_GROUP_SIZE: tl.constexpr,
):
    output_block = tl.program_id(0)
    offsets_n = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start_k in range(0, SIZE_K, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        activation_vector = tl.load(activation_ptr + offsets_k * stride_ak)
        activation = tl.broadcast_to(
            activation_vector[None, :],
            (BLOCK_M, BLOCK_K),
        )

        vector_rows = offsets_n // VECTOR_LENGTH
        words = tl.load(
            packed_indices_ptr + vector_rows[:, None] * stride_pn + (offsets_k[None, :] // INDICES_PER_WORD) * stride_pk
        )
        shifts = (offsets_k[None, :] % INDICES_PER_WORD) * INDEX_BITS
        codes = (words >> shifts) & (CODEBOOK_SIZE - 1)
        source_tiles = tl.load(codebook_tile_ids_ptr + offsets_k).to(tl.int32)
        codebook_offsets = (
            source_tiles[None, :] * stride_cbt
            + (offsets_n // ROW_GROUP_SIZE)[:, None] * stride_cbr
            + codes * stride_cbi
            + (offsets_n % VECTOR_LENGTH)[:, None] * stride_cbc
        )
        weights = tl.load(codebooks_ptr + codebook_offsets)
        accumulator += tl.dot(activation, tl.trans(weights))

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

    ``activation`` and ``codebooks`` are BF16 mirrors of already-quantized
    FP8 values.  No dense ``[N, K]`` weight is created on either host or
    device by this function.
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
        codebooks.stride(2),
        codebooks.stride(3),
        output.stride(1),
        SIZE_N=shape.size_n,
        SIZE_K=shape.size_k,
        BLOCK_M=VQ2_BLOCK_M,
        BLOCK_N=VQ2_BLOCK_N,
        BLOCK_K=VQ2_BLOCK_K,
        CODEBOOK_SIZE=VQ2_CODEBOOK_SIZE,
        VECTOR_LENGTH=VQ2_VECTOR_LENGTH,
        INDICES_PER_WORD=VQ2_INDICES_PER_WORD,
        INDEX_BITS=VQ2_INDEX_BITS,
        ROW_GROUP_SIZE=VQ2_ROW_GROUP_SIZE,
    )
    return output
