# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-4 experimental packed Vector kernel, never selected by serving.

The accepted M=1 kernel remains in vq2a8_triton.py. This candidate loads an
aligned codebook table per 16-output program, then gathers from that shared
one-dimensional on-chip tensor. It does not gather single-byte GM addresses
or create a dense expert matrix in GM. Each activation row retains the accepted K=512 reduction
geometry; multiple rows share a launch, not their activation quantization.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.quantization.vq2a8_kernel_contract import validate_vq2a8_tp1_m1_inputs

# The shared 1-D FP32 table is at most 32 * 32 * 4 = 4 KiB, not
# broadcast to every output. Indices/decoded tiles are [16,512]. These
# source-level bounds are not a compiler UB allocation proof.
MAX_COLUMN_TILES = 32
MAX_BATCH_ROWS = 32
# 16 BF16 outputs give aligned 32-byte stores; retain full 32-byte FP8
# table loads. Only split N, keeping each K=512 reduction intact.
OUTPUTS_PER_PROGRAM = 16


def validate_vector_gather_inputs(activation, scale, bias, packed, codebooks, tile_ids):
    """Host metadata only; values must come from the validated artifact/preparer."""
    if not isinstance(activation, torch.Tensor) or activation.ndim != 2:
        raise ValueError("Experimental Vector activation requires [M,K].")
    rows = activation.shape[0]
    if not 1 <= rows <= MAX_BATCH_ROWS:
        raise ValueError(f"Experimental Vector M must be in [1,{MAX_BATCH_ROWS}].")
    if not activation.is_contiguous():
        raise ValueError("Experimental Vector activation must be contiguous.")
    for name, tensor in (("scale", scale), ("bias", bias)):
        if not isinstance(tensor, torch.Tensor) or tensor.shape != (rows,) or not tensor.is_contiguous():
            raise ValueError(f"Experimental Vector {name} requires contiguous [M].")
    shape = validate_vq2a8_tp1_m1_inputs(activation[:1], scale[:1], bias[:1], packed, codebooks, tile_ids)
    if shape.column_tiles > MAX_COLUMN_TILES:
        raise ValueError(f"Experimental Vector supports at most {MAX_COLUMN_TILES} column tiles.")
    return shape


@triton.jit
def _packed_vector_gather_kernel(
    X,
    SCALE,
    BIAS,
    PACKED,
    CODEBOOK,
    TILE_IDS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    COLUMN_TILES: tl.constexpr,
    TABLE_TILES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(BLOCK_N == 16, "Use the bounded, aligned phase-4 output tile.")
    output_block, row = tl.program_id(0), tl.program_id(1)
    group = output_block // (32 // BLOCK_N)
    outputs = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    pairs = tl.arange(0, BLOCK_N // 2)
    words = tl.arange(0, 64)
    nibbles = tl.arange(0, 8)
    columns = tl.arange(0, 512)
    table_tiles = tl.arange(0, TABLE_TILES)
    table_entries = tl.arange(0, 32)
    # Affine [tiles,32] FP8 GM transfer, then reinterpret loaded values as
    # bytes, as in the accepted Ascend kernel. Do not cast the GM pointer:
    # it left an unrealized pointer conversion in the failing A5 scope IR.
    # Ascend gather rejects int32 sources.
    # FP32 exactly represents every byte value (0..255); this is an on-chip
    # byte carrier, NOT FP8 dequantization or a resident FP32 codebook.
    table = (
        tl.load(
            CODEBOOK + table_tiles[:, None] * (N // 32 * 32) + group * 32 + table_entries[None, :],
            mask=table_tiles[:, None] < COLUMN_TILES,
            # A float zero is also valid for the FP8 masked-load fallback;
            # an int32 zero is not supported by every frontend's FP8 cast.
            other=0.0,
        )
        .to(tl.uint8, bitcast=True)
        .to(tl.float32)
    )
    table = tl.reshape(table, (TABLE_TILES * 32,))
    accumulator = tl.zeros((BLOCK_N,), tl.float32)
    for start in range(0, K, 512):
        packed = tl.load(
            PACKED + (output_block * (BLOCK_N // 2) + pairs[:, None]) * (K // 8) + start // 8 + words[None, :]
        )
        codes = (packed[:, :, None] >> (nibbles[None, None, :] * 4)) & 15
        codes = tl.reshape(codes, (BLOCK_N // 2, 512))
        codes = tl.reshape(tl.broadcast_to(codes[:, None, :], (BLOCK_N // 2, 2, 512)), (BLOCK_N, 512))
        tile = tl.load(TILE_IDS + start + columns).to(tl.int32)
        lookup = tile[None, :] * 32 + codes * 2 + outputs[:, None] % 2
        # All outputs index the SAME table. Flatten just the indices so
        # gather does not need an N-times-replicated source tensor in UB.
        selected = tl.gather(table, tl.reshape(lookup, (BLOCK_N * 512,)), axis=0)
        # Restore the selected byte before interpreting its E4M3 bits.
        weights = tl.reshape(selected, (BLOCK_N, 512)).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        x = tl.load(X + row * K + start + columns)
        accumulator += tl.sum(weights.to(tl.float32) * x[None, :].to(tl.float32), axis=1)
    scale, bias = tl.load(SCALE + row), tl.load(BIAS + row)
    tl.store(Y + row * N + outputs, accumulator * scale + bias)


def vq2a8_packed_vector_gather(activation, scale, bias, packed, codebooks, tile_ids):
    """Experimental M<=32 Vector MAC; FP8 storage does not mean FP8 dot."""
    shape = validate_vector_gather_inputs(activation, scale, bias, packed, codebooks, tile_ids)
    if activation.device.type not in ("npu", "cuda"):
        raise ValueError("Experimental Vector kernel requires an accelerator.")
    output = torch.empty((activation.shape[0], shape.size_n), device=activation.device, dtype=torch.bfloat16)
    _packed_vector_gather_kernel[(shape.size_n // OUTPUTS_PER_PROGRAM, activation.shape[0])](
        activation,
        scale,
        bias,
        packed,
        codebooks,
        tile_ids,
        output,
        N=shape.size_n,
        K=shape.size_k,
        COLUMN_TILES=shape.column_tiles,
        TABLE_TILES=triton.next_power_of_2(shape.column_tiles),
        BLOCK_N=OUTPUTS_PER_PROGRAM,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
