# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-4 experimental packed Vector kernel, never selected by serving.

The accepted M=1 kernel remains in vq2a8_triton.py. This candidate loads an
aligned codebook table once per output group, then gathers from that on-chip
tensor. It does not gather single-byte GM addresses or create a dense expert
matrix in GM. Each activation row retains the accepted K=512 reduction
geometry; multiple rows share a launch, not their activation quantization.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.quantization.vq2a8_kernel_contract import validate_vq2a8_tp1_m1_inputs

# Bound the on-chip lookup tile: [32, 32 * 32] float32 = 128 KiB, plus
# indices/decoded tile/temporaries. This is not a compiler UB allocation proof.
MAX_COLUMN_TILES = 32
MAX_BATCH_ROWS = 32


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
):
    group, row = tl.program_id(0), tl.program_id(1)
    outputs = tl.arange(0, 32)
    pairs = tl.arange(0, 16)
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
    table = tl.broadcast_to(table[None, :], (32, TABLE_TILES * 32))
    accumulator = tl.zeros((32,), tl.float32)
    for start in range(0, K, 512):
        packed = tl.load(PACKED + (group * 16 + pairs[:, None]) * (K // 8) + start // 8 + words[None, :])
        codes = (packed[:, :, None] >> (nibbles[None, None, :] * 4)) & 15
        codes = tl.reshape(codes, (16, 512))
        codes = tl.reshape(tl.broadcast_to(codes[:, None, :], (16, 2, 512)), (32, 512))
        tile = tl.load(TILE_IDS + start + columns).to(tl.int32)
        lookup = tile[None, :] * 32 + codes * 2 + outputs[:, None] % 2
        # Restore the selected byte before interpreting its E4M3 bits.
        weights = tl.gather(table, lookup, axis=1).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        x = tl.load(X + row * K + start + columns)
        accumulator += tl.sum(weights.to(tl.float32) * x[None, :].to(tl.float32), axis=1)
    scale, bias = tl.load(SCALE + row), tl.load(BIAS + row)
    tl.store(Y + row * N + group * 32 + outputs, accumulator * scale + bias)


def vq2a8_packed_vector_gather(activation, scale, bias, packed, codebooks, tile_ids):
    """Experimental M<=32 Vector MAC; FP8 storage does not mean FP8 dot."""
    shape = validate_vector_gather_inputs(activation, scale, bias, packed, codebooks, tile_ids)
    if activation.device.type not in ("npu", "cuda"):
        raise ValueError("Experimental Vector kernel requires an accelerator.")
    output = torch.empty((activation.shape[0], shape.size_n), device=activation.device, dtype=torch.bfloat16)
    _packed_vector_gather_kernel[(shape.size_n // 32, activation.shape[0])](
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
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
