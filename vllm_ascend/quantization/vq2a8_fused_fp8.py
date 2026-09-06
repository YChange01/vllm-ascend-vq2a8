# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in VQ decode -> E4M3 Cube prototype. NOT a model backend.

The kernel consumes the frozen packed artifact and already prepared A8 rows.
It decodes only a [32,128] weight tile and feeds that tile directly to an
FP8 dot, with FP32 accumulation and the original per-row scale/bias epilogue.
No dense expert tensor or decode scratch tensor is allocated by the wrapper.
Compiler-inserted scratch/spills and actual Ascend CV transfers still require
hardware/IR inspection; source-level fusion is not proof of on-chip execution.

The Ascend M=32 internal tile is masked for M=1..32. CUDA development uses
M=64 internally for SM90 native FP8 WGMMA (M=32 lowered to FP16 MMA on the
developer compiler); this does not certify Ascend M=32 lowering.
The K=128 tile keeps packed
int32 row loads (64 bytes), FP8 loads (128 bytes), codebook rows (32 bytes)
and BF16 output rows (64 bytes) aligned. This is a new reduction geometry,
not an assertion of bitwise equivalence to the accepted K=512 Vector MAC.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.quantization.vq2a8_kernel_contract import validate_vq2a8_tp1_m1_inputs

BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 128
MAX_COLUMN_TILES = 32


def validate_fused_fp8_inputs(activation, scale, bias, packed, codebooks, tile_ids):
    """Metadata-only validation; artifact/preparer must validate tensor values."""
    if not isinstance(activation, torch.Tensor) or activation.ndim != 2:
        raise ValueError("Fused prototype activation requires [M,K].")
    rows = activation.shape[0]
    if not 1 <= rows <= BLOCK_M or not activation.is_contiguous():
        raise ValueError("Fused prototype requires contiguous activation with M in [1,32].")
    for name, tensor in (("scale", scale), ("bias", bias)):
        if not isinstance(tensor, torch.Tensor) or tensor.shape != (rows,) or not tensor.is_contiguous():
            raise ValueError(f"Fused prototype {name} requires contiguous [M].")
    shape = validate_vq2a8_tp1_m1_inputs(activation[:1], scale[:1], bias[:1], packed, codebooks, tile_ids)
    if shape.column_tiles > MAX_COLUMN_TILES:
        raise ValueError(f"Fused prototype supports at most {MAX_COLUMN_TILES} column tiles.")
    return shape


@triton.jit
def _vq_decode_cube_kernel(
    X,
    SCALE,
    BIAS,
    PACKED,
    CODEBOOK,
    TILE_IDS,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    COLUMN_TILES: tl.constexpr,
    TABLE_TILES: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    ASCEND: tl.constexpr,
):
    tl.static_assert(BK == 128, "Keep the initial aligned, bounded CV geometry.")
    group = tl.program_id(0)
    rows = tl.arange(0, BM)
    outputs = tl.arange(0, 32)
    columns = tl.arange(0, BK)
    pairs = tl.arange(0, 16)
    words = tl.arange(0, BK // 8)
    nibbles = tl.arange(0, 8)
    tiles = tl.arange(0, TABLE_TILES)
    entries = tl.arange(0, 32)
    # Keep the GM pointer FP8-typed. Reinterpret loaded VALUES only, avoiding
    # the unsupported FP8->byte pointer cast in A5 memory-scope inference.
    table = (
        tl.load(
            CODEBOOK + tiles[:, None] * N + group * 32 + entries[None, :],
            mask=tiles[:, None] < COLUMN_TILES,
            other=0.0,
        )
        .to(tl.uint8, bitcast=True)
        .to(tl.float32)
    )
    # A shared FP32 carrier preserves byte values 0..255 exactly; it is not
    # a dequantized weight table and is NOT broadcast across output rows.
    table = tl.reshape(table, (TABLE_TILES * 32,))
    accumulator = tl.zeros((BM, 32), tl.float32)
    for start in range(0, K, BK):
        packed = tl.load(PACKED + (group * 16 + pairs[:, None]) * (K // 8) + start // 8 + words[None, :])
        codes = (packed[:, :, None] >> (nibbles[None, None, :] * 4)) & 15
        codes = tl.reshape(codes, (16, BK))
        codes = tl.reshape(tl.broadcast_to(codes[:, None, :], (16, 2, BK)), (32, BK))
        tile = tl.load(TILE_IDS + start + columns).to(tl.int32)
        lookup = tile[None, :] * 32 + codes * 2 + outputs[:, None] % 2
        selected = tl.gather(table, tl.reshape(lookup, (32 * BK,)), axis=0)
        weights = tl.reshape(selected, (32, BK)).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        activation = tl.load(X + rows[:, None] * K + start + columns[None, :], mask=rows[:, None] < M, other=0.0)
        # The decoded E4M3 tensor is the actual dot operand, not a diagnostic
        # side branch. No FP32 Vector multiply/reduce or dense GM fallback.
        if ASCEND:
            accumulator = tl.dot_scaled(
                activation,
                None,
                "e4m3",
                tl.trans(weights),
                None,
                "e4m3",
                acc=accumulator,
                out_dtype=tl.float32,
            )
        else:
            # SM90 WGMMA's default permits long reduced-precision partial
            # accumulation. Bound it before adding partials in FP32; otherwise
            # real prepared-A8 expert inputs fail the unchanged CPU oracle.
            accumulator = tl.dot(
                activation, tl.trans(weights), acc=accumulator, out_dtype=tl.float32, max_num_imprecise_acc=32
            )
    scale = tl.load(SCALE + rows, mask=rows < M, other=0.0)
    bias = tl.load(BIAS + rows, mask=rows < M, other=0.0)
    result = accumulator * scale[:, None] + bias[:, None]
    tl.store(Y + rows[:, None] * N + group * 32 + outputs[None, :], result, mask=rows[:, None] < M)


def launch_fused_fp8(activation, scale, bias, packed, codebooks, tile_ids):
    """Return output and compiled kernel so isolated probes can retain codegen."""
    shape = validate_fused_fp8_inputs(activation, scale, bias, packed, codebooks, tile_ids)
    if activation.device.type not in ("npu", "cuda"):
        raise ValueError("Fused prototype requires an accelerator; no CPU fallback.")
    output = torch.empty((activation.shape[0], shape.size_n), dtype=torch.bfloat16, device=activation.device)
    compiled = _vq_decode_cube_kernel[(shape.size_n // BLOCK_N,)](
        activation,
        scale,
        bias,
        packed,
        codebooks,
        tile_ids,
        output,
        M=activation.shape[0],
        N=shape.size_n,
        K=shape.size_k,
        COLUMN_TILES=shape.column_tiles,
        TABLE_TILES=triton.next_power_of_2(shape.column_tiles),
        BK=BLOCK_K,
        BM=BLOCK_M if activation.device.type == "npu" else 64,
        ASCEND=activation.device.type == "npu",
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, compiled


def vq2a8_fused_fp8(activation, scale, bias, packed, codebooks, tile_ids):
    return launch_fused_fp8(activation, scale, bias, packed, codebooks, tile_ids)[0]


@triton.jit
def _cube_bridge_kernel(
    X,
    W,
    Y,
    M: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    BRIDGE: tl.constexpr,
    ASCEND: tl.constexpr,
):
    """Bounded synthetic control with the same M/N/K tiling as the prototype."""
    rows, outputs, columns = tl.arange(0, BM), tl.arange(0, 32), tl.arange(0, BK)
    accumulator = tl.zeros((BM, 32), tl.float32)
    for start in range(0, K, BK):
        a = tl.load(X + rows[:, None] * K + start + columns[None, :], mask=rows[:, None] < M, other=0.0)
        b = tl.load(W + outputs[:, None] * K + start + columns[None, :])
        if BRIDGE:
            b = (b.to(tl.uint8, bitcast=True) ^ 128).to(tl.float8e4nv, bitcast=True)
        if ASCEND:
            accumulator = tl.dot_scaled(
                a, None, "e4m3", tl.trans(b), None, "e4m3", acc=accumulator, out_dtype=tl.float32
            )
        else:
            accumulator = tl.dot(a, tl.trans(b), acc=accumulator, out_dtype=tl.float32, max_num_imprecise_acc=32)
    tl.store(Y + rows[:, None] * 32 + outputs[None, :], accumulator, mask=rows[:, None] < M)


def launch_cube_control(activation, weight, *, bridge=False):
    """Synthetic-only direct/CV control, never invoked by the model/prototype."""
    if not isinstance(activation, torch.Tensor) or activation.ndim != 2 or not 1 <= activation.shape[0] <= BLOCK_M:
        raise ValueError("Cube control requires [M,K], M in [1,32].")
    if not isinstance(weight, torch.Tensor) or weight.shape != (BLOCK_N, activation.shape[1]):
        raise ValueError("Cube control weight requires [32,K].")
    if activation.shape[1] <= 0 or activation.shape[1] % BLOCK_K:
        raise ValueError("Cube control K must be positive and divisible by 128.")
    for tensor in (activation, weight):
        if tensor.dtype != torch.float8_e4m3fn or not tensor.is_contiguous() or tensor.data_ptr() % 32:
            raise ValueError("Cube control requires aligned contiguous E4M3 tensors.")
        if tensor.device != activation.device or tensor.device.type not in ("npu", "cuda"):
            raise ValueError("Cube control requires one accelerator device.")
    output = torch.empty((activation.shape[0], BLOCK_N), dtype=torch.bfloat16, device=activation.device)
    compiled = _cube_bridge_kernel[(1,)](
        activation,
        weight,
        output,
        M=activation.shape[0],
        K=activation.shape[1],
        BK=BLOCK_K,
        BM=BLOCK_M if activation.device.type == "npu" else 64,
        BRIDGE=bridge,
        ASCEND=activation.device.type == "npu",
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, compiled
