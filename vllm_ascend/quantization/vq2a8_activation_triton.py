# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RHT128 + per-row E4M3 preparation. Experimental Ascend device code."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _transform(X, WS, WB, SIGN, JOB, Z, MAX, BIAS, K: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, 128)
    column = block * 128 + lane
    job = tl.load(JOB + row)
    value = tl.load(X + row * K + column).to(tl.float32)
    value *= tl.load(SIGN + job * K + column).to(tl.float32)
    for stage in tl.static_range(7):
        stride = 1 << stage
        other = tl.gather(value, lane ^ stride, axis=0)
        value = tl.where((lane & stride) == 0, value + other, other - value)
    value *= 0.08838834764831845  # 1/sqrt(128), normalized Sylvester RHT
    bias = tl.sum(value * tl.load(WB + job * K + column), axis=0)
    value *= tl.load(WS + job * K + column)
    tl.store(Z + row * K + column, value)
    tl.store(MAX + row * (K // 128) + block, tl.max(tl.abs(value), axis=0))
    tl.store(BIAS + row * (K // 128) + block, bias)


@triton.jit
def _quantize(Z, MAX, BIAS, Q, SCALE, CORRECTION, K: tl.constexpr, BK: tl.constexpr, BP: tl.constexpr):
    row = tl.program_id(0)
    p = tl.arange(0, BP)
    maximum = tl.max(tl.load(MAX + row * (K // 128) + p, p < K // 128, other=0), axis=0)
    bias = tl.sum(tl.load(BIAS + row * (K // 128) + p, p < K // 128, other=0), axis=0)
    scale = tl.maximum(maximum / 448.0, 1.0e-12)
    column = tl.arange(0, BK)
    value = tl.load(Z + row * K + column, column < K, other=0) / scale
    value = tl.minimum(tl.maximum(value, -448.0), 448.0)
    tl.store(Q + row * K + column, value.to(tl.float8e4nv), column < K)
    tl.store(SCALE + row, scale)
    tl.store(CORRECTION + row, bias)


def prepare(x, weight_scale, weight_bias, sign, row_jobs):
    rows, width = x.shape
    transformed = torch.empty((rows, width), dtype=torch.float32, device=x.device)
    partial_max = torch.empty((rows, width // 128), dtype=torch.float32, device=x.device)
    partial_bias = torch.empty_like(partial_max)
    q = torch.empty((rows, width), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
    bias = torch.empty_like(scale)
    # FMA disabled: butterfly/bias sums have a declared, reproducible expression.
    # This still does NOT promise equality with baseline GEMM accumulation.
    _transform[(rows, width // 128)](
        x,
        weight_scale,
        weight_bias,
        sign,
        row_jobs,
        transformed,
        partial_max,
        partial_bias,
        width,
        enable_fp_fusion=False,
    )
    _quantize[(rows,)](
        transformed,
        partial_max,
        partial_bias,
        q,
        scale,
        bias,
        width,
        triton.next_power_of_2(width),
        triton.next_power_of_2(width // 128),
        enable_fp_fusion=False,
    )
    return q, scale, bias
