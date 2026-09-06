# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit FP32 FMA for root inverse RoPE; no FP8 dot or Vector/Cube bridge."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _root_fma_fp32_kernel(A, B, C, OUT, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < SIZE
    a = tl.load(A + offsets, mask=mask, other=0)
    b = tl.load(B + offsets, mask=mask, other=0)
    c = tl.load(C + offsets, mask=mask, other=0)
    tl.store(OUT + offsets, tl.fma(a, b, c), mask=mask)


def root_fma_fp32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """One FP32 rounding for a*b+c, including on the A5 Vector path.

    Inputs to this small kernel are contiguous FP32 vectors. In particular,
    c is an already-rounded partner product, computed by a separate eager
    multiply. The compiler cannot reassociate the two RoPE products. CUDA
    support exists to compare this exact kernel with the original SM90 ops.
    """
    if a.device.type not in ("npu", "cuda") or any(t.device != a.device for t in (b, c)):
        raise ValueError("Root FMA requires tensors on the same accelerator.")
    if any(t.dtype != torch.float32 for t in (a, b, c)):
        raise ValueError("Root FMA requires FP32 operands.")
    a, b, c = (t.contiguous() for t in torch.broadcast_tensors(a, b, c))
    result = torch.empty_like(a)
    if a.numel():
        _root_fma_fp32_kernel[(triton.cdiv(a.numel(), 1024),)](
            a, b, c, result, SIZE=a.numel(), BLOCK=1024, enable_fp_fusion=False
        )
    return result
