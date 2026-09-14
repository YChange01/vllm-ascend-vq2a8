# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in synthetic Cube/CV microtests; never imported by model execution.

Run each variant in a separate child process. Past A5 mixed kernels aborted
with MTE alignment errors; a PASS here does not certify packed expert GEMM.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.quantization.vq2a8_fp8_cube import ascend_fp8_dot_unit_scale


@triton.jit
def _fp8_cube_micro(A, B, Y, BRIDGE: tl.constexpr, ASCEND: tl.constexpr):
    rows, columns = tl.arange(0, 32), tl.arange(0, 512)
    a = tl.load(A + rows[:, None] * 512 + columns[None, :])
    b = tl.load(B + rows[:, None] * 512 + columns[None, :])
    if BRIDGE:
        # Runtime byte XOR is a genuine Vector producer of a Cube operand.
        # No packed decode, dynamic GM gather or tiny unaligned transfer.
        b = (b.to(tl.uint8, bitcast=True) ^ 128).to(tl.float8e4nv, bitcast=True)
    if ASCEND:
        result = ascend_fp8_dot_unit_scale(a, tl.trans(b), tl.zeros((32, 32), tl.float32))
    else:
        result = tl.dot(a, tl.trans(b), out_dtype=tl.float32)
    tl.store(Y + rows[:, None] * 32 + rows[None, :], result)


def fp8_cube_micro(a, b, *, bridge=False):
    for t in (a, b):
        if (
            t.shape != (32, 512)
            or t.dtype != torch.float8_e4m3fn
            or not t.is_contiguous()
            or t.device != a.device
            or t.data_ptr() % 32
        ):
            raise ValueError("FP8 micro requires aligned contiguous E4M3 [32,512] operands on one device.")
    if a.device.type not in ("cuda", "npu"):
        raise ValueError("FP8 micro requires an accelerator.")
    result = torch.empty((32, 32), dtype=torch.bfloat16, device=a.device)
    _fp8_cube_micro[(1,)](a, b, result, BRIDGE=bridge, ASCEND=a.device.type == "npu", num_warps=4)
    return result
