# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit identity microscales for the Triton-Ascend 3.2.2/A5 prototype.

This is a compiler-call contract, not a native-instruction certification.
The installed 3.2.2 frontend requires byte scale tensors. Its A5 FP8 scale
layout is [M,K/16], [N,K/16]; this must not be inferred from newer upstream
Triton, which has a different scale contract. E8M0 byte 127 represents one.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def ascend_fp8_dot_unit_scale(lhs, rhs, accumulator):
    """E4M3 dot with mandatory unit scales; no A8 requantization or GM input."""
    scale_k: tl.constexpr = 16
    unit_e8m0: tl.constexpr = 127
    tl.static_assert(lhs.dtype == tl.float8e4nv and rhs.dtype == tl.float8e4nv)
    tl.static_assert(lhs.shape[1] == rhs.shape[0] and lhs.shape[1] % 64 == 0)
    # The RHS is [K,N], but its scale is [N,K/16], not [K/16,N].
    lhs_scale = tl.full((lhs.shape[0], lhs.shape[1] // scale_k), unit_e8m0, tl.uint8)
    rhs_scale = tl.full((rhs.shape[1], rhs.shape[0] // scale_k), unit_e8m0, tl.uint8)
    return tl.dot_scaled(lhs, lhs_scale, "e4m3", rhs, rhs_scale, "e4m3", acc=accumulator, out_dtype=tl.float32)


def ascend_fp8_unit_scale_contract(block_m=32, block_n=32, block_k=128):
    """Report the source contract only; generated memory placement is unreviewed."""
    return {
        "target": "triton-ascend-3.2.2-a5",
        "dtype": "uint8",
        "format": "e8m0",
        "identity_byte": 127,
        "scale_k": 16,
        "lhs_shape": [block_m, block_k // 16],
        "rhs_shape": [block_n, block_k // 16],
        "source": "kernel_constant",
        "row_scale_bias": "unchanged_epilogue",
    }
