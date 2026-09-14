# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integer single-rounding FP32 FMA for root inverse RoPE.

The tested A5 compiler lowers both eager addcmul and tl.fma with a rounding
fingerprint different from the SM90 reference. Do not use floating FMA or
FP64 here: keep the product exact and round the integer sum to nearest-even.
This is a correctness path, not a native FP8 dot or a Vector/Cube bridge.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _highest_bit(x):
    # x is a nonnegative int64. Returning zero for x=0 is intentional;
    # zero significands and exact cancellation are handled separately.
    bit = tl.full(x.shape, 0, tl.int64)
    for power in tl.static_range(5, -1, -1):
        shift = 1 << power
        high = x >> shift
        take = high != 0
        bit += tl.where(take, shift, 0)
        x = tl.where(take, high, x)
    return bit


@triton.jit
def _shift_right_jam(x, distance):
    # Preserve a sticky bit for ALL discarded bits. Clamp even the unused
    # tl.where branch so neither backend can emit an undefined >=64 shift.
    shift = tl.minimum(tl.maximum(distance, 0), 62)
    one = tl.full(x.shape, 1, tl.int64)
    discarded = (x & ((one << shift) - 1)) != 0
    shifted = (x >> shift) | discarded.to(tl.int64)
    return tl.where(distance >= 63, (x != 0).to(tl.int64), shifted)


@triton.jit
def _normalized_significand(bits):
    field = (bits >> 23) & 255
    mantissa = (bits & 0x7FFFFF) | tl.where(field != 0, 0x800000, 0)
    shift = 23 - _highest_bit(mantissa)
    exponent = tl.where(field != 0, field, 1) - 127 - shift
    return mantissa << shift, exponent


@triton.jit
def _single_rounding_fma_bits(a, b, c):
    ab = a.to(tl.uint32, bitcast=True).to(tl.int64)
    bb = b.to(tl.uint32, bitcast=True).to(tl.int64)
    cb = c.to(tl.uint32, bitcast=True).to(tl.int64)
    am, ae = _normalized_significand(ab)
    bm, be = _normalized_significand(bb)
    cm, ce = _normalized_significand(cb)
    product_sign = ((ab ^ bb) >> 31) != 0
    c_sign = (cb >> 31) != 0

    # 24x24 -> exact 48-bit product. Put its top bit at 60 or 61, and
    # the addend's top bit at 61. Signed sum fits int64 (magnitude <2**63).
    # Cancellation of nearby terms loses no information; distant terms
    # retain a sticky bit below the guard/round bits.
    pe = ae + be + 1
    exponent = tl.maximum(pe, ce)
    product = _shift_right_jam((am * bm) << 14, exponent - pe)
    addend = _shift_right_jam(cm << 38, exponent - ce)
    signed = tl.where(product_sign, -product, product) + tl.where(c_sign, -addend, addend)
    negative = signed < 0
    magnitude = tl.where(negative, -signed, signed)
    top = _highest_bit(magnitude)
    out_exp = exponent + top - 61

    # Normal: retain 24 bits. Subnormal: round at the fixed 2**-149
    # quantum. Ties choose the even retained bit, including underflow.
    distance = tl.maximum(top - 23, -exponent - 88)
    shift = tl.minimum(tl.maximum(distance, 0), 62)
    one = tl.full(magnitude.shape, 1, tl.int64)
    retained = magnitude >> shift
    remainder = magnitude & ((one << shift) - 1)
    half = one << tl.maximum(shift - 1, 0)
    increment = (distance > 0) & ((remainder > half) | ((remainder == half) & ((retained & 1) != 0)))
    rounded = retained + increment.to(tl.int64)
    left = magnitude << tl.minimum(tl.maximum(-distance, 0), 62)
    rounded = tl.where(distance < 0, left, rounded)
    rounded = tl.where(distance >= 63, 0, rounded)
    carry = rounded >= 0x1000000
    rounded = tl.where(carry, rounded >> 1, rounded)
    out_exp += carry.to(tl.int64)
    encoded = tl.where(out_exp < -126, rounded, ((out_exp + 127) << 23) | (rounded & 0x7FFFFF))
    encoded = tl.where(out_exp > 127, 0x7F800000, encoded)
    encoded |= negative.to(tl.int64) << 31
    # Exact cancellation is +0; two negative zero terms produce -0.
    encoded = tl.where(magnitude == 0, (product_sign & c_sign).to(tl.int64) << 31, encoded)

    ap, bp, cp = ab & 0x7FFFFFFF, bb & 0x7FFFFFFF, cb & 0x7FFFFFFF
    product_zero = (ap == 0) | (bp == 0)
    encoded = tl.where(product_zero & (cp != 0), cb, encoded)
    product_inf = (ap == 0x7F800000) | (bp == 0x7F800000)
    encoded = tl.where(product_inf, (product_sign.to(tl.int64) << 31) | 0x7F800000, encoded)
    encoded = tl.where(cp == 0x7F800000, cb, encoded)
    invalid = (
        (ap > 0x7F800000)
        | (bp > 0x7F800000)
        | (cp > 0x7F800000)
        | (product_inf & product_zero)
        | (product_inf & (cp == 0x7F800000) & (product_sign != c_sign))
    )
    # NaNs are canonicalized; matching reference NaN payloads is not a contract.
    return tl.where(invalid, 0x7FC00000, encoded).to(tl.uint32)


@triton.jit
def _root_fma_fp32_kernel(A, B, C, OUT, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < SIZE
    a = tl.load(A + offsets, mask=mask, other=0)
    b = tl.load(B + offsets, mask=mask, other=0)
    c = tl.load(C + offsets, mask=mask, other=0)
    result = _single_rounding_fma_bits(a, b, c).to(tl.float32, bitcast=True)
    tl.store(OUT + offsets, result, mask=mask)


def root_fma_fp32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Round a*b+c once using integer significands on the accelerator.

    Inputs to this small kernel are contiguous FP32 vectors. In particular,
    c is an already-rounded partner product, computed by a separate eager
    multiply. No floating multiply/add exists inside this kernel to fuse or
    reassociate. CUDA support tests the algorithm against hardware FMA;
    acceptance on the installed A5 compiler/device is a separate gate.
    """
    if a.device.type not in ("npu", "cuda") or any(t.device != a.device for t in (b, c)):
        raise ValueError("Root FMA requires tensors on the same accelerator.")
    if any(t.dtype != torch.float32 for t in (a, b, c)):
        raise ValueError("Root FMA requires FP32 operands.")
    a, b, c = (t.contiguous() for t in torch.broadcast_tensors(a, b, c))
    result = torch.empty_like(a)
    if a.numel():
        _root_fma_fp32_kernel[(triton.cdiv(a.numel(), 256),)](a, b, c, result, SIZE=a.numel(), BLOCK=256)
    return result
