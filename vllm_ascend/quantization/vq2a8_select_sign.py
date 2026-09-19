# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in resident metadata selection/sign fusion; RHT/quantizer stay unchanged."""

from __future__ import annotations

import math

import torch

SELECT_SIGN_ABI = 1
SWIGLU_SELECT_SIGN_ABI = 1


class FusedSelectSign:
    """No exposed pointer table, host tensor reads or implicit fallback."""

    def __init__(self, native_ops=None):
        native = torch.ops.vq2a8_ascendc_v4_v2 if native_ops is None else native_ops
        try:
            version = native.select_sign_version()
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError("Fused select/sign requires rebuilt ABI 1; no implicit fallback.") from error
        if type(version) is not int or version != SELECT_SIGN_ABI:
            raise RuntimeError(f"Unsupported select/sign ABI {version}; require {SELECT_SIGN_ABI}.")

    def __call__(self, bank, hidden, slots):
        if (
            not isinstance(hidden, torch.Tensor)
            or not isinstance(slots, torch.Tensor)
            or hidden.ndim != 2
            or hidden.dtype not in (torch.bfloat16, torch.float32)
            or not 1 <= hidden.shape[0] <= 6
            or hidden.shape[1] not in (2048, 4096)
            or hidden.stride(1) != 1
            or (hidden.stride(0) != 0 and hidden.stride(0) < hidden.shape[1])
            or hidden.data_ptr() % 32
            or (hidden.shape[0] > 1 and hidden.stride(0) * hidden.element_size() % 32)
            or slots.ndim != 1
            or slots.dtype != torch.int64
            or slots.numel() != hidden.shape[0]
            or slots.device != hidden.device
            or not slots.is_contiguous()
        ):
            raise ValueError("Select/sign requires aligned BF16/FP32[G,K] and contiguous INT64[G], G1..6/K2048,4096.")
        try:
            operation = bank.select_sign
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError("Resident bank lacks select_sign ABI 1; no implicit fallback.") from error
        return operation(hidden, slots)


class FusedSwigluSelectSign:
    """I: tested native BF16 SwiGLU plus select/sign; never falls back eagerly."""

    def __init__(self, native_ops=None):
        native = torch.ops.vq2a8_ascendc_v4_v2 if native_ops is None else native_ops
        try:
            version = native.swiglu_select_sign_version()
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError("Fused SwiGLU/select/sign requires rebuilt ABI 1; no implicit fallback.") from error
        if type(version) is not int or version != SWIGLU_SELECT_SIGN_ABI:
            raise RuntimeError(f"Unsupported SwiGLU/select/sign ABI {version}; require {SWIGLU_SELECT_SIGN_ABI}.")

    def __call__(self, bank, gate_up, slots, swiglu_limit):
        if (
            not isinstance(gate_up, torch.Tensor)
            or not isinstance(slots, torch.Tensor)
            or gate_up.ndim != 2
            or gate_up.dtype != torch.bfloat16
            or not 1 <= gate_up.shape[0] <= 6
            or gate_up.shape[1] not in (4096, 8192)
            or gate_up.stride(1) != 1
            or (gate_up.stride(0) != 0 and gate_up.stride(0) < gate_up.shape[1])
            or gate_up.data_ptr() % 32
            or (gate_up.shape[0] > 1 and gate_up.stride(0) * gate_up.element_size() % 32)
            or slots.ndim != 1
            or slots.dtype != torch.int64
            or slots.numel() != gate_up.shape[0]
            or slots.device != gate_up.device
            or not slots.is_contiguous()
        ):
            raise ValueError("SwiGLU/select/sign requires aligned BF16[G,2K] and INT64[G], G1..6/K2048,4096.")
        if swiglu_limit is not None and (
            isinstance(swiglu_limit, bool)
            or not isinstance(swiglu_limit, int | float)
            or not math.isfinite(swiglu_limit)
            or not 0 <= swiglu_limit <= torch.finfo(torch.float32).max
        ):
            raise ValueError("SwiGLU/select/sign limit must be None or a finite nonnegative FP32 scalar.")
        try:
            operation = bank.swiglu_select_sign
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError("Resident bank lacks swiglu_select_sign ABI 1; no implicit fallback.") from error
        # Existing Torch contract uses no clamp for either None or zero.
        return operation(gate_up, slots, 0.0 if swiglu_limit is None else float(swiglu_limit))
