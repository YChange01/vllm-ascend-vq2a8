# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in resident metadata selection/sign fusion; RHT/quantizer stay unchanged."""

from __future__ import annotations

import torch

SELECT_SIGN_ABI = 1


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
