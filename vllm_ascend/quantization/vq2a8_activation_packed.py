# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact row-wise resident activation preparation for the VQ2A8 decode path.

The native bank fuses metadata selection and sign multiplication. RHT, bias
GEMVs and FP8 rounding retain their accepted Torch order and geometry.
"""

from __future__ import annotations

import torch

from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_reference import VQ2_FP8_MIN_SCALE
from vllm_ascend.quantization.vq2a8_select_sign import FusedSelectSign

PACKED_GROUP_LIMIT = 6
PACKED_WIDTHS = (2048, 4096)


class PackedRowwiseVQ2A8Preparation(RowwiseVQ2A8Preparation):
    """Use fused resident selection for decode and rowwise Torch for prefill."""

    def __init__(self, *, compact=False, validity=None, native_ops=None):
        super().__init__(compact=compact, validity=validity)
        self._select_sign = FusedSelectSign(native_ops=native_ops)

    def packed_resident(self, bank, hidden, slots, spec, *, validity, raw_statuses=None):
        """Retain fresh selection and input statuses without materializing signs."""
        if self._select_sign is None:
            raise ValueError("Resident preparation requires explicit select/sign fusion.")
        if hidden.ndim != 2 or not 1 <= hidden.shape[0] <= PACKED_GROUP_LIMIT:
            raise ValueError("Resident preparation requires 1..6 rows.")
        self._check_strided_hidden(hidden)
        width = hidden.shape[1]
        if width not in PACKED_WIDTHS or (spec.columns, spec.rht_true_columns, spec.rht_block_size) != (
            width,
            width,
            128,
        ):
            raise ValueError("Resident preparation requires unpadded K2048/4096 and RHT128.")
        signed, weight_scale, weight_bias, select_valid, input_valid = self._select_sign(bank, hidden, slots)
        if raw_statuses is not None:
            raw_statuses.extend((select_valid, input_valid))
        else:
            validity((select_valid != 0).all())
            validity(input_valid.all())
        self._ensure_hadamard(hidden.device, spec.rht_block_size)
        return self._from_signed(signed, weight_scale, weight_bias, spec)

    def _from_signed(self, signed, weight_scale, weight_bias, spec):
        groups, width = signed.shape
        block = spec.rht_block_size
        signed_blocks = signed.reshape(groups, width // block, block)
        rotated = torch.empty((groups, width), dtype=torch.float32, device=signed.device)
        bias = torch.empty(groups, dtype=torch.float32, device=signed.device)
        for group_index in range(groups):
            # Keep these as independent one-row GEMMs.  A batched GEMM is not
            # byte-equivalent to RowwiseVQ2A8Preparation on Ascend 950.
            row = rotated[group_index : group_index + 1]
            torch.matmul(
                signed_blocks[group_index : group_index + 1],
                self._hadamard,
                out=row.reshape(1, width // block, block),
            )
            torch.matmul(row, weight_bias[group_index], out=bias[group_index : group_index + 1])

        transformed = rotated * weight_scale
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = torch.clamp(transformed.abs().amax(dim=-1) / fp8_max, min=VQ2_FP8_MIN_SCALE)
        normalized = transformed / scale.unsqueeze(-1)
        quantized = torch.clamp(normalized, -fp8_max, fp8_max).to(torch.float8_e4m3fn)
        return quantized.contiguous(), scale.contiguous(), bias.contiguous()

    @staticmethod
    def _check_strided_hidden(hidden):
        if hidden.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Strided sign input requires BF16 or FP32.")
        row_stride, column_stride = hidden.stride()
        if column_stride != 1 or (row_stride != 0 and row_stride < hidden.shape[1]):
            raise ValueError("Strided sign input requires unit columns and expanded or non-overlapping rows.")
        if hidden.data_ptr() % 32 or (hidden.shape[0] > 1 and (row_stride * hidden.element_size()) % 32):
            raise ValueError("Strided sign input base and row stride require 32-byte alignment.")
