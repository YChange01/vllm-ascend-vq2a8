# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed M=1 activation preparation for the opt-in V4 device route.

The resident bank already selects one metadata row per routed expert.  This
module consumes those batched rows directly instead of rebuilding dictionaries
and stacking them again.  It deliberately retains the original one-row RHT and
bias GEMVs: batching either operation changes rounding on NPU.

The optional native path is intentionally narrow: sign multiplication and input
validation only. The baseline uses the shipped ``activation_sign``; separate
strided/direct candidates consume BF16/FP32 input views and optionally write
MatMul outputs directly into final buffers. Row scaling, amax, division,
clamping, and FP8 conversion remain Torch operations because the native
quantizer has not met the byte-exact oracle.
"""

from __future__ import annotations

import torch

from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_reference import VQ2_FP8_MIN_SCALE

PACKED_GROUP_LIMIT = 6
PACKED_WIDTHS = (2048, 4096)
SIGN_FUSION_ABI = 1
STRIDED_SIGN_ABI = 1


class PackedRowwiseVQ2A8Preparation(RowwiseVQ2A8Preparation):
    """Prepare one activation row per selected expert without Python repacking."""

    def __init__(
        self, *, compact=False, validity=None, fuse_sign=False, native_ops=None, strided_sign=False, direct_output=False
    ):
        super().__init__(compact=compact, validity=validity)
        if (strided_sign and not fuse_sign) or (direct_output and not strided_sign):
            raise ValueError("Direct output requires strided sign fusion; strided sign requires fuse_sign.")
        self.fuse_sign = fuse_sign
        self.strided_sign = strided_sign
        self.direct_output = direct_output
        self._sign = None
        if not fuse_sign:
            return
        native = native_ops if native_ops is not None else torch.ops.vq2a8_ascendc_v4_v2
        try:
            version = native.activation_preparation_version()
            sign = native.activation_sign
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(
                "Packed sign fusion requires V4/v2 activation_sign ABI 1; no unfused fallback is selected implicitly."
            ) from error
        if type(version) is not int or version != SIGN_FUSION_ABI:
            raise RuntimeError(f"Unsupported packed sign fusion ABI {version}; require {SIGN_FUSION_ABI}.")
        if strided_sign:
            try:
                version = native.activation_sign_strided_version()
                sign = native.activation_sign_strided
            except (AttributeError, RuntimeError) as error:
                raise RuntimeError("Strided sign fusion requires a rebuilt library; no implicit fallback.") from error
            if type(version) is not int or version != STRIDED_SIGN_ABI:
                raise RuntimeError(f"Unsupported strided sign ABI {version}; require {STRIDED_SIGN_ABI}.")
        self._sign = sign

    def packed(self, hidden, weight_scale, weight_bias, signs, spec, *, validity=None):
        """Return contiguous ``(FP8 rows, row scales, row biases)``.

        ``hidden`` and every metadata tensor contain one row for each selected
        expert.  Only the bounded M=1 decode geometry is accepted.  General
        prefill and padded widths continue through the inherited ``many`` path.
        """

        groups, width, block = self._check_packed(hidden, weight_scale, weight_bias, signs, spec)
        # ``hidden`` may be an expanded stride-zero view in gate/up. Preserve
        # baseline materialization unless direct native view access is chosen.
        if self.strided_sign:
            self._check_strided_hidden(hidden)
            x = hidden
        else:
            x = hidden.to(dtype=torch.float32).contiguous()
        self._ensure_hadamard(hidden.device, block)

        if self.fuse_sign:
            signed, input_valid = self._sign(x, weight_scale, weight_bias, signs)
            valid = input_valid.all()
        else:
            valid = torch.isfinite(x).all() & torch.isfinite(weight_scale).all()
            valid = valid & torch.isfinite(weight_bias).all()
            valid = valid & ((signs == -1) | (signs == 1)).all()
            signed = x * signs.float()
        if validity is None:
            self._validate(valid)
        else:
            validity(valid)

        signed_blocks = signed.reshape(groups, width // block, block)
        rotated = torch.empty((groups, width), dtype=torch.float32, device=hidden.device)
        bias = torch.empty(groups, dtype=torch.float32, device=hidden.device)
        for group_index in range(groups):
            # Keep these as independent one-row GEMMs.  A batched GEMM is not
            # byte-equivalent to RowwiseVQ2A8Preparation on Ascend 950.
            if self.direct_output:
                # Identical input ranks and GEMM geometries; only destinations
                # change. NPU out= dispatch still needs independent bitwise
                # acceptance, so this remains a separate opt-in candidate.
                row = rotated[group_index : group_index + 1]
                torch.matmul(
                    signed_blocks[group_index : group_index + 1],
                    self._hadamard,
                    out=row.reshape(1, width // block, block),
                )
                torch.matmul(row, weight_bias[group_index], out=bias[group_index : group_index + 1])
            else:
                row = (signed_blocks[group_index : group_index + 1] @ self._hadamard).reshape(1, width)
                rotated[group_index : group_index + 1].copy_(row)
                row_bias = row @ weight_bias[group_index]
                bias[group_index : group_index + 1].copy_(row_bias)

        transformed = rotated * weight_scale
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = torch.clamp(transformed.abs().amax(dim=-1) / fp8_max, min=VQ2_FP8_MIN_SCALE)
        quantized = torch.clamp(transformed / scale.unsqueeze(-1), -fp8_max, fp8_max).to(torch.float8_e4m3fn)
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

    @staticmethod
    def _check_packed(hidden, weight_scale, weight_bias, signs, spec):
        if hidden.ndim != 2 or not 1 <= hidden.shape[0] <= PACKED_GROUP_LIMIT:
            raise ValueError("Packed preparation requires 1..6 activation rows, one per selected expert.")
        groups, width = hidden.shape
        if width not in PACKED_WIDTHS:
            raise ValueError("Packed preparation supports activation width 2048 or 4096 only.")
        try:
            geometry = (spec.columns, spec.rht_true_columns, spec.rht_block_size)
        except AttributeError as error:
            raise ValueError("Packed preparation requires columns, rht_true_columns, and rht_block_size.") from error
        block = geometry[2]
        if geometry[:2] != (width, width):
            raise ValueError("Packed preparation does not support padded or mismatched activation geometry.")
        if type(block) is not int or block <= 0 or block & (block - 1) or width % block:
            raise ValueError("rht_block_size must be a positive power of two dividing the activation width.")
        for name, tensor, dtype in (
            ("weight_scale", weight_scale, torch.float32),
            ("weight_bias", weight_bias, torch.float32),
            ("signs", signs, torch.int8),
        ):
            if (
                tensor.shape != (groups, width)
                or tensor.dtype != dtype
                or tensor.device != hidden.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"{name} must be contiguous {dtype}[{groups},{width}] on {hidden.device}.")
        return groups, width, block
