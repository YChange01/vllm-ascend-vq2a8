# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-wise preparation for the opt-in AscendC execution policy.

Retain the reference's arithmetic and every input-value check. Combine the
four host validity decisions into one device scalar read and reuse the RHT
matrix within one layer runtime. No weight or activation validity is cached.
The independent reference implementation remains unchanged.
"""

from __future__ import annotations

import torch

from vllm_ascend.quantization.vq2a8_reference import VQ2_FP8_MIN_SCALE, _sylvester_hadamard


class RowwiseVQ2A8Preparation:
    """One bounded constant cache owned by the single-stream offline runtime."""

    def __init__(self, *, compact=False):
        self._key = None
        self._hadamard = None
        self.compact = compact

    def rows(self, hidden, payload, spec):
        if self.compact:
            return self.many([(hidden, payload, spec)])[0]
        prepared = []
        for row in hidden.split(1):
            if spec.columns != spec.rht_true_columns:
                row = torch.nn.functional.pad(row, (0, spec.columns - spec.rht_true_columns))
            with torch.device("cpu"):
                prepared.append(
                    self(row, payload["weight_scale"], payload["weight_bias"], payload["rht_sign"], spec.rht_block_size)
                )
        if not prepared:
            raise ValueError("AscendC preparation requires at least one row.")
        if len(prepared) == 1:
            return tuple(value.contiguous() for value in prepared[0])
        return tuple(torch.cat(values, dim=0).contiguous() for values in zip(*prepared))

    def many(self, requests):
        """Batch pointwise work, but keep every RHT and bias GEMV one-row.

        All inputs are revalidated on every call, in one collective decision.
        There is no cached sign/scale/bias validity or transformed weight copy.
        At most six projections of <=32 rows share a same-width workspace.
        """
        if not 1 <= len(requests) <= 6:
            raise ValueError("Grouped preparation requires 1..6 projections.")
        first, _, spec = requests[0]
        width, block = spec.columns, spec.rht_block_size
        values, counts = [], []
        for hidden, payload, current in requests:
            if (
                (current.columns, current.rht_true_columns, current.rht_block_size)
                != (width, spec.rht_true_columns, block)
                or hidden.ndim != 2
                or not 1 <= hidden.shape[0] <= 32
                or hidden.shape[1] != current.rht_true_columns
                or hidden.device != first.device
                or not 0 < current.rht_true_columns <= width
            ):
                raise ValueError("Grouped preparation requires matching geometry/device and 1..32 rows per projection.")
            counts.append(hidden.shape[0])
            for row in hidden.split(1):
                if width != current.rht_true_columns:
                    row = torch.nn.functional.pad(row, (0, width - current.rht_true_columns))
                self._check_metadata(row, payload["weight_scale"], payload["weight_bias"], payload["rht_sign"], block)
                values.append((row, payload["weight_scale"], payload["weight_bias"], payload["rht_sign"]))
        # One conversion instead of one allocation/launch per row. Mixed input
        # dtypes retain the old conversion-before-concatenation contract.
        compact = self.compact and len({value[0].dtype for value in values}) == 1
        x = torch.cat([value[0] if compact else value[0].float() for value in values]).float()
        weight_scale, weight_bias, rht_sign = (torch.stack([value[i] for value in values]) for i in (1, 2, 3))
        valid = torch.isfinite(x).all() & torch.isfinite(weight_scale).all() & torch.isfinite(weight_bias).all()
        # Equality with -1/+1 is exact in int8; no widening buffer is needed.
        signs = rht_sign if self.compact else rht_sign.to(torch.int16)
        valid = valid & ((signs == -1) | (signs == 1)).all()
        if not bool(valid):
            raise ValueError("Invalid activation/weight_scale/weight_bias (non-finite) or rht_sign (not -1/+1).")
        self._ensure_hadamard(first.device, block)
        signed = x.reshape(-1, width // block, block) * rht_sign.float().reshape(-1, width // block, block)
        # Do not turn these into a batched GEMM: its rounding may differ on NPU.
        rotated = [(row @ self._hadamard).reshape(1, width) for row in signed.split(1)]
        bias = torch.cat([row @ values[i][2] for i, row in enumerate(rotated)])
        transformed = torch.cat(rotated) * weight_scale
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = torch.clamp(transformed.abs().amax(dim=-1) / fp8_max, min=VQ2_FP8_MIN_SCALE)
        quantized = torch.clamp(transformed / scale.unsqueeze(-1), -fp8_max, fp8_max).to(torch.float8_e4m3fn)
        return list(zip(quantized.split(counts), scale.split(counts), bias.split(counts)))

    def __call__(self, activation, weight_scale, weight_bias, rht_sign, rht_block_size):
        self._check_metadata(activation, weight_scale, weight_bias, rht_sign, rht_block_size)
        width = activation.shape[1]
        x = activation.float()
        valid = torch.isfinite(x).all() & torch.isfinite(weight_scale).all() & torch.isfinite(weight_bias).all()
        signs = rht_sign.to(torch.int16)
        valid = valid & ((signs == -1) | (signs == 1)).all()
        if not bool(valid):
            raise ValueError("Invalid activation/weight_scale/weight_bias (non-finite) or rht_sign (not -1/+1).")

        self._ensure_hadamard(activation.device, rht_block_size)
        blocks = x.reshape(1, width // rht_block_size, rht_block_size)
        blocks = blocks * rht_sign.float().reshape(width // rht_block_size, rht_block_size)
        rotated = (blocks @ self._hadamard).reshape(1, width)
        bias_correction = rotated @ weight_bias.float()
        transformed = rotated * weight_scale.float().unsqueeze(0)
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = torch.clamp(transformed.abs().amax(dim=-1) / fp8_max, min=VQ2_FP8_MIN_SCALE)
        quantized = torch.clamp(transformed / scale.unsqueeze(-1), -fp8_max, fp8_max).to(torch.float8_e4m3fn)
        return quantized, scale, bias_correction

    @staticmethod
    def _check_metadata(activation, weight_scale, weight_bias, rht_sign, rht_block_size):
        if activation.ndim != 2 or activation.shape[0] != 1 or activation.shape[1] <= 0:
            raise ValueError("AscendC preparation requires exactly one nonempty activation row.")
        width = activation.shape[1]
        if (
            type(rht_block_size) is not int
            or rht_block_size <= 0
            or rht_block_size & (rht_block_size - 1)
            or width % rht_block_size
        ):
            raise ValueError("rht_block_size must be a positive power of two dividing the activation width.")
        for name, tensor, dtype in (
            ("weight_scale", weight_scale, torch.float32),
            ("weight_bias", weight_bias, torch.float32),
            ("rht_sign", rht_sign, torch.int8),
        ):
            if tensor.shape != (width,) or tensor.dtype != dtype or tensor.device != activation.device:
                raise ValueError(f"{name} must be {dtype}[{width}] on {activation.device}.")

    def _ensure_hadamard(self, device, rht_block_size):
        key = (device, rht_block_size)
        if self._key != key:
            # Never generate the FP64 Sylvester matrix on the default NPU.
            with torch.device("cpu"):
                hadamard = _sylvester_hadamard(rht_block_size)
            self._hadamard = hadamard.to(device=device, dtype=torch.float32)
            self._key = key
