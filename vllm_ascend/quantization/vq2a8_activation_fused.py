# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in V4/v2 pointwise fusion, preserving the one-row RHT/bias GEMMs.

The native operators fuse signing/input checks and post-RHT row quantization.
They do not use a different Hadamard algorithm or cache metadata validity.
Reference preparation remains the default and the numerical acceptance oracle.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation


class FusedV4V2Preparation(RowwiseVQ2A8Preparation):
    """Two native vector launches surrounding unchanged row-wise GEMMs."""

    def __init__(self, *, compact=True, validity=None, native_ops=None):
        super().__init__(compact=compact, validity=validity)
        self._native = native_ops if native_ops is not None else torch.ops.vq2a8_ascendc_v4_v2
        try:
            version = self._native.activation_preparation_version()
            self._sign = self._native.activation_sign
            self._quantize = self._native.activation_quantize
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError("Fused activation preparation requires a rebuilt V4/v2 native library.") from error
        if type(version) is not int or version != 1:
            raise RuntimeError(f"Unsupported fused activation preparation ABI {version}; require 1.")

    def rows(self, hidden, payload, spec):
        return self.many([(hidden, payload, spec)])[0]

    def __call__(self, activation, weight_scale, weight_bias, rht_sign, rht_block_size):
        self._check_metadata(activation, weight_scale, weight_bias, rht_sign, rht_block_size)
        spec = SimpleNamespace(
            columns=activation.shape[1], rht_true_columns=activation.shape[1], rht_block_size=rht_block_size
        )
        payload = {"weight_scale": weight_scale, "weight_bias": weight_bias, "rht_sign": rht_sign}
        return self.rows(activation, payload, spec)

    def many(self, requests, *, validity=None):
        if not 1 <= len(requests) <= 6:
            raise ValueError("Grouped preparation requires 1..6 projections.")
        first, _, spec = requests[0]
        width, block = spec.columns, spec.rht_block_size
        if width not in (2048, 4096):
            raise ValueError("Fused V4/v2 activation preparation requires width 2048 or 4096.")
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
        # Match conversion-before-cat for mixed inputs, including padded rows.
        compact = self.compact and len({value[0].dtype for value in values}) == 1
        x = torch.cat([value[0] if compact else value[0].float() for value in values]).float()
        weight_scale, weight_bias, signs = (torch.stack([value[i] for value in values]) for i in (1, 2, 3))
        self._ensure_hadamard(first.device, block)
        signed, input_valid = self._sign(x, weight_scale, weight_bias, signs)
        signed = signed.reshape(-1, width // block, block)
        # Intentionally identical to RowwiseVQ2A8Preparation.many. Neither
        # batched GEMM nor FWHT is substituted for these rounding-sensitive ops.
        rotated = [(row @ self._hadamard).reshape(1, width) for row in signed.split(1)]
        bias = torch.cat([row @ values[i][2] for i, row in enumerate(rotated)])
        quantized, scale, output_valid = self._quantize(torch.cat(rotated), weight_scale, bias)
        valid = (input_valid & output_valid).all()
        if validity is None:
            self._validate(valid)
        else:
            validity(valid)
        return list(zip(quantized.split(counts), scale.split(counts), bias.split(counts)))
