# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from vllm_ascend.quantization.vq2a8_kernel_contract import (
    VQ2A8M1Shape,
    validate_vq2a8_tp1_m1_inputs,
)


def _inputs() -> tuple[torch.Tensor, ...]:
    size_n = 64
    size_k = 128
    return (
        torch.zeros((1, size_k), dtype=torch.bfloat16),
        torch.ones(1, dtype=torch.float32),
        torch.zeros(1, dtype=torch.float32),
        torch.zeros((size_n // 2, size_k // 8), dtype=torch.int32),
        torch.zeros((2, size_n // 32, 16, 2), dtype=torch.bfloat16),
        torch.zeros(size_k, dtype=torch.uint8),
    )


def test_validate_tp1_m1_packed_kernel_contract() -> None:
    shape = validate_vq2a8_tp1_m1_inputs(*_inputs())

    assert shape == VQ2A8M1Shape(
        size_n=64,
        size_k=128,
        column_tiles=2,
        row_tiles=2,
    )


def test_validate_tp1_m1_rejects_multiple_rows() -> None:
    values = list(_inputs())
    values[0] = values[0].expand(2, -1).contiguous()

    with pytest.raises(ValueError, match="M=1"):
        validate_vq2a8_tp1_m1_inputs(*values)


def test_validate_tp1_m1_rejects_partial_packed_k() -> None:
    values = list(_inputs())
    values[3] = values[3][:, :-1].contiguous()

    with pytest.raises(ValueError, match="cover K exactly"):
        validate_vq2a8_tp1_m1_inputs(*values)


def test_validate_tp1_m1_rejects_wrong_row_tiles() -> None:
    values = list(_inputs())
    values[4] = torch.zeros((2, 1, 16, 2), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="row tile count"):
        validate_vq2a8_tp1_m1_inputs(*values)


def test_validate_tp1_m1_requires_bf16_fp8_mirrors() -> None:
    values = list(_inputs())
    values[4] = values[4].float()

    with pytest.raises(ValueError, match="codebooks must use torch.bfloat16"):
        validate_vq2a8_tp1_m1_inputs(*values)
