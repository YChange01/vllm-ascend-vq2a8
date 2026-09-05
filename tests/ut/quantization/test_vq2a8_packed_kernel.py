# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from vllm_ascend.quantization.vq2a8_kernel_contract import (
    VQ2_BLOCK_K,
    VQ2_BLOCK_M,
    VQ2_BLOCK_N,
    VQ2_CODEBOOK_SIZE,
    VQ2_INDICES_PER_WORD,
    VQ2_VECTOR_LENGTH,
    VQ2A8M1Shape,
    validate_vq2a8_tp1_m1_inputs,
)
from vllm_ascend.quantization.vq2a8_triton import _vq2a8_ascend_launch_options


def _inputs() -> tuple[torch.Tensor, ...]:
    size_n = 64
    size_k = 512
    return (
        torch.zeros((1, size_k), dtype=torch.float8_e4m3fn),
        torch.ones(1, dtype=torch.float32),
        torch.zeros(1, dtype=torch.float32),
        torch.zeros((size_n // 2, size_k // 8), dtype=torch.int32),
        torch.zeros((2, size_n // 32, 16, 2), dtype=torch.float8_e4m3fn),
        torch.zeros(size_k, dtype=torch.uint8),
    )


def test_validate_tp1_m1_packed_kernel_contract() -> None:
    shape = validate_vq2a8_tp1_m1_inputs(*_inputs())

    assert shape == VQ2A8M1Shape(
        size_n=64,
        size_k=512,
        column_tiles=2,
        row_tiles=2,
    )


def test_native_e4m3_transfers_preserve_a5_alignment() -> None:
    fp8_bytes = torch.empty((), dtype=torch.float8_e4m3fn).element_size()
    int32_bytes = torch.empty((), dtype=torch.int32).element_size()

    assert (VQ2_BLOCK_M, VQ2_BLOCK_N) == (32, 32)
    assert VQ2_BLOCK_K * fp8_bytes == 512
    assert VQ2_BLOCK_K // VQ2_INDICES_PER_WORD * int32_bytes == 256
    assert VQ2_CODEBOOK_SIZE * VQ2_VECTOR_LENGTH * fp8_bytes == 32


def test_ascend_kernel_uses_explicit_cv_compile_profile() -> None:
    assert _vq2a8_ascend_launch_options() == {
        "multibuffer": False,
        "enable_auto_bind_sub_block": False,
        "num_warps": 4,
    }


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
    values[4] = torch.zeros((2, 1, 16, 2), dtype=torch.float8_e4m3fn)

    with pytest.raises(ValueError, match="row tile count"):
        validate_vq2a8_tp1_m1_inputs(*values)


def test_validate_tp1_m1_requires_native_e4m3_inputs() -> None:
    values = list(_inputs())
    values[4] = values[4].float()

    with pytest.raises(ValueError, match="codebooks must use torch.float8_e4m3fn"):
        validate_vq2a8_tp1_m1_inputs(*values)

    values = list(_inputs())
    values[0] = values[0].bfloat16()

    with pytest.raises(ValueError, match="activation must use torch.float8_e4m3fn"):
        validate_vq2a8_tp1_m1_inputs(*values)
