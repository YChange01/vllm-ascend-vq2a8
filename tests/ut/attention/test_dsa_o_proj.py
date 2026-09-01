# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.attention.dsa_o_proj import (
    apply_unquantized_grouped_o_proj,
    can_all_gather_dsa_cp_o_proj,
    reshape_grouped_o_proj_weight,
)


def test_dsa_cp_full_weight_gather_requires_two_scaled_projections() -> None:
    scaled = SimpleNamespace(weight_scale=torch.ones(1))
    unscaled = SimpleNamespace()

    assert can_all_gather_dsa_cp_o_proj(scaled, scaled)
    assert not can_all_gather_dsa_cp_o_proj(scaled, unscaled)
    assert not can_all_gather_dsa_cp_o_proj(unscaled, scaled)
    assert not can_all_gather_dsa_cp_o_proj(unscaled, unscaled)


def test_grouped_o_proj_restores_flattened_column_parallel_weight() -> None:
    num_tokens = 5
    num_groups = 2
    o_lora_rank = 3
    input_size = 4
    activations = torch.randn(num_tokens, num_groups, input_size)
    flat_weight = torch.randn(num_groups * o_lora_rank, input_size)

    def batch_matmul(
        actual_activations: torch.Tensor,
        actual_weight: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        assert actual_activations.ndim == 3
        assert actual_weight.ndim == 3
        assert actual_weight.shape == (num_groups, input_size, o_lora_rank)
        assert kwargs["perm_x1"] == (1, 0, 2)
        assert kwargs["perm_x2"] == (0, 1, 2)
        assert kwargs["perm_y"] == (1, 0, 2)
        x1 = actual_activations.permute(kwargs["perm_x1"])
        x2 = actual_weight.permute(kwargs["perm_x2"])
        assert x1.shape[-1] == x2.shape[-2]
        return torch.bmm(x1, x2).permute(kwargs["perm_y"])

    actual = apply_unquantized_grouped_o_proj(
        activations,
        flat_weight,
        num_groups,
        o_lora_rank,
        batch_matmul,
    )
    expected = torch.stack(
        [
            F.linear(
                activations[:, group],
                flat_weight[group * o_lora_rank : (group + 1) * o_lora_rank],
            )
            for group in range(num_groups)
        ],
        dim=1,
    )

    assert actual.shape == (num_tokens, num_groups, o_lora_rank)
    torch.testing.assert_close(actual, expected)


def test_grouped_o_proj_rejects_incompatible_flat_weight() -> None:
    with pytest.raises(ValueError, match="expected \\(6, input_size\\)"):
        reshape_grouped_o_proj_weight(torch.randn(5, 4), 2, 3)


def test_grouped_o_proj_rejects_different_reduction_dimensions() -> None:
    with pytest.raises(ValueError, match="different reduction dimensions"):
        apply_unquantized_grouped_o_proj(
            torch.randn(5, 2, 7),
            torch.randn(6, 4),
            2,
            3,
            torch.bmm,
        )


@pytest.mark.parametrize(
    ("num_groups", "flat_rows"),
    [(2, 2048), (8, 8192)],
)
def test_vq2a8_model_o_proj_shapes(
    num_groups: int,
    flat_rows: int,
) -> None:
    weight = torch.empty(flat_rows, 4096, device="meta")

    actual = reshape_grouped_o_proj_weight(weight, num_groups, 1024)

    assert actual.shape == (num_groups, 1024, 4096)


def test_vq2a8_tp4_o_proj_npu_k_axis_layout() -> None:
    num_tokens = 12
    num_groups = 2
    input_size = 4096
    o_lora_rank = 1024
    activations = torch.empty(num_tokens, num_groups, input_size, device="meta")
    flat_weight = torch.empty(num_groups * o_lora_rank, input_size, device="meta")

    def check_npu_layout(
        actual_activations: torch.Tensor,
        actual_weight: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        x1_shape = tuple(actual_activations.permute(kwargs["perm_x1"]).shape)
        x2_shape = tuple(actual_weight.permute(kwargs["perm_x2"]).shape)
        assert x1_shape == (num_groups, num_tokens, input_size)
        assert x2_shape == (num_groups, input_size, o_lora_rank)
        assert x1_shape[-1] == x2_shape[-2]
        return torch.empty(num_tokens, num_groups, o_lora_rank, device="meta")

    actual = apply_unquantized_grouped_o_proj(
        activations,
        flat_weight,
        num_groups,
        o_lora_rank,
        check_npu_layout,
    )

    assert actual.shape == (num_tokens, num_groups, o_lora_rank)
