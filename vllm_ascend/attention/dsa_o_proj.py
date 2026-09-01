# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections.abc import Callable

import torch


def has_o_proj_weight_scale(layer: torch.nn.Module) -> bool:
    """Return whether an o-projection layer carries quantized weight scales."""
    return hasattr(layer, "weight_scale")


def can_all_gather_dsa_cp_o_proj(
    wo_a: torch.nn.Module,
    wo_b: torch.nn.Module,
) -> bool:
    """Full-weight DSA-CP o-proj requires the A5 quantized weight layout."""
    return has_o_proj_weight_scale(wo_a) and has_o_proj_weight_scale(wo_b)


def reshape_grouped_o_proj_weight(
    weight: torch.Tensor,
    num_groups: int,
    o_lora_rank: int,
) -> torch.Tensor:
    """Restore the group axis flattened into ColumnParallelLinear weights."""
    expected_shape = (num_groups, o_lora_rank)
    if weight.ndim == 3:
        if weight.shape[:2] != expected_shape:
            raise ValueError(
                "Grouped o-proj weight has incompatible leading dimensions: "
                f"got {tuple(weight.shape)}, expected ({num_groups}, "
                f"{o_lora_rank}, input_size)"
            )
        return weight

    expected_rows = num_groups * o_lora_rank
    if weight.ndim != 2 or weight.shape[0] != expected_rows:
        raise ValueError(
            "Flattened o-proj weight has incompatible dimensions: "
            f"got {tuple(weight.shape)}, expected ({expected_rows}, input_size)"
        )
    return weight.view(num_groups, o_lora_rank, weight.shape[1])


def apply_unquantized_grouped_o_proj(
    o_proj_input: torch.Tensor,
    weight: torch.Tensor,
    num_groups: int,
    o_lora_rank: int,
    batch_matmul: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Apply the BF16 grouped o-projection with an injected NPU operator."""
    if o_proj_input.ndim != 3 or o_proj_input.shape[1] != num_groups:
        raise ValueError(
            "Grouped o-proj input has incompatible dimensions: "
            f"got {tuple(o_proj_input.shape)}, expected "
            f"(num_tokens, {num_groups}, input_size)"
        )
    grouped_weight = reshape_grouped_o_proj_weight(
        weight,
        num_groups,
        o_lora_rank,
    )
    return batch_matmul(
        o_proj_input,
        grouped_weight,
        bias=None,
        scale=None,
        perm_x1=(1, 0, 2),
        perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2),
        batch_split_factor=1,
    )
