# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch

from vllm_ascend.quantization.vq2a8_format import decode_repacked_vq2_weight
from vllm_ascend.quantization.vq2a8_ops import (
    reference_vq2a8_down_reduce,
    reference_vq2a8_gate_up,
)
from vllm_ascend.quantization.vq2a8_repack import pack_repacked_indices


def _payload(
    output_size: int,
    input_size: int,
    row_group_size: int,
) -> tuple[torch.Tensor, ...]:
    output_pairs = output_size // 2
    row_tiles = output_size // row_group_size
    indices = torch.arange(output_pairs * input_size).reshape(
        output_pairs, input_size
    )
    packed_indices = pack_repacked_indices(indices.remainder(16)).unsqueeze(0)
    codebooks = (
        torch.arange(row_tiles * 16 * 2, dtype=torch.float32)
        .reshape(1, 1, row_tiles, 16, 2)
        .div(23)
    )
    codebook_tile_ids = torch.zeros((1, input_size), dtype=torch.uint8)
    weight_scale = torch.linspace(0.5, 1.0, input_size).unsqueeze(0)
    weight_bias = torch.linspace(-0.1, 0.2, input_size).unsqueeze(0)
    rht_sign = torch.tensor(
        [[1 if index % 3 else -1 for index in range(input_size)]],
        dtype=torch.int8,
    )
    return (
        packed_indices,
        codebooks,
        codebook_tile_ids,
        weight_scale,
        weight_bias,
        rht_sign,
    )


def test_reference_two_stage_moe_matches_decoded_dense_weights() -> None:
    gate_payload = _payload(output_size=4, input_size=8, row_group_size=2)
    down_payload = _payload(output_size=8, input_size=2, row_group_size=2)
    x = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 17
    expert_ids = torch.zeros(3, dtype=torch.int32)
    token_ids = torch.arange(3)
    routing_weights = torch.tensor([0.2, 0.5, 0.8])

    gate_up = reference_vq2a8_gate_up(
        x,
        expert_ids,
        *gate_payload,
        rht_block_size=4,
        row_group_size=2,
    )
    actual = reference_vq2a8_down_reduce(
        gate_up,
        expert_ids,
        token_ids,
        routing_weights,
        *down_payload,
        rht_block_size=2,
        row_group_size=2,
        num_tokens=3,
    )

    gate_weight = decode_repacked_vq2_weight(
        *[tensor[0] for tensor in gate_payload],
        rht_block_size=4,
        row_group_size=2,
    ).float()
    down_weight = decode_repacked_vq2_weight(
        *[tensor[0] for tensor in down_payload],
        rht_block_size=2,
        row_group_size=2,
    ).float()
    gate, up = (x @ gate_weight.T).chunk(2, dim=-1)
    expected = (torch.nn.functional.silu(gate) * up) @ down_weight.T
    expected *= routing_weights.unsqueeze(-1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
