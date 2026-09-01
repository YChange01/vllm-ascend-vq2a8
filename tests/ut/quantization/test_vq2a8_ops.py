# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.quantization import vq2a8_ops
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


@pytest.mark.parametrize(
    "activation",
    ["silu", "swiglu", MoEActivation.SILU],
)
def test_reference_two_stage_moe_matches_decoded_dense_weights(
    activation: str | MoEActivation,
) -> None:
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
        activation=activation,
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


def test_reference_gate_up_rejects_unsupported_activation() -> None:
    gate_payload = _payload(output_size=4, input_size=8, row_group_size=2)
    x = torch.arange(8, dtype=torch.float32).reshape(1, 8) / 17
    expert_ids = torch.zeros(1, dtype=torch.int32)

    with pytest.raises(NotImplementedError, match="only supports SwiGLU"):
        reference_vq2a8_gate_up(
            x,
            expert_ids,
            *gate_payload,
            rht_block_size=4,
            row_group_size=2,
            activation=MoEActivation.GELU,
        )


def test_reference_gate_up_casts_fp8_codebooks_before_lookup() -> None:
    gate_payload = list(_payload(output_size=4, input_size=8, row_group_size=2))
    gate_payload[1] = gate_payload[1].to(torch.float8_e4m3fn)
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 17
    expert_ids = torch.zeros(2, dtype=torch.int32)

    actual = reference_vq2a8_gate_up(
        x,
        expert_ids,
        *gate_payload,
        rht_block_size=4,
        row_group_size=2,
    )
    gate_payload[1] = gate_payload[1].to(torch.bfloat16)
    expected = reference_vq2a8_gate_up(
        x,
        expert_ids,
        *gate_payload,
        rht_block_size=4,
        row_group_size=2,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gate_up_dispatch_forwards_the_native_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(output_size=4, input_size=8, row_group_size=2)
    x = torch.zeros((2, 8), dtype=torch.bfloat16)
    expert_ids = torch.zeros(2, dtype=torch.int32)
    sentinel = torch.ones((2, 2), dtype=torch.bfloat16)
    calls: list[tuple[object, ...]] = []

    def fake_op(*args: object) -> torch.Tensor:
        calls.append(args)
        return sentinel

    monkeypatch.setattr(
        vq2a8_ops,
        "_custom_op",
        lambda name: fake_op if name == "vq2a8_gate_up" else None,
    )
    actual = vq2a8_ops.vq2a8_gate_up(
        x,
        expert_ids,
        *payload,
        rht_block_size=4,
        row_group_size=2,
        activation="silu",
        allow_reference_fallback=False,
    )

    assert actual is sentinel
    assert len(calls) == 1
    assert calls[0][0] is x
    assert calls[0][1] is expert_ids
    assert all(
        actual_arg is expected_arg
        for actual_arg, expected_arg in zip(calls[0][2:8], payload)
    )
    assert calls[0][8:] == (4, 2, "silu")


def test_down_reduce_dispatch_forwards_the_native_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(output_size=8, input_size=2, row_group_size=2)
    x = torch.zeros((3, 2), dtype=torch.bfloat16)
    expert_ids = torch.zeros(3, dtype=torch.int32)
    token_ids = torch.arange(3, dtype=torch.int64)
    routing_weights = torch.ones(3, dtype=torch.float32)
    sentinel = torch.ones((3, 8), dtype=torch.bfloat16)
    calls: list[tuple[object, ...]] = []

    def fake_op(*args: object) -> torch.Tensor:
        calls.append(args)
        return sentinel

    monkeypatch.setattr(
        vq2a8_ops,
        "_custom_op",
        lambda name: fake_op if name == "vq2a8_down_reduce" else None,
    )
    actual = vq2a8_ops.vq2a8_down_reduce(
        x,
        expert_ids,
        token_ids,
        routing_weights,
        *payload,
        rht_block_size=2,
        row_group_size=2,
        num_tokens=3,
        allow_reference_fallback=False,
    )

    assert actual is sentinel
    assert len(calls) == 1
    assert calls[0][0] is x
    assert calls[0][1] is expert_ids
    assert calls[0][2] is token_ids
    assert calls[0][3] is routing_weights
    assert all(
        actual_arg is expected_arg
        for actual_arg, expected_arg in zip(calls[0][4:10], payload)
    )
    assert calls[0][10:] == (2, 2, 3)
