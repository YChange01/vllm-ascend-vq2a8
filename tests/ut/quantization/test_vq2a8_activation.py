# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.quantization import vq2a8_activation as activation_module
from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_reference import prepare_repacked_vq2a8_activation_reference


def inputs(width=512):
    rng = torch.Generator().manual_seed(73)
    return [
        torch.randn(1, width, generator=rng).bfloat16(),
        torch.randn(width, generator=rng),
        torch.randn(width, generator=rng),
        torch.where(torch.arange(width) % 3 == 0, -1, 1).to(torch.int8),
    ]


@pytest.mark.parametrize("width", [128, 512, 2048, 4096])
@pytest.mark.parametrize("block", [32, 128])
@pytest.mark.parametrize("case", ["normal", "zero", "impulse", "small"])
def test_preparation_preserves_rowwise_bits_and_fp32_epilogue(width, block, case):
    values = inputs(width)
    if case == "zero":
        values[0].zero_()
    elif case == "impulse":
        values[0].zero_()
        values[0][0, -1] = -1
    elif case == "small":
        values[0] *= 1e-5
    prepare = RowwiseVQ2A8Preparation()
    expected = prepare_repacked_vq2a8_activation_reference(*values, block)
    for _ in range(2):
        actual = prepare(*values, block)
        for got, want in zip(actual, expected):
            assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


def test_constant_is_reused_bounded_and_owned_by_runtime(monkeypatch):
    calls = []
    original = activation_module._sylvester_hadamard

    def matrix(size):
        calls.append(size)
        return original(size)

    monkeypatch.setattr(activation_module, "_sylvester_hadamard", matrix)
    first, second = RowwiseVQ2A8Preparation(), RowwiseVQ2A8Preparation()
    first(*inputs(), 128)
    tensor = first._hadamard
    first(*inputs(), 128)
    assert first._hadamard is tensor
    second(*inputs(), 128)
    first(*inputs(), 32)
    assert calls == [128, 128, 32]
    assert first._hadamard.shape == (32, 32)


@pytest.mark.parametrize("field", range(4))
def test_no_input_validation_decision_is_cached(field):
    values = inputs()
    prepare = RowwiseVQ2A8Preparation()
    prepare(*values, 128)
    values[field].view(-1)[-1] = 0 if field == 3 else float("nan")
    with pytest.raises(ValueError, match="Invalid activation"):
        prepare(*values, 128)


@pytest.mark.parametrize("block", [0, -1, 3, 1024, True])
def test_invalid_block_rejected(block):
    with pytest.raises(ValueError, match="rht_block_size"):
        RowwiseVQ2A8Preparation()(*inputs(), block)


def test_one_host_validity_decision_and_no_batch_geometry_change(monkeypatch):
    values = inputs()
    bool_calls = []
    original = torch.Tensor.__bool__

    def count(value):
        bool_calls.append(value.numel())
        return original(value)

    monkeypatch.setattr(torch.Tensor, "__bool__", count)
    RowwiseVQ2A8Preparation()(*values, 128)
    assert bool_calls == [1]
    values[0] = values[0].repeat(2, 1)
    with pytest.raises(ValueError, match="exactly one"):
        RowwiseVQ2A8Preparation()(*values, 128)
