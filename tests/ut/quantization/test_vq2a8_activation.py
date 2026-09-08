# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS

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


def requests(count=6, width=512, true_width=512):
    result = []
    for i in range(count):
        x, scale, bias, sign = inputs(width)
        hidden = torch.cat([x[:, :true_width] * (i + j + 1) / 8 for j in range((1, 2, 15, 16, 17, 32)[i])])
        result.append(
            (
                hidden,
                {"weight_scale": scale * (i + 1), "weight_bias": bias, "rht_sign": sign},
                NS(columns=width, rht_true_columns=true_width, rht_block_size=128),
            )
        )
    return result


@pytest.mark.parametrize("width,true_width", [(512, 512), (512, 480), (2048, 2048), (4096, 4000)])
@pytest.mark.parametrize("case", ["normal", "zero", "impulse", "small"])
def test_grouped_preparation_exact_for_distinct_experts_and_mixed_rows(width, true_width, case):
    batch = requests(width=width, true_width=true_width)
    for hidden, _, _ in batch:
        if case in ("zero", "impulse"):
            hidden.zero_()
        if case == "impulse":
            hidden[:, -1] = -2
        if case == "small":
            hidden *= 1e-7
    prepare = RowwiseVQ2A8Preparation()
    expected = [prepare.rows(*request) for request in batch]
    for order in (batch, batch, list(reversed(batch))):
        got = prepare.many(order)
        want = list(reversed(expected)) if order is not batch else expected
        for actual, reference in zip(got, want):
            for a, b in zip(actual, reference):
                assert a.is_contiguous() and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("field", ["hidden", "weight_scale", "weight_bias", "rht_sign"])
def test_grouped_input_value_validation_not_cached(field):
    batch = requests()
    prepare = RowwiseVQ2A8Preparation()
    prepare.many(batch)
    tensor = batch[-1][0] if field == "hidden" else batch[-1][1][field]
    tensor.view(-1)[-1] = 0 if field == "rht_sign" else float("nan")
    with pytest.raises(ValueError, match="Invalid activation"):
        prepare.many(batch)


def test_grouped_validation_one_decision_and_reference_matmul_shapes(monkeypatch):
    batch = requests(count=2)
    decisions, shapes = [], []
    old_bool, old_matmul = torch.Tensor.__bool__, torch.Tensor.__matmul__

    def boolean(t):
        decisions.append(t.numel())
        return old_bool(t)

    def matmul(a, b):
        shapes.append((tuple(a.shape), tuple(b.shape)))
        return old_matmul(a, b)

    monkeypatch.setattr(torch.Tensor, "__bool__", boolean)
    monkeypatch.setattr(torch.Tensor, "__matmul__", matmul)
    RowwiseVQ2A8Preparation().many(batch)
    assert decisions == [1]
    assert shapes == [((1, 4, 128), (128, 128))] * 3 + [((1, 512), (512,))] * 3


@pytest.mark.parametrize("bad", ["empty", "too_many", "m0", "m33", "width", "sign_dtype", "block"])
def test_grouped_metadata_fails_before_compute(bad):
    batch = requests(count=2)
    if bad == "empty":
        batch = []
    elif bad == "too_many":
        batch = batch * 4
    elif bad in ("m0", "m33", "width"):
        x = torch.zeros(0 if bad == "m0" else 33 if bad == "m33" else 1, 500 if bad == "width" else 512)
        batch[-1] = (x, *batch[-1][1:])
    elif bad == "sign_dtype":
        batch[-1][1]["rht_sign"] = batch[-1][1]["rht_sign"].float()
    elif bad == "block":
        batch[-1][2].rht_block_size = 3
    with pytest.raises(ValueError):
        RowwiseVQ2A8Preparation().many(batch)
