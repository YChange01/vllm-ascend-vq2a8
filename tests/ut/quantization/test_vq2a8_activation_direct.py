# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct destination and strided sign CPU contracts, not Ascend acceptance."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from tests.ut.quantization.test_vq2a8_activation_packed import TorchSignOps, assert_bytes, fixture, reference
from vllm_ascend.quantization.vq2a8_activation_packed import PackedRowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_v4_v2 import require_v4_v2_features


class StridedSignOracle(TorchSignOps):
    def __init__(self):
        super().__init__()
        self.input_views = []

    def activation_sign_strided_version(self):
        return 1

    def activation_sign_strided(self, x, scale, bias, signs):
        self.input_views.append((x.data_ptr(), x.dtype, x.stride()))
        return self.activation_sign(x.float(), scale, bias, signs)


def candidate(native, direct):
    return PackedRowwiseVQ2A8Preparation(fuse_sign=True, strided_sign=True, direct_output=direct, native_ops=native)


@pytest.mark.parametrize("width", [2048, 4096])
@pytest.mark.parametrize("groups", range(1, 7))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["contiguous", "expanded", "padded"])
@pytest.mark.parametrize("direct", [False, True])
def test_strided_input_and_direct_outputs_match_rowwise(width, groups, dtype, layout, direct):
    x, scale, bias, signs, spec = fixture(width, groups, dtype=dtype)
    if layout == "expanded":
        x = x[:1].expand(groups, -1)
    elif layout == "padded":
        backing = torch.empty((groups, width + 32), dtype=dtype)
        backing[:, :width].copy_(x)
        x = backing[:, :width]
    before = tuple(value.clone() for value in (x, scale, bias, signs))
    native = StridedSignOracle()
    result = candidate(native, direct).packed(x, scale, bias, signs, spec)
    assert native.input_views == [(x.data_ptr(), dtype, x.stride())]
    for got, expected in zip(result, reference(x, scale, bias, signs, spec)):
        assert_bytes(got, expected)
        assert got.is_contiguous()
    for value, original in zip((x, scale, bias, signs), before):
        assert torch.equal(value, original)


def test_direct_matmul_preserves_input_ranks_and_uses_distinct_output_views(monkeypatch):
    original = torch.matmul
    calls = []

    def record(left, right, *, out=None):
        calls.append((left.shape, right.shape, out.shape, out.data_ptr()))
        return original(left, right, out=out)

    def forbidden(*args, **kwargs):
        raise AssertionError("Direct output must not perform Python-level copy_ or scalar reads")

    values = fixture(2048, 6)
    monkeypatch.setattr(torch, "matmul", record)
    monkeypatch.setattr(torch.Tensor, "copy_", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    flags = []
    candidate(StridedSignOracle(), True).packed(*values, validity=flags.append)
    assert len(calls) == 12 and len(flags) == 1
    assert len({call[3] for call in calls}) == 12
    for rht, bias in zip(calls[::2], calls[1::2]):
        assert rht[:3] == (torch.Size([1, 16, 128]), torch.Size([128, 128]), torch.Size([1, 16, 128]))
        assert bias[:3] == (torch.Size([1, 2048]), torch.Size([2048]), torch.Size([1]))


@pytest.mark.parametrize("direct", [False, True])
def test_direct_result_owns_storage_across_calls(direct):
    values = fixture(2048, 2)
    references = [weakref.ref(value) for value in values[:4]]
    preparation = candidate(StridedSignOracle(), direct)
    result = preparation.packed(*values)
    snapshot = tuple(value.view(torch.uint8).clone() for value in result)
    del values
    gc.collect()
    assert all(ref() is None for ref in references)
    second = preparation.packed(*fixture(2048, 2))
    for first, later, expected in zip(result, second, snapshot):
        assert first.data_ptr() != later.data_ptr()
        assert torch.equal(first.view(torch.uint8), expected)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_single_row_unused_stride_does_not_need_alignment(direct, dtype):
    x, scale, bias, signs, spec = fixture(2048, 1, dtype=dtype)
    x = x.as_strided((1, 2048), (2049, 1))
    native = StridedSignOracle()
    actual = candidate(native, direct).packed(x, scale, bias, signs, spec)
    assert native.input_views == [(x.data_ptr(), dtype, (2049, 1))]
    for got, expected in zip(actual, reference(x, scale, bias, signs, spec)):
        assert_bytes(got, expected)


@pytest.mark.parametrize("kind", ["dtype", "columns", "overlap", "alignment"])
def test_reject_unsupported_strided_input_without_fallback(kind):
    x, scale, bias, signs, spec = fixture(2048, 2)
    if kind == "dtype":
        x = x.half()
    elif kind == "columns":
        x = torch.empty(2, 4096, dtype=x.dtype)[:, ::2]
    elif kind == "overlap":
        x = x.as_strided((2, 2048), (32, 1))
    else:
        x = torch.empty(2 * 2048 + 1, dtype=x.dtype)[1:].view(2, 2048)
    native = StridedSignOracle()
    with pytest.raises(ValueError, match="Strided sign"):
        candidate(native, True).packed(x, scale, bias, signs, spec)
    assert native.input_views == []


@pytest.mark.parametrize("mode", ["sign_fused_strided", "sign_fused_direct"])
def test_feature_check_requires_new_abi_without_changing_old_sign_fused(mode):
    old = SimpleNamespace(activation_preparation_version=lambda: 1)
    require_v4_v2_features(preparation="sign_fused", native_ops=old)
    with pytest.raises(RuntimeError, match="activation_sign_strided_version"):
        require_v4_v2_features(preparation=mode, native_ops=old)
    with pytest.raises(RuntimeError, match="rebuilt library"):
        candidate(TorchSignOps(), False)
    for invalid in (True, 0, 2, "1"):
        native = StridedSignOracle()
        native.activation_sign_strided_version = lambda value=invalid: value
        with pytest.raises(RuntimeError, match="strided sign ABI"):
            candidate(native, False)


@pytest.mark.parametrize(
    "options", [{"strided_sign": True}, {"direct_output": True}, {"fuse_sign": True, "direct_output": True}]
)
def test_inconsistent_direct_switches_rejected(options):
    with pytest.raises(ValueError, match="requires"):
        PackedRowwiseVQ2A8Preparation(**options)
