# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU source/geometry contracts, not evidence of native NPU execution.

Numerical, graph and lifetime execution remain gated by the bounded packed
activation validator on Ascend hardware with the rebuilt native library.
"""

from pathlib import Path

import pytest
import torch

NATIVE_ROOT = Path(__file__).resolve().parents[3] / "csrc/vq2a8_ascendc_v4_v2"


def source(name):
    return (NATIVE_ROOT / name).read_text(encoding="utf-8")


def section(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


def test_strided_sign_is_an_independent_feature_without_changing_old_abi():
    binding = source("activation_binding.cpp")
    assert 'm.def("activation_preparation_version() -> int", []() -> int64_t { return 1; });' in binding
    assert 'm.def("activation_sign_strided_version() -> int", []() -> int64_t { return 1; });' in binding
    assert 'm.impl("activation_sign", &vq2a8_ascendc_v4_v2::ActivationSign);' in binding
    assert 'm.impl("activation_sign_strided", &vq2a8_ascendc_v4_v2::ActivationSignStrided);' in binding
    assert 'm.def("activation_sign_strided(Tensor x, Tensor weight_scale, Tensor weight_bias, Tensor signs)' in binding


def test_strided_sign_rejects_unbounded_geometry_and_unsupported_layouts():
    binding = source("activation_binding.cpp")
    check = section(binding, "void CheckStridedMatrix", "uint32_t CoreCount")
    assert "constexpr int64_t kMaxStridedRows = 6;" in binding
    assert "c10::DeviceType::PrivateUse1" in check
    assert "x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16" in check
    assert "x.dim() == 2" in check
    assert "x.size(0) >= 1 && x.size(0) <= kMaxStridedRows" in check
    assert "x.size(1) == 2048 || x.size(1) == 4096" in check
    assert "x.stride(1) == 1 && (x.stride(0) == 0 || x.stride(0) >= x.size(1))" in check
    assert "reinterpret_cast<uintptr_t>(x.data_ptr()) % kActivationAlignment == 0" in check
    assert "rowStride % (kActivationAlignment / elementBytes) == 0" in check


def test_storage_span_check_uses_division_before_unsigned_kernel_multiply():
    check = section(source("activation_binding.cpp"), "void CheckStridedMatrix", "uint32_t CoreCount")
    assert "x.storage().nbytes()" in check
    assert "x.storage_offset() >= 0" in check
    assert "storageElements - static_cast<uint64_t>(x.storage_offset())" in check
    assert "width <= available &&" in check
    assert "rowStride <= (available - width) / static_cast<uint64_t>(x.size(0) - 1)" in check


def test_strided_sign_keeps_metadata_contract_and_returns_dense_fp32():
    body = section(source("activation_binding.cpp"), "ActivationSignStrided(", "ActivationQuantize(")
    for expression in (
        'CheckLike(weightScale, x, at::kFloat, "weight_scale")',
        'CheckLike(weightBias, x, at::kFloat, "weight_bias")',
        'CheckLike(signs, x, at::kChar, "signs")',
        "at::empty(x.sizes(), x.options().dtype(at::kFloat))",
        "at::empty({x.size(0)}, x.options().dtype(at::kInt))",
    ):
        assert expression in body
    assert ".contiguous(" not in body
    assert ".to(" not in body
    assert "empty_like(x)" not in body


def test_strided_sign_resolves_stream_before_enqueue_and_keeps_owners():
    body = section(source("activation_binding.cpp"), "ActivationSignStrided(", "ActivationQuantize(")
    resolve = body.index("const auto launchStream = stream.stream();")
    enqueue = body.index('OpCommand::RunOpApi("Vq2a8V4V2ActivationSignStrided"')
    assert resolve < enqueue
    assert "for (const auto& input : {x, weightScale, weightBias, signs}) Record(input, stream);" in body
    assert "[launchStream, blocks, x, weightScale, weightBias, signs, output, valid, rows, width," in body
    assert "rowStride, inputIsBf16]() -> int" in body
    assert "stream.stream()" not in body[enqueue:]
    assert "RunOpApiV2" not in source("activation_binding.cpp")


def test_kernel_uses_independent_wide_input_stride_and_dense_metadata_offsets():
    kernel = source("activation_kernel.cpp")
    sign = section(kernel, "class ActivationSignKernel", "class ActivationQuantizeKernel")
    assert "uint64_t rowStride_;" in sign
    assert "const uint64_t inputBase = uint64_t(row) * rowStride_;" in sign
    assert "const uint64_t base = uint64_t(row) * width_;" in sign
    assert "DataCopy(scale, scale_[base], width_);" in sign
    assert "DataCopy(bias, bias_[base], width_);" in sign
    assert "DataCopy(sign, signs_[base], width_);" in sign
    assert "DataCopy(output_[base], x, width_);" in sign


def test_bf16_widens_after_load_fence_before_existing_checks_and_multiply():
    kernel = source("activation_kernel.cpp")
    sign = section(kernel, "class ActivationSignKernel", "class ActivationQuantizeKernel")
    load_fence = sign.index("ActivationFence<HardEvent::MTE2_V>();")
    cast = sign.index("Cast(x, xInputUb_.Get<InputT>(), RoundMode::CAST_NONE, width_);")
    vector_fence = sign.index("PipeBarrier<PIPE_V>();", cast)
    finite = sign.index("FiniteMask(x, scratch, mask, width_);")
    multiply = sign.index("Mul(x, x, signFloat, width_);")
    output_fence = sign.index("ActivationFence<HardEvent::V_MTE3>();", multiply)
    assert load_fence < cast < vector_fence < finite < multiply < output_fence
    for expression in (
        "FiniteMask(scale, scratch, other, width_);",
        "FiniteMask(bias, scratch, other, width_);",
        "Compares(other, scratch, 1.0f, CMPMODE::EQ, width_);",
        "const bool valid = AllMask(maskUb_.Get<uint32_t>(), width_);",
        "ActivationFence<HardEvent::MTE3_S>();",
    ):
        assert expression in sign


def test_old_dense_fp32_launch_still_uses_width_stride_and_separate_quantizer():
    kernel = source("activation_kernel.cpp")
    old = section(kernel, "void vq2a8_v4_v2_activation_sign(", "void vq2a8_v4_v2_activation_sign_strided_fp32(")
    assert "ActivationSignKernel<float> op;" in old
    assert "op.Init(x, scale, bias, signs, output, valid, rows, width, width);" in old
    launch = section(kernel, "void LaunchActivationSignStrided(", "void LaunchActivationQuantize(")
    assert "if (inputIsBf16)" in launch
    assert "vq2a8_v4_v2_activation_sign_strided_bf16<<<" in launch
    assert "vq2a8_v4_v2_activation_sign_strided_fp32<<<" in launch
    assert "quantize" not in launch
    assert "uint64_t rowStride, bool inputIsBf16" in source("activation_launch.h")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [2048, 4096])
@pytest.mark.parametrize("rows", [1, 6])
@pytest.mark.parametrize("layout", ["dense", "expanded", "padded", "offset"])
def test_supported_view_geometry_has_exact_dense_reference(dtype, width, rows, layout):
    """Address arithmetic and CPU oracle only; not a native arithmetic test."""
    step = 32 // torch.empty((), dtype=dtype).element_size()
    if layout == "expanded":
        x = torch.arange(width, dtype=torch.float32).to(dtype).view(1, width).expand(rows, width)
    elif layout == "padded":
        x = torch.arange(rows * (width + step), dtype=torch.float32).to(dtype).view(rows, width + step)[:, :width]
    elif layout == "offset":
        x = torch.arange(step + rows * width, dtype=torch.float32).to(dtype)[step:].view(rows, width)
    else:
        x = torch.arange(rows * width, dtype=torch.float32).to(dtype).view(rows, width)
    row_stride = x.stride(0)
    element_bytes = x.element_size()
    available = x.untyped_storage().nbytes() // element_bytes - x.storage_offset()
    assert x.stride(1) == 1
    assert row_stride == 0 or row_stride >= width
    assert x.data_ptr() % 32 == 0
    assert rows == 1 or row_stride % step == 0
    assert width <= available
    assert rows == 1 or row_stride <= (available - width) // (rows - 1)
    signs = torch.where(torch.arange(rows * width).view(rows, width) % 3 == 0, -1, 1).to(torch.int8)
    expected = x.contiguous().float() * signs.float()
    for row in range(rows):
        assert x[row].data_ptr() == x.data_ptr() + row * row_stride * element_bytes
        actual = x[row].float() * signs[row].float()
        assert torch.equal(actual.view(torch.uint8), expected[row].view(torch.uint8))
