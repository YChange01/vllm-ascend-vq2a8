// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/ATen.h>
#include <torch/library.h>
#include <algorithm>
#include <cstring>
#include <tuple>
#include "acl/acl_rt.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "activation_launch.h"

namespace vq2a8_ascendc_v4_v2 {
namespace {
constexpr int64_t kMaxPreparationRows = 6 * 32;

void CheckMatrix(const at::Tensor& x) {
  TORCH_CHECK(x.defined() && x.device().type() == c10::DeviceType::PrivateUse1,
              "Fused V4/v2 activation preparation requires NPU tensors");
  TORCH_CHECK(x.scalar_type() == at::kFloat && x.dim() == 2 && x.is_contiguous(),
              "Fused activation input must be contiguous FP32[rows,width]");
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= kMaxPreparationRows && (x.size(1) == 2048 || x.size(1) == 4096),
              "Fused activation requires rows 1..192, width 2048/4096");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 32 == 0, "Fused activation requires aligned input");
}

void CheckLike(const at::Tensor& tensor, const at::Tensor& x, at::ScalarType dtype, const char* name) {
  TORCH_CHECK(tensor.defined() && tensor.device() == x.device() && tensor.scalar_type() == dtype &&
              tensor.sizes() == x.sizes() && tensor.is_contiguous(), name, ": shape/device/dtype/layout mismatch");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 32 == 0, name, ": alignment must be 32 bytes");
}

uint32_t CoreCount(int64_t rows) {
  const char* soc = aclrtGetSocName();
  TORCH_CHECK(soc && std::strncmp(soc, "Ascend950", 9) == 0, "Fused V4/v2 activation requires Ascend950");
  auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  TORCH_CHECK(platform && platform->GetCoreNumAiv() > 0, "Cannot query Ascend vector cores");
  return std::min(static_cast<uint32_t>(rows), platform->GetCoreNumAiv());
}

void Record(const at::Tensor& tensor, c10_npu::NPUStream stream) {
  c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
}
}  // namespace

std::tuple<at::Tensor, at::Tensor> ActivationSign(const at::Tensor& x, const at::Tensor& weightScale,
                                                const at::Tensor& weightBias, const at::Tensor& signs) {
  CheckMatrix(x);
  CheckLike(weightScale, x, at::kFloat, "weight_scale");
  CheckLike(weightBias, x, at::kFloat, "weight_bias");
  CheckLike(signs, x, at::kChar, "signs");
  const c10_npu::OptionalNPUGuard guard(x.device());
  const auto blocks = CoreCount(x.size(0));
  auto output = at::empty_like(x);
  auto valid = at::empty({x.size(0)}, x.options().dtype(at::kInt));
  const auto stream = c10_npu::getCurrentNPUStream();
  for (const auto& input : {x, weightScale, weightBias, signs}) Record(input, stream);
  // Resolve before enqueue: stream() drains the task queue and must NEVER
  // run inside its own launch callback. RunOpApi preserves safe owner release.
  const auto launchStream = stream.stream();
  const uint32_t rows = x.size(0), width = x.size(1);
  at_npu::native::OpCommand::RunOpApi("Vq2a8V4V2ActivationSign",
      [launchStream, blocks, x, weightScale, weightBias, signs, output, valid, rows, width]() -> int {
    LaunchActivationSign(launchStream, blocks, x.data_ptr(), weightScale.data_ptr(), weightBias.data_ptr(),
                         signs.data_ptr(), output.data_ptr(), valid.data_ptr(), rows, width);
    return 0;
  }, false);
  return {output, valid};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> ActivationQuantize(const at::Tensor& rotated,
                                                               const at::Tensor& weightScale,
                                                               const at::Tensor& rowBias) {
  CheckMatrix(rotated);
  CheckLike(weightScale, rotated, at::kFloat, "weight_scale");
  TORCH_CHECK(rowBias.defined() && rowBias.device() == rotated.device() && rowBias.scalar_type() == at::kFloat &&
              rowBias.dim() == 1 && rowBias.size(0) == rotated.size(0) && rowBias.is_contiguous(),
              "row_bias must be contiguous FP32[rows] on the activation device");
  const c10_npu::OptionalNPUGuard guard(rotated.device());
  const auto blocks = CoreCount(rotated.size(0));
  auto quantized = at::empty_like(rotated, rotated.options().dtype(at::kFloat8_e4m3fn));
  auto scale = at::empty({rotated.size(0)}, rotated.options());
  auto valid = at::empty({rotated.size(0)}, rotated.options().dtype(at::kInt));
  const auto stream = c10_npu::getCurrentNPUStream();
  for (const auto& input : {rotated, weightScale, rowBias}) Record(input, stream);
  const auto launchStream = stream.stream();
  const uint32_t rows = rotated.size(0), width = rotated.size(1);
  at_npu::native::OpCommand::RunOpApi("Vq2a8V4V2ActivationQuantize",
      [launchStream, blocks, rotated, weightScale, rowBias, quantized, scale, valid, rows, width]() -> int {
    LaunchActivationQuantize(launchStream, blocks, rotated.data_ptr(), weightScale.data_ptr(), rowBias.data_ptr(),
                             quantized.data_ptr(), scale.data_ptr(), valid.data_ptr(), rows, width);
    return 0;
  }, false);
  return {quantized, scale, valid};
}
}  // namespace vq2a8_ascendc_v4_v2

TORCH_LIBRARY_FRAGMENT(vq2a8_ascendc_v4_v2, m) {
  m.def("activation_preparation_version() -> int", []() -> int64_t { return 1; });
  m.def("activation_sign(Tensor x, Tensor weight_scale, Tensor weight_bias, Tensor signs) -> (Tensor, Tensor)");
  m.def("activation_quantize(Tensor rotated, Tensor weight_scale, Tensor row_bias) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(vq2a8_ascendc_v4_v2, PrivateUse1, m) {
  m.impl("activation_sign", &vq2a8_ascendc_v4_v2::ActivationSign);
  m.impl("activation_quantize", &vq2a8_ascendc_v4_v2::ActivationQuantize);
}
