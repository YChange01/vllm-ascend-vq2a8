// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/ATen.h>
#include <torch/library.h>
#include <algorithm>
#include <cstring>
#include "acl/acl_rt.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "bias_dot_probe_launch.h"

namespace vq2a8_ascendc_v4_v2 {
namespace {
void CheckBiasDotInput(const at::Tensor& value) {
  TORCH_CHECK(value.defined() && value.device().type() == c10::DeviceType::PrivateUse1 &&
                  value.scalar_type() == at::kFloat && value.dim() == 2 && value.is_contiguous(),
              "Bias dot probe requires contiguous NPU FP32[R,K]");
  TORCH_CHECK(value.size(0) >= 1 && value.size(0) <= 6 &&
                  (value.size(1) == 2048 || value.size(1) == 4096),
              "Bias dot probe requires R1..6 and K2048/4096");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(value.data_ptr()) % 32 == 0,
              "Bias dot probe requires 32-byte aligned input");
  const auto storageElements = static_cast<uint64_t>(value.storage().nbytes()) / sizeof(float);
  TORCH_CHECK(value.storage_offset() >= 0 &&
                  static_cast<uint64_t>(value.storage_offset()) <= storageElements &&
                  static_cast<uint64_t>(value.numel()) <=
                      storageElements - static_cast<uint64_t>(value.storage_offset()),
              "Bias dot probe input span exceeds storage");
}
}  // namespace

at::Tensor BiasDotRowsProbe(const at::Tensor& rotated, const at::Tensor& weightBias) {
  CheckBiasDotInput(rotated);
  CheckBiasDotInput(weightBias);
  TORCH_CHECK(rotated.device() == weightBias.device() && rotated.sizes() == weightBias.sizes(),
              "Bias dot probe inputs must match device and shape");
  const c10_npu::OptionalNPUGuard guard(rotated.device());
  const char* soc = aclrtGetSocName();
  TORCH_CHECK(soc && std::strncmp(soc, "Ascend950", 9) == 0, "Bias dot probe requires Ascend950");
  auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  TORCH_CHECK(platform && platform->GetCoreNumAiv() > 0, "Cannot query bias dot probe vector cores");
  const uint32_t rows = rotated.size(0), width = rotated.size(1);
  const uint32_t blocks = std::min(rows, platform->GetCoreNumAiv());
  auto output = at::empty({rows}, rotated.options());
  const auto stream = c10_npu::getCurrentNPUStream();
  for (const auto& tensor : {rotated, weightBias, output}) {
    c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
  }
  const auto launchStream = stream.stream();
  at_npu::native::OpCommand::RunOpApi("Vq2a8V4V2BiasDotRowsProbe",
      [launchStream, blocks, rotated, weightBias, output, rows, width]() -> int {
    LaunchBiasDotRowsProbe(launchStream, blocks, rotated.data_ptr(), weightBias.data_ptr(),
                           output.data_ptr(), rows, width);
    return 0;
  }, false);
  return output;
}
}  // namespace vq2a8_ascendc_v4_v2

TORCH_LIBRARY_FRAGMENT(vq2a8_ascendc_v4_v2, m) {
  m.def("bias_dot_rows_probe_version() -> int", []() -> int64_t { return 1; });
  m.def("bias_dot_rows_probe(Tensor rotated, Tensor weight_bias) -> Tensor");
}
TORCH_LIBRARY_IMPL(vq2a8_ascendc_v4_v2, PrivateUse1, m) {
  m.impl("bias_dot_rows_probe", &vq2a8_ascendc_v4_v2::BiasDotRowsProbe);
}
