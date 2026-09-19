// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/ATen.h>
#include <torch/library.h>
#include <algorithm>
#include <cstring>
#include <vector>
#include "acl/acl_rt.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "activation_diagnostic_launch.h"

namespace vq2a8_ascendc_v4_v2 {
namespace {
constexpr int64_t kDiagnosticMaxRows = 6 * 32;
void CheckDiagnosticStorage(const at::Tensor& value) {
  const auto storageElements = static_cast<uint64_t>(value.storage().nbytes()) / value.element_size();
  TORCH_CHECK(value.storage_offset() >= 0 && static_cast<uint64_t>(value.storage_offset()) <= storageElements,
              "Tail diagnostic storage offset is out of bounds");
  TORCH_CHECK(static_cast<uint64_t>(value.numel()) <=
                  storageElements - static_cast<uint64_t>(value.storage_offset()),
              "Tail diagnostic contiguous input span exceeds storage");
}
void CheckDiagnosticInput(const at::Tensor& value) {
  TORCH_CHECK(value.defined() && value.device().type() == c10::DeviceType::PrivateUse1 &&
                  value.scalar_type() == at::kFloat && value.dim() == 2 && value.is_contiguous(),
              "Tail diagnostic requires contiguous NPU FP32[R,K]");
  TORCH_CHECK(value.size(0) >= 1 && value.size(0) <= kDiagnosticMaxRows &&
                  (value.size(1) == 2048 || value.size(1) == 4096),
              "Tail diagnostic requires R1..192, K2048/4096");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(value.data_ptr()) % 32 == 0,
              "Tail diagnostic requires 32-byte aligned input");
  CheckDiagnosticStorage(value);
}
void RecordDiagnostic(const at::Tensor& value, c10_npu::NPUStream stream) {
  c10_npu::NPUCachingAllocator::recordStream(value.storage().data_ptr(), stream);
}
}  // namespace

std::vector<at::Tensor> ActivationTailDiagnostic(const at::Tensor& rotated,
                                                const at::Tensor& weightScale,
                                                const at::Tensor& rowBias) {
  CheckDiagnosticInput(rotated);
  CheckDiagnosticInput(weightScale);
  TORCH_CHECK(weightScale.device() == rotated.device() && weightScale.sizes() == rotated.sizes(),
              "Tail diagnostic weight_scale must match rotated");
  TORCH_CHECK(rowBias.defined() && rowBias.device() == rotated.device() &&
                  rowBias.scalar_type() == at::kFloat && rowBias.dim() == 1 &&
                  rowBias.size(0) == rotated.size(0) && rowBias.is_contiguous(),
              "Tail diagnostic row_bias requires contiguous FP32[R] on input device");
  CheckDiagnosticStorage(rowBias);
  const c10_npu::OptionalNPUGuard guard(rotated.device());
  const char* soc = aclrtGetSocName();
  TORCH_CHECK(soc && std::strncmp(soc, "Ascend950", 9) == 0, "Tail diagnostic requires Ascend950");
  auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  TORCH_CHECK(platform && platform->GetCoreNumAiv() > 0, "Cannot query diagnostic vector cores");
  const uint32_t rows = rotated.size(0), width = rotated.size(1);
  const auto blocks = std::min(rows, platform->GetCoreNumAiv());
  // transformed, amax, /448, clamped scale, /scale, clamped values, FP8, valid.
  std::vector<at::Tensor> outputs{
      at::empty_like(rotated), at::empty({rows}, rotated.options()),
      at::empty({rows}, rotated.options()), at::empty({rows}, rotated.options()),
      at::empty_like(rotated), at::empty_like(rotated),
      at::empty_like(rotated, rotated.options().dtype(at::kFloat8_e4m3fn)),
      at::empty({rows}, rotated.options().dtype(at::kInt))};
  const auto stream = c10_npu::getCurrentNPUStream();
  for (const auto& value : {rotated, weightScale, rowBias}) RecordDiagnostic(value, stream);
  for (const auto& value : outputs) RecordDiagnostic(value, stream);
  // Resolve before queue insertion and retain owning Tensors in RunOpApi.
  // Diagnostic snapshots add fences, so the Python probe independently checks
  // q/scale/valid against the *unmodified* native quantizer on identical inputs.
  const auto launchStream = stream.stream();
  at_npu::native::OpCommand::RunOpApi("Vq2a8V4V2ActivationTailDiagnostic",
      [launchStream, blocks, rotated, weightScale, rowBias, outputs, rows, width]() -> int {
    LaunchActivationTailDiagnostic(
        launchStream, blocks, rotated.data_ptr(), weightScale.data_ptr(), rowBias.data_ptr(),
        outputs[0].data_ptr(), outputs[1].data_ptr(), outputs[2].data_ptr(), outputs[3].data_ptr(),
        outputs[4].data_ptr(), outputs[5].data_ptr(), outputs[6].data_ptr(), outputs[7].data_ptr(),
        rows, width);
    return 0;
  }, false);
  return outputs;
}
}  // namespace vq2a8_ascendc_v4_v2

TORCH_LIBRARY_FRAGMENT(vq2a8_ascendc_v4_v2, m) {
  m.def("activation_tail_diagnostic_version() -> int", []() -> int64_t { return 1; });
  m.def("activation_tail_diagnostic(Tensor rotated, Tensor weight_scale, Tensor row_bias) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(vq2a8_ascendc_v4_v2, PrivateUse1, m) {
  m.impl("activation_tail_diagnostic", &vq2a8_ascendc_v4_v2::ActivationTailDiagnostic);
}
