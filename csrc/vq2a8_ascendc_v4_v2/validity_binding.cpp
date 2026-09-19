// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/ATen.h>
#include <torch/library.h>
#include <array>
#include <cstring>
#include <vector>
#include "acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "validity_launch.h"

namespace vq2a8_ascendc_v4_v2 {
namespace {
constexpr int64_t kMaximumGroups = 6;
constexpr uintptr_t kOutputAlignment = 32;

void CheckSpan(const at::Tensor& tensor) {
  const auto available = tensor.storage().nbytes() / tensor.element_size();
  TORCH_CHECK(tensor.storage_offset() >= 0 &&
                  static_cast<uint64_t>(tensor.storage_offset()) <= available &&
                  static_cast<uint64_t>(tensor.numel()) <= available - tensor.storage_offset(),
              "Layer validity tensor span exceeds its storage");
}

void CheckTensor(const at::Tensor& tensor, const c10::Device& device, at::ScalarType dtype) {
  TORCH_CHECK(tensor.defined() && tensor.device() == device && tensor.scalar_type() == dtype &&
                  tensor.is_contiguous(), "Layer validity requires matching device/dtype and contiguous tensors");
  CheckSpan(tensor);
}

void CheckOutput(const at::Tensor& tensor, const c10::Device& device, int64_t groups, bool combined) {
  CheckTensor(tensor, device, at::kBFloat16);
  TORCH_CHECK((tensor.dim() == 2 || (!combined && tensor.dim() == 3 && tensor.size(1) == 1)) &&
                  tensor.size(0) == (combined ? 1 : groups) &&
                  (tensor.size(-1) == 2048 || tensor.size(-1) == 4096),
              "Layer validity outputs require BF16[G,N]/[G,1,N] and result[1,H], N/H 2048 or 4096");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % kOutputAlignment == 0,
              "Layer validity outputs require 32-byte alignment");
}
}  // namespace

at::Tensor LayerValidity(at::TensorList statusList, at::TensorList outputList, at::TensorList flagList) {
  TORCH_CHECK(statusList.size() == kLayerValidityStatuses && outputList.size() == kLayerValidityOutputs &&
                  flagList.size() <= kLayerValidityRouteFlags,
              "Layer validity requires six statuses, three outputs, and at most eight route flags");
  TORCH_CHECK(statusList[0].defined() && statusList[0].device().type() == c10::DeviceType::PrivateUse1,
              "Layer validity requires NPU tensors");
  const auto device = statusList[0].device();
  TORCH_CHECK(statusList[0].dim() == 1 && statusList[0].size(0) >= 1 &&
                  statusList[0].size(0) <= kMaximumGroups,
              "Layer validity statuses require INT32[G], G 1..6");
  const auto groups = statusList[0].size(0);
  for (const auto& status : statusList) {
    CheckTensor(status, device, at::kInt);
    TORCH_CHECK(status.dim() == 1 && status.size(0) == groups, "Layer validity status shapes must match");
  }
  for (size_t i = 0; i < outputList.size(); ++i) CheckOutput(outputList[i], device, groups, i == 2);
  TORCH_CHECK(outputList[1].size(-1) == outputList[2].size(-1), "Down and combined output widths must match");
  for (const auto& flag : flagList) {
    CheckTensor(flag, device, at::kBool);
    TORCH_CHECK(flag.dim() == 0, "Layer validity route flags must be BOOL scalars");
  }
  const c10_npu::OptionalNPUGuard guard(device);
  const char* soc = aclrtGetSocName();
  TORCH_CHECK(soc && std::strncmp(soc, "Ascend950", 9) == 0, "Layer validity requires Ascend950");
  auto result = at::empty({}, statusList[0].options().dtype(at::kBool));
  // TensorList is a borrowed ArrayRef. Capture owning vectors, never the lists.
  std::vector<at::Tensor> statuses(statusList.begin(), statusList.end());
  std::vector<at::Tensor> outputs(outputList.begin(), outputList.end());
  std::vector<at::Tensor> flags(flagList.begin(), flagList.end());
  const auto stream = c10_npu::getCurrentNPUStream();
  for (const auto* tensors : {&statuses, &outputs, &flags}) {
    for (const auto& tensor : *tensors) c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
  }
  // stream() may drain the runtime queue. Resolve it before RunOpApi, never in
  // its callback. RunOpApi also avoids legacy callback destruction reentrancy.
  const auto launchStream = stream.stream();
  const uint32_t gateWidth = outputs[0].size(-1), downWidth = outputs[1].size(-1);
  at_npu::native::OpCommand::RunOpApi("Vq2a8V4V2LayerValidity",
      [launchStream, statuses, outputs, flags, result, groups, gateWidth, downWidth]() -> int {
    std::array<void*, kLayerValidityStatuses> statusPointers{};
    std::array<void*, kLayerValidityOutputs> outputPointers{};
    std::array<void*, kLayerValidityRouteFlags> flagPointers{};
    for (size_t i = 0; i < statuses.size(); ++i) statusPointers[i] = statuses[i].data_ptr();
    for (size_t i = 0; i < outputs.size(); ++i) outputPointers[i] = outputs[i].data_ptr();
    for (size_t i = 0; i < flags.size(); ++i) flagPointers[i] = flags[i].data_ptr();
    LaunchLayerValidity(launchStream, statusPointers.data(), outputPointers.data(), flagPointers.data(),
                        result.data_ptr(), groups, gateWidth, downWidth, flags.size());
    return 0;
  }, false);
  return result;
}
}  // namespace vq2a8_ascendc_v4_v2

TORCH_LIBRARY_FRAGMENT(vq2a8_ascendc_v4_v2, m) {
  m.def("layer_validity_version() -> int", []() -> int64_t { return 1; });
  m.def("layer_validity(Tensor[] statuses, Tensor[] outputs, Tensor[] route_flags) -> Tensor");
}
TORCH_LIBRARY_IMPL(vq2a8_ascendc_v4_v2, PrivateUse1, m) {
  m.impl("layer_validity", &vq2a8_ascendc_v4_v2::LayerValidity);
}
