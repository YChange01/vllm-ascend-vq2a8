// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "select_sign_binding.h"
#include "select_sign_launch.h"
#include <torch/library.h>
#include <algorithm>
#include <cstring>
#include "acl/acl_rt.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace vq2a8_ascendc_v4_v2 {
namespace {
void CheckInput(const at::Tensor& hidden, const at::Tensor& ids, const at::Tensor& table, uint32_t width) {
  TORCH_CHECK(hidden.defined() && table.defined() &&
                  hidden.device().type() == c10::DeviceType::PrivateUse1 && hidden.device() == table.device(),
              "Resident select/sign requires the bank NPU");
  TORCH_CHECK(hidden.dim() == 2 && (hidden.scalar_type() == at::kFloat || hidden.scalar_type() == at::kBFloat16) &&
                  hidden.size(0) >= 1 && hidden.size(0) <= 6 && hidden.size(1) == width &&
                  (width == 2048 || width == 4096), "Resident select/sign requires BF16/FP32[G,K], G1..6, K2048/4096");
  TORCH_CHECK(hidden.stride(1) == 1 && (hidden.stride(0) == 0 || hidden.stride(0) >= width),
              "Resident select/sign requires unit columns and expanded or nonoverlapping rows");
  const uint64_t elementBytes = hidden.element_size(), rowStride = hidden.stride(0);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(hidden.data_ptr()) % 32 == 0 &&
                  (hidden.size(0) == 1 || rowStride % (32 / elementBytes) == 0),
              "Resident select/sign input rows require 32-byte alignment");
  const uint64_t storageElements = hidden.storage().nbytes() / elementBytes;
  TORCH_CHECK(hidden.storage_offset() >= 0 && static_cast<uint64_t>(hidden.storage_offset()) <= storageElements,
              "Resident select/sign input storage offset is invalid");
  const uint64_t available = storageElements - hidden.storage_offset();
  TORCH_CHECK(width <= available && (hidden.size(0) == 1 ||
                  rowStride <= (available - width) / static_cast<uint64_t>(hidden.size(0) - 1)),
              "Resident select/sign input row span exceeds storage");
  TORCH_CHECK(ids.defined() && ids.device() == table.device() && ids.scalar_type() == at::kLong &&
                  ids.dim() == 1 && ids.is_contiguous() && ids.numel() == hidden.size(0),
              "Resident select/sign requires contiguous INT64 slots[G] on the bank device");
  const uint64_t idElements = ids.storage().nbytes() / sizeof(int64_t);
  TORCH_CHECK(ids.storage_offset() >= 0 && static_cast<uint64_t>(ids.storage_offset()) <= idElements &&
                  static_cast<uint64_t>(ids.numel()) <= idElements - ids.storage_offset() &&
                  reinterpret_cast<uintptr_t>(ids.data_ptr()) % sizeof(int64_t) == 0,
              "Resident select/sign slot storage is invalid");
}
void Record(const at::Tensor& tensor, c10_npu::NPUStream stream) {
  c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
}
}  // namespace

std::vector<at::Tensor> ResidentSelectSign(
    const at::Tensor& hidden, const at::Tensor& ids, const at::Tensor& table,
    const std::vector<at::Tensor>& scaleOwners, const std::vector<at::Tensor>& biasOwners,
    const std::vector<at::Tensor>& signOwners, uint32_t experts, uint32_t width, void* bankStream) {
  CheckInput(hidden, ids, table, width);
  TORCH_CHECK(experts >= 1 && experts <= 256 && scaleOwners.size() == experts &&
                  biasOwners.size() == experts && signOwners.size() == experts,
              "Resident select/sign requires complete immutable bank owners");
  const c10_npu::OptionalNPUGuard guard(table.device());
  const char* soc = aclrtGetSocName();
  TORCH_CHECK(soc && std::strncmp(soc, "Ascend950", 9) == 0, "Resident select/sign requires Ascend950");
  auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  TORCH_CHECK(platform && platform->GetCoreNumAiv() > 0, "Cannot query Ascend vector cores");
  auto signedOutput = at::empty(hidden.sizes(), hidden.options().dtype(at::kFloat));
  auto selectedScale = at::empty_like(signedOutput), selectedBias = at::empty_like(signedOutput);
  auto selectStatus = at::empty({hidden.size(0)}, hidden.options().dtype(at::kInt));
  auto inputStatus = at::empty_like(selectStatus);
  const auto stream = c10_npu::getCurrentNPUStream();
  // Resolve BEFORE enqueue, never drain a queue from its own launch callback.
  const auto launchStream = stream.stream();
  TORCH_CHECK(launchStream == bankStream, "Resident select/sign must use the bank creation stream");
  for (const auto& input : {hidden, ids, table}) Record(input, stream);
  for (const auto& owner : scaleOwners) Record(owner, stream);
  for (const auto& owner : biasOwners) Record(owner, stream);
  for (const auto& owner : signOwners) Record(owner, stream);
  const uint32_t groups = hidden.size(0), blocks = std::min(groups, platform->GetCoreNumAiv());
  const uint64_t rowStride = hidden.stride(0);
  const bool inputIsBf16 = hidden.scalar_type() == at::kBFloat16;
  at_npu::native::OpCommand::RunOpApi("Vq2a8V4V2ResidentSelectSign",
      [launchStream, blocks, table, ids, hidden, signedOutput, selectedScale, selectedBias, selectStatus,
       inputStatus, scaleOwners, biasOwners, signOwners, experts, groups, width, rowStride, inputIsBf16]() -> int {
    // Strong Tensor captures retain every indirect metadata pointer even when
    // the Python bank/input owners are dropped before the queue reaches us.
    (void)scaleOwners; (void)biasOwners; (void)signOwners;
    LaunchResidentSelectSign(launchStream, blocks, table.data_ptr(), ids.data_ptr(), hidden.data_ptr(),
        signedOutput.data_ptr(), selectedScale.data_ptr(), selectedBias.data_ptr(), selectStatus.data_ptr(),
        inputStatus.data_ptr(), experts, groups, width, rowStride, inputIsBf16);
    return 0;
  }, false);
  return {signedOutput, selectedScale, selectedBias, selectStatus, inputStatus};
}
}  // namespace vq2a8_ascendc_v4_v2

TORCH_LIBRARY_FRAGMENT(vq2a8_ascendc_v4_v2, m) {
  m.def("select_sign_version() -> int", []() -> int64_t { return 1; });
}
