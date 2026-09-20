// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cstring>
#include <tuple>
#include "acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "route_mapping_launch.h"

namespace vq2a8_ascendc_v4_v2 {
namespace {
void CheckRouteMappingTensor(const at::Tensor& tensor, const c10::Device& device,
                             int64_t maximum, const char* name) {
  TORCH_CHECK(tensor.defined() && tensor.device() == device && tensor.scalar_type() == at::kLong &&
                  tensor.dim() == 1 && tensor.is_contiguous() && tensor.numel() >= 1 &&
                  tensor.numel() <= maximum,
              name, ": require matching NPU and contiguous INT64 vector within the supported bounds");
  const auto available = tensor.storage().nbytes() / tensor.element_size();
  TORCH_CHECK(tensor.storage_offset() >= 0 &&
                  static_cast<uint64_t>(tensor.storage_offset()) <= available &&
                  static_cast<uint64_t>(tensor.numel()) <= available - tensor.storage_offset(),
              name, ": tensor span exceeds its storage");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % sizeof(int64_t) == 0,
              name, ": INT64 storage must be naturally aligned");
}
}  // namespace

std::tuple<at::Tensor, at::Tensor> RouteMapping(const at::Tensor& ids, const at::Tensor& lookup) {
  TORCH_CHECK(ids.defined() && ids.device().type() == c10::DeviceType::PrivateUse1,
              "Route mapping requires NPU tensors");
  const auto device = ids.device();
  CheckRouteMappingTensor(ids, device, kRouteMappingMaximumGroups, "route_ids");
  CheckRouteMappingTensor(lookup, device, kRouteMappingMaximumExperts, "route_lookup");
  const c10_npu::OptionalNPUGuard guard(device);
  const char* soc = aclrtGetSocName();
  TORCH_CHECK(soc && std::strncmp(soc, "Ascend950", 9) == 0, "Route mapping requires Ascend950");
  // No aliasing, cached validity latch, or input mutation across calls/replays.
  auto slots = at::empty(ids.sizes(), ids.options());
  auto valid = at::empty({}, ids.options().dtype(at::kBool));
  const auto stream = c10_npu::getCurrentNPUStream();
  c10_npu::NPUCachingAllocator::recordStream(ids.storage().data_ptr(), stream);
  c10_npu::NPUCachingAllocator::recordStream(lookup.storage().data_ptr(), stream);
  // Resolve stream() BEFORE the callback: it may drain the runtime queue.
  const auto launchStream = stream.stream();
  const uint32_t groups = ids.numel(), experts = lookup.numel();
  at_npu::native::OpCommand::RunOpApi("Vq2a8V4V2RouteMapping",
      [launchStream, ids, lookup, slots, valid, groups, experts]() -> int {
    // Owning Tensor captures keep indirect/input/output storage alive until
    // submission. RunOpApi releases callbacks outside the legacy enqueue lock.
    LaunchRouteMapping(launchStream, ids.data_ptr(), lookup.data_ptr(), slots.data_ptr(),
                       valid.data_ptr(), groups, experts);
    return 0;
  }, false);
  return {slots, valid};
}
}  // namespace vq2a8_ascendc_v4_v2

TORCH_LIBRARY_FRAGMENT(vq2a8_ascendc_v4_v2, m) {
  m.def("route_mapping_version() -> int", []() -> int64_t { return 1; });
  m.def("route_mapping(Tensor ids, Tensor lookup) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(vq2a8_ascendc_v4_v2, PrivateUse1, m) {
  m.impl("route_mapping", &vq2a8_ascendc_v4_v2::RouteMapping);
}
