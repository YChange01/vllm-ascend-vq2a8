// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/ATen.h>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <cstring>
#include <vector>
#include "acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "input_plan_launch.h"

namespace vq2a8_ascendc_v4_v2 {
namespace {
struct InputTensorContract {
  const void* pointer;
  std::vector<int64_t> sizes, strides;
  int64_t offset;
  explicit InputTensorContract(const at::Tensor& value)
      : pointer(value.data_ptr()), sizes(value.sizes().vec()), strides(value.strides().vec()),
        offset(value.storage_offset()) {}
  void Check(const at::Tensor& value) const {
    TORCH_CHECK(value.data_ptr() == pointer && value.sizes().vec() == sizes &&
                    value.strides().vec() == strides && value.storage_offset() == offset,
                "B1 input plan target storage/layout changed");
  }
};
void CheckTensor(const at::Tensor& value, const c10::Device& device, at::ScalarType dtype) {
  TORCH_CHECK(value.defined() && value.device() == device && value.scalar_type() == dtype &&
                  value.is_contiguous(), "B1 input plan requires matching contiguous NPU tensors");
  const auto bytes = value.storage().nbytes();
  const auto unit = value.element_size();
  TORCH_CHECK(value.storage_offset() >= 0 && static_cast<uint64_t>(value.storage_offset()) <= bytes / unit &&
                  static_cast<uint64_t>(value.numel()) <= bytes / unit - value.storage_offset(),
              "B1 input plan tensor exceeds storage");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(value.data_ptr()) % unit == 0,
              "B1 input plan requires naturally aligned tensors");
}
bool Overlap(const at::Tensor& left, const at::Tensor& right) {
  auto a = reinterpret_cast<uintptr_t>(left.data_ptr()), b = reinterpret_cast<uintptr_t>(right.data_ptr());
  return a < b + right.numel() * right.element_size() && b < a + left.numel() * left.element_size();
}
void Record(const at::Tensor& value, const c10_npu::NPUStream& stream) {
  c10_npu::NPUCachingAllocator::recordStream(value.storage().data_ptr(), stream);
}
}  // namespace

class DecoderInputPlan : public torch::CustomClassHolder {
 public:
  DecoderInputPlan(std::vector<at::Tensor> tables, std::vector<at::Tensor> slots,
                   std::vector<int64_t> blockSizes, std::vector<int64_t> limits)
      : tables_(std::move(tables)), slots_(std::move(slots)) {
    const size_t count = tables_.size();
    TORCH_CHECK(count > 0 && count <= kInputPlanMaxGroups && slots_.size() == count &&
                    blockSizes.size() == count && limits.size() == count, "B1 input plan group schema differs");
    TORCH_CHECK(tables_[0].defined() && tables_[0].device().type() == c10::DeviceType::PrivateUse1,
                "B1 input plan requires NPU tensors");
    device_ = tables_[0].device();
    const c10_npu::OptionalNPUGuard guard(device_);
    const char* soc = aclrtGetSocName();
    TORCH_CHECK(soc && std::strncmp(soc, "Ascend950", 9) == 0, "B1 input plan requires Ascend950");
    auto cpu = at::empty({static_cast<int64_t>(count), kInputPlanDescriptorWords},
                         at::TensorOptions().device(at::kCPU).dtype(at::kLong));
    auto* words = cpu.data_ptr<int64_t>();
    std::vector<at::Tensor> seen;
    for (size_t i = 0; i < count; ++i) {
      auto& table = tables_[i]; auto& slot = slots_[i];
      CheckTensor(table, device_, at::kInt); CheckTensor(slot, device_, at::kInt);
      TORCH_CHECK(table.dim() == 2 && table.size(0) >= 1 && table.size(1) >= 1 &&
                      table.size(1) <= kInputPlanMaxElements && slot.dim() == 1 &&
                      blockSizes[i] > 0 && blockSizes[i] <= INT32_MAX && limits[i] > 0 &&
                      limits[i] <= kInputPlanMaxElements && slot.numel() >= limits[i],
                  "B1 input plan unsupported row/slot geometry");
      for (const auto& value : {table, slot}) {
        for (const auto& previous : seen) TORCH_CHECK(!Overlap(value, previous), "B1 input plan target alias");
        seen.push_back(value);
      }
      tableContracts_.emplace_back(table); slotContracts_.emplace_back(slot);
      const size_t base = i * kInputPlanDescriptorWords;
      words[base] = reinterpret_cast<int64_t>(table.data_ptr());
      words[base + 1] = reinterpret_cast<int64_t>(slot.data_ptr());
      words[base + 2] = table.size(1); words[base + 3] = elements_;
      words[base + 4] = blockSizes[i]; words[base + 5] = limits[i]; words[base + 6] = 0;
      elements_ += table.size(1);
    }
    // Startup-only blocking upload: no per-token descriptor construction,
    // mutable host descriptor, host scalar read or asynchronous owner hazard.
    descriptors_ = cpu.to(device_, at::kLong, false, true);
    stream_ = c10_npu::getCurrentNPUStream().stream();
  }

  void CopyRows(const at::Tensor& packed) {
    const c10_npu::OptionalNPUGuard guard(device_);
    CheckTensor(packed, device_, at::kInt);
    TORCH_CHECK(packed.dim() == 1 && packed.numel() == elements_, "B1 packed row length changed");
    CheckTargets(); CheckInputAlias(packed);
    auto stream = c10_npu::getCurrentNPUStream();
    auto launchStream = stream.stream();
    TORCH_CHECK(launchStream == stream_, "B1 input plan caller stream changed");
    RecordOwners(stream); Record(packed, stream);
    const auto descriptors = descriptors_; const auto tables = tables_; const auto slots = slots_;
    const uint32_t groups = tables.size();
    at_npu::native::OpCommand::RunOpApi("Vq2a8InputRows",
        [launchStream, descriptors, tables, slots, packed, groups]() -> int {
          LaunchInputRows(launchStream, descriptors.data_ptr(), packed.data_ptr(), groups); return 0;
        }, false);
  }

  void SlotMapping(const at::Tensor& query, const at::Tensor& positions) {
    const c10_npu::OptionalNPUGuard guard(device_);
    CheckTensor(query, device_, at::kInt); CheckTensor(positions, device_, at::kLong);
    TORCH_CHECK(query.dim() == 1 && query.numel() == 2 && positions.dim() == 1 && positions.numel() == 1,
                "B1 slot mapping requires query[2], positions[1]");
    CheckTargets(); CheckInputAlias(query); CheckInputAlias(positions);
    TORCH_CHECK(!Overlap(query, positions), "B1 slot input alias");
    auto stream = c10_npu::getCurrentNPUStream(); auto launchStream = stream.stream();
    TORCH_CHECK(launchStream == stream_, "B1 input plan caller stream changed");
    RecordOwners(stream); Record(query, stream); Record(positions, stream);
    const auto descriptors = descriptors_; const auto tables = tables_; const auto slots = slots_;
    const uint32_t groups = tables.size();
    at_npu::native::OpCommand::RunOpApi("Vq2a8InputSlots",
        [launchStream, descriptors, tables, slots, query, positions, groups]() -> int {
          LaunchInputSlots(launchStream, descriptors.data_ptr(), query.data_ptr(), positions.data_ptr(), groups);
          return 0;
        }, false);
  }
 private:
  void CheckTargets() const {
    for (size_t i = 0; i < tables_.size(); ++i) {
      CheckTensor(tables_[i], device_, at::kInt); CheckTensor(slots_[i], device_, at::kInt);
      tableContracts_[i].Check(tables_[i]); slotContracts_[i].Check(slots_[i]);
    }
  }
  void CheckInputAlias(const at::Tensor& input) const {
    for (size_t i = 0; i < tables_.size(); ++i)
      TORCH_CHECK(!Overlap(input, tables_[i]) && !Overlap(input, slots_[i]), "B1 input/output alias");
  }
  void RecordOwners(const c10_npu::NPUStream& stream) const {
    Record(descriptors_, stream);
    for (const auto& table : tables_) Record(table, stream);
    for (const auto& slot : slots_) Record(slot, stream);
  }
  std::vector<at::Tensor> tables_, slots_;
  std::vector<InputTensorContract> tableContracts_, slotContracts_;
  at::Tensor descriptors_;
  c10::Device device_{c10::DeviceType::PrivateUse1, 0};
  int64_t elements_ = 0;
  void* stream_ = nullptr;
};
}  // namespace vq2a8_ascendc_v4_v2

TORCH_LIBRARY_FRAGMENT(vq2a8_ascendc_v4_v2, m) {
  m.def("decoder_input_plan_version() -> int", []() -> int64_t { return 1; });
  m.class_<vq2a8_ascendc_v4_v2::DecoderInputPlan>("DecoderInputPlan")
      .def(torch::init<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<int64_t>,
                       std::vector<int64_t>>())
      .def("copy_rows", &vq2a8_ascendc_v4_v2::DecoderInputPlan::CopyRows)
      .def("slot_mapping", &vq2a8_ascendc_v4_v2::DecoderInputPlan::SlotMapping);
}
