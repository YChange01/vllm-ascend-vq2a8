// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Host metadata only. This file deliberately has no CANN/torch_npu headers,
// device guards, stream access, tensor kernels, or device synchronization.
#include <ATen/ATen.h>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vq2a8_ascendc_v4_v2 {
namespace {
constexpr const char* kContractError =
    "V4 MoE graph runtime/root/geometry signature changed; no implicit recapture.";

struct TensorSnapshot {
  at::Tensor owner;
  // set_()/data replacement can mutate an existing TensorImpl. Retain the
  // original allocation too, preventing address reuse from hiding a change.
  c10::Storage storage_owner;
  const c10::TensorImpl* implementation;
  const void* pointer;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int64_t offset;
  at::ScalarType dtype;
  c10::Device device;
  c10::Layout layout;
  std::string label;

  TensorSnapshot(const at::Tensor& tensor, std::string name)
      : owner(tensor), storage_owner(tensor.storage()), implementation(tensor.unsafeGetTensorImpl()),
        pointer(tensor.const_data_ptr()),
        sizes(tensor.sizes().vec()), strides(tensor.strides().vec()), offset(tensor.storage_offset()),
        dtype(tensor.scalar_type()), device(tensor.device()), layout(tensor.layout()), label(std::move(name)) {}

  void Check(const at::Tensor& current, size_t index) const {
    // Check layout/device before touching data_ptr/strides: a sparse or meta
    // replacement must fail closed without running any backend operation.
    TORCH_CHECK(current.defined() && current.unsafeGetTensorImpl() == implementation &&
                    current.layout() == layout && current.device() == device &&
                    current.scalar_type() == dtype && current.sizes().equals(sizes) &&
                    current.strides().equals(strides) && current.storage_offset() == offset &&
                    current.const_data_ptr() == pointer,
                kContractError, " (tensor ", index, ": ", label, ")");
  }
};
}  // namespace

class RuntimeTensorGuard : public torch::CustomClassHolder {
 public:
  RuntimeTensorGuard(std::vector<at::Tensor> tensors, std::vector<std::string> labels) {
    TORCH_CHECK(tensors.size() == labels.size(), "Runtime guard tensors/labels length differs");
    snapshots_.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
      TORCH_CHECK(tensors[i].defined() && tensors[i].layout() == c10::kStrided && !tensors[i].is_meta(),
                  "Runtime guard capture requires defined, strided, storage-backed tensors");
      snapshots_.emplace_back(tensors[i], std::move(labels[i]));
    }
  }

  void Append(const c10::intrusive_ptr<RuntimeTensorGuard>& other) {
    TORCH_CHECK(!checked_ && other.get() != this, "Runtime guard aggregation is startup-only, not self-append");
    snapshots_.insert(snapshots_.end(), other->snapshots_.begin(), other->snapshots_.end());
  }

  void Check(std::vector<at::Tensor> current) {
    checked_ = true;
    TORCH_CHECK(current.size() == snapshots_.size(), kContractError, " (tensor count)");
    for (size_t i = 0; i < snapshots_.size(); ++i) snapshots_[i].Check(current[i], i);
  }

 private:
  std::vector<TensorSnapshot> snapshots_;
  bool checked_ = false;
};
}  // namespace vq2a8_ascendc_v4_v2

TORCH_LIBRARY_FRAGMENT(vq2a8_ascendc_v4_v2, m) {
  m.def("runtime_guard_version() -> int", []() -> int64_t { return 1; });
  m.class_<vq2a8_ascendc_v4_v2::RuntimeTensorGuard>("RuntimeTensorGuard")
      .def(torch::init<std::vector<at::Tensor>, std::vector<std::string>>())
      .def("append", &vq2a8_ascendc_v4_v2::RuntimeTensorGuard::Append)
      .def("check", &vq2a8_ascendc_v4_v2::RuntimeTensorGuard::Check);
}
