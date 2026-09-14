// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/ATen.h>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>
#include "acl/acl_rt.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "launch.h"
#include "resident_layout.h"

namespace vq2a8_ascendc {
namespace {
using Tensors = std::vector<at::Tensor>;

void RecordResidentInputs(const Tensors& tensors, c10_npu::NPUStream stream) {
  for (const auto& tensor : tensors) {
    c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
  }
}

void CheckResidentTensor(const at::Tensor& tensor, const at::Tensor& anchor, at::ScalarType dtype, int64_t rank,
                         const char* name, uintptr_t alignment = 32) {
  TORCH_CHECK(tensor.defined(), name, ": undefined tensor");
  TORCH_CHECK(tensor.device() == anchor.device(), name, ": all tensors must share one NPU");
  TORCH_CHECK(tensor.scalar_type() == dtype && tensor.dim() == rank, name, ": wrong dtype/rank");
  TORCH_CHECK(tensor.is_contiguous(), name, ": contiguous storage is required");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0, name, ": unaligned storage offset");
}

struct ResidentBankState {
  Tensors packed, books, tile_ids, weight_scale, weight_bias, signs;
  at::Tensor table;
  void* stream = nullptr;
  uint32_t experts = 0, n = 0, k = 0, aic_cores = 0, aiv_cores = 0;
};
}  // namespace

// An opt-in single-stream, M=1-per-route bank. It neither clones/decodes
// expert weights nor changes any original V1 operator schema. Payload Tensor
// storage must remain immutable while the bank is alive (the V4 contract).
class ResidentBank : public torch::CustomClassHolder {
 public:
  ResidentBank(Tensors packed, Tensors books, Tensors tile_ids, Tensors weight_scale, Tensors weight_bias,
               Tensors signs) {
    const size_t experts = packed.size();
    TORCH_CHECK(experts > 0 && experts <= kMaxResidentExperts, "Resident bank requires 1..256 experts");
    TORCH_CHECK(books.size() == experts && tile_ids.size() == experts && weight_scale.size() == experts &&
                    weight_bias.size() == experts && signs.size() == experts,
                "Resident bank tensor lists must have identical lengths");
    TORCH_CHECK(packed[0].defined() && packed[0].device().type() == c10::DeviceType::PrivateUse1,
                "Resident bank requires an NPU");
    const at::Tensor anchor = packed[0];
    CheckResidentTensor(anchor, anchor, at::kInt, 2, "packed_indices");
    TORCH_CHECK(anchor.size(0) > 0 && anchor.size(0) <= kMaxDimension / 2 && anchor.size(1) > 0 &&
                    anchor.size(1) <= kMaxDimension / 8,
                "Resident bank packed dimensions exceed N,K<=65536");
    const int64_t n = anchor.size(0) * 2, k = anchor.size(1) * 8;
    TORCH_CHECK(ValidDimensions(1, n, k, 1), "Resident bank requires N%32=0, K%512=0 and N,K<=65536");
    // Validate every owner before allocating or uploading a pointer table.
    for (size_t expert = 0; expert < experts; ++expert) {
      CheckResidentTensor(packed[expert], anchor, at::kInt, 2, "packed_indices");
      CheckResidentTensor(books[expert], anchor, at::kFloat8_e4m3fn, 4, "codebooks");
      CheckResidentTensor(tile_ids[expert], anchor, at::kByte, 1, "tile_ids");
      CheckResidentTensor(weight_scale[expert], anchor, at::kFloat, 1, "weight_scale");
      CheckResidentTensor(weight_bias[expert], anchor, at::kFloat, 1, "weight_bias");
      CheckResidentTensor(signs[expert], anchor, at::kChar, 1, "rht_sign");
      TORCH_CHECK(packed[expert].size(0) == n / 2 && packed[expert].size(1) == k / 8,
                  "Resident bank experts must share N and K");
      TORCH_CHECK(ValidDimensions(1, n, k, books[expert].size(0)) && books[expert].size(1) == n / kN &&
                      books[expert].size(2) == 16 && books[expert].size(3) == 2,
                  "Resident bank codebooks must have shape [tiles,N/32,16,2]");
      TORCH_CHECK(tile_ids[expert].numel() == k && weight_scale[expert].numel() == k &&
                      weight_bias[expert].numel() == k && signs[expert].numel() == k,
                  "Resident bank tile IDs and preparation metadata must cover K exactly");
    }
    const c10_npu::OptionalNPUGuard guard(anchor.device());
    const char* soc = aclrtGetSocName();
    TORCH_CHECK(soc && std::strncmp(soc, "Ascend950", 9) == 0, "Resident VQ2A8 is Ascend950-only");
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "CANN platform query failed");
    auto state = std::make_shared<ResidentBankState>();
    state->aic_cores = platform->GetCoreNumAic();
    state->aiv_cores = platform->GetCoreNumAiv();
    TORCH_CHECK(state->aic_cores > 0 && state->aiv_cores >= 2 * state->aic_cores,
                "Resident VQ2A8 requires a 1C:2V core topology");
    const auto constructionStream = c10_npu::getCurrentNPUStream();
    state->stream = constructionStream.stream();
    state->experts = experts;
    state->n = n;
    state->k = k;
    state->packed = std::move(packed);
    state->books = std::move(books);
    state->tile_ids = std::move(tile_ids);
    state->weight_scale = std::move(weight_scale);
    state->weight_bias = std::move(weight_bias);
    state->signs = std::move(signs);
    // Record all indirect owners ONCE, including allocations originally made
    // on another stream. If the caller drops the bank immediately after a
    // launch, the caching allocator still waits for this stream before reuse.
    // This is ownership tracking, not an implicit wait for external producers;
    // the caller must make payload writes ready before constructing the bank.
    RecordResidentInputs(state->packed, constructionStream);
    RecordResidentInputs(state->books, constructionStream);
    RecordResidentInputs(state->tile_ids, constructionStream);
    RecordResidentInputs(state->weight_scale, constructionStream);
    RecordResidentInputs(state->weight_bias, constructionStream);
    RecordResidentInputs(state->signs, constructionStream);
    auto host =
        at::zeros({static_cast<int64_t>(experts), kBankWords}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
    auto* records = host.data_ptr<int64_t>();
    for (size_t expert = 0; expert < experts; ++expert) {
      auto* entry = records + expert * kBankWords;
      entry[kBankPacked] = reinterpret_cast<int64_t>(state->packed[expert].data_ptr());
      entry[kBankBook] = reinterpret_cast<int64_t>(state->books[expert].data_ptr());
      entry[kBankTileIds] = reinterpret_cast<int64_t>(state->tile_ids[expert].data_ptr());
      entry[kBankScale] = reinterpret_cast<int64_t>(state->weight_scale[expert].data_ptr());
      entry[kBankBias] = reinterpret_cast<int64_t>(state->weight_bias[expert].data_ptr());
      entry[kBankSign] = reinterpret_cast<int64_t>(state->signs[expert].data_ptr());
      entry[kBankTiles] = state->books[expert].size(0);
    }
    // The ONLY pointer-table H2D copy. Blocking is intentional at startup;
    // runtime select/project do not construct host descriptors or read IDs.
    state->table = host.to(anchor.device(), at::kLong, false, true);
    state_ = std::move(state);
  }

  // Use EXECUTE_OPAPI for Tensor-owning callbacks. The legacy
  // SetCustomHandler/Run path can retain its handler in a queue slot until
  // that slot is reused under the enqueue mutex. Destroying captured Tensors
  // there may record an allocator event and recursively enqueue, deadlocking.
  // RunOpApi transfers the handler to the release queue instead. Keep both
  // strong owners and recordStream: submission is still asynchronous.
  Tensors Select(const at::Tensor& ids) {
    const auto state = state_;
    const c10_npu::OptionalNPUGuard guard(state->table.device());
    CheckIds(ids);
    auto options = state->table.options();
    const int64_t routes = ids.numel();
    auto scale = at::empty({routes, state->k}, options.dtype(at::kFloat));
    auto bias = at::empty({routes, state->k}, options.dtype(at::kFloat));
    auto sign = at::empty({routes, state->k}, options.dtype(at::kChar));
    auto valid = at::empty({routes}, options.dtype(at::kInt));
    const uint32_t blocks = std::min(state->aiv_cores, static_cast<uint32_t>(routes) * state->k / kSelectColumns);
    RecordResidentInputs({ids}, c10_npu::getCurrentNPUStream());
    at_npu::native::OpCommand::RunOpApi(
        "Vq2a8AscendCResidentSelect",
        [state, ids, scale, bias, sign, valid, routes, blocks]() -> int {
          // Capture the complete bank, not just its table; every indirect GM
          // pointer remains owned during asynchronous launch submission.
          LaunchResidentSelect(state->stream, blocks, state->table.data_ptr(), ids.data_ptr(), scale.data_ptr(),
                               bias.data_ptr(), sign.data_ptr(), valid.data_ptr(), state->experts, routes, state->k);
          return 0;
        },
        false);
    return {scale, bias, sign, valid};
  }

  Tensors Project(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias, const at::Tensor& ids) {
    const auto state = state_;
    const c10_npu::OptionalNPUGuard guard(state->table.device());
    CheckIds(ids);
    CheckResidentTensor(x, state->table, at::kFloat8_e4m3fn, 2, "activation");
    CheckResidentTensor(scale, state->table, at::kFloat, 1, "activation_scale", 4);
    CheckResidentTensor(bias, state->table, at::kFloat, 1, "bias_correction", 4);
    const int64_t routes = ids.numel();
    TORCH_CHECK(x.size(0) == routes && x.size(1) == state->k && scale.numel() == routes && bias.numel() == routes,
                "Resident projection requires FP8[R,K], FP32[R] scale/bias and int64[R] IDs, 1<=R<=6");
    auto output = at::empty({routes, state->n}, x.options().dtype(at::kBFloat16));
    auto valid = at::empty({routes}, x.options().dtype(at::kInt));
    const uint32_t blocks = std::min(state->aic_cores, static_cast<uint32_t>(routes) * state->n / kN);
    RecordResidentInputs({x, scale, bias, ids}, c10_npu::getCurrentNPUStream());
    at_npu::native::OpCommand::RunOpApi(
        "Vq2a8AscendCResidentProjection",
        [state, x, scale, bias, ids, output, valid, routes, blocks]() -> int {
          LaunchResident(state->stream, blocks, state->table.data_ptr(), ids.data_ptr(), x.data_ptr(), scale.data_ptr(),
                         bias.data_ptr(), output.data_ptr(), valid.data_ptr(), state->experts, routes, state->n,
                         state->k);
          return 0;
        },
        false);
    return {output, valid};
  }

  std::vector<int64_t> Metadata() const {
    return {state_->experts, state_->n, state_->k, static_cast<int64_t>(state_->table.nbytes())};
  }

 private:
  void CheckIds(const at::Tensor& ids) const {
    CheckResidentTensor(ids, state_->table, at::kLong, 1, "resident_ids", sizeof(int64_t));
    TORCH_CHECK(ids.numel() > 0 && ids.numel() <= kMaxJobs, "Resident decode requires 1..6 route slots");
    // The underlying V1 prototype uses same-stream allocator ordering. Do
    // not silently allow a second stream to race Tensor-owner destruction.
    TORCH_CHECK(c10_npu::getCurrentNPUStream().stream() == state_->stream,
                "Resident bank must be used on its construction NPU stream");
  }

  std::shared_ptr<ResidentBankState> state_;
};
}  // namespace vq2a8_ascendc

TORCH_LIBRARY_FRAGMENT(vq2a8_ascendc, m) {
  using Tensors = std::vector<at::Tensor>;
  m.class_<vq2a8_ascendc::ResidentBank>("ResidentBank")
      .def(torch::init<Tensors, Tensors, Tensors, Tensors, Tensors, Tensors>())
      .def("select", &vq2a8_ascendc::ResidentBank::Select)
      .def("project", &vq2a8_ascendc::ResidentBank::Project)
      .def("metadata", &vq2a8_ascendc::ResidentBank::Metadata);
}
