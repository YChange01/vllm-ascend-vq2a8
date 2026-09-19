// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// V4 residency adapter for V2 arithmetic. No V3 native implementation reused.
#include <ATen/ATen.h>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <tuple>
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
#include "select_sign_binding.h"
#include "swiglu_select_sign_binding.h"

namespace vq2a8_ascendc_v4_v2 {
namespace {
using Tensors = std::vector<at::Tensor>;
constexpr int64_t kActivationReorderVersion = 1;
void RecordInputs(const Tensors& tensors, c10_npu::NPUStream stream) {
  for (const auto& tensor : tensors)
    c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
}
void CheckTensor(const at::Tensor& tensor, const at::Tensor& anchor, at::ScalarType dtype,
                 int64_t rank, const char* name, uintptr_t alignment = 32) {
  TORCH_CHECK(tensor.defined(), name, ": undefined tensor");
  TORCH_CHECK(tensor.device() == anchor.device(), name, ": all tensors must share one NPU");
  TORCH_CHECK(tensor.scalar_type() == dtype && tensor.dim() == rank, name, ": incorrect dtype/rank");
  TORCH_CHECK(tensor.is_contiguous(), name, ": contiguous storage required");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0, name, ": unaligned storage offset");
}
struct BankState {
  Tensors packed, books, order, weight_scale, weight_bias, signs;
  at::Tensor table;
  void* stream = nullptr;
  uint32_t experts = 0, n = 0, k = 0, aic_cores = 0, aiv_cores = 0;
};
}  // namespace

class ResidentBank : public torch::CustomClassHolder {
 public:
  ResidentBank(Tensors packed, Tensors books, Tensors order, Tensors weight_scale, Tensors weight_bias,
               Tensors signs) {
    const size_t experts = packed.size();
    TORCH_CHECK(experts > 0 && experts <= kMaxResidentExperts, "V4+V2 bank requires 1..256 experts");
    TORCH_CHECK(books.size() == experts && order.size() == experts && weight_scale.size() == experts &&
                    weight_bias.size() == experts && signs.size() == experts,
                "V4+V2 bank tensor lists must have identical lengths");
    TORCH_CHECK(packed[0].defined() && packed[0].device().type() == c10::DeviceType::PrivateUse1,
                "V4+V2 bank requires an NPU");
    const auto anchor = packed[0];
    CheckTensor(anchor, anchor, at::kByte, 4, "packed_zn");
    const int64_t n = anchor.size(0) * kN0, k = anchor.size(1) * kK0;
    TORCH_CHECK(ValidDimensions(1, n, k), "V4+V2 bank supports N4096, K2048/4096 only");
    // Validate every rank/shape/owner before allocating metadata or reading a
    // value. Payload storage is immutable for the complete bank lifetime.
    for (size_t expert = 0; expert < experts; ++expert) {
      CheckTensor(packed[expert], anchor, at::kByte, 4, "packed_zn");
      CheckTensor(books[expert], anchor, at::kByte, 3, "pair_lut");
      CheckTensor(order[expert], anchor, at::kLong, 1, "activation_order");
      CheckTensor(weight_scale[expert], anchor, at::kFloat, 1, "weight_scale");
      CheckTensor(weight_bias[expert], anchor, at::kFloat, 1, "weight_bias");
      CheckTensor(signs[expert], anchor, at::kChar, 1, "rht_sign");
      TORCH_CHECK(packed[expert].size(0) == n / kN0 && packed[expert].size(1) == k / kK0 &&
                      packed[expert].size(2) == kK0 && packed[expert].size(3) == kN0 / 4,
                  "packed_zn must have shared shape [N/32,K/16,16,8]");
      TORCH_CHECK(books[expert].size(0) == k / kCodebookK && books[expert].size(1) == n / kN0 &&
                      books[expert].size(2) == 32, "pair_lut must have shape [K/256,N/32,32]");
      TORCH_CHECK(order[expert].numel() == k && weight_scale[expert].numel() == k &&
                      weight_bias[expert].numel() == k && signs[expert].numel() == k,
                  "activation_order and preparation metadata must cover K exactly");
    }
    const c10_npu::OptionalNPUGuard guard(anchor.device());
    const char* soc = aclrtGetSocName();
    TORCH_CHECK(soc && std::strncmp(soc, "Ascend950", 9) == 0, "V4+V2 requires Ascend950");
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "CANN platform query failed");
    auto state = std::make_shared<BankState>();
    state->aic_cores = platform->GetCoreNumAic();
    state->aiv_cores = platform->GetCoreNumAiv();
    TORCH_CHECK(state->aic_cores > 0 && state->aiv_cores >= 2 * state->aic_cores,
                "V4+V2 requires a 1C:2V core topology");
    // STARTUP ONLY synchronization: protect the byte-gather's indirect input
    // reads against malformed/manual orders. No sort/equal/item in hot paths.
    const auto expectedOrder = at::arange(k, order[0].options());
    for (const auto& permutation : order) {
      TORCH_CHECK(std::get<0>(at::sort(permutation)).equal(expectedOrder),
                  "activation_order must be a permutation of 0..K-1");
    }
    const auto stream = c10_npu::getCurrentNPUStream();
    state->stream = stream.stream();
    state->experts = experts; state->n = n; state->k = k;
    state->packed = std::move(packed); state->books = std::move(books); state->order = std::move(order);
    state->weight_scale = std::move(weight_scale); state->weight_bias = std::move(weight_bias);
    state->signs = std::move(signs);
    RecordInputs(state->packed, stream); RecordInputs(state->books, stream); RecordInputs(state->order, stream);
    RecordInputs(state->weight_scale, stream); RecordInputs(state->weight_bias, stream); RecordInputs(state->signs, stream);
    auto host = at::zeros({static_cast<int64_t>(experts), kBankWords}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
    auto* records = host.data_ptr<int64_t>();
    for (size_t expert = 0; expert < experts; ++expert) {
      auto* entry = records + expert * kBankWords;
      entry[kBankPacked] = reinterpret_cast<int64_t>(state->packed[expert].data_ptr());
      entry[kBankBook] = reinterpret_cast<int64_t>(state->books[expert].data_ptr());
      entry[kBankOrder] = reinterpret_cast<int64_t>(state->order[expert].data_ptr());
      entry[kBankScale] = reinterpret_cast<int64_t>(state->weight_scale[expert].data_ptr());
      entry[kBankBias] = reinterpret_cast<int64_t>(state->weight_bias[expert].data_ptr());
      entry[kBankSign] = reinterpret_cast<int64_t>(state->signs[expert].data_ptr());
    }
    // The ONLY bank H2D pointer transfer, performed at construction.
    state->table = host.to(anchor.device(), at::kLong, false, true);
    state_ = std::move(state);
  }

  Tensors Select(const at::Tensor& ids) {
    const auto state = state_;
    const c10_npu::OptionalNPUGuard guard(state->table.device());
    CheckIds(ids);
    const int64_t routes = ids.numel();
    auto scale = at::empty({routes, state->k}, state->table.options().dtype(at::kFloat));
    auto bias = at::empty_like(scale);
    auto sign = at::empty({routes, state->k}, state->table.options().dtype(at::kChar));
    auto valid = at::empty({routes}, state->table.options().dtype(at::kInt));
    const uint32_t blocks = std::min(state->aiv_cores, static_cast<uint32_t>(routes) * state->k / kSelectColumns);
    RecordInputs({ids}, c10_npu::getCurrentNPUStream());
    at_npu::native::OpCommand::RunOpApi("Vq2a8AscendCV4V2Select",
        [state, ids, scale, bias, sign, valid, routes, blocks]() -> int {
          LaunchResidentSelect(state->stream, blocks, state->table.data_ptr(), ids.data_ptr(), scale.data_ptr(),
                               bias.data_ptr(), sign.data_ptr(), valid.data_ptr(), state->experts, routes, state->k);
          return 0;
        }, false);
    return {scale, bias, sign, valid};
  }

  Tensors SelectSign(const at::Tensor& hidden, const at::Tensor& ids) {
    const auto state = state_;
    const c10_npu::OptionalNPUGuard guard(state->table.device());
    CheckIds(ids);
    return ResidentSelectSign(hidden, ids, state->table, state->weight_scale,
                              state->weight_bias, state->signs, state->experts, state->k, state->stream);
  }

  Tensors SwigluSelectSign(const at::Tensor& gateUp, const at::Tensor& ids, double limit) {
    const auto state = state_;
    const c10_npu::OptionalNPUGuard guard(state->table.device());
    CheckIds(ids);
    return ResidentSwigluSelectSign(gateUp, ids, state->table, state->weight_scale,
                                   state->weight_bias, state->signs, state->experts,
                                   state->k, state->stream, limit);
  }

  template <bool Vectorized = false, bool PrepareOnly = false, bool NormalizedTail = false,
            bool RowReuse = false, uint32_t ChunkReuse = 0, bool B1Schedule = false>
  Tensors Project(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias, const at::Tensor& ids) {
    const auto state = state_;
    const c10_npu::OptionalNPUGuard guard(state->table.device());
    CheckIds(ids);
    TORCH_CHECK(x.defined() && (x.dim() == 2 || x.dim() == 3), "activation requires [R,K] or [R,M,K]");
    CheckTensor(x, state->table, NormalizedTail ? at::kFloat : at::kFloat8_e4m3fn, x.dim(), "activation");
    if constexpr (NormalizedTail) TORCH_CHECK(x.dim() == 2, "Tail fusion requires M=1 [R,K]");
    CheckTensor(scale, state->table, at::kFloat, x.dim() - 1, "activation_scale", 4);
    CheckTensor(bias, state->table, at::kFloat, x.dim() - 1, "bias_correction", 4);
    const int64_t routes = ids.numel(), m = x.dim() == 2 ? 1 : x.size(1);
    if constexpr (RowReuse) TORCH_CHECK(m == 1, "Row-reuse reorder requires M=1");
    if constexpr (ChunkReuse != 0 || B1Schedule) {
      static_assert(Vectorized && !NormalizedTail && !RowReuse);
      static_assert(ChunkReuse == 0 || ChunkReuse == 2 || ChunkReuse == 4);
      TORCH_CHECK(m == 1, "Chunk-reuse/B1 schedule requires M=1");
    }
    TORCH_CHECK(ValidDimensions(m, state->n, state->k) && x.size(0) == routes && x.size(-1) == state->k &&
                    scale.size(0) == routes && bias.size(0) == routes && scale.numel() == routes * m &&
                    bias.numel() == routes * m, "projection requires matching [R,M,K], [R,M], R<=6, M1..32");
    auto outputShape = x.sizes().vec(); outputShape.back() = state->n;
    auto output = at::empty(outputShape, x.options().dtype(at::kBFloat16));
    auto reordered = NormalizedTail ? at::empty(x.sizes(), x.options().dtype(at::kFloat8_e4m3fn))
                                   : at::empty_like(x);
    auto descriptors = at::empty({routes, kJobWords}, state->table.options());
    auto valid = at::empty({routes}, state->table.options().dtype(at::kInt));
    const uint32_t blocks = std::min(state->aic_cores, static_cast<uint32_t>(routes) * state->n / kN);
    const uint32_t prepareWork = RowReuse ? static_cast<uint32_t>(routes) :
        static_cast<uint32_t>(routes * m) * std::max(state->n, state->k) /
            (kSelectColumns * (ChunkReuse == 0 ? 1 : ChunkReuse));
    const uint32_t prepareBlocks = std::min(state->aiv_cores, prepareWork);
    RecordInputs({x, scale, bias, ids}, c10_npu::getCurrentNPUStream());
    // Safe V4 ownership pattern. RunOpApi releases Tensor-owning callbacks
    // outside the legacy enqueue slot lock. Keep strong owners AND stream
    // records; dropping Python handles must not recycle any indirect pointer.
    at_npu::native::OpCommand::RunOpApi(
        ChunkReuse == 2 ? "Vq2a8AscendCV4V2ChunkReuse2" :
        ChunkReuse == 4 ? "Vq2a8AscendCV4V2ChunkReuse4" :
        B1Schedule ? "Vq2a8AscendCV4V2B1Schedule" :
        RowReuse ? "Vq2a8AscendCV4V2RowReuseReorder" :
        NormalizedTail ? "Vq2a8AscendCV4V2TailReorder" :
        PrepareOnly ? "Vq2a8AscendCV4V2PrepareVectorizedProbe" :
            (Vectorized ? "Vq2a8AscendCV4V2ProjectionVectorized" : "Vq2a8AscendCV4V2Projection"),
        [state, x, scale, bias, ids, reordered, descriptors, output, valid, routes, m, blocks, prepareBlocks]() -> int {
          if constexpr (NormalizedTail) {
            LaunchResidentPrepareTail(state->stream, prepareBlocks, state->table.data_ptr(), ids.data_ptr(),
                x.data_ptr(), scale.data_ptr(), bias.data_ptr(), reordered.data_ptr(), descriptors.data_ptr(),
                output.data_ptr(), valid.data_ptr(), state->experts, routes, m, state->n, state->k);
          } else if constexpr (RowReuse) {
            LaunchResidentPrepareRowReuse(state->stream, prepareBlocks, state->table.data_ptr(), ids.data_ptr(),
                x.data_ptr(), scale.data_ptr(), bias.data_ptr(), reordered.data_ptr(), descriptors.data_ptr(),
                output.data_ptr(), valid.data_ptr(), state->experts, routes, m, state->n, state->k);
          } else if constexpr (ChunkReuse != 0) {
            LaunchResidentPrepareChunkReuse(state->stream, prepareBlocks, state->table.data_ptr(), ids.data_ptr(),
                x.data_ptr(), scale.data_ptr(), bias.data_ptr(), reordered.data_ptr(), descriptors.data_ptr(),
                output.data_ptr(), valid.data_ptr(), state->experts, routes, m, state->n, state->k, ChunkReuse);
          } else if constexpr (Vectorized) {
            LaunchResidentPrepareVectorized(state->stream, prepareBlocks, state->table.data_ptr(), ids.data_ptr(),
                x.data_ptr(), scale.data_ptr(), bias.data_ptr(), reordered.data_ptr(), descriptors.data_ptr(),
                output.data_ptr(), valid.data_ptr(), state->experts, routes, m, state->n, state->k);
          } else {
            LaunchResidentPrepare(state->stream, prepareBlocks, state->table.data_ptr(), ids.data_ptr(), x.data_ptr(),
                                  scale.data_ptr(), bias.data_ptr(), reordered.data_ptr(), descriptors.data_ptr(),
                                  output.data_ptr(), valid.data_ptr(), state->experts, routes, m, state->n, state->k);
          }
          if constexpr (!PrepareOnly) {
            if constexpr (B1Schedule) {
              LaunchGroupedB1(state->stream, blocks, descriptors.data_ptr(), routes, state->n / kN);
            } else {
              LaunchGrouped(state->stream, blocks, descriptors.data_ptr(), routes, state->n / kN);
            }
          }
          return 0;
        }, false);
    if constexpr (PrepareOnly && (NormalizedTail || RowReuse || ChunkReuse != 0)) {
      return {reordered, valid, descriptors, output};
    }
    if constexpr (PrepareOnly) return {reordered, valid};
    return {output, valid};
  }

  Tensors ProjectVectorized(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias,
                           const at::Tensor& ids) {
    return Project<true>(x, scale, bias, ids);
  }

  Tensors ProjectChunkReuse(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias,
                            const at::Tensor& ids, int64_t chunks) {
    TORCH_CHECK(chunks == 2 || chunks == 4, "Chunk-reuse requires chunks=2 or 4");
    if (chunks == 2) return Project<true, false, false, false, 2>(x, scale, bias, ids);
    return Project<true, false, false, false, 4>(x, scale, bias, ids);
  }

  Tensors PrepareChunkReuse(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias,
                            const at::Tensor& ids, int64_t chunks) {
    TORCH_CHECK(chunks == 2 || chunks == 4, "Chunk-reuse requires chunks=2 or 4");
    if (chunks == 2) return Project<true, true, false, false, 2>(x, scale, bias, ids);
    return Project<true, true, false, false, 4>(x, scale, bias, ids);
  }

  Tensors ProjectCandidate(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias,
                           const at::Tensor& ids, int64_t chunks, int64_t schedule) {
    TORCH_CHECK(chunks == 0 || chunks == 2 || chunks == 4, "Candidate reorder chunks must be 0, 2 or 4");
    TORCH_CHECK(schedule == 0 || schedule == 1, "Candidate schedule must be 0 or 1");
    if (schedule == 0) {
      if (chunks == 0) return Project<true>(x, scale, bias, ids);
      return ProjectChunkReuse(x, scale, bias, ids, chunks);
    }
    if (chunks == 0) return Project<true, false, false, false, 0, true>(x, scale, bias, ids);
    if (chunks == 2) return Project<true, false, false, false, 2, true>(x, scale, bias, ids);
    return Project<true, false, false, false, 4, true>(x, scale, bias, ids);
  }

  Tensors ProjectRowReuse(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias,
                          const at::Tensor& ids) {
    return Project<true, false, false, true>(x, scale, bias, ids);
  }

  // Four diagnostic outputs expose descriptor/invalid-output contracts too.
  Tensors PrepareRowReuse(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias,
                          const at::Tensor& ids) {
    return Project<true, true, false, true>(x, scale, bias, ids);
  }

  Tensors ProjectTail(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias, const at::Tensor& ids) {
    return Project<true, false, true>(x, scale, bias, ids);
  }

  Tensors PrepareTail(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias, const at::Tensor& ids) {
    return Project<true, true, true>(x, scale, bias, ids);
  }

  // DIAGNOSTIC ONLY: expose the exact gathered bytes before any GEMM can hide
  // permutation defects by cancellation. Invalid route rows are unspecified;
  // callers must check the returned validity mask before reading those rows.
  Tensors PrepareVectorized(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias,
                           const at::Tensor& ids) {
    return Project<true, true>(x, scale, bias, ids);
  }

  std::vector<int64_t> Metadata() const {
    return {state_->experts, state_->n, state_->k, static_cast<int64_t>(state_->table.nbytes())};
  }

 private:
  void CheckIds(const at::Tensor& ids) const {
    CheckTensor(ids, state_->table, at::kLong, 1, "resident_ids", sizeof(int64_t));
    TORCH_CHECK(ids.numel() > 0 && ids.numel() <= kMaxJobs, "V4+V2 requires 1..6 route slots");
    TORCH_CHECK(c10_npu::getCurrentNPUStream().stream() == state_->stream,
                "V4+V2 bank must be used on its construction NPU stream");
  }
  std::shared_ptr<BankState> state_;
};
}  // namespace vq2a8_ascendc_v4_v2

TORCH_LIBRARY_FRAGMENT(vq2a8_ascendc_v4_v2, m) {
  using Tensors = std::vector<at::Tensor>;
  m.def("activation_reorder_version() -> int", []() -> int64_t {
    return vq2a8_ascendc_v4_v2::kActivationReorderVersion;
  });
  m.def("activation_tail_reorder_version() -> int", []() -> int64_t { return 1; });
  m.def("activation_reorder_row_reuse_version() -> int", []() -> int64_t { return 1; });
  m.def("activation_reorder_chunk_reuse_version() -> int", []() -> int64_t { return 1; });
  m.def("b1_schedule_version() -> int", []() -> int64_t { return 1; });
  m.class_<vq2a8_ascendc_v4_v2::ResidentBank>("ResidentBank")
      .def(torch::init<Tensors, Tensors, Tensors, Tensors, Tensors, Tensors>())
      .def("select", &vq2a8_ascendc_v4_v2::ResidentBank::Select)
      .def("select_sign", &vq2a8_ascendc_v4_v2::ResidentBank::SelectSign)
      .def("swiglu_select_sign", &vq2a8_ascendc_v4_v2::ResidentBank::SwigluSelectSign)
      .def("project", &vq2a8_ascendc_v4_v2::ResidentBank::Project<false>)
      .def("project_vectorized", &vq2a8_ascendc_v4_v2::ResidentBank::ProjectVectorized)
      .def("prepare_vectorized", &vq2a8_ascendc_v4_v2::ResidentBank::PrepareVectorized)
      .def("project_row_reuse", &vq2a8_ascendc_v4_v2::ResidentBank::ProjectRowReuse)
      .def("prepare_row_reuse", &vq2a8_ascendc_v4_v2::ResidentBank::PrepareRowReuse)
      .def("project_chunk_reuse", &vq2a8_ascendc_v4_v2::ResidentBank::ProjectChunkReuse)
      .def("prepare_chunk_reuse", &vq2a8_ascendc_v4_v2::ResidentBank::PrepareChunkReuse)
      .def("project_candidate", &vq2a8_ascendc_v4_v2::ResidentBank::ProjectCandidate)
      .def("project_tail", &vq2a8_ascendc_v4_v2::ResidentBank::ProjectTail)
      .def("prepare_tail", &vq2a8_ascendc_v4_v2::ResidentBank::PrepareTail)
      .def("metadata", &vq2a8_ascendc_v4_v2::ResidentBank::Metadata);
}
