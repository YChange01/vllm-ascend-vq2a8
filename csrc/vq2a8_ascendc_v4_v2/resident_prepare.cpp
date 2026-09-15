// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// New V4 adapter: no dependency on the experimental V3 resident implementation.
#include "kernel_operator.h"
#include "launch.h"
#define VQ2_V2_LAYOUT_FN __aicore__ inline
#include "resident_layout.h"
#undef VQ2_V2_LAYOUT_FN

namespace vq2a8_ascendc_v4_v2 {
using namespace AscendC;
template <HardEvent E>
__aicore__ inline void PrepareFence() {
  event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
  SetFlag<E>(event);
  WaitFlag<E>(event);
}

// Correctness-first byte gather. Only activation bytes are reordered, AFTER
// original RHT/FP8 preparation; compressed expert weights never move at runtime.
// Route/row/K chunks are independent, including repeated expert IDs. This
// scalar gather is deliberately isolated for later profiling/vectorization.
class ResidentPrepareKernel {
 public:
  __aicore__ inline void Init(GM_ADDR bank, GM_ADDR routeIds, GM_ADDR x, GM_ADDR scale, GM_ADDR bias,
                             GM_ADDR reordered, GM_ADDR descriptors, GM_ADDR output, GM_ADDR valid,
                             uint32_t experts, uint32_t routes, uint32_t m, uint32_t n, uint32_t k) {
    experts_ = experts; routes_ = routes; m_ = m; n_ = n; k_ = k;
    xAddr_ = reinterpret_cast<uint64_t>(reordered);
    scaleAddr_ = reinterpret_cast<uint64_t>(scale);
    biasAddr_ = reinterpret_cast<uint64_t>(bias);
    outputAddr_ = reinterpret_cast<uint64_t>(output);
    bank_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(bank));
    ids_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(routeIds));
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    reordered_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(reordered));
    descriptors_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(descriptors));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(output));
    valid_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valid));
    pipe_.InitBuffer(gatherUb_, kSelectColumns);
    pipe_.InitBuffer(outputUb_, kSelectColumns * sizeof(uint16_t));
    pipe_.InitBuffer(recordUb_, 96);  // Nine uint64 words, rounded to a DMA block.
    pipe_.InitBuffer(statusUb_, 32);
  }

  __aicore__ inline void Process() {
    const uint32_t chunks = (n_ > k_ ? n_ : k_) / kSelectColumns;
    for (uint32_t work = GetBlockIdx(); work < routes_ * m_ * chunks; work += GetBlockNum()) {
      const uint32_t route = work / (m_ * chunks);
      const uint32_t row = (work / chunks) % m_;
      const uint32_t column = (work % chunks) * kSelectColumns;
      const int64_t expert = ids_.GetValue(route);
      const bool valid = ValidResidentSlot(expert, experts_);
      if (valid && column < k_) {
        // Never truncate/index an unchecked signed int64 route ID.
        const uint32_t entry = static_cast<uint32_t>(expert) * kBankWords;
        GlobalTensor<int64_t> order;
        order.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(bank_.GetValue(entry + kBankOrder)));
        const uint64_t rowBase = (uint64_t(route) * m_ + row) * k_;
        auto gathered = gatherUb_.Get<uint8_t>();
        for (uint32_t i = 0; i < kSelectColumns; ++i) {
          // Constructor verified a complete 0..K-1 permutation before upload.
          gathered.SetValue(i, x_.GetValue(rowBase + static_cast<uint64_t>(order.GetValue(column + i))));
        }
        PrepareFence<HardEvent::S_MTE3>();
        DataCopy(reordered_[rowBase + column], gathered, kSelectColumns);
        PrepareFence<HardEvent::MTE3_S>();
      } else if (!valid && column < n_) {
        Duplicate(outputUb_.Get<uint16_t>(), kInvalidBf16, kSelectColumns);
        PrepareFence<HardEvent::V_MTE3>();
        DataCopy(output_[(uint64_t(route) * m_ + row) * n_ + column], outputUb_.Get<uint16_t>(), kSelectColumns);
        PrepareFence<HardEvent::MTE3_S>();
      }
      if (row == 0 && column == 0) WriteDescriptor(route, expert, valid);
    }
  }

 private:
  __aicore__ inline void WriteDescriptor(uint32_t route, int64_t expert, bool valid) {
    auto record = recordUb_.Get<uint64_t>();
    for (uint32_t field = 0; field < kJobWords; ++field) record.SetValue(field, uint64_t(0));
    if (valid) {
      const uint32_t entry = static_cast<uint32_t>(expert) * kBankWords;
      record.SetValue(kX, xAddr_ + uint64_t(route) * m_ * k_);
      record.SetValue(kScale, scaleAddr_ + uint64_t(route) * m_ * sizeof(float));
      record.SetValue(kBias, biasAddr_ + uint64_t(route) * m_ * sizeof(float));
      record.SetValue(kPacked, bank_.GetValue(entry + kBankPacked));
      record.SetValue(kTable, bank_.GetValue(entry + kBankBook));
      record.SetValue(kOutput, outputAddr_ + uint64_t(route) * m_ * n_ * sizeof(uint16_t));
      record.SetValue(kRows, uint64_t(m_));
      record.SetValue(kColumns, uint64_t(n_));
      record.SetValue(kReduction, uint64_t(k_));
    }
    // Invalid routes carry m=0. Both AIC and both AIV peers skip that work
    // before loading any indirect pointer or emitting cross-core flags.
    statusUb_.Get<int32_t>().SetValue(0, valid ? 1 : 0);
    PrepareFence<HardEvent::S_MTE3>();
    DataCopyExtParams descriptorCopy{1, kJobWords * sizeof(uint64_t), 0, 0, 0};
    DataCopyPad(descriptors_[uint64_t(route) * kJobWords], record, descriptorCopy);
    DataCopyExtParams statusCopy{1, sizeof(int32_t), 0, 0, 0};
    DataCopyPad(valid_[route], statusUb_.Get<int32_t>(), statusCopy);
    PrepareFence<HardEvent::MTE3_S>();
  }

  TPipe pipe_;
  TBuf<TPosition::VECCALC> gatherUb_, outputUb_, recordUb_, statusUb_;
  GlobalTensor<uint64_t> bank_, descriptors_;
  GlobalTensor<int64_t> ids_;
  GlobalTensor<uint8_t> x_, reordered_;
  GlobalTensor<uint16_t> output_;
  GlobalTensor<int32_t> valid_;
  uint64_t xAddr_, scaleAddr_, biasAddr_, outputAddr_;
  uint32_t experts_, routes_, m_, n_, k_;
};
}  // namespace vq2a8_ascendc_v4_v2

extern "C" __global__ __aicore__ void vq2a8_ascendc_v4_v2_prepare(
    GM_ADDR bank, GM_ADDR routeIds, GM_ADDR x, GM_ADDR scale, GM_ADDR bias, GM_ADDR reordered,
    GM_ADDR descriptors, GM_ADDR output, GM_ADDR valid, uint32_t experts, uint32_t routes,
    uint32_t m, uint32_t n, uint32_t k) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::ResidentPrepareKernel kernel;
  kernel.Init(bank, routeIds, x, scale, bias, reordered, descriptors, output, valid, experts, routes, m, n, k);
  kernel.Process();
}

namespace vq2a8_ascendc_v4_v2 {
void LaunchResidentPrepare(void* stream, uint32_t blocks, void* bank, void* routeIds, void* x, void* scale,
                           void* bias, void* reordered, void* descriptors, void* output, void* valid,
                           uint32_t experts, uint32_t routes, uint32_t m, uint32_t n, uint32_t k) {
  vq2a8_ascendc_v4_v2_prepare<<<blocks, nullptr, stream>>>(
      static_cast<GM_ADDR>(bank), static_cast<GM_ADDR>(routeIds), static_cast<GM_ADDR>(x),
      static_cast<GM_ADDR>(scale), static_cast<GM_ADDR>(bias), static_cast<GM_ADDR>(reordered),
      static_cast<GM_ADDR>(descriptors), static_cast<GM_ADDR>(output), static_cast<GM_ADDR>(valid),
      experts, routes, m, n, k);
}
}  // namespace vq2a8_ascendc_v4_v2
