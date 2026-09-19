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

// Baseline activation bytes are reordered AFTER original RHT/FP8 preparation.
// The opt-in normalized tail starts AFTER the original FP32 division instead.
// Compressed expert weights never move; all paths use separate entry points.
template <bool Vectorized = false, bool NormalizedTail = false, bool RowReuse = false,
          uint32_t ChunkReuse = 0>
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
    if constexpr (NormalizedTail) normalized_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    reordered_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(reordered));
    descriptors_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(descriptors));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(output));
    valid_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valid));
    pipe_.InitBuffer(gatherUb_, kSelectColumns);
    pipe_.InitBuffer(outputUb_, kSelectColumns * sizeof(uint16_t));
    pipe_.InitBuffer(recordUb_, 96);  // Nine uint64 words, rounded to a DMA block.
    pipe_.InitBuffer(statusUb_, 32);
    if constexpr (Vectorized) {
      if constexpr (NormalizedTail) {
        pipe_.InitBuffer(inputUb_, k_ * sizeof(float));
        pipe_.InitBuffer(tailGatherUb_, kSelectColumns * sizeof(float));
        pipe_.InitBuffer(tailClampUb_, kSelectColumns * sizeof(float));
        pipe_.InitBuffer(tailMaskUb_, kSelectColumns / 8);
      } else {
        pipe_.InitBuffer(inputUb_, k_);
      }
      pipe_.InitBuffer(orderUb_, kSelectColumns * sizeof(int64_t));
      pipe_.InitBuffer(orderOffsetsUb_, kSelectColumns * sizeof(uint32_t));
      pipe_.InitBuffer(lowWordOffsetsUb_, kSelectColumns * sizeof(uint32_t));
      // Gather offsets are BYTE offsets. Orders are validated nonnegative and
      // less than K<=4096 at construction, so their low uint32 word is exact.
      // Load the existing int64 metadata; do not allocate a second bank copy.
      auto offsets = lowWordOffsetsUb_.Get<int32_t>();
      CreateVecIndex(offsets, int32_t(0), kSelectColumns);
      PipeBarrier<PIPE_V>();
      Muls(offsets, offsets, int32_t(sizeof(int64_t)), kSelectColumns);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void Process() {
    if constexpr (ChunkReuse != 0) {
      static_assert(Vectorized && !NormalizedTail && !RowReuse && (ChunkReuse == 2 || ChunkReuse == 4));
      ProcessChunkReuse();
      return;
    }
    if constexpr (RowReuse) {
      ProcessRowReuse();
      return;
    }
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
        if constexpr (NormalizedTail) {
          GatherNormalizedTail(order, rowBase, column);
        } else if constexpr (Vectorized) {
          GatherVectorized(order, rowBase, column);
        } else {
          auto gathered = gatherUb_.Get<uint8_t>();
          for (uint32_t i = 0; i < kSelectColumns; ++i) {
            // Constructor verified a complete 0..K-1 permutation before upload.
            gathered.SetValue(i, x_.GetValue(rowBase + static_cast<uint64_t>(order.GetValue(column + i))));
          }
          PrepareFence<HardEvent::S_MTE3>();
          DataCopy(reordered_[rowBase + column], gathered, kSelectColumns);
          PrepareFence<HardEvent::MTE3_S>();
        }
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
  // Candidate J: one work item owns only 2/4 adjacent 256-column chunks.
  // Reload the K-byte row once per chunk group, retaining parallel work across
  // route x chunk-group. Both supported widths and N4096 divide these groups.
  // Arithmetic and the descriptor/invalid-output contract remain unchanged.
  __aicore__ inline void ProcessChunkReuse() {
    const uint32_t chunkGroups = (n_ > k_ ? n_ : k_) / (kSelectColumns * ChunkReuse);
    for (uint32_t work = GetBlockIdx(); work < routes_ * chunkGroups; work += GetBlockNum()) {
      const uint32_t route = work / chunkGroups;
      const uint32_t firstColumn = (work % chunkGroups) * kSelectColumns * ChunkReuse;
      const int64_t expert = ids_.GetValue(route);
      const bool valid = ValidResidentSlot(expert, experts_);
      if (valid && firstColumn < k_) {
        const uint32_t entry = static_cast<uint32_t>(expert) * kBankWords;
        GlobalTensor<int64_t> order;
        order.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(bank_.GetValue(entry + kBankOrder)));
        const uint64_t rowBase = uint64_t(route) * k_;
        DataCopy(inputUb_.Get<uint8_t>(), x_[rowBase], k_);
        for (uint32_t chunk = 0; chunk < ChunkReuse; ++chunk) {
          GatherVectorized(order, rowBase, firstColumn + chunk * kSelectColumns);
        }
      } else if (!valid && firstColumn < n_) {
        Duplicate(outputUb_.Get<uint16_t>(), kInvalidBf16, kSelectColumns);
        PrepareFence<HardEvent::V_MTE3>();
        for (uint32_t chunk = 0; chunk < ChunkReuse; ++chunk) {
          DataCopy(output_[uint64_t(route) * n_ + firstColumn + chunk * kSelectColumns],
                   outputUb_.Get<uint16_t>(), kSelectColumns);
        }
        PrepareFence<HardEvent::MTE3_V>();
        PrepareFence<HardEvent::MTE3_S>();
      }
      if (firstColumn == 0) WriteDescriptor(route, expert, valid);
    }
  }

  // Candidate F is a pure FP8-byte permutation, M=1 only at the binding.
  // Assign a full route row to one AIV, retaining its K-byte input in UB
  // across all 256-column chunks. This removes the baseline's 8/16 repeated
  // row DMAs, but trades away chunk-level parallelism; benchmark separately.
  __aicore__ inline void ProcessRowReuse() {
    for (uint32_t route = GetBlockIdx(); route < routes_; route += GetBlockNum()) {
      const int64_t expert = ids_.GetValue(route);
      const bool valid = ValidResidentSlot(expert, experts_);
      if (valid) {
        // Validate the full signed int64 before narrowing or dereferencing.
        const uint32_t entry = static_cast<uint32_t>(expert) * kBankWords;
        GlobalTensor<int64_t> order;
        order.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(bank_.GetValue(entry + kBankOrder)));
        const uint64_t rowBase = uint64_t(route) * k_;
        DataCopy(inputUb_.Get<uint8_t>(), x_[rowBase], k_);
        // GatherVectorized fences MTE2->V after loading the first order chunk.
        for (uint32_t column = 0; column < k_; column += kSelectColumns) {
          GatherVectorized(order, rowBase, column);
        }
      } else {
        // Match the original contract: invalid reordered bytes are unspecified,
        // every invalid output is poisoned, and its descriptor is all zero.
        Duplicate(outputUb_.Get<uint16_t>(), kInvalidBf16, kSelectColumns);
        PrepareFence<HardEvent::V_MTE3>();
        for (uint32_t column = 0; column < n_; column += kSelectColumns) {
          DataCopy(output_[uint64_t(route) * n_ + column], outputUb_.Get<uint16_t>(), kSelectColumns);
        }
        PrepareFence<HardEvent::MTE3_V>();
        PrepareFence<HardEvent::MTE3_S>();
      }
      WriteDescriptor(route, expert, valid);
    }
  }

  // Candidate D starts AFTER the original Torch RealDiv. Reordering commutes
  // with elementwise clamp/cast, but not with RHT or scale reductions. Keep
  // those upstream computations unchanged. All indirect orders are validated
  // at bank construction; invalid slots never reach this method.
  __aicore__ inline void GatherNormalizedTail(const GlobalTensor<int64_t>& order,
                                             uint64_t rowBase, uint32_t column) {
    auto input = inputUb_.Get<float>();
    auto orderWords = orderUb_.Get<int64_t>();
    auto offsets = orderOffsetsUb_.Get<uint32_t>();
    auto gathered = tailGatherUb_.Get<float>();
    auto clamped = tailClampUb_.Get<float>();
    auto mask = tailMaskUb_.Get<uint8_t>();
    DataCopy(input, normalized_[rowBase], k_);
    DataCopy(orderWords, order[column], kSelectColumns);
    PrepareFence<HardEvent::MTE2_V>();
    Gather(offsets, orderWords.ReinterpretCast<uint32_t>(),
           lowWordOffsetsUb_.Get<uint32_t>(), uint32_t(0), kSelectColumns);
    PipeBarrier<PIPE_V>();
    // Gather uses BYTE offsets, unlike the int64 element-index permutation.
    Muls(orderOffsetsUb_.Get<int32_t>(), orderOffsetsUb_.Get<int32_t>(),
         int32_t(sizeof(float)), kSelectColumns);
    PipeBarrier<PIPE_V>();
    Gather(gathered, input, offsets, uint32_t(0), kSelectColumns);
    PipeBarrier<PIPE_V>();
    // Explicitly restore NaNs: hardware min/max NaN selection must not turn
    // an invalid activation into a finite value. Infinities clamp normally.
    Compare(mask, gathered, gathered, CMPMODE::EQ, kSelectColumns);
    PipeBarrier<PIPE_V>();
    constexpr float kFp8FiniteMaximum = 448.0f;
    Mins(clamped, gathered, kFp8FiniteMaximum, kSelectColumns);
    PipeBarrier<PIPE_V>();
    Maxs(clamped, clamped, -kFp8FiniteMaximum, kSelectColumns);
    PipeBarrier<PIPE_V>();
    Select(clamped, mask, clamped, gathered, SELMODE::VSEL_TENSOR_TENSOR_MODE, kSelectColumns);
    PipeBarrier<PIPE_V>();
    Cast(gatherUb_.Get<fp8_e4m3fn_t>(), clamped, RoundMode::CAST_RINT, kSelectColumns);
    PrepareFence<HardEvent::V_MTE3>();
    DataCopy(reordered_[rowBase + column], gatherUb_.Get<uint8_t>(), kSelectColumns);
    PrepareFence<HardEvent::MTE3_V>();
    PrepareFence<HardEvent::V_MTE2>();
  }

  __aicore__ inline void GatherVectorized(const GlobalTensor<int64_t>& order,
                                        uint64_t rowBase, uint32_t column) {
    auto input = inputUb_.Get<uint8_t>();
    auto orderWords = orderUb_.Get<int64_t>();
    auto orderOffsets = orderOffsetsUb_.Get<uint32_t>();
    auto gathered = gatherUb_.Get<uint8_t>();
    // One contiguous DMA per activation row and per 256-entry order segment,
    // replacing 512 scalar GM loads. K is small enough to keep the entire
    // activation row in UB (maximum 4096 bytes); no decoded weights in GM.
    if constexpr (ChunkReuse == 0) {
      if constexpr (!RowReuse) DataCopy(input, x_[rowBase], k_);
    }
    DataCopy(orderWords, order[column], kSelectColumns);
    PrepareFence<HardEvent::MTE2_V>();
    Gather(orderOffsets, orderWords.ReinterpretCast<uint32_t>(),
           lowWordOffsetsUb_.Get<uint32_t>(), uint32_t(0), kSelectColumns);
    PipeBarrier<PIPE_V>();
    // Ascend950/dav3510 supports the count-based Gather<uint8_t> overload:
    // asc-devkit include/basic_api/kernel_operator_vec_gather_intf.h and
    // impl/basic_api/dav_3510/kernel_operator_vec_gather_impl.h,
    // GatherApi2B8Impl/VfGatherApi2B8. Do not use the B16/B32-only level-0 API.
    // This is a byte copy, NOT a cast: FP8 signed zero and NaN bits survive.
    Gather(gathered, input, orderOffsets, uint32_t(0), kSelectColumns);
    PrepareFence<HardEvent::V_MTE3>();
    DataCopy(reordered_[rowBase + column], gathered, kSelectColumns);
    // Protect both the output buffer and input/order buffers on reuse.
    PrepareFence<HardEvent::MTE3_V>();
    PrepareFence<HardEvent::V_MTE2>();
  }

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
  TBuf<TPosition::VECCALC> inputUb_, orderUb_, orderOffsetsUb_, lowWordOffsetsUb_;
  TBuf<TPosition::VECCALC> tailGatherUb_, tailClampUb_, tailMaskUb_;
  GlobalTensor<float> normalized_;
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
  vq2a8_ascendc_v4_v2::ResidentPrepareKernel<false> kernel;
  kernel.Init(bank, routeIds, x, scale, bias, reordered, descriptors, output, valid, experts, routes, m, n, k);
  kernel.Process();
}

extern "C" __global__ __aicore__ void vq2a8_ascendc_v4_v2_prepare_vectorized(
    GM_ADDR bank, GM_ADDR routeIds, GM_ADDR x, GM_ADDR scale, GM_ADDR bias, GM_ADDR reordered,
    GM_ADDR descriptors, GM_ADDR output, GM_ADDR valid, uint32_t experts, uint32_t routes,
    uint32_t m, uint32_t n, uint32_t k) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::ResidentPrepareKernel<true> kernel;
  kernel.Init(bank, routeIds, x, scale, bias, reordered, descriptors, output, valid, experts, routes, m, n, k);
  kernel.Process();
}

extern "C" __global__ __aicore__ void vq2a8_ascendc_v4_v2_prepare_tail(
    GM_ADDR bank, GM_ADDR routeIds, GM_ADDR x, GM_ADDR scale, GM_ADDR bias, GM_ADDR reordered,
    GM_ADDR descriptors, GM_ADDR output, GM_ADDR valid, uint32_t experts, uint32_t routes,
    uint32_t m, uint32_t n, uint32_t k) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::ResidentPrepareKernel<true, true> kernel;
  kernel.Init(bank, routeIds, x, scale, bias, reordered, descriptors, output, valid, experts, routes, m, n, k);
  kernel.Process();
}

extern "C" __global__ __aicore__ void vq2a8_ascendc_v4_v2_prepare_row_reuse(
    GM_ADDR bank, GM_ADDR routeIds, GM_ADDR x, GM_ADDR scale, GM_ADDR bias, GM_ADDR reordered,
    GM_ADDR descriptors, GM_ADDR output, GM_ADDR valid, uint32_t experts, uint32_t routes,
    uint32_t m, uint32_t n, uint32_t k) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::ResidentPrepareKernel<true, false, true> kernel;
  kernel.Init(bank, routeIds, x, scale, bias, reordered, descriptors, output, valid, experts, routes, m, n, k);
  kernel.Process();
}

extern "C" __global__ __aicore__ void vq2a8_ascendc_v4_v2_prepare_chunk_reuse2(
    GM_ADDR bank, GM_ADDR routeIds, GM_ADDR x, GM_ADDR scale, GM_ADDR bias, GM_ADDR reordered,
    GM_ADDR descriptors, GM_ADDR output, GM_ADDR valid, uint32_t experts, uint32_t routes,
    uint32_t m, uint32_t n, uint32_t k) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::ResidentPrepareKernel<true, false, false, 2> kernel;
  kernel.Init(bank, routeIds, x, scale, bias, reordered, descriptors, output, valid, experts, routes, m, n, k);
  kernel.Process();
}

extern "C" __global__ __aicore__ void vq2a8_ascendc_v4_v2_prepare_chunk_reuse4(
    GM_ADDR bank, GM_ADDR routeIds, GM_ADDR x, GM_ADDR scale, GM_ADDR bias, GM_ADDR reordered,
    GM_ADDR descriptors, GM_ADDR output, GM_ADDR valid, uint32_t experts, uint32_t routes,
    uint32_t m, uint32_t n, uint32_t k) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::ResidentPrepareKernel<true, false, false, 4> kernel;
  kernel.Init(bank, routeIds, x, scale, bias, reordered, descriptors, output, valid, experts, routes, m, n, k);
  kernel.Process();
}

namespace vq2a8_ascendc_v4_v2 {
void LaunchResidentPrepareChunkReuse(void* stream, uint32_t blocks, void* bank, void* routeIds, void* x,
                                   void* scale, void* bias, void* reordered, void* descriptors,
                                   void* output, void* valid, uint32_t experts, uint32_t routes,
                                   uint32_t m, uint32_t n, uint32_t k, uint32_t chunks) {
  // Host binding rejects every other value before enqueue.
  if (chunks == 2) {
    vq2a8_ascendc_v4_v2_prepare_chunk_reuse2<<<blocks, nullptr, stream>>>(
        static_cast<GM_ADDR>(bank), static_cast<GM_ADDR>(routeIds), static_cast<GM_ADDR>(x),
        static_cast<GM_ADDR>(scale), static_cast<GM_ADDR>(bias), static_cast<GM_ADDR>(reordered),
        static_cast<GM_ADDR>(descriptors), static_cast<GM_ADDR>(output), static_cast<GM_ADDR>(valid),
        experts, routes, m, n, k);
  } else {
    vq2a8_ascendc_v4_v2_prepare_chunk_reuse4<<<blocks, nullptr, stream>>>(
        static_cast<GM_ADDR>(bank), static_cast<GM_ADDR>(routeIds), static_cast<GM_ADDR>(x),
        static_cast<GM_ADDR>(scale), static_cast<GM_ADDR>(bias), static_cast<GM_ADDR>(reordered),
        static_cast<GM_ADDR>(descriptors), static_cast<GM_ADDR>(output), static_cast<GM_ADDR>(valid),
        experts, routes, m, n, k);
  }
}

void LaunchResidentPrepareRowReuse(void* stream, uint32_t blocks, void* bank, void* routeIds, void* x,
                                 void* scale, void* bias, void* reordered, void* descriptors,
                                 void* output, void* valid, uint32_t experts, uint32_t routes,
                                 uint32_t m, uint32_t n, uint32_t k) {
  vq2a8_ascendc_v4_v2_prepare_row_reuse<<<blocks, nullptr, stream>>>(
      static_cast<GM_ADDR>(bank), static_cast<GM_ADDR>(routeIds), static_cast<GM_ADDR>(x),
      static_cast<GM_ADDR>(scale), static_cast<GM_ADDR>(bias), static_cast<GM_ADDR>(reordered),
      static_cast<GM_ADDR>(descriptors), static_cast<GM_ADDR>(output), static_cast<GM_ADDR>(valid),
      experts, routes, m, n, k);
}

void LaunchResidentPrepareTail(void* stream, uint32_t blocks, void* bank, void* routeIds, void* x,
                              void* scale, void* bias, void* reordered, void* descriptors,
                              void* output, void* valid, uint32_t experts, uint32_t routes,
                              uint32_t m, uint32_t n, uint32_t k) {
  vq2a8_ascendc_v4_v2_prepare_tail<<<blocks, nullptr, stream>>>(
      static_cast<GM_ADDR>(bank), static_cast<GM_ADDR>(routeIds), static_cast<GM_ADDR>(x),
      static_cast<GM_ADDR>(scale), static_cast<GM_ADDR>(bias), static_cast<GM_ADDR>(reordered),
      static_cast<GM_ADDR>(descriptors), static_cast<GM_ADDR>(output), static_cast<GM_ADDR>(valid),
      experts, routes, m, n, k);
}

void LaunchResidentPrepare(void* stream, uint32_t blocks, void* bank, void* routeIds, void* x, void* scale,
                           void* bias, void* reordered, void* descriptors, void* output, void* valid,
                           uint32_t experts, uint32_t routes, uint32_t m, uint32_t n, uint32_t k) {
  vq2a8_ascendc_v4_v2_prepare<<<blocks, nullptr, stream>>>(
      static_cast<GM_ADDR>(bank), static_cast<GM_ADDR>(routeIds), static_cast<GM_ADDR>(x),
      static_cast<GM_ADDR>(scale), static_cast<GM_ADDR>(bias), static_cast<GM_ADDR>(reordered),
      static_cast<GM_ADDR>(descriptors), static_cast<GM_ADDR>(output), static_cast<GM_ADDR>(valid),
      experts, routes, m, n, k);
}

void LaunchResidentPrepareVectorized(void* stream, uint32_t blocks, void* bank, void* routeIds, void* x,
                                    void* scale, void* bias, void* reordered, void* descriptors,
                                    void* output, void* valid, uint32_t experts, uint32_t routes,
                                    uint32_t m, uint32_t n, uint32_t k) {
  vq2a8_ascendc_v4_v2_prepare_vectorized<<<blocks, nullptr, stream>>>(
      static_cast<GM_ADDR>(bank), static_cast<GM_ADDR>(routeIds), static_cast<GM_ADDR>(x),
      static_cast<GM_ADDR>(scale), static_cast<GM_ADDR>(bias), static_cast<GM_ADDR>(reordered),
      static_cast<GM_ADDR>(descriptors), static_cast<GM_ADDR>(output), static_cast<GM_ADDR>(valid),
      experts, routes, m, n, k);
}
}  // namespace vq2a8_ascendc_v4_v2
