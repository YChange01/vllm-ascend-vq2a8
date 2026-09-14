// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "kernel_operator.h"
#include "kernel_utils.h"
#define VQ2A8_V3_PREPARE_FN __aicore__ inline
#include "prepare_launch.h"

// This fuses the preparation AFTER the original dense RHT and bias GEMV.
// Neither reduction is rewritten here. FP8 conversion uses the existing A5
// recipe in attention/mla_prolog_v3/op_kernel/arch35/vf/vf_dynamic_quant.h:
// RegLayout::ZERO, SAT, CAST_RINT, and DIST_PACK4_B32. Ordinary vector APIs
// implement scaling, maximum reduction, validation and byte Gather.
namespace vq2a8_v3 {
using namespace AscendC;

template <HardEvent E>
__aicore__ inline void PrepareFence() {
  event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
  SetFlag<E>(event);
  WaitFlag<E>(event);
}

__simd_vf__ void PrepareCastFp8(__ubuf__ float* input, __ubuf__ fp8_e4m3fn_t* output, uint32_t count) {
  static constexpr MicroAPI::CastTrait kCast = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
  constexpr uint16_t kLanes = VECTOR_REG_WIDTH / sizeof(float);
  MicroAPI::RegTensor<float> source;
  MicroAPI::RegTensor<fp8_e4m3fn_t> target;
  uint32_t remaining = count;
  for (uint16_t repeat = 0; repeat < count / kLanes; ++repeat) {
    auto mask = MicroAPI::UpdateMask<float>(remaining);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(source, input + repeat * kLanes);
    MicroAPI::Cast<fp8_e4m3fn_t, float, kCast>(target, source, mask);
    MicroAPI::StoreAlign<fp8_e4m3fn_t, MicroAPI::StoreDist::DIST_PACK4_B32>(output + repeat * kLanes, target, mask);
  }
}

class PrepareKernel {
 public:
  __aicore__ inline void Init(GM_ADDR rotated, GM_ADDR weightScale, GM_ADDR order, GM_ADDR inputBias, GM_ADDR quantized,
                              GM_ADDR rowScale, GM_ADDR outputBias, GM_ADDR valid, uint32_t jobs, uint32_t k) {
    jobs_ = jobs;
    k_ = k;
    rotated_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rotated));
    weightScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weightScale));
    order_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(order));
    inputBias_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputBias));
    quantized_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(quantized));
    rowScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rowScale));
    outputBias_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outputBias));
    valid_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valid));
    pipe_.InitBuffer(input_, kPrepareTile * sizeof(float));
    pipe_.InitBuffer(weight_, kPrepareTile * sizeof(float));
    pipe_.InitBuffer(values_, kPrepareTile * sizeof(float));
    pipe_.InitBuffer(temporary_, kPrepareTile * sizeof(float));
    pipe_.InitBuffer(maximum_, kPrepareTile * sizeof(float));
    pipe_.InitBuffer(exponents_, kPrepareTile * sizeof(float));
    pipe_.InitBuffer(scratch_, kPrepareTile * sizeof(float));
    pipe_.InitBuffer(reduction_, kPrepareBlockBytes);
    pipe_.InitBuffer(scalars_, 3 * kPrepareBlockBytes);
    pipe_.InitBuffer(fp8_, kPrepareMaxK);
    pipe_.InitBuffer(orderUb_, kPrepareTile * sizeof(int64_t));
    pipe_.InitBuffer(lowOffsets_, kPrepareTile * sizeof(uint32_t));
    pipe_.InitBuffer(highOffsets_, kPrepareTile * sizeof(uint32_t));
    pipe_.InitBuffer(output_, kPrepareTile);
    // Byte offsets extract both halves of int64 order entries. Inspecting only
    // the low word would incorrectly accept e.g. 2**32 + valid_column.
    ArithProgression(lowOffsets_.Get<int32_t>(), int32_t(0), int32_t(sizeof(int64_t)), kPrepareTile);
    ArithProgression(highOffsets_.Get<int32_t>(), int32_t(sizeof(int32_t)), int32_t(sizeof(int64_t)), kPrepareTile);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Process() {
    for (uint32_t job = GetBlockIdx(); job < jobs_; job += GetBlockNum()) {
      PrepareRow(job);
    }
  }

 private:
  __aicore__ inline void LoadProduct(uint64_t offset, uint32_t count) {
    PrepareFence<HardEvent::V_MTE2>();
    DataCopy(input_.Get<float>(), rotated_[offset], count);
    DataCopy(weight_.Get<float>(), weightScale_[offset], count);
    PrepareFence<HardEvent::MTE2_V>();
    Mul(values_.Get<float>(), input_.Get<float>(), weight_.Get<float>(), count);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void AccumulateExponents(LocalTensor<float> source, uint32_t count) {
    // Masking away mantissas maps BOTH infinity and NaN to +infinity. This
    // detects NaNs even on a ReduceMax implementation that ignores NaN lanes.
    Ands(temporary_.Get<uint64_t>(), source.ReinterpretCast<uint64_t>(), kPrepareExponentPairMask, count / 2);
    PipeBarrier<PIPE_V>();
    Max(exponents_.Get<float>(), exponents_.Get<float>(), temporary_.Get<float>(), count);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline float Maximum(LocalTensor<float> source, uint32_t count) {
    ReduceMax(reduction_.Get<float>(), source, scratch_.Get<float>(), count, false);
    PrepareFence<HardEvent::V_S>();
    return reduction_.Get<float>().GetValue(0);
  }

  __aicore__ inline bool GatherRow(uint32_t job) {
    bool valid = true;
    uint64_t row = uint64_t(job) * k_;
    for (uint32_t start = 0; start < k_; start += kPrepareTile) {
      uint32_t count = PrepareTileCount(k_, start);
      PrepareFence<HardEvent::V_MTE2>();
      DataCopy(orderUb_.Get<int64_t>(), order_[row + start], count);
      PrepareFence<HardEvent::MTE2_V>();
      Gather(input_.Get<int32_t>(), orderUb_.Get<int32_t>(), lowOffsets_.Get<uint32_t>(), uint32_t(0), count);
      Gather(weight_.Get<int32_t>(), orderUb_.Get<int32_t>(), highOffsets_.Get<uint32_t>(), uint32_t(0), count);
      PipeBarrier<PIPE_V>();
      Cast(temporary_.Get<float>(), weight_.Get<int32_t>(), RoundMode::CAST_ROUND, count);
      Cast(values_.Get<float>(), input_.Get<int32_t>(), RoundMode::CAST_ROUND, count);
      PipeBarrier<PIPE_V>();
      Abs(temporary_.Get<float>(), temporary_.Get<float>(), count);
      PipeBarrier<PIPE_V>();
      valid = (Maximum(temporary_.Get<float>(), count) == 0.0f) && valid;
      // Valid indices <= 65535 are represented exactly in FP32. Out-of-range
      // low words are clamped BEFORE Gather, so malformed metadata cannot read
      // outside the FP8 UB row, including negative and overflowing int64s.
      Maxs(input_.Get<float>(), values_.Get<float>(), 0.0f, count);
      PipeBarrier<PIPE_V>();
      Mins(input_.Get<float>(), input_.Get<float>(), float(k_ - 1), count);
      PipeBarrier<PIPE_V>();
      Sub(temporary_.Get<float>(), values_.Get<float>(), input_.Get<float>(), count);
      PipeBarrier<PIPE_V>();
      Abs(temporary_.Get<float>(), temporary_.Get<float>(), count);
      PipeBarrier<PIPE_V>();
      valid = (Maximum(temporary_.Get<float>(), count) == 0.0f) && valid;
      Cast(values_.Get<int32_t>(), input_.Get<float>(), RoundMode::CAST_RINT, count);
      PipeBarrier<PIPE_V>();
      Gather(output_.Get<uint8_t>(), fp8_.Get<uint8_t>(), values_.Get<uint32_t>(), uint32_t(0), count);
      PrepareFence<HardEvent::V_MTE3>();
      DataCopy(quantized_[row + start], output_.Get<uint8_t>(), count);
      PrepareFence<HardEvent::MTE3_V>();
    }
    return valid;
  }

  __aicore__ inline void PrepareRow(uint32_t job) {
    constexpr uint32_t kScalarWordsPerBlock = kPrepareBlockBytes / sizeof(float);
    constexpr uint32_t kBiasWord = kScalarWordsPerBlock;
    constexpr uint32_t kValidWord = 2 * kScalarWordsPerBlock;
    DataCopyExtParams scalarCopy{1, sizeof(float), 0, 0, 0};
    DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
    DataCopyPad(scalars_.Get<float>()[kBiasWord], inputBias_[job], scalarCopy, noPad);
    PrepareFence<HardEvent::MTE2_S>();
    bool valid = PrepareFiniteBits(scalars_.Get<uint32_t>().GetValue(kBiasWord));
    Duplicate(maximum_.Get<float>(), 0.0f, kPrepareTile);
    Duplicate(exponents_.Get<float>(), 0.0f, kPrepareTile);
    PipeBarrier<PIPE_V>();
    uint64_t row = uint64_t(job) * k_;
    for (uint32_t start = 0; start < k_; start += kPrepareTile) {
      uint32_t count = PrepareTileCount(k_, start);
      LoadProduct(row + start, count);
      AccumulateExponents(input_.Get<float>(), count);
      AccumulateExponents(weight_.Get<float>(), count);
      AccumulateExponents(values_.Get<float>(), count);
      Abs(temporary_.Get<float>(), values_.Get<float>(), count);
      PipeBarrier<PIPE_V>();
      Max(maximum_.Get<float>(), maximum_.Get<float>(), temporary_.Get<float>(), count);
      PipeBarrier<PIPE_V>();
    }
    Maximum(exponents_.Get<float>(), kPrepareTile);
    valid = PrepareFiniteBits(reduction_.Get<uint32_t>().GetValue(0)) && valid;
    float scale = kPrepareMinScale;
    if (valid) {
      scale = Maximum(maximum_.Get<float>(), kPrepareTile) / kPrepareFp8Max;
      scale = scale < kPrepareMinScale ? kPrepareMinScale : scale;
      for (uint32_t start = 0; start < k_; start += kPrepareTile) {
        uint32_t count = PrepareTileCount(k_, start);
        LoadProduct(row + start, count);
        Duplicate(weight_.Get<float>(), scale, count);
        PipeBarrier<PIPE_V>();
        Div(values_.Get<float>(), values_.Get<float>(), weight_.Get<float>(), count);
        PipeBarrier<PIPE_V>();
        Maxs(values_.Get<float>(), values_.Get<float>(), -kPrepareFp8Max, count);
        PipeBarrier<PIPE_V>();
        Mins(values_.Get<float>(), values_.Get<float>(), kPrepareFp8Max, count);
        PipeBarrier<PIPE_V>();
        PrepareCastFp8(reinterpret_cast<__ubuf__ float*>(values_.Get<float>().GetPhyAddr()),
                       reinterpret_cast<__ubuf__ fp8_e4m3fn_t*>(fp8_.Get<uint8_t>()[start].GetPhyAddr()), count);
        PipeBarrier<PIPE_V>();
      }
    } else {
      // Deterministic bytes for an invalid numerical row. Runtime consumes
      // the device status; it must not treat these zeros as a valid result.
      Duplicate(fp8_.Get<uint32_t>(), uint32_t(0), k_ / sizeof(uint32_t));
      PipeBarrier<PIPE_V>();
    }
    valid = GatherRow(job) && valid;
    scalars_.Get<float>().SetValue(0, scale);
    scalars_.Get<int32_t>().SetValue(kValidWord, valid ? 1 : 0);
    PrepareFence<HardEvent::S_MTE3>();
    DataCopyPad(rowScale_[job], scalars_.Get<float>(), scalarCopy);
    // Raw FP32 bias is forwarded without a new reduction or BF16 conversion.
    DataCopyPad(outputBias_[job], scalars_.Get<float>()[kBiasWord], scalarCopy);
    DataCopyPad(valid_[job], scalars_.Get<int32_t>()[kValidWord], scalarCopy);
    PrepareFence<HardEvent::MTE3_S>();
    PrepareFence<HardEvent::MTE3_V>();
  }

  TPipe pipe_;
  TBuf<TPosition::VECCALC> input_, weight_, values_, temporary_, maximum_, exponents_, scratch_;
  TBuf<TPosition::VECCALC> reduction_, scalars_, fp8_, orderUb_, lowOffsets_, highOffsets_, output_;
  GlobalTensor<float> rotated_, weightScale_, inputBias_, rowScale_, outputBias_;
  GlobalTensor<int64_t> order_;
  GlobalTensor<uint8_t> quantized_;
  GlobalTensor<int32_t> valid_;
  uint32_t jobs_, k_;
};
}  // namespace vq2a8_v3

extern "C" __global__ __aicore__ void vq2a8_v3_prepare(GM_ADDR rotated, GM_ADDR weightScale, GM_ADDR order,
                                                       GM_ADDR inputBias, GM_ADDR quantized, GM_ADDR rowScale,
                                                       GM_ADDR outputBias, GM_ADDR valid, uint32_t jobs, uint32_t k) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_v3::PrepareKernel op;
  op.Init(rotated, weightScale, order, inputBias, quantized, rowScale, outputBias, valid, jobs, k);
  op.Process();
}

namespace vq2a8_v3 {
void LaunchPrepareV3(void* stream, uint32_t blocks, void* rotated, void* weightScale, void* order, void* inputBias,
                     void* quantized, void* rowScale, void* outputBias, void* valid, uint32_t jobs, uint32_t k) {
  vq2a8_v3_prepare<<<blocks, nullptr, stream>>>(static_cast<GM_ADDR>(rotated), static_cast<GM_ADDR>(weightScale),
                                                static_cast<GM_ADDR>(order), static_cast<GM_ADDR>(inputBias),
                                                static_cast<GM_ADDR>(quantized), static_cast<GM_ADDR>(rowScale),
                                                static_cast<GM_ADDR>(outputBias), static_cast<GM_ADDR>(valid), jobs, k);
}
}  // namespace vq2a8_v3
