// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "kernel_operator.h"
#include "activation_launch.h"

namespace vq2a8_ascendc_v4_v2 {
using namespace AscendC;
constexpr float kFp8Maximum = 448.0f;
constexpr float kMinimumScale = 1.0e-12f;
constexpr float kFiniteMaximum = 3.4028234663852886e+38f;
constexpr uint32_t kScalarBytes = 32;
constexpr uint32_t kMaskBitsPerWord = 32;

template <HardEvent Event>
__aicore__ inline void ActivationFence() {
  const event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
  SetFlag<Event>(event);
  WaitFlag<Event>(event);
}

// Compare writes packed bits, so all input checks need only K/32 scalar UB
// reads, not K scalar GM loads. NaN comparisons are false; +/-Inf exceed max.
__aicore__ inline void FiniteMask(const LocalTensor<float>& x, const LocalTensor<float>& scratch,
                                const LocalTensor<uint8_t>& mask, uint32_t width) {
  Abs(scratch, x, width);
  PipeBarrier<PIPE_V>();
  Compares(mask, scratch, kFiniteMaximum, CMPMODE::LE, width);
  PipeBarrier<PIPE_V>();
}

__aicore__ inline bool AllMask(const LocalTensor<uint32_t>& mask, uint32_t width) {
  ActivationFence<HardEvent::V_S>();
  uint32_t result = 0xffffffffU;
  for (uint32_t i = 0; i < width / kMaskBitsPerWord; ++i) result &= mask.GetValue(i);
  return result == 0xffffffffU;
}

class ActivationSignKernel {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR bias, GM_ADDR signs, GM_ADDR output,
                            GM_ADDR valid, uint32_t rows, uint32_t width) {
    rows_ = rows;
    width_ = width;
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    scale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
    bias_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
    signs_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(signs));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    valid_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valid));
    pipe_.InitBuffer(xUb_, width * sizeof(float));
    pipe_.InitBuffer(scaleUb_, width * sizeof(float));
    pipe_.InitBuffer(biasUb_, width * sizeof(float));
    pipe_.InitBuffer(signUb_, width);
    pipe_.InitBuffer(halfUb_, width * sizeof(half));
    pipe_.InitBuffer(signFloatUb_, width * sizeof(float));
    pipe_.InitBuffer(scratchUb_, width * sizeof(float));
    pipe_.InitBuffer(maskUb_, width / 8);
    pipe_.InitBuffer(otherMaskUb_, width / 8);
    pipe_.InitBuffer(statusUb_, kScalarBytes);
  }

  __aicore__ inline void Process() {
    const auto x = xUb_.Get<float>();
    const auto scale = scaleUb_.Get<float>();
    const auto bias = biasUb_.Get<float>();
    const auto sign = signUb_.Get<int8_t>();
    const auto signFloat = signFloatUb_.Get<float>();
    const auto scratch = scratchUb_.Get<float>();
    const auto mask = maskUb_.Get<uint8_t>();
    const auto other = otherMaskUb_.Get<uint8_t>();
    for (uint32_t row = GetBlockIdx(); row < rows_; row += GetBlockNum()) {
      const uint64_t base = uint64_t(row) * width_;
      DataCopy(x, x_[base], width_);
      DataCopy(scale, scale_[base], width_);
      DataCopy(bias, bias_[base], width_);
      DataCopy(sign, signs_[base], width_);
      ActivationFence<HardEvent::MTE2_V>();
      FiniteMask(x, scratch, mask, width_);
      FiniteMask(scale, scratch, other, width_);
      And(maskUb_.Get<uint16_t>(), maskUb_.Get<uint16_t>(), otherMaskUb_.Get<uint16_t>(), width_ / 16);
      PipeBarrier<PIPE_V>();
      FiniteMask(bias, scratch, other, width_);
      And(maskUb_.Get<uint16_t>(), maskUb_.Get<uint16_t>(), otherMaskUb_.Get<uint16_t>(), width_ / 16);
      PipeBarrier<PIPE_V>();
      // int8 -> half -> float is exact for every int8 value, including -128.
      Cast(halfUb_.Get<half>(), sign, RoundMode::CAST_NONE, width_);
      PipeBarrier<PIPE_V>();
      Cast(signFloat, halfUb_.Get<half>(), RoundMode::CAST_NONE, width_);
      PipeBarrier<PIPE_V>();
      Abs(scratch, signFloat, width_);
      PipeBarrier<PIPE_V>();
      Compares(other, scratch, 1.0f, CMPMODE::EQ, width_);
      PipeBarrier<PIPE_V>();
      And(maskUb_.Get<uint16_t>(), maskUb_.Get<uint16_t>(), otherMaskUb_.Get<uint16_t>(), width_ / 16);
      PipeBarrier<PIPE_V>();
      const bool valid = AllMask(maskUb_.Get<uint32_t>(), width_);
      Mul(x, x, signFloat, width_);
      ActivationFence<HardEvent::V_MTE3>();
      DataCopy(output_[base], x, width_);
      statusUb_.Get<int32_t>().SetValue(0, valid ? 1 : 0);
      ActivationFence<HardEvent::S_MTE3>();
      const DataCopyExtParams scalar{1, sizeof(int32_t), 0, 0, 0};
      DataCopyPad(valid_[row], statusUb_.Get<int32_t>(), scalar);
      ActivationFence<HardEvent::MTE3_S>();
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> xUb_, scaleUb_, biasUb_, signUb_, halfUb_, signFloatUb_, scratchUb_;
  TBuf<TPosition::VECCALC> maskUb_, otherMaskUb_, statusUb_;
  GlobalTensor<float> x_, scale_, bias_, output_;
  GlobalTensor<int8_t> signs_;
  GlobalTensor<int32_t> valid_;
  uint32_t rows_, width_;
};

class ActivationQuantizeKernel {
 public:
  __aicore__ inline void Init(GM_ADDR rotated, GM_ADDR weightScale, GM_ADDR rowBias, GM_ADDR quantized,
                            GM_ADDR rowScale, GM_ADDR valid, uint32_t rows, uint32_t width) {
    rows_ = rows;
    width_ = width;
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rotated));
    weightScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weightScale));
    rowBias_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rowBias));
    quantized_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(quantized));
    rowScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rowScale));
    valid_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valid));
    pipe_.InitBuffer(xUb_, width * sizeof(float));
    pipe_.InitBuffer(weightUb_, width * sizeof(float));
    pipe_.InitBuffer(absUb_, width * sizeof(float));
    pipe_.InitBuffer(scratchUb_, width * sizeof(float));
    pipe_.InitBuffer(divisorUb_, width * sizeof(float));
    pipe_.InitBuffer(quantUb_, width);
    pipe_.InitBuffer(maskUb_, width / 8);
    pipe_.InitBuffer(reduceUb_, kScalarBytes);
    pipe_.InitBuffer(statusUb_, kScalarBytes);
  }

  __aicore__ inline void Process() {
    const auto x = xUb_.Get<float>();
    const auto weight = weightUb_.Get<float>();
    const auto absolute = absUb_.Get<float>();
    const auto reduced = reduceUb_.Get<float>();
    const auto divisor = divisorUb_.Get<float>();
    for (uint32_t row = GetBlockIdx(); row < rows_; row += GetBlockNum()) {
      const uint64_t base = uint64_t(row) * width_;
      DataCopy(x, x_[base], width_);
      DataCopy(weight, weightScale_[base], width_);
      ActivationFence<HardEvent::MTE2_V>();
      Mul(x, x, weight, width_);
      PipeBarrier<PIPE_V>();
      FiniteMask(x, absolute, maskUb_.Get<uint8_t>(), width_);
      const float bias = rowBias_.GetValue(row);
      const bool valid = AllMask(maskUb_.Get<uint32_t>(), width_) &&
                         bias >= -kFiniteMaximum && bias <= kFiniteMaximum;
      if (valid) {
        ReduceMax(reduced, absolute, scratchUb_.Get<float>(), width_, false);
        PipeBarrier<PIPE_V>();
        // Use division (not a reciprocal/multiply approximation), and keep
        // the same min-scale/clamp/nearest-even FP8 contract as the reference.
        Divs(reduced, reduced, kFp8Maximum, 1);
        PipeBarrier<PIPE_V>();
        Maxs(reduced, reduced, kMinimumScale, 1);
        ActivationFence<HardEvent::V_S>();
        const float scale = reduced.GetValue(0);
        Duplicate(divisor, scale, width_);
        PipeBarrier<PIPE_V>();
        Div(x, x, divisor, width_);
        PipeBarrier<PIPE_V>();
        Mins(x, x, kFp8Maximum, width_);
        PipeBarrier<PIPE_V>();
        Maxs(x, x, -kFp8Maximum, width_);
        PipeBarrier<PIPE_V>();
        Cast(quantUb_.Get<fp8_e4m3fn_t>(), x, RoundMode::CAST_RINT, width_);
        ActivationFence<HardEvent::V_MTE3>();
      } else {
        // Deterministic poison, plus a device flag consumed before sampling.
        // No invalid FP32 values are fed into reduce/division/cast operations.
        Duplicate(quantUb_.Get<uint16_t>(), uint16_t(0x7f7f), width_ / 2);
        Duplicate(reduceUb_.Get<uint32_t>(), uint32_t(0x7fc00000), 8);
        ActivationFence<HardEvent::V_MTE3>();
      }
      DataCopy(quantized_[base], quantUb_.Get<uint8_t>(), width_);
      const DataCopyExtParams scalar{1, sizeof(float), 0, 0, 0};
      DataCopyPad(rowScale_[row], reduced, scalar);
      statusUb_.Get<int32_t>().SetValue(0, valid ? 1 : 0);
      ActivationFence<HardEvent::S_MTE3>();
      DataCopyPad(valid_[row], statusUb_.Get<int32_t>(), scalar);
      ActivationFence<HardEvent::MTE3_S>();
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> xUb_, weightUb_, absUb_, scratchUb_, divisorUb_, quantUb_, maskUb_, reduceUb_, statusUb_;
  GlobalTensor<float> x_, weightScale_, rowBias_, rowScale_;
  GlobalTensor<uint8_t> quantized_;
  GlobalTensor<int32_t> valid_;
  uint32_t rows_, width_;
};
}  // namespace vq2a8_ascendc_v4_v2

extern "C" __global__ __aicore__ void vq2a8_v4_v2_activation_sign(
    GM_ADDR x, GM_ADDR scale, GM_ADDR bias, GM_ADDR signs, GM_ADDR output, GM_ADDR valid,
    uint32_t rows, uint32_t width) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::ActivationSignKernel op;
  op.Init(x, scale, bias, signs, output, valid, rows, width);
  op.Process();
}

extern "C" __global__ __aicore__ void vq2a8_v4_v2_activation_quantize(
    GM_ADDR rotated, GM_ADDR weightScale, GM_ADDR rowBias, GM_ADDR quantized, GM_ADDR rowScale, GM_ADDR valid,
    uint32_t rows, uint32_t width) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::ActivationQuantizeKernel op;
  op.Init(rotated, weightScale, rowBias, quantized, rowScale, valid, rows, width);
  op.Process();
}

namespace vq2a8_ascendc_v4_v2 {
void LaunchActivationSign(void* stream, uint32_t blocks, void* x, void* weightScale, void* weightBias,
                          void* signs, void* output, void* valid, uint32_t rows, uint32_t width) {
  vq2a8_v4_v2_activation_sign<<<blocks, nullptr, stream>>>(
      static_cast<GM_ADDR>(x), static_cast<GM_ADDR>(weightScale), static_cast<GM_ADDR>(weightBias),
      static_cast<GM_ADDR>(signs), static_cast<GM_ADDR>(output), static_cast<GM_ADDR>(valid), rows, width);
}

void LaunchActivationQuantize(void* stream, uint32_t blocks, void* rotated, void* weightScale, void* rowBias,
                              void* quantized, void* rowScale, void* valid, uint32_t rows, uint32_t width) {
  vq2a8_v4_v2_activation_quantize<<<blocks, nullptr, stream>>>(
      static_cast<GM_ADDR>(rotated), static_cast<GM_ADDR>(weightScale), static_cast<GM_ADDR>(rowBias),
      static_cast<GM_ADDR>(quantized), static_cast<GM_ADDR>(rowScale), static_cast<GM_ADDR>(valid), rows, width);
}
}  // namespace vq2a8_ascendc_v4_v2
