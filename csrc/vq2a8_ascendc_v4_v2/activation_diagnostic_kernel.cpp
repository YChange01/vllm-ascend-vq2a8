// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "kernel_operator.h"
#include "activation_diagnostic_launch.h"

namespace vq2a8_ascendc_v4_v2 {
namespace tail_diagnostic {
using namespace AscendC;
constexpr float kFp8Maximum = 448.0f;
constexpr float kMinimumScale = 1.0e-12f;
constexpr float kFiniteMaximum = 3.4028234663852886e+38f;
constexpr uint32_t kScalarBytes = 32;
constexpr uint32_t kMaskBitsPerWord = 32;

template <HardEvent Event>
__aicore__ inline void Fence() {
  const event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
  SetFlag<Event>(event);
  WaitFlag<Event>(event);
}

class Kernel {
 public:
  __aicore__ inline void Init(
      GM_ADDR rotated, GM_ADDR weightScale, GM_ADDR rowBias, GM_ADDR transformed,
      GM_ADDR maximum, GM_ADDR dividedScale, GM_ADDR scale, GM_ADDR normalized,
      GM_ADDR clamped, GM_ADDR quantized, GM_ADDR valid, uint32_t rows, uint32_t width) {
    rows_ = rows;
    width_ = width;
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rotated));
    weightScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weightScale));
    rowBias_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rowBias));
    transformed_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(transformed));
    maximum_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(maximum));
    dividedScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dividedScale));
    scale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
    normalized_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(normalized));
    clamped_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(clamped));
    quantized_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(quantized));
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
      Fence<HardEvent::MTE2_V>();
      // Same arithmetic and validity branch as ActivationQuantizeKernel.
      // Snapshot stores are the only extra work and are not a serving path.
      Mul(x, x, weight, width_);
      PipeBarrier<PIPE_V>();
      StoreVector(transformed_, base, x);
      Abs(absolute, x, width_);
      PipeBarrier<PIPE_V>();
      Compares(maskUb_.Get<uint8_t>(), absolute, kFiniteMaximum, CMPMODE::LE, width_);
      PipeBarrier<PIPE_V>();
      const float bias = rowBias_.GetValue(row);
      Fence<HardEvent::V_S>();
      uint32_t mask = 0xffffffffU;
      for (uint32_t i = 0; i < width_ / kMaskBitsPerWord; ++i) mask &= maskUb_.Get<uint32_t>().GetValue(i);
      const bool valid = mask == 0xffffffffU && bias >= -kFiniteMaximum && bias <= kFiniteMaximum;
      if (valid) {
        ReduceMax(reduced, absolute, scratchUb_.Get<float>(), width_, false);
        PipeBarrier<PIPE_V>();
        StoreScalar(maximum_, row, reduced);
        Divs(reduced, reduced, kFp8Maximum, 1);
        PipeBarrier<PIPE_V>();
        StoreScalar(dividedScale_, row, reduced);
        Maxs(reduced, reduced, kMinimumScale, 1);
        StoreScalar(scale_, row, reduced);
        Fence<HardEvent::V_S>();
        const float scale = reduced.GetValue(0);
        Duplicate(divisor, scale, width_);
        PipeBarrier<PIPE_V>();
        Div(x, x, divisor, width_);
        PipeBarrier<PIPE_V>();
        StoreVector(normalized_, base, x);
        Mins(x, x, kFp8Maximum, width_);
        PipeBarrier<PIPE_V>();
        Maxs(x, x, -kFp8Maximum, width_);
        PipeBarrier<PIPE_V>();
        StoreVector(clamped_, base, x);
        Cast(quantUb_.Get<fp8_e4m3fn_t>(), x, RoundMode::CAST_RINT, width_);
        Fence<HardEvent::V_MTE3>();
      } else {
        // Match legacy deterministic q/scale poison. Undefined intermediate
        // stages are explicitly filled with NaN, never stale UB contents.
        Duplicate(quantUb_.Get<uint16_t>(), uint16_t(0x7f7f), width_ / 2);
        Duplicate(reduceUb_.Get<uint32_t>(), uint32_t(0x7fc00000), 8);
        Duplicate(xUb_.Get<uint32_t>(), uint32_t(0x7fc00000), width_);
        StoreScalar(maximum_, row, reduced);
        StoreScalar(dividedScale_, row, reduced);
        StoreScalar(scale_, row, reduced);
        StoreVector(normalized_, base, x);
        StoreVector(clamped_, base, x);
        Fence<HardEvent::V_MTE3>();
      }
      DataCopy(quantized_[base], quantUb_.Get<uint8_t>(), width_);
      statusUb_.Get<int32_t>().SetValue(0, valid ? 1 : 0);
      Fence<HardEvent::S_MTE3>();
      const DataCopyExtParams scalar{1, sizeof(int32_t), 0, 0, 0};
      DataCopyPad(valid_[row], statusUb_.Get<int32_t>(), scalar);
      Fence<HardEvent::MTE3_S>();
      Fence<HardEvent::MTE3_V>();
    }
  }

 private:
  __aicore__ inline void StoreVector(GlobalTensor<float>& target, uint64_t base,
                                    const LocalTensor<float>& value) {
    Fence<HardEvent::V_MTE3>();
    DataCopy(target[base], value, width_);
    // The next in-place vector operation must not race this snapshot read.
    Fence<HardEvent::MTE3_V>();
  }
  __aicore__ inline void StoreScalar(GlobalTensor<float>& target, uint32_t row,
                                    const LocalTensor<float>& value) {
    Fence<HardEvent::V_MTE3>();
    const DataCopyExtParams scalar{1, sizeof(float), 0, 0, 0};
    DataCopyPad(target[row], value, scalar);
    Fence<HardEvent::MTE3_V>();
  }
  TPipe pipe_;
  TBuf<TPosition::VECCALC> xUb_, weightUb_, absUb_, scratchUb_, divisorUb_, quantUb_, maskUb_, reduceUb_, statusUb_;
  GlobalTensor<float> x_, weightScale_, rowBias_, transformed_, maximum_, dividedScale_, scale_, normalized_, clamped_;
  GlobalTensor<uint8_t> quantized_;
  GlobalTensor<int32_t> valid_;
  uint32_t rows_, width_;
};
}  // namespace tail_diagnostic
}  // namespace vq2a8_ascendc_v4_v2

extern "C" __global__ __aicore__ void vq2a8_v4_v2_activation_tail_diagnostic(
    GM_ADDR rotated, GM_ADDR weightScale, GM_ADDR rowBias, GM_ADDR transformed,
    GM_ADDR maximum, GM_ADDR dividedScale, GM_ADDR scale, GM_ADDR normalized,
    GM_ADDR clamped, GM_ADDR quantized, GM_ADDR valid, uint32_t rows, uint32_t width) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::tail_diagnostic::Kernel kernel;
  kernel.Init(rotated, weightScale, rowBias, transformed, maximum, dividedScale, scale,
              normalized, clamped, quantized, valid, rows, width);
  kernel.Process();
}

namespace vq2a8_ascendc_v4_v2 {
void LaunchActivationTailDiagnostic(
    void* stream, uint32_t blocks, void* rotated, void* weightScale, void* rowBias,
    void* transformed, void* maximum, void* dividedScale, void* scale,
    void* normalized, void* clamped, void* quantized, void* valid, uint32_t rows, uint32_t width) {
  vq2a8_v4_v2_activation_tail_diagnostic<<<blocks, nullptr, stream>>>(
      static_cast<GM_ADDR>(rotated), static_cast<GM_ADDR>(weightScale), static_cast<GM_ADDR>(rowBias),
      static_cast<GM_ADDR>(transformed), static_cast<GM_ADDR>(maximum), static_cast<GM_ADDR>(dividedScale),
      static_cast<GM_ADDR>(scale), static_cast<GM_ADDR>(normalized), static_cast<GM_ADDR>(clamped),
      static_cast<GM_ADDR>(quantized), static_cast<GM_ADDR>(valid), rows, width);
}
}  // namespace vq2a8_ascendc_v4_v2
