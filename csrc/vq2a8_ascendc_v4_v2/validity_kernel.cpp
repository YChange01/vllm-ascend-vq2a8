// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "kernel_operator.h"
#include "validity_launch.h"

namespace vq2a8_ascendc_v4_v2 {
using namespace AscendC;
constexpr uint32_t kValidityMaximumWidth = 4096;
constexpr uint32_t kValidityScalarBytes = 32;
constexpr uint32_t kValidityMaskBits = 32;
constexpr float kValidityFiniteMaximum = 3.4028234663852886e+38f;

template <HardEvent Event>
__aicore__ inline void ValidityFence() {
  const event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
  SetFlag<Event>(event);
  WaitFlag<Event>(event);
}

class LayerValidityKernel {
 public:
  __aicore__ inline void Init(GM_ADDR result) {
    result_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(result));
    pipe_.InitBuffer(inputUb_, kValidityMaximumWidth * sizeof(bfloat16_t));
    pipe_.InitBuffer(floatUb_, kValidityMaximumWidth * sizeof(float));
    pipe_.InitBuffer(scratchUb_, kValidityMaximumWidth * sizeof(float));
    pipe_.InitBuffer(maskUb_, kValidityMaximumWidth / 8);
    pipe_.InitBuffer(resultUb_, kValidityScalarBytes);
    valid_ = true;
  }

  __aicore__ inline void CheckStatus(GM_ADDR pointer, uint32_t groups) {
    GlobalTensor<int32_t> status;
    status.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pointer));
    for (uint32_t row = 0; row < groups; ++row) valid_ = (status.GetValue(row) != 0) && valid_;
  }

  __aicore__ inline void CheckFlag(GM_ADDR pointer) {
    GlobalTensor<uint8_t> flag;
    flag.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(pointer));
    valid_ = (flag.GetValue(0) != 0) && valid_;
  }

  __aicore__ inline void CheckOutput(GM_ADDR pointer, uint32_t rows, uint32_t width) {
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(pointer));
    const auto input = inputUb_.Get<bfloat16_t>();
    const auto widened = floatUb_.Get<float>();
    const auto scratch = scratchUb_.Get<float>();
    const auto mask = maskUb_.Get<uint8_t>();
    for (uint32_t row = 0; row < rows; ++row) {
      DataCopy(input, output[row * width], width);
      ValidityFence<HardEvent::MTE2_V>();
      // BF16 widening is exact. Only classification is performed; no model
      // arithmetic or reduction order changes. NaNs fail LE, +/-Inf exceed max.
      Cast(widened, input, RoundMode::CAST_NONE, width);
      PipeBarrier<PIPE_V>();
      Abs(scratch, widened, width);
      PipeBarrier<PIPE_V>();
      Compares(mask, scratch, kValidityFiniteMaximum, CMPMODE::LE, width);
      ValidityFence<HardEvent::V_S>();
      uint32_t allBits = 0xffffffffU;
      for (uint32_t word = 0; word < width / kValidityMaskBits; ++word) {
        allBits &= maskUb_.Get<uint32_t>().GetValue(word);
      }
      valid_ = (allBits == 0xffffffffU) && valid_;
      // Do not overwrite the input UB while the prior vector read is in flight.
      ValidityFence<HardEvent::V_MTE2>();
    }
  }

  __aicore__ inline void Finish() {
    // One writer and a fresh result per invocation: no atomics, shared status
    // cache, cross-core reduction, or stale invalid flag on graph replay.
    resultUb_.Get<uint8_t>().SetValue(0, valid_ ? 1 : 0);
    ValidityFence<HardEvent::S_MTE3>();
    const DataCopyExtParams scalar{1, sizeof(uint8_t), 0, 0, 0};
    DataCopyPad(result_, resultUb_.Get<uint8_t>(), scalar);
    ValidityFence<HardEvent::MTE3_S>();
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> inputUb_, floatUb_, scratchUb_, maskUb_, resultUb_;
  GlobalTensor<uint8_t> result_;
  bool valid_;
};
}  // namespace vq2a8_ascendc_v4_v2

extern "C" __global__ __aicore__ void vq2a8_v4_v2_layer_validity(
    GM_ADDR s0, GM_ADDR s1, GM_ADDR s2, GM_ADDR s3, GM_ADDR s4, GM_ADDR s5,
    GM_ADDR gate, GM_ADDR down, GM_ADDR combined,
    GM_ADDR f0, GM_ADDR f1, GM_ADDR f2, GM_ADDR f3, GM_ADDR f4, GM_ADDR f5, GM_ADDR f6, GM_ADDR f7,
    GM_ADDR result, uint32_t groups, uint32_t gateWidth, uint32_t downWidth, uint32_t flagCount) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::LayerValidityKernel op;
  op.Init(result);
  op.CheckStatus(s0, groups);
  op.CheckStatus(s1, groups);
  op.CheckStatus(s2, groups);
  op.CheckStatus(s3, groups);
  op.CheckStatus(s4, groups);
  op.CheckStatus(s5, groups);
  op.CheckOutput(gate, groups, gateWidth);
  op.CheckOutput(down, groups, downWidth);
  op.CheckOutput(combined, 1, downWidth);
  if (flagCount > 0) op.CheckFlag(f0);
  if (flagCount > 1) op.CheckFlag(f1);
  if (flagCount > 2) op.CheckFlag(f2);
  if (flagCount > 3) op.CheckFlag(f3);
  if (flagCount > 4) op.CheckFlag(f4);
  if (flagCount > 5) op.CheckFlag(f5);
  if (flagCount > 6) op.CheckFlag(f6);
  if (flagCount > 7) op.CheckFlag(f7);
  op.Finish();
}

namespace vq2a8_ascendc_v4_v2 {
void LaunchLayerValidity(void* stream, void* const* statuses, void* const* outputs,
                         void* const* flags, void* result, uint32_t groups,
                         uint32_t gateWidth, uint32_t downWidth, uint32_t flagCount) {
  vq2a8_v4_v2_layer_validity<<<1, nullptr, stream>>>(
      static_cast<GM_ADDR>(statuses[0]), static_cast<GM_ADDR>(statuses[1]), static_cast<GM_ADDR>(statuses[2]),
      static_cast<GM_ADDR>(statuses[3]), static_cast<GM_ADDR>(statuses[4]), static_cast<GM_ADDR>(statuses[5]),
      static_cast<GM_ADDR>(outputs[0]), static_cast<GM_ADDR>(outputs[1]), static_cast<GM_ADDR>(outputs[2]),
      static_cast<GM_ADDR>(flags[0]), static_cast<GM_ADDR>(flags[1]), static_cast<GM_ADDR>(flags[2]),
      static_cast<GM_ADDR>(flags[3]), static_cast<GM_ADDR>(flags[4]), static_cast<GM_ADDR>(flags[5]),
      static_cast<GM_ADDR>(flags[6]), static_cast<GM_ADDR>(flags[7]), static_cast<GM_ADDR>(result),
      groups, gateWidth, downWidth, flagCount);
}
}  // namespace vq2a8_ascendc_v4_v2
