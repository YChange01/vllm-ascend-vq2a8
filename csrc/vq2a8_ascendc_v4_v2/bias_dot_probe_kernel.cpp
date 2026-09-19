// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "kernel_operator.h"
#include "bias_dot_probe_launch.h"

// EXPERIMENT ONLY. Vector multiply/reduction is not assumed to reproduce the
// CANN FP32 MatMul accumulation order. The independent probe rejects even one
// differing bit. No model dispatcher calls this entry point.
namespace vq2a8_ascendc_v4_v2 {
namespace bias_dot_probe {
using namespace AscendC;
template <HardEvent Event>
__aicore__ inline void Fence() {
  const event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
  SetFlag<Event>(event);
  WaitFlag<Event>(event);
}

class Kernel {
 public:
  __aicore__ inline void Init(GM_ADDR rotated, GM_ADDR weightBias, GM_ADDR output,
                             uint32_t rows, uint32_t width) {
    rows_ = rows;
    width_ = width;
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rotated));
    bias_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weightBias));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    pipe_.InitBuffer(xUb_, width * sizeof(float));
    pipe_.InitBuffer(biasUb_, width * sizeof(float));
    pipe_.InitBuffer(productUb_, width * sizeof(float));
    pipe_.InitBuffer(workUb_, width * sizeof(float));
    pipe_.InitBuffer(resultUb_, 32);
  }

  __aicore__ inline void Process() {
    for (uint32_t row = GetBlockIdx(); row < rows_; row += GetBlockNum()) {
      const uint64_t base = uint64_t(row) * width_;
      auto x = xUb_.Get<float>();
      auto bias = biasUb_.Get<float>();
      auto product = productUb_.Get<float>();
      auto result = resultUb_.Get<float>();
      DataCopy(x, x_[base], width_);
      DataCopy(bias, bias_[base], width_);
      Fence<HardEvent::MTE2_V>();
      Mul(product, x, bias, width_);
      PipeBarrier<PIPE_V>();
      ReduceSum(result, product, workUb_.Get<float>(), int32_t(width_));
      Fence<HardEvent::V_MTE3>();
      const DataCopyExtParams scalar{1, sizeof(float), 0, 0, 0};
      DataCopyPad(output_[row], result, scalar);
      Fence<HardEvent::MTE3_V>();
      Fence<HardEvent::V_MTE2>();
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> xUb_, biasUb_, productUb_, workUb_, resultUb_;
  GlobalTensor<float> x_, bias_, output_;
  uint32_t rows_, width_;
};
}  // namespace bias_dot_probe
}  // namespace vq2a8_ascendc_v4_v2

extern "C" __global__ __aicore__ void vq2a8_v4_v2_bias_dot_rows_probe(
    GM_ADDR rotated, GM_ADDR weightBias, GM_ADDR output, uint32_t rows, uint32_t width) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::bias_dot_probe::Kernel kernel;
  kernel.Init(rotated, weightBias, output, rows, width);
  kernel.Process();
}

namespace vq2a8_ascendc_v4_v2 {
void LaunchBiasDotRowsProbe(void* stream, uint32_t blocks, void* rotated,
                           void* weightBias, void* output, uint32_t rows, uint32_t width) {
  vq2a8_v4_v2_bias_dot_rows_probe<<<blocks, nullptr, stream>>>(
      static_cast<GM_ADDR>(rotated), static_cast<GM_ADDR>(weightBias),
      static_cast<GM_ADDR>(output), rows, width);
}
}  // namespace vq2a8_ascendc_v4_v2
