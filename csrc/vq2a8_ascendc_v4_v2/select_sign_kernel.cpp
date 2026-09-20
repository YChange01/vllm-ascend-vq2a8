// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// The arithmetic below is the existing ActivationSignKernel, fed directly
// from immutable resident metadata instead of its freshly selected GM copies.
#include "kernel_operator.h"
#include "select_sign_launch.h"
#define VQ2_V2_LAYOUT_FN __aicore__ inline
#include "resident_layout.h"
#undef VQ2_V2_LAYOUT_FN

namespace vq2a8_ascendc_v4_v2 {
namespace select_sign {
using namespace AscendC;
constexpr float kFiniteMaximum = 3.4028234663852886e+38f;
constexpr uint32_t kScalarBytes = 32, kMaskBitsPerWord = 32;

template <HardEvent Event>
__aicore__ inline void Fence() {
  const event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
  SetFlag<Event>(event);
  WaitFlag<Event>(event);
}
__aicore__ inline void FiniteMask(const LocalTensor<float>& value, const LocalTensor<float>& scratch,
                                const LocalTensor<uint8_t>& mask, uint32_t width) {
  Abs(scratch, value, width);
  PipeBarrier<PIPE_V>();
  Compares(mask, scratch, kFiniteMaximum, CMPMODE::LE, width);
  PipeBarrier<PIPE_V>();
}

template <typename InputT>
class Kernel {
 public:
  __aicore__ inline void Init(GM_ADDR bank, GM_ADDR ids, GM_ADDR input, GM_ADDR signedOutput,
                            GM_ADDR selectedScale, GM_ADDR selectedBias, GM_ADDR selectStatus,
                            GM_ADDR inputStatus, uint32_t experts, uint32_t groups,
                            uint32_t width, uint64_t rowStride) {
    experts_ = experts; groups_ = groups; width_ = width; rowStride_ = rowStride;
    bank_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(bank));
    ids_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ids));
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(input));
    signedOutput_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(signedOutput));
    selectedScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(selectedScale));
    selectedBias_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(selectedBias));
    selectStatus_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(selectStatus));
    inputStatus_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(inputStatus));
    pipe_.InitBuffer(xUb_, width * sizeof(float));
    if constexpr (sizeof(InputT) != sizeof(float)) pipe_.InitBuffer(inputUb_, width * sizeof(InputT));
    pipe_.InitBuffer(scaleUb_, width * sizeof(float));
    pipe_.InitBuffer(biasUb_, width * sizeof(float));
    pipe_.InitBuffer(signUb_, width);
    pipe_.InitBuffer(halfUb_, width * sizeof(half));
    pipe_.InitBuffer(signFloatUb_, width * sizeof(float));
    pipe_.InitBuffer(scratchUb_, width * sizeof(float));
    pipe_.InitBuffer(maskUb_, width / 8);
    pipe_.InitBuffer(otherMaskUb_, width / 8);
    pipe_.InitBuffer(selectStatusUb_, kScalarBytes);
    pipe_.InitBuffer(inputStatusUb_, kScalarBytes);
  }
  __aicore__ inline void Process() {
    const auto x = xUb_.Get<float>(), scale = scaleUb_.Get<float>(), bias = biasUb_.Get<float>();
    const auto sign = signUb_.Get<int8_t>();
    const auto signFloat = signFloatUb_.Get<float>();
    const auto scratch = scratchUb_.Get<float>();
    const auto mask = maskUb_.Get<uint8_t>(), other = otherMaskUb_.Get<uint8_t>();
    for (uint32_t row = GetBlockIdx(); row < groups_; row += GetBlockNum()) {
      const uint64_t base = uint64_t(row) * width_;
      const int64_t slot = ids_.GetValue(row);
      const bool selected = ValidResidentSlot(slot, experts_);
      if constexpr (sizeof(InputT) == sizeof(float)) DataCopy(x, input_[uint64_t(row) * rowStride_], width_);
      else DataCopy(inputUb_.Get<InputT>(), input_[uint64_t(row) * rowStride_], width_);
      if (selected) {
        // Full signed int64 bounds precede every indirect pointer formation.
        const uint32_t record = static_cast<uint32_t>(slot) * kBankWords;
        GlobalTensor<float> weightScale, weightBias;
        GlobalTensor<int8_t> signs;
        weightScale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bank_.GetValue(record + kBankScale)));
        weightBias.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bank_.GetValue(record + kBankBias)));
        signs.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(bank_.GetValue(record + kBankSign)));
        DataCopy(scale, weightScale, width_);
        DataCopy(bias, weightBias, width_);
        DataCopy(sign, signs, width_);
      }
      Fence<HardEvent::MTE2_V>();
      if (!selected) {
        // Match Select's exact NaN payload and zero signs, not expert zero.
        Duplicate(scaleUb_.Get<int32_t>(), int32_t(0x7fc00000), width_);
        Duplicate(biasUb_.Get<int32_t>(), int32_t(0x7fc00000), width_);
        Duplicate(signUb_.Get<int16_t>(), int16_t(0), width_ / sizeof(int16_t));
        PipeBarrier<PIPE_V>();
      }
      if constexpr (sizeof(InputT) != sizeof(float)) {
        Cast(x, inputUb_.Get<InputT>(), RoundMode::CAST_NONE, width_);
        PipeBarrier<PIPE_V>();
      }
      FiniteMask(x, scratch, mask, width_);
      FiniteMask(scale, scratch, other, width_);
      And(maskUb_.Get<uint16_t>(), maskUb_.Get<uint16_t>(), otherMaskUb_.Get<uint16_t>(), width_ / 16);
      PipeBarrier<PIPE_V>();
      FiniteMask(bias, scratch, other, width_);
      And(maskUb_.Get<uint16_t>(), maskUb_.Get<uint16_t>(), otherMaskUb_.Get<uint16_t>(), width_ / 16);
      PipeBarrier<PIPE_V>();
      Cast(halfUb_.Get<half>(), sign, RoundMode::CAST_NONE, width_);
      PipeBarrier<PIPE_V>();
      Cast(signFloat, halfUb_.Get<half>(), RoundMode::CAST_NONE, width_);
      PipeBarrier<PIPE_V>();
      Abs(scratch, signFloat, width_);
      PipeBarrier<PIPE_V>();
      Compares(other, scratch, 1.0f, CMPMODE::EQ, width_);
      PipeBarrier<PIPE_V>();
      And(maskUb_.Get<uint16_t>(), maskUb_.Get<uint16_t>(), otherMaskUb_.Get<uint16_t>(), width_ / 16);
      Fence<HardEvent::V_S>();
      uint32_t allBits = 0xffffffffU;
      for (uint32_t word = 0; word < width_ / kMaskBitsPerWord; ++word) allBits &= maskUb_.Get<uint32_t>().GetValue(word);
      Mul(x, x, signFloat, width_);
      Fence<HardEvent::V_MTE3>();
      DataCopy(signedOutput_[base], x, width_);
      DataCopy(selectedScale_[base], scale, width_);
      DataCopy(selectedBias_[base], bias, width_);
      selectStatusUb_.Get<int32_t>().SetValue(0, selected ? 1 : 0);
      inputStatusUb_.Get<int32_t>().SetValue(0, allBits == 0xffffffffU ? 1 : 0);
      Fence<HardEvent::S_MTE3>();
      const DataCopyExtParams scalar{1, sizeof(int32_t), 0, 0, 0};
      DataCopyPad(selectStatus_[row], selectStatusUb_.Get<int32_t>(), scalar);
      DataCopyPad(inputStatus_[row], inputStatusUb_.Get<int32_t>(), scalar);
      Fence<HardEvent::MTE3_S>();
      // UB may be reused by either next DMA or next invalid-row vector fill.
      Fence<HardEvent::MTE3_V>();
    }
  }
 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> xUb_, inputUb_, scaleUb_, biasUb_, signUb_, halfUb_, signFloatUb_, scratchUb_;
  TBuf<TPosition::VECCALC> maskUb_, otherMaskUb_, selectStatusUb_, inputStatusUb_;
  GlobalTensor<uint64_t> bank_;
  GlobalTensor<int64_t> ids_;
  GlobalTensor<InputT> input_;
  GlobalTensor<float> signedOutput_, selectedScale_, selectedBias_;
  GlobalTensor<int32_t> selectStatus_, inputStatus_;
  uint32_t experts_, groups_, width_;
  uint64_t rowStride_;
};
}  // namespace select_sign
}  // namespace vq2a8_ascendc_v4_v2

extern "C" __global__ __aicore__ void vq2a8_v4_v2_select_sign_fp32(
    GM_ADDR bank, GM_ADDR ids, GM_ADDR input, GM_ADDR signedOutput, GM_ADDR selectedScale,
    GM_ADDR selectedBias, GM_ADDR selectStatus, GM_ADDR inputStatus, uint32_t experts,
    uint32_t groups, uint32_t width, uint64_t rowStride) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::select_sign::Kernel<float> op;
  op.Init(bank, ids, input, signedOutput, selectedScale, selectedBias, selectStatus, inputStatus,
          experts, groups, width, rowStride);
  op.Process();
}
extern "C" __global__ __aicore__ void vq2a8_v4_v2_select_sign_bf16(
    GM_ADDR bank, GM_ADDR ids, GM_ADDR input, GM_ADDR signedOutput, GM_ADDR selectedScale,
    GM_ADDR selectedBias, GM_ADDR selectStatus, GM_ADDR inputStatus, uint32_t experts,
    uint32_t groups, uint32_t width, uint64_t rowStride) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::select_sign::Kernel<bfloat16_t> op;
  op.Init(bank, ids, input, signedOutput, selectedScale, selectedBias, selectStatus, inputStatus,
          experts, groups, width, rowStride);
  op.Process();
}
namespace vq2a8_ascendc_v4_v2 {
void LaunchResidentSelectSign(void* stream, uint32_t blocks, void* bank, void* ids, void* input,
                              void* signedOutput, void* selectedScale, void* selectedBias,
                              void* selectStatus, void* inputStatus, uint32_t experts,
                              uint32_t groups, uint32_t width, uint64_t rowStride, bool inputIsBf16) {
  if (inputIsBf16) {
    vq2a8_v4_v2_select_sign_bf16<<<blocks, nullptr, stream>>>(static_cast<GM_ADDR>(bank), static_cast<GM_ADDR>(ids),
        static_cast<GM_ADDR>(input), static_cast<GM_ADDR>(signedOutput), static_cast<GM_ADDR>(selectedScale),
        static_cast<GM_ADDR>(selectedBias), static_cast<GM_ADDR>(selectStatus), static_cast<GM_ADDR>(inputStatus),
        experts, groups, width, rowStride);
  } else {
    vq2a8_v4_v2_select_sign_fp32<<<blocks, nullptr, stream>>>(static_cast<GM_ADDR>(bank), static_cast<GM_ADDR>(ids),
        static_cast<GM_ADDR>(input), static_cast<GM_ADDR>(signedOutput), static_cast<GM_ADDR>(selectedScale),
        static_cast<GM_ADDR>(selectedBias), static_cast<GM_ADDR>(selectStatus), static_cast<GM_ADDR>(inputStatus),
        experts, groups, width, rowStride);
  }
}
}  // namespace vq2a8_ascendc_v4_v2
