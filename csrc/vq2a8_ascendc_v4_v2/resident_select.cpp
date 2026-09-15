// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "kernel_operator.h"
#include "launch.h"
#define VQ2_V2_LAYOUT_FN __aicore__ inline
#include "resident_layout.h"

namespace vq2a8_ascendc_v4_v2 {
using namespace AscendC;

template <HardEvent E>
__aicore__ inline void SelectFence() {
  event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
  SetFlag<E>(event);
  WaitFlag<E>(event);
}

// Gather only the three preparation vectors. No arithmetic, dense decoded
// weights, global expert bank duplication or activation transform is added.
class ResidentSelectKernel {
 public:
  __aicore__ inline void Init(GM_ADDR bank, GM_ADDR routeIds, GM_ADDR scale, GM_ADDR bias, GM_ADDR sign, GM_ADDR valid,
                              uint32_t experts, uint32_t routes, uint32_t k) {
    experts_ = experts;
    routes_ = routes;
    k_ = k;
    bank_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(bank));
    routeIds_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(routeIds));
    scale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
    bias_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
    sign_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(sign));
    valid_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valid));
    pipe_.InitBuffer(scaleUb_, kSelectColumns * sizeof(float));
    pipe_.InitBuffer(biasUb_, kSelectColumns * sizeof(float));
    pipe_.InitBuffer(signUb_, kSelectColumns * sizeof(int8_t));
    pipe_.InitBuffer(statusUb_, 32);
  }

  __aicore__ inline void Process() {
    const uint32_t chunks = k_ / kSelectColumns;
    for (uint32_t work = GetBlockIdx(); work < routes_ * chunks; work += GetBlockNum()) {
      const uint32_t route = work / chunks;
      const uint32_t column = (work % chunks) * kSelectColumns;
      const int64_t expert = routeIds_.GetValue(route);
      const bool valid = ValidResidentSlot(expert, experts_);
      if (valid) {
        // Do not form a pointer-table offset until the full int64 ID passed.
        const uint32_t record = static_cast<uint32_t>(expert) * kBankWords;
        GlobalTensor<float> weightScale, weightBias;
        GlobalTensor<int8_t> rhtSign;
        weightScale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bank_.GetValue(record + kBankScale)));
        weightBias.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bank_.GetValue(record + kBankBias)));
        rhtSign.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(bank_.GetValue(record + kBankSign)));
        DataCopy(scaleUb_.Get<float>(), weightScale[column], kSelectColumns);
        DataCopy(biasUb_.Get<float>(), weightBias[column], kSelectColumns);
        DataCopy(signUb_.Get<int8_t>(), rhtSign[column], kSelectColumns);
        SelectFence<HardEvent::MTE2_MTE3>();
      } else {
        // Deterministic invalid metadata also fails the unchanged row-wise
        // preparation's finite/sign checks. Never read expert zero instead.
        Duplicate(scaleUb_.Get<int32_t>(), int32_t(0x7fc00000), kSelectColumns);
        Duplicate(biasUb_.Get<int32_t>(), int32_t(0x7fc00000), kSelectColumns);
        Duplicate(signUb_.Get<int16_t>(), int16_t(0), kSelectColumns / sizeof(int16_t));
        SelectFence<HardEvent::V_MTE3>();
      }
      const uint64_t output = uint64_t(route) * k_ + column;
      DataCopy(scale_[output], scaleUb_.Get<float>(), kSelectColumns);
      DataCopy(bias_[output], biasUb_.Get<float>(), kSelectColumns);
      DataCopy(sign_[output], signUb_.Get<int8_t>(), kSelectColumns);
      // Finish reads of reusable UB before either the next MTE2 load or
      // invalid-row vector fill. Use TPipe-assigned event IDs exclusively.
      SelectFence<HardEvent::MTE3_S>();
      if (column == 0) {
        statusUb_.Get<int32_t>().SetValue(0, valid ? 1 : 0);
        SelectFence<HardEvent::S_MTE3>();
        DataCopyExtParams scalarCopy{1, sizeof(int32_t), 0, 0, 0};
        DataCopyPad(valid_[route], statusUb_.Get<int32_t>(), scalarCopy);
        SelectFence<HardEvent::MTE3_S>();
      }
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> scaleUb_, biasUb_, signUb_, statusUb_;
  GlobalTensor<uint64_t> bank_;
  GlobalTensor<int64_t> routeIds_;
  GlobalTensor<float> scale_, bias_;
  GlobalTensor<int8_t> sign_;
  GlobalTensor<int32_t> valid_;
  uint32_t experts_, routes_, k_;
};
}  // namespace vq2a8_ascendc_v4_v2

extern "C" __global__ __aicore__ void vq2a8_ascendc_v4_v2_resident_select(GM_ADDR bank, GM_ADDR routeIds, GM_ADDR scale,
                                                                    GM_ADDR bias, GM_ADDR sign, GM_ADDR valid,
                                                                    uint32_t experts, uint32_t routes, uint32_t k) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::ResidentSelectKernel op;
  op.Init(bank, routeIds, scale, bias, sign, valid, experts, routes, k);
  op.Process();
}

namespace vq2a8_ascendc_v4_v2 {
void LaunchResidentSelect(void* stream, uint32_t blocks, void* bank, void* routeIds, void* scale, void* bias,
                          void* sign, void* valid, uint32_t experts, uint32_t routes, uint32_t k) {
  vq2a8_ascendc_v4_v2_resident_select<<<blocks, nullptr, stream>>>(
      static_cast<GM_ADDR>(bank), static_cast<GM_ADDR>(routeIds), static_cast<GM_ADDR>(scale),
      static_cast<GM_ADDR>(bias), static_cast<GM_ADDR>(sign), static_cast<GM_ADDR>(valid), experts, routes, k);
}
}  // namespace vq2a8_ascendc_v4_v2
