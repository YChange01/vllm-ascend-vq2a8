// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "kernel_operator.h"
#include "input_plan_launch.h"

namespace vq2a8_ascendc_v4_v2 {
using namespace AscendC;
template <HardEvent Event>
__aicore__ inline void InputFence() {
  const event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
  SetFlag<Event>(event);
  WaitFlag<Event>(event);
}

class InputPlanKernel {
 public:
  __aicore__ inline void Init(GM_ADDR descriptors) {
    descriptors_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(descriptors));
    pipe_.InitBuffer(data_, kInputPlanMaxElements * sizeof(int32_t));
    base_ = GetBlockIdx() * kInputPlanDescriptorWords;
  }
  __aicore__ inline void Rows(GM_ADDR packed) {
    GlobalTensor<int32_t> source, target;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(packed));
    target.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(descriptors_.GetValue(base_)));
    const uint32_t count = descriptors_.GetValue(base_ + 2);
    const uint32_t offset = descriptors_.GetValue(base_ + 3);
    const auto local = data_.Get<int32_t>();
    // Exact scalar loads support short/unaligned rows, with no DMA overread.
    for (uint32_t i = 0; i < count; ++i) local.SetValue(i, source.GetValue(offset + i));
    InputFence<HardEvent::S_MTE3>();
    const DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(target, local, copy);
    InputFence<HardEvent::MTE3_S>();
  }
  __aicore__ inline void Slots(GM_ADDR query, GM_ADDR positions) {
    GlobalTensor<int32_t> table, slots, starts;
    GlobalTensor<int64_t> pos;
    table.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(descriptors_.GetValue(base_)));
    slots.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(descriptors_.GetValue(base_ + 1)));
    starts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(query));
    pos.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(positions));
    const int64_t position = pos.GetValue(0);
    const int64_t columns = descriptors_.GetValue(base_ + 2);
    const int64_t blockSize = descriptors_.GetValue(base_ + 4);
    const uint32_t count = descriptors_.GetValue(base_ + 5);
    const auto local = data_.Get<int32_t>();
    for (uint32_t i = 0; i < count; ++i) local.SetValue(i, -1);
    // B1 CPU eligibility establishes this range. Also guard indirect GM reads
    // defensively; invalid device inputs produce PAD rather than OOB access.
    if (starts.GetValue(0) == 0 && starts.GetValue(1) == 1 && position >= 0 &&
        position / blockSize < columns) {
      const int64_t number = table.GetValue(static_cast<uint32_t>(position / blockSize));
      local.SetValue(0, static_cast<int32_t>(number * blockSize + position % blockSize));
    }
    InputFence<HardEvent::S_MTE3>();
    const DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    // Exactly [0,max_num_batched_tokens); extra MTP/CP capacity stays untouched.
    DataCopyPad(slots, local, copy);
    InputFence<HardEvent::MTE3_S>();
  }
 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> data_;
  GlobalTensor<int64_t> descriptors_;
  uint32_t base_;
};
}  // namespace vq2a8_ascendc_v4_v2

extern "C" __global__ __aicore__ void vq2a8_input_rows(GM_ADDR descriptors, GM_ADDR packed) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::InputPlanKernel op;
  op.Init(descriptors);
  op.Rows(packed);
}
extern "C" __global__ __aicore__ void vq2a8_input_slots(
    GM_ADDR descriptors, GM_ADDR query, GM_ADDR positions) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::InputPlanKernel op;
  op.Init(descriptors);
  op.Slots(query, positions);
}
namespace vq2a8_ascendc_v4_v2 {
void LaunchInputRows(void* stream, void* descriptors, void* packed, uint32_t groups) {
  vq2a8_input_rows<<<groups, nullptr, stream>>>(static_cast<GM_ADDR>(descriptors), static_cast<GM_ADDR>(packed));
}
void LaunchInputSlots(void* stream, void* descriptors, void* query, void* positions, uint32_t groups) {
  vq2a8_input_slots<<<groups, nullptr, stream>>>(
      static_cast<GM_ADDR>(descriptors), static_cast<GM_ADDR>(query), static_cast<GM_ADDR>(positions));
}
}  // namespace vq2a8_ascendc_v4_v2
