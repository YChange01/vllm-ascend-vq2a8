// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "kernel_operator.h"
#include "route_mapping_launch.h"

namespace vq2a8_ascendc_v4_v2 {
using namespace AscendC;
constexpr uint32_t kRouteMappingDmaBlockBytes = 32;
constexpr uint32_t kRouteMappingSlotsBytes =
    ((kRouteMappingMaximumGroups * sizeof(int64_t) + kRouteMappingDmaBlockBytes - 1) /
     kRouteMappingDmaBlockBytes) * kRouteMappingDmaBlockBytes;

template <HardEvent Event>
__aicore__ inline void RouteMappingFence() {
  const event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
  SetFlag<Event>(event);
  WaitFlag<Event>(event);
}

class RouteMappingKernel {
 public:
  __aicore__ inline void Init(GM_ADDR ids, GM_ADDR lookup, GM_ADDR slots, GM_ADDR valid) {
    ids_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ids));
    lookup_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(lookup));
    slots_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots));
    valid_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(valid));
    pipe_.InitBuffer(slotsUb_, kRouteMappingSlotsBytes);
    pipe_.InitBuffer(validUb_, kRouteMappingDmaBlockBytes);
  }

  __aicore__ inline void Process(uint32_t groups, uint32_t experts) {
    const auto slots = slotsUb_.Get<int64_t>();
    bool valid = true;
    for (uint32_t group = 0; group < groups; ++group) {
      // Scalar GM reads cover exactly the requested INT64 elements. In
      // particular G=1 and naturally aligned offset views never overread a
      // 32-byte DMA block. Never narrow/index an unchecked signed route ID.
      const int64_t id = ids_.GetValue(group);
      int64_t slot = -1;
      if (id >= int64_t(0) && id < static_cast<int64_t>(experts)) {
        slot = lookup_.GetValue(static_cast<uint32_t>(id));
      }
      slots.SetValue(group, slot);
      valid = (slot >= int64_t(0)) && valid;
    }
    // This exactly reproduces all(slots >= 0), not all(slots < bank_size).
    // Preserve invalid positive mappings so the existing bank select/project
    // range gates reject them without reading out-of-range expert pointers.
    validUb_.Get<uint8_t>().SetValue(0, valid ? 1 : 0);
    RouteMappingFence<HardEvent::S_MTE3>();
    const DataCopyExtParams slotCopy{1, groups * static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
    const DataCopyExtParams flagCopy{1, sizeof(uint8_t), 0, 0, 0};
    // Write exact byte counts: output storage can be only 8 and 1 bytes.
    DataCopyPad(slots_, slots, slotCopy);
    DataCopyPad(valid_, validUb_.Get<uint8_t>(), flagCopy);
    RouteMappingFence<HardEvent::MTE3_S>();
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> slotsUb_, validUb_;
  GlobalTensor<int64_t> ids_, lookup_, slots_;
  GlobalTensor<uint8_t> valid_;
};
}  // namespace vq2a8_ascendc_v4_v2

extern "C" __global__ __aicore__ void vq2a8_v4_v2_route_mapping(
    GM_ADDR ids, GM_ADDR lookup, GM_ADDR slots, GM_ADDR valid, uint32_t groups, uint32_t experts) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  vq2a8_ascendc_v4_v2::RouteMappingKernel op;
  op.Init(ids, lookup, slots, valid);
  op.Process(groups, experts);
}

namespace vq2a8_ascendc_v4_v2 {
void LaunchRouteMapping(void* stream, void* ids, void* lookup, void* slots, void* valid,
                        uint32_t groups, uint32_t experts) {
  vq2a8_v4_v2_route_mapping<<<1, nullptr, stream>>>(
      static_cast<GM_ADDR>(ids), static_cast<GM_ADDR>(lookup), static_cast<GM_ADDR>(slots),
      static_cast<GM_ADDR>(valid), groups, experts);
}
}  // namespace vq2a8_ascendc_v4_v2
