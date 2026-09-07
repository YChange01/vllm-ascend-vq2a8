// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Host-only event ownership model. Includes the actual extracted Fence body,
// not an independently reimplemented candidate. This is not a CANN simulator.
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#define __aicore__
enum class HardEvent { M_MTE1, MTE1_M, S_MTE3, Count };
using event_t = int;
struct Pool {
  std::array<uint32_t, 3> owned{7, 0, 3};
  std::array<uint32_t, 3> flags{7, 0, 3};
  int queries = 0;

  int FetchEventID(HardEvent event) {
    ++queries;
    for (int i = 0; i < 8; ++i) {
      if (!(owned[static_cast<int>(event)] & (1u << i))) return i;
    }
    throw std::runtime_error("No free event");
  }
};
Pool* active = nullptr;
Pool* GetTPipePtr() { return active; }
template <HardEvent E>
void SetFlag(event_t id) {
  auto& flags = active->flags[static_cast<int>(E)];
  if (flags & (1u << id)) throw std::runtime_error("duplicate set_flag");
  flags |= 1u << id;
}
template <HardEvent E>
void WaitFlag(event_t id) {
  auto& flags = active->flags[static_cast<int>(E)];
  if (!(flags & (1u << id))) throw std::runtime_error("wait without token");
  flags &= ~(1u << id);
}

#include "fence_under_test.h"

int main() {
  Pool pool;
  active = &pool;
  // Confirm this test detects the old hard-coded-ID defect.
  bool detected = false;
  try {
    SetFlag<HardEvent::M_MTE1>(0);
  } catch (const std::runtime_error&) {
    detected = true;
  }
  assert(detected);
  const auto initial = pool.flags;
  for (int group = 0; group < 3; ++group) {
    for (int k = 0; k < 4; ++k) {
      Fence<HardEvent::M_MTE1>();
      Fence<HardEvent::MTE1_M>();
      Fence<HardEvent::S_MTE3>();
      assert(pool.flags == initial);  // framework tokens remain untouched
    }
  }
  assert(pool.queries == 36);
  // Framework teardown must still have its initialization tokens.
  for (int id = 0; id < 3; ++id) WaitFlag<HardEvent::M_MTE1>(id);
  assert(pool.flags[0] == 0);
  std::cout << "ASCENDC_HOST_SYNC=PASS DEVICE_EXECUTION_VERIFIED=False\n";
}
