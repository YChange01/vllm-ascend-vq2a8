// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cstdint>

namespace vq2a8_ascendc_v4_v2 {
#ifndef VQ2_B1_SCHEDULE_FN
  #define VQ2_B1_SCHEDULE_FN constexpr
  #define VQ2_B1_SCHEDULE_UNDEF_FN
#endif
// A bijection over jobs*nTiles. Only task placement changes: each task keeps
// its original descriptor, output tile, K traversal and accumulation order.
// The caller validates jobs>0, nTiles>0 and M=1 before scheduling this kernel.
VQ2_B1_SCHEDULE_FN uint32_t B1TileMajorWork(uint32_t work, uint32_t jobs, uint32_t nTiles) {
  return (work % jobs) * nTiles + work / jobs;
}
#ifdef VQ2_B1_SCHEDULE_UNDEF_FN
  #undef VQ2_B1_SCHEDULE_FN
  #undef VQ2_B1_SCHEDULE_UNDEF_FN
#endif
}  // namespace vq2a8_ascendc_v4_v2
