// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cstdint>

namespace vq2a8_ascendc_v4_v2 {
void LaunchResidentSwigluSelectSign(void* stream, uint32_t blocks, void* bank, void* ids, void* input,
                                  void* signedOutput, void* selectedScale, void* selectedBias,
                                  void* selectStatus, void* inputStatus, uint32_t experts,
                                  uint32_t groups, uint32_t width, uint64_t rowStride, float clampLimit);
}  // namespace vq2a8_ascendc_v4_v2
