// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cstdint>

namespace vq2a8_ascendc_v4_v2 {
void LaunchBiasDotRowsProbe(void* stream, uint32_t blocks, void* rotated,
                           void* weightBias, void* output, uint32_t rows, uint32_t width);
}
