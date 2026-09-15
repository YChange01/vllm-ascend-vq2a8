// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cstdint>

namespace vq2a8_ascendc_v4_v2 {
void LaunchActivationSign(void* stream, uint32_t blocks, void* x, void* weightScale, void* weightBias,
                          void* signs, void* output, void* valid, uint32_t rows, uint32_t width);
void LaunchActivationQuantize(void* stream, uint32_t blocks, void* rotated, void* weightScale, void* rowBias,
                              void* quantized, void* rowScale, void* valid, uint32_t rows, uint32_t width);
}  // namespace vq2a8_ascendc_v4_v2
