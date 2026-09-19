// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cstdint>

namespace vq2a8_ascendc_v4_v2 {
// Diagnostic-only snapshots of the existing activation quantizer. Never used
// by a preparation mode or by serving; output order is diagnostic ABI 1.
void LaunchActivationTailDiagnostic(
    void* stream, uint32_t blocks, void* rotated, void* weightScale, void* rowBias,
    void* transformed, void* maximum, void* dividedScale, void* scale,
    void* normalized, void* clamped, void* quantized, void* valid,
    uint32_t rows, uint32_t width);
}  // namespace vq2a8_ascendc_v4_v2
