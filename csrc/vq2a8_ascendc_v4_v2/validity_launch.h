// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cstdint>

namespace vq2a8_ascendc_v4_v2 {
constexpr uint32_t kLayerValidityStatuses = 6;
constexpr uint32_t kLayerValidityOutputs = 3;
constexpr uint32_t kLayerValidityRouteFlags = 8;

// These are caller-side arrays only. The launch expands each pointer into a
// kernel argument; no host pointer or unowned GM descriptor table is uploaded.
void LaunchLayerValidity(void* stream, void* const* statuses, void* const* outputs,
                         void* const* flags, void* result, uint32_t groups,
                         uint32_t gateWidth, uint32_t downWidth, uint32_t flagCount);
void LaunchLayerValidityVectorized(void* stream, void* const* statuses, void* const* outputs,
                                   void* const* flags, void* result, uint32_t groups,
                                   uint32_t gateWidth, uint32_t downWidth, uint32_t flagCount);
}  // namespace vq2a8_ascendc_v4_v2
