// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cstdint>

namespace vq2a8_ascendc_v4_v2 {
constexpr uint32_t kRouteMappingMaximumGroups = 6;
constexpr uint32_t kRouteMappingMaximumExperts = 256;

void LaunchRouteMapping(void* stream, void* ids, void* lookup, void* slots, void* valid,
                        uint32_t groups, uint32_t experts);
}  // namespace vq2a8_ascendc_v4_v2
