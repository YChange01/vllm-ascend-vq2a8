// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cstdint>

namespace vq2a8_ascendc_v4_v2 {
constexpr uint32_t kInputPlanMaxGroups = 16;
constexpr uint32_t kInputPlanMaxElements = 4096;
constexpr uint32_t kInputPlanDescriptorWords = 7;
// Per group: table address, slot address, row columns, packed offset,
// logical block size, padding limit, reserved zero.
void LaunchInputRows(void* stream, void* descriptors, void* packed, uint32_t groups);
void LaunchInputSlots(void* stream, void* descriptors, void* query, void* positions, uint32_t groups);
}  // namespace vq2a8_ascendc_v4_v2
