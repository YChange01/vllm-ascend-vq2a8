// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include "layout.h"

namespace vq2a8_ascendc_v4_v2 {
// Uploaded once at construction. All indirect pointers have strong owners.
constexpr uint32_t kBankWords = 8;
constexpr uint32_t kBankPacked = 0;
constexpr uint32_t kBankBook = 1;
constexpr uint32_t kBankOrder = 2;
constexpr uint32_t kBankScale = 3;
constexpr uint32_t kBankBias = 4;
constexpr uint32_t kBankSign = 5;
constexpr uint32_t kMaxResidentExperts = 256;
constexpr uint32_t kSelectColumns = 256;
constexpr uint16_t kInvalidBf16 = 0x7fc0;
#ifndef VQ2_V2_LAYOUT_FN
#define VQ2_V2_LAYOUT_FN constexpr
#define VQ2_V4_V2_UNDEF_RESIDENT_FN
#endif
VQ2_V2_LAYOUT_FN bool ValidResidentSlot(int64_t slot, uint32_t experts) {
  return slot >= 0 && static_cast<uint64_t>(slot) < experts;
}
#ifdef VQ2_V4_V2_UNDEF_RESIDENT_FN
#undef VQ2_V2_LAYOUT_FN
#undef VQ2_V4_V2_UNDEF_RESIDENT_FN
#endif
}  // namespace vq2a8_ascendc_v4_v2
