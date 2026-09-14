// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include "layout.h"

namespace vq2a8_ascendc {
// One immutable, startup-uploaded record per densely numbered expert. The
// owning custom class retains all six Tensor payloads for every pointer.
constexpr uint32_t kBankWords = 8;
constexpr uint32_t kBankPacked = 0;
constexpr uint32_t kBankBook = 1;
constexpr uint32_t kBankTileIds = 2;
constexpr uint32_t kBankScale = 3;
constexpr uint32_t kBankBias = 4;
constexpr uint32_t kBankSign = 5;
constexpr uint32_t kBankTiles = 6;
constexpr uint32_t kMaxResidentExperts = 256;
constexpr uint32_t kSelectColumns = 256;
constexpr uint16_t kInvalidBf16 = 0x7fc0;

// Check the complete signed int64 value BEFORE indexing the pointer table.
// In particular, do not truncate an ID such as 2**40 to a valid uint32 slot.
VQ2A8_LAYOUT_FN constexpr bool ValidResidentSlot(int64_t slot, uint32_t experts) {
  return slot >= 0 && static_cast<uint64_t>(slot) < experts;
}
}  // namespace vq2a8_ascendc
