// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cstdint>
#ifndef VQ2A8_V3_PREPARE_FN
  #define VQ2A8_V3_PREPARE_FN inline
#endif

namespace vq2a8_v3 {
constexpr uint32_t kPrepareMaxJobs = 6;
constexpr uint32_t kPrepareMaxK = 65536;
constexpr uint32_t kPrepareKAlignment = 512;
constexpr uint32_t kPrepareTile = 2048;
constexpr uint32_t kPrepareBlockBytes = 32;
constexpr float kPrepareFp8Max = 448.0f;
// vq2a8_reference.VQ2_FP8_MIN_SCALE; do not substitute another quantizer's epsilon.
constexpr float kPrepareMinScale = 1e-12f;
constexpr uint32_t kPrepareExponentMask = 0x7f800000u;
constexpr uint64_t kPrepareExponentPairMask = (uint64_t(kPrepareExponentMask) << 32) | kPrepareExponentMask;
// Seven FP32 tile buffers, four scalar blocks, one complete FP8 row, one
// int64 order tile, two uint32 gather-offset tiles, and one output byte tile.
constexpr uint32_t kPrepareUbBytes = 7 * kPrepareTile * sizeof(float) + 4 * kPrepareBlockBytes + kPrepareMaxK +
                                     kPrepareTile * sizeof(int64_t) + 2 * kPrepareTile * sizeof(uint32_t) +
                                     kPrepareTile;
static_assert(kPrepareUbBytes == 157824, "Preparation UB accounting changed");
static_assert(kPrepareMaxK % kPrepareTile == 0 && kPrepareTile % kPrepareKAlignment == 0);

VQ2A8_V3_PREPARE_FN constexpr bool ValidPrepareDimensions(int64_t jobs, int64_t k) {
  return jobs > 0 && jobs <= kPrepareMaxJobs && k > 0 && k <= kPrepareMaxK && k % kPrepareKAlignment == 0;
}

VQ2A8_V3_PREPARE_FN constexpr uint32_t PrepareTileCount(uint32_t k, uint32_t start) {
  return k - start < kPrepareTile ? k - start : kPrepareTile;
}

VQ2A8_V3_PREPARE_FN constexpr bool PrepareFiniteBits(uint32_t bits) {
  return (bits & kPrepareExponentMask) != kPrepareExponentMask;
}
}  // namespace vq2a8_v3
