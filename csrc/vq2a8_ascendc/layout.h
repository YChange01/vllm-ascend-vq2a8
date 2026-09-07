// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cstdint>
#ifndef VQ2A8_LAYOUT_FN
  #define VQ2A8_LAYOUT_FN inline
#endif

// Shared by the device implementation and the ordinary C++ layout tests.
namespace vq2a8_ascendc {
constexpr uint32_t kM = 32;
constexpr uint32_t kN = 32;
constexpr uint32_t kK = 128;
constexpr uint32_t kHalf = 16;
constexpr uint32_t kC0 = 32;  // bytes/elements for E4M3
constexpr uint32_t kHalfTileBytes = kHalf * kK;
constexpr uint32_t kMaxTiles = 256;
constexpr uint32_t kMaxDimension = 65536;
constexpr uint8_t kInvalidFp8 = 0x7f;  // fail closed on an invalid tile ID

// Scalar row clipping, shared with host tests. AscendC::Min is a vector
// tensor API, not a two-scalar overload. Guard before unsigned subtraction.
VQ2A8_LAYOUT_FN constexpr uint32_t HalfRows(uint32_t rows, uint32_t firstRow) {
  if (rows <= firstRow) {
    return 0;
  }
  uint32_t remaining = rows - firstRow;
  return remaining < kHalf ? remaining : kHalf;
}

// Local [16,K] -> NZ [K/32,16,32]. The two AIVs interleave their
// halves into L1 [K/32,32,32], never into a dense GM weight tensor.
VQ2A8_LAYOUT_FN constexpr uint32_t HalfNz(uint32_t row, uint32_t col) {
  return (col / kC0) * kHalf * kC0 + row * kC0 + col % kC0;
}
VQ2A8_LAYOUT_FN constexpr uint32_t FullNz(uint32_t row, uint32_t col) {
  return (col / kC0) * kN * kC0 + row * kC0 + col % kC0;
}
VQ2A8_LAYOUT_FN constexpr uint32_t Code(uint32_t word, uint32_t col) { return (word >> ((col % 8) * 4)) & 15u; }
VQ2A8_LAYOUT_FN constexpr uint32_t BookOffset(uint32_t tile, uint32_t code, uint32_t row) {
  return tile * 32 + code * 2 + row % 2;
}
VQ2A8_LAYOUT_FN constexpr uint32_t PackedOffset(uint32_t row, uint32_t col, uint32_t k) {
  return (row / 2) * (k / 8) + col / 8;
}
VQ2A8_LAYOUT_FN constexpr bool ValidDimensions(int64_t m, int64_t n, int64_t k, int64_t tiles) {
  return m > 0 && m <= kM && n > 0 && n <= kMaxDimension && n % kN == 0 && k > 0 && k <= kMaxDimension &&
         k % 512 == 0 && tiles > 0 && tiles <= kMaxTiles;
}
}  // namespace vq2a8_ascendc
