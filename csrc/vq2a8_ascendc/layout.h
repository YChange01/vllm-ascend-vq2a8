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
constexpr uint32_t kPairs = kHalfTileBytes / 2;
constexpr uint32_t kMaxJobs = 6;
// Seven pointers, four dimensions, one reserved word. Host-only construction.
constexpr uint32_t kJobWords = 12;

// Gather offsets are BYTES, not elements. These pure index functions are
// tested on host and used to build bounded per-core constant UB tables.
VQ2A8_LAYOUT_FN constexpr uint32_t NzRow(uint32_t byte) { return (byte / kC0) % kHalf; }
VQ2A8_LAYOUT_FN constexpr uint32_t NzCol(uint32_t byte) { return byte / (kHalf * kC0) * kC0 + byte % kC0; }
VQ2A8_LAYOUT_FN constexpr uint32_t NdGatherOffset(uint32_t word) { return NzRow(word * 4) * kK + NzCol(word * 4); }
VQ2A8_LAYOUT_FN constexpr uint32_t PackedGatherOffset(uint32_t pair) {
  return (pair / kK * (kK / 8) + pair % kK / 8) * 4;
}
VQ2A8_LAYOUT_FN constexpr uint32_t IdGatherOffset(uint32_t pair) { return (pair % kK / 4) * 4; }
VQ2A8_LAYOUT_FN constexpr uint32_t PairGatherOffset(uint32_t byte) {
  return (NzRow(byte) / 2 * kK + NzCol(byte)) * 2 + NzRow(byte) % 2;
}

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

// Four columns from one packed word, two output rows per codebook entry.
// Reader.GetValue() returns a little-endian uint16_t pair, not a converted
// FP8 value. This same implementation is exercised by the host layout test.
// Invalid tile IDs produce NaNs without reading beyond the compact UB table.
template <typename Reader>
VQ2A8_LAYOUT_FN void DecodeFour(uint32_t codes, uint32_t tileIds, uint32_t tiles, const Reader& table, uint32_t& even,
                                uint32_t& odd) {
  even = 0;
  odd = 0;
  for (uint32_t lane = 0; lane < 4; ++lane) {
    uint32_t tile = (tileIds >> (lane * 8)) & 255u;
    uint32_t code = (codes >> (lane * 4)) & 15u;
    uint32_t pair = uint32_t(kInvalidFp8) * 0x101u;
    if (tile < tiles) {
      pair = table.GetValue(tile * 16 + code);
    }
    even |= (pair & 255u) << (lane * 8);
    odd |= (pair >> 8) << (lane * 8);
  }
}
VQ2A8_LAYOUT_FN constexpr bool ValidDimensions(int64_t m, int64_t n, int64_t k, int64_t tiles) {
  return m > 0 && m <= kM && n > 0 && n <= kMaxDimension && n % kN == 0 && k > 0 && k <= kMaxDimension &&
         k % 512 == 0 && tiles > 0 && tiles <= kMaxTiles;
}
}  // namespace vq2a8_ascendc
