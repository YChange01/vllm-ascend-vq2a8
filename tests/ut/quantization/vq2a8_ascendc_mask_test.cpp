// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Host model of the reported Ands type contract, NOT a CANN implementation.
#include "csrc/vq2a8_ascendc/layout.h"
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <type_traits>

using namespace vq2a8_ascendc;
#define __aicore__

template <typename T>
struct LocalTensor {
  unsigned char* bytes;
  template <typename U>
  LocalTensor<U> ReinterpretCast() const {
    return {bytes};
  }
};

template <typename T>
void Ands(const LocalTensor<T>& dst, const LocalTensor<T>& src, T scalar, uint32_t count) {
  // Exact whitelist reported by the target CANN 9.1 compiler.
  static_assert(std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int64_t> ||
                    std::is_same_v<T, uint64_t>,
                "Ands unsupported dtype");
  assert(sizeof(T) == 8 && count == kPairs / 2 && dst.bytes == src.bytes);
  // memcpy avoids strict-aliasing UB in this host-only model.
  for (uint32_t i = 0; i < count; ++i) {
    T value;
    std::memcpy(&value, src.bytes + i * sizeof(T), sizeof(T));
    value &= scalar;
    std::memcpy(dst.bytes + i * sizeof(T), &value, sizeof(T));
  }
}

// Extracted verbatim from kernel.cpp by the Python test.
#include "mask_under_test.h"

int main() {
  static_assert(kCodeLanePairMask == 0x0000000f0000000full);
  static_assert(kTileLanePairMask == 0x000000ff000000ffull);
  alignas(32) std::array<uint32_t, kPairs + 8> codes{}, tiles{};
  // All nibble/ID values, high garbage bits, distinct neighboring lanes.
  // Guard words detect an unhalved count or a write beyond the last lane.
  for (uint32_t seed = 0; seed < 256; ++seed) {
    for (uint32_t i = 0; i < codes.size(); ++i) {
      codes[i] = (i * 0x9e3779b9u + seed) ^ 0xf0f08000u;
      tiles[i] = (i * 0x85ebca6bu + seed) ^ 0xff008000u;
    }
    auto originalCodes = codes, originalTiles = tiles;
    MaskDecodeLanes({reinterpret_cast<unsigned char*>(codes.data())}, {reinterpret_cast<unsigned char*>(tiles.data())});
    for (uint32_t i = 0; i < codes.size(); ++i) {
      assert(codes[i] == (i < kPairs ? originalCodes[i] & 15u : originalCodes[i]));
      assert(tiles[i] == (i < kPairs ? originalTiles[i] & 255u : originalTiles[i]));
    }
  }
  std::cout << "ASCENDC_HOST_MASK=PASS DEVICE_EXECUTION_VERIFIED=False\n";
}
