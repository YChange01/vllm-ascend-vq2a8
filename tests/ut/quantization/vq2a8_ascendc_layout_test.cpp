// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Host-only tests of exact shared indexing, not an AscendC simulator.
#include "csrc/vq2a8_ascendc/layout.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <vector>

using namespace vq2a8_ascendc;

int main() {
  static_assert(HalfRows(0, 0) == 0);
  static_assert(HalfRows(1, 0) == 1);
  static_assert(HalfRows(1, 16) == 0);
  static_assert(HalfRows(16, 16) == 0);
  static_assert(HalfRows(17, 16) == 1);
  static_assert(HalfRows(32, 0) == 16);
  static_assert(HalfRows(32, 16) == 16);
  for (uint32_t rows = 0; rows <= 32; ++rows) {
    for (uint32_t first = 0; first <= 32; ++first) {
      uint32_t expected = 0;
      for (uint32_t row = first; row < first + 16; ++row) {
        expected += row < rows;
      }
      assert(HalfRows(rows, first) == expected);
    }
  }
  assert(ValidDimensions(1, 32, 512, 1));
  assert(ValidDimensions(32, 65536, 65536, 256));
  for (auto shape : std::vector<std::array<int64_t, 4>>{{0, 32, 512, 1},
                                                        {33, 32, 512, 1},
                                                        {1, 31, 512, 1},
                                                        {1, 32, 128, 1},
                                                        {1, 32, 512, 0},
                                                        {1, 32, 512, 257},
                                                        {1, 65568, 512, 1},
                                                        {1, 32, 66048, 1}}) {
    assert(!ValidDimensions(shape[0], shape[1], shape[2], shape[3]));
  }
  for (uint32_t tiles : {1u, 3u, 32u, 256u}) {
    constexpr uint32_t n = 96, k = 1536;
    std::vector<uint32_t> packed(n / 2 * (k / 8), 0);
    std::vector<uint8_t> ids(k), book(tiles * n);
    for (uint32_t pair = 0; pair < n / 2; ++pair) {
      for (uint32_t col = 0; col < k; ++col) {
        // Top nibble frequently >=8: tests sign-bit packed words.
        uint32_t code = (pair * 3 + col * 7 + 3) % 16;
        packed[pair * (k / 8) + col / 8] |= code << ((col % 8) * 4);
      }
    }
    for (uint32_t col = 0; col < k; ++col) {
      ids[col] = (col * 13 + col / 17) % tiles;
    }
    for (uint32_t i = 0; i < book.size(); ++i) {
      book[i] = (i * 53 + i / 32) % 256;
    }
    for (uint32_t group = 0; group < n / 32; ++group) {
      std::vector<uint8_t> compact(tiles * 32);
      for (uint32_t tile = 0; tile < tiles; ++tile) {
        std::copy_n(book.begin() + tile * n + group * 32, 32, compact.begin() + tile * 32);
      }
      for (uint32_t start = 0; start < k; start += kK) {
        std::vector<uint8_t> l1(32 * kK);
        std::vector<uint32_t> writes(32 * kK, 0);
        for (uint32_t half = 0; half < 2; ++half) {
          std::vector<uint8_t> ub(16 * kK);
          // Same packed DMA and UB decode as kernel.cpp.
          std::array<uint32_t, 8 * (kK / 8)> words{};
          for (uint32_t pair = 0; pair < 8; ++pair) {
            std::copy_n(packed.begin() + PackedOffset(group * 32 + half * 16 + pair * 2, start, k), kK / 8,
                        words.begin() + pair * (kK / 8));
          }
          for (uint32_t row = 0; row < 16; ++row) {
            for (uint32_t col = 0; col < kK; ++col) {
              auto code = Code(words[PackedOffset(row, col, kK)], col);
              ub[HalfNz(row, col)] = compact[BookOffset(ids[start + col], code, row)];
            }
          }
          // Four 512-byte UB->L1 blocks, 512-byte destination gaps.
          for (uint32_t block = 0; block < kK / 32; ++block) {
            for (uint32_t b = 0; b < 16 * 32; ++b) {
              auto dst = block * 32 * 32 + half * 16 * 32 + b;
              l1[dst] = ub[block * 16 * 32 + b];
              ++writes[dst];
            }
          }
        }
        for (uint32_t row = 0; row < 32; ++row) {
          for (uint32_t col = 0; col < kK; ++col) {
            // Independent literal code formula (not Code/BookOffset).
            auto globalRow = group * 32 + row, globalCol = start + col;
            auto code = ((globalRow / 2) * 3 + globalCol * 7 + 3) % 16;
            auto expected = book[uint32_t(ids[globalCol]) * n + group * 32 + code * 2 + globalRow % 2];
            auto offset = FullNz(row, col);
            assert(writes[offset] == 1);
            assert(l1[offset] == expected);
          }
        }
      }
    }
  }
  // Activation word reorder, partial M padding, and bridge XOR byte order.
  for (uint32_t m = 1; m <= 32; ++m) {
    for (uint32_t half = 0; half < 2; ++half) {
      std::array<uint32_t, kHalfTileBytes / 4> ndTile{}, nz{};
      auto rows = HalfRows(m, half * 16);
      for (uint32_t row = 0; row < 16; ++row) {
        if (row < rows) {
          for (uint32_t word = 0; word < kK / 4; ++word) {
            ndTile[row * (kK / 4) + word] = row * 1000 + word + 1;
          }
        }
        for (uint32_t col = 0; col < kK; col += 4) {
          nz[HalfNz(row, col) / 4] = ndTile[(row * kK + col) / 4];
        }
      }
      for (uint32_t row = 0; row < 16; ++row) {
        for (uint32_t col = 0; col < kK; col += 4) {
          auto value = nz[HalfNz(row, col) / 4];
          assert(value == (half * 16 + row < m ? row * 1000 + col / 4 + 1 : 0));
          for (uint32_t byte = 0; byte < 4; ++byte) {
            assert((((value ^ 0x80808080u) >> (byte * 8)) & 255) == (((value >> (byte * 8)) & 255) ^ 128));
          }
        }
      }
    }
  }
  // AIC block scheduling visits every N group once, including >core-count N.
  for (uint32_t groups : {1u, 2u, 3u, 28u, 29u, 256u, 2048u}) {
    auto blocks = std::min(groups, 28u);
    std::vector<uint32_t> visited(groups, 0);
    for (uint32_t core = 0; core < blocks; ++core) {
      for (uint32_t group = core; group < groups; group += blocks) {
        ++visited[group];
      }
    }
    for (auto count : visited) {
      assert(count == 1);
    }
  }
  std::cout << "ASCENDC_HOST_LAYOUT=PASS DEVICE_EXECUTION_VERIFIED=False\n";
}
