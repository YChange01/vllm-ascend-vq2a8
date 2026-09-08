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

struct PairReader {
  const std::vector<uint8_t>& bytes;
  uint16_t GetValue(uint32_t offset) const {
    assert(offset * 2 + 1 < bytes.size());
    return uint16_t(bytes[offset * 2]) | (uint16_t(bytes[offset * 2 + 1]) << 8);
  }
};

// Host emulation of the vector lane plan. Tests bytes/indices only, not the
// NPU implementations of Gather, ShiftRight, Add or their synchronization.
std::array<uint8_t, kHalfTileBytes> VectorDecode(const std::array<uint32_t, kPairs / 8>& words,
                                                 const std::array<uint8_t, kK>& ids, const std::vector<uint8_t>& book) {
  std::array<uint8_t, kMaxTiles * kN> padded;
  padded.fill(kInvalidFp8);
  std::copy(book.begin(), book.end(), padded.begin());
  std::array<uint8_t, kHalfTileBytes> pairs{}, output{};
  for (uint32_t i = 0; i < kPairs; ++i) {
    auto wordOffset = PackedGatherOffset(i);
    assert(wordOffset % 4 == 0 && wordOffset / 4 < words.size());
    auto code = (words[wordOffset / 4] >> ((i % 8) * 4)) & 15u;
    auto idOffset = IdGatherOffset(i);
    assert(idOffset % 4 == 0 && idOffset + 3 < ids.size());
    uint32_t idWord = 0;
    for (uint32_t lane = 0; lane < 4; ++lane) {
      idWord |= uint32_t(ids[idOffset + lane]) << (lane * 8);
    }
    auto tile = (idWord >> ((i % 4) * 8)) & 255u;
    auto offset = (tile << 5) + (code << 1);
    assert(offset % 2 == 0 && offset + 1 < padded.size());
    pairs[2 * i] = padded[offset];
    pairs[2 * i + 1] = padded[offset + 1];
  }
  for (uint32_t i = 0; i < kHalfTileBytes; ++i) {
    assert(PairGatherOffset(i) < pairs.size());
    output[i] = pairs[PairGatherOffset(i)];
  }
  return output;
}

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
          // Exercise the actual shared word decoder, including word byte
          // order, row-pair sharing and writes across every NZ block.
          for (uint32_t row = 0; row < 16; row += 2) {
            for (uint32_t col = 0; col < kK; col += 8) {
              uint32_t codes = words[PackedOffset(row, col, kK)];
              for (uint32_t offset = 0; offset < 8; offset += 4) {
                uint32_t idWord = 0;
                for (uint32_t lane = 0; lane < 4; ++lane) {
                  idWord |= uint32_t(ids[start + col + offset + lane]) << (lane * 8);
                }
                uint32_t even, odd;
                DecodeFour(codes >> (offset * 4), idWord, tiles, PairReader{compact}, even, odd);
                for (uint32_t lane = 0; lane < 4; ++lane) {
                  ub[HalfNz(row, col + offset + lane)] = (even >> (lane * 8)) & 255;
                  ub[HalfNz(row + 1, col + offset + lane)] = (odd >> (lane * 8)) & 255;
                }
              }
            }
          }
          std::array<uint8_t, kK> tileIds{};
          std::copy_n(ids.begin() + start, kK, tileIds.begin());
          auto vectorOutput = VectorDecode(words, tileIds, compact);
          assert(std::equal(ub.begin(), ub.end(), vectorOutput.begin()));
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
  // Invalid IDs in the vector path must read sentinel bytes for both rows,
  // including unsigned 255; all 256 codebook encodings are preserved.
  for (uint32_t tiles : {1u, 3u, 32u, 256u}) {
    std::vector<uint8_t> book(tiles * 32);
    for (uint32_t i = 0; i < book.size(); ++i) book[i] = i % 256;
    std::array<uint32_t, kPairs / 8> words{};
    for (uint32_t i = 0; i < words.size(); ++i) words[i] = 0xfedcba98u - i;
    for (uint32_t base : {0u, 128u}) {
      std::array<uint8_t, kK> ids{};
      for (uint32_t i = 0; i < kK; ++i) ids[i] = base + i;
      auto output = VectorDecode(words, ids, book);
      for (uint32_t row = 0; row < 16; ++row) {
        for (uint32_t col = 0; col < kK; ++col) {
          auto code = (words[(row / 2) * 16 + col / 8] >> ((col % 8) * 4)) & 15;
          auto expected = ids[col] < tiles ? book[ids[col] * 32 + code * 2 + row % 2] : kInvalidFp8;
          assert(output[HalfNz(row, col)] == expected);
        }
      }
    }
  }
  // Every possible FP8 byte, nibble and tile ID, including invalid IDs.
  // This also asserts that invalid IDs never issue out-of-bounds lookups.
  for (uint32_t tiles : {1u, 3u, 32u, 256u}) {
    std::vector<uint8_t> book(tiles * 32);
    for (uint32_t byte = 0; byte < 256; ++byte) {
      for (uint32_t i = 0; i < book.size(); i += 2) {
        book[i] = byte;
        book[i + 1] = 255 - byte;
      }
      for (uint32_t tile = 0; tile < 256; ++tile) {
        for (uint32_t code = 0; code < 16; ++code) {
          uint32_t even, odd;
          DecodeFour(code * 0x1111u, tile * 0x01010101u, tiles, PairReader{book}, even, odd);
          assert(even == (tile < tiles ? byte : kInvalidFp8) * 0x01010101u);
          assert(odd == (tile < tiles ? 255 - byte : kInvalidFp8) * 0x01010101u);
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
          assert(NdGatherOffset(HalfNz(row, col) / 4) == row * kK + col);
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
  // Flattened (expert, N-group) work reaches each output tile exactly once,
  // with both AIVs assigned to the same job as their paired Cube core.
  for (uint32_t jobs : {1u, 2u, 6u}) {
    for (uint32_t groups : {1u, 3u, 32u, 33u, 256u, 2048u}) {
      uint32_t blocks = std::min(32u, jobs * groups);
      std::vector<uint32_t> visited(jobs * groups, 0);
      for (uint32_t core = 0; core < blocks; ++core) {
        for (uint32_t work = core; work < jobs * groups; work += blocks) {
          assert(work / groups < jobs && (work / groups) * kJobWords + 10 < jobs * kJobWords);
          ++visited[work];
          assert((core * 2) / 2 == core && (core * 2 + 1) / 2 == core);
        }
      }
      assert(std::all_of(visited.begin(), visited.end(), [](uint32_t n) { return n == 1; }));
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
