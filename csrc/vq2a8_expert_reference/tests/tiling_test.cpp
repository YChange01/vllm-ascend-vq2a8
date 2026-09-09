// Host-only exercise of the actual restored header; no CANN or NPU dependency.
#include "../mat_fp4_tiling.h"
#include <cassert>
#include <iostream>

template <typename Get> void CheckTiles(uint64_t total, uint64_t count, uint64_t maximum, Get get) {
    uint64_t covered = 0;
    for (uint64_t i = 0; i < count; ++i) {
        const auto tile = get(i);
        assert(tile.offset == covered && tile.rows > 0 && tile.rows <= maximum);
        assert(tile.offset % 16 == 0);
        covered += tile.rows;
    }
    assert(covered == total);
}
int main() {
    static_assert(sizeof(MatFP4TilingData) == 80, "ABI");
    assert(MatFP4NOnlyTileCount(0) == 0);
    for (uint64_t m = 1; m <= 8192; ++m) {
        CheckTiles(m, MatFP4NOnlyTileCount(m), 256,
                   [=](uint64_t i) { return MatFP4GetNOnlyTile(m, i, 64); });
        if (m < 480) {
            CheckTiles(m, MatFP4NarrowTileCount(m), 256,
                       [=](uint64_t i) { return MatFP4GetNarrowTile(m, i, 96); });
        } else {
            const auto first = MatFP4SplitNearHalf16(m);
            for (auto local : {first, m - first}) {
                assert(local >= 240);
                CheckTiles(local, MatFP4WideTileCount(local), 240,
                           [=](uint64_t i) { return MatFP4GetWideTile(local, i); });
            }
        }
    }
    std::cout << "CPU_TILING_PASS M=1..8192 (not device verification)\n";
}
