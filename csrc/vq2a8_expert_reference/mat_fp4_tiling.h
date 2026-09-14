// User-supplied internal reference; chat transcription restored. See README.md.
#ifndef MAT_FP4_TILING_H
#define MAT_FP4_TILING_H
#include <cstdint>

struct MatFP4TilingData {
    uint64_t narrow_nbs;
    uint64_t wide_nbs;
    uint64_t small_m_limit;
    uint64_t n_only_mode;
    uint64_t small_aiv_kbs;
    uint64_t small_aic_kbs;
    uint64_t small_mad_k;
    uint64_t middle_aiv_kbs;
    uint64_t middle_aic_kbs;
    uint64_t middle_mad_k;
};
constexpr uint64_t kMatFP4MaxTileM = 256;
constexpr uint64_t kMatFP4WideTileM = 240;
constexpr uint64_t kMatFP4WideMThreshold = 480;
constexpr uint64_t kMatFP4NarrowNCoreCount = 32;
constexpr uint64_t kMatFP4WideNCoreCount = 16;
constexpr uint64_t kMatFP4L1SlotBytes = 256 * 1024;
enum MatFP4TileMode : uint64_t { MAT_FP4_SMALL = 0, MAT_FP4_MIDDLE = 1, MAT_FP4_WIDE = 2 };
struct MatFP4TileDesc { uint64_t offset; uint64_t rows; uint64_t mode; };
#ifdef __aicore__
#define MAT_FP4_INLINE __aicore__ inline
#else
#define MAT_FP4_INLINE inline
#endif

// First half is the nearest multiple of 16 to total/2; ties choose the lower.
MAT_FP4_INLINE uint64_t MatFP4SplitNearHalf16(uint64_t total) {
    const uint64_t lower = total / 32 * 16;
    const uint64_t upper = lower + 16;
    const uint64_t lower_twice = lower * 2;
    const uint64_t upper_twice = upper * 2;
    const uint64_t lower_distance = total >= lower_twice ? total - lower_twice : lower_twice - total;
    const uint64_t upper_distance = total >= upper_twice ? total - upper_twice : upper_twice - total;
    return lower_distance <= upper_distance ? lower : upper;
}
MAT_FP4_INLINE uint64_t MatFP4NarrowTileCount(uint64_t group_m) {
    return group_m <= kMatFP4MaxTileM ? 1 : 2;
}
MAT_FP4_INLINE MatFP4TileDesc MatFP4GetNarrowTile(uint64_t group_m, uint64_t tile_id, uint64_t small_m_limit) {
    if (group_m <= kMatFP4MaxTileM) {
        return {0, group_m, group_m <= small_m_limit ? MAT_FP4_SMALL : MAT_FP4_MIDDLE};
    }
    const uint64_t first_m = MatFP4SplitNearHalf16(group_m);
    const uint64_t offset = tile_id == 0 ? 0 : first_m;
    const uint64_t rows = tile_id == 0 ? first_m : group_m - first_m;
    return {offset, rows, rows <= small_m_limit ? MAT_FP4_SMALL : MAT_FP4_MIDDLE};
}
// N-only mode: retain q-1 full M256 tiles, then split 256+r near half.
MAT_FP4_INLINE uint64_t MatFP4NOnlyTileCount(uint64_t group_m) {
    if (group_m == 0) return 0;
    if (group_m <= kMatFP4MaxTileM) return 1;
    const uint64_t quotient = group_m / kMatFP4MaxTileM;
    const uint64_t remainder = group_m % kMatFP4MaxTileM;
    return quotient - 1 + (remainder == 0 ? 1 : 2);
}
MAT_FP4_INLINE MatFP4TileDesc MatFP4GetNOnlyTile(uint64_t group_m, uint64_t tile_id, uint64_t small_m_limit) {
    if (group_m <= kMatFP4MaxTileM) {
        return {0, group_m, group_m <= small_m_limit ? MAT_FP4_SMALL : MAT_FP4_MIDDLE};
    }
    const uint64_t quotient = group_m / kMatFP4MaxTileM;
    const uint64_t remainder = group_m % kMatFP4MaxTileM;
    const uint64_t full_tile_count = quotient - 1;
    if (tile_id < full_tile_count) {
        return {tile_id * kMatFP4MaxTileM, kMatFP4MaxTileM, MAT_FP4_MIDDLE};
    }
    const uint64_t tail_offset = full_tile_count * kMatFP4MaxTileM;
    const uint64_t tail_m = group_m - tail_offset;
    if (remainder == 0) return {tail_offset, tail_m, MAT_FP4_MIDDLE};
    const uint64_t first_m = MatFP4SplitNearHalf16(tail_m);
    const uint64_t offset = tile_id == full_tile_count ? tail_offset : tail_offset + first_m;
    const uint64_t rows = tile_id == full_tile_count ? first_m : tail_m - first_m;
    return {offset, rows, rows <= small_m_limit ? MAT_FP4_SMALL : MAT_FP4_MIDDLE};
}
// Called only with local_m>=240 after splitting groups with M>=480.
MAT_FP4_INLINE uint64_t MatFP4WideTileCount(uint64_t local_m) {
    return local_m / kMatFP4WideTileM + 1;
}
MAT_FP4_INLINE MatFP4TileDesc MatFP4GetWideTile(uint64_t local_m, uint64_t tile_id) {
    const uint64_t full_tile_count = local_m / kMatFP4WideTileM - 1;
    if (tile_id < full_tile_count) {
        return {tile_id * kMatFP4WideTileM, kMatFP4WideTileM, MAT_FP4_WIDE};
    }
    const uint64_t tail_offset = full_tile_count * kMatFP4WideTileM;
    const uint64_t tail_m = local_m - tail_offset;
    const uint64_t first_m = MatFP4SplitNearHalf16(tail_m);
    if (tile_id == full_tile_count) return {tail_offset, first_m, MAT_FP4_WIDE};
    return {tail_offset + first_m, tail_m - first_m, MAT_FP4_WIDE};
}
#undef MAT_FP4_INLINE
#endif
