// Local adaptation of the user-supplied internal expert reference.
// No license grant for the original source is inferred. See ../vq2a8_expert_reference/README.md.
#pragma once
#include <cstdint>

namespace vq2a8_ascendc_v2 {
constexpr uint32_t kAbiVersion = 1;
constexpr uint32_t kMaxJobs = 6;
constexpr uint32_t kMaxM = 32;
constexpr uint32_t kN = 128;
constexpr uint32_t kAicK = 1024;
constexpr uint32_t kAivK = 512;
constexpr uint32_t kMadK = 256;
constexpr uint32_t kK0 = 16;
constexpr uint32_t kN0 = 32;
constexpr uint32_t kCodebookK = 256;
constexpr uint32_t kBuffers = 2;
constexpr uint32_t kPackedBytes = kN * kAivK / 4;
constexpr uint32_t kDecodedBytes = kN * kAivK / kK0 * (kK0 + 1);
// Register table loads read 256 bytes. Only the first 32 are indexed. Seven
// extra 32-byte blocks keep the final full-register read within its UB buffer.
constexpr uint32_t kTableBytes = ((kAivK / kCodebookK) * (kN / kN0) + 7) * 32;
constexpr uint32_t kResultBytes = (kMaxM / 2) * kN * sizeof(float);
constexpr uint32_t kRowBytes = (kMaxM / 2) * sizeof(float);
constexpr uint32_t kOutBytes = (kMaxM / 2) * kN * sizeof(uint16_t);
constexpr uint32_t kUbBytes =
    kResultBytes + kBuffers * (kPackedBytes + kDecodedBytes + kTableBytes) + 2 * kRowBytes + kOutBytes;
constexpr uint32_t kL1Bytes = kBuffers * (kMaxM + kN) * kAicK;
constexpr uint32_t kL0ABytes = kBuffers * kMaxM * kMadK;
constexpr uint32_t kL0BBytes = kBuffers * kN * kMadK;
constexpr uint32_t kL0CBytes = kMaxM * kN * sizeof(float);
static_assert(kUbBytes <= 256 * 1024 && kL1Bytes <= 512 * 1024, "VQ2A8 v2 tile exceeds UB/L1 capacity");
static_assert(kL0ABytes <= 64 * 1024 && kL0BBytes <= 64 * 1024 && kL0CBytes <= 256 * 1024,
              "VQ2A8 v2 tile exceeds L0 capacity");
static_assert(kAicK == 2 * kAivK && kAivK % kCodebookK == 0 && kAicK % kMadK == 0,
              "VQ2A8 v2 AIC/AIV/LUT reduction tiles disagree");

enum JobField : uint32_t {
  kX = 0,
  kScale = 1,
  kBias = 2,
  kPacked = 3,
  kTable = 4,
  kOutput = 5,
  kRows = 6,
  kColumns = 7,
  kReduction = 8,
  kJobWords = 9,
};

#ifndef VQ2_V2_LAYOUT_FN
  #define VQ2_V2_LAYOUT_FN constexpr
  #define VQ2_V2_UNDEF_LAYOUT_FN
#endif
VQ2_V2_LAYOUT_FN bool ValidDimensions(int64_t m, int64_t n, int64_t k) {
  // Deliberately bounded to the real gate/up and down shapes. The standalone
  // reference's N6144 and M>32 branches are not certified by this model path.
  return m >= 1 && m <= kMaxM && n == 4096 && (k == 2048 || k == 4096);
}
VQ2_V2_LAYOUT_FN uint32_t AlignedM(uint32_t m) { return (m + 15) / 16 * 16; }
VQ2_V2_LAYOUT_FN uint32_t HalfRows(uint32_t m, uint32_t half) {
  const uint32_t halfCapacity = AlignedM(m) / 2;
  const uint32_t begin = half * halfCapacity;
  return begin >= m ? 0 : ((m - begin < halfCapacity) ? m - begin : halfCapacity);
}
VQ2_V2_LAYOUT_FN uint64_t PackedOffset(uint32_t nBegin, uint32_t kBegin, uint32_t k) {
  return (uint64_t(nBegin / kN0) * (k / kK0) + kBegin / kK0) * kK0 * (kN0 / 4);
}
VQ2_V2_LAYOUT_FN uint64_t TableOffset(uint32_t nBegin, uint32_t kBegin, uint32_t n) {
  return (uint64_t(kBegin / kCodebookK) * (n / kN0) + nBegin / kN0) * 32;
}
VQ2_V2_LAYOUT_FN uint32_t DecodedOffset(uint32_t n1, uint32_t k1) {
  return (n1 * (kAivK / kK0) + k1) * (kK0 + 1) * kN0;
}
VQ2_V2_LAYOUT_FN uint32_t B1Offset(uint32_t n1, uint32_t half) { return (n1 * kAicK + half * kAivK) * kN0; }
#ifdef VQ2_V2_UNDEF_LAYOUT_FN
  #undef VQ2_V2_LAYOUT_FN
  #undef VQ2_V2_UNDEF_LAYOUT_FN
#endif
}  // namespace vq2a8_ascendc_v2
