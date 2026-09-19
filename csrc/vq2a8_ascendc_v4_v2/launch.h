// VQ2A8 kernel v2; derived from ../vq2a8_expert_reference, see its README.md.
#pragma once
#include <cstdint>

namespace vq2a8_ascendc_v4_v2 {
// Exactly one native MIX_AIC_1_2 launch, including FP32 scale/bias epilogue.
// All pointers are Torch-owned on the caller's current stream. No GM decoded B.
void LaunchGrouped(void* stream, uint32_t blocks, void* descriptors, uint32_t jobs, uint32_t nTiles);
void LaunchGroupedB1(void* stream, uint32_t blocks, void* descriptors, uint32_t jobs, uint32_t nTiles);
void LaunchResidentSelect(void* stream, uint32_t blocks, void* bank, void* routeIds, void* scale, void* bias,
                          void* sign, void* valid, uint32_t experts, uint32_t routes, uint32_t k);
void LaunchResidentPrepare(void* stream, uint32_t blocks, void* bank, void* routeIds, void* x, void* scale,
                           void* bias, void* reordered, void* descriptors, void* output, void* valid,
                           uint32_t experts, uint32_t routes, uint32_t m, uint32_t n, uint32_t k);
void LaunchResidentPrepareVectorized(void* stream, uint32_t blocks, void* bank, void* routeIds, void* x,
                                    void* scale, void* bias, void* reordered, void* descriptors,
                                    void* output, void* valid, uint32_t experts, uint32_t routes,
                                    uint32_t m, uint32_t n, uint32_t k);
void LaunchResidentPrepareTail(void* stream, uint32_t blocks, void* bank, void* routeIds, void* x,
                              void* scale, void* bias, void* reordered, void* descriptors,
                              void* output, void* valid, uint32_t experts, uint32_t routes,
                              uint32_t m, uint32_t n, uint32_t k);
void LaunchResidentPrepareRowReuse(void* stream, uint32_t blocks, void* bank, void* routeIds, void* x,
                                 void* scale, void* bias, void* reordered, void* descriptors,
                                 void* output, void* valid, uint32_t experts, uint32_t routes,
                                 uint32_t m, uint32_t n, uint32_t k);
void LaunchResidentPrepareChunkReuse(void* stream, uint32_t blocks, void* bank, void* routeIds, void* x,
                                   void* scale, void* bias, void* reordered, void* descriptors,
                                   void* output, void* valid, uint32_t experts, uint32_t routes,
                                   uint32_t m, uint32_t n, uint32_t k, uint32_t chunks);
}  // namespace vq2a8_ascendc_v4_v2
