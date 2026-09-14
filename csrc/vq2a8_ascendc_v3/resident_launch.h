// VQ2A8 v3 resident kernel; derived from ../vq2a8_expert_reference, see its README.md.
#pragma once
#include <cstdint>

namespace vq2a8_v3_resident {
// Exactly one native MIX_AIC_1_2 launch, including FP32 scale/bias epilogue.
// All pointers are Torch-owned on the caller's current stream. No GM decoded B.
void LaunchGrouped(void* stream, uint32_t blocks, void* descriptors, uint32_t jobs, uint32_t nTiles);
}  // namespace vq2a8_v3_resident
