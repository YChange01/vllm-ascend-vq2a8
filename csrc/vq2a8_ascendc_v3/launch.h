// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cstdint>

namespace vq2a8_v3 {
// mode: 0=dense synthetic control, 1=sign-bit Vector/Cube bridge,
// 2=packed VQ projection. No workspace or decoded-GM-weight argument.
void Launch(void* stream, uint32_t blocks, void* x, void* scale, void* bias, void* packed, void* book, void* ids,
            void* denseControl, void* y, uint32_t m, uint32_t n, uint32_t k, uint32_t tiles, uint32_t mode);
void LaunchGrouped(void* stream, uint32_t blocks, void* descriptors, uint32_t jobs, uint32_t groups);
void LaunchGroupedPipeline(void* stream, uint32_t blocks, void* descriptors, uint32_t jobs, uint32_t groups);
// ABI 1: preallocated device int64[jobs,12] records and immutable device
// int32[6656] constants. The host binding validates/retains every tensor owner.
// No host-to-device descriptor, constant or weight copies on this path.
void LaunchGroupedV3(void* stream, uint32_t blocks, void* descriptors, void* constants, uint32_t jobs, uint32_t groups,
                     bool pipeline);
}  // namespace vq2a8_v3
