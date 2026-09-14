// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cstdint>

namespace vq2a8_ascendc {
// mode: 0=dense synthetic control, 1=sign-bit Vector/Cube bridge,
// 2=packed VQ projection. No workspace or decoded-GM-weight argument.
void Launch(void* stream, uint32_t blocks, void* x, void* scale, void* bias, void* packed, void* book, void* ids,
            void* denseControl, void* y, uint32_t m, uint32_t n, uint32_t k, uint32_t tiles, uint32_t mode);
void LaunchGrouped(void* stream, uint32_t blocks, void* descriptors, uint32_t jobs, uint32_t groups);
void LaunchGroupedPipeline(void* stream, uint32_t blocks, void* descriptors, uint32_t jobs, uint32_t groups);
// Opt-in B1 decode: device-selected immutable bank, never host route IDs or
// per-request host descriptors. Every route is an independent M=1 V1 job.
void LaunchResident(void* stream, uint32_t blocks, void* bank, void* routeIds, void* x, void* scale, void* bias,
                    void* output, void* valid, uint32_t experts, uint32_t routes, uint32_t n, uint32_t k);
void LaunchResidentSelect(void* stream, uint32_t blocks, void* bank, void* routeIds, void* scale, void* bias,
                          void* sign, void* valid, uint32_t experts, uint32_t routes, uint32_t k);
}  // namespace vq2a8_ascendc
