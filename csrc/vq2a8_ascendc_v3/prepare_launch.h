// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cstdint>
#include "prepare_layout.h"

namespace vq2a8_v3 {
// All tensors are contiguous device storage retained by the caller. Each AIV
// owns a whole row; jobs and K obey ValidPrepareDimensions. No GM workspace.
// FP32 rotated/weightScale[jobs,K], int64 activationOrder[jobs,K], and FP32
// inputBias[jobs] produced by the original one-row RHT / bias GEMV path.
// Outputs: FP8 quantized[jobs,K], FP32 rowScale/outputBias[jobs], and int32
// valid[jobs] (1 = finite inputs/product/bias and in-range order; 0 = invalid).
// The byte gather does not check uniqueness: the resident weight conversion
// validates that activationOrder is a permutation before uploading its bank.
void LaunchPrepareV3(void* stream, uint32_t blocks, void* rotated, void* weightScale, void* activationOrder,
                     void* inputBias, void* quantized, void* rowScale, void* outputBias, void* valid, uint32_t jobs,
                     uint32_t k);
}  // namespace vq2a8_v3
