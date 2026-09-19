// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <ATen/ATen.h>
#include <cstdint>
#include <vector>

namespace vq2a8_ascendc_v4_v2 {
// Internal ResidentBank entry, never expose raw pointer-table tensors to Python.
std::vector<at::Tensor> ResidentSwigluSelectSign(
    const at::Tensor& hidden, const at::Tensor& ids, const at::Tensor& table,
    const std::vector<at::Tensor>& scaleOwners, const std::vector<at::Tensor>& biasOwners,
    const std::vector<at::Tensor>& signOwners, uint32_t experts, uint32_t width, void* bankStream, double limit);
}  // namespace vq2a8_ascendc_v4_v2
