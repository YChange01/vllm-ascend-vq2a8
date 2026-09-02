// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#include <cstdint>

namespace vllm_ascend {

void vq2a8_prepare_impl(
    void* stream,
    void* x,
    void* expert_ids,
    void* weight_scale,
    void* weight_bias,
    void* rht_sign,
    void* transformed,
    void* partial_amax,
    void* partial_bias,
    void* quantized,
    void* activation_scale,
    void* bias_correction,
    uint32_t size_m,
    uint32_t size_k,
    uint32_t rht_block_size);

void vq2a8_gate_up_impl(
    void* stream,
    void* quantized,
    void* activation_scale,
    void* bias_correction,
    void* expert_ids,
    void* packed_indices,
    void* codebooks,
    void* codebook_tile_ids,
    void* output,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k,
    uint32_t num_experts,
    uint32_t num_codebook_tiles,
    uint32_t row_tiles,
    uint32_t row_group_size,
    float swiglu_limit);

void vq2a8_down_reduce_impl(
    void* stream,
    void* quantized,
    void* activation_scale,
    void* bias_correction,
    void* expert_ids,
    void* token_ids,
    void* routing_weights,
    void* packed_indices,
    void* codebooks,
    void* codebook_tile_ids,
    void* output,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k,
    uint32_t num_tokens,
    uint32_t num_experts,
    uint32_t num_codebook_tiles,
    uint32_t row_tiles,
    uint32_t row_group_size);

}  // namespace vllm_ascend
