// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <torch/extension.h>
#include <torch/library.h>

#include <acl/acl_rt.h>
#include <c10/util/string_view.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>

#include "vq2a8/direct/vq2a8_kernel.h"

namespace vllm_ascend {
namespace {

constexpr int64_t kVq2VectorLength = 2;
constexpr int64_t kCodesPerPackedWord = 8;
constexpr int64_t kOutputTile = 32;
constexpr int64_t kMaximumInputWidth = 4096;

uint32_t checked_u32(int64_t value, const char* name)
{
    TORCH_CHECK(
        value >= 0 && value <= std::numeric_limits<uint32_t>::max(),
        "VQ2A8 ", name, " does not fit uint32: ", value);
    return static_cast<uint32_t>(value);
}

void check_npu_contiguous(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(
        tensor.is_privateuseone(),
        "VQ2A8 ", name, " must be an NPU tensor, got ", tensor.device());
    TORCH_CHECK(tensor.is_contiguous(), "VQ2A8 ", name, " must be contiguous.");
}

void check_same_device(const at::Tensor& reference, const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(
        tensor.device() == reference.device(),
        "VQ2A8 ", name, " must be on ", reference.device(), ", got ", tensor.device());
}

void check_transform_payload(
    const at::Tensor& x,
    const at::Tensor& expert_ids,
    const at::Tensor& weight_scale,
    const at::Tensor& weight_bias,
    const at::Tensor& rht_sign,
    int64_t rht_block_size)
{
    check_npu_contiguous(x, "x");
    check_npu_contiguous(expert_ids, "expert_ids");
    check_npu_contiguous(weight_scale, "weight_scale");
    check_npu_contiguous(weight_bias, "weight_bias");
    check_npu_contiguous(rht_sign, "rht_sign");
    check_same_device(x, expert_ids, "expert_ids");
    check_same_device(x, weight_scale, "weight_scale");
    check_same_device(x, weight_bias, "weight_bias");
    check_same_device(x, rht_sign, "rht_sign");

    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "VQ2A8 x must use bfloat16.");
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt, "VQ2A8 expert_ids must use int32.");
    TORCH_CHECK(weight_scale.scalar_type() == at::kFloat, "VQ2A8 weight_scale must use float32.");
    TORCH_CHECK(weight_bias.scalar_type() == at::kFloat, "VQ2A8 weight_bias must use float32.");
    TORCH_CHECK(rht_sign.scalar_type() == at::kChar, "VQ2A8 rht_sign must use int8.");
    TORCH_CHECK(x.dim() == 2, "VQ2A8 x must have shape [assignments, input].");
    TORCH_CHECK(expert_ids.dim() == 1 && expert_ids.size(0) == x.size(0),
                "VQ2A8 expert_ids must contain one ID per input row.");
    TORCH_CHECK(weight_scale.dim() == 2, "VQ2A8 weight_scale must have shape [experts, input].");
    TORCH_CHECK(weight_bias.sizes() == weight_scale.sizes(),
                "VQ2A8 weight_bias shape must match weight_scale.");
    TORCH_CHECK(rht_sign.sizes() == weight_scale.sizes(),
                "VQ2A8 rht_sign shape must match weight_scale.");
    TORCH_CHECK(weight_scale.size(1) == x.size(1),
                "VQ2A8 transform width must match the input width.");
    TORCH_CHECK(x.size(1) > 0 && x.size(1) <= kMaximumInputWidth,
                "VQ2A8 direct kernel supports input widths in [1, ", kMaximumInputWidth,
                "], got ", x.size(1));
    TORCH_CHECK(rht_block_size == 128,
                "VQ2A8 Ascend 950 direct kernels require rht_block_size=128, got ",
                rht_block_size);
    TORCH_CHECK(x.size(1) % rht_block_size == 0,
                "VQ2A8 input width must be divisible by rht_block_size.");
    TORCH_CHECK(x.size(1) % kOutputTile == 0,
                "VQ2A8 input width must be divisible by 32 for FP8 vector operations.");
}

struct LookupShape {
    int64_t num_experts;
    int64_t output_size;
    int64_t num_codebook_tiles;
    int64_t row_tiles;
};

LookupShape check_lookup_payload(
    const at::Tensor& x,
    const at::Tensor& packed_indices,
    const at::Tensor& codebooks,
    const at::Tensor& codebook_tile_ids,
    int64_t row_group_size)
{
    check_npu_contiguous(packed_indices, "packed_indices");
    check_npu_contiguous(codebooks, "codebooks");
    check_npu_contiguous(codebook_tile_ids, "codebook_tile_ids");
    check_same_device(x, packed_indices, "packed_indices");
    check_same_device(x, codebooks, "codebooks");
    check_same_device(x, codebook_tile_ids, "codebook_tile_ids");
    TORCH_CHECK(packed_indices.scalar_type() == at::kInt,
                "VQ2A8 packed_indices must use int32.");
    TORCH_CHECK(codebooks.scalar_type() == at::ScalarType::Float8_e4m3fn,
                "VQ2A8 codebooks must use float8_e4m3fn.");
    TORCH_CHECK(codebook_tile_ids.scalar_type() == at::kByte,
                "VQ2A8 codebook_tile_ids must use uint8.");
    TORCH_CHECK(packed_indices.dim() == 3,
                "VQ2A8 packed_indices must have shape [experts, output_pairs, words].");
    TORCH_CHECK(codebooks.dim() == 5,
                "VQ2A8 codebooks must have shape [experts, column_tiles, row_tiles, 16, 2].");
    TORCH_CHECK(codebook_tile_ids.dim() == 2,
                "VQ2A8 codebook_tile_ids must have shape [experts, input].");
    TORCH_CHECK(codebooks.size(0) == packed_indices.size(0) &&
                    codebook_tile_ids.size(0) == packed_indices.size(0),
                "VQ2A8 payload expert dimensions must agree.");
    TORCH_CHECK(codebook_tile_ids.size(1) == x.size(1),
                "VQ2A8 codebook tile width must match the input width.");
    TORCH_CHECK(packed_indices.size(2) * kCodesPerPackedWord >= x.size(1),
                "VQ2A8 packed indices do not cover the input width.");
    TORCH_CHECK(codebooks.size(3) == 16 && codebooks.size(4) == kVq2VectorLength,
                "VQ2A8 codebooks require geometry [..., 16, 2].");
    TORCH_CHECK(row_group_size > 0 && row_group_size % kVq2VectorLength == 0,
                "VQ2A8 row_group_size must be positive and even.");
    const int64_t output_size = packed_indices.size(1) * kVq2VectorLength;
    TORCH_CHECK(codebooks.size(2) * row_group_size == output_size,
                "VQ2A8 codebook row tiles do not cover the output width.");
    return {
        packed_indices.size(0),
        output_size,
        codebooks.size(1),
        codebooks.size(2),
    };
}

struct PreparedActivation {
    at::Tensor quantized;
    at::Tensor scale;
    at::Tensor bias;
};

PreparedActivation prepare_activation(
    const at::Tensor& x,
    const at::Tensor& expert_ids,
    const at::Tensor& weight_scale,
    const at::Tensor& weight_bias,
    const at::Tensor& rht_sign,
    int64_t rht_block_size)
{
    const int64_t size_m = x.size(0);
    const int64_t size_k = x.size(1);
    const int64_t num_rht_blocks = size_k / rht_block_size;
    const auto float_options = x.options().dtype(at::kFloat);
    auto quantized = at::empty(
        {size_m, size_k}, x.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto activation_scale = at::empty({size_m}, float_options);
    auto bias_correction = at::empty({size_m}, float_options);
    if (size_m == 0) {
        return {quantized, activation_scale, bias_correction};
    }

    auto transformed = at::empty({size_m, size_k}, float_options);
    auto partial_amax = at::empty({size_m, num_rht_blocks}, float_options);
    auto partial_bias = at::empty({size_m, num_rht_blocks}, float_options);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    vq2a8_prepare_impl(
        stream,
        x.data_ptr(),
        expert_ids.data_ptr(),
        weight_scale.data_ptr(),
        weight_bias.data_ptr(),
        rht_sign.data_ptr(),
        transformed.data_ptr(),
        partial_amax.data_ptr(),
        partial_bias.data_ptr(),
        quantized.data_ptr(),
        activation_scale.data_ptr(),
        bias_correction.data_ptr(),
        checked_u32(size_m, "assignment count"),
        checked_u32(size_k, "input width"),
        checked_u32(rht_block_size, "RHT block size"));
    return {quantized, activation_scale, bias_correction};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> vq2a8_prepare_debug(
    const at::Tensor& x,
    const at::Tensor& expert_ids,
    const at::Tensor& weight_scale,
    const at::Tensor& weight_bias,
    const at::Tensor& rht_sign,
    int64_t rht_block_size)
{
    check_transform_payload(
        x, expert_ids, weight_scale, weight_bias, rht_sign, rht_block_size);
    const PreparedActivation prepared = prepare_activation(
        x, expert_ids, weight_scale, weight_bias, rht_sign, rht_block_size);
    return {prepared.quantized, prepared.scale, prepared.bias};
}

at::Tensor vq2a8_gate_up(
    const at::Tensor& x,
    const at::Tensor& expert_ids,
    const at::Tensor& packed_indices,
    const at::Tensor& codebooks,
    const at::Tensor& codebook_tile_ids,
    const at::Tensor& weight_scale,
    const at::Tensor& weight_bias,
    const at::Tensor& rht_sign,
    int64_t rht_block_size,
    int64_t row_group_size,
    c10::string_view activation,
    double swiglu_limit)
{
    check_transform_payload(
        x, expert_ids, weight_scale, weight_bias, rht_sign, rht_block_size);
    const LookupShape shape = check_lookup_payload(
        x, packed_indices, codebooks, codebook_tile_ids, row_group_size);
    const std::string activation_name(activation);
    TORCH_CHECK(activation_name == "silu" || activation_name == "swiglu",
                "VQ2A8 gate_up supports only silu/swiglu, got ", activation_name);
    TORCH_CHECK(std::isfinite(swiglu_limit) && swiglu_limit >= 0.0,
                "VQ2A8 swiglu_limit must be finite and non-negative, got ", swiglu_limit);
    TORCH_CHECK(shape.output_size % 2 == 0,
                "VQ2A8 gate/up projected width must be even.");
    const int64_t intermediate_size = shape.output_size / 2;
    TORCH_CHECK(intermediate_size % kOutputTile == 0,
                "VQ2A8 intermediate width must be divisible by 32.");
    auto output = at::empty({x.size(0), intermediate_size}, x.options());
    if (x.size(0) == 0) {
        return output;
    }
    const PreparedActivation prepared = prepare_activation(
        x, expert_ids, weight_scale, weight_bias, rht_sign, rht_block_size);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    vq2a8_gate_up_impl(
        stream,
        prepared.quantized.data_ptr(),
        prepared.scale.data_ptr(),
        prepared.bias.data_ptr(),
        expert_ids.data_ptr(),
        packed_indices.data_ptr(),
        codebooks.data_ptr(),
        codebook_tile_ids.data_ptr(),
        output.data_ptr(),
        checked_u32(x.size(0), "assignment count"),
        checked_u32(intermediate_size, "intermediate width"),
        checked_u32(x.size(1), "input width"),
        checked_u32(shape.num_experts, "expert count"),
        checked_u32(shape.num_codebook_tiles, "codebook tile count"),
        checked_u32(shape.row_tiles, "row tile count"),
        checked_u32(row_group_size, "row group size"),
        static_cast<float>(swiglu_limit));
    return output;
}

at::Tensor vq2a8_down_reduce(
    const at::Tensor& x,
    const at::Tensor& expert_ids,
    const at::Tensor& token_ids,
    const at::Tensor& routing_weights,
    const at::Tensor& packed_indices,
    const at::Tensor& codebooks,
    const at::Tensor& codebook_tile_ids,
    const at::Tensor& weight_scale,
    const at::Tensor& weight_bias,
    const at::Tensor& rht_sign,
    int64_t rht_block_size,
    int64_t row_group_size,
    int64_t num_tokens)
{
    check_transform_payload(
        x, expert_ids, weight_scale, weight_bias, rht_sign, rht_block_size);
    const LookupShape shape = check_lookup_payload(
        x, packed_indices, codebooks, codebook_tile_ids, row_group_size);
    check_npu_contiguous(token_ids, "token_ids");
    check_npu_contiguous(routing_weights, "routing_weights");
    check_same_device(x, token_ids, "token_ids");
    check_same_device(x, routing_weights, "routing_weights");
    TORCH_CHECK(token_ids.scalar_type() == at::kLong,
                "VQ2A8 token_ids must use int64.");
    TORCH_CHECK(routing_weights.scalar_type() == at::kFloat,
                "VQ2A8 routing_weights must use float32.");
    TORCH_CHECK(token_ids.dim() == 1 && token_ids.size(0) == x.size(0),
                "VQ2A8 token_ids must contain one ID per input row.");
    TORCH_CHECK(routing_weights.dim() == 1 && routing_weights.size(0) == x.size(0),
                "VQ2A8 routing_weights must contain one value per input row.");
    TORCH_CHECK(num_tokens >= 0, "VQ2A8 num_tokens must be non-negative.");
    TORCH_CHECK(shape.output_size % kOutputTile == 0,
                "VQ2A8 down output width must be divisible by 32.");
    auto accumulator = at::zeros(
        {num_tokens, shape.output_size}, x.options().dtype(at::kFloat));
    if (x.size(0) == 0 || num_tokens == 0) {
        return accumulator.to(x.scalar_type());
    }
    const PreparedActivation prepared = prepare_activation(
        x, expert_ids, weight_scale, weight_bias, rht_sign, rht_block_size);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    vq2a8_down_reduce_impl(
        stream,
        prepared.quantized.data_ptr(),
        prepared.scale.data_ptr(),
        prepared.bias.data_ptr(),
        expert_ids.data_ptr(),
        token_ids.data_ptr(),
        routing_weights.data_ptr(),
        packed_indices.data_ptr(),
        codebooks.data_ptr(),
        codebook_tile_ids.data_ptr(),
        accumulator.data_ptr(),
        checked_u32(x.size(0), "assignment count"),
        checked_u32(shape.output_size, "output width"),
        checked_u32(x.size(1), "input width"),
        checked_u32(num_tokens, "token count"),
        checked_u32(shape.num_experts, "expert count"),
        checked_u32(shape.num_codebook_tiles, "codebook tile count"),
        checked_u32(shape.row_tiles, "row tile count"),
        checked_u32(row_group_size, "row group size"));
    return accumulator.to(x.scalar_type());
}

namespace meta {

std::tuple<at::Tensor, at::Tensor, at::Tensor> vq2a8_prepare_debug_meta(
    const at::Tensor& x,
    const at::Tensor& expert_ids,
    const at::Tensor& weight_scale,
    const at::Tensor& weight_bias,
    const at::Tensor& rht_sign,
    int64_t rht_block_size)
{
    (void)expert_ids;
    (void)weight_scale;
    (void)weight_bias;
    (void)rht_sign;
    (void)rht_block_size;
    auto quantized = at::empty_symint(
        {x.sym_size(0), x.sym_size(1)},
        x.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto scale = at::empty_symint(
        {x.sym_size(0)}, x.options().dtype(at::kFloat));
    auto bias = at::empty_symint(
        {x.sym_size(0)}, x.options().dtype(at::kFloat));
    return {quantized, scale, bias};
}

at::Tensor vq2a8_gate_up_meta(
    const at::Tensor& x,
    const at::Tensor& expert_ids,
    const at::Tensor& packed_indices,
    const at::Tensor& codebooks,
    const at::Tensor& codebook_tile_ids,
    const at::Tensor& weight_scale,
    const at::Tensor& weight_bias,
    const at::Tensor& rht_sign,
    int64_t rht_block_size,
    int64_t row_group_size,
    c10::string_view activation,
    double swiglu_limit)
{
    (void)expert_ids;
    (void)codebooks;
    (void)codebook_tile_ids;
    (void)weight_scale;
    (void)weight_bias;
    (void)rht_sign;
    (void)rht_block_size;
    (void)row_group_size;
    (void)activation;
    (void)swiglu_limit;
    return at::empty_symint({x.sym_size(0), packed_indices.sym_size(1)}, x.options());
}

at::Tensor vq2a8_down_reduce_meta(
    const at::Tensor& x,
    const at::Tensor& expert_ids,
    const at::Tensor& token_ids,
    const at::Tensor& routing_weights,
    const at::Tensor& packed_indices,
    const at::Tensor& codebooks,
    const at::Tensor& codebook_tile_ids,
    const at::Tensor& weight_scale,
    const at::Tensor& weight_bias,
    const at::Tensor& rht_sign,
    int64_t rht_block_size,
    int64_t row_group_size,
    int64_t num_tokens)
{
    (void)expert_ids;
    (void)token_ids;
    (void)routing_weights;
    (void)codebooks;
    (void)codebook_tile_ids;
    (void)weight_scale;
    (void)weight_bias;
    (void)rht_sign;
    (void)rht_block_size;
    (void)row_group_size;
    return at::empty_symint(
        {c10::SymInt(num_tokens), packed_indices.sym_size(1) * kVq2VectorLength},
        x.options());
}

}  // namespace meta
}  // namespace vllm_ascend

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)
{
    ops.def(
        "vq2a8_prepare_debug(Tensor x, Tensor expert_ids, Tensor weight_scale, "
        "Tensor weight_bias, Tensor rht_sign, int rht_block_size) "
        "-> (Tensor, Tensor, Tensor)");
    ops.impl(
        "vq2a8_prepare_debug",
        torch::kPrivateUse1,
        &vllm_ascend::vq2a8_prepare_debug);
    ops.def(
        "vq2a8_gate_up(Tensor x, Tensor expert_ids, Tensor packed_indices, "
        "Tensor codebooks, Tensor codebook_tile_ids, Tensor weight_scale, "
        "Tensor weight_bias, Tensor rht_sign, int rht_block_size, "
        "int row_group_size, str activation, float swiglu_limit) -> Tensor");
    ops.impl("vq2a8_gate_up", torch::kPrivateUse1, &vllm_ascend::vq2a8_gate_up);
    ops.def(
        "vq2a8_down_reduce(Tensor x, Tensor expert_ids, Tensor token_ids, "
        "Tensor routing_weights, Tensor packed_indices, Tensor codebooks, "
        "Tensor codebook_tile_ids, Tensor weight_scale, Tensor weight_bias, "
        "Tensor rht_sign, int rht_block_size, int row_group_size, "
        "int num_tokens) -> Tensor");
    ops.impl("vq2a8_down_reduce", torch::kPrivateUse1, &vllm_ascend::vq2a8_down_reduce);
}

TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops)
{
    ops.impl("vq2a8_prepare_debug", &vllm_ascend::meta::vq2a8_prepare_debug_meta);
    ops.impl("vq2a8_gate_up", &vllm_ascend::meta::vq2a8_gate_up_meta);
    ops.impl("vq2a8_down_reduce", &vllm_ascend::meta::vq2a8_down_reduce_meta);
}
