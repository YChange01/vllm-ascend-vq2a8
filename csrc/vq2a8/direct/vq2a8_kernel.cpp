// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "kernel_operator.h"

#include "vq2a8_kernel.h"

namespace {

constexpr uint32_t kOutputTile = 32;
constexpr float kFp8Max = 448.0F;
constexpr float kMinScale = 1.0e-12F;

class VQ2A8TransformKernel {
public:
    __aicore__ inline VQ2A8TransformKernel(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(
        __gm__ void* x,
        __gm__ void* expertIds,
        __gm__ void* weightScale,
        __gm__ void* weightBias,
        __gm__ void* rhtSign,
        __gm__ void* transformed,
        __gm__ void* partialAmax,
        __gm__ void* partialBias,
        uint32_t sizeM,
        uint32_t sizeK,
        uint32_t rhtBlockSize)
    {
        sizeM_ = sizeM;
        sizeK_ = sizeK;
        rhtBlockSize_ = rhtBlockSize;
        numRhtBlocks_ = sizeK / rhtBlockSize;

        xGm_.SetGlobalBuffer((__gm__ bfloat16_t*)x);
        expertIdsGm_.SetGlobalBuffer((__gm__ int32_t*)expertIds);
        weightScaleGm_.SetGlobalBuffer((__gm__ float*)weightScale);
        weightBiasGm_.SetGlobalBuffer((__gm__ float*)weightBias);
        rhtSignGm_.SetGlobalBuffer((__gm__ int8_t*)rhtSign);
        transformedGm_.SetGlobalBuffer((__gm__ float*)transformed);
        partialAmaxGm_.SetGlobalBuffer((__gm__ float*)partialAmax);
        partialBiasGm_.SetGlobalBuffer((__gm__ float*)partialBias);
        pipe_->InitBuffer(inputBf16Buffer_, rhtBlockSize * sizeof(bfloat16_t));
        pipe_->InitBuffer(valuesBuffer_, rhtBlockSize * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIndex = AscendC::GetBlockIdx();
        const uint32_t assignment = blockIndex / numRhtBlocks_;
        if (assignment >= sizeM_) {
            return;
        }
        const uint32_t rhtBlock = blockIndex % numRhtBlocks_;
        const uint32_t expert = static_cast<uint32_t>(expertIdsGm_.GetValue(assignment));
        const uint32_t columnStart = rhtBlock * rhtBlockSize_;
        const uint64_t xOffset = static_cast<uint64_t>(assignment) * sizeK_ + columnStart;
        const uint64_t expertOffset = static_cast<uint64_t>(expert) * sizeK_ + columnStart;
        AscendC::LocalTensor<bfloat16_t> inputBf16 = inputBf16Buffer_.Get<bfloat16_t>();
        AscendC::LocalTensor<float> values = valuesBuffer_.Get<float>();
        AscendC::DataCopy(inputBf16, xGm_[xOffset], rhtBlockSize_);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Cast(values, inputBf16, AscendC::RoundMode::CAST_NONE, rhtBlockSize_);
        // The FWHT below reads the cast result through scalar GetValue calls.
        // PIPE_V only orders vector instructions; it does not make vector
        // writes visible to the scalar pipeline.
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);

        for (uint32_t column = 0; column < rhtBlockSize_; ++column) {
            const float value = values.GetValue(column);
            const float sign = static_cast<float>(rhtSignGm_.GetValue(expertOffset + column));
            values.SetValue(column, value * sign);
        }

        for (uint32_t stride = 1; stride < rhtBlockSize_; stride <<= 1) {
            const uint32_t groupWidth = stride << 1;
            for (uint32_t group = 0; group < rhtBlockSize_; group += groupWidth) {
                for (uint32_t column = 0; column < stride; ++column) {
                    const uint32_t low = group + column;
                    const uint32_t high = low + stride;
                    const float lowValue = values.GetValue(low);
                    const float highValue = values.GetValue(high);
                    values.SetValue(low, lowValue + highValue);
                    values.SetValue(high, lowValue - highValue);
                }
            }
        }

        const float inverseNorm = 1.0F / sqrt(static_cast<float>(rhtBlockSize_));
        float amax = 0.0F;
        float bias = 0.0F;
        for (uint32_t column = 0; column < rhtBlockSize_; ++column) {
            const float rotated = values.GetValue(column) * inverseNorm;
            const float scaled = rotated * weightScaleGm_.GetValue(expertOffset + column);
            transformedGm_.SetValue(xOffset + column, scaled);
            const float magnitude = scaled < 0.0F ? -scaled : scaled;
            amax = magnitude > amax ? magnitude : amax;
            bias += rotated * weightBiasGm_.GetValue(expertOffset + column);
        }
        const uint64_t partialOffset = static_cast<uint64_t>(assignment) * numRhtBlocks_ + rhtBlock;
        partialAmaxGm_.SetValue(partialOffset, amax);
        partialBiasGm_.SetValue(partialOffset, bias);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> inputBf16Buffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> valuesBuffer_;
    AscendC::GlobalTensor<bfloat16_t> xGm_;
    AscendC::GlobalTensor<int32_t> expertIdsGm_;
    AscendC::GlobalTensor<float> weightScaleGm_;
    AscendC::GlobalTensor<float> weightBiasGm_;
    AscendC::GlobalTensor<int8_t> rhtSignGm_;
    AscendC::GlobalTensor<float> transformedGm_;
    AscendC::GlobalTensor<float> partialAmaxGm_;
    AscendC::GlobalTensor<float> partialBiasGm_;
    uint32_t sizeM_ = 0;
    uint32_t sizeK_ = 0;
    uint32_t rhtBlockSize_ = 0;
    uint32_t numRhtBlocks_ = 0;
};

class VQ2A8QuantizeKernel {
public:
    __aicore__ inline VQ2A8QuantizeKernel(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(
        __gm__ void* transformed,
        __gm__ void* partialAmax,
        __gm__ void* partialBias,
        __gm__ void* quantized,
        __gm__ void* activationScale,
        __gm__ void* biasCorrection,
        uint32_t sizeM,
        uint32_t sizeK,
        uint32_t rhtBlockSize)
    {
        sizeM_ = sizeM;
        sizeK_ = sizeK;
        numRhtBlocks_ = sizeK / rhtBlockSize;
        transformedGm_.SetGlobalBuffer((__gm__ float*)transformed);
        partialAmaxGm_.SetGlobalBuffer((__gm__ float*)partialAmax);
        partialBiasGm_.SetGlobalBuffer((__gm__ float*)partialBias);
        quantizedGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t*)quantized);
        activationScaleGm_.SetGlobalBuffer((__gm__ float*)activationScale);
        biasCorrectionGm_.SetGlobalBuffer((__gm__ float*)biasCorrection);
        pipe_->InitBuffer(floatBuffer_, sizeK * sizeof(float));
        pipe_->InitBuffer(fp8Buffer_, sizeK * sizeof(fp8_e4m3fn_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t assignment = AscendC::GetBlockIdx();
        if (assignment >= sizeM_) {
            return;
        }
        const uint64_t partialOffset = static_cast<uint64_t>(assignment) * numRhtBlocks_;
        float amax = 0.0F;
        float bias = 0.0F;
        for (uint32_t block = 0; block < numRhtBlocks_; ++block) {
            const float blockAmax = partialAmaxGm_.GetValue(partialOffset + block);
            amax = blockAmax > amax ? blockAmax : amax;
            bias += partialBiasGm_.GetValue(partialOffset + block);
        }
        const float unboundedScale = amax / kFp8Max;
        const float scale = unboundedScale > kMinScale ? unboundedScale : kMinScale;
        activationScaleGm_.SetValue(assignment, scale);
        biasCorrectionGm_.SetValue(assignment, bias);

        const uint64_t rowOffset = static_cast<uint64_t>(assignment) * sizeK_;
        AscendC::LocalTensor<float> values = floatBuffer_.Get<float>();
        AscendC::LocalTensor<fp8_e4m3fn_t> fp8Values = fp8Buffer_.Get<fp8_e4m3fn_t>();
        AscendC::DataCopy(values, transformedGm_[rowOffset], sizeK_);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Muls(values, values, 1.0F / scale, sizeK_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Maxs(values, values, -kFp8Max, sizeK_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(values, values, kFp8Max, sizeK_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(fp8Values, values, AscendC::RoundMode::CAST_RINT, sizeK_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopy(quantizedGm_[rowOffset], fp8Values, sizeK_);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> floatBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> fp8Buffer_;
    AscendC::GlobalTensor<float> transformedGm_;
    AscendC::GlobalTensor<float> partialAmaxGm_;
    AscendC::GlobalTensor<float> partialBiasGm_;
    AscendC::GlobalTensor<fp8_e4m3fn_t> quantizedGm_;
    AscendC::GlobalTensor<float> activationScaleGm_;
    AscendC::GlobalTensor<float> biasCorrectionGm_;
    uint32_t sizeM_ = 0;
    uint32_t sizeK_ = 0;
    uint32_t numRhtBlocks_ = 0;
};

class VQ2A8LookupMatmulBase {
protected:
    __aicore__ inline void InitLookupBuffers(AscendC::TPipe* pipe, uint32_t sizeK)
    {
        pipe->InitBuffer(activationFp8Buffer_, sizeK * sizeof(fp8_e4m3fn_t));
        pipe->InitBuffer(activationFloatBuffer_, sizeK * sizeof(float));
        pipe->InitBuffer(weightFp8Buffer_, sizeK * sizeof(fp8_e4m3fn_t));
        pipe->InitBuffer(weightFloatBuffer_, sizeK * sizeof(float));
        pipe->InitBuffer(productBuffer_, sizeK * sizeof(float));
        pipe->InitBuffer(reduceBuffer_, sizeK * sizeof(float));
        // Keep ReduceSum's destination disjoint from its source instead of
        // relying on backend-specific alias handling in the CANN 9.1 path.
        pipe->InitBuffer(sumBuffer_, kOutputTile * sizeof(float));
    }

    __aicore__ inline void LoadActivation(uint32_t assignment, uint32_t sizeK)
    {
        AscendC::LocalTensor<fp8_e4m3fn_t> activationFp8 = activationFp8Buffer_.Get<fp8_e4m3fn_t>();
        AscendC::LocalTensor<float> activationFloat = activationFloatBuffer_.Get<float>();
        const uint64_t activationOffset = static_cast<uint64_t>(assignment) * sizeK;
        AscendC::DataCopy(activationFp8, quantizedGm_[activationOffset], sizeK);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Cast(activationFloat, activationFp8, AscendC::RoundMode::CAST_NONE, sizeK);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float Dot(
        uint32_t expert,
        uint32_t outputColumn,
        uint32_t sizeK,
        uint32_t packedRows,
        uint32_t packedWords,
        uint32_t numCodebookTiles,
        uint32_t rowTiles,
        uint32_t rowGroupSize)
    {
        AscendC::LocalTensor<fp8_e4m3fn_t> weightFp8 = weightFp8Buffer_.Get<fp8_e4m3fn_t>();
        AscendC::LocalTensor<float> weightFloat = weightFloatBuffer_.Get<float>();
        AscendC::LocalTensor<float> activationFloat = activationFloatBuffer_.Get<float>();
        AscendC::LocalTensor<float> product = productBuffer_.Get<float>();
        AscendC::LocalTensor<float> reduce = reduceBuffer_.Get<float>();
        AscendC::LocalTensor<float> sum = sumBuffer_.Get<float>();
        const uint32_t vectorRow = outputColumn >> 1;
        const uint32_t component = outputColumn & 1U;
        const uint32_t outputTile = outputColumn / rowGroupSize;
        const uint64_t packedExpertOffset = static_cast<uint64_t>(expert) * packedRows * packedWords;
        const uint64_t tileExpertOffset = static_cast<uint64_t>(expert) * sizeK;

        for (uint32_t column = 0; column < sizeK; ++column) {
            const int32_t word = packedIndicesGm_.GetValue(
                packedExpertOffset + static_cast<uint64_t>(vectorRow) * packedWords + column / 8);
            const uint32_t code = (static_cast<uint32_t>(word) >> ((column & 7U) * 4U)) & 15U;
            const uint32_t codebookTile = static_cast<uint32_t>(
                codebookTileIdsGm_.GetValue(tileExpertOffset + column));
            const uint64_t codebookOffset =
                (((static_cast<uint64_t>(expert) * numCodebookTiles + codebookTile) * rowTiles + outputTile) * 16U +
                 code) *
                    2U +
                component;
            weightFp8.SetValue(column, codebooksGm_.GetValue(codebookOffset));
        }
        // weightFp8 is populated by scalar SetValue calls and consumed by the
        // vector Cast below.
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        AscendC::Cast(weightFloat, weightFp8, AscendC::RoundMode::CAST_NONE, sizeK);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(product, activationFloat, weightFloat, sizeK);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum<float>(sum, product, reduce, sizeK);
        // The reduced value is returned through scalar GetValue.
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        return sum.GetValue(0);
    }

    AscendC::TBuf<AscendC::QuePosition::VECCALC> activationFp8Buffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> activationFloatBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> weightFp8Buffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> weightFloatBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> productBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduceBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> sumBuffer_;
    AscendC::GlobalTensor<fp8_e4m3fn_t> quantizedGm_;
    AscendC::GlobalTensor<int32_t> packedIndicesGm_;
    AscendC::GlobalTensor<fp8_e4m3fn_t> codebooksGm_;
    AscendC::GlobalTensor<uint8_t> codebookTileIdsGm_;
};

class VQ2A8GateUpKernel : public VQ2A8LookupMatmulBase {
public:
    __aicore__ inline VQ2A8GateUpKernel(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(
        __gm__ void* quantized,
        __gm__ void* activationScale,
        __gm__ void* biasCorrection,
        __gm__ void* expertIds,
        __gm__ void* packedIndices,
        __gm__ void* codebooks,
        __gm__ void* codebookTileIds,
        __gm__ void* output,
        uint32_t sizeM,
        uint32_t sizeN,
        uint32_t sizeK,
        uint32_t numCodebookTiles,
        uint32_t rowTiles,
        uint32_t rowGroupSize,
        float swigluLimit)
    {
        sizeM_ = sizeM;
        sizeN_ = sizeN;
        sizeK_ = sizeK;
        numCodebookTiles_ = numCodebookTiles;
        rowTiles_ = rowTiles;
        rowGroupSize_ = rowGroupSize;
        swigluLimit_ = swigluLimit;
        packedRows_ = sizeN;
        packedWords_ = (sizeK + 7U) / 8U;
        quantizedGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t*)quantized);
        activationScaleGm_.SetGlobalBuffer((__gm__ float*)activationScale);
        biasCorrectionGm_.SetGlobalBuffer((__gm__ float*)biasCorrection);
        expertIdsGm_.SetGlobalBuffer((__gm__ int32_t*)expertIds);
        packedIndicesGm_.SetGlobalBuffer((__gm__ int32_t*)packedIndices);
        codebooksGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t*)codebooks);
        codebookTileIdsGm_.SetGlobalBuffer((__gm__ uint8_t*)codebookTileIds);
        outputGm_.SetGlobalBuffer((__gm__ bfloat16_t*)output);
        InitLookupBuffers(pipe_, sizeK);
        pipe_->InitBuffer(gateBuffer_, kOutputTile * sizeof(float));
        pipe_->InitBuffer(upBuffer_, kOutputTile * sizeof(float));
        pipe_->InitBuffer(sigmoidBuffer_, kOutputTile * sizeof(float));
        pipe_->InitBuffer(sigmoidTmpBuffer_, 2048);
        pipe_->InitBuffer(gateBf16Buffer_, kOutputTile * sizeof(bfloat16_t));
        pipe_->InitBuffer(upBf16Buffer_, kOutputTile * sizeof(bfloat16_t));
        pipe_->InitBuffer(outputBuffer_, kOutputTile * sizeof(bfloat16_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t tilesPerAssignment = sizeN_ / kOutputTile;
        const uint32_t blockIndex = AscendC::GetBlockIdx();
        const uint32_t assignment = blockIndex / tilesPerAssignment;
        if (assignment >= sizeM_) {
            return;
        }
        const uint32_t tile = blockIndex % tilesPerAssignment;
        const uint32_t expert = static_cast<uint32_t>(expertIdsGm_.GetValue(assignment));
        const float scale = activationScaleGm_.GetValue(assignment);
        const float bias = biasCorrectionGm_.GetValue(assignment);
        LoadActivation(assignment, sizeK_);

        AscendC::LocalTensor<float> gate = gateBuffer_.Get<float>();
        AscendC::LocalTensor<float> up = upBuffer_.Get<float>();
        for (uint32_t column = 0; column < kOutputTile; ++column) {
            const uint32_t outputColumn = tile * kOutputTile + column;
            gate.SetValue(
                column,
                Dot(
                    expert,
                    outputColumn,
                    sizeK_,
                    packedRows_,
                    packedWords_,
                    numCodebookTiles_,
                    rowTiles_,
                    rowGroupSize_) *
                        scale +
                    bias);
            up.SetValue(
                column,
                Dot(
                    expert,
                    outputColumn + sizeN_,
                    sizeK_,
                    packedRows_,
                    packedWords_,
                    numCodebookTiles_,
                    rowTiles_,
                    rowGroupSize_) *
                        scale +
                    bias);
        }

        AscendC::LocalTensor<bfloat16_t> gateBf16 = gateBf16Buffer_.Get<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> upBf16 = upBf16Buffer_.Get<bfloat16_t>();
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        AscendC::Cast(gateBf16, gate, AscendC::RoundMode::CAST_RINT, kOutputTile);
        AscendC::Cast(upBf16, up, AscendC::RoundMode::CAST_RINT, kOutputTile);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(gate, gateBf16, AscendC::RoundMode::CAST_NONE, kOutputTile);
        AscendC::Cast(up, upBf16, AscendC::RoundMode::CAST_NONE, kOutputTile);
        AscendC::PipeBarrier<PIPE_V>();
        if (swigluLimit_ > 0.0F) {
            AscendC::Mins(gate, gate, swigluLimit_, kOutputTile);
            AscendC::Maxs(up, up, -swigluLimit_, kOutputTile);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(up, up, swigluLimit_, kOutputTile);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::LocalTensor<float> sigmoid = sigmoidBuffer_.Get<float>();
        AscendC::LocalTensor<uint8_t> sigmoidTmp = sigmoidTmpBuffer_.Get<uint8_t>();
        AscendC::Sigmoid(sigmoid, gate, sigmoidTmp, kOutputTile);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(gate, gate, sigmoid, kOutputTile);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(gateBf16, gate, AscendC::RoundMode::CAST_RINT, kOutputTile);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(gate, gateBf16, AscendC::RoundMode::CAST_NONE, kOutputTile);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(gate, gate, up, kOutputTile);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<bfloat16_t> output = outputBuffer_.Get<bfloat16_t>();
        AscendC::Cast(output, gate, AscendC::RoundMode::CAST_RINT, kOutputTile);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        const uint64_t outputOffset = static_cast<uint64_t>(assignment) * sizeN_ + tile * kOutputTile;
        AscendC::DataCopy(outputGm_[outputOffset], output, kOutputTile);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> gateBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> upBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> sigmoidBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> sigmoidTmpBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> gateBf16Buffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> upBf16Buffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> outputBuffer_;
    AscendC::GlobalTensor<float> activationScaleGm_;
    AscendC::GlobalTensor<float> biasCorrectionGm_;
    AscendC::GlobalTensor<int32_t> expertIdsGm_;
    AscendC::GlobalTensor<bfloat16_t> outputGm_;
    uint32_t sizeM_ = 0;
    uint32_t sizeN_ = 0;
    uint32_t sizeK_ = 0;
    uint32_t packedRows_ = 0;
    uint32_t packedWords_ = 0;
    uint32_t numCodebookTiles_ = 0;
    uint32_t rowTiles_ = 0;
    uint32_t rowGroupSize_ = 0;
    float swigluLimit_ = 0.0F;
};

class VQ2A8DownReduceKernel : public VQ2A8LookupMatmulBase {
public:
    __aicore__ inline VQ2A8DownReduceKernel(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(
        __gm__ void* quantized,
        __gm__ void* activationScale,
        __gm__ void* biasCorrection,
        __gm__ void* expertIds,
        __gm__ void* tokenIds,
        __gm__ void* routingWeights,
        __gm__ void* packedIndices,
        __gm__ void* codebooks,
        __gm__ void* codebookTileIds,
        __gm__ void* output,
        uint32_t sizeM,
        uint32_t sizeN,
        uint32_t sizeK,
        uint32_t numCodebookTiles,
        uint32_t rowTiles,
        uint32_t rowGroupSize)
    {
        sizeM_ = sizeM;
        sizeN_ = sizeN;
        sizeK_ = sizeK;
        numCodebookTiles_ = numCodebookTiles;
        rowTiles_ = rowTiles;
        rowGroupSize_ = rowGroupSize;
        packedRows_ = sizeN / 2U;
        packedWords_ = (sizeK + 7U) / 8U;
        quantizedGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t*)quantized);
        activationScaleGm_.SetGlobalBuffer((__gm__ float*)activationScale);
        biasCorrectionGm_.SetGlobalBuffer((__gm__ float*)biasCorrection);
        expertIdsGm_.SetGlobalBuffer((__gm__ int32_t*)expertIds);
        tokenIdsGm_.SetGlobalBuffer((__gm__ int64_t*)tokenIds);
        routingWeightsGm_.SetGlobalBuffer((__gm__ float*)routingWeights);
        packedIndicesGm_.SetGlobalBuffer((__gm__ int32_t*)packedIndices);
        codebooksGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t*)codebooks);
        codebookTileIdsGm_.SetGlobalBuffer((__gm__ uint8_t*)codebookTileIds);
        outputGm_.SetGlobalBuffer((__gm__ float*)output);
        InitLookupBuffers(pipe_, sizeK);
        pipe_->InitBuffer(outputFloatBuffer_, kOutputTile * sizeof(float));
        pipe_->InitBuffer(outputBf16Buffer_, kOutputTile * sizeof(bfloat16_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t tilesPerAssignment = sizeN_ / kOutputTile;
        const uint32_t blockIndex = AscendC::GetBlockIdx();
        const uint32_t assignment = blockIndex / tilesPerAssignment;
        if (assignment >= sizeM_) {
            return;
        }
        const uint32_t tile = blockIndex % tilesPerAssignment;
        const uint32_t expert = static_cast<uint32_t>(expertIdsGm_.GetValue(assignment));
        const int64_t token = tokenIdsGm_.GetValue(assignment);
        const float scale = activationScaleGm_.GetValue(assignment);
        const float bias = biasCorrectionGm_.GetValue(assignment);
        const float routingWeight = routingWeightsGm_.GetValue(assignment);
        LoadActivation(assignment, sizeK_);

        AscendC::LocalTensor<float> output = outputFloatBuffer_.Get<float>();
        for (uint32_t column = 0; column < kOutputTile; ++column) {
            const uint32_t outputColumn = tile * kOutputTile + column;
            output.SetValue(
                column,
                Dot(
                    expert,
                    outputColumn,
                    sizeK_,
                    packedRows_,
                    packedWords_,
                    numCodebookTiles_,
                    rowTiles_,
                    rowGroupSize_) *
                        scale +
                    bias);
        }
        AscendC::LocalTensor<bfloat16_t> outputBf16 = outputBf16Buffer_.Get<bfloat16_t>();
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        AscendC::Cast(outputBf16, output, AscendC::RoundMode::CAST_RINT, kOutputTile);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(output, outputBf16, AscendC::RoundMode::CAST_NONE, kOutputTile);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(output, output, routingWeight, kOutputTile);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        const uint64_t outputOffset = static_cast<uint64_t>(token) * sizeN_ + tile * kOutputTile;
        AscendC::SetAtomicAdd<float>();
        AscendC::DataCopy(outputGm_[outputOffset], output, kOutputTile);
        AscendC::SetAtomicNone();
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> outputFloatBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> outputBf16Buffer_;
    AscendC::GlobalTensor<float> activationScaleGm_;
    AscendC::GlobalTensor<float> biasCorrectionGm_;
    AscendC::GlobalTensor<int32_t> expertIdsGm_;
    AscendC::GlobalTensor<int64_t> tokenIdsGm_;
    AscendC::GlobalTensor<float> routingWeightsGm_;
    AscendC::GlobalTensor<float> outputGm_;
    uint32_t sizeM_ = 0;
    uint32_t sizeN_ = 0;
    uint32_t sizeK_ = 0;
    uint32_t packedRows_ = 0;
    uint32_t packedWords_ = 0;
    uint32_t numCodebookTiles_ = 0;
    uint32_t rowTiles_ = 0;
    uint32_t rowGroupSize_ = 0;
};

extern "C" __global__ __aicore__ void vq2a8_transform_kernel(
    __gm__ void* x,
    __gm__ void* expertIds,
    __gm__ void* weightScale,
    __gm__ void* weightBias,
    __gm__ void* rhtSign,
    __gm__ void* transformed,
    __gm__ void* partialAmax,
    __gm__ void* partialBias,
    uint32_t sizeM,
    uint32_t sizeK,
    uint32_t rhtBlockSize)
{
    AscendC::TPipe pipe;
    VQ2A8TransformKernel kernel(&pipe);
    kernel.Init(
        x,
        expertIds,
        weightScale,
        weightBias,
        rhtSign,
        transformed,
        partialAmax,
        partialBias,
        sizeM,
        sizeK,
        rhtBlockSize);
    kernel.Process();
}

extern "C" __global__ __aicore__ void vq2a8_quantize_kernel(
    __gm__ void* transformed,
    __gm__ void* partialAmax,
    __gm__ void* partialBias,
    __gm__ void* quantized,
    __gm__ void* activationScale,
    __gm__ void* biasCorrection,
    uint32_t sizeM,
    uint32_t sizeK,
    uint32_t rhtBlockSize)
{
    AscendC::TPipe pipe;
    VQ2A8QuantizeKernel kernel(&pipe);
    kernel.Init(
        transformed,
        partialAmax,
        partialBias,
        quantized,
        activationScale,
        biasCorrection,
        sizeM,
        sizeK,
        rhtBlockSize);
    kernel.Process();
}

extern "C" __global__ __aicore__ void vq2a8_gate_up_kernel(
    __gm__ void* quantized,
    __gm__ void* activationScale,
    __gm__ void* biasCorrection,
    __gm__ void* expertIds,
    __gm__ void* packedIndices,
    __gm__ void* codebooks,
    __gm__ void* codebookTileIds,
    __gm__ void* output,
    uint32_t sizeM,
    uint32_t sizeN,
    uint32_t sizeK,
    uint32_t numCodebookTiles,
    uint32_t rowTiles,
    uint32_t rowGroupSize,
    float swigluLimit)
{
    AscendC::TPipe pipe;
    VQ2A8GateUpKernel kernel(&pipe);
    kernel.Init(
        quantized,
        activationScale,
        biasCorrection,
        expertIds,
        packedIndices,
        codebooks,
        codebookTileIds,
        output,
        sizeM,
        sizeN,
        sizeK,
        numCodebookTiles,
        rowTiles,
        rowGroupSize,
        swigluLimit);
    kernel.Process();
}

extern "C" __global__ __aicore__ void vq2a8_down_reduce_kernel(
    __gm__ void* quantized,
    __gm__ void* activationScale,
    __gm__ void* biasCorrection,
    __gm__ void* expertIds,
    __gm__ void* tokenIds,
    __gm__ void* routingWeights,
    __gm__ void* packedIndices,
    __gm__ void* codebooks,
    __gm__ void* codebookTileIds,
    __gm__ void* output,
    uint32_t sizeM,
    uint32_t sizeN,
    uint32_t sizeK,
    uint32_t numCodebookTiles,
    uint32_t rowTiles,
    uint32_t rowGroupSize)
{
    AscendC::TPipe pipe;
    VQ2A8DownReduceKernel kernel(&pipe);
    kernel.Init(
        quantized,
        activationScale,
        biasCorrection,
        expertIds,
        tokenIds,
        routingWeights,
        packedIndices,
        codebooks,
        codebookTileIds,
        output,
        sizeM,
        sizeN,
        sizeK,
        numCodebookTiles,
        rowTiles,
        rowGroupSize);
    kernel.Process();
}

}  // namespace

namespace vllm_ascend {

void vq2a8_prepare_impl(
    void* stream,
    void* x,
    void* expertIds,
    void* weightScale,
    void* weightBias,
    void* rhtSign,
    void* transformed,
    void* partialAmax,
    void* partialBias,
    void* quantized,
    void* activationScale,
    void* biasCorrection,
    uint32_t sizeM,
    uint32_t sizeK,
    uint32_t rhtBlockSize)
{
    const uint32_t numRhtBlocks = sizeK / rhtBlockSize;
    vq2a8_transform_kernel<<<sizeM * numRhtBlocks, nullptr, stream>>>(
        x,
        expertIds,
        weightScale,
        weightBias,
        rhtSign,
        transformed,
        partialAmax,
        partialBias,
        sizeM,
        sizeK,
        rhtBlockSize);
    vq2a8_quantize_kernel<<<sizeM, nullptr, stream>>>(
        transformed,
        partialAmax,
        partialBias,
        quantized,
        activationScale,
        biasCorrection,
        sizeM,
        sizeK,
        rhtBlockSize);
}

void vq2a8_gate_up_impl(
    void* stream,
    void* quantized,
    void* activationScale,
    void* biasCorrection,
    void* expertIds,
    void* packedIndices,
    void* codebooks,
    void* codebookTileIds,
    void* output,
    uint32_t sizeM,
    uint32_t sizeN,
    uint32_t sizeK,
    uint32_t numExperts,
    uint32_t numCodebookTiles,
    uint32_t rowTiles,
    uint32_t rowGroupSize,
    float swigluLimit)
{
    (void)numExperts;
    const uint32_t blockDim = sizeM * (sizeN / kOutputTile);
    vq2a8_gate_up_kernel<<<blockDim, nullptr, stream>>>(
        quantized,
        activationScale,
        biasCorrection,
        expertIds,
        packedIndices,
        codebooks,
        codebookTileIds,
        output,
        sizeM,
        sizeN,
        sizeK,
        numCodebookTiles,
        rowTiles,
        rowGroupSize,
        swigluLimit);
}

void vq2a8_down_reduce_impl(
    void* stream,
    void* quantized,
    void* activationScale,
    void* biasCorrection,
    void* expertIds,
    void* tokenIds,
    void* routingWeights,
    void* packedIndices,
    void* codebooks,
    void* codebookTileIds,
    void* output,
    uint32_t sizeM,
    uint32_t sizeN,
    uint32_t sizeK,
    uint32_t numTokens,
    uint32_t numExperts,
    uint32_t numCodebookTiles,
    uint32_t rowTiles,
    uint32_t rowGroupSize)
{
    (void)numTokens;
    (void)numExperts;
    const uint32_t blockDim = sizeM * (sizeN / kOutputTile);
    vq2a8_down_reduce_kernel<<<blockDim, nullptr, stream>>>(
        quantized,
        activationScale,
        biasCorrection,
        expertIds,
        tokenIds,
        routingWeights,
        packedIndices,
        codebooks,
        codebookTileIds,
        output,
        sizeM,
        sizeN,
        sizeK,
        numCodebookTiles,
        rowTiles,
        rowGroupSize);
}

}  // namespace vllm_ascend
