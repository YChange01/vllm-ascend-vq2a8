# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the real AICPU scheduler on the host with a minimal CANN context shim.

This tests C++ validation/scheduling/ABI writes, not CANN loading or NPU code.
"""

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
OP = REPO / "csrc/attention/vllm_quant_lightning_indexer_metadata"

CONTEXT = r"""
#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
namespace aicpu {
struct TensorShape {
    int64_t length;
    int64_t GetDimSize(int) const { return length; }
};
struct Tensor {
    void* data;
    TensorShape shape;
    void* GetData() { return data; }
    TensorShape* GetTensorShape() { return &shape; }
};
struct Attr {
    int64_t integer = 0;
    std::string text;
    int64_t GetInt() const { return integer; }
    std::string GetString() const { return text; }
    bool GetBool() const { return integer != 0; }
};
struct CpuKernelContext {
    std::vector<Tensor*> inputs, outputs;
    std::map<std::string, Attr> attrs;
    Tensor* Input(uint32_t n) { return inputs.at(n); }
    Tensor* Output(uint32_t n) { return outputs.at(n); }
    Attr* GetAttr(const std::string& key) {
        auto it = attrs.find(key);
        return it == attrs.end() ? nullptr : &it->second;
    }
};
class CpuKernel {
public:
    virtual ~CpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext&) = 0;
};
}
#define REGISTER_CPU_KERNEL(...)
#define KERNEL_LOG_ERROR(...) std::fprintf(stderr, __VA_ARGS__)
#define KERNEL_CHECK_NULLPTR(ptr, result, ...) if (!(ptr)) { return result; }
"""

HARNESS = r"""
#include "cpu_context.h"
#include "op_kernel_aicpu/vllm_quant_lightning_indexer_metadata_aicpu.h"
#include "../vllm_quant_lightning_indexer/op_kernel/vllm_quant_lightning_indexer_metadata.h"

using namespace aicpu;
using namespace optiling;
int main(int argc, char** argv) {
    assert(argc == 5);
    const auto aic = std::atoi(argv[1]), aiv = std::atoi(argv[2]);
    int32_t seq = std::atoi(argv[3]);
    const bool valid = std::atoi(argv[4]);
    // Canary on either side; the unused 160-int ABI tail must stay untouched.
    std::array<uint32_t, QLI_META_SIZE + 2> storage;
    std::vector<uint32_t> previous;
    for (int repeat = 0; repeat < 3; ++repeat) {
        const uint32_t poison = 0xa5a50000U + repeat;
        storage.fill(poison);
        Tensor q{&seq, {1}}, k{&seq, {1}}, out{storage.data() + 1, {QLI_META_SIZE}};
        CpuKernelContext ctx{{&q, &k}, {&out}, {}};
        const std::map<std::string, int64_t> attrs{
            {"aic_core_num", aic}, {"aiv_core_num", aiv}, {"num_heads_q", 64},
            {"num_heads_k", 1}, {"head_dim", 128}, {"query_quant_mode", 0},
            {"key_quant_mode", 0}, {"batch_size", 1}, {"max_seqlen_q", seq},
            {"max_seqlen_k", seq}, {"sparse_count", 512}, {"sparse_mode", 3},
            {"pre_tokens", INT64_MAX}, {"next_tokens", INT64_MAX}, {"cmp_ratio", 4}
        };
        for (const auto& item : attrs) ctx.attrs[item.first].integer = item.second;
        ctx.attrs["soc_version"].text = "Ascend950PR_958b";
        ctx.attrs["layout_query"].text = "TND";
        ctx.attrs["layout_key"].text = "PA_BSND";
        VllmQuantLightningIndexerMetadataCpuKernel kernel;
        const auto status = kernel.Compute(ctx);
        assert((status == 0) == valid);
        if (!valid) {
            for (auto value : storage) assert(value == poison);
            continue;
        }
        assert(storage.front() == poison && storage.back() == poison);
        const auto* meta = reinterpret_cast<const detail::QliMetaData*>(out.data);
        bool disabled = false;
        std::array<uint32_t, 3> end{};
        for (uint32_t i = 0; i < AIC_CORE_NUM; ++i) {
            const auto* row = meta->LIMetadata[i];
            if (row[LI_CORE_ENABLE_INDEX] == 0) {
                disabled = true;
                for (uint32_t j = 0; j < LI_METADATA_SIZE; ++j) assert(row[j] == 0);
            } else {
                assert(!disabled && i < static_cast<uint32_t>(aic));
                assert(row[LI_CORE_ENABLE_INDEX] == 1);
                const std::array<uint32_t, 3> start{
                    row[LI_BN2_START_INDEX], row[LI_M_START_INDEX], row[LI_S2_START_INDEX]};
                assert(start == end); // no gaps or overlapping assigned intervals
                end = {row[LI_BN2_END_INDEX], row[LI_M_END_INDEX], row[LI_S2_END_INDEX]};
                assert(end > start);
            }
        }
        assert((end == std::array<uint32_t, 3>{1, 0, 0}));
        // FD is disabled in this operator; every Vector FD slot must be zero.
        for (const auto& row : meta->LDMetadata) for (auto value : row) assert(value == 0);
        constexpr size_t words = sizeof(detail::QliMetaData) / sizeof(uint32_t);
        std::vector<uint32_t> current(storage.begin() + 1, storage.begin() + 1 + words);
        if (repeat) assert(current == previous);
        previous = current;
        for (size_t i = 1 + words; i < storage.size(); ++i) assert(storage[i] == poison);
    }
}
"""


@pytest.fixture(scope="module")
def scheduler(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("Host C++ compiler required for real AICPU scheduler regression")
    work = tmp_path_factory.mktemp("qli-host-scheduler")
    (work / "context_shim.h").write_text(CONTEXT, encoding="utf-8")
    for name in ("cpu_context.h", "cpu_kernel.h", "cpu_tensor.h", "log.h"):
        (work / name).write_text('#include "context_shim.h"\n', encoding="utf-8")
    main = work / "main.cpp"
    main.write_text(HARNESS, encoding="utf-8")
    binary = work / "scheduler"
    command = [
        compiler,
        "-std=c++17",
        "-O1",
        "-g",
        "-fsanitize=undefined",
        "-fno-sanitize-recover=all",
        f"-I{work}",
        f"-I{OP}",
        str(main),
        str(OP / "op_kernel_aicpu/vllm_quant_lightning_indexer_metadata_aicpu.cpp"),
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    return binary


@pytest.mark.parametrize("cores", [(28, 64), (28, 56), (32, 64), (24, 48), (36, 72)])
@pytest.mark.parametrize("tokens", [1, 3, 10, 32, 128, 129])
def test_real_scheduler_available_core_counts_and_disabled_slots(scheduler, cores, tokens):
    subprocess.run([str(scheduler), *map(str, cores), str(tokens), "1"], check=True, timeout=10)


@pytest.mark.parametrize("cores", [(0, 64), (28, 0), (28, 55), (32, 32), (37, 74), (36, 73)])
def test_real_scheduler_rejects_unsafe_counts_before_writing(scheduler, cores):
    subprocess.run([str(scheduler), *map(str, cores), "10", "0"], check=True, timeout=10)


def valid_metadata():
    values = [0] * 1024
    values[:8] = [1, 0, 0, 0, 1, 0, 0, 0]
    return values


def test_preflight_ignores_only_reserved_tail():
    from tools.validate_vq2a8_qli_metadata import DEFINED_WORDS, check_metadata

    values = valid_metadata()
    values[DEFINED_WORDS:] = [0x55555555] * (1024 - DEFINED_WORDS)
    assert check_metadata(values) == 1


@pytest.mark.parametrize("index,value", [(0, 0), (0, 2), (1, 1), (4, 0), (8, 1), (9, 99), (28 * 8, 1), (36 * 8, 1)])
def test_preflight_rejects_invalid_or_uninitialized_schedule(index, value):
    from tools.validate_vq2a8_qli_metadata import check_metadata

    values = valid_metadata()
    values[index] = value
    with pytest.raises(ValueError):
        check_metadata(values)


def test_preflight_rejects_wrong_size():
    from tools.validate_vq2a8_qli_metadata import check_metadata

    with pytest.raises(ValueError, match="1024"):
        check_metadata([0] * 864)


def test_offline_checks_metadata_before_expensive_engine_construction():
    source = REPO / "tools/validate_vq2a8_tp1_offline.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    preflight, construct = (
        next(
            n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
        )
        for name in ("run_preflight", "LLM")
    )
    assert preflight.lineno < construct.lineno
    assert any(k.arg == "prompt_tokens" for k in preflight.keywords)
