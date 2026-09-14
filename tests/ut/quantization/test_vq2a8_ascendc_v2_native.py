# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU layout/protocol checks for VQ2A8 kernel v2; not device execution proof."""

import importlib.util
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import regex as re

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "csrc/vq2a8_ascendc_v2"


def _bridge():
    path = REPO / "csrc/vq2a8_expert_reference/scripts/vq2_bridge.py"
    spec = importlib.util.spec_from_file_location("vq2_v2_test_bridge", path)
    module = importlib.util.module_from_spec(spec)
    # dataclass inspects sys.modules during module execution.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v2_host_header_geometry_and_bounds(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Host C++ compiler unavailable; run on development Linux (not an NPU test).")
    source = tmp_path / "layout.cpp"
    source.write_text(
        r"""#include "layout.h"
#include <cassert>
using namespace vq2a8_ascendc_v2;
int main() {
    static_assert(kAbiVersion == 1 && kJobWords == 9 && kMaxJobs == 6);
    static_assert(kUbBytes == 185408 && kL1Bytes == 327680 && kL0BBytes == 65536);
    for (int m = -1; m <= 34; ++m) {
        assert(ValidDimensions(m, 4096, 2048) == (m >= 1 && m <= 32));
        assert(ValidDimensions(m, 4096, 4096) == (m >= 1 && m <= 32));
    }
    assert(!ValidDimensions(1, 6144, 4096));
    assert(!ValidDimensions(1, 4096, 1024));
    for (unsigned m = 1; m <= 32; ++m) {
        assert(HalfRows(m, 0) + HalfRows(m, 1) == m);
        assert(AlignedM(m) >= m && AlignedM(m) % 16 == 0);
    }
    for (unsigned k: {2048u, 4096u}) {
        for (unsigned n = 0; n < 4096; n += 128) {
            for (unsigned start = 0; start < k; start += 512) {
                auto offset = PackedOffset(n, start, k);
                assert(offset == (uint64_t(n / 32) * (k / 16) + start / 16) * 128);
                assert(TableOffset(n, start, 4096) == (uint64_t(start / 256) * 128 + n / 32) * 32);
            }
        }
    }
}""".replace("#include <cassert>", "#include <cassert>\n#include <initializer_list>"),
        encoding="utf-8",
    )
    exe = tmp_path / "layout"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(SOURCE), str(source), "-o", str(exe)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(exe)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("k", [2048, 4096])
@pytest.mark.parametrize("n_begin", [0, 384, 3968])
def test_v2_register_decode_and_padded_l1_layout(k, n_begin):
    """Independent lane-level emulation of AIV's register operations and DMA.

    Check all K for a 128-column tile at beginning/middle/end of a real N4096
    matrix, arbitrary 16x2 codebooks (not scalar W2), both AIVs and ping/pong.
    This tests indexing/dataflow, not instruction support or NPU scheduling.
    """
    rng = np.random.default_rng(k + n_begin)
    pair_codes = rng.integers(0, 16, size=(2048, k), dtype=np.uint8)
    packed = _bridge().pack_pairs_zn(pair_codes).reshape(-1)
    table = rng.integers(0, 127, size=(k // 256, 128, 16, 2), dtype=np.uint8)
    for aic_begin in range(0, k, 1024):
        l1 = np.full((4, 64, 16, 32), 255, dtype=np.uint8)
        for half in range(2):
            k_begin = aic_begin + half * 512
            base = ((n_begin // 32) * (k // 16) + k_begin // 16) * 128
            ub_packed = np.concatenate([packed[base + n1 * k * 8 : base + n1 * k * 8 + 4096] for n1 in range(4)])
            ub_decoded = np.full((4, 32, 17, 32), 255, dtype=np.uint8)
            local_tables = np.ascontiguousarray(
                table[k_begin // 256 : k_begin // 256 + 2, n_begin // 32 : n_begin // 32 + 4]
            )
            for n1 in range(4):
                for lut_k in range(2):
                    # uint16 pairs preserve arbitrary FP8 bytes.
                    lut = local_tables[lut_k, n1].reshape(-1).view("<u2")
                    for local_k1 in range(16):
                        k1 = lut_k * 16 + local_k1
                        offset = (n1 * 32 + k1) * 128
                        for repeat in range(2):
                            words = ub_packed[offset + repeat * 64 : offset + (repeat + 1) * 64].astype("<u4")
                            # Literal operations from the recovered register path.
                            indices = (words | ((words >> 4) << 16)) & np.uint32(0x000F000F)
                            values = lut[indices.view("<u2")].view(np.uint8)
                            ub_decoded[n1, k1, repeat * 8 : (repeat + 1) * 8] = values.reshape(8, 32)
            assert np.all(ub_decoded[:, :, 16] == 255), "DMA must not overwrite the bank-conflict padding row"
            l1[:, half * 32 : (half + 1) * 32] = ub_decoded[:, :, :16]
        decoded_kn = l1.transpose(1, 2, 0, 3).reshape(1024, 128)
        ks = np.arange(aic_begin, aic_begin + 1024)[:, None]
        ns = np.arange(n_begin, n_begin + 128)[None, :]
        codes = pair_codes[ns // 2, ks]
        expected = table[ks // 256, ns // 32, codes, ns % 2]
        np.testing.assert_array_equal(decoded_kn, expected)


@pytest.mark.parametrize("cores", [1, 2, 7, 28, 32, 40])
@pytest.mark.parametrize("jobs", [1, 2, 6])
def test_v2_dynamic_mixed_core_mapping_has_no_holes(cores, jobs):
    total = jobs * 32  # N4096 / N128
    blocks = min(cores, total)
    cube_work = {core: list(range(core, total, blocks)) for core in range(blocks)}
    vector_work = {vector: list(range(vector // 2, total, blocks)) for vector in range(2 * blocks)}
    assert sorted(work for works in cube_work.values() for work in works) == list(range(total))
    for cube in range(blocks):
        assert vector_work[2 * cube] == vector_work[2 * cube + 1] == cube_work[cube]


@pytest.mark.parametrize("m", range(1, 33))
def test_v2_fixpipe_actual_m_row_partition(m):
    aligned = (m + 15) // 16 * 16
    half_capacity = aligned // 2
    covered = []
    for half in range(2):
        first = half * half_capacity
        rows = max(0, min(m - first, half_capacity))
        covered.extend(range(first, first + rows))
        assert rows <= 16
    assert covered == list(range(m))


def _actions(k_tiles, jobs, actor):
    actions = []
    for job in range(jobs):
        if actor == 0:
            for ki in range(k_tiles):
                slot = ki % 2
                actions += [("wait", ("ready", slot, peer)) for peer in (1, 2)]
                actions += [("read", (slot, job, ki))]
                actions += [("set", ("free", slot, peer)) for peer in (1, 2)]
            actions += [("set", ("result", peer)) for peer in (1, 2)]
            actions += [("wait", ("stored", peer)) for peer in (1, 2)]
        else:
            for ki in range(k_tiles):
                slot = ki % 2
                if ki >= 2:
                    actions.append(("wait", ("free", slot, actor)))
                actions += [("write", (slot, actor, job, ki)), ("set", ("ready", slot, actor))]
            actions += [("wait", ("free", slot, actor)) for slot in (0, 1)]
            actions += [("wait", ("result", actor)), ("set", ("stored", actor))]
    return actions


@pytest.mark.parametrize("k_tiles", [2, 4])
@pytest.mark.parametrize("seed", range(8))
def test_v2_cross_core_protocol_drains_between_pointer_jobs(k_tiles, seed):
    """Finite state model of inter-core tokens; no claim to simulate pipelines."""
    rng = random.Random(seed)
    sequences = [_actions(k_tiles, 3, actor) for actor in range(3)]
    positions = [0, 0, 0]
    tokens = set()
    buffers = {}
    while any(positions[i] < len(sequences[i]) for i in range(3)):
        actors = list(range(3))
        rng.shuffle(actors)
        for actor in actors:
            if positions[actor] == len(sequences[actor]):
                continue
            action, key = sequences[actor][positions[actor]]
            if action == "wait" and key not in tokens:
                continue
            if action == "set":
                assert key not in tokens, "double-set/lost event across jobs"
                tokens.add(key)
            elif action == "wait":
                tokens.remove(key)
            elif action == "write":
                slot, peer, job, ki = key
                assert (slot, peer) not in buffers, "AIV overwrote L1 before AIC consumed it"
                buffers[slot, peer] = (job, ki)
            else:
                slot, job, ki = key
                for peer in (1, 2):
                    assert buffers.pop((slot, peer)) == (job, ki)
            positions[actor] += 1
            break
        else:
            pytest.fail("mixed-core protocol deadlock")
    assert not tokens and not buffers


@pytest.mark.parametrize(("count_register", "count"), [("shiftRight", 4), ("shiftLeft", 16)])
def test_v2_shift_counts_are_signed(count_register, count):
    """Guard the CANN 9.1 u32-data/s32-count contract; not a device compile test."""
    kernel = (SOURCE / "kernel.cpp").read_text(encoding="utf-8")
    register_types = {
        name.strip(): dtype
        for dtype, names in re.findall(r"RegTensor<(\w+)>\s+([\w,\s]+);", kernel)
        for name in names.split(",")
    }
    assert register_types[count_register] == "int32_t", "vshr/vshl require signed per-lane shift counts"
    assert re.search(rf"Duplicate\(\s*{count_register},\s*int32_t\({count}\),\s*all32\s*\)", kernel)
    # Only the counts change type: preserve logical shifts and packed LUT bits.
    for data_register in ("word", "highNibble", "highIndex", "index", "mask"):
        assert register_types[data_register] == "uint32_t"


def test_v2_native_source_contract_and_distinct_backend():
    kernel = (SOURCE / "kernel.cpp").read_text(encoding="utf-8")
    binding = (SOURCE / "torch_binding.cpp").read_text(encoding="utf-8")
    layout = (SOURCE / "layout.h").read_text(encoding="utf-8")
    assert "DIST_UNPACK4_B8" in kernel and "0x000F000F" in kernel
    assert "b.ifTranspose = true" in kernel
    assert "Fixpipe<float, float, kToUb>" in kernel
    assert "QuantMode_t::NoQuant" in kernel
    assert kernel.index("Muls(result[") < kernel.index("Adds(result[") < kernel.index("Cast(out, result")
    assert kernel.count("Cast(out, result") == 1
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)" in kernel
    assert "work += cores" in kernel and "core /= 2" in kernel
    assert "AllocEventID<E>" in kernel and "ReleaseEventID<E>" in kernel
    assert "packedFree_.Drain()" in kernel and "l0Free_.Drain()" in kernel
    assert "TORCH_LIBRARY(vq2a8_ascendc_v2, m)" in binding
    assert "at::kFloat8_e4m3fn" in binding and "at::kFloat" in binding
    assert "RecordInputStream(packed, stream)" in binding
    assert "NPUCachingAllocator::recordStream" in binding
    assert "[stream, blocks, descriptors, jobs, nTiles, x, scale, bias, packed, table, output]" in binding
    assert "ValidDimensions(m, n, k)" in binding
    assert "kMaxJobs = 6" in layout and "kJobWords = 9" in layout
    assert not re.search(r"\b(?:aclrtResetDevice|rtDeviceReset|rtStreamCreate|at::matmul|at::cat)\s*\(", binding)
