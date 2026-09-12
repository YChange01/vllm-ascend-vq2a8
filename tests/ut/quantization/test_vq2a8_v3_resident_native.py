# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host ABI/layout regressions; these do not compile or execute AscendC."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import regex as re

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "csrc/vq2a8_ascendc_v3"
V2_SOURCE = REPO / "csrc/vq2a8_ascendc_v2"


def _function(source, name):
    match = re.search(rf"\b{name}\([^;{{}}]*\)\s*\{{", source)
    assert match is not None, f"Missing function {name}"
    depth = 1
    end = match.end()
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start() : end]


def test_resident_header_descriptor_geometry_and_bounds(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Host C++ compiler unavailable; this is not an NPU test")
    code = tmp_path / "resident_layout.cpp"
    code.write_text(
        """#include "resident_layout.h"
#include <cassert>
#include <initializer_list>
using namespace vq2a8_v3_resident;
int main() {
  static_assert(kAbiVersion == 1 && (kCapabilities & 1) != 0);
  static_assert(kJobWords == 9 && kMaxJobs == 6);
  static_assert(kX == 0 && kScale == 1 && kBias == 2 && kPacked == 3);
  static_assert(kTable == 4 && kOutput == 5 && kRows == 6);
  static_assert(kColumns == 7 && kReduction == 8);
  static_assert(kUbBytes == 185408 && kL1Bytes == 327680);
  static_assert(kL0ABytes == 16384 && kL0BBytes == 65536 && kL0CBytes == 16384);
  for (int m = -1; m <= 34; ++m) {
    assert(ValidDimensions(m, 4096, 2048) == (m >= 1 && m <= 32));
    assert(ValidDimensions(m, 4096, 4096) == (m >= 1 && m <= 32));
  }
  for (auto k : {0, 512, 1024, 3072, 8192}) assert(!ValidDimensions(1, 4096, k));
  for (auto n : {0, 32, 128, 2048, 6144}) assert(!ValidDimensions(1, n, 4096));
  for (unsigned m = 1; m <= 32; ++m) {
    assert(HalfRows(m, 0) + HalfRows(m, 1) == m);
    assert(AlignedM(m) >= m && AlignedM(m) % 16 == 0);
  }
  for (unsigned k : {2048u, 4096u}) {
    for (unsigned n = 0; n < 4096; n += 128) {
      for (unsigned start = 0; start < k; start += 512) {
        auto packed = PackedOffset(n, start, k);
        assert(packed == (uint64_t(n / 32) * (k / 16) + start / 16) * 128);
        assert(TableOffset(n, start, 4096) == (uint64_t(start / 256) * 128 + n / 32) * 32);
      }
    }
  }
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / "resident_layout"
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(SOURCE), str(code), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("k", [2048, 4096])
@pytest.mark.parametrize("n_begin", [0, 3968])
def test_resident_pair_lookup_and_shared_l1_match_independent_byte_oracle(k, n_begin):
    rng = np.random.default_rng(k + n_begin)
    packed = rng.integers(0, 256, (128, k // 16, 16, 8), dtype=np.uint8)
    tables = rng.integers(0, 256, (k // 256, 128, 16, 2), dtype=np.uint8)
    for aic_begin in range(0, k, 1024):
        l1 = np.full((4, 64, 16, 32), 199, dtype=np.uint8)
        for half in range(2):
            start = aic_begin + half * 512
            packed_ub = packed[n_begin // 32 : n_begin // 32 + 4, start // 16 : start // 16 + 32].copy()
            decoded_ub = np.full((4, 32, 17, 32), 199, dtype=np.uint8)
            for n1 in range(4):
                for k1 in range(32):
                    lut = tables[(start + k1 * 16) // 256, n_begin // 32 + n1].reshape(-1).view("<u2")
                    for repeat in range(2):
                        words = packed_ub[n1, k1].reshape(-1)[repeat * 64 : (repeat + 1) * 64].astype("<u4")
                        indices = (words | ((words >> 4) << 16)) & np.uint32(0x000F000F)
                        values = lut[indices.view("<u2")].view(np.uint8)
                        decoded_ub[n1, k1, repeat * 8 : (repeat + 1) * 8] = values.reshape(8, 32)
            assert np.all(decoded_ub[:, :, 16] == 199)
            l1[:, half * 32 : (half + 1) * 32] = decoded_ub[:, :, :16]
        actual = l1.transpose(1, 2, 0, 3).reshape(1024, 128)
        ks = np.arange(aic_begin, aic_begin + 1024)[:, None]
        ns = np.arange(n_begin, n_begin + 128)[None, :]
        encoded = packed[ns // 32, ks // 16, ks % 16, ns % 32 // 4]
        code = (encoded >> ((ns // 2 % 2) * 4)) & 15
        expected = tables[ks // 256, ns // 32, code, ns % 2]
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("name", ["RunLut", "LoadVectorTile", "StoreB1", "Vector", "LoadA1", "LoadL0", "Cube"])
def test_resident_port_preserves_v2_arithmetic_and_pipeline(name):
    resident = (SOURCE / "resident_kernel.cpp").read_text(encoding="utf-8")
    original = (V2_SOURCE / "kernel.cpp").read_text(encoding="utf-8")
    assert _function(resident, name) == _function(original, name)


def test_resident_out_binding_has_no_host_staging_and_retains_all_owners():
    binding = (SOURCE / "torch_binding.cpp").read_text(encoding="utf-8")
    resident = _function(binding, "GroupedProjectionResidentOut")
    for forbidden in ("at::empty", "at::zeros", ".to(", ".cpu(", ".item(", "at::matmul", "aclrtSynchronize"):
        assert forbidden not in resident
    assert "m == 1" in resident
    assert "vq2a8_v3_resident::ValidDimensions(m, n, k)" in resident
    assert "descriptors.size(1) == vq2a8_v3_resident::kJobWords" in resident
    assert "!owners.empty()" in resident and "owner.defined()" in resident
    assert "RecordInputStream(owners, npuStream);" in resident
    assert "RecordInputStream({descriptors}, npuStream);" in resident
    assert resident.index("RecordInputStream(owners") < resident.index("command.SetCustomHandler")
    assert "[stream, blocks, descriptors, owners, jobs, groups]" in resident
    assert "vq2a8_v3_resident::LaunchGrouped" in resident
    assert "Tensor(a!)[] owners" in binding


def test_resident_eager_checks_all_jobs_before_work_and_keeps_output_owners():
    binding = (SOURCE / "torch_binding.cpp").read_text(encoding="utf-8")
    eager = _function(binding, "GroupedProjectionResident")
    validation = _function(binding, "CheckResidentProjection")
    assert eager.index("CheckResidentProjection(") < eager.index("at::empty(")
    for value in ("scale", "bias", "packed", "table"):
        assert f"{value}.size() == jobs" in eager
        assert f"RecordInputStream({value}, npuStream);" in eager
    assert "[stream, blocks, descriptors, jobs, groups, x, scale, bias, packed, table, output]" in eager
    assert "vq2a8_v3_resident::kOutput" in eager
    for declaration in ("at::kFloat8_e4m3fn, 2", "at::kByte, 4", "at::kByte, 3"):
        assert declaration in validation


def test_prepare_out_checks_geometry_aliases_and_retains_every_input_and_output():
    binding = (SOURCE / "torch_binding.cpp").read_text(encoding="utf-8")
    preparation = _function(binding, "PrepareOut")
    for forbidden in ("at::empty", "at::zeros", ".to(", ".cpu(", ".item(", "aclrtSynchronize"):
        assert forbidden not in preparation
    assert "ValidPrepareDimensions(jobs, k)" in preparation
    assert "weightScale.sizes() == rotated.sizes()" in preparation
    assert "order.sizes() == rotated.sizes()" in preparation
    assert "quantized.sizes() == rotated.sizes()" in preparation
    assert "inputBias.numel() == jobs" in preparation
    assert "scale.numel() == jobs" in preparation
    assert "bias.numel() == jobs" in preparation
    assert "valid.numel() == jobs" in preparation
    assert "owners{rotated, weightScale, order, inputBias, quantized, scale, bias, valid}" in preparation
    assert "kFirstOutput = 4" in preparation
    assert "!owners[output].is_alias_of(owners[other])" in preparation
    assert preparation.index("is_alias_of") < preparation.index("command.SetCustomHandler")
    assert "RecordInputStream(owners, npuStream);" in preparation
    assert "[stream, blocks, owners, jobs, k]" in preparation
    assert "LaunchPrepareV3(" in preparation
    assert 'm.impl("prepare_out", &vq2a8_v3::PrepareOut)' in binding


def test_new_resident_namespace_is_linked_alongside_unchanged_diagnostics():
    binding = (SOURCE / "torch_binding.cpp").read_text(encoding="utf-8")
    kernel = (SOURCE / "resident_kernel.cpp").read_text(encoding="utf-8")
    cmake = (SOURCE / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "namespace vq2a8_v3_resident" in kernel
    assert "void vq2a8_v3_resident_grouped(" in kernel
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)" in kernel
    assert "kernel.cpp resident_kernel.cpp" in cmake
    assert "prepare_kernel.cpp" in cmake
    assert "vq2a8_ascendc_v2" not in cmake
    for operator in (
        "projection",
        "grouped_projection",
        "grouped_projection_out",
        "make_constants",
        "grouped_projection_resident",
        "grouped_projection_resident_out",
    ):
        assert f'm.impl("{operator}"' in binding
    assert 'm.def("resident_abi_version() -> int"' in binding
    assert 'm.def("resident_capabilities() -> int"' in binding
