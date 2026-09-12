# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host geometry checks and native preparation regression cases.

Device cases require an NPU and an already loaded, pinned V3 library. Host
checks alone do not establish CANN compilation or FP8 numerical equivalence.
"""

import ast
import importlib
import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "csrc/vq2a8_ascendc_v3"


class PrepareHostTests(unittest.TestCase):
    def test_host_geometry_and_nonfinite_bit_detection(self):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            self.skipTest("No host C++ compiler")
        reference = ast.parse((REPO / "vllm_ascend/quantization/vq2a8_reference.py").read_text(encoding="utf-8"))
        minimum_scale = next(
            ast.literal_eval(node.value)
            for node in reference.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "VQ2_FP8_MIN_SCALE" for target in node.targets)
        )
        program = r"""#include "csrc/vq2a8_ascendc_v3/prepare_layout.h"
#include <cassert>
#include <cstdint>
#include <initializer_list>
int main() {
  using namespace vq2a8_v3;
  static_assert(kPrepareFp8Max == 448.0f && kPrepareMinScale == 1e-12f);
  static_assert(kPrepareUbBytes == 157824 && kPrepareUbBytes < 192 * 1024);
  for (int64_t jobs = -1; jobs <= 7; ++jobs) {
    for (int64_t k : {-512, 0, 1, 511, 512, 1536, 2048, 2560, 4096, 65536, 66048}) {
      bool expected = jobs >= 1 && jobs <= 6 && k >= 512 && k <= 65536 && k % 512 == 0;
      assert(ValidPrepareDimensions(jobs, k) == expected);
    }
  }
  for (uint32_t k : {512u, 1536u, 2048u, 2560u, 4096u, 65536u}) {
    uint32_t covered = 0;
    for (uint32_t start = 0; start < k; start += kPrepareTile) {
      auto count = PrepareTileCount(k, start);
      assert(count > 0 && count <= 2048 && count % 512 == 0);
      assert(start + count <= k);
      covered += count;
    }
    assert(covered == k);
  }
  for (uint32_t sign : {0u, 0x80000000u}) {
    for (uint32_t bits : {0u, 1u, 0x007fffffu, 0x00800000u, 0x3f800000u, 0x7f7fffffu}) {
      assert(PrepareFiniteBits(sign | bits));
    }
    for (uint32_t mantissa : {0u, 1u, 0x00400000u, 0x007fffffu}) {
      assert(!PrepareFiniteBits(sign | 0x7f800000u | mantissa));
    }
  }
}
""".replace("== 1e-12f", f"== {minimum_scale!r}f")
        with tempfile.TemporaryDirectory(prefix="vq2a8-v3-prepare-host-") as directory:
            source = Path(directory) / "prepare.cpp"
            binary = Path(directory) / "prepare"
            source.write_text(program, encoding="utf-8")
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(REPO),
                    str(source),
                    "-o",
                    str(binary),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run([str(binary)], check=True, capture_output=True, text=True)

    def test_fp8_conversion_uses_existing_a5_recipe(self):
        kernel = (SOURCE / "prepare_kernel.cpp").read_text(encoding="utf-8")
        precedent = (REPO / "csrc/attention/mla_prolog_v3/op_kernel/arch35/vf/vf_dynamic_quant.h").read_text(
            encoding="utf-8"
        )
        for expression in (
            "RegLayout::ZERO",
            "SatMode::SAT",
            "MaskMergeMode::ZEROING",
            "RoundMode::CAST_RINT",
            "StoreDist::DIST_PACK4_B32",
        ):
            self.assertIn(expression, kernel)
            self.assertIn(expression, precedent)
        self.assertIn("MicroAPI::Cast<fp8_e4m3fn_t, float, kCast>", kernel)
        self.assertIn("KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)", kernel)
        self.assertIn("kPrepareExponentPairMask, count / 2", kernel)
        self.assertIn("Gather(weight_.Get<int32_t>(), orderUb_.Get<int32_t>(), highOffsets_", kernel)
        self.assertLess(kernel.index("Mins(input_.Get<float>()"), kernel.index("Gather(output_.Get<uint8_t>()"))
        self.assertIn("DataCopyPad(outputBias_[job], scalars_.Get<float>()[kBiasWord], scalarCopy)", kernel)


class PrepareDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("PyTorch unavailable; host checks are not device validation")
        torch = importlib.import_module("torch")
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise unittest.SkipTest("Native preparation requires Ascend NPU hardware")
        namespace = torch.ops.vq2a8_ascendc_v3
        if not hasattr(namespace, "prepare_out"):
            raise unittest.SkipTest("Load the explicitly pinned V3 candidate before these device tests")
        cls.prepare = namespace.prepare_out
        spec = importlib.util.spec_from_file_location(
            "vq2a8_v3_prepare_check", REPO / "tools/vq2a8_v3_prepare_check.py"
        )
        cls.checks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.checks)

    def test_reference_bytes_dirty_reuse_and_chunk_boundaries(self):
        self.checks.check_prepare_reuse(self.prepare)

    def test_midpoints_negative_zero_and_preserved_bias(self):
        self.checks.check_prepare_midpoints(self.prepare)

    def test_nonfinite_and_full_int64_order_validation(self):
        self.checks.check_prepare_invalid(self.prepare)


if __name__ == "__main__":
    unittest.main()
