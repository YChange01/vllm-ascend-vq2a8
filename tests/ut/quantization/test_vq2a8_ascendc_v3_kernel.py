# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU byte-layout/protocol and source regressions, NOT Ascend950 execution proof."""

import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import regex as re

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "csrc/vq2a8_ascendc_v3"
ORIGINAL = REPO / "csrc/vq2a8_ascendc"


def _method(source, name):
    match = re.search(rf"  __aicore__ inline void {name}\(.*?\n  }}", source, re.S)
    if match is None:
        raise AssertionError(f"Missing device method {name}")
    return match.group()


def _protocol_actions(buffers, k_tiles, jobs, actor):
    actions = []
    for job in range(jobs):
        if actor == 0:
            for ki in range(k_tiles):
                slot = ki % buffers
                actions += [("wait", ("ready", slot, peer)) for peer in (1, 2)]
                actions.append(("read", (slot, job, ki)))
                actions += [("set", ("free", slot, peer)) for peer in (1, 2)]
            actions += [("set", ("result", peer)) for peer in (1, 2)]
            actions += [("wait", ("stored", peer)) for peer in (1, 2)]
        else:
            for ki in range(k_tiles):
                slot = ki % buffers
                if ki >= buffers:
                    actions.append(("wait", ("free", slot, actor)))
                actions += [("write", (slot, actor, job, ki)), ("set", ("ready", slot, actor))]
            actions += [("wait", ("free", slot, actor)) for slot in range(min(buffers, k_tiles))]
            actions += [("wait", ("result", actor)), ("set", ("stored", actor))]
    return actions


class V3KernelCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = (SOURCE / "kernel.cpp").read_text(encoding="utf-8")
        cls.original_kernel = (ORIGINAL / "kernel.cpp").read_text(encoding="utf-8")
        cls.layout = (SOURCE / "layout.h").read_text(encoding="utf-8")

    def test_cube_decode_and_epilogue_keep_v1_order(self):
        for name in ("Cube", "Decode", "CopyHalfToL1", "LoadNd"):
            with self.subTest(method=name):
                self.assertEqual(_method(self.kernel, name), _method(self.original_kernel, name))
        begin = "    CrossCoreWaitFlag<4, PIPE_V>(kResult);"
        end = "    Fence<HardEvent::MTE3_S>();"
        old_epilogue = self.original_kernel.split(begin, 1)[1].split(end, 1)[0]
        self.assertEqual(self.kernel.split(begin, 1)[1].split(end, 1)[0], old_epilogue)
        self.assertIn("p.m = kM;", _method(self.kernel, "Cube"))
        self.assertIn("constexpr uint32_t kM = 32;", self.layout)
        self.assertIn("constexpr uint32_t kK = 128;", self.layout)

    def test_prepared_constant_abi_and_dma_counts(self):
        offsets = [0, 512, 1536, 2560, 3584, 4608]
        counts = [512, 1024, 1024, 1024, 1024, 2048]
        self.assertEqual(sum(counts), 6656)
        self.assertEqual([offsets[i] + counts[i] for i in range(5)], offsets[1:])
        self.assertTrue(all(offset * 4 % 32 == 0 for offset in offsets))
        self.assertTrue(all(count * 4 % 32 == 0 for count in counts))
        initializer = _method(self.kernel, "InitBuffers")
        prepared = initializer.split("if (constants != nullptr) {", 1)[1].split("} else {", 1)[0]
        self.assertEqual(prepared.count("DataCopy("), 6)
        self.assertNotIn("SetValue", prepared)
        self.assertIn("Fence<HardEvent::MTE2_V>();", prepared)
        tables = (
            ("ndOffsets_", "kNdConstants", "kHalfTileBytes / 4"),
            ("packedOffsets_", "kPackedConstants", "kPairs"),
            ("idOffsets_", "kIdConstants", "kPairs"),
            ("codeShifts_", "kCodeShiftConstants", "kPairs"),
            ("idShifts_", "kIdShiftConstants", "kPairs"),
            ("pairOffsets_", "kPairConstants", "kHalfTileBytes"),
        )
        for name, offset, count in tables:
            self.assertIn(f"DataCopy({name}.Get<uint32_t>(), words[{offset}], {count});", prepared)
        for assignment in (
            "words[kNdConstants + i] = NdGatherOffset(i)",
            "words[kPackedConstants + i] = PackedGatherOffset(i)",
            "words[kIdConstants + i] = IdGatherOffset(i)",
            "words[kCodeShiftConstants + i] = (i % 8) * 4",
            "words[kIdShiftConstants + i] = (i % 4) * 8",
            "words[kPairConstants + i] = PairGatherOffset(i)",
        ):
            self.assertIn(assignment, self.layout)
        self.assertIn("static_assert(kConstantWords == 6656", self.layout)

    def test_m1_nz_dma_is_byte_exact_across_dirty_jobs(self):
        load = _method(self.kernel, "LoadM1")
        for expression in (
            "if (half == 0)",
            "p.blockCount = kK / kC0;",
            "p.blockLen = 1;",
            "p.srcStride = 0;",
            "p.dstStride = kHalf - 1;",
            "DataCopy(aUb_.Get<uint8_t>(), x_[start], p);",
            "Fence<HardEvent::V_MTE2>();",
            "Fence<HardEvent::MTE2_V>();",
        ):
            self.assertIn(expression, load)
        self.assertNotIn("Gather(", load)
        vector = self.kernel.split("  __aicore__ inline void Vector(", 1)[1].split(
            "  __aicore__ inline void Cube()", 1
        )[0]
        initial_zero = vector.index("Duplicate(aUb_.Get<uint32_t>(), uint32_t(0)")
        self.assertLess(initial_zero, vector.index("for (uint32_t start = 0;"))
        self.assertIn("LoadM1(start, half);", vector)
        self.assertIn("Fence<HardEvent::MTE3_V>();", vector)
        for seed in range(4):
            for k in (512, 2048, 4096):
                # Both halves begin dirty, as after a prior M32 output tile.
                halves = [bytearray([199] * 2048), bytearray([211] * 2048)]
                for job in range(3):
                    halves[0][:] = bytes(2048)
                    halves[1][:] = bytes(2048)
                    for start in range(0, k, 128):
                        row = bytes((start + col + 37 * seed + job) % 256 for col in range(128))
                        for block in range(4):
                            # AscendC DataCopy uses 32-byte block units.
                            target = block * (1 + 15) * 32
                            halves[0][target : target + 32] = row[block * 32 : (block + 1) * 32]
                        expected = bytearray(2048)
                        for col, value in enumerate(row):
                            expected[(col // 32) * 16 * 32 + col % 32] = value
                        self.assertEqual(halves[0], expected)
                        self.assertEqual(halves[1], bytes(2048))
                    # Simulate a non-M1 job before the next reinitialization.
                    halves[0][:] = bytes([97] * 2048)
                    halves[1][:] = bytes([123] * 2048)

    def test_cross_core_slots_drain_between_jobs(self):
        """Finite token model, not a timing or hardware pipeline simulation."""
        for buffers in (1, 2):
            for k_tiles in (1, 3, 4, 16, 32):
                for seed in range(8):
                    rng = random.Random(seed)
                    sequences = [_protocol_actions(buffers, k_tiles, 3, actor) for actor in range(3)]
                    positions, tokens, occupied = [0, 0, 0], set(), {}
                    while any(positions[i] < len(sequences[i]) for i in range(3)):
                        actors = [0, 1, 2]
                        rng.shuffle(actors)
                        for actor in actors:
                            if positions[actor] == len(sequences[actor]):
                                continue
                            action, key = sequences[actor][positions[actor]]
                            if action == "wait" and key not in tokens:
                                continue
                            if action == "set":
                                self.assertNotIn(key, tokens)
                                tokens.add(key)
                            elif action == "wait":
                                tokens.remove(key)
                            elif action == "write":
                                slot, peer, job, ki = key
                                self.assertNotIn((slot, peer), occupied)
                                occupied[slot, peer] = (job, ki)
                            else:
                                slot, job, ki = key
                                for peer in (1, 2):
                                    self.assertEqual(occupied.pop((slot, peer)), (job, ki))
                            positions[actor] += 1
                            break
                        else:
                            self.fail("cross-core token model deadlocked")
                    self.assertFalse(tokens)
                    self.assertFalse(occupied)

    def test_new_namespace_preserves_legacy_diagnostic_launchers(self):
        launch = (SOURCE / "launch.h").read_text(encoding="utf-8")
        cmake = (SOURCE / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("namespace vq2a8_v3", launch)
        for name in ("Launch", "LaunchGrouped", "LaunchGroupedPipeline", "LaunchGroupedV3"):
            self.assertRegex(launch, rf"void {name}\(")
            self.assertRegex(self.kernel, rf"void {name}\(")
        for name in ("vq2a8_v3_prepared", "vq2a8_v3_prepared_pipeline"):
            body = self.kernel.split(f"void {name}(", 1)[1].split("\n}", 1)[0]
            self.assertIn("op.InitBuffers(constants);", body)
            self.assertIn("op.ProcessGrouped(descriptors, jobs, groups, cores);", body)
        self.assertIn("add_library(vq2a8_ascendc_v3 SHARED torch_binding.cpp)", cmake)
        self.assertNotIn("add_subdirectory", cmake)

    def test_cmake_consumes_builder_selected_sdk_recipe(self):
        cmake = (SOURCE / "CMakeLists.txt").read_text(encoding="utf-8")
        commands = (REPO / "tools/build_vq2a8_ascendc_v2.py").read_text(encoding="utf-8")
        self.assertIn("-DVQ2A8_ASCENDC_CMAKE={cmake_file}", commands)
        self.assertIn('include("${VQ2A8_ASCENDC_CMAKE}")', cmake)
        self.assertIn('if(NOT EXISTS "${VQ2A8_ASCENDC_CMAKE}")', cmake)
        self.assertNotIn("ASCENDC_CMAKE_DIR", cmake)
        # Discovery is only a direct-CMake fallback, never an override of the
        # build helper's already-probed and manifest-pinned recipe.
        discovery = cmake.index("if(NOT VQ2A8_ASCENDC_CMAKE)")
        self.assertLess(discovery, cmake.index("foreach(candidate"))
        self.assertLess(cmake.index("endforeach()"), cmake.index('include("${VQ2A8_ASCENDC_CMAKE}")'))
        self.assertIn("aarch64-linux/asc/cmake/ascendc.cmake", cmake)
        self.assertIn("x86_64-linux/asc/cmake/ascendc.cmake", cmake)
        self.assertIn('LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"', cmake)

    def test_prepared_native_launch_records_all_consumer_owners(self):
        binding = (SOURCE / "torch_binding.cpp").read_text(encoding="utf-8")
        prepared = binding.split("void GroupedProjectionOut(", 1)[1].split("TORCH_LIBRARY(", 1)[0]
        self.assertIn("NPUCachingAllocator::recordStream", binding)
        self.assertIn("RecordInputStream(owners, npuStream);", prepared)
        self.assertIn("RecordInputStream({descriptors, constants}, npuStream);", prepared)
        self.assertLess(prepared.index("RecordInputStream(owners"), prepared.index("command.SetCustomHandler"))
        self.assertIn("[stream, blocks, descriptors, constants, owners, jobs, groups, pipeline]", prepared)

    def test_host_header_constants_match_original_layout(self):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            self.skipTest("No host C++ compiler; source/byte-model checks do not replace this compile test.")
        program = r"""#include "csrc/vq2a8_ascendc/layout.h"
#include "csrc/vq2a8_ascendc_v3/layout.h"
#include <array>
#include <cassert>
int main() {
  namespace old = vq2a8_ascendc;
  namespace v3 = vq2a8_v3;
  static_assert(v3::kAbiVersion == 1 && v3::kJobWords == 12 && v3::kConstantWords == 6656);
  static_assert(v3::kNdConstants == 0 && v3::kPackedConstants == 512 && v3::kIdConstants == 1536);
  static_assert(v3::kCodeShiftConstants == 2560 && v3::kIdShiftConstants == 3584 && v3::kPairConstants == 4608);
  std::array<uint32_t, v3::kConstantWords> words{};
  v3::FillConstantWords(words.data());
  for (uint32_t i = 0; i < 512; ++i) assert(words[v3::kNdConstants + i] == old::NdGatherOffset(i));
  for (uint32_t i = 0; i < 1024; ++i) {
    assert(words[v3::kPackedConstants + i] == old::PackedGatherOffset(i));
    assert(words[v3::kIdConstants + i] == old::IdGatherOffset(i));
    assert(words[v3::kCodeShiftConstants + i] == (i % 8) * 4);
    assert(words[v3::kIdShiftConstants + i] == (i % 4) * 8);
  }
  for (uint32_t i = 0; i < 2048; ++i) assert(words[v3::kPairConstants + i] == old::PairGatherOffset(i));
  for (int64_t m = -1; m <= 34; ++m) {
    assert(v3::ValidDimensions(m, 4096, 2048, 8) == old::ValidDimensions(m, 4096, 2048, 8));
  }
}
"""
        with tempfile.TemporaryDirectory(prefix="vq2a8-v3-host-") as directory:
            source = Path(directory) / "layout.cpp"
            executable = Path(directory) / "layout.exe"
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
                    str(executable),
                ],
                check=True,
            )
            subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    unittest.main()
