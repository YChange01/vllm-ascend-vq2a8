# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU build-contract tests; these do not replace CANN/NPU verification."""

import ast
import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import regex as re

ROOT = Path(__file__).resolve().parents[3]
NATIVE = ROOT / "csrc" / "vq2a8_ascendc_v4_v2"


class TestCleanNativeBuild(unittest.TestCase):
    def test_build_option_is_centralized_and_disabled_by_default(self):
        spec = importlib.util.spec_from_file_location("vq2_build_envs", ROOT / "vllm_ascend" / "envs.py")
        envs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(envs)
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(envs.VLLM_ASCEND_BUILD_VQ2A8)
        with patch.dict(os.environ, {"VLLM_ASCEND_BUILD_VQ2A8": "1"}):
            self.assertTrue(envs.VLLM_ASCEND_BUILD_VQ2A8)
        setup = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn("envs.VLLM_ASCEND_BUILD_VQ2A8", setup)
        self.assertIn("-DVLLM_ASCEND_BUILD_VQ2A8=", setup)

    def test_cmake_sources_are_complete_and_only_include_retained_operators(self):
        cmake = (NATIVE / "CMakeLists.txt").read_text(encoding="utf-8")
        listed = set(re.findall(r"\b\w+\.cpp\b", cmake))
        self.assertEqual(
            listed,
            {
                "kernel.cpp",
                "resident_prepare.cpp",
                "validity_kernel.cpp",
                "route_mapping_kernel.cpp",
                "select_sign_kernel.cpp",
                "torch_binding.cpp",
                "validity_binding.cpp",
                "route_mapping_binding.cpp",
                "select_sign_binding.cpp",
            },
        )
        self.assertEqual(listed, {p.name for p in NATIVE.glob("*.cpp")})
        self.assertNotIn("tools/", cmake)
        self.assertNotIn("find_package(Torch", cmake)
        root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("include(${CMAKE_CURRENT_LIST_DIR}/cmake/vq2a8.cmake)", root_cmake)
        gate = (ROOT / "cmake/vq2a8.cmake").read_text(encoding="utf-8")
        self.assertIn("add_dependencies(vllm_ascend_C vq2a8_ascendc_v4_v2)", gate)

    def test_installed_library_and_provenance_are_packaged(self):
        tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
        setup = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setup"
        )
        data = ast.literal_eval(next(kw.value for kw in setup.keywords if kw.arg == "package_data"))
        self.assertIn("libvq2a8_ascendc_v4_v2.so", data["vllm_ascend"])
        self.assertIn("VQ2A8_NOTICE", data["vllm_ascend"])
        cmake = (NATIVE / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn('LIBRARY DESTINATION "${VLLM_ASCEND_INSTALL_PATH}"', cmake)
        self.assertIn("RENAME VQ2A8_NOTICE", cmake)
        notice = (NATIVE / "NOTICE").read_text(encoding="utf-8")
        self.assertIn("not a license grant", notice)

    def test_experimental_abis_are_absent(self):
        source = "\n".join(p.read_text(encoding="utf-8") for p in NATIVE.glob("*.cpp"))
        methods = set(re.findall(r'\.def\("([a-z_]+)"', source))
        self.assertEqual(methods, {"select_sign", "project_vectorized", "metadata"})
        for rejected in (
            "swiglu_select_sign",
            "bias_dot_rows_probe",
            "runtime_guard",
            "decoder_input_plan",
            "activation_tail",
            "chunk_reuse",
            "row_reuse",
            "tile_major",
        ):
            self.assertNotIn(rejected, source)
        for retained in (
            "abi_version()",
            "activation_reorder_version()",
            "select_sign_version()",
            "route_mapping_version()",
            "layer_validity_vectorized_version()",
        ):
            self.assertIn(retained, source)

    def test_notice_copy_covers_first_native_build_and_existing_binary(self):
        tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "copy_vq2a8_notice"
        )
        # Exercise the actual packaging helper without executing setup(),
        # probing an NPU, or importing torch_npu in a CPU test environment.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "build" / "vllm_ascend"
            package.mkdir(parents=True)
            source = root / "csrc" / "vq2a8_ascendc_v4_v2" / "NOTICE"
            namespace = {"os": os, "shutil": shutil, "ROOT_DIR": str(root)}
            exec(compile(ast.Module(body=[function], type_ignores=[]), "setup.py", "exec"), namespace)
            copy_notice = namespace["copy_vq2a8_notice"]
            copy_notice(str(package))
            self.assertFalse((package / "VQ2A8_NOTICE").exists())
            (package / "libvq2a8_ascendc_v4_v2.so").write_bytes(b"test library")
            with self.assertRaises(FileNotFoundError):
                copy_notice(str(package))
            source.parent.mkdir(parents=True)
            source.write_bytes((NATIVE / "NOTICE").read_bytes())
            copy_notice(str(package))
            self.assertEqual((package / "VQ2A8_NOTICE").read_bytes(), source.read_bytes())
        for class_name, method_name in (("custom_build_info", "run"), ("cmake_build_ext", "build_extensions")):
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
            method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
            self.assertTrue(
                any(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "copy_vq2a8_notice"
                    for node in ast.walk(method)
                )
            )

    def test_retained_kernel_ownership_and_numeric_contracts(self):
        binding = (NATIVE / "torch_binding.cpp").read_text(encoding="utf-8")
        self.assertIn("NPUCachingAllocator::recordStream", binding)
        self.assertIn("[state, x, scale, bias, ids, reordered, descriptors, output, valid", binding)
        self.assertIn("RunOpApi", binding)
        self.assertNotIn("SetCustomHandler", binding)
        self.assertIn("x.dim() == 2 || x.dim() == 3", binding)
        self.assertIn("at::sort(permutation)", binding)
        self.assertIn("must be used on its construction NPU stream", binding)
        kernel = (NATIVE / "kernel.cpp").read_text(encoding="utf-8")
        self.assertIn("KERNEL_TYPE_MIX_AIC_1_2", kernel)
        self.assertNotIn("B1TileMajor", kernel)
        reorder = (NATIVE / "resident_prepare.cpp").read_text(encoding="utf-8")
        self.assertIn("Gather(gathered, input, orderOffsets", reorder)
        self.assertNotIn("RoundMode", reorder)

    @unittest.skipUnless(shutil.which("cmake"), "CMake unavailable; NPU build remains unverified")
    def test_cmake_hardware_gate(self):
        # Execute the production gate with target-creation mocks. This checks
        # CMake branching without a compiler, Torch import, or NPU toolchain.
        cases = (
            ("OFF", "Windows", "ascend910b1", "npu", True, False),
            ("ON", "Windows", "Ascend950PR_957d", "npu", False, False),
            ("ON", "Linux", "ascend910b1", "npu", False, False),
            ("ON", "Linux", "Ascend950PR_957d", "sim", False, False),
            ("ON", "Linux", "Ascend950PR_957d", "npu", True, True),
        )
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "gate.cmake"
            for enabled, system, soc, mode, success, added in cases:
                with self.subTest(enabled=enabled, system=system, soc=soc, mode=mode):
                    script.write_text(
                        "cmake_minimum_required(VERSION 3.16)\n"
                        f"set(VLLM_ASCEND_BUILD_VQ2A8 {enabled})\n"
                        f"set(CMAKE_SYSTEM_NAME {system})\n"
                        f"set(SOC_VERSION {soc})\nset(RUN_MODE {mode})\n"
                        'function(add_subdirectory directory)\nmessage(STATUS "VQ2_TARGET_ADDED")\nendfunction()\n'
                        "function(add_dependencies target dependency)\nendfunction()\n"
                        f'include("{(ROOT / "cmake/vq2a8.cmake").as_posix()}")\n',
                        encoding="utf-8",
                    )
                    result = subprocess.run(
                        [shutil.which("cmake"), "-P", str(script)], capture_output=True, text=True, check=False
                    )
                    self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
                    self.assertEqual("VQ2_TARGET_ADDED" in result.stdout, added)


if __name__ == "__main__":
    unittest.main()
