#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the isolated V4 + v2 compute candidate; never overwrite the V4 baseline."""

from __future__ import annotations

# Keep tools/bisect out of the direct-script import path.
# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
from pathlib import Path

import regex as re

from tools.build_vq2a8_ascendc_v2 import discover_cmake, toolchain_probe
from tools.vq2a8_live_log import LiveChildLog

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "csrc/vq2a8_ascendc_v4_v2"
TARGET = "vq2a8_ascendc_v4_v2"
LIBRARY_NAME = f"lib{TARGET}.so"


def source_hashes(source=SOURCE):
    required = (
        "CMakeLists.txt",
        "kernel.cpp",
        "torch_binding.cpp",
        "grouped_binding.cpp",
        "resident_select.cpp",
        "resident_prepare.cpp",
        "activation_kernel.cpp",
        "activation_binding.cpp",
        "activation_launch.h",
        "validity_kernel.cpp",
        "validity_binding.cpp",
        "validity_launch.h",
        "route_mapping_kernel.cpp",
        "route_mapping_binding.cpp",
        "route_mapping_launch.h",
        "select_sign_kernel.cpp",
        "select_sign_binding.cpp",
        "select_sign_binding.h",
        "select_sign_launch.h",
        "activation_diagnostic_kernel.cpp",
        "activation_diagnostic_binding.cpp",
        "activation_diagnostic_launch.h",
        "runtime_guard_binding.cpp",
        "input_plan_kernel.cpp",
        "input_plan_binding.cpp",
        "input_plan_launch.h",
        "bias_dot_probe_kernel.cpp",
        "bias_dot_probe_binding.cpp",
        "bias_dot_probe_launch.h",
        "swiglu_select_sign_kernel.cpp",
        "swiglu_select_sign_binding.cpp",
        "swiglu_select_sign_binding.h",
        "swiglu_select_sign_launch.h",
        "b1_schedule.h",
        "layout.h",
        "launch.h",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise ValueError("Candidate sources missing: " + ", ".join(missing))
    return {
        path.relative_to(REPO).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.rglob("*"))
        if path.is_file()
        and not {"tests", "build", "__pycache__"}.intersection(path.relative_to(source).parts)
        and (path.suffix in {".h", ".cpp", ".cmake"} or path.name == "CMakeLists.txt")
    }


def validate_options(args):
    if not re.fullmatch(r"Ascend950[A-Za-z0-9_]+", args.soc, re.IGNORECASE) or args.jobs < 1:
        raise ValueError("Require an exact Ascend950 SoC and positive --jobs")
    directory = args.build_dir.resolve()
    if directory in (REPO, SOURCE) or directory in REPO.parents or SOURCE in directory.parents:
        raise ValueError("Choose a dedicated out-of-source build directory")
    cache = directory / "CMakeCache.txt"
    if cache.is_file():
        match = re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$", cache.read_text(), re.MULTILINE)
        if match and Path(match.group(1).strip()).resolve() != SOURCE:
            raise ValueError("Build directory belongs to another project; do not overwrite the baseline")
    for baseline in ("libvq2a8_ascendc.so", "libvq2a8_ascendc_v2.so", "libvq2a8_ascendc_v3.so"):
        if (directory / baseline).exists():
            raise ValueError("Build directory contains another backend; choose a new candidate directory")
    return directory


def cmake_commands(args, directory, recipe, npu_path, torch_cmake):
    return [
        [
            "cmake",
            "-S",
            str(SOURCE),
            "-B",
            str(directory),
            f"-DASCEND_HOME_PATH={args.cann.resolve()}",
            f"-DSOC_VERSION={args.soc}",
            f"-DVQ2A8_ASCENDC_CMAKE={recipe}",
            f"-DTORCH_NPU_PATH={npu_path}",
            f"-DCMAKE_PREFIX_PATH={torch_cmake}",
            f"-DASCEND_PYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        ["cmake", "--build", str(directory), "--target", TARGET, "-j", str(args.jobs), "--verbose"],
    ]


def build(args):
    directory = validate_options(args)
    report = {
        "status": "planned" if args.dry_run else "building",
        "implementation": "ascendc_v4_v2",
        "abi_version": 1,
        "soc": args.soc,
        "source_sha256": source_hashes(),
        "python": sys.executable,
        "device_execution_verified": False,
        "model_execution_verified": False,
        "performance_verified": False,
        "default_model_backend": "unchanged",
    }
    if args.dry_run:
        report["commands"] = cmake_commands(
            args,
            directory,
            args.ascendc_cmake or "<installed-CANN-ascendc.cmake>",
            "<selected-Python-torch_npu>",
            "<selected-Python-torch-cmake>",
        )
        print(json.dumps(report, indent=2))
        return 0
    if platform.system() != "Linux":
        raise RuntimeError("Actual compilation requires Linux CANN/torch_npu; use --dry-run on Windows")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "build-manifest.json"

    def save():
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        cann = args.cann.resolve(strict=True)
        recipe = discover_cmake(cann, args.ascendc_cmake)
        report["toolchain"] = toolchain_probe(cann, args.soc, recipe)
        import torch

        spec = importlib.util.find_spec("torch_npu")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("torch_npu is not installed in the selected Python")
        report["torch"] = torch.__version__
        report["commands"] = cmake_commands(
            args,
            directory,
            recipe,
            next(iter(spec.submodule_search_locations)),
            torch.utils.cmake_prefix_path,
        )
        for name, command in zip(("configure", "compile"), report["commands"]):
            report["current_step"] = name
            save()
            log = directory / f"{name}.log"
            print(f"V4_V2_BUILD_STAGE={name} LOG={log}", flush=True)
            print("COMMAND=" + json.dumps(command), flush=True)
            with log.open("w", encoding="utf-8") as stream, LiveChildLog(log, f"v4-v2-{name}"):
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f"{name} failed ({result.returncode}); inspect {log}")
        library = directory / LIBRARY_NAME
        if not library.is_file() or not library.stat().st_size:
            raise RuntimeError(f"Candidate library was not produced: {library}")
        if source_hashes() != report["source_sha256"]:
            raise RuntimeError("Candidate sources changed while compiling; rebuild before validation")
        report.update(
            status="built", library=str(library), library_sha256=hashlib.sha256(library.read_bytes()).hexdigest()
        )
        save()
        print(f"V4_V2_BUILD=PASS LIBRARY={library} REPORT={manifest}", flush=True)
        print("DEVICE_EXECUTION_VERIFIED=False MODEL_EXECUTION_VERIFIED=False", flush=True)
        return 0
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        save()
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soc", required=True)
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann-9.1.0"))
    parser.add_argument("--ascendc-cmake", type=Path)
    parser.add_argument("--build-dir", type=Path, default=REPO / "build/vq2a8-ascendc-v4-v2")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="print commands/hashes; do not build or write files")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        return build(parse_args(argv))
    except (OSError, RuntimeError, ValueError, ImportError, subprocess.TimeoutExpired) as exc:
        print(f"V4_V2_BUILD=FAIL ERROR={exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
