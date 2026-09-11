#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the independent, opt-in v3 library; a build is not device validation."""

from __future__ import annotations

# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import importlib.util
import json
import platform
import subprocess
from pathlib import Path

from tools.build_vq2a8_ascendc_v2 import (
    cmake_commands as sdk_commands,
)
from tools.build_vq2a8_ascendc_v2 import (
    discover_cmake,
    sha256,
    toolchain_probe,
    validate_options,
)
from tools.vq2a8_live_log import LiveChildLog

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "csrc/vq2a8_ascendc_v3"
ABI_VERSION = 1
LIBRARY_NAME = "libvq2a8_ascendc_v3.so"


def source_hashes():
    """Bind native definitions and the reused SDK build helper, not v2 repacking."""
    for name in ("CMakeLists.txt", "kernel.cpp", "torch_binding.cpp"):
        if not (SOURCE / name).is_file():
            raise ValueError(f"V3 source missing: {SOURCE / name}")
    helper = "tools/build_vq2a8_ascendc_v2.py"
    result = {helper: sha256(REPO / helper)}
    for directory in (SOURCE, REPO / "csrc/vq2a8_ascendc"):
        for path in sorted(directory.rglob("*")):
            if (
                path.is_file()
                and not {"build", "tests", "__pycache__"}.intersection(path.relative_to(directory).parts)
                and (path.suffix in {".cpp", ".h", ".hpp", ".cc", ".cce", ".cmake"} or path.name == "CMakeLists.txt")
            ):
                result[path.relative_to(REPO).as_posix()] = sha256(path)
    return result


def commands(args, directory, cmake_file, npu_path, torch_cmake):
    result = sdk_commands(
        SOURCE, directory, args.cann.resolve(), args.soc, cmake_file, npu_path, torch_cmake, args.jobs
    )
    result[1][result[1].index("--target") + 1] = "vq2a8_ascendc_v3"
    return result


def build(args):
    directory = validate_options(args, source=SOURCE)
    report = {
        "status": "planned" if args.plan_only else "building",
        "implementation": "ascendc_v3",
        "abi_version": ABI_VERSION,
        "soc": args.soc,
        "python": sys.executable,
        "source_sha256": source_hashes(),
        "build_tool_sha256": sha256(Path(__file__)),
        "device_execution_verified": False,
        "model_execution_verified": False,
        "full_model_graph_verified": False,
        "default_model_backend": "unchanged",
    }
    if args.plan_only:
        report["commands"] = commands(
            args, directory, str(args.ascendc_cmake or "<installed-ascendc.cmake>"), "<torch_npu>", "<torch-cmake>"
        )
        print(json.dumps(report, indent=2))
        return 0
    if platform.system() != "Linux":
        raise RuntimeError("Actual build requires Linux CANN; use --plan-only on CPU/Windows.")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "build-manifest.json"

    def save():
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        cann = args.cann.resolve(strict=True)
        cmake_file = discover_cmake(cann, args.ascendc_cmake)
        report["toolchain"] = toolchain_probe(cann, args.soc, cmake_file)
        import torch

        spec = importlib.util.find_spec("torch_npu")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("torch_npu missing in selected Python environment.")
        report["torch"] = torch.__version__
        report["cann"] = str(cann)
        report["commands"] = commands(
            args, directory, str(cmake_file), next(iter(spec.submodule_search_locations)), torch.utils.cmake_prefix_path
        )
        for stage, command in zip(("configure", "compile"), report["commands"]):
            report["current_stage"] = stage
            save()
            log = directory / f"{stage}.log"
            print(f"VQ2A8_V3_BUILD_START={stage} TIMEOUT_S={args.timeout} LOG={log}", flush=True)
            with log.open("w", encoding="utf-8") as stream, LiveChildLog(log, f"v3-{stage}"):
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
        library = directory / LIBRARY_NAME
        if not library.is_file() or not library.stat().st_size:
            raise RuntimeError(f"Build did not produce {library}")
        if source_hashes() != report["source_sha256"]:
            raise RuntimeError("Native sources changed during compilation; rebuild.")
        report.update(status="built", library=str(library), library_sha256=sha256(library))
        report["toolchain"]["api_compilation_verified"] = True
        save()
        print(f"VQ2A8_V3_BUILD=PASS LIBRARY={library} DEVICE_EXECUTION_VERIFIED=False", flush=True)
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
    parser.add_argument("--build-dir", type=Path, default=REPO / "build/vq2a8-ascendc-v3")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per configure/compile stage")
    parser.add_argument("--plan-only", "--dry-run", dest="plan_only", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    return args


def main():
    try:
        return build(parse_args())
    except (OSError, RuntimeError, ValueError, ImportError, subprocess.SubprocessError) as exc:
        print(f"VQ2A8_V3_BUILD=FAIL ERROR={exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
