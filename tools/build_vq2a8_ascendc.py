#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build only the standalone AscendC prototype, not vLLM or custom OPP packages."""

from __future__ import annotations

# Direct scripts must not put tools/bisect ahead of the stdlib bisect module.
# ruff: noqa: E402
import os as _bootstrap_os
import sys as _bootstrap_sys

if not __package__:
    _bootstrap_sys.path[0] = _bootstrap_os.path.dirname(
        _bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))
    )

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.vq2a8_live_log import LiveChildLog  # noqa: E402

SOURCE = REPO / "csrc/vq2a8_ascendc"


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCE.iterdir()) if p.is_file()}


def build(args):
    if platform.system() != "Linux":
        raise RuntimeError("Build on the Linux Ascend950 machine with CANN 9.1 installed.")
    if not args.soc.lower().startswith("ascend950") or args.jobs < 1:
        raise ValueError("Require an Ascend950 target and positive --jobs.")
    cann = args.cann.resolve(strict=True)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    directory = args.build_dir.resolve()
    # No in-source build or destructive cleanup. Rebuilds use this one target.
    if directory == SOURCE or SOURCE in directory.parents:
        raise ValueError("Use a build directory outside csrc/vq2a8_ascendc.")
    import torch

    spec = importlib.util.find_spec("torch_npu")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("torch_npu must be installed in the selected Python environment.")
    npu = Path(next(iter(spec.submodule_search_locations)))
    configure = [
        "cmake",
        "-S",
        str(SOURCE),
        "-B",
        str(directory),
        f"-DASCEND_HOME_PATH={cann}",
        f"-DSOC_VERSION={args.soc}",
        f"-DTORCH_NPU_PATH={npu}",
        f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    compile_cmd = ["cmake", "--build", str(directory), "--target", "vq2a8_ascendc", "-j", str(args.jobs), "--verbose"]
    report = {
        "status": "building",
        "implementation": "ascendc",
        "python": sys.executable,
        "cann": str(cann),
        "soc": args.soc,
        "torch": torch.__version__,
        "source_sha256": source_hashes(),
        "commands": [configure, compile_cmd],
        "device_execution_verified": False,
        "native_instruction_verified": False,
        "on_chip_decode_verified": False,
        "default_model_backend": "unchanged",
    }
    manifest = directory / "build-manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    for name, command in (("configure", configure), ("compile", compile_cmd)):
        log = directory / f"{name}.log"
        print(f"ASCENDC_BUILD_START={name} LOG={log}", flush=True)
        print("COMMAND " + json.dumps(command), flush=True)
        try:
            with log.open("w") as stream, LiveChildLog(log, f"ascendc-{name}"):
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        except OSError as exc:
            report.update(status="failed", failed_step=name, error=str(exc))
            manifest.write_text(json.dumps(report, indent=2) + "\n")
            raise
        if result.returncode:
            report.update(status="failed", failed_step=name, returncode=result.returncode)
            manifest.write_text(json.dumps(report, indent=2) + "\n")
            print(f"ASCENDC_BUILD=FAIL REPORT={manifest}", flush=True)
            return 1
    library = directory / "libvq2a8_ascendc.so"
    if not library.is_file():
        raise RuntimeError(f"Native shared library was not produced: {library}")
    report.update(status="built", library=str(library), library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"ASCENDC_BUILD=PASS LIBRARY={library} REPORT={manifest}", flush=True)
    print("DEVICE_EXECUTION_VERIFIED=False NATIVE_INSTRUCTION_VERIFIED=False", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cann", type=Path, default=Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.1.0"))
    )
    parser.add_argument("--soc", default="Ascend950PR_957d")
    parser.add_argument("--build-dir", type=Path, default=REPO / "build/vq2a8-ascendc")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    try:
        return build(args)
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        print(f"ASCENDC_BUILD=FAIL ERROR={exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
