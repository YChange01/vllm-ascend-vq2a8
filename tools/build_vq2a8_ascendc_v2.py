#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build VQ2A8 operator v2 using the installed CANN MIX recipe.

No pip installation, weights, old backend library or NPU security settings are
changed. A successful build is NOT evidence of device or model correctness.
"""

from __future__ import annotations

# Avoid tools/bisect shadowing the Python standard library during direct use.
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
import platform
import subprocess
import sys
from pathlib import Path

import regex as re

from tools.vq2a8_live_log import LiveChildLog

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "csrc/vq2a8_ascendc_v2"
REFERENCE = REPO / "csrc/vq2a8_expert_reference"
ABI_VERSION = 1
MANDATORY_SOURCES = (
    "CMakeLists.txt",
    "kernel.cpp",
    "torch_binding.cpp",
    "launch.h",
    "layout.h",
)
ASCENDC_CMAKE_CANDIDATES = (
    "tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake",
    "compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake",
    "ascendc_devkit/tikcpp/samples/cmake/ascendc.cmake",
    "aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake",
    "x86_64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake",
    "aarch64-linux/asc/cmake/ascendc.cmake",
    "x86_64-linux/asc/cmake/ascendc.cmake",
    "tools/ascendc_kernel_cmake/ascendc.cmake",
    "cmake/asc/ascendc.cmake",
)
COMPILER_PREFIXES = (
    "aarch64-linux/ccec_compiler/bin",
    "x86_64-linux/ccec_compiler/bin",
    "compiler/ccec_compiler/bin",
    "tools/ccec_compiler/bin",
    "ascendc_devkit/ccec_compiler/bin",
    "ccec_compiler/bin",
    "bin",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(source: Path = SOURCE, reference: Path = REFERENCE) -> dict[str, str]:
    missing = [name for name in MANDATORY_SOURCES if not (source / name).is_file()]
    if missing:
        raise ValueError("VQ2A8 v2 source files missing: " + ", ".join(missing))
    if not (reference / "code.txt").is_file():
        raise ValueError("Reference code.txt is missing")
    if not (reference / "code.txt").stat().st_size:
        raise ValueError("code.txt is empty; cannot record the supplied reference provenance")
    paths = [
        p
        for p in source.rglob("*")
        if p.is_file()
        and not {"tests", "__pycache__", "build"}.intersection(p.relative_to(source).parts)
        and (p.suffix in {".h", ".hpp", ".cpp", ".cc", ".cce", ".cmake"} or p.name in {"CMakeLists.txt", "code.txt"})
    ]
    hashes = {"csrc/vq2a8_ascendc_v2/" + p.relative_to(source).as_posix(): sha256(p) for p in sorted(paths)}
    reference_paths = [
        p for p in reference.iterdir() if p.is_file() and (p.suffix in {".h", ".cc", ".cce"} or p.name == "code.txt")
    ]
    hashes.update({"csrc/vq2a8_expert_reference/" + p.name: sha256(p) for p in sorted(reference_paths)})
    return hashes


def discover_cmake(cann: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        candidate = explicit.resolve(strict=True)
        if not candidate.is_file() or candidate.name != "ascendc.cmake":
            raise ValueError("--ascendc-cmake must select an installed ascendc.cmake file")
        return candidate
    for name in ASCENDC_CMAKE_CANDIDATES:
        candidate = cann / name
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(
        f"CANN ascendc.cmake was not found under {cann}. "
        "Supply --ascendc-cmake with the installed SDK file; do not download a mismatched recipe."
    )


def discover_compiler(cann: Path) -> Path:
    for prefix in COMPILER_PREFIXES:
        for name in ("bisheng", "ccec"):
            candidate = cann / prefix / name
            if candidate.is_file():
                return candidate.resolve()
    raise RuntimeError(f"CANN bisheng/ccec was not found under {cann}")


def discover_soc_config(cann: Path, soc: str) -> Path:
    for prefix in ("aarch64-linux/data/platform_config", "x86_64-linux/data/platform_config", "data/platform_config"):
        directory = cann / prefix
        if directory.is_dir():
            for path in directory.glob("*.ini"):
                if path.stem.lower() == soc.lower():
                    return path.resolve()
    raise RuntimeError(
        f"Exact SoC {soc!r} is absent from installed CANN platform_config; do not substitute another SoC"
    )


def validate_soc_config(path: Path) -> dict[str, str]:
    values = dict(re.findall(r"^\s*([A-Za-z0-9_]+)\s*=\s*([^\r\n#;]+)", path.read_text(), re.MULTILINE))
    selected = {
        key: value.strip()
        for key, value in values.items()
        if key
        in {
            "SoC_version",
            "Short_SoC_version",
            "CCEC_AIC_version",
            "CCEC_VECTOR_version",
            "cube_core_cnt",
            "vector_core_cnt",
        }
    }
    # Different CANN patch levels spell the arch values differently. An explicit
    # non-c310 architecture is a hard error; absent metadata is recorded, not guessed.
    for key in ("CCEC_AIC_version", "CCEC_VECTOR_version"):
        if key in selected and "310" not in selected[key]:
            raise RuntimeError(f"{path}: {key}={selected[key]} is not an Ascend950 c310 target")
    if "cube_core_cnt" in selected and int(selected["cube_core_cnt"]) <= 0:
        raise RuntimeError("The selected SoC has no AIC cores")
    return selected


def toolchain_probe(cann: Path, soc: str, cmake_file: Path) -> dict:
    """Read SDK provenance and run compiler identification only; no device init."""
    compiler = discover_compiler(cann)
    soc_file = discover_soc_config(cann, soc)
    soc_metadata = validate_soc_config(soc_file)
    identification = {}
    for label, flag in (("version", "--version"), ("help", "--help")):
        result = subprocess.run([str(compiler), flag], capture_output=True, text=True, timeout=30, check=False)
        if result.returncode:
            raise RuntimeError(f"Compiler {flag} failed ({result.returncode}): {result.stderr[-2000:]}")
        identification[label] = (result.stdout + result.stderr)[-32768:]
    recipe_files = [cmake_file]
    for subdir in (cmake_file.parent, cmake_file.parent / "legacy_modules"):
        for name in ("function.cmake", "bisheng_intf.cmake", "util/extract_host_stub.py", "util/merge_device_obj.py"):
            candidate = subdir / name
            if candidate.is_file():
                recipe_files.append(candidate)
    recipe_text = "\n".join(p.read_text(errors="replace") for p in recipe_files)
    if "ascendc_library" not in recipe_text:
        raise RuntimeError("The selected SDK recipe does not expose ascendc_library")
    # The actual compiler checks every used API in the next build step. This
    # textual check prevents selecting an old c220-only SDK without guessing flags.
    explicit_mix = "KERNEL_TYPE_MIX_AIC_1_2" in recipe_text
    c310 = "c310" in recipe_text
    return {
        "compiler": str(compiler),
        "compiler_identification": identification,
        "soc_config": str(soc_file),
        "soc_config_sha256": sha256(soc_file),
        "soc_metadata": soc_metadata,
        "recipe_sha256": {str(p): sha256(p) for p in sorted(set(recipe_files))},
        "recipe_mentions_c310": c310,
        "recipe_mentions_explicit_mix_1_2": explicit_mix,
        "api_compilation_verified": False,
    }


def cmake_commands(
    source: Path, directory: Path, cann: Path, soc: str, cmake_file: str, npu_path: str, torch_cmake: str, jobs: int
) -> list[list[str]]:
    return [
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(directory),
            f"-DASCEND_HOME_PATH={cann}",
            f"-DSOC_VERSION={soc}",
            f"-DVQ2A8_ASCENDC_CMAKE={cmake_file}",
            f"-DTORCH_NPU_PATH={npu_path}",
            f"-DCMAKE_PREFIX_PATH={torch_cmake}",
            f"-DASCEND_PYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        ["cmake", "--build", str(directory), "--target", "vq2a8_ascendc_v2", "-j", str(jobs), "--verbose"],
    ]


def validate_options(args: argparse.Namespace, source: Path = SOURCE) -> Path:
    if not re.fullmatch(r"Ascend950[A-Za-z0-9_]+", args.soc, flags=re.IGNORECASE) or args.jobs < 1:
        raise ValueError("Require an exact Ascend950 target and positive --jobs")
    directory = args.build_dir.resolve()
    if directory == source.resolve() or source.resolve() in directory.parents:
        raise ValueError("Use an out-of-source build directory outside csrc/vq2a8_ascendc_v2")
    if directory == REPO or directory in REPO.parents:
        raise ValueError("Do not use the repository root or its ancestors as a build directory")
    cache = directory / "CMakeCache.txt"
    if cache.is_file():
        match = re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$", cache.read_text(), re.MULTILINE)
        if match and Path(match.group(1).strip()).resolve() != source.resolve():
            raise ValueError("Build directory belongs to a different CMake project; choose a new directory")
    return directory


def build(args: argparse.Namespace) -> int:
    directory = validate_options(args)
    hashes = source_hashes()
    report = {
        "status": "planned" if args.dry_run else "building",
        "implementation": "ascendc_v2",
        "abi_version": ABI_VERSION,
        "python": sys.executable,
        "soc": args.soc,
        "source_sha256": hashes,
        "build_tool_sha256": sha256(Path(__file__)),
        "build_recipe": "installed_cann_ascendc_library_mix_aic_1_2",
        "device_execution_verified": False,
        "model_execution_verified": False,
        "native_instruction_verified": False,
        "on_chip_decode_verified": False,
        "default_model_backend": "unchanged",
        "original_standalone_cce_compiled": False,
    }
    cann = args.cann.resolve()
    if args.dry_run:
        report["commands"] = cmake_commands(
            SOURCE,
            directory,
            cann,
            args.soc,
            str(args.ascendc_cmake) if args.ascendc_cmake else "<installed-CANN-ascendc.cmake>",
            "<selected-Python-torch_npu>",
            "<selected-Python-torch-cmake>",
            args.jobs,
        )
        report["toolchain_verified"] = False
        print(json.dumps(report, indent=2))
        return 0
    if platform.system() != "Linux":
        raise RuntimeError("Actual build requires Linux CANN/torch_npu; --dry-run is available on CPU/Windows")
    cann = cann.resolve(strict=True)
    cmake_file = discover_cmake(cann, args.ascendc_cmake)
    # Keep the diagnostic manifest even when configuration/compilation fails.
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "build-manifest.json"

    def save() -> None:
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        report["cann"] = str(cann)
        report["toolchain"] = toolchain_probe(cann, args.soc, cmake_file)
        import torch

        spec = importlib.util.find_spec("torch_npu")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("torch_npu is not installed in the selected Python environment")
        npu = Path(next(iter(spec.submodule_search_locations))).resolve()
        report["torch"] = torch.__version__
        report["commands"] = cmake_commands(
            SOURCE, directory, cann, args.soc, str(cmake_file), str(npu), torch.utils.cmake_prefix_path, args.jobs
        )
        save()
        for name, command in zip(("configure", "compile"), report["commands"]):
            report["current_step"] = name
            save()
            log = directory / f"{name}.log"
            print(f"VQ2A8_V2_BUILD_START={name} LOG={log}", flush=True)
            print("COMMAND " + json.dumps(command), flush=True)
            with log.open("w", encoding="utf-8") as stream, LiveChildLog(log, f"vq2a8-v2-{name}"):
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f"{name} failed ({result.returncode}); inspect {log}")
        library = directory / "libvq2a8_ascendc_v2.so"
        if not library.is_file() or not library.stat().st_size:
            raise RuntimeError(f"Native library not produced: {library}")
        if source_hashes() != hashes:
            raise RuntimeError("VQ2A8 v2 sources changed during compilation; rerun the build")
        report.update(status="built", library=str(library), library_sha256=sha256(library))
        report["toolchain"]["api_compilation_verified"] = True
        save()
        print(f"VQ2A8_V2_BUILD=PASS LIBRARY={library} REPORT={manifest}", flush=True)
        print("DEVICE_EXECUTION_VERIFIED=False MODEL_EXECUTION_VERIFIED=False", flush=True)
        return 0
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        save()
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann-9.1.0"))
    parser.add_argument("--soc", required=True)
    parser.add_argument("--build-dir", type=Path, default=REPO / "build/vq2a8-ascendc-v2")
    parser.add_argument("--ascendc-cmake", type=Path, help="Override SDK recipe discovery with an installed file")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print source hashes and planned commands only; write nothing"
    )
    return parser.parse_args(argv)


def main() -> int:
    try:
        return build(parse_args())
    except (OSError, RuntimeError, ValueError, ImportError, subprocess.TimeoutExpired) as exc:
        print(f"VQ2A8_V2_BUILD=FAIL ERROR={exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
