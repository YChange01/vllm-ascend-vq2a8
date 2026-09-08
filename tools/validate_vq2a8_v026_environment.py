#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the VQ2A8 0.26 Python stack before model allocation. No inference PASS."""

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
import inspect
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

V026_REQUIREMENTS = {
    "vllm": "==0.26.0",
    "torch": "==2.10.0",
    "torch-npu": "==2.10.0.post4",
    "transformers": "==5.14.1",
    "triton-ascend": "==3.2.2",
    "fastapi": ">=0.133.0,<0.137.0",
}


def stack_errors(packages, python_version, system):
    errors = []
    if system != "Linux":
        errors.append("NPU execution requires Linux; PC repacking is a separate CPU workflow.")
    if not (3, 10) <= tuple(python_version[:2]) < (3, 13):
        errors.append("Use Python >=3.10,<3.13 for this Ascend release.")
    for name, spec in V026_REQUIREMENTS.items():
        try:
            actual = Version(packages.get(name) or "missing")
            if not SpecifierSet(spec).contains(actual):
                errors.append(f"{name}{spec} required, found {actual}.")
        except InvalidVersion:
            errors.append(f"{name}{spec} required, found {packages.get(name)!r}.")
    try:
        ascend = Version(packages.get("vllm-ascend") or "missing")
        if ascend.release != (0, 26, 0) or ascend < Version("0.26.0rc1"):
            errors.append(f"Use the VQ2A8 0.26 migration branch, found vllm-ascend {ascend}.")
    except InvalidVersion:
        errors.append("vllm-ascend 0.26 migration package is missing or has no usable version.")
    return errors


def environment_report():
    packages = {}
    for name in (*V026_REQUIREMENTS, "vllm-ascend"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "system": platform.system(),
        "packages": packages,
        "errors": stack_errors(packages, sys.version_info, platform.system()),
        "scope": "python_environment_only",
        "device_execution_verified": False,
        "model_integration_verified": False,
    }


def require_v026_stack():
    report = environment_report()
    if report["errors"]:
        raise RuntimeError("VQ2A8 0.26 environment mismatch: " + " ".join(report["errors"]))
    return report


def _check_structured_output_manager(manager_type):
    """Check the paired scheduler API without constructing an engine/backend."""
    advance = inspect.signature(manager_type.should_advance)
    trim = inspect.signature(manager_type.trim_reasoning_for_advance)
    try:
        advance.bind(None, None, new_token_ids=[223])
        trim.bind(None, None, [223])
    except TypeError as exc:
        raise RuntimeError(
            "Ascend 0.26 scheduler/structured-output API mismatch: "
            f"should_advance{advance}, trim_reasoning_for_advance{trim}. "
            "Check local-build version recognition and VLLM_VERSION; "
            "the paired reasoning-boundary patch must be installed before model loading."
        ) from exc
    manager = object.__new__(manager_type)
    request = SimpleNamespace(use_structured_output=False)
    if manager.should_advance(request, new_token_ids=[223]) is not False:
        raise RuntimeError("Plain generation must not advance a structured-output grammar.")
    return {
        "scope": "python_scheduler_contract_only",
        "should_advance": str(advance),
        "trim_reasoning_for_advance": str(trim),
        "plain_request_checked": True,
        "npu_kernel_execution": False,
    }


def check_scheduler_apis():
    # Install the same platform patches used by the engine before checking the
    # effective method. The upstream, unpatched method lacks new_token_ids.
    from vllm.v1.structured_output import StructuredOutputManager

    import vllm_ascend.patch.platform  # noqa: F401

    return _check_structured_output_manager(StructuredOutputManager)


def check_runtime_imports():
    """Import real 0.26 APIs, but do not load model weights or run a kernel."""
    import torch
    import torch_npu  # noqa: F401
    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs

    from vllm_ascend.models import register_model
    from vllm_ascend.patch.worker.vq2a8_offline_model import VQ2A8TP1OfflineForCausalLM
    from vllm_ascend.quantization.vq2a8_offline import offline_engine_options

    register_model()
    options = offline_engine_options(Path("/model"), Path("/artifact"))
    missing = set(options) - set(inspect.signature(EngineArgs).parameters)
    if missing or not callable(getattr(LLM, "collective_rpc", None)):
        raise RuntimeError(f"vLLM 0.26 offline APIs/options missing: {sorted(missing)}.")
    return {
        "torch_runtime_version": torch.__version__,
        "model_class": VQ2A8TP1OfflineForCausalLM.__name__,
        "engine_options_checked": sorted(options),
        "collective_rpc": True,
        "scheduler_apis": check_scheduler_apis(),
        "npu_kernel_execution": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true", help="Do not import NPU/vLLM or run pip check.")
    args = parser.parse_args()
    report = environment_report()
    if not report["errors"] and not args.metadata_only:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=120, check=False
            )
            report["pip_check"] = {"exit": result.returncode, "output": result.stdout + result.stderr}
            if result.returncode:
                report["errors"].append(
                    "pip check failed; do not start model acceptance with conflicting dependencies."
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["errors"].append(f"pip check could not complete: {type(exc).__name__}: {exc}")
        if not report["errors"]:
            try:
                report["runtime_imports"] = check_runtime_imports()
            except Exception as exc:
                report["errors"].append(f"Runtime import failed: {type(exc).__name__}: {exc}")
    report["status"] = "failed" if report["errors"] else "metadata_pass" if args.metadata_only else "passed"
    print("VQ2A8_V026_ENVIRONMENT " + json.dumps(report), flush=True)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
