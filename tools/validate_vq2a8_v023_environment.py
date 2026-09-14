#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the VQ2A8 0.23 Python stack before model allocation. No inference PASS."""

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

V023_REQUIREMENTS = {
    "vllm": "==0.23.0",
    "torch": "==2.10.0",
    "torch-npu": "==2.10.0.post4",
    "transformers": "==5.5.4",
    "triton-ascend": "==3.2.2",
    "fastapi": ">=0.115.0,<0.124.0",
}


def stack_errors(packages, python_version, system):
    errors = []
    if system != "Linux":
        errors.append("NPU execution requires Linux; PC repacking is a separate CPU workflow.")
    if not (3, 10) <= tuple(python_version[:2]) < (3, 13):
        errors.append("Use Python >=3.10,<3.13 for this Ascend release.")
    for name, spec in V023_REQUIREMENTS.items():
        try:
            actual = Version(packages.get(name) or "missing")
            if not SpecifierSet(spec).contains(actual):
                errors.append(f"{name}{spec} required, found {actual}.")
        except InvalidVersion:
            errors.append(f"{name}{spec} required, found {packages.get(name)!r}.")
    try:
        ascend = Version(packages.get("vllm-ascend") or "missing")
        if ascend.release != (0, 23, 0) or ascend < Version("0.23.0"):
            errors.append(f"Use the VQ2A8 0.23 migration branch, found vllm-ascend {ascend}.")
    except InvalidVersion:
        errors.append("vllm-ascend 0.23 migration package is missing or has no usable version.")
    return errors


def environment_report():
    packages = {}
    for name in (*V023_REQUIREMENTS, "vllm-ascend"):
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


def require_v023_stack():
    report = environment_report()
    if report["errors"]:
        raise RuntimeError("VQ2A8 0.23 environment mismatch: " + " ".join(report["errors"]))
    return report


def _check_structured_output_manager(manager_type):
    """Check the paired scheduler API without constructing an engine/backend."""
    advance = inspect.signature(manager_type.should_advance)
    try:
        advance.bind(None, None)
    except TypeError as exc:
        raise RuntimeError(
            "Ascend 0.23 scheduler/structured-output API mismatch: "
            f"should_advance{advance}. "
            "Check local-build version recognition and VLLM_VERSION; "
            "do not install the 0.26 reasoning-boundary patch on this branch."
        ) from exc
    manager = object.__new__(manager_type)
    request = SimpleNamespace(use_structured_output=False)
    if manager.should_advance(request) is not False:
        raise RuntimeError("Plain generation must not advance a structured-output grammar.")
    return {
        "scope": "python_scheduler_contract_only",
        "should_advance": str(advance),
        "plain_request_checked": True,
        "npu_kernel_execution": False,
    }


def check_scheduler_apis():
    # Install the same platform patches used by the 0.23 engine. Its scheduler
    # calls should_advance(request), without the 0.26 new_token_ids/trim API.
    from vllm.v1.structured_output import StructuredOutputManager

    import vllm_ascend.patch.platform  # noqa: F401

    return _check_structured_output_manager(StructuredOutputManager)


def check_runtime_imports():
    """Import real 0.23 APIs, but do not load model weights or run a kernel."""
    import torch
    import torch_npu  # noqa: F401
    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    from vllm_ascend.models import register_model
    from vllm_ascend.patch.worker.vq2a8_offline_model import (
        VQ2A8TP1OfflineForCausalLM,
        VQ2A8TP2OfflineForCausalLM,
    )
    from vllm_ascend.quantization.vq2a8_offline import offline_engine_options

    register_model()
    options = offline_engine_options(Path("/model"), Path("/artifact"))
    missing = set(options) - set(inspect.signature(EngineArgs).parameters)
    if missing or not callable(getattr(LLM, "collective_rpc", None)):
        raise RuntimeError(f"vLLM 0.23 offline APIs/options missing: {sorted(missing)}.")
    device_mapper = getattr(current_platform, "device_id_to_physical_device_id", None)
    if not callable(device_mapper):
        raise RuntimeError("vLLM 0.23 platform must provide device_id_to_physical_device_id for TP2.")
    inspect.signature(device_mapper).bind(0)
    return {
        "torch_runtime_version": torch.__version__,
        "model_class": VQ2A8TP1OfflineForCausalLM.__name__,
        "tp2_model_class": VQ2A8TP2OfflineForCausalLM.__name__,
        "engine_options_checked": sorted(options),
        "collective_rpc": True,
        "device_mapping_api": "device_id_to_physical_device_id",
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
    print("VQ2A8_V023_ENVIRONMENT " + json.dumps(report), flush=True)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
