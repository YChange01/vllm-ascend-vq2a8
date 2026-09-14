#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check runtime imports; dependency consistency auditing is manual/opt-in."""

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
import inspect
import json
import platform
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
from urllib.request import url2pathname

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

V023_REQUIREMENTS = {
    "vllm": "==0.23.0",
    "torch": "==2.10.0",
    "torch-npu": "==2.10.0.post4",
    "transformers": "==5.5.4",
    "triton-ascend": "==3.2.2",
    "fastapi": ">=0.115.0,<0.124.0",
}
# Used only by the explicit --audit-consistency diagnostic.
# This exception is not a claim of ABI, device, or model compatibility.
# Do not generalize this to every torch-npu 2.10/post4 development build.
V023_TORCH_NPU_TEST_VERSIONS = frozenset({"2.10.0.post4.dev20260715"})

# setuptools-scm advances v0.23.0 to 0.23.1.devN after migration commits.
# This is not permission to run an arbitrary 0.23.1/0.26 framework. These
# upstream files are also pinned by test_vq2a8_v023.py; intentional framework
# changes need a compatibility review before updating this provenance fence.
V023_MIGRATION_COMMIT = "4817b8a019380c300051f7caec25b83638a8eba3"
V023_FRAMEWORK_BLOBS = {
    "vllm_ascend/core/recompute_scheduler.py": "85b2590e98b248700baeb5d4cc5e9954a54f486f",
    "vllm_ascend/patch/platform/patch_structured_output.py": "d69af3f620028751816a7c5e8a96913a982cd739",
    "vllm_ascend/worker/model_runner_v1.py": "70ef1d79a8d52d79b9d808f16257d39951a3d48b",
}
REPO = Path(__file__).resolve().parents[1]


def _is_v023_scm_version(value):
    return bool(re.fullmatch(r"0\.23\.1\.dev\d+\+g[0-9a-f]{7,40}(?:\.d\d{8})?", value or ""))


def ascend_source_provenance(installed_version, repo=REPO):
    """Verify the narrow editable-SCM exception without importing NPU code.

    Editable metadata can predate a git pull, so its SCM hash need not equal
    HEAD. Check the source actually resolved by Python and the current tree.
    Never infer provenance from a directory/branch name or a version override.
    """
    proof = {"version": installed_version, "verified": False, "checks": [], "errors": []}
    if not _is_v023_scm_version(installed_version):
        proof["errors"].append("Not a supported v023 SCM development version.")
        return proof
    repo = Path(repo).resolve()
    try:
        direct = json.loads(distribution("vllm-ascend").read_text("direct_url.json") or "null")
        if not isinstance(direct, dict) or direct.get("dir_info", {}).get("editable") is not True:
            raise ValueError("not editable")
        url = urlsplit(direct["url"])
        if url.scheme != "file" or url.netloc not in ("", "localhost") or url.query or url.fragment:
            raise ValueError("not a local directory")
        source = Path(url2pathname(url.path))
        if not source.is_absolute() or source.resolve() != repo:
            raise ValueError("different checkout")
        proof["checks"].append("editable_distribution_matches_checkout")
    except (PackageNotFoundError, OSError, ValueError, TypeError, KeyError, AttributeError):
        # Do not echo arbitrary direct_url content (may contain credentials).
        proof["errors"].append("vllm-ascend must be editable-installed from this checkout with this Python.")
        return proof
    try:
        spec = find_spec("vllm_ascend")  # Top-level lookup does not execute __init__.
        if spec is None or not spec.origin or Path(spec.origin).resolve() != repo / "vllm_ascend/__init__.py":
            raise ValueError("different import source")
        proof["checks"].append("python_import_matches_checkout")
    except (ImportError, OSError, ValueError, TypeError):
        proof["errors"].append("Python resolves vllm_ascend outside this checkout; check the active Python/PYTHONPATH.")
        return proof
    try:
        ancestry = subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", V023_MIGRATION_COMMIT, "HEAD"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        if ancestry.returncode:
            proof["errors"].append(
                "Cannot verify v023 migration ancestry; check the checkout and available Git history."
            )
            return proof
        proof["checks"].append("v023_migration_ancestor")
    except (OSError, subprocess.TimeoutExpired):
        proof["errors"].append("Cannot inspect v023 migration ancestry; Git is unavailable or timed out.")
        return proof
    for relative, expected in V023_FRAMEWORK_BLOBS.items():
        try:
            contents = (repo / relative).read_bytes().replace(b"\r\n", b"\n")
            blob = hashlib.sha1(f"blob {len(contents)}\0".encode() + contents, usedforsecurity=False).hexdigest()
        except OSError:
            blob = None
        if blob != expected:
            proof["errors"].append(f"Source differs from the verified v0.23 framework: {relative}.")
    if not proof["errors"]:
        proof["checks"].append("v023_framework_fingerprints")
        proof["verified"] = True
    return proof


def _accepted_version_difference(name, required, actual, *, ascend_source=None):
    """One policy for metadata and pip conflicts in the manual audit."""
    name = canonicalize_name(name)
    official = "==0.23.0" if name == "vllm-ascend" else V023_REQUIREMENTS.get(name)
    try:
        if official is None or SpecifierSet(required) != SpecifierSet(official):
            return None
        installed = str(Version(actual or "missing"))
    except (InvalidVersion, InvalidSpecifier):
        return None
    reason = None
    if name == "torch-npu" and installed in V023_TORCH_NPU_TEST_VERSIONS:
        reason = "Explicitly allowed Ascend950 development build for testing; runtime/device checks are still required."
    elif (
        name == "vllm-ascend"
        and _is_v023_scm_version(installed)
        and ascend_source is not None
        and ascend_source.get("verified") is True
        and ascend_source.get("version") == installed
    ):
        reason = "Editable SCM version with verified v023 source; runtime/device checks are still required."
    if reason is None:
        return None
    return {"package": name, "required": official, "actual": installed, "reason": reason}


def accepted_version_differences(packages, *, ascend_source=None):
    differences = []
    for name, required in {**V023_REQUIREMENTS, "vllm-ascend": "==0.23.0"}.items():
        difference = _accepted_version_difference(name, required, packages.get(name), ascend_source=ascend_source)
        if difference:
            differences.append(difference)
    return differences


def stack_errors(packages, python_version, system, *, ascend_source=None):
    errors = []
    if system != "Linux":
        errors.append("NPU execution requires Linux; PC repacking is a separate CPU workflow.")
    if not (3, 10) <= tuple(python_version[:2]) < (3, 13):
        errors.append("Use Python >=3.10,<3.13 for this Ascend release.")
    for name, spec in V023_REQUIREMENTS.items():
        try:
            actual = Version(packages.get(name) or "missing")
            if not SpecifierSet(spec).contains(actual) and not _accepted_version_difference(
                name, spec, str(actual), ascend_source=ascend_source
            ):
                errors.append(f"{name}{spec} required, found {actual}.")
        except InvalidVersion:
            errors.append(f"{name}{spec} required, found {packages.get(name)!r}.")
    try:
        ascend = Version(packages.get("vllm-ascend") or "missing")
        verified_scm = _accepted_version_difference("vllm-ascend", "==0.23.0", str(ascend), ascend_source=ascend_source)
        if (ascend.release != (0, 23, 0) or ascend < Version("0.23.0")) and not verified_scm:
            errors.append(f"vllm-ascend 0.23.0 or a verified v023 editable SCM build required, found {ascend}.")
            if ascend_source:
                errors.extend(ascend_source.get("errors", []))
    except InvalidVersion:
        errors.append("vllm-ascend 0.23 migration package is missing or has no usable version.")
    return errors


def environment_snapshot():
    """Record actual metadata without version, source, or global pip gates."""
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
        "validation_profile": "runtime_only",
        "consistency_checked": False,
        "pip_check_run": False,
        "errors": [],
        "scope": "python_environment_only",
        "status": "recorded",
        "device_execution_verified": False,
        "model_integration_verified": False,
    }


def environment_report():
    """Explicit consistency audit, not used by automatic model workflows."""
    snapshot = environment_snapshot()
    packages = snapshot["packages"]
    ascend_source = None
    if _is_v023_scm_version(packages["vllm-ascend"]):
        ascend_source = ascend_source_provenance(packages["vllm-ascend"])
    differences = accepted_version_differences(packages, ascend_source=ascend_source)
    return {
        **snapshot,
        "consistency_checked": True,
        "ascend_source": ascend_source,
        "accepted_version_differences": differences,
        "validation_profile": "accepted_version_differences" if differences else "official_pins",
        "warnings": [
            f"Allowing {item['package']} {item['actual']} instead of {item['required']}: {item['reason']}"
            for item in differences
        ],
        "errors": stack_errors(packages, sys.version_info, platform.system(), ascend_source=ascend_source),
        "scope": "python_environment_only",
        "device_execution_verified": False,
        "model_integration_verified": False,
    }


def require_v023_stack():
    """Legacy strict audit helper; do not call from automatic model workflows."""
    report = environment_report()
    if report["errors"]:
        raise RuntimeError("VQ2A8 0.23 environment mismatch: " + " ".join(report["errors"]))
    return report


def _accepted_pip_conflict(line, packages, *, ascend_source=None):
    # Public `pip check` output grammar. Unknown/new formats remain blockers;
    # never match a substring or discard every line mentioning torch-npu.
    match = re.fullmatch(
        r"(?P<owner>[A-Za-z0-9][A-Za-z0-9._-]*) (?P<owner_version>\S+) has requirement "
        r"(?P<requirement>.+), but you have (?P<dependency>[A-Za-z0-9][A-Za-z0-9._-]*) (?P<actual>\S+)\.",
        line,
    )
    if not match:
        return False
    try:
        requirement = Requirement(match["requirement"])
        name = canonicalize_name(requirement.name)
        if (
            name != canonicalize_name(match["dependency"])
            or requirement.url is not None
            or requirement.marker is not None
            or requirement.extras
            or Version(match["actual"]) != Version(packages.get(name) or "missing")
        ):
            return False
        Version(match["owner_version"])
        return bool(
            _accepted_version_difference(name, str(requirement.specifier), match["actual"], ascend_source=ascend_source)
        )
    except (InvalidRequirement, InvalidVersion):
        return False


def classify_pip_check(returncode, stdout, stderr, packages, *, ascend_source=None):
    """Retain raw pip failure evidence while classifying only known differences.

    The original exit status is never rewritten to success. A failed pip check
    may continue only if every issue is the same explicitly accepted version
    difference that the manual metadata audit recognizes. No pip private APIs.
    """
    result = {
        "exit": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "output": stdout + stderr,
        "accepted_differences": [],
        "blocking_issues": [],
        "status": "failed",
    }
    lines = [line.strip() for text in (stdout, stderr) for line in text.splitlines() if line.strip()]
    if returncode not in (0, 1):
        result["blocking_issues"] = [f"pip check exited with unexpected status {returncode}.", *lines]
    elif returncode == 0:
        if lines == ["No broken requirements found."]:
            result["status"] = "passed"
        else:
            result["blocking_issues"] = ["pip check returned unexpected success output.", *lines]
    elif not lines:
        result["blocking_issues"] = ["pip check failed without diagnostic output."]
    else:
        for line in lines:
            target = (
                "accepted_differences"
                if _accepted_pip_conflict(line, packages, ascend_source=ascend_source)
                else "blocking_issues"
            )
            result[target].append(line)
        if not result["blocking_issues"]:
            result["status"] = "passed_with_accepted_differences"
    return result


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


def check_python_environment(*, metadata_only=False):
    """Manual consistency audit; never invoked automatically by acceptance."""
    report = environment_report()
    if not report["errors"] and not metadata_only:
        report["pip_check_run"] = True
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=120, check=False
            )
            report["pip_check"] = classify_pip_check(
                result.returncode,
                result.stdout,
                result.stderr,
                report.get("packages", {}),
                ascend_source=report.get("ascend_source"),
            )
            if report["pip_check"]["blocking_issues"]:
                report["errors"].append(
                    "pip check failed; do not start model acceptance with conflicting dependencies."
                )
            if report["pip_check"]["accepted_differences"]:
                report.setdefault("warnings", []).append(
                    "pip check included explicitly accepted version differences; "
                    "the original exit/output are retained. Any remaining conflict still blocks execution."
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["errors"].append(f"pip check could not complete: {type(exc).__name__}: {exc}")
        if not report["errors"]:
            try:
                report["runtime_imports"] = check_runtime_imports()
            except Exception as exc:
                report["errors"].append(f"Runtime import failed: {type(exc).__name__}: {exc}")
    report["status"] = "failed" if report["errors"] else "metadata_pass" if metadata_only else "passed"
    return report


def check_runtime_environment():
    """Check imports/APIs actually used by the model, not unrelated packages."""
    report = environment_snapshot()
    try:
        report["runtime_imports"] = check_runtime_imports()
    except Exception as exc:
        report["errors"].append(f"Runtime import failed: {type(exc).__name__}: {exc}")
    report["status"] = "failed" if report["errors"] else "passed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only record metadata; with --audit-consistency also audit versions/source. No runtime imports.",
    )
    parser.add_argument(
        "--audit-consistency",
        action="store_true",
        help="Manually audit versions/source and global pip dependencies; not an acceptance prerequisite.",
    )
    args = parser.parse_args()
    if args.audit_consistency:
        report = check_python_environment(metadata_only=args.metadata_only)
    elif args.metadata_only:
        report = environment_snapshot()
    else:
        report = check_runtime_environment()
    print("VQ2A8_V023_ENVIRONMENT " + json.dumps(report), flush=True)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
