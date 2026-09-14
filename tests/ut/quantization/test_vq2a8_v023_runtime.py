# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default execution checks actual APIs, never metadata consistency or pip."""

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tools import validate_vq2a8_v023_environment as environment

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def nonmatching_stack(monkeypatch):
    """Deliberately outside the old whitelist; this is not an NPU runtime."""
    packages = {
        "vllm": "0.26.0+empty",
        "vllm-ascend": "0.23.1.dev99+gabcdef012",
        "torch": "2.11.0.dev20260716",
        "torch-npu": "2.10.0.post4.dev20260716",
        "transformers": "5.5.5",
        "triton-ascend": "3.2.2.dev20260729205041",
        "fastapi": "0.137.0",
    }
    monkeypatch.setattr(environment, "version", packages.__getitem__)
    monkeypatch.setattr(environment.platform, "system", lambda: "Linux")
    monkeypatch.setattr(environment.platform, "python_version", lambda: "3.11.10")
    monkeypatch.setattr(environment, "sys", NS(executable=sys.executable, version_info=(3, 11, 10)))

    def forbidden(*args, **kwargs):
        pytest.fail("Default execution must not run consistency/provenance/pip checks")

    for name in (
        "stack_errors",
        "ascend_source_provenance",
        "accepted_version_differences",
        "check_python_environment",
        "require_v023_stack",
        "environment_report",
        "distribution",
        "find_spec",
    ):
        monkeypatch.setattr(environment, name, forbidden)
    # This also ensures a pre-existing global pip conflict cannot gate the run:
    # querying pip or starting an external process is forbidden altogether.
    monkeypatch.setattr(environment.subprocess, "run", forbidden)
    return packages


def _assert_runtime_only(report, packages):
    assert report["packages"] == packages
    assert report["validation_profile"] == "runtime_only"
    assert report["consistency_checked"] is False
    assert report["pip_check_run"] is False
    assert report["device_execution_verified"] is False
    assert report["model_integration_verified"] is False
    assert "pip_check" not in report
    assert not report.get("accepted_version_differences")
    assert not report.get("ascend_source")


def _printed_report(capsys):
    return json.loads(capsys.readouterr().out.split("VQ2A8_V023_ENVIRONMENT ", 1)[1])


def test_snapshot_records_nonmatching_versions_without_audit_or_runtime(monkeypatch, nonmatching_stack):
    monkeypatch.setattr(environment, "check_runtime_imports", lambda: pytest.fail("Snapshot must not import runtime"))
    report = environment.environment_snapshot()
    _assert_runtime_only(report, nonmatching_stack)
    assert report["status"] == "recorded"
    assert report["errors"] == []
    assert "runtime_imports" not in report


@pytest.mark.parametrize("missing", ["vllm", "vllm-ascend", "torch-npu"])
def test_snapshot_missing_package_is_recorded_not_certified(monkeypatch, nonmatching_stack, missing):
    def version(name):
        if name == missing:
            raise environment.PackageNotFoundError(name)
        return nonmatching_stack[name]

    monkeypatch.setattr(environment, "version", version)
    report = environment.environment_snapshot()
    _assert_runtime_only(report, {**nonmatching_stack, missing: None})
    assert report["status"] == "recorded"
    assert report["errors"] == []
    assert "runtime_imports" not in report


def test_runtime_environment_checks_real_api_hook_despite_nonmatching_metadata(monkeypatch, nonmatching_stack):
    calls = []

    def runtime():
        calls.append("runtime")
        return {"scheduler_apis": {"plain_request_checked": True}, "npu_kernel_execution": False}

    monkeypatch.setattr(environment, "check_runtime_imports", runtime)
    report = environment.check_runtime_environment()
    assert calls == ["runtime"]
    _assert_runtime_only(report, nonmatching_stack)
    assert report["status"] == "passed"
    assert report["errors"] == []
    assert report["runtime_imports"]["scheduler_apis"]["plain_request_checked"] is True


@pytest.mark.parametrize("failure", [ImportError("missing runtime dependency"), RuntimeError("scheduler API mismatch")])
def test_runtime_environment_still_fails_on_actual_import_or_api_error(monkeypatch, nonmatching_stack, failure):
    def runtime():
        raise failure

    monkeypatch.setattr(environment, "check_runtime_imports", runtime)
    report = environment.check_runtime_environment()
    _assert_runtime_only(report, nonmatching_stack)
    assert report["status"] == "failed"
    assert any(str(failure) in error for error in report["errors"])
    assert "runtime_imports" not in report


@pytest.mark.parametrize("metadata_only", [False, True])
def test_default_cli_does_not_audit_versions_provenance_or_global_pip(
    monkeypatch, nonmatching_stack, capsys, metadata_only
):
    monkeypatch.setattr(
        sys, "argv", ["validate_vq2a8_v023_environment.py", *(["--metadata-only"] if metadata_only else [])]
    )
    calls = []

    def runtime():
        assert not metadata_only
        calls.append("runtime")
        return {"npu_kernel_execution": False}

    monkeypatch.setattr(environment, "check_runtime_imports", runtime)
    assert environment.main() == 0
    report = _printed_report(capsys)
    _assert_runtime_only(report, nonmatching_stack)
    assert report["status"] == ("recorded" if metadata_only else "passed")
    assert calls == ([] if metadata_only else ["runtime"])
    assert report["errors"] == []


def test_default_cli_runtime_failure_remains_nonzero(monkeypatch, nonmatching_stack, capsys):
    monkeypatch.setattr(sys, "argv", ["validate_vq2a8_v023_environment.py"])

    def runtime():
        raise ImportError("required operator module unavailable")

    monkeypatch.setattr(environment, "check_runtime_imports", runtime)
    assert environment.main() == 1
    report = _printed_report(capsys)
    _assert_runtime_only(report, nonmatching_stack)
    assert report["status"] == "failed"
    assert any("required operator module unavailable" in error for error in report["errors"])


def test_default_release_worker_uses_runtime_check_and_fails_before_device(
    monkeypatch, nonmatching_stack, tmp_path, capsys
):
    from tools import accept_vq2a8_release as release
    from tools import validate_vq2a8_ascendc as native

    def runtime():
        raise RuntimeError("runtime API unavailable")

    monkeypatch.setattr(environment, "check_runtime_imports", runtime)
    monkeypatch.setattr(native, "require_hardware_runtime", lambda: pytest.fail("Runtime failure must precede device"))
    monkeypatch.setattr(release, "library_evidence", lambda path: pytest.fail("Runtime failure must precede library"))
    args = NS(worker="environment", library=tmp_path / "library.so", output_dir=tmp_path / "result")
    with pytest.raises(RuntimeError, match="runtime API unavailable"):
        release.worker(args)
    report = _printed_report(capsys)
    _assert_runtime_only(report, nonmatching_stack)
    assert report["status"] == "failed"
    assert not args.output_dir.exists()


@pytest.mark.parametrize("failure_stage", ["hardware", "device", "smoke", "library", "soc"])
def test_release_runtime_only_mode_retains_hardware_and_library_failure_gates(
    monkeypatch, nonmatching_stack, tmp_path, capsys, failure_stage
):
    from tools import accept_vq2a8_release as release
    from tools import validate_vq2a8_ascendc as native

    calls = []

    def runtime():
        calls.append("runtime")
        return {"npu_kernel_execution": False}

    def hardware():
        calls.append("hardware")
        if failure_stage == "hardware":
            raise RuntimeError("hardware runtime unavailable")
        return {"simulator": False}

    def set_device(index):
        assert index == 0
        calls.append("device")
        if failure_stage == "device":
            raise RuntimeError("NPU device initialization failed")

    class SmokeTensor:
        def __add__(self, value):
            assert value == 1
            return self

        def cpu(self):
            return self

        def item(self):
            calls.append("smoke")
            return 1 if failure_stage == "smoke" else 2

    def library(path):
        assert path == args.library
        calls.append("library")
        if failure_stage == "library":
            raise ValueError("Native sources changed since build")
        return {"sha256": "cpu-test-only", "build": {"soc": "Ascend950DT_9582"}}

    def device_name(index):
        assert index == 0
        calls.append("soc")
        return "Ascend950DT_9574" if failure_stage == "soc" else "Ascend950DT_9582"

    monkeypatch.setattr(environment, "check_runtime_imports", runtime)
    monkeypatch.setattr(native, "require_hardware_runtime", hardware)
    monkeypatch.setattr(release, "library_evidence", library)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        NS(
            npu=NS(set_device=set_device, get_device_name=device_name),
            ones=lambda *args, **kwargs: SmokeTensor(),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", NS())
    args = NS(worker="environment", library=tmp_path / "library.so", output_dir=tmp_path / "result")
    with pytest.raises((RuntimeError, ValueError)):
        release.worker(args)
    order = ["runtime", "hardware", "device", "smoke", "library", "soc"]
    assert calls == order[: order.index(failure_stage) + 1]
    report = _printed_report(capsys)
    _assert_runtime_only(report, nonmatching_stack)
    assert report["status"] == "passed"  # Python API success is not device success.
    assert not args.output_dir.exists()


def _call_names(tree):
    return {ast.unparse(node.func).split(".")[-1] for node in ast.walk(tree) if isinstance(node, ast.Call)}


def test_automatic_vq2_entrypoints_cannot_import_or_call_consistency_gate():
    forbidden = {"require_v023_stack", "check_python_environment", "stack_errors", "ascend_source_provenance"}
    checked = []
    for directory in (REPO / "tools", REPO / "vllm_ascend"):
        for path in directory.rglob("*vq2a8*.py"):
            if path.name == "validate_vq2a8_v023_environment.py":
                continue  # Explicit manual audit retains its implementation.
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = {
                item.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module == "tools.validate_vq2a8_v023_environment"
                for item in node.names
            }
            assert not imported & forbidden, str(path)
            assert not _call_names(tree) & forbidden, str(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    assert node.value != "--audit-consistency", str(path)
            checked.append(path)
    assert REPO / "tools/accept_vq2a8_release.py" in checked
    assert REPO / "tools/benchmark_vq2a8_offline.py" in checked


@pytest.mark.parametrize(
    "name,required_calls",
    [
        ("benchmark_vq2a8_offline.py", {"environment_snapshot", "require_hardware_runtime", "checked_model_preflight"}),
        (
            "benchmark_vq2a8_ascendc_v2.py",
            {"environment_snapshot", "require_hardware_runtime", "checked_model_preflight"},
        ),
        ("benchmark_vq2a8_ascendc_v3.py", {"environment_snapshot", "require_hardware_runtime", "library_evidence"}),
        ("quick_benchmark_vq2a8_v3.py", {"environment_snapshot", "require_hardware_runtime", "library_identity"}),
        ("serve_vq2a8_demo.py", {"environment_snapshot", "require_hardware_runtime", "checked_model_preflight"}),
        (
            "validate_vq2a8_tp1_offline.py",
            {"environment_snapshot", "require_hardware_runtime", "checked_model_preflight"},
        ),
        ("accept_vq2a8_release.py", {"check_runtime_environment", "require_hardware_runtime", "library_evidence"}),
    ],
)
def test_environment_change_does_not_remove_automatic_native_proof_checks(name, required_calls):
    tree = ast.parse((REPO / "tools" / name).read_text(encoding="utf-8"))
    assert required_calls <= _call_names(tree)
