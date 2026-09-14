# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Narrow development-package exceptions; CPU metadata checks, not NPU proof."""

import ast
import copy
import inspect
import json
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from tools import validate_vq2a8_v023_environment as environment

TORCH_NPU_DEV = "2.10.0.post4.dev20260715"
ASCEND_SCM = "0.23.1.dev5+g32c3714e4"
PIP_CLEAN = "No broken requirements found.\n"


def _packages(*, torch_npu=TORCH_NPU_DEV, ascend="0.23.0+vq2a8.v023"):
    return {
        "vllm": "0.23.0+empty",
        "vllm-ascend": ascend,
        "torch": "2.10.0+cpu",
        "torch-npu": torch_npu,
        "transformers": "5.5.4",
        "triton-ascend": "3.2.2",
        "fastapi": "0.123.0",
    }


def _proof(*, version=ASCEND_SCM, verified=True):
    return {"version": version, "verified": verified, "checks": ["test-proof"], "errors": []}


def _mismatch(package="torch-npu", required="==2.10.0.post4", actual=TORCH_NPU_DEV):
    return f"vllm-ascend 0.23.0 has requirement {package}{required}, but you have {package} {actual}."


def _check(stdout, *, stderr="", returncode=1, packages=None, proof=None):
    return environment.classify_pip_check(
        returncode, stdout, stderr, _packages() if packages is None else packages, ascend_source=proof
    )


def test_exact_releases_need_no_development_exception():
    values = _packages(torch_npu="2.10.0.post4")
    assert environment.accepted_version_differences(values) == []
    assert environment.stack_errors(values, (3, 11, 10), "Linux") == []


def test_explicit_torch_npu_development_build_is_narrowly_reported_not_rewritten():
    values = _packages()
    original = copy.deepcopy(values)
    differences = environment.accepted_version_differences(values)
    assert len(differences) == 1
    assert differences[0]["package"] == "torch-npu"
    assert differences[0]["required"] == "==2.10.0.post4"
    assert differences[0]["actual"] == TORCH_NPU_DEV
    assert differences[0]["reason"]
    assert environment.stack_errors(values, (3, 11, 10), "Linux") == []
    assert values == original


@pytest.mark.parametrize(
    "bad",
    [
        "2.10.0.post4.dev20260714",
        "2.10.0.post4.dev20260716",
        "2.10.0.post4.dev202607150",
        "2.10.0.post4.dev20260715+local",
        "2.10.0.post3.dev20260715",
        "2.10.0.post5.dev20260715",
        "2.10.0.dev20260715",
        "2.11.0.post4.dev20260715",
        "2.10.0.post4rc1",
        "invalid",
        None,
    ],
)
def test_other_torch_npu_development_or_missing_versions_still_block(bad):
    values = _packages(torch_npu=bad)
    assert environment.accepted_version_differences(values) == []
    assert any("torch-npu" in item for item in environment.stack_errors(values, (3, 11), "Linux"))


@pytest.mark.parametrize(
    "package,actual",
    [
        ("vllm", "0.26.0+empty"),
        ("torch", "2.10.0.dev20260715"),
        ("triton-ascend", "3.2.2.dev20260729205041"),
        ("transformers", "5.5.5"),
        ("fastapi", "0.124.0"),
    ],
)
def test_torch_exception_never_relaxes_other_package_pins(package, actual):
    values = _packages()
    values[package] = actual
    assert {item["package"] for item in environment.accepted_version_differences(values)} == {"torch-npu"}
    assert any(package in item for item in environment.stack_errors(values, (3, 11), "Linux"))


def test_both_exceptions_require_and_keep_matching_verified_ascend_proof():
    values, proof = _packages(ascend=ASCEND_SCM), _proof()
    original = copy.deepcopy((values, proof))
    by_package = {
        item["package"]: item for item in environment.accepted_version_differences(values, ascend_source=proof)
    }
    assert set(by_package) == {"torch-npu", "vllm-ascend"}
    assert by_package["vllm-ascend"]["required"] == "==0.23.0"
    assert by_package["vllm-ascend"]["actual"] == ASCEND_SCM
    assert by_package["vllm-ascend"]["reason"]
    assert environment.stack_errors(values, (3, 11), "Linux", ascend_source=proof) == []
    assert (values, proof) == original


@pytest.mark.parametrize(
    "proof",
    [
        None,
        {},
        _proof(verified=False),
        _proof(verified=1),
        _proof(verified="true"),
        _proof(version="0.23.1.dev1+gabcdef0"),
    ],
)
def test_unverified_or_mismatched_ascend_proof_cannot_use_exception(proof):
    values = _packages(ascend=ASCEND_SCM)
    differences = environment.accepted_version_differences(values, ascend_source=proof)
    assert {item["package"] for item in differences} == {"torch-npu"}
    assert any(
        "vllm-ascend" in item for item in environment.stack_errors(values, (3, 11), "Linux", ascend_source=proof)
    )


@pytest.mark.parametrize("actual", ["0.23.1", "0.23.2.dev5+g32c3714e4", "0.26.1.dev5+g32c3714e4"])
def test_even_proof_shaped_input_cannot_relax_ascend_release_family(actual):
    values = _packages(ascend=actual)
    proof = _proof(version=actual)
    assert {item["package"] for item in environment.accepted_version_differences(values, ascend_source=proof)} == {
        "torch-npu"
    }
    assert environment.stack_errors(values, (3, 11), "Linux", ascend_source=proof)


def test_report_and_worker_gate_keep_actual_packages_warnings_and_no_device_claim(monkeypatch):
    _mock_metadata(monkeypatch, _packages(ascend=ASCEND_SCM), _proof())
    for report in (environment.environment_report(), environment.require_v023_stack()):
        assert report["packages"] == _packages(ascend=ASCEND_SCM)
        assert report["errors"] == []
        assert report["warnings"]
        assert {item["package"] for item in report["accepted_version_differences"]} == {"torch-npu", "vllm-ascend"}
        assert report["device_execution_verified"] is False
        assert report["model_integration_verified"] is False


@pytest.mark.parametrize("stdout", [PIP_CLEAN, "\n" + PIP_CLEAN + "\n"])
def test_clean_pip_check_preserves_success(stdout):
    result = _check(stdout, returncode=0)
    assert result["exit"] == 0
    assert result["status"] == "passed"
    assert result["accepted_differences"] == []
    assert result["blocking_issues"] == []
    assert result["stdout"] == stdout
    assert result["stderr"] == ""
    assert result["output"] == stdout


@pytest.mark.parametrize("package", ["torch-npu", "torch_npu", "Torch.Npu"])
def test_only_approved_exact_pin_pip_mismatch_is_accepted(package):
    line = _mismatch(package=package)
    result = _check(line + "\n")
    assert result["exit"] == 1  # Never pretend pip itself returned success.
    assert result["status"] == "passed_with_accepted_differences"
    assert result["accepted_differences"] == [line]
    assert result["blocking_issues"] == []
    assert result["output"] == line + "\n"


def test_both_approved_pip_mismatches_accept_without_losing_source_proof():
    values, proof = _packages(ascend=ASCEND_SCM), _proof()
    lines = [_mismatch(), _mismatch("vllm-ascend", "==0.23.0", ASCEND_SCM)]
    result = _check("\n".join(lines) + "\n", packages=values, proof=proof)
    assert result["exit"] == 1
    assert result["status"] == "passed_with_accepted_differences"
    assert result["accepted_differences"] == lines
    assert result["blocking_issues"] == []


@pytest.mark.parametrize(
    "line",
    [
        _mismatch(required="==2.10.0.post3"),
        _mismatch(required=">=2.10.0.post4"),
        _mismatch(required="~=2.10.0.post4"),
        _mismatch(required="==2.10.0.*"),
        _mismatch(required="==2.10.0.post4; python_version > '3.9'"),
        _mismatch(package="torch-npu[extra]"),
        _mismatch(required=" @ https://example.invalid/wheel.whl"),
        _mismatch(actual="2.10.0.post4.dev20260714"),
        _mismatch("torch", "==2.10.0", TORCH_NPU_DEV),
        "vllm-ascend 0.23.0 requires torch-npu, which is not installed.",
        "vllm-ascend 0.23.0 has requirement torch-npu==2.10.0.post4, but you have torch 2.10.0.post4.dev20260715.",
        "ERROR: pip failed before dependency analysis",
        "dependency check",
        _mismatch() + " unexpected trailing content",
    ],
)
def test_other_requirements_versions_and_unknown_lines_are_blocking(line):
    result = _check(line + "\n")
    assert result["status"] == "failed"
    assert result["blocking_issues"]
    assert result["accepted_differences"] == []
    assert result["exit"] == 1


def test_pip_found_version_must_match_installed_metadata():
    result = _check(_mismatch(), packages=_packages(torch_npu="2.10.0.post4"))
    assert result["status"] == "failed"
    assert result["blocking_issues"]
    assert result["accepted_differences"] == []


@pytest.mark.parametrize("proof", [None, _proof(verified=False), _proof(version="0.23.1.dev1+gabcdef0")])
def test_pip_ascend_scm_exception_also_requires_matching_verified_source(proof):
    line = _mismatch("vllm-ascend", "==0.23.0", ASCEND_SCM)
    result = _check(line, packages=_packages(ascend=ASCEND_SCM), proof=proof)
    assert result["status"] == "failed"
    assert result["blocking_issues"]
    assert not result["accepted_differences"]


@pytest.mark.parametrize("where", ["stdout", "stderr"])
def test_mixed_approved_and_unapproved_output_never_hides_blocking_issue(where):
    line, unknown = _mismatch(), "ERROR: another dependency is missing"
    stdout = line + "\n" + (unknown + "\n" if where == "stdout" else "")
    stderr = unknown + "\n" if where == "stderr" else ""
    result = _check(stdout, stderr=stderr)
    assert result["status"] == "failed"
    assert result["accepted_differences"] == [line]
    assert any(unknown in issue for issue in result["blocking_issues"])
    assert result["output"] == stdout + stderr
    assert result["stdout"] == stdout
    assert result["stderr"] == stderr


@pytest.mark.parametrize("returncode", [-15, 2, 127])
def test_abnormal_exit_cannot_be_accepted_even_with_approved_mismatch(returncode):
    result = _check(_mismatch(), returncode=returncode)
    assert result["exit"] == returncode
    assert result["status"] == "failed"
    assert result["blocking_issues"]


@pytest.mark.parametrize(
    "returncode,stdout,stderr",
    [
        (0, "", ""),
        (1, "", ""),
        (1, " \n", ""),
        (1, PIP_CLEAN, ""),
        (0, _mismatch(), ""),
        (0, "unknown output\n", ""),
        (0, PIP_CLEAN + "unknown output\n", ""),
        (0, PIP_CLEAN, "unexpected warning\n"),
    ],
)
def test_inconsistent_empty_or_unknown_pip_results_fail_closed(returncode, stdout, stderr):
    result = _check(stdout, stderr=stderr, returncode=returncode)
    assert result["exit"] == returncode
    assert result["status"] == "failed"
    assert result["blocking_issues"]


def _mock_metadata(monkeypatch, values=None, proof=None):
    values = _packages() if values is None else values
    monkeypatch.setattr(environment, "version", values.__getitem__)
    monkeypatch.setattr(environment, "ascend_source_provenance", lambda actual: proof)
    monkeypatch.setattr(environment.platform, "system", lambda: "Linux")
    monkeypatch.setattr(environment.platform, "python_version", lambda: "3.11.10")
    monkeypatch.setattr(environment, "sys", NS(executable=sys.executable, version_info=(3, 11, 10)))


def _printed_report(capsys):
    return json.loads(capsys.readouterr().out.split("VQ2A8_V023_ENVIRONMENT ", 1)[1])


@pytest.mark.parametrize("runtime_failure", [False, True])
def test_accepted_pip_difference_still_runs_real_api_gate(monkeypatch, capsys, runtime_failure):
    _mock_metadata(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["validate_vq2a8_v023_environment.py", "--audit-consistency"])
    calls = []

    def pip_check(command, **kwargs):
        assert command == [sys.executable, "-m", "pip", "check"]
        assert kwargs == {"capture_output": True, "text": True, "timeout": 120, "check": False}
        calls.append("pip")
        return NS(returncode=1, stdout=_mismatch() + "\n", stderr="")

    def runtime():
        calls.append("runtime")
        if runtime_failure:
            raise RuntimeError("scheduler API mismatch")
        return {"npu_kernel_execution": False}

    monkeypatch.setattr(environment.subprocess, "run", pip_check)
    monkeypatch.setattr(environment, "check_runtime_imports", runtime)
    assert environment.main() == int(runtime_failure)
    report = _printed_report(capsys)
    assert calls == ["pip", "runtime"]
    assert report["status"] == ("failed" if runtime_failure else "passed")
    assert report["pip_check"]["exit"] == 1
    assert report["pip_check"]["status"] == "passed_with_accepted_differences"
    assert report["packages"]["torch-npu"] == TORCH_NPU_DEV
    assert report["accepted_version_differences"]
    assert report["warnings"]
    assert report["validation_profile"] == "accepted_version_differences"
    assert report["device_execution_verified"] is False
    if runtime_failure:
        assert any("scheduler API mismatch" in error for error in report["errors"])
    else:
        assert report["runtime_imports"] == {"npu_kernel_execution": False}


def test_mixed_pip_failure_skips_runtime_even_when_torch_variant_approved(monkeypatch, capsys):
    _mock_metadata(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["validate_vq2a8_v023_environment.py", "--audit-consistency"])
    monkeypatch.setattr(
        environment.subprocess,
        "run",
        lambda *args, **kwargs: NS(returncode=1, stdout=_mismatch() + "\nMissing another package\n", stderr=""),
    )
    monkeypatch.setattr(environment, "check_runtime_imports", lambda: pytest.fail("Must not import after pip failure"))
    assert environment.main() == 1
    report = _printed_report(capsys)
    assert report["status"] == "failed"
    assert report["pip_check"]["blocking_issues"]
    assert "runtime_imports" not in report


@pytest.mark.parametrize("failure", [OSError("pip unavailable"), subprocess.TimeoutExpired("pip check", 120)])
def test_process_failure_does_not_become_accepted_variant(monkeypatch, capsys, failure):
    _mock_metadata(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["validate_vq2a8_v023_environment.py", "--audit-consistency"])

    def failed(*args, **kwargs):
        raise failure

    monkeypatch.setattr(environment.subprocess, "run", failed)
    monkeypatch.setattr(environment, "check_runtime_imports", lambda: pytest.fail("Must not import after pip failure"))
    assert environment.main() == 1
    report = _printed_report(capsys)
    assert report["status"] == "failed"
    assert any("pip check could not complete" in error for error in report["errors"])
    assert "runtime_imports" not in report


def test_metadata_only_reports_exception_without_pip_or_runtime(monkeypatch, capsys):
    _mock_metadata(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["validate_vq2a8_v023_environment.py", "--audit-consistency", "--metadata-only"])
    monkeypatch.setattr(environment.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unexpected subprocess"))
    monkeypatch.setattr(environment, "check_runtime_imports", lambda: pytest.fail("Unexpected NPU runtime import"))
    assert environment.main() == 0
    report = _printed_report(capsys)
    assert report["status"] == "metadata_pass"
    assert report["packages"]["torch-npu"] == TORCH_NPU_DEV
    assert report["accepted_version_differences"]
    assert report["warnings"]
    assert "pip_check" not in report
    assert "runtime_imports" not in report


@pytest.mark.parametrize("issue", ["runtime import failed", "missing engine entry point", "runtime API mismatch"])
def test_release_environment_uses_shared_failure_gate_before_any_device_work(monkeypatch, tmp_path, capsys, issue):
    from tools import accept_vq2a8_release as release
    from tools import validate_vq2a8_ascendc as native

    report = {
        "status": "failed",
        "errors": [issue],
        "packages": _packages(),
        "validation_profile": "runtime_only",
        "consistency_checked": False,
        "pip_check_run": False,
    }
    calls = []

    def shared_gate():
        calls.append("shared-python-check")
        return copy.deepcopy(report)

    monkeypatch.setattr(environment, "check_runtime_environment", shared_gate)
    monkeypatch.setattr(native, "require_hardware_runtime", lambda: pytest.fail("Device guard must not run"))
    monkeypatch.setattr(release, "library_evidence", lambda *args: pytest.fail("Library must not be examined"))
    monkeypatch.setattr(release.subprocess, "run", lambda *args, **kwargs: pytest.fail("No second pip invocation"))
    args = NS(worker="environment", library=tmp_path / "not-built.so", output_dir=tmp_path / "not-created")
    with pytest.raises(RuntimeError, match=issue):
        release.worker(args)
    assert calls == ["shared-python-check"]
    assert _printed_report(capsys) == report
    assert not args.output_dir.exists()


def test_release_runtime_only_check_continues_hardware_checks_and_preserves_runtime_summary(
    monkeypatch, tmp_path, capsys
):
    from tools import accept_vq2a8_release as release
    from tools import validate_vq2a8_ascendc as native

    calls = []
    runtime = {"scheduler_apis": {"plain_request_checked": True}, "npu_kernel_execution": False}
    report = {
        "status": "passed",
        "errors": [],
        "packages": _packages(),
        "validation_profile": "runtime_only",
        "consistency_checked": False,
        "pip_check_run": False,
        "runtime_imports": runtime,
    }

    def shared_gate():
        calls.append("shared-python-check")
        return copy.deepcopy(report)

    def hardware():
        calls.append("hardware-runtime")
        return {"simulator": False}

    def set_device(index):
        assert index == 0
        calls.append("set-device")

    class SmokeTensor:
        def __add__(self, value):
            assert value == 1
            return self

        def cpu(self):
            return self

        def item(self):
            return 2

    def ones(count, *, device):
        assert count == 1 and device == "npu:0"
        calls.append("smoke")
        return SmokeTensor()

    def library_evidence(path):
        assert path == args.library
        calls.append("library")
        return {"sha256": "unit-test-only", "build": {"soc": "Ascend950DT_9582"}}

    fake_torch = NS(
        ones=ones,
        npu=NS(
            set_device=set_device,
            get_device_name=lambda index: "Ascend950DT_9582",
            get_device_properties=lambda index: "unit-test-properties",
        ),
    )
    monkeypatch.setattr(environment, "check_runtime_environment", shared_gate)
    monkeypatch.setattr(native, "require_hardware_runtime", hardware)
    monkeypatch.setattr(release, "library_evidence", library_evidence)
    monkeypatch.setattr(release.subprocess, "run", lambda *args, **kwargs: pytest.fail("No second pip invocation"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_npu", NS())
    args = NS(worker="environment", library=tmp_path / "native.so", output_dir=tmp_path / "environment")
    assert release.worker(args) == 0
    assert calls == ["shared-python-check", "hardware-runtime", "set-device", "smoke", "library", "hardware-runtime"]
    assert _printed_report(capsys) == report
    saved = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert saved["status"] == "PASS"
    assert saved["runtime"] == saved["runtime_imports"] == runtime
    assert saved["pip_check_run"] is False
    assert saved["consistency_checked"] is False
    assert "pip_check" not in saved
    assert saved["packages"]["torch-npu"] == TORCH_NPU_DEV
    assert saved["validation_profile"] == "runtime_only"
    assert saved["library_sha256"] == "unit-test-only"


def test_release_worker_has_no_separate_bare_pip_or_runtime_import_gate():
    from tools import accept_vq2a8_release as release

    source = inspect.getsource(release.worker)
    tree = ast.parse(source)
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert calls.count("check_runtime_environment") == 1
    assert "check_python_environment" not in calls
    assert "subprocess.run" not in calls
    assert "check_runtime_imports" not in calls
    assert "require_v023_stack" not in calls
