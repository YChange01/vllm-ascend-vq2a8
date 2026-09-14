# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only editable provenance contracts; no imports or execution on an NPU."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tools import validate_vq2a8_v023_environment as environment

REPO = Path(__file__).resolve().parents[3]
OLD_EDITABLE_VERSION = "0.23.1.dev2+ge4f48fe5d"
CURRENT_EDITABLE_VERSION = "0.23.1.dev5+g32c3714e4"


def _packages(ascend=OLD_EDITABLE_VERSION):
    return {
        "vllm": "0.23.0+empty",
        "vllm-ascend": ascend,
        "torch": "2.10.0+cpu",
        "torch-npu": "2.10.0.post4",
        "transformers": "5.5.4",
        "triton-ascend": "3.2.2",
        "fastapi": "0.123.0",
    }


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """Real pinned framework bytes, fake metadata/Git, and no package import."""
    root = tmp_path / "editable checkout with spaces"
    root.mkdir()
    for relative in environment.V023_FRAMEWORK_BLOBS:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Windows and source archives may have CRLF; normalize like Git does.
        source = (REPO / relative).read_bytes().replace(b"\r\n", b"\n")
        target.write_bytes(source.replace(b"\n", b"\r\n"))
    (root / "vllm_ascend/__init__.py").write_text("raise RuntimeError('must not import')\n")
    # A linked worktree has a .git file, not a directory.
    (root / ".git").write_text("gitdir: /not-used-by-this-test/worktrees/v023\n")
    state = NS(
        root=root,
        direct=json.dumps({"url": root.as_uri(), "dir_info": {"editable": True}}),
        git_calls=[],
        metadata_calls=[],
        import_calls=[],
    )

    def read_text(name):
        assert name == "direct_url.json"
        return state.direct

    def distribution(name):
        state.metadata_calls.append(name)
        assert name == "vllm-ascend"
        return NS(read_text=read_text)

    def find_spec(name):
        state.import_calls.append(name)
        assert name == "vllm_ascend"
        return NS(origin=str(root / "vllm_ascend/__init__.py"))

    def git_run(command, **kwargs):
        state.git_calls.append(command)
        assert command == [
            "git",
            "-C",
            str(root.resolve()),
            "merge-base",
            "--is-ancestor",
            environment.V023_MIGRATION_COMMIT,
            "HEAD",
        ]
        assert kwargs == {"capture_output": True, "timeout": 10, "check": False}
        return NS(returncode=0, stdout=b"", stderr=b"")

    state.git_run = git_run
    monkeypatch.setattr(environment, "distribution", distribution)
    monkeypatch.setattr(environment, "find_spec", find_spec)
    monkeypatch.setattr(environment.subprocess, "run", git_run)
    return state


def _use_checkout_report(monkeypatch, checkout, installed_version=OLD_EDITABLE_VERSION):
    versions = _packages(installed_version)
    calls = []
    real_provenance = environment.ascend_source_provenance

    def source_provenance(value):
        calls.append(value)
        return real_provenance(value, repo=checkout.root)

    monkeypatch.setattr(environment, "version", versions.__getitem__)
    monkeypatch.setattr(environment, "ascend_source_provenance", source_provenance)
    monkeypatch.setattr(environment.platform, "system", lambda: "Linux")
    monkeypatch.setattr(environment.platform, "python_version", lambda: "3.11.10")
    monkeypatch.setattr(environment, "sys", NS(executable=sys.executable, version_info=(3, 11, 10)))
    return calls


@pytest.mark.parametrize(
    "installed_version",
    [OLD_EDITABLE_VERSION, CURRENT_EDITABLE_VERSION, CURRENT_EDITABLE_VERSION + ".d20260914"],
)
def test_actual_v023_scm_labels_accept_pinned_editable_source(checkout, installed_version):
    proof = environment.ascend_source_provenance(installed_version, repo=checkout.root)
    assert proof["verified"] is True
    assert proof["version"] == installed_version
    assert proof["errors"] == []
    assert proof["checks"] == [
        "editable_distribution_matches_checkout",
        "python_import_matches_checkout",
        "v023_migration_ancestor",
        "v023_framework_fingerprints",
    ]
    assert not environment.stack_errors(_packages(installed_version), (3, 11), "Linux", ascend_source=proof)
    # Editable metadata from e4f48fe5d remains usable after later source pulls:
    # the current source lineage, not equality with the metadata hash, is checked.
    assert len(checkout.git_calls) == 1
    assert installed_version.split("+g", 1)[1].split(".", 1)[0] not in checkout.git_calls[0]
    assert checkout.import_calls == ["vllm_ascend"]


@pytest.mark.parametrize(
    "label",
    [
        "0.23.1",
        "0.23.1+g32c3714e4",
        "0.23.1rc1",
        "0.23.1.dev5",
        "0.23.1.dev5+empty",
        "0.23.0.dev5+g32c3714e4",
        "0.23.2.dev5+g32c3714e4",
        "0.26.1.dev5+g32c3714e4",
        "0.23.1.dev5+g123",
        "0.23.1.dev5+g32c3714e4.d2026",
        "0.23.1.dev5+g32c3714e4.extra",
    ],
)
def test_other_releases_and_unrecognized_labels_cannot_use_exception(checkout, label):
    proof = environment.ascend_source_provenance(label, repo=checkout.root)
    assert not proof["verified"]
    assert not checkout.metadata_calls
    # Even a proof-shaped input must not relax the candidate version boundary.
    errors = environment.stack_errors(
        _packages(label), (3, 11), "Linux", ascend_source={"version": label, "verified": True}
    )
    assert any("vllm-ascend" in item for item in errors)


def test_development_version_without_matching_proof_is_still_rejected():
    for proof in (None, {"verified": False}, {"version": CURRENT_EDITABLE_VERSION, "verified": True}):
        errors = environment.stack_errors(_packages(), (3, 11), "Linux", ascend_source=proof)
        assert any("vllm-ascend" in item for item in errors)


@pytest.mark.parametrize(
    "direct",
    [
        None,
        "",
        "{incomplete",
        "null",
        "[]",
        "{}",
        '{"dir_info": null}',
        '{"url":"https://secret:token@example.com/repo","dir_info":{"editable":true}}',
    ],
)
def test_missing_or_malformed_direct_url_fails_without_leaking_content(checkout, direct):
    checkout.direct = direct
    proof = environment.ascend_source_provenance(OLD_EDITABLE_VERSION, repo=checkout.root)
    assert not proof["verified"]
    assert "editable-installed" in proof["errors"][0]
    assert "secret" not in json.dumps(proof)
    assert not checkout.import_calls
    assert not checkout.git_calls


@pytest.mark.parametrize("editable", [False, "true", 1, None])
def test_noneditable_or_loosely_truthy_metadata_is_not_provenance(checkout, editable):
    checkout.direct = json.dumps({"url": checkout.root.as_uri(), "dir_info": {"editable": editable}})
    assert not environment.ascend_source_provenance(OLD_EDITABLE_VERSION, repo=checkout.root)["verified"]


@pytest.mark.parametrize("suffix", ["?token=secret", "#fragment"])
def test_file_url_with_query_or_fragment_is_rejected(checkout, suffix):
    checkout.direct = json.dumps({"url": checkout.root.as_uri() + suffix, "dir_info": {"editable": True}})
    assert not environment.ascend_source_provenance(OLD_EDITABLE_VERSION, repo=checkout.root)["verified"]


def test_different_editable_checkout_is_rejected(checkout, tmp_path):
    checkout.direct = json.dumps({"url": (tmp_path / "other-v023").as_uri(), "dir_info": {"editable": True}})
    proof = environment.ascend_source_provenance(OLD_EDITABLE_VERSION, repo=checkout.root)
    assert not proof["verified"]
    assert not checkout.import_calls


@pytest.mark.parametrize("failure", [environment.PackageNotFoundError("vllm-ascend"), OSError("unreadable")])
def test_distribution_metadata_errors_fail_closed(checkout, monkeypatch, failure):
    def unavailable(name):
        raise failure

    monkeypatch.setattr(environment, "distribution", unavailable)
    proof = environment.ascend_source_provenance(OLD_EDITABLE_VERSION, repo=checkout.root)
    assert not proof["verified"]
    assert "editable-installed" in proof["errors"][0]
    assert not checkout.git_calls


@pytest.mark.parametrize("origin", [None, "outside/vllm_ascend/__init__.py"])
def test_missing_or_different_python_import_path_is_rejected(checkout, monkeypatch, origin):
    monkeypatch.setattr(environment, "find_spec", lambda name: NS(origin=origin))
    proof = environment.ascend_source_provenance(OLD_EDITABLE_VERSION, repo=checkout.root)
    assert not proof["verified"]
    assert "Python resolves" in proof["errors"][0]
    assert not checkout.git_calls


@pytest.mark.parametrize("failure", [ImportError("not found"), ValueError("module.__spec__ is None")])
def test_package_resolution_errors_fail_closed(checkout, monkeypatch, failure):
    def unavailable(name):
        raise failure

    monkeypatch.setattr(environment, "find_spec", unavailable)
    assert not environment.ascend_source_provenance(OLD_EDITABLE_VERSION, repo=checkout.root)["verified"]
    assert not checkout.git_calls


@pytest.mark.parametrize("returncode", [1, 128])
def test_nonancestor_or_missing_git_history_is_rejected(checkout, monkeypatch, returncode):
    monkeypatch.setattr(environment.subprocess, "run", lambda *args, **kwargs: NS(returncode=returncode))
    proof = environment.ascend_source_provenance(OLD_EDITABLE_VERSION, repo=checkout.root)
    assert not proof["verified"]
    assert "Cannot verify v023 migration ancestry" in proof["errors"][0]


@pytest.mark.parametrize("failure", [OSError("git unavailable"), subprocess.TimeoutExpired("git", 10)])
def test_git_unavailable_or_timeout_is_rejected(checkout, monkeypatch, failure):
    def failed(*args, **kwargs):
        raise failure

    monkeypatch.setattr(environment.subprocess, "run", failed)
    proof = environment.ascend_source_provenance(OLD_EDITABLE_VERSION, repo=checkout.root)
    assert not proof["verified"]
    assert "Git is unavailable or timed out" in proof["errors"][0]


@pytest.mark.parametrize("relative", environment.V023_FRAMEWORK_BLOBS)
def test_one_mixed_framework_file_rejects_even_matching_migration_ancestry(checkout, relative):
    (checkout.root / relative).write_text("# different / v0.26 framework implementation\n")
    proof = environment.ascend_source_provenance(OLD_EDITABLE_VERSION, repo=checkout.root)
    assert "v023_migration_ancestor" in proof["checks"]
    assert not proof["verified"]
    assert proof["errors"] == [f"Source differs from the verified v0.23 framework: {relative}."]


def test_missing_framework_file_is_rejected(checkout):
    relative = next(iter(environment.V023_FRAMEWORK_BLOBS))
    (checkout.root / relative).unlink()
    assert not environment.ascend_source_provenance(OLD_EDITABLE_VERSION, repo=checkout.root)["verified"]


def test_environment_report_and_worker_gate_use_same_proof_and_keep_actual_version(checkout, monkeypatch):
    calls = _use_checkout_report(monkeypatch, checkout)
    for report in (environment.environment_report(), environment.require_v023_stack()):
        assert report["errors"] == []
        assert report["packages"]["vllm-ascend"] == OLD_EDITABLE_VERSION
        assert report["ascend_source"]["version"] == OLD_EDITABLE_VERSION
        assert report["ascend_source"]["verified"] is True
        assert report["device_execution_verified"] is False
        assert report["model_integration_verified"] is False
    assert calls == [OLD_EDITABLE_VERSION, OLD_EDITABLE_VERSION]


def test_worker_gate_retains_provenance_failure_reason(checkout, monkeypatch):
    _use_checkout_report(monkeypatch, checkout)
    checkout.direct = None
    with pytest.raises(RuntimeError, match="editable-installed"):
        environment.require_v023_stack()


@pytest.mark.parametrize("vllm_version", ["0.23.1", "0.26.0+empty"])
def test_verified_ascend_source_never_relaxes_vllm_release_pin(checkout, vllm_version):
    proof = environment.ascend_source_provenance(OLD_EDITABLE_VERSION, repo=checkout.root)
    packages = _packages()
    packages["vllm"] = vllm_version
    errors = environment.stack_errors(packages, (3, 11), "Linux", ascend_source=proof)
    assert any("vllm==0.23.0" in item for item in errors)


def test_exact_v023_release_build_does_not_require_editable_metadata(checkout, monkeypatch):
    calls = _use_checkout_report(monkeypatch, checkout, "0.23.0+vq2a8.v023")
    checkout.direct = None
    assert environment.require_v023_stack()["ascend_source"] is None
    assert not calls
    assert not checkout.metadata_calls


@pytest.mark.parametrize("outcome", ["pass", "pip-fail", "runtime-fail"])
def test_verified_scm_does_not_skip_pip_or_runtime_api_gates(checkout, monkeypatch, capsys, outcome):
    _use_checkout_report(monkeypatch, checkout)
    monkeypatch.setattr(sys, "argv", ["validate_vq2a8_v023_environment.py"])
    calls = []

    def run(command, **kwargs):
        if command[0] == "git":
            return checkout.git_run(command, **kwargs)
        assert command == [sys.executable, "-m", "pip", "check"]
        calls.append("pip")
        return NS(returncode=int(outcome == "pip-fail"), stdout="dependency check\n", stderr="")

    def runtime_check():
        calls.append("runtime")
        if outcome == "runtime-fail":
            raise RuntimeError("scheduler API mismatch")
        return {"npu_kernel_execution": False}

    monkeypatch.setattr(environment.subprocess, "run", run)
    monkeypatch.setattr(environment, "check_runtime_imports", runtime_check)
    assert environment.main() == (0 if outcome == "pass" else 1)
    report = json.loads(capsys.readouterr().out.split("VQ2A8_V023_ENVIRONMENT ", 1)[1])
    assert report["packages"]["vllm-ascend"] == OLD_EDITABLE_VERSION
    assert report["ascend_source"]["verified"] is True
    assert calls == (["pip"] if outcome == "pip-fail" else ["pip", "runtime"])
    assert report["status"] == ("passed" if outcome == "pass" else "failed")


def test_metadata_only_checks_provenance_without_pip_or_runtime_imports(checkout, monkeypatch, capsys):
    _use_checkout_report(monkeypatch, checkout)
    monkeypatch.setattr(sys, "argv", ["validate_vq2a8_v023_environment.py", "--metadata-only"])

    def unexpected_runtime():
        pytest.fail("metadata-only must not import torch_npu or the real vLLM runtime")

    monkeypatch.setattr(environment, "check_runtime_imports", unexpected_runtime)
    assert environment.main() == 0
    report = json.loads(capsys.readouterr().out.split("VQ2A8_V023_ENVIRONMENT ", 1)[1])
    assert report["status"] == "metadata_pass"
    assert report["ascend_source"]["verified"] is True
    assert "runtime_imports" not in report
    assert "pip_check" not in report
    assert len(checkout.git_calls) == 1
