# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tools import vq2a8_baseline as baseline
from tools.validate_vq2a8_tp1_acceptance import format_compact_summary


def write_fixture(tmp_path, monkeypatch):
    repo, model, artifact, report = [tmp_path / name for name in ("repo", "model", "artifact", "old-report")]
    for folder in (repo, model, artifact, report / "model-evidence"):
        folder.mkdir(parents=True)
    for path in (model / "config.json", model / "tokenizer.json", artifact / "manifest.json"):
        path.write_text("{}")
    environment = {"git": {"head": "old-head", "status": " M tools/test.py"}, "packages": {}, "source_sha256": {}}
    for name in (
        "vq2a8_triton.py",
        "vq2a8_kernel_contract.py",
        "vq2a8_reference.py",
        "vq2a8_moe.py",
        "vq2a8_offline.py",
        "vllm_ascend/patch/worker/vq2a8_offline_model.py",
        "vllm_ascend/models/deepseek_v4.py",
        "vllm_ascend/attention/dsa_v1.py",
        "vllm_ascend/ops/linear.py",
    ):
        path = repo / name if "/" in name else repo / "vllm_ascend/quantization" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# original source\n")
        environment["source_sha256"][name] = baseline.file_sha256(path)
    logits = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    runs = []
    for run in range(2):
        path = report / "model-evidence" / f"run-{run}-logits.safetensors"
        save_file({"logits": logits}, path)
        runs.append(
            {
                "run": run,
                "execution_policy": "cached",
                "finite_logits": True,
                "generated_token_ids": [7, 7, 7, 7],
                "logits_sha256": baseline.file_sha256(path),
                "logits_file": "/old/absolute/path/is/not/followed.safetensors",
                "prompt_token_ids": [1, 2],
                "prefill_tokens": 2,
                "decode_steps": 3,
                "layers_executed": 43,
            }
        )
    records = [{"type": "ENVIRONMENT", "data": environment}]
    records += [{"type": "MODEL_RESULT", "data": run} for run in runs]
    records.append(
        {
            "type": "VQ2A8_TP1_OFFLINE_EXECUTION_GATE=PASS",
            "data": {
                "runs": 2,
                "repeat_exact": True,
                "offline_execution_verified": True,
            },
        }
    )
    summary = {
        "stage": "model",
        "status": "passed",
        "probes": ["full_model"],
        "device": "npu:0",
        "physical_npu": 4,
        "model": str(model),
        "artifact": str(artifact),
        "results": [{"passed": True, "returncode": 0, "timed_out": False, "records": records}],
    }
    (report / "summary.json").write_text(json.dumps(summary))
    (report / "probe-full_model.log").write_text("retained full log\n")
    monkeypatch.setattr(baseline.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=b"capture-time\n"))
    return repo, model, artifact, report, environment, logits


def freeze_fixture(tmp_path, monkeypatch):
    repo, model, artifact, report, environment, logits = write_fixture(tmp_path, monkeypatch)
    frozen = baseline.freeze_baseline(report, tmp_path / "frozen", repo, model, artifact, 4)
    return model, artifact, report, frozen, environment, logits


def native_fixture(tmp_path, monkeypatch):
    repo, model, artifact, report, environment, logits = write_fixture(tmp_path, monkeypatch)
    library = tmp_path / "known-good.so"
    library.write_bytes(b"host fixture, not a device binary")
    identity = {"path": str(library), "sha256": baseline.file_sha256(library)}
    path = report / "summary.json"
    summary = json.loads(path.read_text())
    summary["execution_policy"] = "ascendc"
    for record in summary["results"][0]["records"]:
        data = record["data"]
        if record["type"] == "MODEL_RESULT":
            data.update(
                execution_policy="ascendc",
                expert_backend={"policy": "ascendc", "library": identity, "fallback_enabled": False},
            )
        elif record["type"] == "VQ2A8_TP1_OFFLINE_EXECUTION_GATE=PASS":
            data.update(ascendc_model_execution_verified=True, root_linear_mode="bf16")
    path.write_text(json.dumps(summary))
    return repo, model, artifact, report, environment, logits, library


def test_native_baseline_is_frozen_without_loading_or_overwriting_library(tmp_path, monkeypatch):
    repo, model, artifact, report, environment, logits, library = native_fixture(tmp_path, monkeypatch)
    frozen = baseline.freeze_baseline(report, tmp_path / "frozen", repo, model, artifact, 4, execution_policy="ascendc")
    assert (frozen.parent / "native-baseline/libvq2a8_ascendc.so").read_bytes() == library.read_bytes()
    # Offline option plumbing can change; reference and router cannot.
    environment["source_sha256"]["vq2a8_offline.py"] = "candidate-options"
    runs, values = baseline.load_baseline(frozen, environment, model, artifact, execution_policy="ascendc")
    assert runs[0]["execution_policy"] == "ascendc" and torch.equal(values[0], logits)
    environment["source_sha256"]["vq2a8_moe.py"] = "changed-router"
    with pytest.raises(ValueError, match="compute source changed"):
        baseline.load_baseline(frozen, environment, model, artifact, execution_policy="ascendc")


@pytest.mark.parametrize("change", ["fallback", "library", "gate", "root", "policy", "mixed-library"])
def test_native_baseline_rejects_missing_or_mixed_execution_evidence(tmp_path, monkeypatch, change):
    _, _, _, report, _, _, _ = native_fixture(tmp_path, monkeypatch)
    path = report / "summary.json"
    summary = json.loads(path.read_text())
    records = summary["results"][0]["records"]
    backend = records[1]["data"]["expert_backend"]
    if change == "fallback":
        backend["fallback_enabled"] = True
    elif change == "library":
        backend["library"]["sha256"] = "unknown"
    elif change == "mixed-library":
        backend["library"]["sha256"] = "f" * 64
    elif change == "gate":
        records[-1]["data"]["ascendc_model_execution_verified"] = False
    elif change == "root":
        records[-1]["data"]["root_linear_mode"] = "online_fp8_sm90"
    else:
        records[1]["data"]["execution_policy"] = "cached"
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        baseline.baseline_records(report, "ascendc")


def test_native_baseline_requires_unchanged_original_binary_and_matching_policy(tmp_path, monkeypatch):
    repo, model, artifact, report, _, _, library = native_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="cached execution policy"):
        baseline.baseline_records(report)
    library.write_bytes(b"overwritten")
    with pytest.raises(ValueError, match="missing or changed"):
        baseline.freeze_baseline(report, tmp_path / "frozen", repo, model, artifact, 4, execution_policy="ascendc")


def test_generation_comparison_requires_numeric_pass_and_does_not_claim_serving():
    data = dict(run=1, baseline_exact=True, baseline_generation_s=60.0, candidate_generation_s=20.0)
    summary = {
        "stage": "model",
        "baseline_report": "frozen",
        "results": [{"records": [{"type": "MODEL_BASELINE_RESULT", "data": data}]}],
    }
    text = format_compact_summary(summary)
    assert "observed_ratio=3.000 SCOPE=offline_validation_not_serving" in text
    data["baseline_exact"] = False
    assert "observed_ratio" not in format_compact_summary(summary)


def test_freeze_is_copy_only_with_explicit_historical_identity_limits(tmp_path, monkeypatch):
    model, artifact, report, frozen, environment, logits = freeze_fixture(tmp_path, monkeypatch)
    assert (report / "probe-full_model.log").read_bytes() == (frozen / "probe-full_model.log").read_bytes()
    snapshot = json.loads((frozen.parent / "snapshot.json").read_text())
    assert snapshot["recorded_run_git"]["status"] == " M tools/test.py"
    assert all(snapshot["source_matches_recorded"].values())
    assert "not proof of historical" in snapshot["source_snapshot_scope"]
    assert snapshot["input_identity"]["payload_hashes_verified"] is False
    runs, values = baseline.load_baseline(frozen, environment, model, artifact)
    assert torch.equal(values[0], logits)
    current = dict(runs[0])
    result = baseline.compare_baseline_run(runs[0], values[0], current, logits)
    assert result["baseline_exact"] and result["max_abs_error"] == 0
    assert result["independent_reference"] is False


@pytest.mark.parametrize("field,value", [("status", "failed"), ("stage", "moe"), ("probes", []), ("results", [])])
def test_baseline_requires_completed_full_model_pass(tmp_path, monkeypatch, field, value):
    *_, report, _, _ = write_fixture(tmp_path, monkeypatch)
    path = report / "summary.json"
    summary = json.loads(path.read_text())
    summary[field] = value
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="successful full-model"):
        baseline.baseline_records(report)


def test_tampered_or_missing_logits_rejected_before_model_initialization(tmp_path, monkeypatch):
    *_, report, _, _ = write_fixture(tmp_path, monkeypatch)
    path = report / "model-evidence/run-0-logits.safetensors"
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        baseline.baseline_records(report)


@pytest.mark.parametrize("change", ["package", "compute", "model", "snapshot_log"])
def test_preflight_rejects_changed_environment_compute_or_snapshot(tmp_path, monkeypatch, change):
    model, artifact, _, frozen, environment, _ = freeze_fixture(tmp_path, monkeypatch)
    if change == "package":
        environment["packages"]["torch-npu"] = "different"
    elif change == "compute":
        environment["source_sha256"]["vq2a8_triton.py"] = "different"
    elif change == "model":
        (model / "config.json").write_text('{"changed": true}')
    else:
        (frozen / "probe-full_model.log").write_text("changed")
    with pytest.raises(ValueError):
        baseline.load_baseline(frozen, environment, model, artifact)


def test_dirty_capture_does_not_claim_changed_source_matches_old_hash(tmp_path, monkeypatch):
    repo, model, artifact, report, environment, _ = write_fixture(tmp_path, monkeypatch)
    (repo / "vllm_ascend/quantization/vq2a8_triton.py").write_text("# modified at capture\n")
    frozen = baseline.freeze_baseline(report, tmp_path / "frozen", repo, model, artifact, 4)
    snapshot = json.loads((frozen.parent / "snapshot.json").read_text())
    assert snapshot["source_matches_recorded"]["vq2a8_triton.py"] is False
    assert snapshot["recorded_source_sha256"] == environment["source_sha256"]
    assert (frozen.parent / "worktree-at-capture.patch").read_text() == "capture-time\n"


def test_freeze_refuses_overwrite_nested_copy_and_device_mismatch(tmp_path, monkeypatch):
    repo, model, artifact, report, _, _ = write_fixture(tmp_path, monkeypatch)
    for dest, device in ((report / "nested", 4), (tmp_path / "different", 5)):
        with pytest.raises(ValueError):
            baseline.freeze_baseline(report, dest, repo, model, artifact, device)
    destination = tmp_path / "exists"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        baseline.freeze_baseline(report, destination, repo, model, artifact, 4)


@pytest.mark.parametrize("change", ["logits", "tokens", "prompt", "dtype", "shape"])
def test_historical_comparison_is_exact_not_tolerance_based(tmp_path, monkeypatch, change):
    *_, report, _, logits = write_fixture(tmp_path, monkeypatch)
    _, runs, _ = baseline.baseline_records(report)
    previous, current, actual = runs[0], dict(runs[0]), logits.clone()
    if change == "logits":
        actual[0, 0] = 1e-10
    elif change == "tokens":
        current["generated_token_ids"] = [6, 7, 7, 7]
    elif change == "prompt":
        current["prompt_token_ids"] = [1, 3]
    elif change == "dtype":
        actual = actual.double()
    else:
        actual = actual[:, :2]
    assert baseline.compare_baseline_run(previous, logits, current, actual)["baseline_exact"] is False


@pytest.mark.parametrize("count,passed", [(0, False), (1, True), (2, False), (2, True)])
def test_summary_never_mistakes_missing_comparisons_for_pass(count, passed):
    records = [{"type": "MODEL_BASELINE_RESULT", "data": {"baseline_exact": passed}} for _ in range(count)]
    summary = {"stage": "model", "baseline_report": "frozen", "results": [{"records": records}]}
    text = format_compact_summary(summary)
    expected = "PASS" if count == 2 and passed else "NOT_PASSED"
    assert f"BASELINE_EXACT={expected}" in text
    assert "INDEPENDENT_REFERENCE=False" in text


def test_summary_host_breakdown_is_not_added_to_parent_time():
    timing = dict(
        host_load_validate_s=10,
        host_read_s=1,
        host_validate_s=8,
        h2d_s=2,
        prepare_s=3,
        packed_projection_s=4,
        cache_loads=1,
        cache_hits=2,
        evictions=0,
    )
    summary = {"stage": "model", "results": [{"records": [{"type": "MODEL_MOE_TIMING", "data": timing}]}]}
    text = format_compact_summary(summary)
    assert "host_load_validate_s=10.000" in text and "host_validate_s=8.000" in text
    assert "included_in_host_load_validate_s" in text
    del timing["host_read_s"]
    assert "HOST_BREAKDOWN" not in format_compact_summary(summary)


@pytest.mark.parametrize("fail_step,expected_count", [("host", 1), ("moe", 2), (None, 3)])
def test_phase1_driver_stops_at_failure_and_never_runs_deferred_phases(
    tmp_path, monkeypatch, fail_step, expected_count
):
    import sys
    from contextlib import nullcontext

    from tools import validate_vq2a8_tp1_phase1 as driver

    model = tmp_path / "model"
    (model / "experts_vq_ascend_v2").mkdir(parents=True)
    output = tmp_path / "new-output"
    monkeypatch.setattr(
        sys, "argv", ["phase1", "--model", str(model), "--baseline-report", "old", "--output-dir", str(output)]
    )
    monkeypatch.setattr(driver, "freeze_baseline", lambda *args: tmp_path / "frozen-report")
    monkeypatch.setattr(driver, "LiveChildLog", lambda *args: nullcontext())
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        step = command[command.index("--stage") + 1] if "--stage" in command else "host"
        assert kwargs["timeout"] == (1800 if step == "host" else None)
        if step == "model" and fail_step is None:
            folder = output / "model"
            folder.mkdir()
            (folder / "summary.txt").write_text("BASELINE_EXACT=PASS\n")
        return SimpleNamespace(returncode=1 if step == fail_step else 0)

    monkeypatch.setattr(driver.subprocess, "run", run)
    assert driver.main() == (0 if fail_step is None else 1)
    assert len(commands) == expected_count
    report = json.loads((output / "phase1.json").read_text())
    assert report["phase2"] == "skipped_by_request" and report["deferred_phases"] == [3, 4, 5]
    assert report["logits_reference_verified"] is False
    if expected_count == 3:
        assert "--baseline-report" in commands[-1]
