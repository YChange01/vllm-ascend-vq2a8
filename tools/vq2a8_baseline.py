# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve a prior local PASS and compare logits; not an independent oracle.

Only the comparison helpers import torch, inside the isolated model child.
Historical dirty source bytes cannot be reconstructed from recorded hashes.
The snapshot explicitly distinguishes run-time records from capture-time data.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def baseline_records(report: Path) -> tuple[dict, list[dict], dict]:
    """Fail before model construction if the previous gate/evidence is incomplete."""
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    results = summary.get("results", [])
    if (
        summary.get("stage") != "model"
        or summary.get("status") != "passed"
        or summary.get("probes") != ["full_model"]
        or len(results) != 1
        or results[0].get("passed") is not True
        or results[0].get("returncode") != 0
        or results[0].get("timed_out") is not False
    ):
        raise ValueError("Baseline must be one completed, successful full-model acceptance report.")
    records = results[0]["records"]
    gates = [r["data"] for r in records if r["type"] == "VQ2A8_TP1_OFFLINE_EXECUTION_GATE=PASS"]
    runs = [r["data"] for r in records if r["type"] == "MODEL_RESULT"]
    environments = [r["data"] for r in records if r["type"] == "ENVIRONMENT"]
    if (
        len(gates) != 1
        or gates[0].get("repeat_exact") is not True
        or gates[0].get("offline_execution_verified") is not True
        or gates[0].get("runs") != 2
        or len(runs) != 2
        or [r.get("run") for r in runs] != [0, 1]
        or len(environments) != 1
    ):
        raise ValueError("Baseline lacks two repeat-exact runs or environment identity.")
    for run in runs:
        # Do not follow the absolute logits_file stored in a copied report.
        path = report / "model-evidence" / f"run-{run['run']}-logits.safetensors"
        if path.is_symlink() or not path.is_file() or file_sha256(path) != run.get("logits_sha256"):
            raise ValueError(f"Baseline logits missing or SHA256 mismatch: {path}")
        if run.get("finite_logits") is not True or run.get("execution_policy") != "cached":
            raise ValueError("Baseline must have finite logits and the cached execution policy.")
    return summary, runs, environments[0]


def capture_input_identity(model: Path, artifact: Path) -> dict:
    """Small identity files only; a manifest hash is NOT a payload hash scan."""
    files = {
        "model_config": model / "config.json",
        "tokenizer": model / "tokenizer.json",
        "artifact_manifest": artifact / "manifest.json",
    }
    index = model / "model.safetensors.index.json"
    if index.is_file():
        files["root_index"] = index
    return {
        "files_sha256": {key: file_sha256(path) for key, path in files.items()},
        "payload_hashes_verified": False,
        "scope": "capture_time_not_retrospective_run_identity",
    }


def freeze_baseline(
    report: Path, destination: Path, repo: Path, model: Path, artifact: Path, physical_npu: int
) -> Path:
    report = report.resolve(strict=True)
    destination = destination.resolve()
    if destination == report or report in destination.parents:
        raise ValueError("Baseline destination must be outside the original report.")
    summary, _, environment = baseline_records(report)
    if (
        summary.get("device") != "npu:0"
        or summary.get("physical_npu") != physical_npu
        or Path(summary["model"]).resolve() != model.resolve()
        or Path(summary["artifact"]).resolve() != artifact.resolve()
    ):
        raise ValueError("Baseline device/model/artifact location differs from this regression run.")
    # Reports contain logs, JSON and logits only. Refuse links rather than copy
    # unrelated files outside the report directory through a symlink.
    for path in report.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"Baseline report contains a link or special file: {path}")
    destination.mkdir(parents=True, exist_ok=False)
    frozen = destination / "report"
    shutil.copytree(report, frozen)
    baseline_records(frozen)
    snapshot = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "original_report": str(report),
        "recorded_run_git": environment.get("git"),
        "recorded_source_sha256": environment.get("source_sha256", {}),
        "input_identity": capture_input_identity(model, artifact),
        "source_snapshot_scope": "capture_time_only; not proof of historical dirty source bytes",
        "source_matches_recorded": {},
    }
    for name, expected in environment.get("source_sha256", {}).items():
        relative = Path(name) if "/" in name else Path("vllm_ascend/quantization") / name
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe baseline source locator: {name}")
        source = (repo / relative).resolve()
        if repo.resolve() not in source.parents or not source.is_file():
            raise ValueError(f"Invalid or missing baseline source locator: {name}")
        target = destination / "source-at-capture" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        snapshot["source_matches_recorded"][name] = file_sha256(target) == expected
    # Preserve the current tracked dirty diff separately, without cleaning or
    # overwriting the user's worktree. Recorded run status remains above.
    for name, arguments in (
        ("git-head.txt", ["rev-parse", "HEAD"]),
        ("git-status.txt", ["status", "--porcelain"]),
        ("worktree-at-capture.patch", ["diff", "--binary", "HEAD", "--"]),
    ):
        result = subprocess.run(["git", *arguments], cwd=repo, capture_output=True, timeout=30, check=True)
        (destination / name).write_bytes(result.stdout)
    snapshot["snapshot_files_sha256"] = {
        path.relative_to(destination).as_posix(): file_sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    (destination / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(f"BASELINE_FROZEN={destination} HISTORICAL_DIRTY_SOURCE_RECONSTRUCTED=False", flush=True)
    return frozen


def load_baseline(report: Path, environment: dict, model: Path, artifact: Path) -> tuple[list[dict], list]:
    """CPU-only preflight, after importing torch in the supervised child."""
    import torch
    from safetensors.torch import load_file

    _, runs, previous_environment = baseline_records(report)
    for name in ("torch", "torch-npu", "triton", "triton-ascend", "vllm", "safetensors"):
        if previous_environment.get("packages", {}).get(name) != environment.get("packages", {}).get(name):
            raise ValueError(f"Baseline package changed: {name}; same-environment exact regression required.")
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
        expected = previous_environment.get("source_sha256", {}).get(name)
        if not expected or environment.get("source_sha256", {}).get(name) != expected:
            raise ValueError(f"Phase-1 compute source changed or identity absent: {name}")
    snapshot = json.loads((report.parent / "snapshot.json").read_text(encoding="utf-8"))
    if snapshot["input_identity"] != capture_input_identity(model, artifact):
        raise ValueError("Model/artifact identity changed after baseline capture.")
    for relative, expected in snapshot["snapshot_files_sha256"].items():
        path = (report.parent / relative).resolve()
        if report.parent.resolve() not in path.parents or file_sha256(path) != expected:
            raise ValueError(f"Frozen baseline file changed: {relative}")
    logits = []
    for run in runs:
        tensors = load_file(report / "model-evidence" / f"run-{run['run']}-logits.safetensors", device="cpu")
        value = tensors["logits"]
        if (
            set(tensors) != {"logits"}
            or value.ndim != 2
            or value.shape[0] != 4
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("Invalid baseline logits tensor.")
        logits.append(value)
    if not torch.equal(logits[0], logits[1]) or runs[0]["generated_token_ids"] != runs[1]["generated_token_ids"]:
        raise ValueError("Baseline files do not support their repeat-exact claim.")
    return runs, logits


def compare_baseline_run(previous: dict, expected, current: dict, actual) -> dict:
    import torch

    fields = ("prompt_token_ids", "prefill_tokens", "decode_steps", "layers_executed", "execution_policy")
    metadata_equal = all(previous.get(key) == current.get(key) and key in previous for key in fields)
    tokens_equal = previous["generated_token_ids"] == current["generated_token_ids"]
    shape_dtype_equal = expected.shape == actual.shape and expected.dtype == actual.dtype
    logits_equal = shape_dtype_equal and torch.equal(expected, actual)
    return {
        "run": current["run"],
        "baseline_exact": metadata_equal and tokens_equal and logits_equal,
        "metadata_equal": metadata_equal,
        "tokens_equal": tokens_equal,
        "logits_equal": logits_equal,
        "max_abs_error": float((actual - expected).abs().max()) if shape_dtype_equal else None,
        "independent_reference": False,
    }
