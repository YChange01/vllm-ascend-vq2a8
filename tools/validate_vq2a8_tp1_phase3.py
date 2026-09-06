#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 3 only: CANN FP8 smoke -> real root projections -> offline FP8 model.

Streaming logs and bounded child processes. No repack, TP4 or HTTP serving.
The accepted BF16 historical baseline is preserved, not used as an FP8 oracle.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from tools.validate_vq2a8_tp1_acceptance import acceptance_environment
    from tools.vq2a8_live_log import LiveChildLog
except ModuleNotFoundError:
    from validate_vq2a8_tp1_acceptance import acceptance_environment
    from vq2a8_live_log import LiveChildLog


def step_evidence_passed(output: Path, step: str) -> bool:
    """Exit zero alone cannot certify the requested backend or completed gate."""
    path = output / ("model/summary.json" if step == "model" else f"{step}.json")
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence.get("status") != "passed" or not evidence.get("results"):
            return False
        if step != "model":
            return (
                evidence.get("stage") == step
                and evidence.get("device") == "npu:0"
                and all(r.get("passed") is True for r in evidence["results"])
            )
        results = evidence["results"]
        if len(results) != 1 or results[0].get("passed") is not True:
            return False
        gates = [
            r["data"] for r in results[0].get("records", []) if r.get("type") == "VQ2A8_TP1_OFFLINE_EXECUTION_GATE=PASS"
        ]
        return (
            len(gates) == 1
            and gates[0].get("root_linear_mode") == "online_fp8_sm90"
            and all(
                gates[0].get(key) is True
                for key in (
                    "root_fp8_execution_verified",
                    "native_fp8_root_matmul",
                    "offline_execution_verified",
                    "repeat_exact",
                )
            )
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--physical-npu", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--operators-only",
        action="store_true",
        help="Run smoke and root projections only; never start or certify the full model/phase 3.",
    )
    args = parser.parse_args()
    if args.physical_npu < 0:
        parser.error("Physical NPU must be non-negative.")
    repo = Path(__file__).resolve().parents[1]
    model = args.model.resolve(strict=True)
    artifact = (args.artifact or model / "experts_vq_ascend_v2").resolve(strict=not args.operators_only)
    if args.output_dir:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = Path(tempfile.mkdtemp(prefix="vq2a8-phase3-"))
    print(f"PHASE3_REPORT_DIR={output}", flush=True)
    print("PHASE2=SKIPPED PHASE4=DEFERRED PHASE5=DEFERRED", flush=True)
    # Execute native capability first; later steps cannot hide a failed or
    # unsupported primitive behind a reference backend or a full-model PASS.
    steps = [
        (
            stage,
            [
                sys.executable,
                str(repo / "tools/validate_vq2a8_root_fp8.py"),
                "--model",
                str(model),
                "--stage",
                stage,
                "--output",
                str(output / f"{stage}.json"),
            ],
            1800,
        )
        for stage in ("smoke", "roots")
    ]
    model_step = (
        "model",
        [
            sys.executable,
            str(repo / "tools/validate_vq2a8_tp1_acceptance.py"),
            "--stage",
            "model",
            "--model",
            str(model),
            "--artifact",
            str(artifact),
            "--physical-npu",
            str(args.physical_npu),
            "--root-linear-mode",
            "online_fp8_sm90",
            "--execution-policy",
            "cached",
            "--output-dir",
            str(output / "model"),
        ],
        None,
    )
    if not args.operators_only:
        steps.append(model_step)
    report = {
        "phase": 3,
        "status": "running",
        "results": [],
        "phase2": "skipped_by_request",
        "deferred_phases": [4, 5],
        "logits_reference_verified": False,
        "quality_verified": False,
        "serving_verified": False,
        "native_fp8_expert_dot": False,
        "scope": "operators_only" if args.operators_only else "operators_and_model",
        "planned_steps": [step[0] for step in steps],
        "root_fp8_operators_verified": False,
        "root_fp8_execution_verified": False,
    }
    for name, command, timeout in steps:
        log = output / f"{name}.log"
        print(f"PHASE3_STEP_START={name} LOG={log}", flush=True)
        code = None
        try:
            with log.open("w", encoding="utf-8") as stream, LiveChildLog(log, f"phase3-{name}"):
                result = subprocess.run(
                    command,
                    cwd=repo,
                    env=acceptance_environment(repo, args.physical_npu, "npu:0"),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                )
            code = result.returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            print(f"PHASE3_STEP_ERROR={name} {error}", flush=True)
        passed = code == 0 and step_evidence_passed(output, name)
        report["results"].append({"step": name, "returncode": code, "passed": passed, "log": str(log)})
        report["status"] = "failed" if not passed else ("passed" if name == "model" else "running")
        if passed and name == "roots":
            report["root_fp8_operators_verified"] = True
            if args.operators_only:
                report["status"] = "incomplete"
        report["root_fp8_execution_verified"] = passed and name == "model"
        (output / "phase3.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        short = [
            f"PHASE3={report['status'].upper()} completed={len(report['results'])}/3",
            f"SCOPE={report['scope'].upper()} requested_completed={len(report['results'])}/{len(steps)}",
            *[f"STEP={r['step']} {'PASS' if r['passed'] else 'FAIL'}" for r in report["results"]],
            f"ROOT_FP8_EXECUTION_VERIFIED={report['root_fp8_execution_verified']}",
            f"ROOT_FP8_OPERATORS_VERIFIED={report['root_fp8_operators_verified']}",
            f"MODEL={'NOT_RUN' if args.operators_only else 'REQUIRED'}",
            "PHASE2=SKIPPED NATIVE_FP8_EXPERT_DOT=False QUALITY_VERIFIED=False SERVING_VERIFIED=False",
        ]
        (output / "summary.txt").write_text("\n".join(short) + "\n", encoding="utf-8")
        if not passed:
            print(f"PHASE3=FAIL step={name} REPORT={output} (remaining steps skipped)", flush=True)
            return 1
        print(f"PHASE3_STEP_PASS={name}", flush=True)
    if args.operators_only:
        print((output / "summary.txt").read_text(encoding="utf-8"), end="", flush=True)
        print(f"PHASE3_OPERATORS=PASS PHASE3=INCOMPLETE MODEL=NOT_RUN REPORT={output}", flush=True)
        return 0
    print((output / "model/summary.txt").read_text(encoding="utf-8"), end="", flush=True)
    print(f"PHASE3=PASS REPORT={output} (operator + offline execution only)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
