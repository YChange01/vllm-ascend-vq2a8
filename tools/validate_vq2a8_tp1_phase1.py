#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run phase 1 only: preserve baseline, host benchmark, MoE, exact model regression.

Phase 2 (independent reference) is skipped by request, not marked verified.
Phases 3/4/5 (root FP8, kernel performance, serving) are deferred.
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
    from tools.vq2a8_baseline import freeze_baseline
    from tools.vq2a8_live_log import LiveChildLog
except ModuleNotFoundError:
    from validate_vq2a8_tp1_acceptance import acceptance_environment
    from vq2a8_baseline import freeze_baseline
    from vq2a8_live_log import LiveChildLog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--physical-npu", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, help="New directory; never overwrites a previous run.")
    args = parser.parse_args()
    if args.physical_npu < 0:
        parser.error("Physical NPU must be non-negative.")
    repo = Path(__file__).resolve().parents[1]
    model = args.model.resolve(strict=True)
    artifact = (args.artifact or model / "experts_vq_ascend_v2").resolve(strict=True)
    if args.output_dir:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = Path(tempfile.mkdtemp(prefix="vq2a8-phase1-"))
    print(f"PHASE1_REPORT_DIR={output}", flush=True)
    frozen = freeze_baseline(args.baseline_report, output / "baseline", repo, model, artifact, args.physical_npu)
    shared = ["--model", str(model), "--artifact", str(artifact)]
    accept = [
        sys.executable,
        str(repo / "tools/validate_vq2a8_tp1_acceptance.py"),
        *shared,
        "--physical-npu",
        str(args.physical_npu),
        "--execution-policy",
        "cached",
    ]
    steps = [
        (
            "host",
            [
                sys.executable,
                str(repo / "tools/benchmark_vq2a8_host_load.py"),
                *shared,
                "--output",
                str(output / "host.json"),
            ],
            1800,
        ),
        (
            "moe",
            [
                *accept,
                "--stage",
                "moe",
                "--layers",
                "0,3",
                "--token-counts",
                "1",
                "3",
                "10",
                "--cases",
                "deterministic",
                "zero",
                "--warmups",
                "0",
                "--repeats",
                "3",
                "--output-dir",
                str(output / "moe"),
            ],
            None,
        ),
        (
            "model",
            [
                *accept,
                "--stage",
                "model",
                "--baseline-report",
                str(frozen),
                "--output-dir",
                str(output / "model"),
                "--timeout",
                "3600",
            ],
            None,
        ),
    ]
    report = {
        "phase": 1,
        "results": [],
        "phase2": "skipped_by_request",
        "deferred_phases": [3, 4, 5],
        "logits_reference_verified": False,
        "quality_verified": False,
        "serving_verified": False,
    }
    for name, command, timeout in steps:
        print(f"PHASE1_STEP_START={name}", flush=True)
        log = output / f"{name}.log"
        # Acceptance owns its device child's timeout. Do not add a competing
        # outer timeout that could kill that supervisor and orphan its worker.
        try:
            with log.open("w", encoding="utf-8") as stream, LiveChildLog(log, f"phase1-{name}"):
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
        except (subprocess.TimeoutExpired, OSError) as error:
            print(f"PHASE1_STEP_ERROR={name} {error}", flush=True)
            code = None
        report["results"].append({"step": name, "returncode": code, "log": str(log)})
        report["status"] = "failed" if code != 0 else ("passed" if name == "model" else "running")
        (output / "phase1.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if code != 0:
            print(f"PHASE1=FAIL step={name} REPORT={output} (remaining steps skipped)", flush=True)
            return 1
        print(f"PHASE1_STEP_PASS={name}", flush=True)
    print((output / "model/summary.txt").read_text(encoding="utf-8"), end="", flush=True)
    print(f"PHASE1_REGRESSION=PASS REPORT={output} PERFORMANCE_REQUIRES_TIMING_REVIEW=True", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
