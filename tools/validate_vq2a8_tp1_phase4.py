#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated phase-4 development gates, not promotion into the accepted model.

lookup -> CANN native micro -> real packed correctness -> warm benchmarks.
Optional Cube/CV compiler microtests run in separate bounded processes.
No parent NPU import, process reset, model launch, repack or serving changes.
"""

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
import json
import math
import re
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path

try:
    from tools.validate_vq2a8_tp1_acceptance import acceptance_environment
    from tools.vq2a8_live_log import LiveChildLog
except ModuleNotFoundError:
    from validate_vq2a8_tp1_acceptance import acceptance_environment
    from vq2a8_live_log import LiveChildLog


MAX_ERROR_EXCERPT_LINES = 12
MAX_ERROR_LINE_CHARS = 1200


def error_excerpt(path):
    """Keep diagnostic lines even when a compiler dumps thousands of IR lines.

    Read the existing child log, without truncating it or retrying the kernel.
    This is reporting only: evidence and return codes still control the gate.
    """
    diagnostics = deque(maxlen=MAX_ERROR_EXCERPT_LINES)
    pattern = re.compile(
        r"\berror:|\berrorStr:|\b\w*(?:Error|Exception):|IR Dump After .* Failed|encounters error", re.IGNORECASE
    )
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if pattern.search(line):
                    diagnostic = line.strip()[:MAX_ERROR_LINE_CHARS]
                    if diagnostic not in diagnostics:
                        diagnostics.append(diagnostic)
    except OSError:
        pass
    return list(diagnostics)


def probe_list(value):
    items = value.split(",")
    if not items or len(set(items)) != len(items) or any(re.fullmatch(r"[0-9]+:[0-9]+", p) is None for p in items):
        raise argparse.ArgumentTypeError("Use distinct layer:expert pairs, e.g. 0:0,3:127.")
    return items


def evidence_passed(path, stage, device, probe, rows, cases):
    """Fail closed on missing coverage, wrong backend and stale/partial reports."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        records = report["results"]
        if (
            report["status"] != "passed"
            or report["stage"] != stage
            or report["device"] != device
            or report["native_fp8_expert_dot"] is not False
            or report["model_integration_verified"] is not False
            or report["requested"] != {"probe": probe, "rows": rows, "cases": cases}
            or report["npu_verified"] is not device.startswith("npu")
        ):
            return False
        if not records or any(r.get("passed") is not True or r.get("repeat_exact") is not True for r in records):
            return False
        if stage in ("native", "cube_direct", "cv_bridge"):
            return (
                len(records) == 1
                and records[0].get("synthetic_only") is True
                and records[0]["comparison"]["allclose"] is True
                and records[0]["micro_backend"]
                == ("cann" if stage == "native" and device.startswith("npu") else "triton")
            )
        if any(
            r["oracle"]["allclose"] is not True
            or r["baseline"]["allclose"] is not True
            or r["row_chunk_exact"] is not True
            or r["dense_expert_weight_on_device"] is not False
            for r in records
        ):
            return False
        if stage == "lookup":
            expected = {(m, t) for m in (1, 3, 32) for t in (1, 3, 16, 32)}
            actual = {(r["rows"], r["column_tiles"]) for r in records}
        else:
            if any(r["chain_baseline"]["allclose"] is not True for r in records):
                return False
            expected = {(probe, kind, m, case) for kind in ("gate_up", "down") for m in rows for case in cases}
            actual = {(r["probe"], r["projection"], r["rows"], r["case"]) for r in records}
        if actual != expected or len(records) != len(expected):
            return False
        if stage == "benchmark":
            for r in records:
                for name in ("accepted", "candidate"):
                    if len(r["timing"][name]) != 2:
                        return False
                    for sample in r["timing"][name]:
                        if sample["warmups"] < 3 or sample["repeats"] < 10:
                            return False
                        for clock in ("event_ms", "wall_ms"):
                            values = [sample[clock][key] for key in ("min", "median", "p95")]
                            if any(not math.isfinite(v) or v <= 0 for v in values) or values != sorted(values):
                                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def short_report(report):
    lines = [
        f"PHASE4=INCOMPLETE KERNEL_GATES={report['status'].upper()}",
        f"SCOPE={report['scope']} DEVICE={report['device']} PHYSICAL_NPU={report['physical_npu']}",
        f"COMPLETED={len(report['results'])}/{len(report['planned_steps'])}",
    ]
    for r in report["results"]:
        lines.append(
            f"STEP={r['name']} {'PASS' if r['passed'] else 'FAIL'} exit={r['returncode']} timeout={r['timeout']}"
        )
        if not r["passed"]:
            if r.get("error"):
                lines.append(f"  ERROR={r['error']}")
            lines.extend(f"  {line}" for line in r.get("error_excerpt", []))
            if r.get("log"):
                lines.append(f"  LOG={r['log']}")
        grouped = {}
        for sample in r.get("speedups", []):
            grouped.setdefault((sample["probe"], sample["projection"]), []).append(sample)
        for (probe, projection), samples in grouped.items():
            # Keep the smallest/largest M on screen; all sizes and timing
            # distributions remain in JSON. Seven probes must stay copyable.
            samples.sort(key=lambda sample: sample["rows"])
            selected = samples if len(samples) < 2 else [samples[0], samples[-1]]
            lines.append(
                f"SPEEDUP={probe}:{projection} "
                + " ".join(
                    f"m={s['rows']}:event={s['event_ms_speedup']:.3f}x,wall={s['wall_ms_speedup']:.3f}x"
                    for s in selected
                )
            )
    lines += [
        "PERFORMANCE_VERIFIED=False (standalone timings require hardware/profile review)",
        "DEFAULT_MODEL_BACKEND=UNCHANGED MODEL_INTEGRATION_VERIFIED=False",
        "NATIVE_FP8_EXPERT_DOT=False QUALITY_VERIFIED=False SERVING_VERIFIED=False",
        "PHASE2=SKIPPED PHASE3_MODEL=NOT_EVALUATED_BY_THIS_RUN PHASE5=DEFERRED",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--physical-npu", type=int, default=4)
    parser.add_argument("--device", choices=["npu:0", "cuda:0"], default="npu:0")
    parser.add_argument("--probes", type=probe_list, default=probe_list("0:0,1:0,2:0,3:0,3:127,3:255,42:255"))
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 3, 10, 32])
    parser.add_argument(
        "--micro-only", action="store_true", help="Only bounded synthetic tests; no real weights or benchmark."
    )
    parser.add_argument(
        "--include-cv",
        action="store_true",
        help="Opt in to potentially aborting experimental Triton Cube/CV microtests.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if (
        args.physical_npu < 0
        or args.timeout <= 0
        or len(set(args.rows)) != len(args.rows)
        or any(not 1 <= m <= 32 for m in args.rows)
    ):
        parser.error("Require nonnegative physical NPU, positive timeout and unique rows in [1,32].")
    if not args.micro_only and args.model is None:
        parser.error("--model is required unless --micro-only is selected.")
    repo = Path(__file__).resolve().parents[1]
    if args.model:
        args.model = args.model.resolve(strict=True)
    if not args.micro_only:
        args.artifact = (args.artifact or args.model / "experts_vq_ascend_v2").resolve(strict=True)
    if args.output_dir:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = Path(tempfile.mkdtemp(prefix="vq2a8-phase4-"))
    cases = ["deterministic", "zero", "impulse", "small"]
    steps = [(stage, stage, "0:0", cases) for stage in ("lookup", "native")]
    if args.include_cv:
        steps += [(stage, stage, "0:0", cases) for stage in ("cube_direct", "cv_bridge")]
    if not args.micro_only:
        steps += [(f"packed-{p.replace(':', '-')}", "packed", p, cases) for p in args.probes]
        steps += [(f"benchmark-{p.replace(':', '-')}", "benchmark", p, ["deterministic"]) for p in args.probes]
    report = {
        "status": "running",
        "scope": "MICRO_ONLY" if args.micro_only else "STANDALONE_KERNELS",
        "device": args.device,
        "physical_npu": args.physical_npu,
        "planned_steps": [s[0] for s in steps],
        "results": [],
        "phase4_complete": False,
        "model_integration_verified": False,
        "native_fp8_expert_dot": False,
        "serving_verified": False,
    }

    def save():
        (output / "phase4.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (output / "summary.txt").write_text(short_report(report), encoding="utf-8")

    save()
    print(f"PHASE4_REPORT_DIR={output} DEFAULT_MODEL_BACKEND=UNCHANGED", flush=True)
    for name, stage, probe, requested_cases in steps:
        log, evidence = output / f"{name}.log", output / f"{name}.json"
        command = [
            sys.executable,
            str(repo / "tools/validate_vq2a8_phase4_kernel.py"),
            "--stage",
            stage,
            "--device",
            args.device,
            "--probe",
            probe,
            "--output",
            str(evidence),
            "--rows",
            *map(str, args.rows),
            "--cases",
            *requested_cases,
        ]
        if args.model:
            command += ["--model", str(args.model)]
        if args.artifact:
            command += ["--artifact", str(args.artifact)]
        env = acceptance_environment(repo, args.physical_npu, args.device)
        # Existing acceptance forces launch blocking. Warm event benchmarks
        # explicitly disable it; correctness remains synchronized per result.
        if args.device.startswith("npu") and stage == "benchmark":
            env["ASCEND_LAUNCH_BLOCKING"] = "0"
        print(f"PHASE4_STEP_START={name} LOG={log}", flush=True)
        code, timed_out, error = None, False, None
        try:
            with log.open("w", encoding="utf-8") as stream, LiveChildLog(log, name):
                child = subprocess.run(
                    command,
                    cwd=repo,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
            code = child.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
        except OSError as exception:
            error = str(exception)
        passed = (
            code == 0
            and not timed_out
            and evidence_passed(evidence, stage, args.device, probe, args.rows, requested_cases)
        )
        step = {
            "name": name,
            "passed": passed,
            "returncode": code,
            "timeout": timed_out,
            "error": error,
            "log": str(log),
            "evidence": str(evidence),
            "error_excerpt": error_excerpt(log) if not passed else [],
        }
        if passed and stage == "benchmark":
            results = json.loads(evidence.read_text())["results"]
            step["speedups"] = [
                {key: r[key] for key in ("probe", "projection", "rows", "event_ms_speedup", "wall_ms_speedup")}
                for r in results
            ]
        report["results"].append(step)
        report["status"] = "failed" if not passed else "running"
        save()
        if not passed:
            print(short_report(report), end="", flush=True)
            print(f"PHASE4_KERNEL_GATES=FAIL step={name} REPORT={output} (remaining steps skipped)", flush=True)
            return 1
        print(f"PHASE4_STEP_PASS={name}", flush=True)
    report["status"] = "passed"
    save()
    print(short_report(report), end="", flush=True)
    print(f"PHASE4_KERNEL_GATES=PASS PHASE4=INCOMPLETE REPORT={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
