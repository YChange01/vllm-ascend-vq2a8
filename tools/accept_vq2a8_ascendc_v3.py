#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in v3: build -> short operator gate -> isolated v1 -> exact v3 -> timing.

Default 10:4 is a quick case, not a long-output performance claim. Full v3
residency may exceed the default budget/reserve; never reduce safety margins
automatically. No install, repack, service, fallback, or full-model graph claim.
"""

from __future__ import annotations

# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import contextlib
import datetime
import json
import math
import platform
import signal
import subprocess
import time
from pathlib import Path

from tools.benchmark_vq2a8_ascendc_v3 import parse_cases
from tools.build_vq2a8_ascendc_v3 import LIBRARY_NAME, REPO
from tools.validate_vq2a8_ascendc_v3 import library_identity, validate_receipt
from tools.validate_vq2a8_tp1_acceptance import acceptance_environment
from tools.vq2a8_live_log import LiveChildLog


class ProgressLog(LiveChildLog):
    def _record_stage(self, text):
        lines = (self._pending_line + text).split("\n")
        for line in lines[:-1]:
            if line.startswith(("V3_", "PERF_V3_", "MODEL ", "MODEL_", "VQ2A8_V3_")):
                self._last_stage = line.strip()[:240]
        self._pending_line = lines[-1][-1024:]


def supervise(command, log, environment, timeout):
    """Process-group timeout; log progress is forwarded even during model load."""
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream, ProgressLog(log, log.stem):
        child = subprocess.Popen(
            command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        timed_out = False
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise
    return dict(
        exit=child.returncode, timeout=timed_out, elapsed_s=time.monotonic() - started, log=str(log), command=command
    )


def commands(args, output):
    base = [sys.executable, "-u"]
    library = (args.library or args.build_dir / LIBRARY_NAME).resolve()
    model = args.model.resolve()
    steps = [("environment", [*base, str(REPO / "tools/validate_vq2a8_v026_environment.py")])]
    if args.library is None:
        steps.append(
            (
                "build",
                [
                    *base,
                    str(REPO / "tools/build_vq2a8_ascendc_v3.py"),
                    "--soc",
                    args.soc,
                    "--cann",
                    str(args.cann),
                    "--build-dir",
                    str(args.build_dir.resolve()),
                    "--jobs",
                    str(args.jobs),
                    "--timeout",
                    str(args.timeout),
                ],
            )
        )
    steps.append(
        (
            "preflight",
            [
                *base,
                str(REPO / "tools/validate_vq2a8_ascendc_v3.py"),
                "--model",
                str(model),
                "--library",
                str(library),
                "--output",
                str(output / "preflight.json"),
            ],
        )
    )
    if args.preflight_only:
        return steps
    common = [
        *base,
        str(REPO / "tools/benchmark_vq2a8_ascendc_v3.py"),
        "--model",
        str(model),
        "--cases",
        args.cases,
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--cache-budget-gib",
        str(args.cache_budget_gib),
        "--cache-reserve-gib",
        str(args.cache_reserve_gib),
        "--memory-fraction",
        str(args.memory_fraction),
    ]
    reference = args.reference_report.resolve() if args.reference_report else output / "v1-reference/summary.json"
    if args.reference_report is None:
        steps.append(
            (
                "v1-reference",
                [
                    *common,
                    "--library",
                    str(args.baseline_library.resolve()),
                    "--reference-only",
                    "--output-dir",
                    str(output / "v1-reference"),
                ],
            )
        )
    candidate = [
        *common,
        "--library",
        str(library),
        "--preflight",
        str(output / "preflight.json"),
        "--reference-report",
        str(reference),
    ]
    if args.benchmark:
        performance = [*candidate, "--output-dir", str(output / "performance")]
        if args.target_tpot_ms is not None:
            performance += ["--target-tpot-ms", str(args.target_tpot_ms)]
        if args.profile:
            performance += ["--profile"]
        steps.append(("performance", performance))
    else:
        steps.append(("model-exact", [*candidate, "--correctness-only", "--output-dir", str(output / "model-exact")]))
    return steps


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--soc", help="Exact Ascend950 device name, required to build")
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann-9.1.0"))
    parser.add_argument("--library", type=Path)
    parser.add_argument("--build-dir", type=Path, default=REPO / "build/vq2a8-ascendc-v3")
    parser.add_argument("--baseline-library", type=Path, default=REPO / "build/vq2a8-ascendc-v026/libvq2a8_ascendc.so")
    parser.add_argument("--reference-report", type=Path, help="Reuse hash-bound v1 reference produced by this workflow")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--physical-npu", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3600, help="Maximum seconds for EACH child stage")
    parser.add_argument("--cache-budget-gib", type=float, default=0.0, help="0 computes budget from available memory")
    parser.add_argument(
        "--cache-reserve-gib",
        type=float,
        default=16.0,
        help="Explicit headroom for KV/allocator; never automatically reduced",
    )
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Extra untimed CPU/NPU trace after benchmark cases")
    parser.add_argument(
        "--target-tpot-ms", type=float, help="Optional measured observation target; not a correctness threshold"
    )
    parser.add_argument("--cases", default="10:4", help="Quick case; also supports 10:64,32:64")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        parse_cases(args.cases)
        if not args.library and (not args.soc or not args.soc.startswith("Ascend950")):
            raise ValueError("Provide an exact Ascend950 --soc or --library")
        if min(args.jobs, args.timeout) < 1 or args.physical_npu < 0 or args.warmups < 2 or args.repeats < 5:
            raise ValueError("Invalid jobs/timeout/NPU/warmups/repeats")
        if args.preflight_only and args.benchmark:
            raise ValueError("Benchmark cannot skip full-model exact gate")
        if args.profile and not args.benchmark:
            raise ValueError("--profile requires --benchmark")
        if (
            not math.isfinite(args.cache_budget_gib)
            or args.cache_budget_gib < 0
            or not math.isfinite(args.cache_reserve_gib)
            or args.cache_reserve_gib < 1
            or not math.isfinite(args.memory_fraction)
            or not 0 < args.memory_fraction <= 1
        ):
            raise ValueError("Invalid budget/reserve/memory fraction")
        if args.target_tpot_ms is not None and (
            not args.benchmark or not math.isfinite(args.target_tpot_ms) or args.target_tpot_ms <= 0
        ):
            raise ValueError("Finite positive --target-tpot-ms requires --benchmark")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    output = (
        args.output_dir
        or REPO
        / "reports"
        / ("vq2a8-ascendc-v3-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    ).resolve()
    steps = commands(args, output)
    if args.plan_only:
        print(
            json.dumps(
                dict(
                    scope="plan_only_no_device_execution",
                    steps=steps,
                    default_backend="unchanged",
                    full_model_graph_verified=False,
                    performance_target_met=None,
                ),
                indent=2,
            )
        )
        return 0
    if platform.system() != "Linux":
        raise RuntimeError("Run on Linux Ascend950; CPU/Windows supports --plan-only")
    output.mkdir(parents=True, exist_ok=False)
    environment = acceptance_environment(REPO, args.physical_npu, "npu:0")
    report = dict(
        status="RUNNING",
        stages=[],
        default_backend="unchanged",
        full_model_graph_verified=False,
        device_execution_verified=False,
        model_integration_verified=False,
        baseline_exact=False,
        performance_measurement_verified=False,
        performance_target_met=None,
    )

    def save():
        (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    try:
        for name, command in steps:
            print(f"VQ2A8_V3_STAGE={name} TIMEOUT_S={args.timeout} LOG={output / (name + '.log')}", flush=True)
            stage_env = environment.copy()
            if name == "performance":
                stage_env["ASCEND_LAUNCH_BLOCKING"] = "0"
            result = supervise(command, output / f"{name}.log", stage_env, args.timeout)
            report["stages"].append(dict(name=name, **result))
            save()
            if result["exit"] != 0 or result["timeout"]:
                raise RuntimeError(f"{name} failed; inspect {result['log']}")
            if name == "preflight":
                library = (args.library or args.build_dir / LIBRARY_NAME).resolve()
                receipt = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
                validate_receipt(receipt, library_identity(library), args.model, str(args.physical_npu))
                report["device_execution_verified"] = True
            elif name in ("v1-reference", "model-exact", "performance"):
                from tools.benchmark_vq2a8_ascendc_v3 import parse_args as child_args
                from tools.benchmark_vq2a8_ascendc_v3 import verify_report

                parsed = child_args(command[3:])
                child_report = json.loads((parsed.output_dir / "summary.json").read_text(encoding="utf-8"))
                # Receipt check is host-only, but follows the isolated child's physical-device identity.
                previous = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
                try:
                    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.physical_npu)
                    verify_report(child_report, parsed)
                finally:
                    if previous is None:
                        os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
                    else:
                        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = previous
                if name == "model-exact":
                    report.update(model_integration_verified=True, baseline_exact=True)
                elif name == "performance":
                    report.update(
                        model_integration_verified=True,
                        baseline_exact=True,
                        performance_measurement_verified=True,
                        performance_target_met=child_report["performance_target_met"],
                        summaries=child_report["summaries"],
                        profile=child_report.get("profile", {"status": "NOT_REQUESTED"}),
                    )
        report["status"] = "PASS"
        return 0
    except Exception as exc:
        report.update(status="FAIL", error=str(exc))
        return 1
    finally:
        save()
        print(
            f"VQ2A8_V3_ACCEPTANCE={report['status']} REPORT={output / 'summary.json'} "
            f"BASELINE_EXACT={report['baseline_exact']} PERFORMANCE_TARGET_MET={report['performance_target_met']} "
            "FULL_MODEL_GRAPH_VERIFIED=False",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
