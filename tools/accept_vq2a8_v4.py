#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent V4 TP1 acceptance. Reuse the V1 library and direct TP1 artifact.

No automatic repack/build/reinstall, V3 worker, version pin or global pip-check gate.
Default: one full V4 model load. --compare-v1 adds an earlier, separate V1 load.
Opt-in --device-route-decode needs a rebuilt library and compares paths in one engine.
"""

from __future__ import annotations

# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import datetime
import json
import math
import platform
import subprocess
import time
from pathlib import Path

from tools.accept_vq2a8_optimizations import failure_causes
from tools.benchmark_vq2a8_v4 import require_idle_device
from tools.diagnose_vq2a8_tp1_startup import terminate_child
from tools.profile_vq2a8_ascendc import write_json
from tools.vq2a8_live_log import LiveChildLog
from tools.vq2a8_perf_report import validate_cases

REPO = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/home/g00872988/vq2a8"))
    parser.add_argument("--library", type=Path, default=REPO / "build/vq2a8-ascendc-v023-v1/libvq2a8_ascendc.so")
    parser.add_argument("--physical-npu", type=int, default=1)
    parser.add_argument("--cache-reserve-gib", type=float, default=8.0)
    parser.add_argument("--cache-budget-gib", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--kv-cache-mib", type=int, default=1024)
    parser.add_argument("--memory-fraction", type=float, default=None, help="Independent expert budget fraction")
    parser.add_argument("--engine-memory-fraction", type=float, default=0.9)
    parser.add_argument("--cases", default="10:4")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=14400, help="Per-child timeout, including preload")
    parser.add_argument("--compare-v1", action="store_true", help="First run V1 batched in a separate process")
    parser.add_argument(
        "--device-route-decode",
        action="store_true",
        help="Compare batched/device-route decode in one V4 resident engine; needs the rebuilt native library",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        cases = validate_cases([tuple(map(int, case.split(":"))) for case in args.cases.split(",")])
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.physical_npu < 0 or args.warmups < 2 or args.repeats < 5 or args.timeout < 1:
        parser.error("Require one nonnegative NPU, >=2 warmups, >=5 repeats and a positive timeout.")
    if not 1 <= args.max_model_len <= 128 or any(sum(case) > args.max_model_len for case in cases):
        parser.error("Each case must fit --max-model-len, which must be in [1,128].")
    if args.kv_cache_mib <= 0:
        parser.error("--kv-cache-mib must be a positive integer.")
    if any(
        not math.isfinite(value) or not 0 < value <= 1
        for value in (args.memory_fraction, args.engine_memory_fraction)
        if value is not None
    ):
        parser.error("Memory fractions must be finite and in (0,1].")
    if (
        not math.isfinite(args.cache_reserve_gib)
        or args.cache_reserve_gib < max(1.0, args.kv_cache_mib / 1024)
        or not math.isfinite(args.cache_budget_gib)
        or args.cache_budget_gib < 0
    ):
        parser.error("Require finite reserve >=1 GiB covering KV, and budget >=0 GiB.")
    args.artifact = args.model / "experts_vq_ascend_v2"
    return args


def commands(args, output):
    base = [sys.executable, "-u"]
    steps = [("environment", [*base, str(REPO / "tools/validate_vq2a8_v023_environment.py")])]
    steps.append(
        (
            "preflight",
            [
                *base,
                str(REPO / "tools/accept_vq2a8_release.py"),
                "--worker",
                "preflight",
                "--model",
                str(args.model.resolve()),
                "--library",
                str(args.library.resolve()),
                "--physical-npu",
                str(args.physical_npu),
                "--output-dir",
                str(output / "preflight"),
            ],
        )
    )
    if args.device_route_decode:
        steps.append(
            (
                "device_route_preflight",
                [
                    *base,
                    str(REPO / "tools/validate_vq2a8_v4_device_route.py"),
                    "--library",
                    str(args.library.resolve()),
                    "--physical-npu",
                    str(args.physical_npu),
                    "--report-dir",
                    str(output / "device_route_preflight"),
                ],
            )
        )
    worker = [
        *base,
        str(REPO / "tools/benchmark_vq2a8_v4.py"),
        "--model",
        str(args.model.resolve()),
        "--library",
        str(args.library.resolve()),
        "--preflight",
        str(output / "preflight/preflight.json"),
        "--physical-npu",
        str(args.physical_npu),
        "--cache-reserve-gib",
        str(args.cache_reserve_gib),
        "--cache-budget-gib",
        str(args.cache_budget_gib),
        "--max-model-len",
        str(args.max_model_len),
        "--kv-cache-mib",
        str(args.kv_cache_mib),
        "--engine-memory-fraction",
        str(args.engine_memory_fraction),
        *(["--memory-fraction", str(args.memory_fraction)] if args.memory_fraction is not None else []),
        "--cases",
        args.cases,
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
    ]
    if args.compare_v1:
        steps.append(("v1_reference", [*worker, "--reference-only", "--output-dir", str(output / "v1_reference")]))
    steps.append(
        (
            "v4",
            [
                *worker,
                "--output-dir",
                str(output / "v4"),
                *(["--device-route-decode"] if args.device_route_decode else []),
                *(["--reference-report", str(output / "v1_reference/summary.json")] if args.compare_v1 else []),
            ],
        )
    )
    return steps


class V4SummaryChildLog(LiveChildLog):
    """Keep complete disk logs; surface V4 progress without raw framework spam."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._screen_pending = ""
        self._screen_updated = time.monotonic()

    def _emit(self, text):
        lines = (self._screen_pending + text).split("\n")
        self._screen_pending = lines[-1][-65536:]
        for line in lines[:-1]:
            if line.startswith(("V4_", "MODEL_V4_", "MODEL_CACHE_BUDGET", "PROBE_WAIT=", "ERROR=", "Traceback")) or (
                line.startswith("MODEL ") and ("stage=v4_" in line or "stage=root_weight_load" in line)
            ):
                super()._emit(line[:2000] + "\n")
                self._screen_updated = time.monotonic()
        if time.monotonic() - self._screen_updated >= 30:
            super()._emit(f"PROBE_WAIT={self.probe} last_stage={self._last_stage} LOG={self.path}\n")
            self._screen_updated = time.monotonic()


def supervise(command, log, environment, timeout):
    """Fresh worker and bounded cleanup of only the session created here."""
    started = time.monotonic()
    timed_out = False
    reaped = True
    with log.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(
            command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        with V4SummaryChildLog(log, log.stem):
            try:
                child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                reaped = terminate_child(child)
            except BaseException:
                terminate_child(child)
                raise
    return {
        "exit": child.returncode,
        "timeout": timed_out,
        "reaped": reaped,
        "elapsed_s": time.monotonic() - started,
        "command": command,
        "log": str(log),
    }


def run(args):
    output = (
        args.output_dir
        or REPO / "reports" / ("vq2a8-v4-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    ).resolve()
    steps = commands(args, output)
    if args.plan_only:
        print(
            json.dumps(
                dict(scope="plan_only_no_device_execution", policy="ascendc_v4", steps=steps, output=str(output)),
                indent=2,
            )
        )
        return 0
    if platform.system() != "Linux":
        raise ValueError("Run on the Linux NPU server; --plan-only is CPU-only.")
    from tools.validate_vq2a8_tp1_acceptance import acceptance_environment

    output.mkdir(parents=True, exist_ok=False)
    env = acceptance_environment(REPO, args.physical_npu, "npu:0")
    env.update(ASCEND_LAUNCH_BLOCKING="0", VLLM_ENABLE_V1_MULTIPROCESSING="0")
    report = dict(
        status="RUNNING",
        execution_policy="ascendc_v4",
        stages=[],
        performance_measurement_verified=False,
        v1_comparison="NOT_RUN",
        default_v1_unchanged=True,
        device_snapshots=[],
        device_route_comparison="NOT_RUN",
    )
    try:
        for name, command in steps:
            print(f"V4_STAGE={name} REPORT={output}", flush=True)
            if name != "environment":
                report["device_snapshots"].append(require_idle_device(args.physical_npu, output / f"{name}-npu.log"))
            result = supervise(command, output / f"{name}.log", env, args.timeout)
            report["stages"].append(dict(name=name, **result))
            write_json(output / "run.json", report)
            if result["exit"] != 0 or result["timeout"]:
                causes = failure_causes(result["log"], timed_out=result["timeout"])
                report["stages"][-1]["failure_causes"] = causes
                raise RuntimeError(f"{name} failed: {' | '.join(causes)}; full log: {result['log']}")
        result = json.loads((output / "v4/summary.json").read_text(encoding="utf-8"))
        if (
            result.get("status") != "PASS"
            or result.get("execution_policy") != "ascendc_v4"
            or result.get("performance_measurement_verified") is not True
            or (args.compare_v1 and result.get("v1_comparison") != "PASS")
            or (
                args.device_route_decode
                and (
                    result.get("optimization") != "device_route_decode"
                    or result.get("device_route_comparison") != "PASS"
                )
            )
        ):
            raise ValueError("V4 child did not provide complete requested acceptance evidence.")
        report.update(
            status="PASS",
            performance_measurement_verified=True,
            v1_comparison=result["v1_comparison"],
            result=str(output / "v4/summary.json"),
            device_route_comparison=result.get("device_route_comparison", "NOT_RUN"),
        )
        for name, case in result["cases"].items():
            metrics = case["metrics"]
            measured = [
                sample
                for sample in result["samples"]
                if sample["case"] == name
                and sample["kind"] == "measured"
                and (not args.device_route_decode or sample.get("optimization") == "device_route_decode")
            ]
            if len(measured) != args.repeats or any(
                type(sample.get("expert_payload_h2d_bytes")) is not int or sample["expert_payload_h2d_bytes"] != 0
                for sample in measured
            ):
                raise ValueError("Missing zero-transfer evidence for the measured V4 requests.")
            comparison = {}
            if args.device_route_decode:
                baseline = [
                    sample
                    for sample in result["samples"]
                    if sample["case"] == name
                    and sample["kind"] == "measured"
                    and sample.get("optimization") == "batched"
                ]
                if (
                    len(baseline) != args.repeats
                    or any(
                        type(sample.get("expert_payload_h2d_bytes")) is not int
                        or sample["expert_payload_h2d_bytes"] != 0
                        for sample in baseline
                    )
                    or case.get("device_route_comparison", {}).get("accepted") is not True
                ):
                    raise ValueError("Missing same-engine baseline or exact device-route comparison evidence.")
                comparison = {
                    "same_engine_batched_tpot_median_s": case["baseline_metrics"]["tpot_s"]["median"],
                    "tpot_ratio_vs_same_engine_batched": case["tpot_ratio_vs_same_engine_batched"],
                    "device_route_comparison": "PASS",
                }
            print(
                "V4_RESULT "
                + json.dumps(
                    dict(
                        case=name,
                        ttft_median_s=metrics["ttft_s"]["median"],
                        tpot_median_s=metrics["tpot_s"]["median"],
                        e2e_median_s=metrics["e2e_s"]["median"],
                        expert_payload_h2d_bytes_per_request=max(
                            sample["expert_payload_h2d_bytes"] for sample in measured
                        ),
                        v1_comparison=result["v1_comparison"],
                        **comparison,
                    )
                ),
                flush=True,
            )
        return 0
    except KeyboardInterrupt:
        report.update(status="INTERRUPTED", error="Interrupted by user.", performance_measurement_verified=False)
        raise
    except Exception as exc:
        report.update(status="FAIL", error=str(exc), performance_measurement_verified=False)
        print(f"V4_ERROR={exc}", flush=True)
        return 1
    finally:
        write_json(output / "run.json", report)
        print(f"V4_ACCEPTANCE={report['status']} V1_COMPARISON={report['v1_comparison']} REPORT={output}", flush=True)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
