#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build + preflight + ordered opt-in optimization experiments on Ascend950.

No repack, pip reinstall, cache deletion or change to the accepted default path.
Use a separate build directory so the existing known-good .so is preserved.
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
import platform
import re
import traceback
from pathlib import Path

from tools.profile_vq2a8_ascendc import write_json
from tools.vq2a8_perf_report import validate_cases

PRESETS = ("fast", "batched", "fwht", "pipeline", "prepare_graph")
REPO = Path(__file__).resolve().parents[1]
ENVIRONMENT_PREFIX = "VQ2A8_V023_ENVIRONMENT "
MAX_FAILURE_LOG_BYTES = 256 * 1024
MAX_FAILURE_LINE_BYTES = 64 * 1024
MAX_FAILURE_LINES = 200
MAX_FAILURE_CAUSES = 6
MAX_FAILURE_CAUSE_CHARS = 300


def _redact_cause(value):
    """Bound selected diagnostic text; never forward arbitrary log records."""
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
    value = " ".join(value.split())
    # Suppress whole URLs, including userinfo, proxy addresses and query keys.
    value = re.sub(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s<>\"']+", "[REDACTED_URL]", value)
    value = re.sub(r"(?i)\b(?:Bearer|Basic)\s+[\w.~+/=-]+", "[REDACTED_AUTH]", value)
    sensitive = (
        r"[\w-]*(?:proxy|password|passwd|pwd|token|secret|api[_-]?key|access[_-]?key|authorization|credential)[\w-]*"
    )
    value = re.sub(
        rf"(?i)(\b{sensitive}[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
        r"\1[REDACTED]",
        value,
    )
    value = re.sub(rf"(?i)(--{sensitive}\s+)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)", r"\1[REDACTED]", value)
    value = re.sub(
        r"\b(?:sk-[A-Za-z0-9_-]{6,}|hf_[A-Za-z0-9]{6,}|gh[pousr]_[A-Za-z0-9]{6,}|github_pat_[A-Za-z0-9_]{6,})\b",
        "[REDACTED_TOKEN]",
        value,
    )
    value = "".join(char for char in value if char.isprintable())
    if len(value) > MAX_FAILURE_CAUSE_CHARS:
        value = value[: MAX_FAILURE_CAUSE_CHARS - 3] + "..."
    return value


def failure_causes(log, *, timed_out=False):
    """Read a bounded tail and select environment errors, never dump raw logs.

    Unknown or malformed logs produce a fixed message. Only the environment
    tool's errors and unsuccessful pip-check output are candidates for display;
    package/environment dictionaries and successful subprocess output are not.
    """
    causes = []
    if timed_out:
        causes.append("Child exceeded its stage timeout.")
    try:
        with Path(log).open("rb") as source:
            size = os.fstat(source.fileno()).st_size
            offset = max(0, size - MAX_FAILURE_LOG_BYTES)
            source.seek(offset)
            raw = source.read(MAX_FAILURE_LOG_BYTES)
        if offset:
            raw = raw.partition(b"\n")[2]
        for line in reversed(raw.splitlines()[-MAX_FAILURE_LINES:]):
            if len(line) > MAX_FAILURE_LINE_BYTES:
                continue
            text = line.decode("utf-8", errors="replace")
            if not text.startswith(ENVIRONMENT_PREFIX):
                continue
            try:
                report = json.loads(text[len(ENVIRONMENT_PREFIX) :])
            except (ValueError, RecursionError):
                continue
            if not isinstance(report, dict):
                continue
            errors = report.get("errors")
            if isinstance(errors, list):
                causes.extend(_redact_cause(error) for error in errors[:MAX_FAILURE_CAUSES] if isinstance(error, str))
            pip_check = report.get("pip_check")
            if (
                isinstance(pip_check, dict)
                and type(pip_check.get("exit")) is int
                and pip_check["exit"] != 0
                and isinstance(pip_check.get("output"), str)
            ):
                causes.extend(
                    _redact_cause("pip check: " + line)
                    for line in pip_check["output"].splitlines()[:MAX_FAILURE_CAUSES]
                    if line.strip()
                )
            break
    except (OSError, ValueError):
        causes.append("Child log is unavailable; no diagnostic text was read.")
    causes = list(dict.fromkeys(cause for cause in causes if cause))
    return causes[:MAX_FAILURE_CAUSES] or [
        "No recognized environment cause in the bounded log tail; inspect the full log."
    ]


def commands(args, output):
    library = args.library.resolve() if args.library else args.build_dir.resolve() / "libvq2a8_ascendc.so"
    base = [sys.executable, "-u"]
    steps = [("environment", [*base, str(REPO / "tools/validate_vq2a8_v023_environment.py")])]
    if not args.library:
        steps.append(
            (
                "build",
                [
                    *base,
                    str(REPO / "tools/build_vq2a8_ascendc.py"),
                    "--soc",
                    args.soc,
                    "--build-dir",
                    str(args.build_dir.resolve()),
                    "--jobs",
                    str(args.jobs),
                ],
            )
        )
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
                str(library),
                "--physical-npu",
                str(args.physical_npu),
                "--output-dir",
                str(output / "preflight"),
            ],
        )
    )
    steps.append(
        (
            "performance",
            [
                *base,
                str(REPO / "tools/benchmark_vq2a8_offline.py"),
                "--model",
                str(args.model.resolve()),
                "--library",
                str(library),
                "--preflight",
                str(output / "preflight/preflight.json"),
                "--output-dir",
                str(output / "result"),
                "--cases",
                args.cases,
                "--warmups",
                str(args.warmups),
                "--repeats",
                str(args.repeats),
                "--optimization-presets",
                args.presets,
                *(["--profile-optimization"] if args.profile else []),
            ],
        )
    )
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, help="Reuse an explicitly rebuilt candidate library")
    parser.add_argument("--soc", help="Exact torch.npu.get_device_name(0) value; required when building")
    parser.add_argument("--build-dir", type=Path, default=REPO / "build/vq2a8-ascendc-opt3")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--physical-npu", type=int, default=0)
    parser.add_argument("--presets", default=",".join(PRESETS))
    parser.add_argument("--cases", default="10:4,32:32,96:32")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true", help="Separate untimed CPU/NPU traces, not speed samples")
    parser.add_argument("--timeout", type=int, default=14400, help="Per-step child timeout in seconds")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true", help="Print commands without building or touching NPU")
    args = parser.parse_args()
    if not args.library and (not args.soc or not args.soc.startswith("Ascend950")):
        parser.error("Supply the exact --soc Ascend950... or an existing --library")
    if args.physical_npu < 0 or args.jobs < 1 or args.timeout < 1 or args.warmups < 2 or args.repeats < 5:
        parser.error("Require a nonnegative NPU, positive jobs/timeout, >=2 warmups and >=5 repeats")
    selected = args.presets.split(",")
    if len(set(selected)) != len(selected) or any(name not in PRESETS for name in selected):
        parser.error(f"Presets must be distinct members of {PRESETS}")
    validate_cases([tuple(map(int, case.split(":"))) for case in args.cases.split(",")])
    output = (
        args.output_dir
        or REPO
        / "reports"
        / ("vq2a8-opt3-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    ).resolve()
    steps = commands(args, output)
    if args.plan_only:
        print(json.dumps(dict(scope="plan_only_no_device_execution", output=str(output), steps=steps), indent=2))
        return 0
    if platform.system() != "Linux":
        parser.error("Execute on the Linux Ascend950 server; Windows supports --plan-only")
    from tools.accept_vq2a8_release import supervise

    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(ASCEND_RT_VISIBLE_DEVICES=str(args.physical_npu), ASCEND_LAUNCH_BLOCKING="0")
    report = dict(status="RUNNING", stages=[], output=str(output), device_execution_verified=False)
    try:
        for name, command in steps:
            print(f"OPTIMIZATION_STAGE={name} REPORT={output}", flush=True)
            result = supervise(command, output / f"{name}.log", env, args.timeout)
            failed = result["exit"] != 0 or result["timeout"]
            if failed:
                result["failure_causes"] = failure_causes(result["log"], timed_out=result["timeout"])
            report["stages"].append(dict(name=name, **result))
            write_json(output / "run.json", report)
            if failed:
                cause = " | ".join(result["failure_causes"])
                raise RuntimeError(f"{name} failed: {cause}; full traceback/log: {result['log']}")
        result = json.loads((output / "result/summary.json").read_text())
        report.update(
            status=result["status"], device_execution_verified=result.get("performance_measurement_verified", False)
        )
        return 0 if report["status"] == "PASS" else 2
    except Exception as exc:
        report.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
        print(f"OPTIMIZATION_ERROR={exc}", flush=True)
        return 1
    finally:
        write_json(output / "run.json", report)
        print(f"OPTIMIZATION_STATUS={report['status']} REPORT={output}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
