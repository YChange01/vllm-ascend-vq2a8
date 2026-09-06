#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run bounded TP1 checks and retain a report even if a device child aborts.

This supervisor imports no torch/NPU modules. Each expert probe runs in its
own child process, with full output on disk and a small summary on stdout.
It does not start vLLM or exercise any experimental Cube implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_RESULT_PREFIXES = (
    "ENVIRONMENT ",
    "DEVICE ",
    "ARTIFACT_RESULT ",
    "MODEL_AUDIT ",
    "PREPARED_INPUT_RESULT ",
    "NUMERIC_FAILURE ",
    "KERNEL_RESULT ",
    "CHAIN_RESULT ",
    "VQ2A8_TP1_M1_PACKED_KERNEL_GATE=PASS ",
)


def format_compact_summary(summary: dict[str, Any]) -> str:
    """Render a copyable report without changing or rerunning the original checks."""
    results = summary.get("results", [])
    expected_count = len(summary.get("probes", []))
    passed_count = sum(result.get("passed") is True for result in results)
    status = str(summary.get("status", "unknown")).upper()
    if status == "PASSED":
        status = "PASS" if expected_count > 0 and passed_count == len(results) == expected_count else "INCOMPLETE"
    elif status == "FAILED":
        status = "FAIL"
    lines = [
        f"ACCEPTANCE={status} completed={len(results)}/{expected_count} passed={passed_count}",
        f"DEVICE={summary.get('device', '?')} PHYSICAL_NPU={summary.get('physical_npu', '?')}",
        "CASES=" + ",".join(summary.get("cases", [])),
    ]
    records = [record for result in results for record in result.get("records", [])]
    environment = next((r["data"] for r in records if r["type"] == "ENVIRONMENT"), {})
    git = environment.get("git", {})
    lines.append(f"RUN_GIT={git.get('head', '?') if isinstance(git, dict) else git}")
    if isinstance(git, dict) and git.get("status"):
        lines.append("RUN_WORKTREE=DIRTY (details in summary.json)")

    def max_metric(comparisons: list[dict[str, Any]], key: str) -> str:
        values = [c[key] for c in comparisons if isinstance(c.get(key), int | float)]
        if not values:
            return "?"
        if not all(math.isfinite(value) for value in values):
            return "NONFINITE"
        return f"{max(values):.6g}"

    for result in results:
        comparisons, prepared, kernels, chains = [], [], [], set()
        for record in result.get("records", []):
            kind, data = record["type"], record["data"]
            if kind in ("KERNEL_RESULT", "CHAIN_RESULT"):
                kernels.append(data)
                comparisons.extend(data[key] for key in ("comparison", "swiglu") if key in data)
                if "same_prepared_input_comparison" in data:
                    prepared.append(data["same_prepared_input_comparison"])
            if kind == "CHAIN_RESULT":
                chains.add(data.get("case", "unknown"))
            elif kind == "PREPARED_INPUT_RESULT":
                prepared.append(data["comparison"])
            elif kind == "NUMERIC_FAILURE":
                comparisons.append(data)
        deterministic = bool(kernels) and all(k.get("determinism", {}).get("allclose") is True for k in kernels)
        repeat_counts = [k["repeats_checked"] for k in kernels if "repeats_checked" in k]
        lines.append(
            f"PROBE={result.get('probe', '?')} {'PASS' if result.get('passed') is True else 'FAIL'} "
            f"chain_cases={len(chains)} abs={max_metric(comparisons, 'max_abs_error')} "
            f"rel_l2={max_metric(comparisons, 'relative_l2_error')} "
            f"same_fp8_abs={max_metric(prepared, 'max_abs_error')} "
            f"repeat_min={min(repeat_counts) if repeat_counts else '?'} "
            f"det={'PASS' if deterministic else 'UNKNOWN_OR_FAIL'}"
        )
        if result.get("passed") is not True:
            lines.append(
                f"  exit={result.get('returncode')} timeout={result.get('timed_out', False)} "
                f"stage={result.get('last_stage', '?')}"
            )
            lines.extend("  " + error.replace("\n", " ")[:240] for error in result.get("error_excerpt", [])[:2])

    gates = [r["data"] for r in records if r["type"] == "VQ2A8_TP1_M1_PACKED_KERNEL_GATE=PASS"]
    modes = sorted({str(g.get("native_fp8_dot", "unknown")) for g in gates})
    lines.append("NATIVE_FP8_DOT=" + (",".join(modes) if modes else "unknown"))
    lines.append(f"SERVING_VERIFIED={summary.get('serving_integration_verified', False)}")
    return "\n".join(lines) + "\n"


def summarize_log(path: Path, returncode: int | None, *, timed_out: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "passed": False,
        "returncode": returncode,
        "timed_out": timed_out,
        "log": str(path),
        "records": [],
        "last_stage": None,
        "error_excerpt": [],
    }
    saw_pass = False
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith(("KERNEL ", "CHAIN ")):
                result["last_stage"] = line.strip()
            for prefix in _RESULT_PREFIXES:
                if line.startswith(prefix):
                    try:
                        record = json.loads(line[len(prefix) :])
                    except json.JSONDecodeError:
                        continue
                    result["records"].append({"type": prefix.strip(), "data": record})
                    saw_pass |= prefix.endswith("=PASS ")
                    break
            if len(result["error_excerpt"]) < 12 and any(
                marker in line for marker in ("what():", "errorStr:", "Error:", "Assertion", "error code", "error:")
            ):
                result["error_excerpt"].append(line.strip()[:1200])
    result["passed"] = returncode == 0 and saw_pass and not timed_out
    return result


def acceptance_environment(repo: Path, physical_npu: int, device: str) -> dict[str, str]:
    child_env = dict(os.environ)
    if device.startswith("npu"):
        for key in (
            "ASCEND_VISIBLE_DEVICES",
            "NPU_VISIBLE_DEVICES",
            "ASCEND_DEVICE_ID",
            "DEVICE_ID",
            "RANK_ID",
            "LOCAL_RANK",
            "RANK",
            "WORLD_SIZE",
            "PYTHONOPTIMIZE",
        ):
            child_env.pop(key, None)
        child_env["ASCEND_RT_VISIBLE_DEVICES"] = str(physical_npu)
        child_env["ASCEND_LAUNCH_BLOCKING"] = "1"
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["PYTHONPATH"] = str(repo) + (os.pathsep + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else "")
    return child_env


def main() -> int:
    """Execute each expert in isolation and publish progress before continuing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", type=Path, help="Print a short existing summary.json report; no device access.")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--physical-npu", type=int, default=4)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--probes", default="0:0,1:0,2:0,3:0,3:127,3:255,42:255")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["deterministic", "zero", "impulse", "small", "large"],
        default=["deterministic", "zero", "impulse", "small"],
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per isolated expert probe.")
    parser.add_argument("--output-dir", type=Path, help="New directory; existing paths are refused.")
    parser.add_argument("--verify-tensor-hashes", action="store_true")
    parser.add_argument(
        "--allow-partial-artifact", action="store_true", help="Developer checks only; never serving readiness."
    )
    args = parser.parse_args()
    if args.summarize is not None:
        summary = json.loads(args.summarize.read_text(encoding="utf-8"))
        print(format_compact_summary(summary), end="")
        return 0
    if args.model is None:
        parser.error("--model is required unless --summarize is used.")
    if args.physical_npu < 0 or args.warmups < 0 or args.repeats < 1 or args.timeout < 1:
        parser.error("Invalid physical NPU, warmup, repeat or timeout value.")
    probes = args.probes.split(",")
    if len(set(probes)) != len(probes) or any(
        len(probe.split(":")) != 2 or not all(part.isdigit() for part in probe.split(":")) for probe in probes
    ):
        parser.error("Probes must be unique layer:expert pairs.")
    repo = Path(__file__).resolve().parents[1]
    model = args.model.resolve(strict=True)
    artifact = (args.artifact or model / "experts_vq_ascend_v2").resolve(strict=True)
    if args.output_dir:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = Path(tempfile.mkdtemp(prefix="vq2a8-acceptance-"))
    child_env = acceptance_environment(repo, args.physical_npu, args.device)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "device": args.device,
        "physical_npu": args.physical_npu if args.device.startswith("npu") else None,
        "python": sys.executable,
        "model": str(model),
        "artifact": str(artifact),
        "probes": probes,
        "cases": args.cases,
        "results": [],
        "serving_integration_verified": False,
        "device_kernel_performance_verified": False,
    }
    summary_path = output / "summary.json"
    short_path = output / "summary.txt"

    def save_summary() -> None:
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        short_path.write_text(format_compact_summary(summary), encoding="utf-8")

    save_summary()
    print(f"REPORT_DIR={output}", flush=True)
    for index, probe in enumerate(probes):
        log = output / f"probe-{probe.replace(':', '-')}.log"
        command = [
            sys.executable,
            str(repo / "tools/validate_vq2a8_tp1_packed_kernel.py"),
            "--model",
            str(model),
            "--artifact",
            str(artifact),
            "--device",
            args.device,
            "--probes",
            probe,
            "--cases",
            *args.cases,
            "--chain",
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
        ]
        if index == 0:
            command.append("--audit-model")
            if args.verify_tensor_hashes:
                command.append("--verify-tensor-hashes")
        if args.allow_partial_artifact:
            command.append("--allow-partial-artifact")
        print(f"PROBE_START={probe} LOG={log}", flush=True)
        timed_out = False
        returncode = None
        try:
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command,
                    cwd=repo,
                    env=child_env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
                returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
        except OSError as error:
            with log.open("a", encoding="utf-8") as stream:
                stream.write(f"SupervisorError: {error}\n")
        result = summarize_log(log, returncode, timed_out=timed_out)
        result["probe"] = probe
        result["command"] = command
        summary["results"].append(result)
        if not result["passed"]:
            summary["status"] = "failed"
            save_summary()
            print(format_compact_summary(summary), end="", flush=True)
            print(f"SHORT_REPORT={short_path}", flush=True)
            print(f"REPORT={summary_path}", flush=True)
            return 1
        save_summary()
        print(f"PROBE_PASS={probe}", flush=True)
    summary["status"] = "passed"
    save_summary()
    print(format_compact_summary(summary), end="", flush=True)
    print(f"SHORT_REPORT={short_path}", flush=True)
    print(f"ACCEPTANCE=PASS REPORT={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
