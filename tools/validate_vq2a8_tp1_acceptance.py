#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run bounded TP1 checks and retain a report even if a device child aborts.

This supervisor imports no torch/NPU modules. Each expert probe runs in its
own child process, with full output both live on stdout and saved on disk.
Only --stage model starts an offline vLLM instance. No stage starts an HTTP
server. Experimental AscendC experts require an explicit model policy/library.
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

try:
    from tools.vq2a8_baseline import freeze_baseline
    from tools.vq2a8_live_log import LiveChildLog
except ModuleNotFoundError:  # Direct script invocation without repo PYTHONPATH.
    from vq2a8_baseline import freeze_baseline
    from vq2a8_live_log import LiveChildLog

_RESULT_PREFIXES = (
    "ENVIRONMENT ",
    "DEVICE ",
    "ARTIFACT_RESULT ",
    "MODEL_AUDIT ",
    "PREPARED_INPUT_RESULT ",
    "NUMERIC_FAILURE ",
    "KERNEL_RESULT ",
    "CHAIN_RESULT ",
    "ROUTER_RESULT ",
    "MOE_RESULT ",
    "MODEL_PLAN ",
    "MODEL_LOAD_RESULT ",
    "MODEL_ROOT_FP8_RESULT ",
    "MODEL_ASCENDC_LIBRARY ",
    "MODEL_RESULT ",
    "MODEL_REPEAT_FAILURE ",
    "MODEL_BASELINE_RESULT ",
    "MODEL_CACHE_PLAN ",
    "MODEL_MOE_TIMING ",
    "MODEL_FORWARD_TIMING ",
    "MODEL_STARTUP_TIMING ",
    "VQ2A8_TP1_OFFLINE_EXECUTION_GATE=PASS ",
    "VQ2A8_TP1_MOE_GATE=PASS ",
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
    if summary.get("stage") == "moe":
        lines.insert(1, "STAGE=MOE_STANDALONE")
    elif summary.get("stage") == "model":
        lines.insert(1, "STAGE=MODEL_OFFLINE_EXECUTION")
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
        if summary.get("stage") == "model":
            runs = [r["data"] for r in result.get("records", []) if r["type"] == "MODEL_RESULT"]
            gate = next(
                (r["data"] for r in result.get("records", []) if r["type"] == "VQ2A8_TP1_OFFLINE_EXECUTION_GATE=PASS"),
                {},
            )
            latest = runs[-1] if runs else {}
            if summary.get("execution_policy") == "ascendc":
                library = latest.get("expert_backend", {}).get("library") or {}
                executed = result.get("passed") is True and gate.get("ascendc_model_execution_verified") is True
                lines.append(
                    "EXPERT_BACKEND=ascendc "
                    f"ASCENDC_MODEL_EXECUTION_VERIFIED={executed} "
                    f"LIBRARY_SHA256={library.get('sha256', 'unknown')}"
                )
            lines.append(
                f"PROBE={result.get('probe', '?')} {'PASS' if result.get('passed') is True else 'FAIL'} "
                f"runs={len(runs)} layers={latest.get('layers_executed', '?')} "
                f"prefill={latest.get('prefill_tokens', '?')} decode={latest.get('decode_steps', '?')} "
                f"finite={latest.get('finite_logits', 'unknown')} repeat_exact={gate.get('repeat_exact', 'unknown')}"
            )
            if runs:
                lines.append("TOKENS=" + ",".join(map(str, latest["generated_token_ids"])))
                lines.append(
                    f"PEAK_ALLOCATED_GIB={latest['peak_allocated_bytes'] / 1024**3:.3f} "
                    f"PEAK_RESERVED_GIB={latest['peak_reserved_bytes'] / 1024**3:.3f}"
                )
            if summary.get("root_linear_mode") == "online_fp8_sm90":
                lines.append(
                    "ROOT_LINEAR_MODE=online_fp8_sm90 "
                    f"ROOT_FP8_EXECUTION_VERIFIED={gate.get('root_fp8_execution_verified', False)}"
                )
                lines.append(
                    f"NATIVE_FP8_ROOT_MATMUL={gate.get('native_fp8_root_matmul', False)} NATIVE_FP8_EXPERT_DOT=False"
                )
            if summary.get("baseline_report"):
                comparisons = [r["data"] for r in result.get("records", []) if r["type"] == "MODEL_BASELINE_RESULT"]
                exact = len(comparisons) == 2 and all(c.get("baseline_exact") is True for c in comparisons)
                lines.append(
                    f"BASELINE_EXACT={'PASS' if exact else 'NOT_PASSED'} "
                    f"completed={len(comparisons)}/2 INDEPENDENT_REFERENCE=False"
                )
            if not result.get("passed"):
                lines.append(
                    f"  exit={result.get('returncode')} timeout={result.get('timed_out', False)} "
                    f"stage={result.get('last_stage', '?')}"
                )
                lines.extend("  " + e[:240] for e in result.get("error_excerpt", [])[:2])
            run_records = result.get("records", [])
            startups = [r["data"] for r in run_records if r["type"] == "MODEL_STARTUP_TIMING"]
            if startups:
                startup = startups[-1]
                lines.append(
                    f"STARTUP_S={startup['startup_s']:.3f} "
                    f"ENGINE_INIT_PROFILE_KV_S={startup['construct_load_profile_kv_s']:.3f} "
                    f"POLICY={startup['execution_policy']}"
                )
            for phase in ("profile", "prefill", "decode"):
                if not any(r["type"] == "MODEL_FORWARD_TIMING" for r in run_records):
                    # Legacy summaries have no timing records. Absence of
                    # instrumentation cannot establish zero executed steps.
                    break
                completed = [
                    r["data"]
                    for r in run_records
                    if r["type"] == "MODEL_FORWARD_TIMING" and r["data"].get("phase") == phase
                ]
                seconds = sum(r["elapsed_s"] for r in completed)
                lines.append(f"FORWARD={phase} completed={len(completed)} total_s={seconds:.3f}")
            timings = [r["data"] for r in run_records if r["type"] == "MODEL_MOE_TIMING"]
            if timings:
                fields = ("host_load_validate_s", "h2d_s", "prepare_s", "packed_projection_s")
                lines.append(
                    "MOE_COMPLETED_TOTAL " + " ".join(f"{key}={sum(t[key] for t in timings):.3f}" for key in fields)
                )
                if all("host_read_s" in t and "host_validate_s" in t for t in timings):
                    lines.append(
                        "HOST_BREAKDOWN "
                        + " ".join(
                            f"{key}={sum(t[key] for t in timings):.3f}" for key in ("host_read_s", "host_validate_s")
                        )
                        + " (included_in_host_load_validate_s)"
                    )
                lines.append(
                    "CACHE_COMPLETED_TOTAL "
                    + " ".join(
                        f"{key}={sum(t[key] for t in timings)}" for key in ("cache_loads", "cache_hits", "evictions")
                    )
                )
            plans = [r["data"] for r in run_records if r["type"] == "MODEL_CACHE_PLAN"]
            if plans:
                plan = plans[-1]
                lines.append(
                    f"CACHE_PLAN_GIB={plan['planned_bytes'] / 1024**3:.3f} "
                    f"ALL_EXPERTS_FIT={plan['all_experts_fit']} CAP={plan['per_layer_cache_limit']}"
                )
            continue
        comparisons, prepared, kernels, chains, moe_cases = [], [], [], set(), set()
        for record in result.get("records", []):
            kind, data = record["type"], record["data"]
            if kind in ("KERNEL_RESULT", "CHAIN_RESULT", "MOE_RESULT"):
                kernels.append(data)
                comparisons.extend(data[key] for key in ("comparison", "swiglu") if key in data)
                if "same_prepared_input_comparison" in data:
                    prepared.append(data["same_prepared_input_comparison"])
            if kind == "CHAIN_RESULT":
                chains.add(data.get("case", "unknown"))
            elif kind == "MOE_RESULT":
                moe_cases.add(data.get("case", "unknown"))
            elif kind == "PREPARED_INPUT_RESULT":
                prepared.append(data["comparison"])
            elif kind == "NUMERIC_FAILURE":
                comparisons.append(data)
        deterministic = bool(kernels) and all(k.get("determinism", {}).get("allclose") is True for k in kernels)
        repeat_counts = [k["repeats_checked"] for k in kernels if "repeats_checked" in k]
        case_count = f"moe_cases={len(moe_cases)}" if summary.get("stage") == "moe" else f"chain_cases={len(chains)}"
        if summary.get("stage") == "moe":
            route_ok = bool(kernels) and all(k.get("router_comparison", {}).get("allclose") is True for k in kernels)
            chunk_ok = bool(kernels) and all(
                k.get("token_chunk_comparison", {}).get("allclose") is True for k in kernels
            )
            extra = f"router={'PASS' if route_ok else '?'} chunks={'PASS' if chunk_ok else '?'} "
            if any(k.get("execution_policy") == "cached" for k in kernels):
                exact = all(
                    (k.get("accepted_policy_comparison") or {}).get("allclose") is True
                    and (k.get("accepted_policy_comparison") or {}).get("max_abs_error") == 0
                    for k in kernels
                )
                extra += f"baseline_exact={'PASS' if exact else 'UNKNOWN_OR_FAIL'} "
        else:
            extra = f"same_fp8_abs={max_metric(prepared, 'max_abs_error')} "
        lines.append(
            f"PROBE={result.get('probe', '?')} {'PASS' if result.get('passed') is True else 'FAIL'} "
            f"{case_count} "
            f"abs={max_metric(comparisons, 'max_abs_error')} "
            f"rel_l2={max_metric(comparisons, 'relative_l2_error')} "
            f"{extra}"
            f"repeat_min={min(repeat_counts) if repeat_counts else '?'} "
            f"det={'PASS' if deterministic else 'UNKNOWN_OR_FAIL'}"
        )
        if result.get("passed") is not True:
            lines.append(
                f"  exit={result.get('returncode')} timeout={result.get('timed_out', False)} "
                f"stage={result.get('last_stage', '?')}"
            )
            lines.extend("  " + error.replace("\n", " ")[:240] for error in result.get("error_excerpt", [])[:2])

    gates = [
        r["data"]
        for r in records
        if r["type"]
        in ("VQ2A8_TP1_M1_PACKED_KERNEL_GATE=PASS", "VQ2A8_TP1_MOE_GATE=PASS", "VQ2A8_TP1_OFFLINE_EXECUTION_GATE=PASS")
    ]
    modes = sorted({str(g.get("native_fp8_dot")) if g.get("native_fp8_dot") is not None else "unknown" for g in gates})
    lines.append("NATIVE_FP8_DOT=" + (",".join(modes) if modes else "unknown"))
    if summary.get("stage") == "model":
        lines.extend(["LOGITS_REFERENCE_VERIFIED=False", "QUALITY_VERIFIED=False"])
    lines.append(f"SERVING_VERIFIED={summary.get('serving_integration_verified', False)}")
    return "\n".join(lines) + "\n"


def summarize_log(
    path: Path, returncode: int | None, *, timed_out: bool = False, expected_pass: str | None = None
) -> dict[str, Any]:
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
            if line.startswith(("KERNEL ", "CHAIN ", "MOE ", "MODEL ")):
                result["last_stage"] = line.strip()
            for prefix in _RESULT_PREFIXES:
                if line.startswith(prefix):
                    try:
                        record = json.loads(line[len(prefix) :])
                    except json.JSONDecodeError:
                        continue
                    result["records"].append({"type": prefix.strip(), "data": record})
                    saw_pass |= prefix.endswith("=PASS ") and (expected_pass is None or prefix.strip() == expected_pass)
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
    parser.add_argument("--stage", choices=["expert", "moe", "model"], default="expert")
    parser.add_argument("--layers", default="0,3", help="MoE stage layer IDs.")
    parser.add_argument("--token-counts", type=int, nargs="+", default=[1, 3], help="MoE stage token counts.")
    parser.add_argument("--physical-npu", type=int, default=4)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--probes", default="0:0,1:0,2:0,3:0,3:127,3:255,42:255")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["deterministic", "zero", "impulse", "small", "large"],
        default=None,
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=None, help="Seconds per child (default: 1800; model: 3600).")
    parser.add_argument("--output-dir", type=Path, help="New directory; existing paths are refused.")
    parser.add_argument(
        "--baseline-report",
        type=Path,
        help="Model only: freeze a previous PASS and require exact logits/tokens (not an independent oracle).",
    )
    parser.add_argument("--verify-tensor-hashes", action="store_true")
    parser.add_argument(
        "--execution-policy",
        choices=["baseline", "cached", "ascendc"],
        default=None,
        help="MoE/model only; defaults to baseline for MoE, cached for model.",
    )
    parser.add_argument("--cache-budget-gib", type=float, default=0.0, help="Model packed cache: 0 = auto.")
    parser.add_argument("--cache-reserve-gib", type=float, default=16.0)
    parser.add_argument("--root-linear-mode", choices=["bf16", "online_fp8_sm90"], default="bf16")
    parser.add_argument(
        "--ascendc-library", type=Path, help="Model only: explicit native library; runs short hardware preflight first."
    )
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
    if args.stage == "expert" and args.execution_policy is not None:
        parser.error("--execution-policy applies to --stage moe/model only.")
    if args.execution_policy == "ascendc":
        if args.stage != "model" or args.ascendc_library is None:
            parser.error("AscendC requires --stage model and --ascendc-library.")
        if args.artifact is not None:
            parser.error("AscendC preflight/model must use the same canonical model/experts_vq_ascend_v2 artifact.")
    elif args.ascendc_library is not None:
        parser.error("--ascendc-library requires explicit --execution-policy ascendc.")
    if args.baseline_report and (args.stage != "model" or args.execution_policy in ("baseline", "ascendc")):
        parser.error("--baseline-report requires the cached model stage.")
    if args.root_linear_mode != "bf16" and (args.stage != "model" or args.baseline_report):
        parser.error("Online root FP8 requires --stage model and cannot use the phase-1 exact BF16 baseline.")
    if (
        not math.isfinite(args.cache_budget_gib)
        or args.cache_budget_gib < 0
        or not math.isfinite(args.cache_reserve_gib)
        or args.cache_reserve_gib < 1
    ):
        parser.error("Cache budget must be finite and >= 0; reserve must be finite and >= 1 GiB.")
    args.execution_policy = args.execution_policy or ("cached" if args.stage == "model" else "baseline")
    if args.cases is None:
        args.cases = ["deterministic", "zero"] if args.stage == "moe" else ["deterministic", "zero", "impulse", "small"]
    if args.timeout is None:
        args.timeout = 3600 if args.stage == "model" else 1800
    if args.stage == "model":
        if args.allow_partial_artifact or args.device != "npu:0":
            parser.error("Model stage requires a complete artifact and logical npu:0.")
        args.cases = ["short_prefill_decode_repeat"]
    if args.physical_npu < 0 or args.warmups < 0 or args.repeats < 1 or args.timeout < 1:
        parser.error("Invalid physical NPU, warmup, repeat or timeout value.")
    probes = args.probes.split(",")
    if len(set(probes)) != len(probes) or any(
        len(probe.split(":")) != 2 or not all(part.isdigit() for part in probe.split(":")) for probe in probes
    ):
        parser.error("Probes must be unique layer:expert pairs.")
    if args.stage == "moe":
        layers = args.layers.split(",")
        if not all(layer.isdigit() for layer in layers) or len(set(layers)) != len(layers):
            parser.error("MoE layers must be unique non-negative integers.")
        if any(count < 1 for count in args.token_counts):
            parser.error("MoE token counts must be positive.")
        probes = [f"layer{layer}" for layer in layers]
    elif args.stage == "model":
        probes = ["full_model"]
    repo = Path(__file__).resolve().parents[1]
    model = args.model.resolve(strict=True)
    artifact = (args.artifact or model / "experts_vq_ascend_v2").resolve(strict=True)
    if args.output_dir:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = Path(tempfile.mkdtemp(prefix="vq2a8-acceptance-"))
    frozen_baseline = None
    if args.baseline_report:
        frozen_baseline = freeze_baseline(
            args.baseline_report, output / "baseline", repo, model, artifact, args.physical_npu
        )
    child_env = acceptance_environment(repo, args.physical_npu, args.device)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "stage": args.stage,
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
        "baseline_report": str(frozen_baseline) if frozen_baseline else None,
        "root_linear_mode": args.root_linear_mode,
        "execution_policy": args.execution_policy,
    }
    summary_path = output / "summary.json"
    short_path = output / "summary.txt"

    def save_summary() -> None:
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        short_path.write_text(format_compact_summary(summary), encoding="utf-8")

    save_summary()
    print(f"REPORT_DIR={output}", flush=True)
    native_preflight = None
    if args.execution_policy == "ascendc":
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from tools.validate_vq2a8_ascendc import run_model_preflight

        try:
            args.ascendc_library = args.ascendc_library.resolve(strict=True)
            native_preflight = run_model_preflight(
                args.ascendc_library, model, args.physical_npu, output / "ascendc-preflight", min(args.timeout, 600)
            )
            summary["ascendc_preflight"] = str(native_preflight)
        except (OSError, ValueError, RuntimeError) as exc:
            summary.update(status="failed", error=str(exc))
            save_summary()
            print(f"ASCENDC_MODEL_PREFLIGHT=FAIL ERROR={exc} REPORT={summary_path}", flush=True)
            return 1
        save_summary()
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
        if args.stage == "moe":
            command = [
                sys.executable,
                str(repo / "tools/validate_vq2a8_tp1_moe.py"),
                "--model",
                str(model),
                "--artifact",
                str(artifact),
                "--device",
                args.device,
                "--layer",
                probe.removeprefix("layer"),
                "--token-counts",
                *map(str, args.token_counts),
                "--cases",
                *args.cases,
                "--warmups",
                str(args.warmups),
                "--repeats",
                str(args.repeats),
                "--execution-policy",
                args.execution_policy,
            ]
        elif args.stage == "model":
            command = [
                sys.executable,
                str(repo / "tools/validate_vq2a8_tp1_offline.py"),
                "--model",
                str(model),
                "--artifact",
                str(artifact),
                "--device",
                args.device,
                "--output-dir",
                str(output / "model-evidence"),
                "--execution-policy",
                args.execution_policy,
                "--cache-budget-gib",
                str(args.cache_budget_gib),
                "--cache-reserve-gib",
                str(args.cache_reserve_gib),
                "--root-linear-mode",
                args.root_linear_mode,
            ]
            if frozen_baseline:
                command.extend(["--baseline-report", str(frozen_baseline)])
            if native_preflight:
                command.extend(
                    ["--ascendc-library", str(args.ascendc_library), "--ascendc-preflight", str(native_preflight)]
                )
        if index == 0:
            command.append("--audit-model")
            if args.verify_tensor_hashes:
                command.append("--verify-tensor-hashes")
        if args.allow_partial_artifact:
            command.append("--allow-partial-artifact")
        print(f"PROBE_START={probe} progress={index + 1}/{len(probes)} LOG={log}", flush=True)
        timed_out = False
        returncode = None
        try:
            with log.open("w", encoding="utf-8") as stream, LiveChildLog(log, probe):
                try:
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
                finally:
                    stream.flush()
        except subprocess.TimeoutExpired:
            timed_out = True
        except OSError as error:
            with log.open("a", encoding="utf-8") as stream:
                stream.write(f"SupervisorError: {error}\n")
        result = summarize_log(
            log,
            returncode,
            timed_out=timed_out,
            expected_pass="VQ2A8_TP1_OFFLINE_EXECUTION_GATE=PASS" if args.stage == "model" else None,
        )
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
