#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and validate VQ2A8 AscendC v2 without replacing the accepted backend.

Environment -> separate build -> real-NPU projection gates -> TP1 model runs
-> optional isolated nonblocking TTFT/TPOT measurement.
No repack, pip reinstall, HTTP listener, automatic fallback, or speed promise.
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
import traceback
from pathlib import Path

from tools.validate_vq2a8_ascendc_v2 import sha256, validate_receipt
from tools.validate_vq2a8_tp1_acceptance import acceptance_environment, summarize_log
from tools.vq2a8_perf_report import validate_cases

REPO = Path(__file__).resolve().parents[1]


def commands(args, output):
    library = (args.library or args.build_dir / "libvq2a8_ascendc_v2.so").resolve()
    model = args.model.resolve()
    base = [sys.executable, "-u"]
    steps = [("environment", [*base, str(REPO / "tools/validate_vq2a8_v026_environment.py")])]
    if args.library is None:
        steps.append(
            (
                "build",
                [
                    *base,
                    str(REPO / "tools/build_vq2a8_ascendc_v2.py"),
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
                str(REPO / "tools/validate_vq2a8_ascendc_v2.py"),
                "--model",
                str(model),
                "--library",
                str(library),
                "--output",
                str(output / "preflight.json"),
            ],
        )
    )
    if not args.preflight_only:
        steps.append(
            (
                "model",
                [
                    *base,
                    str(REPO / "tools/validate_vq2a8_tp1_offline.py"),
                    "--model",
                    str(model),
                    "--artifact",
                    str(model / "experts_vq_ascend_v2"),
                    "--output-dir",
                    str(output / "model-evidence"),
                    "--execution-policy",
                    "ascendc_v2",
                    "--ascendc-v2-library",
                    str(library),
                    "--ascendc-v2-preflight",
                    str(output / "preflight.json"),
                    "--ascendc-v2-preset",
                    args.preset,
                    "--root-linear-mode",
                    "bf16",
                ],
            )
        )
    if getattr(args, "benchmark", False):
        if args.preflight_only:
            raise ValueError("Benchmark requires both preflight and full-model gates.")
        steps.append(
            (
                "performance",
                [
                    *base,
                    str(REPO / "tools/benchmark_vq2a8_ascendc_v2.py"),
                    "--model",
                    str(model),
                    "--library",
                    str(library),
                    "--preflight",
                    str(output / "preflight.json"),
                    "--output-dir",
                    str(output / "performance"),
                    "--preset",
                    args.preset,
                    "--cases",
                    args.cases,
                    "--warmups",
                    str(args.warmups),
                    "--repeats",
                    str(args.repeats),
                ],
            )
        )
    return steps


def stage_environment(environment, name):
    """Only timing runs remove diagnostic launch serialization; preserve isolation."""
    result = environment.copy()
    if name == "performance":
        result["ASCEND_LAUNCH_BLOCKING"] = "0"
    return result


def verify_performance_report(output, library, args):
    # Lazy import keeps build/plan-only independent of optional measurement code.
    from tools.benchmark_vq2a8_ascendc_v2 import verify_report

    report = json.loads((output / "performance/summary.json").read_text(encoding="utf-8"))
    if report.get("measurement_environment", {}).get("physical_npu") != str(args.physical_npu):
        raise ValueError("Benchmark report belongs to a different physical NPU.")
    identity = {"path": str(library), "sha256": sha256(library), "abi_version": 1}
    cases = validate_cases([tuple(map(int, item.split(":"))) for item in args.cases.split(",")])
    return verify_report(report, identity, args.model, cases, args.preset, args.warmups, args.repeats)


def verify_model_report(log, directory, library):
    """Do not turn exit=0, a profile pass, or an old-policy result into v2 PASS."""
    import torch
    from safetensors.torch import load_file

    from vllm_ascend.quantization.vq2a8_offline import validate_offline_evidence

    parsed = summarize_log(log, 0, expected_pass="VQ2A8_TP1_OFFLINE_EXECUTION_GATE=PASS")
    gates = [r["data"] for r in parsed["records"] if r["type"] == "VQ2A8_TP1_OFFLINE_EXECUTION_GATE=PASS"]
    if not parsed["passed"] or len(gates) != 1:
        raise ValueError("Missing unique full-model gate marker.")
    gate = gates[0]
    if (
        type(gate.get("layers")) is not int
        or gate["layers"] <= 0
        or gate.get("expert_execution_policy") != "ascendc_v2"
        or gate.get("ascendc_v2_model_execution_verified") is not True
        or gate.get("offline_execution_verified") is not True
        or gate.get("repeat_exact") is not True
        or gate.get("runs") != 2
        or gate.get("new_tokens") != 4
    ):
        raise ValueError("Full-model gate did not verify repeated v2 execution.")
    results, all_logits = [], []
    for run in range(2):
        result = json.loads((directory / f"run-{run}.json").read_text(encoding="utf-8"))
        backend = result.get("expert_backend", {})
        logits = directory / f"run-{run}-logits.safetensors"
        if (
            result.get("run") != run
            or result.get("execution_policy") != "ascendc_v2"
            or result.get("finite_logits") is not True
            or result.get("greedy_logits_agree") is not True
            or result.get("layers_executed") != gate.get("layers")
            or result.get("decode_steps") != 3
            or backend.get("policy") != "ascendc_v2"
            or backend.get("fallback_enabled") is not False
            or backend.get("library", {}).get("sha256") != sha256(library)
            or result.get("logits_sha256") != sha256(logits)
        ):
            raise ValueError("Saved v2 model result/library/logit identity mismatch.")
        tensors = load_file(str(logits), device="cpu")
        if set(tensors) != {"logits"} or tensors["logits"].ndim != 2 or tensors["logits"].dtype != torch.float32:
            raise ValueError("Saved v2 model logits are missing or malformed.")
        if result.get("root_fp8", {}).get("mode") != "bf16":
            raise ValueError("V2 bring-up requires BF16 roots.")
        validate_offline_evidence(
            {**result, "logits": tensors["logits"]},
            result["prompt_token_ids"],
            result["generated_token_ids"],
            gate["layers"],
            tensors["logits"].shape[1],
            execution_policy="ascendc_v2",
            ascendc_v2_sha256=sha256(library),
        )
        all_logits.append(tensors["logits"])
        results.append(result)
    if (
        results[0]["generated_token_ids"] != results[1]["generated_token_ids"]
        or results[0]["prompt_token_ids"] != results[1]["prompt_token_ids"]
        or not torch.equal(all_logits[0], all_logits[1])
    ):
        raise ValueError("Saved repeated model logits/tokens differ.")
    return {"gate": gate, "runs": results}


def save_report(output, report):
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        f"ASCENDC_V2_ACCEPTANCE={report['status']}",
        f"NPU_PREFLIGHT_VERIFIED={report['device_execution_verified']}",
        f"MODEL_EXECUTION_VERIFIED={report['model_integration_verified']}",
        f"PERFORMANCE_MEASUREMENT_VERIFIED={report.get('performance_measurement_verified', False)}",
        "PERFORMANCE_TARGET_MET=null NO_SPEED_THRESHOLD",
        "DEFAULT_BACKEND=UNCHANGED",
        "BASELINE_EXACT=NOT_VERIFIED",
        "NATIVE_INSTRUCTION_VERIFIED=False QUALITY_VERIFIED=False SERVING_VERIFIED=False",
    ]
    for stage in report["stages"]:
        lines.append(f"STAGE={stage['name']} EXIT={stage['exit']} TIMEOUT={stage['timeout']} LOG={stage['log']}")
    for row in report.get("performance", {}).get("summaries", []):
        lines.append(
            f"PERF {row['case']} PRESET={report['performance']['preset']} n={row['n']} "
            f"TTFT_S={row['ttft_s']:.6f} TPOT_S={row['tpot_s']:.6f} E2E_S={row['e2e_s']:.6f} "
            f"HOT_CACHE_VERIFIED={row['hot_cache_verified']}"
        )
    if "error" in report:
        lines.append(f"ERROR={report['error']}")
    lines.append(f"REPORT={output}")
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """Supervise each gate, retain failures, and isolate optional timing from debugging."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--soc", help="Exact Ascend950 device name; required unless using --library")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--build-dir", type=Path, default=REPO / "build/vq2a8-ascendc-v2")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--physical-npu", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3600, help="Maximum seconds per child stage")
    parser.add_argument("--preset", choices=["fast", "batched"], default="batched")
    parser.add_argument(
        "--benchmark", action="store_true", help="Measure v2 TTFT/TPOT after all correctness gates pass"
    )
    parser.add_argument("--cases", default="10:4", help="Benchmark prompt:output token counts; total <=128")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Print commands without any writes or device access")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.library and (not args.soc or not args.soc.startswith("Ascend950")):
        parser.error("Provide an exact --soc Ascend950... or an existing --library.")
    if args.jobs < 1 or args.physical_npu < 0 or args.timeout < 1:
        parser.error("Require positive jobs/timeout and nonnegative physical NPU.")
    if args.benchmark and args.preflight_only:
        parser.error("--benchmark cannot skip the full-model gate with --preflight-only.")
    if args.warmups < 2 or args.repeats < 5:
        parser.error("Require >=2 warmups and >=5 measured requests per case.")
    try:
        validate_cases([tuple(map(int, item.split(":"))) for item in args.cases.split(",")])
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    output = (
        args.output_dir
        or REPO
        / "reports"
        / ("vq2a8-ascendc-v2-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    ).resolve()
    steps = commands(args, output)
    if args.plan_only:
        print(json.dumps({"scope": "plan_only_no_device_execution", "steps": steps}, indent=2))
        return 0
    if platform.system() != "Linux":
        parser.error("Run on Linux Ascend950; this PC supports --plan-only and CPU contract tests.")
    from tools.accept_vq2a8_release import supervise

    output.mkdir(parents=True, exist_ok=False)
    environment = acceptance_environment(REPO, args.physical_npu, "npu:0")
    report = {
        "status": "RUNNING",
        "stages": [],
        "device_execution_verified": False,
        "model_integration_verified": False,
        "performance_measurement_verified": False,
        "scope": "TP1_OFFLINE_CONTEXT_LE_128_V2_CANDIDATE",
        "default_backend": "unchanged",
        "baseline_exact": None,
        "performance_target_met": None,
        "quality_verified": False,
        "serving_verified": False,
    }
    library = (args.library or args.build_dir / "libvq2a8_ascendc_v2.so").resolve()
    save_report(output, report)
    try:
        for name, command in steps:
            print(f"ASCENDC_V2_STAGE={name} REPORT={output}", flush=True)
            result = supervise(command, output / f"{name}.log", stage_environment(environment, name), args.timeout)
            report["stages"].append({"name": name, **result})
            save_report(output, report)
            if result["exit"] != 0 or result["timeout"]:
                raise RuntimeError(f"{name} failed; see {result['log']}")
            if name == "preflight":
                receipt = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
                identity = {"path": str(library), "sha256": sha256(library), "abi_version": 1}
                validate_receipt(receipt, identity, args.model, str(args.physical_npu))
                report["device_execution_verified"] = True
            if name == "model":
                report["model"] = verify_model_report(output / "model.log", output / "model-evidence", library)
                report["model_integration_verified"] = True
            if name == "performance":
                report["performance"] = verify_performance_report(output, library, args)
                report["performance_measurement_verified"] = True
        report["status"] = "PASS"
        return 0
    except Exception as exc:
        report.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
        return 1
    finally:
        save_report(output, report)
        print((output / "summary.txt").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
