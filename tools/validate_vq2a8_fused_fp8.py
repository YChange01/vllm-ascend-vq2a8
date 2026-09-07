#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated VQ decode + native FP8 Cube prototype gates, not model promotion.

Default: direct FP8 -> Vector/Cube bridge -> fused synthetic VQ projections.
With --model: additionally validate one real gate_up/SwiGLU/down expert chain.
Every stage gets a fresh child process; any failure stops the sequence. The
parent imports no accelerator runtime. No automatic fallback, retry or reset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    from tools.validate_vq2a8_tp1_acceptance import acceptance_environment
    from tools.validate_vq2a8_tp1_phase4 import error_excerpt
    from tools.vq2a8_live_log import LiveChildLog
except ModuleNotFoundError:
    from validate_vq2a8_tp1_acceptance import acceptance_environment
    from validate_vq2a8_tp1_phase4 import error_excerpt
    from vq2a8_live_log import LiveChildLog


CASES = ("deterministic", "zero", "impulse", "small")
ROWS = (32, 1, 3, 10)
# Full M first isolates CV lowering from masked-row handling. Non-power-of-2
# tables, multiple output groups and the maximum table/K workset follow.
SYNTHETIC_SHAPES = ((32, 32, 512, 1), (1, 32, 512, 1), (3, 64, 512, 3), (10, 64, 1024, 16), (32, 96, 4096, 32))


def case_keys(stage, probe):
    if stage in ("direct", "bridge"):
        return {f"{stage}:m{m}" for m in ROWS}
    if stage == "fused":
        return {f"m{m}:n{n}:k{k}:t{t}:{case}" for m, n, k, t in SYNTHETIC_SHAPES for case in CASES}
    return {f"{probe}:{kind}:m{m}:{case}" for kind in ("gate_up", "down") for m in ROWS for case in CASES}


def save_codegen(compiled, folder):
    """Retain compiler outputs verbatim; do NOT edit IR or certify CV lowering."""
    folder.mkdir(parents=True, exist_ok=False)
    records = []
    for name, content in compiled.asm.items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not isinstance(content, (str, bytes)):
            continue
        data = content.encode("utf-8") if isinstance(content, str) else content
        path = folder / f"kernel.{name}"
        path.write_bytes(data)
        records.append({"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    # A successful FP8 source-level dot can silently become FP16 MMA on
    # CUDA. Reject that developer fallback instead of treating it as native
    # evidence. Ascend needs separate backend-IR/binary inspection.
    ptx = compiled.asm.get("ptx")
    cuda_fp8_mma = None
    if isinstance(ptx, str):
        cuda_fp8_mma = bool(re.search(r"(?:wgmma\.mma_async|mma)\.sync\.aligned[^;]*\.e4m3\.e4m3", ptx))
        if not cuda_fp8_mma:
            raise AssertionError("CUDA codegen did not contain native E4M3 MMA; retained codegen for diagnosis.")
    return {
        "files": records,
        "metadata": str(compiled.metadata),
        "reviewed": False,
        "cuda_native_fp8_mma_detected": cuda_fp8_mma,
        "global_scratch_bytes": getattr(compiled.metadata, "global_scratch_size", None),
    }


def check_fused(inputs, dense, device, codegen_dir, *, timed=False):
    # Imports stay in the child. Sync/CPU copies below are validation only;
    # none are in the kernel wrapper or any default model hot path.
    import torch

    from tools.validate_vq2a8_phase4_kernel import accepted_rows, benchmark, bitwise_equal, compare, same_fp8_oracle
    from tools.validate_vq2a8_tp1_packed_kernel import _synchronize
    from vllm_ascend.quantization.vq2a8_fused_fp8 import launch_fused_fp8, vq2a8_fused_fp8
    from vllm_ascend.quantization.vq2a8_validation import tensor_layout

    _synchronize(device)
    started = time.perf_counter()
    actual, compiled = launch_fused_fp8(*inputs)
    _synchronize(device)
    first_ms = (time.perf_counter() - started) * 1000
    codegen = save_codegen(compiled, codegen_dir)
    oracle = compare(same_fp8_oracle(inputs[:3], dense), actual)
    baseline = compare(accepted_rows(inputs), actual)
    split = torch.cat(
        [
            vq2a8_fused_fp8(inputs[0][i : i + 1], inputs[1][i : i + 1], inputs[2][i : i + 1], *inputs[3:])
            for i in range(inputs[0].shape[0])
        ]
    )
    if not bitwise_equal(actual, split):
        raise AssertionError("Fused batching changed the per-row result.")
    for _ in range(3):
        if not bitwise_equal(actual, vq2a8_fused_fp8(*inputs)):
            raise AssertionError("Fused output is not bitwise repeatable.")
    result = {
        "oracle": oracle,
        "baseline": baseline,
        "row_chunk_exact": True,
        "repeat_exact": True,
        "first_call_including_compile_ms": first_ms,
        "codegen": codegen,
        "dot_api": "tl.dot_scaled:e4m3" if device.type == "npu" else "tl.dot:e4m3",
        "dense_expert_weight_on_device": False,
        "decoded_weight_tile": [32, 128],
        "internal_m": 32 if device.type == "npu" else 64,
        "input_layouts": {
            name: tensor_layout(t)
            for name, t in zip(("activation", "scale", "bias", "packed", "codebooks", "tile_ids"), inputs)
        },
    }
    if timed:
        result["warm_timing"] = benchmark(lambda: vq2a8_fused_fp8(*inputs), device, 3, 10)
    return result, actual


def run_expert(args, device, emit, codegen_root):
    import torch
    from safetensors.torch import save_file

    from tools.validate_vq2a8_phase4_kernel import accepted_rows, prepare_rows
    from tools.validate_vq2a8_tp1_packed_kernel import _comparison_summary, activation_case, parse_probes
    from vllm_ascend.quantization.vq2a8_reference import (
        decode_repacked_vq2a8_codebook_weight,
        deepseek_v4_swiglu_reference,
    )
    from vllm_ascend.quantization.vq2a8_runtime import open_vq2a8_tp1_artifact

    artifact = open_vq2a8_tp1_artifact(
        args.artifact or args.model / "experts_vq_ascend_v2",
        args.model / "config.json",
        require_complete=not args.allow_partial_artifact,
    )
    probe = parse_probes(args.probe)[0]
    limit = json.loads((args.model / "config.json").read_text()).get("swiglu_limit")
    payloads, matrices = {}, {}
    for kind in ("gate_up", "down"):
        print(f"FUSED_START stage=expert load={args.probe}:{kind}", flush=True)
        host, spec = artifact.load_expert(probe.layer_index, probe.expert_id, kind)
        # Real dense weights exist ONLY on CPU, never on the accelerator.
        matrices[kind] = decode_repacked_vq2a8_codebook_weight(host, spec, compute_dtype=torch.float64)
        payloads[kind] = ({key: value.to(device) for key, value in host.items()}, spec)
    for m in ROWS:
        for case in CASES:
            spec = payloads["gate_up"][1]
            hidden = torch.cat([activation_case(spec.rht_true_columns, i, 0, case) for i in range(m)]).to(device)
            baseline_hidden = hidden
            for kind in ("gate_up", "down"):
                key = f"{args.probe}:{kind}:m{m}:{case}"
                print(f"FUSED_START stage=expert key={key}", flush=True)
                payload, spec = payloads[kind]
                prepared = prepare_rows(hidden, payload, spec)
                packed = tuple(payload[name] for name in ("packed_indices", "codebooks", "codebook_tile_ids"))
                result, actual = check_fused(
                    (*prepared, *packed), matrices[kind], device, codegen_root / key.replace(":", "-")
                )
                baseline_prepared = prepare_rows(baseline_hidden, payload, spec)
                baseline = accepted_rows((*baseline_prepared, *packed))
                result["preparation_delta"] = {
                    "activation_byte_mismatches": int(
                        torch.count_nonzero(
                            prepared[0].cpu().view(torch.uint8) != baseline_prepared[0].cpu().view(torch.uint8)
                        )
                    ),
                    "scale_max_abs_error": float((prepared[1].cpu() - baseline_prepared[1].cpu()).abs().max()),
                    "bias_max_abs_error": float((prepared[2].cpu() - baseline_prepared[2].cpu()).abs().max()),
                }
                try:
                    result["chain_baseline"] = _comparison_summary(
                        baseline, actual, rtol=0.0 if case == "zero" else 0.03, atol=0.0 if case == "zero" else 0.05
                    )
                except AssertionError:
                    # Preserve passing single-projection evidence AND the
                    # failing chain; never disguise this as a prototype PASS.
                    failure_file = codegen_root / f"{key.replace(':', '-')}-chain-failure.safetensors"
                    tensors = {
                        "candidate_input": hidden,
                        "accepted_input": baseline_hidden,
                        "candidate_output": actual,
                        "accepted_output": baseline,
                    }
                    for i, name in enumerate(("activation", "scale", "bias")):
                        tensors[f"candidate_{name}"] = prepared[i]
                        tensors[f"accepted_{name}"] = baseline_prepared[i]
                    save_file(
                        {name: t.detach().cpu().contiguous().clone() for name, t in tensors.items()}, str(failure_file)
                    )
                    emit(
                        key,
                        {
                            **result,
                            "passed": False,
                            "failure_stage": "chain_baseline",
                            "failure_tensors": str(failure_file),
                        },
                    )
                    raise
                emit(key, result)
                if kind == "gate_up":
                    hidden = deepseek_v4_swiglu_reference(actual, limit)
                    baseline_hidden = deepseek_v4_swiglu_reference(baseline, limit)


def run_child(args):
    import torch

    from tools.validate_vq2a8_phase4_kernel import bitwise_equal, compare, synthetic_dense_oracle, synthetic_inputs
    from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device, environment_report
    from vllm_ascend.quantization.vq2a8_fp8_cube import ascend_fp8_unit_scale_contract
    from vllm_ascend.quantization.vq2a8_fused_fp8 import launch_cube_control

    device = torch.device(args.device)
    report = {
        "status": "running",
        "stage": args.stage,
        "device": args.device,
        "probe": args.probe,
        "environment": environment_report(),
        "dot_scale_contract": ascend_fp8_unit_scale_contract() if device.type == "npu" else None,
        "results": [],
        "npu_execution_verified": False,
        "native_instruction_verified": False,
        "on_chip_decode_verified": False,
        "model_integration_verified": False,
        "performance_verified": False,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def emit(key, result):
        report["results"].append({"key": key, "passed": True, **result})
        save()
        # Keep detailed layouts/IR hashes in JSON, not one huge console line.
        concise = {
            "key": key,
            "passed": result.get("passed", True),
            "repeat_exact": result["repeat_exact"],
            "max_abs_error": result["oracle"]["max_abs_error"],
            "relative_l2_error": result["oracle"]["relative_l2_error"],
            "codegen_dir": str(Path(result["codegen"]["files"][0]["path"]).parent)
            if result["codegen"]["files"]
            else None,
        }
        print("FUSED_RESULT " + json.dumps(concise, allow_nan=False), flush=True)

    save()
    print("ENVIRONMENT " + json.dumps(report["environment"]), flush=True)
    print("FUSED_DOT_SCALE_CONTRACT " + json.dumps(report["dot_scale_contract"]), flush=True)
    codegen_root = args.output.parent / f"{args.output.stem}-codegen"
    try:
        report["device_info"] = _initialize_device(device)
        print("DEVICE " + json.dumps(report["device_info"]), flush=True)
        if args.stage in ("direct", "bridge"):
            for m in ROWS:
                key = f"{args.stage}:m{m}"
                print(f"FUSED_START stage={args.stage} m={m} n=32 k=512 block_k=128", flush=True)
                a = ((torch.arange(m * 512).reshape(m, 512) % 31 - 15) / 8).to(torch.float8_e4m3fn)
                b = ((torch.arange(32 * 512).reshape(32, 512) % 29 - 14) / 8).to(torch.float8_e4m3fn)
                expected = (a.double() @ b.double().T * (-1 if args.stage == "bridge" else 1)).bfloat16()
                da, db = a.to(device), b.to(device)
                actual, compiled = launch_cube_control(da, db, bridge=args.stage == "bridge")
                codegen = save_codegen(compiled, codegen_root / key.replace(":", "-"))
                comparison = compare(expected, actual)
                for _ in range(3):
                    if not bitwise_equal(actual, launch_cube_control(da, db, bridge=args.stage == "bridge")[0]):
                        raise AssertionError("Cube control is not bitwise repeatable.")
                emit(
                    key,
                    {
                        "oracle": comparison,
                        "repeat_exact": True,
                        "synthetic_only": True,
                        "codegen": codegen,
                    },
                )
        elif args.stage == "fused":
            for m, n, k, tiles in SYNTHETIC_SHAPES:
                for case in CASES:
                    key = f"m{m}:n{n}:k{k}:t{tiles}:{case}"
                    print(f"FUSED_START stage=fused key={key}", flush=True)
                    host = list(synthetic_inputs(m, n, k, tiles))
                    if case == "zero":
                        host[0] = torch.zeros((m, k)).to(torch.float8_e4m3fn)
                        host[2].zero_()
                    elif case == "impulse":
                        x = torch.zeros((m, k))
                        x[torch.arange(m), (torch.arange(m) * 137 + 31) % k] = 1.0
                        host[0] = x.to(torch.float8_e4m3fn)
                        host[2].zero_()
                    elif case == "small":
                        host[1] *= 1e-6
                        host[2] *= 1e-6
                    dense = synthetic_dense_oracle(*host[3:])
                    result, _ = check_fused(
                        tuple(t.to(device) for t in host),
                        dense,
                        device,
                        codegen_root / key.replace(":", "-"),
                        timed=case == "deterministic",
                    )
                    emit(key, result)
        else:
            run_expert(args, device, emit, codegen_root)
        report.update(status="passed", npu_execution_verified=device.type == "npu")
        save()
        print(f"FUSED_STAGE=PASS stage={args.stage} checks={len(report['results'])}", flush=True)
        return 0
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


def evidence_passed(path, stage, device, probe):
    """Require complete numerical/repeat coverage, not only child exit=0."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        records = report["results"]
        if (
            report["status"] != "passed"
            or report["stage"] != stage
            or report["device"] != device
            or report["probe"] != probe
            or report["npu_execution_verified"] is not device.startswith("npu")
            or any(
                report[k] is not False
                for k in (
                    "native_instruction_verified",
                    "on_chip_decode_verified",
                    "model_integration_verified",
                    "performance_verified",
                )
            )
        ):
            return False
        if len(records) != len(case_keys(stage, probe)) or {r["key"] for r in records} != case_keys(stage, probe):
            return False
        for record in records:
            if (
                record["passed"] is not True
                or record["repeat_exact"] is not True
                or record["oracle"]["allclose"] is not True
                or not record["codegen"]["files"]
            ):
                return False
            if device.startswith("cuda") and record["codegen"].get("cuda_native_fp8_mma_detected") is not True:
                return False
            if stage in ("fused", "expert") and (
                record["row_chunk_exact"] is not True
                or record["baseline"]["allclose"] is not True
                or record["dense_expert_weight_on_device"] is not False
                or record["decoded_weight_tile"] != [32, 128]
                or record["dot_api"] != ("tl.dot_scaled:e4m3" if device.startswith("npu") else "tl.dot:e4m3")
            ):
                return False
            if stage == "expert" and record["chain_baseline"]["allclose"] is not True:
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def run_parent(args):
    repo = Path(__file__).resolve().parents[1]
    if args.output_dir:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = Path(tempfile.mkdtemp(prefix="vq2a8-fused-fp8-"))
    stages = ["direct", "bridge", "fused"] + (["expert"] if args.model else [])
    report = {
        "status": "running",
        "planned_steps": stages,
        "results": [],
        "device": args.device,
        "physical_npu": args.physical_npu,
        "phase4_complete": False,
        "native_instruction_verified": False,
        "on_chip_decode_verified": False,
        "model_integration_verified": False,
        "fused_projection_npu_execution_verified": False,
    }

    def save():
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    print(f"FUSED_REPORT_DIR={output} DEFAULT_MODEL_BACKEND=UNCHANGED", flush=True)
    for stage in stages:
        evidence, log = output / f"{stage}.json", output / f"{stage}.log"
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--stage",
            stage,
            "--device",
            args.device,
            "--probe",
            args.probe,
            "--output",
            str(evidence),
        ]
        if args.model:
            command += ["--model", str(args.model)]
        if args.artifact:
            command += ["--artifact", str(args.artifact)]
        if args.allow_partial_artifact:
            command += ["--allow-partial-artifact"]
        print(f"FUSED_STEP_START={stage} LOG={log}", flush=True)
        code, timeout, error = None, False, None
        try:
            with log.open("w", encoding="utf-8") as stream, LiveChildLog(log, stage):
                child = subprocess.run(
                    command,
                    cwd=repo,
                    env=acceptance_environment(repo, args.physical_npu, args.device),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
            code = child.returncode
        except subprocess.TimeoutExpired:
            timeout = True
        except OSError as exception:
            error = str(exception)
        passed = code == 0 and not timeout and evidence_passed(evidence, stage, args.device, args.probe)
        report["results"].append(
            {
                "stage": stage,
                "passed": passed,
                "returncode": code,
                "timeout": timeout,
                "error": error,
                "log": str(log),
                "evidence": str(evidence),
                "error_excerpt": error_excerpt(log) if not passed else [],
            }
        )
        report["status"] = "running" if passed else "failed"
        save()
        if not passed:
            for line in report["results"][-1]["error_excerpt"]:
                print(line, flush=True)
            print(f"FUSED_PROTOTYPE=FAIL stage={stage} REPORT={output} (remaining steps skipped)", flush=True)
            return 1
        print(f"FUSED_STEP_PASS={stage}", flush=True)
    report["status"] = "passed"
    report["fused_projection_npu_execution_verified"] = args.device.startswith("npu")
    save()
    print(f"FUSED_PROTOTYPE=PASS DEVICE={args.device} REPORT={output}", flush=True)
    print("SCOPE=STANDALONE_PROTOTYPE NATIVE_INSTRUCTION_VERIFIED=False ON_CHIP_DECODE_VERIFIED=False", flush=True)
    print("PHASE4=INCOMPLETE MODEL_INTEGRATION_VERIFIED=False PERFORMANCE_VERIFIED=False", flush=True)
    print("PHASE2=SKIPPED PHASE5=DEFERRED QUALITY_VERIFIED=False SERVING_VERIFIED=False", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["npu:0", "cuda:0"], default="npu:0")
    parser.add_argument("--physical-npu", type=int, default=4)
    parser.add_argument("--model", type=Path, help="Opt in to one real expert after synthetic gates.")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--allow-partial-artifact", action="store_true", help="Developer CUDA only.")
    parser.add_argument("--probe", default="0:0")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stage", choices=["direct", "bridge", "fused", "expert"], help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.physical_npu < 0 or args.timeout <= 0 or not re.fullmatch(r"[0-9]+:[0-9]+", args.probe):
        parser.error("Require nonnegative physical NPU, positive timeout and one layer:expert probe.")
    if args.allow_partial_artifact and args.device != "cuda:0":
        parser.error("Partial artifacts are developer CUDA only, not NPU acceptance.")
    if (args.artifact or args.stage == "expert" or args.allow_partial_artifact) and not args.model:
        parser.error("Expert/artifact options require --model.")
    if args.stage and (args.output is None or args.output.exists()):
        parser.error("Child stages require a new --output file.")
    if not args.stage and args.output:
        parser.error("Use --output-dir for the supervisor.")
    if args.model:
        args.model = args.model.resolve(strict=True)
        args.artifact = (args.artifact or args.model / "experts_vq_ascend_v2").resolve(strict=True)
    return run_child(args) if args.stage else run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
