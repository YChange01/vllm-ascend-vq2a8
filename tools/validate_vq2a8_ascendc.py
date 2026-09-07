#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated native AscendC gates. Python is the harness, not the device kernel.

Direct -> sign-bit bridge -> packed synthetic -> optional real expert chain.
The accepted Triton/vector implementation is a comparison ONLY, never a
candidate fallback. Failed stages stop the run; default model is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.build_vq2a8_ascendc import source_hashes  # noqa: E402
from tools.validate_vq2a8_tp1_acceptance import acceptance_environment  # noqa: E402
from tools.validate_vq2a8_tp1_phase4 import error_excerpt  # noqa: E402
from tools.vq2a8_live_log import LiveChildLog  # noqa: E402

ROWS = (32, 1, 3, 10, 16, 17)
CASES = ("deterministic", "zero", "impulse", "small")
# Covers both AIVs, masked rows, signed words, multiple output groups and
# K iterations, and the largest uint8-addressable codebook tile count.
SHAPES = (
    (32, 32, 512, 1),
    (1, 32, 512, 1),
    (3, 64, 1024, 3),
    (10, 64, 4096, 32),
    (16, 32, 512, 32),
    (17, 64, 1024, 256),
    (32, 96, 512, 3),
)
BOUNDARY_SHAPES = (
    (2, 32, 512, 1),
    (15, 64, 1024, 3),
    (31, 96, 4096, 32),
    (32, 928, 512, 3),  # 29 output groups: exceeds this 950's 28 AICs
    (1, 65536, 512, 1),
    (1, 32, 65536, 256),
)
TIMING_ROWS = (1, 17, 32)


def library_evidence(library):
    library = library.resolve(strict=True)
    manifest = json.loads((library.parent / "build-manifest.json").read_text())
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    if manifest.get("status") != "built" or manifest.get("library_sha256") != digest:
        raise ValueError("Missing successful build manifest or native library hash mismatch.")
    if manifest.get("source_sha256") != source_hashes():
        raise ValueError("Native sources changed since build; rebuild this standalone library.")
    return {"path": str(library), "sha256": digest, "build": manifest}


def require_hardware_runtime():
    """Keep simulator preload/config out of physical-device regression receipts."""
    if sys.platform != "linux":
        raise RuntimeError("Physical AscendC regression requires the Linux NPU host.")
    mapped = [
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/" in line and ("camodel" in line.lower() or "/simulator/" in line.lower())
    ]
    if mapped or os.environ.get("CAMODEL_CONFIG_PATH") or "camodel" in os.environ.get("LD_PRELOAD", "").lower():
        raise RuntimeError("Simulator runtime/config detected; use a normal hardware CANN shell.")
    return {"checked": True, "simulator_runtime_paths": []}


def check_projection(inputs, dense, *, baseline=True):
    import torch

    from tools.validate_vq2a8_phase4_kernel import accepted_rows, bitwise_equal, compare, same_fp8_oracle
    from vllm_ascend.quantization.vq2a8_ascendc import vq2a8_ascendc

    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    started = time.perf_counter()
    actual = vq2a8_ascendc(*inputs)
    torch.npu.synchronize()
    first_call_ms = (time.perf_counter() - started) * 1000
    peak_delta = torch.npu.max_memory_allocated() - before
    result = {
        "oracle": compare(same_fp8_oracle(inputs[:3], dense), actual),
        "first_projection_call_ms": first_call_ms,
        # Native CANN internal allocations are not covered by this counter.
        "torch_allocator_peak_delta_bytes": peak_delta,
        "output_bytes": actual.numel() * actual.element_size(),
    }
    if baseline:
        result["accepted_baseline"] = compare(accepted_rows(inputs), actual)
    for _ in range(3):
        if not bitwise_equal(actual, vq2a8_ascendc(*inputs)):
            raise AssertionError("AscendC output is not bitwise repeatable.")
    rows = [
        vq2a8_ascendc(inputs[0][i : i + 1], inputs[1][i : i + 1], inputs[2][i : i + 1], *inputs[3:])
        for i in range(inputs[0].shape[0])
    ]
    if not bitwise_equal(actual, torch.cat(rows)):
        raise AssertionError("AscendC row chunking changed the result.")
    result.update(repeat_exact=True, row_chunk_exact=True)
    return result, actual


def run_expert(args, device, emit):
    import torch
    from safetensors.torch import save_file

    from tools.validate_vq2a8_phase4_kernel import accepted_rows, benchmark, prepare_rows
    from tools.validate_vq2a8_tp1_packed_kernel import _comparison_summary, activation_case, parse_probes
    from vllm_ascend.quantization.vq2a8_ascendc import vq2a8_ascendc
    from vllm_ascend.quantization.vq2a8_reference import (
        decode_repacked_vq2a8_codebook_weight,
        deepseek_v4_swiglu_reference,
    )
    from vllm_ascend.quantization.vq2a8_runtime import open_vq2a8_tp1_artifact

    artifact = open_vq2a8_tp1_artifact(
        args.model / "experts_vq_ascend_v2", args.model / "config.json", require_complete=True
    )
    probe = parse_probes(args.probe)[0]
    limit = json.loads((args.model / "config.json").read_text()).get("swiglu_limit")
    payloads, dense = {}, {}
    for kind in ("gate_up", "down"):
        print(f"ASCENDC_EXPERT_LOAD={kind} PROBE={args.probe}", flush=True)
        host, spec = artifact.load_expert(probe.layer_index, probe.expert_id, kind)
        # The full decoded expert weight is CPU oracle data ONLY.
        dense[kind] = decode_repacked_vq2a8_codebook_weight(host, spec, compute_dtype=torch.float64)
        payloads[kind] = ({name: value.to(device) for name, value in host.items()}, spec)
    timed = args.stage == "timing"
    for m in TIMING_ROWS if timed else ROWS:
        for case in ("deterministic",) if timed else CASES:
            spec = payloads["gate_up"][1]
            hidden = torch.cat([activation_case(spec.rht_true_columns, i, 0, case) for i in range(m)]).to(device)
            accepted_hidden = hidden
            for kind in ("gate_up", "down"):
                key = f"{kind}:m{m}:{case}"
                print(f"ASCENDC_START stage={args.stage} key={key}", flush=True)
                payload, spec = payloads[kind]
                prepared = prepare_rows(hidden, payload, spec)
                accepted_prepared = prepare_rows(accepted_hidden, payload, spec)
                packed = tuple(payload[name] for name in ("packed_indices", "codebooks", "codebook_tile_ids"))
                result, actual = check_projection((*prepared, *packed), dense[kind])
                accepted = accepted_rows((*accepted_prepared, *packed))
                try:
                    result["independent_chain"] = _comparison_summary(
                        accepted, actual, rtol=0 if case == "zero" else 0.03, atol=0 if case == "zero" else 0.05
                    )
                except AssertionError:
                    tensors = {
                        "candidate_output": actual,
                        "accepted_output": accepted,
                        "candidate_input": hidden,
                        "accepted_input": accepted_hidden,
                    }
                    for i, name in enumerate(("activation", "scale", "bias")):
                        tensors[f"candidate_{name}"] = prepared[i]
                        tensors[f"accepted_{name}"] = accepted_prepared[i]
                    failure = args.output.parent / f"{args.output.stem}-{key.replace(':', '-')}-failure.safetensors"
                    save_file(
                        {name: t.detach().cpu().contiguous().clone() for name, t in tensors.items()}, str(failure)
                    )
                    emit(key, {**result, "passed": False, "failure_tensors": str(failure)})
                    raise
                if timed:
                    inputs = (*prepared, *packed)
                    print(f"ASCENDC_TIMING_START key={key}", flush=True)
                    candidate = benchmark(
                        lambda inputs=inputs: vq2a8_ascendc(*inputs), device, args.warmups, args.repeats
                    )
                    baseline = benchmark(
                        lambda inputs=inputs: accepted_rows(inputs), device, args.warmups, args.repeats
                    )
                    result["timings"] = {
                        "candidate": candidate,
                        "accepted_baseline": baseline,
                        "wall_median_ratio_baseline_over_candidate": baseline["wall_ms"]["median"]
                        / candidate["wall_ms"]["median"],
                        "launch_blocking": False,
                        "scope": "resident_prepared_projection_only",
                    }
                emit(key, result)
                if kind == "gate_up":
                    hidden = deepseek_v4_swiglu_reference(actual, limit)
                    accepted_hidden = deepseek_v4_swiglu_reference(accepted, limit)


def run_child(args):
    report = {
        "status": "running",
        "stage": args.stage,
        "probe": args.probe,
        "implementation": "ascendc",
        "results": [],
        "native_instruction_verified": False,
        "on_chip_decode_verified": False,
        "model_integration_verified": False,
        "performance_verified": False,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def emit(key, result):
        record = {"key": key, "passed": True, **result}
        report["results"].append(record)
        save()
        print("ASCENDC_RESULT " + json.dumps(record, allow_nan=False), flush=True)

    save()
    try:
        require_hardware_runtime()
        report["library"] = library_evidence(args.library)
        print("ASCENDC_LIBRARY " + json.dumps(report["library"]), flush=True)
        import torch
        import torch_npu  # noqa: F401

        from tools.validate_vq2a8_phase4_kernel import bitwise_equal, compare, synthetic_dense_oracle, synthetic_inputs
        from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device, environment_report
        from vllm_ascend.quantization.vq2a8_ascendc import cube_control, load_library

        torch.set_num_threads(4)
        device = torch.device("npu:0")
        report["environment"] = environment_report()
        report["device"] = _initialize_device(device)
        report["hardware_runtime"] = require_hardware_runtime()
        if not report["device"].get("name", "").startswith("Ascend950"):
            raise RuntimeError("Only Ascend950 is supported.")
        load_library(args.library)
        print("ASCENDC_DEVICE " + json.dumps(report["device"]), flush=True)
        save()
        if args.stage in ("direct", "bridge"):
            for m in ROWS:
                print(f"ASCENDC_START stage={args.stage} m={m} n=32 k=512", flush=True)
                a = ((torch.arange(m * 512).reshape(m, 512) % 31 - 15) / 8).to(torch.float8_e4m3fn)
                b = ((torch.arange(32 * 512).reshape(32, 512) % 29 - 14) / 8).to(torch.float8_e4m3fn)
                bridge = args.stage == "bridge"
                expected = (a.double() @ b.double().T * (-1 if bridge else 1)).bfloat16()
                da, db = a.to(device), b.to(device)
                actual = cube_control(da, db, bridge=bridge)
                torch.npu.synchronize()
                result = {"oracle": compare(expected, actual), "repeat_exact": True}
                for _ in range(3):
                    if not bitwise_equal(actual, cube_control(da, db, bridge=bridge)):
                        raise AssertionError("Native Cube control is not bitwise repeatable.")
                emit(f"m{m}", result)
        elif args.stage in ("fused", "boundaries"):
            for m, n, k, tiles in SHAPES if args.stage == "fused" else BOUNDARY_SHAPES:
                for case in CASES:
                    key = f"m{m}:n{n}:k{k}:t{tiles}:{case}"
                    print(f"ASCENDC_START stage={args.stage} key={key}", flush=True)
                    inputs = list(synthetic_inputs(m, n, k, tiles))
                    if case != "deterministic":
                        x = inputs[0].float()
                        if case == "zero":
                            x.zero_()
                        elif case == "impulse":
                            x.zero_()
                            x[:, 0] = 1
                            x[:, -1] = -2
                        elif case == "small":
                            x *= 2**-8
                        inputs[0] = x.to(torch.float8_e4m3fn)
                    dense = synthetic_dense_oracle(*inputs[3:])
                    result, _ = check_projection(tuple(t.to(device) for t in inputs), dense)
                    emit(key, result)
        elif args.stage in ("expert", "timing"):
            if args.stage == "timing" and os.environ.get("ASCEND_LAUNCH_BLOCKING") != "0":
                raise RuntimeError("Timing requires a separate child with ASCEND_LAUNCH_BLOCKING=0.")
            run_expert(args, device, emit)
        report["status"] = "passed"
        save()
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


def expected_keys(stage):
    if stage in ("direct", "bridge"):
        return {f"m{m}" for m in ROWS}
    if stage in ("fused", "boundaries"):
        shapes = SHAPES if stage == "fused" else BOUNDARY_SHAPES
        return {f"m{m}:n{n}:k{k}:t{t}:{case}" for m, n, k, t in shapes for case in CASES}
    if stage in ("expert", "timing"):
        rows, cases = (TIMING_ROWS, ("deterministic",)) if stage == "timing" else (ROWS, CASES)
        return {f"{kind}:m{m}:{case}" for kind in ("gate_up", "down") for m in rows for case in cases}
    raise ValueError("Unknown stage")


def timing_evidence_passed(record):
    timing = record["timings"]
    if timing["launch_blocking"] is not False or timing["scope"] != "resident_prepared_projection_only":
        return False
    for name in ("candidate", "accepted_baseline"):
        stats = timing[name]
        if (
            type(stats["warmups"]) is not int
            or type(stats["repeats"]) is not int
            or stats["warmups"] < 3
            or stats["repeats"] < 10
        ):
            return False
        for clock in ("event_ms", "wall_ms"):
            values = [stats[clock][key] for key in ("min", "median", "p95")]
            if not all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in values) or values != sorted(
                values
            ):
                return False
    ratio = timing["wall_median_ratio_baseline_over_candidate"]
    return (
        type(ratio) in (float, int)
        and math.isfinite(ratio)
        and math.isclose(
            ratio, timing["accepted_baseline"]["wall_ms"]["median"] / timing["candidate"]["wall_ms"]["median"]
        )
    )


def evidence_passed(path, stage, digest, probe="0:0"):
    try:
        evidence = json.loads(path.read_text())
        rows = evidence["results"]
        keys = expected_keys(stage)
        return (
            evidence["status"] == "passed"
            and evidence["stage"] == stage
            and evidence["implementation"] == "ascendc"
            and evidence["probe"] == probe
            and evidence["device"]["type"] == "npu"
            and evidence["device"]["soc"] == 260
            and evidence["library"]["sha256"] == digest
            and len(rows) == len(keys)
            and {r["key"] for r in rows} == keys
            and all(r["passed"] is True and r["repeat_exact"] is True for r in rows)
            and all(r["oracle"]["allclose"] is True for r in rows)
            and all(
                r.get("row_chunk_exact") is True for r in rows if stage in ("fused", "boundaries", "expert", "timing")
            )
            and all(
                r["accepted_baseline"]["allclose"] is True
                for r in rows
                if stage in ("fused", "boundaries", "expert", "timing")
            )
            and all(r["independent_chain"]["allclose"] is True for r in rows if stage in ("expert", "timing"))
            and all(timing_evidence_passed(r) for r in rows if stage == "timing")
            and all(
                evidence[name] is False
                for name in (
                    "native_instruction_verified",
                    "on_chip_decode_verified",
                    "model_integration_verified",
                    "performance_verified",
                )
            )
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def checked_model_preflight(library_path, receipt_path):
    """Bind both small hardware gates to the library the model worker will load."""
    library = library_evidence(library_path)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "passed" or receipt.get("library_sha256") != library["sha256"]:
        raise ValueError("AscendC model requires a passing same-library short hardware preflight.")
    for stage in ("fused", "timing"):
        path = receipt_path.parent / f"{stage}.json"
        if not evidence_passed(path, stage, library["sha256"], "0:0"):
            raise ValueError(f"AscendC model preflight is missing/incomplete: {stage}.")
        child = json.loads(path.read_text())
        if child.get("hardware_runtime") != {"checked": True, "simulator_runtime_paths": []}:
            raise ValueError("Preflight must use physical NPU execution, not the simulator.")
        if receipt.get("evidence_sha256", {}).get(stage) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Preflight child evidence changed after collection.")
    return library


def run_model_preflight(library_path, model, physical_npu, directory, timeout=600):
    """28 synthetic + six real expert timing/numerical cases; no simulation."""
    from tools.validate_vq2a8_ascendc_suite import run_step

    require_hardware_runtime()
    library = library_evidence(library_path)
    directory.mkdir(parents=True, exist_ok=False)
    receipt_path = directory / "preflight.json"
    receipt = {"status": "running", "library_sha256": library["sha256"], "evidence_sha256": {}, "results": []}
    args = SimpleNamespace(
        library=library_path, model=model, physical_npu=physical_npu, timeout=timeout, warmups=3, repeats=10
    )
    print(f"ASCENDC_MODEL_PREFLIGHT cases=34 simulator_reruns=0 REPORT={receipt_path}", flush=True)
    try:
        for stage in ("fused", "timing"):
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            step = {"id": stage, "stage": stage, "probe": "0:0"}
            result = run_step(args, step, directory, library["sha256"])
            receipt["results"].append(result)
            if not result["passed"]:
                raise RuntimeError(f"AscendC short {stage} regression failed; model loading was not started.")
            path = directory / f"{stage}.json"
            receipt["evidence_sha256"][stage] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt["status"] = "passed"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        checked_model_preflight(library_path, receipt_path)
    except Exception as exc:
        receipt.update(status="failed", error=str(exc))
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"ASCENDC_MODEL_PREFLIGHT=PASS REPORT={receipt_path}", flush=True)
    return receipt_path


def run_parent(args):
    library = library_evidence(args.library)
    directory = args.output_dir or Path(tempfile.mkdtemp(prefix="vq2a8-ascendc-"))
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Use an empty report directory to avoid stale acceptance evidence.")
    stages = ["direct", "bridge", "fused"] + (["expert"] if args.model else [])
    report = {
        "status": "running",
        "implementation": "ascendc",
        "stages": [],
        "library": library,
        "default_model_backend": "unchanged",
        "native_instruction_verified": False,
        "on_chip_decode_verified": False,
        "model_integration_verified": False,
        "performance_verified": False,
        "quality_verified": False,
        "serving_verified": False,
    }
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"ASCENDC_REPORT_DIR={directory} DEFAULT_MODEL_BACKEND=UNCHANGED", flush=True)
    for stage in stages:
        log, evidence = directory / f"{stage}.log", directory / f"{stage}.json"
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--library",
            str(args.library),
            "--stage",
            stage,
            "--output",
            str(evidence),
            "--probe",
            args.probe,
        ]
        if args.model:
            command += ["--model", str(args.model)]
        print(f"ASCENDC_STEP_START={stage} LOG={log}", flush=True)
        code, timed_out = None, False
        with log.open("w") as stream, LiveChildLog(log, stage):
            try:
                child = subprocess.run(
                    command,
                    cwd=REPO,
                    env=acceptance_environment(REPO, args.physical_npu, "npu:0"),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
                code = child.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
        passed = code == 0 and not timed_out and evidence_passed(evidence, stage, library["sha256"], args.probe)
        report["stages"].append(
            {
                "stage": stage,
                "passed": passed,
                "exit": code,
                "timeout": timed_out,
                "log": str(log),
                "evidence": str(evidence),
            }
        )
        report["status"] = "running" if passed else "failed"
        (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if not passed:
            for line in error_excerpt(log):
                print(line, flush=True)
            print(f"ASCENDC_PROTOTYPE=FAIL stage={stage} REPORT={directory} (remaining stages skipped)", flush=True)
            return 1
        print(f"ASCENDC_STEP_PASS={stage}", flush=True)
    report.update(status="passed", standalone_npu_execution_verified=True)
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"ASCENDC_PROTOTYPE=PASS SCOPE=STANDALONE REPORT={directory}", flush=True)
    print("NATIVE_INSTRUCTION_VERIFIED=False ON_CHIP_DECODE_VERIFIED=False PERFORMANCE_VERIFIED=False", flush=True)
    print("MODEL_INTEGRATION_VERIFIED=False QUALITY_VERIFIED=False SERVING_VERIFIED=False", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=REPO / "build/vq2a8-ascendc/libvq2a8_ascendc.so")
    parser.add_argument("--physical-npu", type=int, default=4)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--probe", default="0:0")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--stage", choices=("direct", "bridge", "fused", "boundaries", "expert", "timing"), help=argparse.SUPPRESS
    )
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.physical_npu < 0 or args.timeout <= 0 or not re.fullmatch(r"[0-9]+:[0-9]+", args.probe):
        parser.error("Require nonnegative physical NPU, positive timeout and a layer:expert probe.")
    if args.warmups < 3 or args.repeats < 10:
        parser.error("Timing requires --warmups >= 3 and --repeats >= 10.")
    if args.stage in ("expert", "timing") and not args.model:
        parser.error("Expert validation requires --model.")
    if args.stage and (args.output is None or args.output.exists()):
        parser.error("Child stage requires a new --output file.")
    if not args.stage and args.output:
        parser.error("Use --output-dir for the supervisor.")
    args.library = args.library.resolve()
    if args.model:
        args.model = args.model.resolve()
    if args.output_dir:
        args.output_dir = args.output_dir.resolve()
    return run_child(args) if args.stage else run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
