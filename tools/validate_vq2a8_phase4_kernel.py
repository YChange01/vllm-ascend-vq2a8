#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated phase-4 kernel experiments; no model or serving integration.

Use the phase-4 supervisor on Ascend: a compiled kernel can abort the entire
process. CPU dense weights exist only in the numerical oracle. Device dense
matrices in native/CV microtests are synthetic and bounded, not real experts.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from tools.validate_vq2a8_tp1_packed_kernel import (
    PREPARED_INPUT_ATOL,
    PREPARED_INPUT_RTOL,
    _comparison_summary,
    _initialize_device,
    _synchronize,
    activation_case,
    environment_report,
    parse_probes,
)
from vllm_ascend.quantization.vq2a8_reference import (
    decode_repacked_vq2a8_codebook_weight,
    deepseek_v4_swiglu_reference,
    prepare_repacked_vq2a8_activation_reference,
)
from vllm_ascend.quantization.vq2a8_runtime import open_vq2a8_tp1_artifact
from vllm_ascend.quantization.vq2a8_triton import vq2a8_tp1_m1_packed_gemm
from vllm_ascend.quantization.vq2a8_validation import tensor_layout
from vllm_ascend.quantization.vq2a8_vector_gather import vq2a8_packed_vector_gather


def synthetic_inputs(rows=3, size_n=64, size_k=512, tiles=3):
    """Exercise signed packed words, every nibble/component and scrambled tiles."""
    with torch.device("cpu"):
        codes = (torch.arange(size_k)[None, :] * 7 + torch.arange(size_n // 2)[:, None] * 3 + 3) % 16
        packed = torch.sum(codes.reshape(size_n // 2, size_k // 8, 8) << (torch.arange(8) * 4), -1).int()
        book = (torch.arange(tiles * (size_n // 32) * 32).reshape(tiles, size_n // 32, 16, 2) % 53 - 26) / 8
        book = book.to(torch.float8_e4m3fn)
        tile_ids = ((torch.arange(size_k) * 13 + torch.arange(size_k) // 17) % tiles).byte()
        x = ((torch.arange(rows * size_k).reshape(rows, size_k) * 11 % 37 - 18) / 8).to(torch.float8_e4m3fn)
        scale = (torch.arange(rows).float() + 1) / 64
        bias = (torch.arange(rows).float() - 2) / 32
    return x, scale, bias, packed, book, tile_ids


def synthetic_dense_oracle(packed, book, tile_ids):
    """CPU-only literal indexing; intentionally not the candidate gather path."""
    if any(t.device.type != "cpu" for t in (packed, book, tile_ids)):
        raise ValueError("Dense synthetic oracle is CPU-only.")
    size_n, size_k = packed.shape[0] * 2, packed.shape[1] * 8
    out = torch.empty((size_n, size_k), dtype=torch.float64)
    book = book.double()
    columns = torch.arange(size_k)
    for row in range(size_n):
        codes = (packed[row // 2, columns // 8].long() >> ((columns % 8) * 4)) & 15
        out[row] = book[tile_ids.long(), row // 32, codes, row % 2]
    return out


def same_fp8_oracle(prepared, dense):
    x, scale, bias = (t.cpu().double() for t in prepared)
    return (x @ dense.T * scale[:, None] + bias[:, None]).bfloat16()


def prepare_rows(hidden, payload, spec):
    """Keep the accepted per-row RHT/A8 arithmetic; only batch kernel launches."""
    rows = []
    for row in hidden.split(1):
        if spec.columns != spec.rht_true_columns:
            row = torch.nn.functional.pad(row, (0, spec.columns - spec.rht_true_columns))
        with torch.device("cpu"):
            rows.append(
                prepare_repacked_vq2a8_activation_reference(
                    row, payload["weight_scale"], payload["weight_bias"], payload["rht_sign"], spec.rht_block_size
                )
            )
    return tuple(torch.cat([r[i] for r in rows], dim=0).contiguous() for i in range(3))


def accepted_rows(inputs):
    x, scale, bias, *packed = inputs
    return torch.cat(
        [vq2a8_tp1_m1_packed_gemm(x[i : i + 1], scale[i : i + 1], bias[i : i + 1], *packed) for i in range(x.shape[0])]
    )


def compare(expected, actual):
    return _comparison_summary(expected, actual, rtol=PREPARED_INPUT_RTOL, atol=PREPARED_INPUT_ATOL)


def bitwise_equal(left, right):
    """Validation only: torch.equal alone does not distinguish signed zeros."""
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(
            left.detach().cpu().contiguous().view(torch.uint8),
            right.detach().cpu().contiguous().view(torch.uint8),
        )
    )


def benchmark(call, device, warmups, repeats):
    """Warm device-event and synchronized wall times; excludes JIT/load/prep.

    Event and wall statistics are distinct: wrapper allocation, validation and
    Python dispatch are included in wall time. No speedup claim from CUDA is
    an NPU claim. Caller compares only identically prepared resident tensors.
    """
    if warmups < 3 or repeats < 10:
        raise ValueError("Performance samples require >=3 warmups and >=10 repeats.")
    backend = getattr(torch, device.type)
    for _ in range(warmups):
        call()
    _synchronize(device)
    events, walls = [], []
    start_event, end_event = backend.Event(enable_timing=True), backend.Event(enable_timing=True)
    for _ in range(repeats):
        start = time.perf_counter()
        start_event.record()
        call()
        end_event.record()
        end_event.synchronize()
        walls.append((time.perf_counter() - start) * 1000)
        events.append(start_event.elapsed_time(end_event))
    if any(not (0 < value < float("inf")) for value in events + walls):
        raise AssertionError("Invalid device or wall timing; no performance claim is possible.")

    def stats(values):
        ordered = sorted(values)
        return {"min": ordered[0], "median": statistics.median(values), "p95": ordered[(95 * len(values) - 1) // 100]}

    return {"event_ms": stats(events), "wall_ms": stats(walls), "warmups": warmups, "repeats": repeats}


def check_projection(inputs, dense, device, *, warmups, repeats, timed=False):
    expected = same_fp8_oracle(inputs[:3], dense)
    _synchronize(device)
    backend = getattr(torch, device.type)
    before_bytes = backend.memory_allocated(device)
    backend.reset_peak_memory_stats(device)
    started = time.perf_counter()
    actual = vq2a8_packed_vector_gather(*inputs)
    _synchronize(device)
    first_call_ms = (time.perf_counter() - started) * 1000
    peak_bytes = backend.max_memory_allocated(device)
    reference_comparison = compare(expected, actual)
    baseline = accepted_rows(inputs)
    baseline_comparison = compare(baseline, actual)
    split = torch.cat(
        [
            vq2a8_packed_vector_gather(inputs[0][i : i + 1], inputs[1][i : i + 1], inputs[2][i : i + 1], *inputs[3:])
            for i in range(inputs[0].shape[0])
        ]
    )
    if not bitwise_equal(actual, split):
        raise AssertionError("Candidate batching changed the per-row result.")
    for _ in range(3):
        if not bitwise_equal(actual, vq2a8_packed_vector_gather(*inputs)):
            raise AssertionError("Candidate repeat is not bitwise deterministic.")
    result = {
        "candidate_backend": "experimental_packed_vector_gather_fp32",
        "baseline_backend": "accepted_a5_vector_v7" if device.type == "npu" else "portable_cuda_fp8_dot",
        "oracle": reference_comparison,
        "baseline": baseline_comparison,
        "baseline_exact": bitwise_equal(baseline, actual),
        "row_chunk_exact": True,
        "repeat_exact": True,
        "first_call_including_compile_ms": first_call_ms,
        "projection_launches": {"accepted": inputs[0].shape[0], "candidate": 1},
        "native_fp8_expert_dot": False,
        "dense_expert_weight_on_device": False,
        "resident_input_bytes": sum(t.numel() * t.element_size() for t in inputs),
        "candidate_peak_extra_allocated_bytes": max(0, peak_bytes - before_bytes),
        "input_layouts": {
            name: tensor_layout(t)
            for name, t in zip(("activation", "scale", "bias", "packed", "codebooks", "tile_ids"), inputs)
        },
    }
    if timed:
        # Interleave two rounds with reversed order to expose ordering/cache
        # bias instead of reporting only one candidate-after-baseline ratio.
        samples = {"accepted": [], "candidate": []}
        calls = {"accepted": lambda: accepted_rows(inputs), "candidate": lambda: vq2a8_packed_vector_gather(*inputs)}
        for order in (("accepted", "candidate"), ("candidate", "accepted")):
            for name in order:
                samples[name].append(benchmark(calls[name], device, warmups, repeats))
        result["timing"] = samples
        for clock in ("event_ms", "wall_ms"):
            old = statistics.median(s[clock]["median"] for s in samples["accepted"])
            new = statistics.median(s[clock]["median"] for s in samples["candidate"])
            result[f"{clock}_speedup"] = old / new
    return result, actual


def native_micro(device, mode):
    """Bounded synthetic FP8 Cube/CV experiments; never decode a real expert."""
    from vllm_ascend.quantization.vq2a8_root_fp8 import root_fp8_matmul_npu

    a = ((torch.arange(32 * 512).reshape(32, 512) % 31 - 15) / 8).to(torch.float8_e4m3fn)
    b = ((torch.arange(32 * 512).reshape(32, 512) % 29 - 14) / 8).to(torch.float8_e4m3fn)
    expected = (a.double() @ b.double().T * (-1 if mode == "cv_bridge" else 1)).bfloat16()
    da, db = a.to(device), b.to(device)
    if mode == "native" and device.type == "npu":

        def call():
            return root_fp8_matmul_npu(
                da, torch.ones((32, 1), device=device), db, torch.ones(1, device=device), "tensor"
            )
    else:
        from vllm_ascend.quantization.vq2a8_phase4_micro import fp8_cube_micro

        def call():
            return fp8_cube_micro(da, db, bridge=mode == "cv_bridge")

    actual = call()
    comparison = compare(expected, actual)
    for _ in range(3):
        if not bitwise_equal(actual, call()):
            raise AssertionError("FP8 micro repeat differs.")
    return {
        "comparison": comparison,
        "repeat_exact": True,
        "synthetic_only": True,
        "native_fp8_expert_dot": False,
        "micro_backend": "cann" if mode == "native" and device.type == "npu" else "triton",
    }


def run_real(args, device, emit):
    artifact = open_vq2a8_tp1_artifact(
        args.artifact or args.model / "experts_vq_ascend_v2",
        args.model / "config.json",
        require_complete=not args.allow_partial_artifact,
    )
    swiglu_limit = json.loads((args.model / "config.json").read_text()).get("swiglu_limit")
    for probe in parse_probes(args.probe):
        payloads, matrices = {}, {}
        for kind in ("gate_up", "down"):
            print(f"PHASE4 stage=load expert={probe.layer_index}:{probe.expert_id} projection={kind}", flush=True)
            host, spec = artifact.load_expert(probe.layer_index, probe.expert_id, kind)
            matrices[kind] = decode_repacked_vq2a8_codebook_weight(host, spec, compute_dtype=torch.float64)
            payloads[kind] = ({k: v.to(device) for k, v in host.items()}, spec)
        for rows in args.rows:
            for case in args.cases:
                payload, spec = payloads["gate_up"]
                # Rows are distinct except zero; never expand a single test row
                # and then claim that multi-token indexing was exercised.
                hidden = torch.cat([activation_case(spec.rht_true_columns, i, 0, case) for i in range(rows)]).to(device)
                baseline_hidden = hidden
                for kind in ("gate_up", "down"):
                    print(f"PHASE4 stage=projection probe={args.probe} kind={kind} m={rows} case={case}", flush=True)
                    payload, spec = payloads[kind]
                    _synchronize(device)
                    started = time.perf_counter()
                    prepared = prepare_rows(hidden, payload, spec)
                    _synchronize(device)
                    prepare_ms = (time.perf_counter() - started) * 1000
                    inputs = (*prepared, *(payload[k] for k in ("packed_indices", "codebooks", "codebook_tile_ids")))
                    result, actual = check_projection(
                        inputs,
                        matrices[kind],
                        device,
                        warmups=args.warmups,
                        repeats=args.repeats,
                        timed=args.stage == "benchmark",
                    )
                    # Also compare complete, independent candidate/accepted
                    # chains. A down projection using only the candidate's
                    # activation would otherwise hide upstream discrepancies.
                    baseline_prepared = prepare_rows(baseline_hidden, payload, spec)
                    baseline_actual = accepted_rows((*baseline_prepared, *inputs[3:]))
                    result["chain_baseline"] = _comparison_summary(
                        baseline_actual,
                        actual,
                        rtol=0.0 if case == "zero" else 0.03,
                        atol=0.0 if case == "zero" else 0.05,
                    )
                    emit(
                        {
                            "probe": f"{probe.layer_index}:{probe.expert_id}",
                            "projection": kind,
                            "rows": rows,
                            "case": case,
                            "prepare_ms": prepare_ms,
                            **result,
                        }
                    )
                    if kind == "gate_up":
                        # Down validation uses the candidate chain's real
                        # intermediate, with same-FP8 and accepted-row oracles.
                        hidden = deepseek_v4_swiglu_reference(actual, swiglu_limit)
                        baseline_hidden = deepseek_v4_swiglu_reference(baseline_actual, swiglu_limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["lookup", "native", "cube_direct", "cv_bridge", "packed", "benchmark"], required=True
    )
    parser.add_argument("--device", choices=["npu:0", "cuda:0"], default="npu:0")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument(
        "--allow-partial-artifact", action="store_true", help="Developer CUDA only; never NPU acceptance."
    )
    parser.add_argument("--probe", default="0:0")
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 3, 10, 32])
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["deterministic", "zero", "impulse", "small"],
        default=["deterministic", "zero", "impulse", "small"],
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.allow_partial_artifact and args.device != "cuda:0":
        parser.error("Partial artifacts are permitted only in developer CUDA experiments.")
    if args.output.exists() or len(set(args.rows)) != len(args.rows) or any(not 1 <= r <= 32 for r in args.rows):
        parser.error("Use a new output file and distinct row counts in [1,32].")
    if args.warmups < 3 or args.repeats < 10 or len(set(args.cases)) != len(args.cases):
        parser.error("Require >=3 warmups, >=10 repeats and distinct cases.")
    if args.stage in ("packed", "benchmark") and args.model is None:
        parser.error("Real packed experiments require --model.")
    device = torch.device(args.device)
    report = {
        "status": "running",
        "stage": args.stage,
        "device": str(device),
        "environment": environment_report(),
        "results": [],
        "native_fp8_expert_dot": False,
        "model_integration_verified": False,
        "npu_verified": False,
        "performance_verified": False,
        "requested": {"probe": args.probe, "rows": args.rows, "cases": args.cases},
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def emit(record):
        report["results"].append({"passed": True, **record})
        save()
        print("PHASE4_RESULT " + json.dumps(record, allow_nan=False), flush=True)

    save()
    print("ENVIRONMENT " + json.dumps(report["environment"]), flush=True)
    try:
        report["device_info"] = _initialize_device(device)
        print("DEVICE " + json.dumps(report["device_info"]), flush=True)
        print(f"PHASE4 stage={args.stage}", flush=True)
        if args.stage == "lookup":
            for tiles in (1, 3, 16, 32):
                for rows in (1, 3, 32):
                    print(f"PHASE4 stage=lookup_start m={rows} n=64 k=512 column_tiles={tiles}", flush=True)
                    inputs = synthetic_inputs(rows, tiles=tiles)
                    dense = synthetic_dense_oracle(*inputs[3:])
                    result, _ = check_projection(
                        tuple(t.to(device) for t in inputs), dense, device, warmups=args.warmups, repeats=args.repeats
                    )
                    emit({"rows": rows, "column_tiles": tiles, **result})
        elif args.stage in ("native", "cube_direct", "cv_bridge"):
            emit(native_micro(device, args.stage))
        else:
            run_real(args, device, emit)
        report["status"] = "passed"
        report["npu_verified"] = device.type == "npu"
        report["timing_measured_on_npu"] = device.type == "npu" and args.stage == "benchmark"
        save()
        print(
            f"PHASE4_KERNEL_GATE=PASS stage={args.stage} results={len(report['results'])} DEVICE={device}", flush=True
        )
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
