#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded real-model eager/decoder-graph correctness test, not a timing result.

Runs on one idle NPU in a new worker. Exercises 3->4, 7->8, 11->12 compressor
boundaries, changing token IDs and repeated request/cache-slot reuse. Both modes
use the *same* resident payload and selected arithmetic/preparation backend.
"""

from __future__ import annotations

# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path

from tools.diagnose_vq2a8_tp1_startup import child_environment, emit, run_child, stage_recorder

CASES = ((1, 4), (3, 4), (7, 4), (11, 4), (12, 4), (1, 15))
REUSE_ROUNDS = 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, help="expert artifact; defaults to MODEL/experts_vq_ascend_v2")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--physical-npu", type=int, default=1)
    parser.add_argument("--compute-backend", choices=("v1", "v2"), default="v2")
    parser.add_argument("--activation-reorder", choices=("scalar", "vectorized"), default="scalar")
    parser.add_argument(
        "--activation-preparation", choices=("rowwise", "rowwise_packed", "sign_fused", "fused"), default="rowwise"
    )
    parser.add_argument("--decoder-metadata-mode", choices=("recursive", "planned"), default="recursive")
    parser.add_argument(
        "--host-profile", action="store_true", help="CPU-only ranges/counters; never a timing benchmark"
    )
    parser.add_argument("--kv-cache-mib", type=int, default=256)
    parser.add_argument("--reserve-gib", type=float, default=3.0)
    parser.add_argument("--engine-memory-fraction", type=float, default=0.9)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--plan-only", action="store_true", help="Print cases/options without importing the NPU runtime"
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.physical_npu < 0 or not 1 <= args.timeout_s <= 7200 or args.kv_cache_mib <= 0:
        parser.error("Require nonnegative physical NPU, positive KV bytes and timeout in 1..7200 seconds.")
    if not math.isfinite(args.reserve_gib) or args.reserve_gib < max(1.0, args.kv_cache_mib / 1024):
        parser.error("Reserve must be finite and at least 1 GiB, covering the explicit KV cache.")
    if not math.isfinite(args.engine_memory_fraction) or not 0 < args.engine_memory_fraction <= 1:
        parser.error("Engine memory fraction must be in (0,1].")
    if args.compute_backend != "v2" and (
        args.activation_reorder != "scalar" or args.activation_preparation != "rowwise"
    ):
        parser.error("New activation variants require --compute-backend v2.")
    return args


def graph_switch(worker, enabled):
    return worker.get_model().set_v4_graph_enabled(enabled)


def graph_report(worker):
    return worker.get_model().v4_graph_report()


def compare_outputs(reference, candidate):
    if list(reference.token_ids) != list(candidate.token_ids):
        raise AssertionError("Decoder graph changed greedy tokens.")
    if reference.logprobs is None or candidate.logprobs is None:
        raise AssertionError("Missing device-computed log-probability evidence.")
    if len(reference.logprobs) != len(candidate.logprobs):
        raise AssertionError("Decoder graph changed output/logprob length.")
    maximum_error = 0.0
    for expected, actual in zip(reference.logprobs, candidate.logprobs):
        if expected.keys() != actual.keys():
            raise AssertionError("Decoder graph changed top-five token identities.")
        for token, old in expected.items():
            new = actual[token]
            if not math.isfinite(old.logprob) or not math.isfinite(new.logprob):
                raise AssertionError("Non-finite log probability.")
            maximum_error = max(maximum_error, abs(old.logprob - new.logprob))
            if not math.isclose(old.logprob, new.logprob, rel_tol=1e-5, abs_tol=1e-4):
                raise AssertionError("Decoder graph changed top-five log probabilities beyond 1e-4/1e-5.")
    return maximum_error


def run_model(args):
    import torch
    import torch_npu  # noqa: F401
    from tokenizers import Tokenizer
    from vllm import LLM, SamplingParams

    from tools.benchmark_vq2a8_v4 import require_idle_device
    from tools.validate_vq2a8_tp1_offline import single_worker_result
    from vllm_ascend.quantization.vq2a8_offline import offline_engine_options

    output = args.output_dir
    require_idle_device(args.physical_npu, output / "npu-before.log")
    library = args.library.resolve(strict=True)
    model = args.model.resolve(strict=True)
    artifact = (args.artifact if args.artifact is not None else model / "experts_vq_ascend_v2").resolve(strict=True)
    stage = stage_recorder("v4_decoder_graph", torch.npu.synchronize)
    options = offline_engine_options(
        model,
        artifact,
        execution_policy="ascendc_v4",
        ascendc_library=str(library),
        ascendc_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        cache_reserve_gib=args.reserve_gib,
        cache_memory_fraction=1.0,
        v4_serving=True,
        v4_device_route_decode=True,
        v4_compute_backend=args.compute_backend,
        v4_activation_reorder=args.activation_reorder,
        v4_activation_preparation=args.activation_preparation,
        v4_decode_graph="decoder",
        v4_graph_replay_stream="caller",
        v4_decoder_metadata_mode=args.decoder_metadata_mode,
        v4_host_profile=args.host_profile,
    )
    options.update(
        max_model_len=16,
        max_num_batched_tokens=16,
        kv_cache_memory_bytes=args.kv_cache_mib * 1024**2,
        gpu_memory_utilization=args.engine_memory_fraction,
    )
    with stage("load_and_capture_all_positions"):
        llm = LLM(**options)
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    seeds = [
        tokenizer.encode(text, add_special_tokens=False).ids
        for text in ("Hello, please continue this short example.", "你好，请计算简单加法。")
    ]
    if any(not seed for seed in seeds):
        raise ValueError("Validation tokenizer produced an empty seed.")
    rows = []
    for round_id in range(REUSE_ROUNDS):
        for prompt_length, output_length in CASES:
            seed = seeds[round_id]
            prompt = (seed * (prompt_length // len(seed) + 1))[:prompt_length]
            params = SamplingParams(
                temperature=0, max_tokens=output_length, ignore_eos=True, detokenize=False, logprobs=5
            )
            with stage(f"round{round_id}_p{prompt_length}_o{output_length}_eager"):
                single_worker_result(llm.collective_rpc(graph_switch, args=(False,)))
                reference = llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)[0].outputs[0]
            with stage(f"round{round_id}_p{prompt_length}_o{output_length}_decoder"):
                before = single_worker_result(llm.collective_rpc(graph_switch, args=(True,)))
                candidate = llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)[0].outputs[0]
                after = single_worker_result(llm.collective_rpc(graph_report))
                delta = after["decoder"]["replays"] - before["decoder"]["replays"]
                if delta != output_length - 1 or len(candidate.token_ids) != output_length:
                    raise AssertionError("The requested output count or actual decoder replay count differs.")
                error = compare_outputs(reference, candidate)
                rows.append(
                    {"round": round_id, "prompt": prompt_length, "output": output_length, "max_logprob_error": error}
                )
    receipt = {
        "status": "PASS",
        "scope": "real_model_eager_vs_position_specialized_decoder",
        "hardware_execution_verified": True,
        "timing_valid": False,
        "cases": rows,
        "graph": single_worker_result(llm.collective_rpc(graph_report)),
        "library_sha256": options["additional_config"]["vq2a8_offline"]["ascendc_sha256"],
    }
    (output / "summary.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("V4_DECODER_GRAPH=PASS SUMMARY=" + str(output / "summary.json"), flush=True)
    emit("v4_decoder_graph", "CASE_PASS", scope=receipt["scope"])


def main(argv=None):
    args = parse_args(argv)
    if args.plan_only:
        print(
            json.dumps(
                {
                    "cases": CASES,
                    "reuse_rounds": REUSE_ROUNDS,
                    "max_model_len": 16,
                    "physical_npu": args.physical_npu,
                    "compute_backend": args.compute_backend,
                    "artifact": str(
                        args.artifact if args.artifact is not None else args.model / "experts_vq_ascend_v2"
                    ),
                    "activation_reorder": args.activation_reorder,
                    "activation_preparation": args.activation_preparation,
                    "decoder_metadata_mode": args.decoder_metadata_mode,
                    "host_profile": args.host_profile,
                    "device_execution": False,
                    "startup_capture_positions": list(range(16)),
                }
            )
        )
        return 0
    if args.child:
        run_model(args)
        return 0
    output = args.output_dir or Path(tempfile.mkdtemp(prefix="vq2-v4-decoder-"))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory; existing profiler/validation data is never overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    command += ["--child", "--output-dir", str(output.resolve())]
    args.launch_blocking = "0"
    environment = child_environment(args)
    environment["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    environment["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    result = run_child(command, environment, output / "validation.log", args.timeout_s)
    (output / "supervisor.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["status"] != "PASS":
        print(f"V4_DECODER_GRAPH={result['status']} LOG=" + str(output / "validation.log"), flush=True)
        return 1
    receipt = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    expected_replays = REUSE_ROUNDS * sum(count - 1 for _, count in CASES)
    if (
        receipt.get("status") != "PASS"
        or receipt.get("hardware_execution_verified") is not True
        or len(receipt.get("cases", [])) != REUSE_ROUNDS * len(CASES)
        or receipt.get("graph", {}).get("decoder", {}).get("replays") != expected_replays
    ):
        raise ValueError("Device probe receipt does not confirm all requested real decoder replays.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
