#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quick vLLM-engine TPOT estimate, NOT a numerical or release acceptance gate.

One model load, one warmup and three measured requests by default. No build,
operator preflight, reference model, logits dump, profiler or report directory.
"""

from __future__ import annotations

# Keep tools/bisect from shadowing the standard library during direct execution.
# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import math
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAX_CONTEXT = 128
MODEL_LAYERS = 43
STEP_SLACK = 4
MILLISECONDS = 1000


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=REPO / "build/vq2a8-ascendc-v3/libvq2a8_ascendc_v3.so")
    parser.add_argument("--physical-npu", type=int, default=0)
    parser.add_argument("--prompt-tokens", type=int, default=10)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--engine-memory-fraction", type=float, default=0.98)
    parser.add_argument("--cache-memory-fraction", type=float, default=1.0)
    parser.add_argument(
        "--cache-reserve-gib",
        type=float,
        default=3.0,
        help="Includes fixed 1 GiB KV; never automatically reduced if residency cannot fit",
    )
    args = parser.parse_args(argv)
    if args.prompt_tokens < 1 or args.output_tokens < 2 or args.prompt_tokens + args.output_tokens > MAX_CONTEXT:
        parser.error("Require prompt >=1, output >=2 and prompt + output <=128 tokens")
    if args.physical_npu < 0 or args.warmups < 0 or args.repeats < 1:
        parser.error("Require NPU >=0, warmups >=0, repeats >=1")
    if any(not math.isfinite(v) or not 0 < v <= 1 for v in (args.engine_memory_fraction, args.cache_memory_fraction)):
        parser.error("Memory fractions must be finite and in (0,1]")
    if not math.isfinite(args.cache_reserve_gib) or args.cache_reserve_gib < 1:
        parser.error("Cache reserve must be finite and >=1 GiB (includes KV)")
    return args


def build_options(args, library, options_factory):
    options = options_factory(
        args.model,
        args.model / "experts_vq_ascend_v2",
        execution_policy="ascendc_v3",
        root_linear_mode="bf16",
        ascendc_v3_library=library["path"],
        ascendc_v3_sha256=library["sha256"],
        cache_budget_gib=0.0,
        cache_reserve_gib=args.cache_reserve_gib,
        cache_memory_fraction=args.cache_memory_fraction,
    )
    options.update(
        max_model_len=args.prompt_tokens + args.output_tokens,
        max_num_batched_tokens=args.prompt_tokens + args.output_tokens,
        gpu_memory_utilization=args.engine_memory_fraction,
    )
    return options


def measure_request(engine, prompt, params, request_id, output_tokens, *, synchronize, clock=None):
    """Host token-arrival timing through the real vLLM scheduler and full model.

    TPOT uses first-to-last token / (N-1), excluding prefill and final cleanup.
    No per-token NPU event, explicit synchronization, tensor read or console I/O.
    """
    if output_tokens < 2 or engine.has_unfinished_requests():
        raise ValueError("Require >=2 output tokens and an idle single-request engine")
    clock = time.perf_counter if clock is None else clock
    synchronize()
    started = clock()
    engine.add_request(request_id, {"prompt_token_ids": prompt}, params)
    ready, tokens, finished = [], [], False
    for _ in range(output_tokens + STEP_SLACK):
        results = engine.step()
        if len(results) > 1:
            raise ValueError("Expected at most one request result per engine step")
        for result in results:
            if result.request_id != request_id or len(result.outputs) != 1 or finished:
                raise ValueError("Unexpected request/output in the single-request benchmark")
            current = list(result.outputs[0].token_ids)
            if current[: len(tokens)] != tokens or len(current) > len(tokens) + 1 or len(current) > output_tokens:
                raise ValueError("Expected cumulative outputs, one new token at a time")
            if len(current) > len(tokens):
                ready.append(clock() - started)
            tokens, finished = current, result.finished
        if finished:
            break
    synchronize()
    elapsed = clock() - started
    if not finished or len(ready) != output_tokens or engine.has_unfinished_requests():
        raise ValueError("Incomplete generation; no TPOT estimate reported")
    if (
        any(not math.isfinite(t) or t < 0 for t in [*ready, elapsed])
        or any(b <= a for a, b in zip(ready, ready[1:]))
        or elapsed < ready[-1]
    ):
        raise ValueError("Invalid token timestamps")
    return {
        "ttft_ms": ready[0] * MILLISECONDS,
        "tpot_ms": (ready[-1] - ready[0]) * MILLISECONDS / (output_tokens - 1),
        "e2e_s": elapsed,
        "output_tokens": len(tokens),
    }


def run_samples(engine, prompt, params, args, synchronize):
    samples = []
    for kind, count in (("warmup", args.warmups), ("measured", args.repeats)):
        for index in range(count):
            print(f"QUICK_V3_START={kind} REPEAT={index + 1}/{count} OUTPUT_TOKENS={args.output_tokens}", flush=True)
            sample = measure_request(
                engine, prompt, params, f"quick-v3-{kind}-{index}", args.output_tokens, synchronize=synchronize
            )
            print(
                f"QUICK_V3_SAMPLE={kind} REPEAT={index + 1}/{count} "
                f"TTFT_MS={sample['ttft_ms']:.3f} TPOT_MS={sample['tpot_ms']:.3f} "
                f"E2E_S={sample['e2e_s']:.3f}",
                flush=True,
            )
            if kind == "measured":
                samples.append(sample)
    return samples


def run(args):
    # Select the single physical device BEFORE any torch/vLLM initialization.
    from tools.validate_vq2a8_ascendc import require_hardware_runtime
    from tools.validate_vq2a8_ascendc_v3 import library_identity
    from tools.validate_vq2a8_tp1_acceptance import acceptance_environment
    from tools.validate_vq2a8_v026_environment import require_v026_stack

    if os.environ.get("ASCEND_LAUNCH_BLOCKING") not in (None, "0"):
        raise ValueError("Unset ASCEND_LAUNCH_BLOCKING before timing")
    environment = acceptance_environment(REPO, args.physical_npu, "npu:0")
    environment.update(ASCEND_LAUNCH_BLOCKING="0", VLLM_ENABLE_V1_MULTIPROCESSING="0")
    os.environ.clear()
    os.environ.update(environment)
    print(
        "QUICK_V3_STAGE=setup ENGINE=vllm POLICY=ascendc_v3 SCOPE=TP1_B1_EAGER ACCEPTANCE=NOT_RUN QUALITY=NOT_VERIFIED",
        flush=True,
    )
    require_hardware_runtime()
    require_v026_stack()
    library = library_identity(args.library)
    args.model = args.model.resolve(strict=True)
    print(f"QUICK_V3_LIBRARY={json.dumps(library)}", flush=True)

    import torch
    import torch_npu  # noqa: F401
    from tokenizers import Tokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    from tools.benchmark_vq2a8_offline import configure, snapshot
    from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device
    from vllm_ascend.quantization.vq2a8_offline import offline_engine_options

    device = _initialize_device(torch.device("npu:0"))
    manifest = json.loads((Path(library["path"]).parent / "build-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("soc") != device["name"]:
        raise ValueError("V3 library was built for a different SoC; rebuild for this exact device")
    require_hardware_runtime()
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    if config.get("num_hidden_layers") != MODEL_LAYERS:
        raise ValueError("Expected the 43-layer VQ2A8 model")
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    seed = tokenizer.encode("The answer to 1 + 1 is. Read this short example. ", add_special_tokens=False).ids
    if not seed:
        raise ValueError("Empty prompt seed")
    prefix = [config["bos_token_id"]] if type(config.get("bos_token_id")) is int else []
    prompt = (prefix + seed * (args.prompt_tokens // len(seed) + 1))[: args.prompt_tokens]
    options = build_options(args, library, offline_engine_options)
    print(
        f"QUICK_V3_STAGE=model_load PHYSICAL_NPU={args.physical_npu} "
        f"ENGINE_MEMORY_FRACTION={args.engine_memory_fraction} CACHE_MEMORY_FRACTION={args.cache_memory_fraction} "
        f"CACHE_RESERVE_GIB={args.cache_reserve_gib}",
        flush=True,
    )
    started = time.perf_counter()
    llm = LLM(**options)
    print(f"QUICK_V3_ENGINE_READY STARTUP_S={time.perf_counter() - started:.3f}", flush=True)
    configure(llm, measurement=True, compact=False, optimization="batched")
    params = SamplingParams(
        temperature=0,
        max_tokens=args.output_tokens,
        ignore_eos=True,
        detokenize=False,
        output_kind=RequestOutputKind.CUMULATIVE,
    )
    samples = run_samples(llm.llm_engine, prompt, params, args, torch.npu.synchronize)
    state = snapshot(llm)  # Once, outside timing: finite flags and immutable resident-bank integrity.
    layers = state.get("v3", {})
    expected_decode_calls = (args.warmups + args.repeats) * (args.output_tokens - 1)
    if (
        state.get("finite") is not True
        or set(layers) != {str(i) for i in range(MODEL_LAYERS)}
        or not all(
            layer.get("ready") is True and layer.get("decode_calls", 0) >= expected_decode_calls
            for layer in layers.values()
        )
    ):
        raise ValueError("Non-finite output or incomplete V3 resident decode; discard timing samples")
    tpots = [s["tpot_ms"] for s in samples]
    print(
        f"QUICK_V3_DONE SAMPLES={len(samples)} PROMPT_TOKENS={len(prompt)} OUTPUT_TOKENS={args.output_tokens} "
        f"TTFT_MEDIAN_MS={statistics.median(s['ttft_ms'] for s in samples):.3f} "
        f"TPOT_MEDIAN_MS={statistics.median(tpots):.3f} TPOT_MIN_MS={min(tpots):.3f} "
        f"TPOT_MAX_MS={max(tpots):.3f} METHOD=HOST_TOKEN_ARRIVAL ACCEPTANCE=NOT_RUN",
        flush=True,
    )


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"QUICK_V3_ERROR={exc} TIMING_VALID=False", flush=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
