#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-process paired offline measurement; supervised by accept_vq2a8_release.py.

No HTTP service, no quality claim, no changed floating-point reduction geometry.
The baseline and compact variants use the SAME native library. This measures
Python preparation changes, not a comparison against another kernel binary.
"""

from __future__ import annotations

# ruff: noqa: E402
import os as _bootstrap_os
import sys as _bootstrap_sys

if not __package__:
    _bootstrap_sys.path[0] = _bootstrap_os.path.dirname(
        _bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))
    )

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from tools.profile_vq2a8_ascendc import digest, write_json
from tools.validate_vq2a8_tp1_offline import capture_worker_trace, reset_worker_trace, single_worker_result
from tools.vq2a8_perf_report import MAX_CONTEXT, summarize_performance, token_metrics, validate_cases

EXPECTED_MODEL_LAYERS = 43


def preparation_preflight():
    """Small on-device bit-exact check before spending time loading the model."""
    import torch

    from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation

    baseline, candidate = RowwiseVQ2A8Preparation(), RowwiseVQ2A8Preparation(compact=True)
    rows = []
    for count in (1, 2, 17, 32):
        for true_width in (480, 512):
            with torch.device("cpu"):
                generator = torch.Generator().manual_seed(193 + count + true_width)
                hidden = torch.randn(count, true_width, generator=generator).bfloat16().to("npu:0")
                payload = {
                    "weight_scale": torch.randn(512, generator=generator).to("npu:0"),
                    "weight_bias": torch.randn(512, generator=generator).to("npu:0"),
                    "rht_sign": torch.where(torch.arange(512) % 2 == 0, -1, 1).to(torch.int8).to("npu:0"),
                }
            spec = SimpleNamespace(columns=512, rht_true_columns=true_width, rht_block_size=128)
            requests = [(hidden, payload, spec), (hidden[:1], payload, spec)]
            expected, actual = baseline.many(requests), candidate.many(requests)
            exact = all(
                torch.equal(a.view(torch.uint8).cpu(), b.view(torch.uint8).cpu())
                for left, right in zip(expected, actual)
                for a, b in zip(left, right)
            )
            if not exact:
                raise ValueError(f"Compact activation preparation differs at M={count}, K={true_width}.")
            rows.append({"rows": [count, 1], "true_width": true_width, "bit_exact": exact})
    return rows


def configure_worker(worker, measurement, compact):
    model = worker.get_model()
    return {"pid": os.getpid(), **model.configure_performance_probe(measurement=measurement, compact=compact)}


def snapshot_worker(worker):
    return worker.get_model().performance_snapshot()


def snapshot(llm):
    return single_worker_result(llm.collective_rpc(snapshot_worker))


def configure(llm, *, measurement, compact):
    result = single_worker_result(llm.collective_rpc(configure_worker, args=(measurement, compact)))
    if result["pid"] != os.getpid():
        raise RuntimeError("Benchmark worker must remain inside the supervised process.")


def memory_observation():
    # Read outside timing. Current RSS plus kernel-maintained process RSS peak.
    result = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value, _unit = line.split()
            result[key.rstrip(":") + "_bytes"] = int(value) * 1024
    return result


def timed_request(llm, prompt, output_tokens, request_id):
    import torch
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    before = snapshot(llm)
    params = SamplingParams(
        temperature=0,
        max_tokens=output_tokens,
        ignore_eos=True,
        detokenize=False,
        output_kind=RequestOutputKind.CUMULATIVE,
    )
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    torch.npu.synchronize()
    started = time.perf_counter()
    begin.record()
    engine = llm.llm_engine
    engine.add_request(request_id, {"prompt_token_ids": prompt}, params)
    ready, tokens, finished = [], [], False
    for _ in range(output_tokens + 4):
        for result in engine.step():
            if result.request_id != request_id or len(result.outputs) != 1:
                raise ValueError("Unexpected request multiplexing in single-request offline benchmark.")
            current = list(result.outputs[0].token_ids)
            if current[: len(tokens)] != tokens or len(current) > len(tokens) + 1:
                raise ValueError("Expected ordered single-token output, without speculative batching.")
            ready.extend([time.perf_counter() - started] * (len(current) - len(tokens)))
            tokens, finished = current, result.finished
        if finished:
            break
    end.record()
    torch.npu.synchronize()
    elapsed = time.perf_counter() - started
    if not finished or len(tokens) != output_tokens or engine.has_unfinished_requests():
        raise ValueError("Offline request did not finish with the declared token count.")
    after = snapshot(llm)
    return {
        **token_metrics(ready, elapsed, output_tokens),
        "tokens": tokens,
        "token_ready_s": ready,
        "device_span_ms": begin.elapsed_time(end),
        "device_span_scope": "NPU event interval including host submission gaps; not summed kernel time",
        "finite": after["finite"],
        "forwards": after["forwards"],
        "cache_delta": {k: after["cache"][k] - before["cache"][k] for k in ("loads", "hits", "evictions")},
        "native_calls": after["native_calls"] - before["native_calls"],
        "native_launches": after["native_launches"] - before["native_launches"],
        "expert_payload_h2d_bytes": after["h2d_bytes"] - before["h2d_bytes"],
        "host_observed_timing": {
            k: after["host_observed_timing"][k] - v for k, v in before["host_observed_timing"].items()
        },
        "host_timing_scope": after["timing_scope"],
        "memory": {k: v for k, v in after.items() if "bytes" in k},
        "host_memory": memory_observation(),
        "resident_packed_bytes": after["cache"]["resident_packed_bytes"],
    }


def diagnostic(llm, prompt, output_tokens, compact, target):
    from safetensors.torch import save_file
    from vllm import SamplingParams

    configure(llm, measurement=False, compact=compact)
    single_worker_result(llm.collective_rpc(reset_worker_trace))
    result = llm.generate(
        [{"prompt_token_ids": prompt}],
        SamplingParams(temperature=0, max_tokens=output_tokens, ignore_eos=True, detokenize=False),
        use_tqdm=False,
    )
    evidence = single_worker_result(llm.collective_rpc(capture_worker_trace))
    if len(result) != 1 or len(result[0].outputs) != 1:
        raise ValueError("Expected exactly one completed offline request.")
    tokens = list(result[0].outputs[0].token_ids)
    logits = evidence.pop("logits")
    if len(result) != 1 or not result[0].finished or len(tokens) != output_tokens or logits.shape[0] != output_tokens:
        raise ValueError("Incomplete diagnostic request.")
    calls = evidence["cache"]["layer_calls"]
    backend = evidence["expert_backend"]
    if (
        len(calls) != EXPECTED_MODEL_LAYERS
        or set(calls.values()) != {output_tokens}
        or backend.get("policy") != "ascendc"
        or backend.get("fallback_enabled") is not False
        or len(backend.get("layers", [])) != EXPECTED_MODEL_LAYERS
    ):
        raise ValueError("Diagnostic must execute every layer on the no-fallback AscendC path.")
    if any(
        len(layer["steps"]) != output_tokens or any(step["projection_calls"] <= 0 for step in layer["steps"])
        for layer in backend["layers"]
    ):
        raise ValueError("Missing native projection call coverage.")
    save_file({"logits": logits.contiguous()}, str(target))
    write_json(target.with_suffix(".json"), {"tokens": tokens, "evidence": evidence, "logits_sha256": digest(target)})
    return tokens, logits


def run(args):
    if os.environ.get("ASCEND_LAUNCH_BLOCKING") not in (None, "0"):
        raise ValueError("Measurement requires ASCEND_LAUNCH_BLOCKING unset or 0, not diagnostic blocking.")
    from tools.validate_vq2a8_ascendc import checked_model_preflight, require_hardware_runtime
    from tools.validate_vq2a8_v026_environment import check_scheduler_apis, require_v026_stack

    require_v026_stack()
    require_hardware_runtime()
    library = checked_model_preflight(args.library, args.preflight)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    started = time.perf_counter()
    import torch
    import torch_npu  # noqa: F401
    from tokenizers import Tokenizer
    from vllm import LLM

    from tools.validate_vq2a8_qli_metadata import run_preflight
    from tools.validate_vq2a8_sas_attention import run_sas_preflight
    from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device, environment_report
    from vllm_ascend.quantization.vq2a8_offline import offline_engine_options

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "RUNNING",
        "samples": [],
        "regressions": [],
        "library": library,
        "cases": args.cases,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "environment": environment_report(),
        "scheduler": check_scheduler_apis(),
        "scope": "bounded_offline_not_serving",
        "performance_target_met": None,
    }
    path = output / "summary.json"
    write_json(path, manifest)
    try:
        manifest["device"] = _initialize_device(torch.device("npu:0"))
        require_hardware_runtime()
        manifest["preparation_preflight"] = preparation_preflight()
        root = args.model.resolve(strict=True)
        config = json.loads((root / "config.json").read_text())
        if config["num_hidden_layers"] != EXPECTED_MODEL_LAYERS:
            raise ValueError("This benchmark is scoped to the existing 43-layer VQ2A8 model.")
        tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        seed = tokenizer.encode(
            "The answer to 1 + 1 is. Read the following short example. ", add_special_tokens=False
        ).ids
        if not seed:
            raise ValueError("Empty tokenized benchmark seed.")
        prompts = {}
        for length, _ in args.cases:
            prefix = [config["bos_token_id"]] if isinstance(config.get("bos_token_id"), int) else []
            prompts[length] = prefix + (seed * (length // len(seed) + 1))[: length - len(prefix)]
        manifest["prompts"] = prompts
        run_preflight(torch.device("npu:0"), config, prompt_tokens=10)
        run_sas_preflight(torch.device("npu:0"), config, prompt_tokens=10)
        options = offline_engine_options(
            root,
            root / "experts_vq_ascend_v2",
            execution_policy="ascendc",
            cache_budget_gib=args.cache_budget_gib,
            cache_reserve_gib=args.cache_reserve_gib,
            ascendc_library=library["path"],
            ascendc_sha256=library["sha256"],
        )
        # Within the EXISTING strict offline validator; do not unlock serving.
        options.update(max_model_len=MAX_CONTEXT, max_num_batched_tokens=MAX_CONTEXT)
        manifest["engine_options"] = options
        engine_started = time.perf_counter()
        llm = LLM(**options)
        manifest["startup_s"] = time.perf_counter() - started
        manifest["engine_init_profile_kv_s"] = time.perf_counter() - engine_started
        print(f"PERF_ENGINE_READY startup_s={manifest['startup_s']:.3f}", flush=True)
        write_json(path, manifest)
        for length, count in args.cases:
            case = f"p{length}-o{count}"
            print(f"PERF_PHASE case={case} stage=first_request_and_exact_regression", flush=True)
            prompt = prompts[length]
            configure(llm, measurement=True, compact=False)
            first = timed_request(llm, prompt, count, f"{case}-first")
            first.update(case=case, kind="first_request_after_prior_cases_and_engine_profile", variant="baseline")
            manifest["samples"].append(first)
            write_json(path, manifest)
            if first["finite"] is not True or first["forwards"] != count:
                raise ValueError("First request did not complete finite real forwards.")
            left_tokens, left = diagnostic(llm, prompt, count, False, output / f"{case}-baseline.safetensors")
            right_tokens, right = diagnostic(llm, prompt, count, True, output / f"{case}-compact.safetensors")
            regression = {
                "case": case,
                "tokens_exact": left_tokens == right_tokens,
                "logits_exact": torch.equal(left, right),
                "max_abs_error": (left - right).abs().max().item(),
                "scope": "same_library_diagnostic_baseline_vs_compact_not_independent_reference",
            }
            manifest["regressions"].append(regression)
            write_json(path, manifest)
            if not regression["tokens_exact"] or not regression["logits_exact"] or first["tokens"] != left_tokens:
                raise ValueError("Preparation/measurement regression differs; stop timing, retain evidence.")
            for kind, repeats in (("warmup", args.warmups), ("measured", args.repeats)):
                for index in range(repeats):
                    # AB/BA order reduces fixed-order thermal/cache bias.
                    for compact in (False, True) if index % 2 == 0 else (True, False):
                        variant = "compact" if compact else "baseline"
                        configure(llm, measurement=True, compact=compact)
                        sample = timed_request(llm, prompt, count, f"{case}-{kind}-{index}-{variant}")
                        sample.update(
                            case=case,
                            variant=variant,
                            kind=kind,
                            repeat=index,
                            tokens_exact=sample["tokens"] == left_tokens,
                        )
                        manifest["samples"].append(sample)
                        write_json(path, manifest)
                        print(
                            "PERF_SAMPLE "
                            + json.dumps(
                                {
                                    k: sample[k]
                                    for k in (
                                        "case",
                                        "variant",
                                        "kind",
                                        "repeat",
                                        "ttft_s",
                                        "tpot_s",
                                        "e2e_s",
                                        "cache_delta",
                                    )
                                }
                            ),
                            flush=True,
                        )
                        if not sample["tokens_exact"] or sample["finite"] is not True:
                            raise ValueError("Measurement numerical regression failed.")
        manifest.update(
            summarize_performance(manifest["samples"], args.cases, args.repeats, args.warmups, manifest["regressions"])
        )
        manifest["library_unchanged"] = digest(args.library) == library["sha256"]
        if not manifest["library_unchanged"]:
            raise ValueError("Library changed during benchmark.")
    except Exception as exc:
        manifest.update(status="FAIL", error=str(exc), performance_measurement_verified=False)
        raise
    finally:
        write_json(path, manifest)
        print(f"PERFORMANCE={manifest['status']} REPORT={path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cache-budget-gib", type=float, default=0.0)
    parser.add_argument("--cache-reserve-gib", type=float, default=16.0)
    parser.add_argument("--cases", default="10:4,32:32,96:32")
    args = parser.parse_args()
    args.cases = validate_cases([tuple(map(int, item.split(":"))) for item in args.cases.split(",")])
    if args.warmups < 2 or args.repeats < 5:
        parser.error("Require >=2 warmups and >=5 repeats.")
    run(args)


if __name__ == "__main__":
    main()
