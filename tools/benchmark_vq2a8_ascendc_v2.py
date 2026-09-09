#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preflight-gated VQ2A8 v2 TP1 wall-clock benchmark, not an HTTP benchmark."""

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
import math
import os
import time
import traceback
from pathlib import Path

from tools.benchmark_vq2a8_offline import configure, snapshot, timed_request
from tools.profile_vq2a8_ascendc import digest, write_json
from tools.validate_vq2a8_ascendc_v2 import checked_model_preflight, model_identity, python_source_hashes
from tools.validate_vq2a8_tp1_offline import capture_worker_trace, reset_worker_trace, single_worker_result
from tools.vq2a8_perf_report import MAX_CONTEXT, distribution, token_metrics, validate_cases

LAYERS = 43
SCOPE = "TP1_OFFLINE_SINGLE_REQUEST_CONTEXT_LE_128"
KINDS = ("first_timed_after_diagnostics", "warmup", "measured")


def require_measurement_environment(environ=None):
    environ = os.environ if environ is None else environ
    if environ.get("ASCEND_LAUNCH_BLOCKING") not in (None, "0"):
        raise ValueError("Measurement requires ASCEND_LAUNCH_BLOCKING unset or 0.")
    visible = environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if not visible.isdecimal():
        raise ValueError("Select exactly one physical NPU with ASCEND_RT_VISIBLE_DEVICES.")
    return {"ASCEND_LAUNCH_BLOCKING": environ.get("ASCEND_LAUNCH_BLOCKING"), "physical_npu": visible}


def source_hashes():
    return {
        **python_source_hashes(),
        "tools/benchmark_vq2a8_ascendc_v2.py": digest(Path(__file__)),
        "tools/benchmark_vq2a8_offline.py": digest(Path(__file__).with_name("benchmark_vq2a8_offline.py")),
        "tools/vq2a8_perf_report.py": digest(Path(__file__).with_name("vq2a8_perf_report.py")),
    }


def validate_diagnostic(evidence, logits, prompt, tokens, vocab, library):
    """Recheck real generated-step coverage without the older gate's four-token limit."""
    import torch

    count = len(tokens)
    validate_cases([(len(prompt), count)])
    if any(type(t) is not int or not 0 <= t < vocab for t in prompt + tokens):
        raise ValueError("Invalid diagnostic tokens.")
    expected = [{"tokens": len(prompt), "positions": list(range(len(prompt)))}]
    expected += [{"tokens": 1, "positions": [len(prompt) + i]} for i in range(count - 1)]
    if evidence.get("steps") != expected:
        raise ValueError("Diagnostic must contain real prefill and every decode position.")
    backend = evidence.get("expert_backend", {})
    if (
        backend.get("policy") != "ascendc_v2"
        or backend.get("fallback_enabled") is not False
        or backend.get("library", {}).get("sha256") != library["sha256"]
        or Path(backend.get("library", {}).get("path", "")).resolve() != Path(library["path"]).resolve()
        or evidence.get("root_fp8", {}).get("mode") != "bf16"
    ):
        raise ValueError("Diagnostic must execute the pinned v2 library and BF16 roots without fallback.")
    records = backend.get("layers", [])
    if len(records) != LAYERS or {r.get("layer") for r in records} != set(range(LAYERS)):
        raise ValueError("Missing v2 layer coverage.")
    for record in records:
        steps = record.get("steps", [])
        if len(steps) != count:
            raise ValueError("Missing v2 per-step execution coverage.")
        for step, expected_step in zip(steps, expected):
            keys = ("tokens", "projection_calls", "projection_rows", "expert_calls", "kernel_launches")
            if (
                any(type(step.get(k)) is not int for k in keys)
                or step["tokens"] != expected_step["tokens"]
                or step["expert_calls"] < 1
                or step["projection_calls"] != 2 * step["expert_calls"]
                or step["projection_rows"] < 2 * step["tokens"]
                or step["projection_rows"] % 2
                or not 2 <= step["kernel_launches"] <= step["projection_calls"]
                or step["projection_calls"] > 6 * step["kernel_launches"]
                or step["kernel_launches"] % 2
            ):
                raise ValueError("Invalid v2 native projection/launch coverage.")
    cache = evidence.get("cache", {})
    if {int(k): v for k, v in cache.get("layer_calls", {}).items()} != {i: count for i in range(LAYERS)}:
        raise ValueError("Every diagnostic step must execute all 43 MoE layers.")
    load = evidence.get("load", {})
    if load.get("moe_layers") != LAYERS or load.get("registered_parameters_loaded", 0) <= 0:
        raise ValueError("Missing strict model loading evidence.")
    plan = cache.get("plan")
    if plan and cache["resident_packed_bytes"] > plan["planned_bytes"]:
        raise ValueError("Diagnostic cache exceeds its planned byte budget.")
    if (
        logits.device.type != "cpu"
        or logits.dtype != torch.float32
        or logits.shape != (count, vocab)
        or not bool(torch.isfinite(logits).all())
        or not torch.equal(logits[torch.arange(count), torch.tensor(tokens)], logits.max(-1).values)
    ):
        raise ValueError("Invalid finite FP32 diagnostic logits or greedy agreement.")


def diagnostic(llm, prompt, count, preset, target, vocab, library):
    from safetensors.torch import save_file
    from vllm import SamplingParams

    configure(llm, measurement=False, compact=False, optimization=preset)
    if single_worker_result(llm.collective_rpc(reset_worker_trace)) != os.getpid():
        raise ValueError("Diagnostic worker escaped the supervised process.")
    started = time.perf_counter()
    result = llm.generate(
        [{"prompt_token_ids": prompt}],
        SamplingParams(temperature=0, max_tokens=count, ignore_eos=True, detokenize=False),
        use_tqdm=False,
    )
    elapsed = time.perf_counter() - started
    if len(result) != 1 or not result[0].finished or len(result[0].outputs) != 1:
        raise ValueError("Expected one completed diagnostic request.")
    tokens = list(result[0].outputs[0].token_ids)
    evidence = single_worker_result(llm.collective_rpc(capture_worker_trace))
    logits = evidence.pop("logits")
    if len(tokens) != count or snapshot(llm)["finite"] is not True:
        raise ValueError("Diagnostic token count or deferred validity failure.")
    save_file({"logits": logits.contiguous()}, str(target))
    record = {
        "prompt": prompt,
        "tokens": tokens,
        "preset": preset,
        "evidence": evidence,
        "logits_file": str(target),
        "logits_sha256": digest(target),
        "generation_s": elapsed,
        "generation_scope": "diagnostic_with_checks_not_performance_TTFT_or_TPOT",
    }
    write_json(target.with_suffix(".json"), record)
    validate_diagnostic(evidence, logits, prompt, tokens, vocab, library)
    return record, logits


def validate_sample(sample, prompt, tokens, preset):
    count = len(tokens)
    if (
        sample.get("case") != f"p{len(prompt)}-o{count}"
        or sample.get("preset") != preset
        or sample.get("kind") not in KINDS
        or sample.get("tokens") != tokens
        or sample.get("tokens_exact") is not True
        or sample.get("finite") is not True
        or type(sample.get("forwards")) is not int
        or sample["forwards"] != count
    ):
        raise ValueError("Invalid benchmark case, tokens, finite flags or real forward count.")
    for key in ("native_calls", "native_launches"):
        if type(sample.get(key)) is not int or sample[key] < 2 * LAYERS * count:
            raise ValueError("Missing model native calls or launches.")
    if not sample["native_launches"] <= sample["native_calls"] <= 6 * sample["native_launches"]:
        raise ValueError("Invalid grouped native call/launch ratio.")
    cache = sample.get("cache_delta", {})
    if any(type(cache.get(k)) is not int or cache[k] < 0 for k in ("loads", "hits", "evictions")):
        raise ValueError("Invalid cache counter deltas.")
    ready = sample.get("token_ready_s", [])
    metrics = token_metrics(ready, sample["e2e_s"], count)
    if any(b <= a for a, b in zip(ready, ready[1:])):
        raise ValueError("Each token must have a distinct increasing wall-clock delivery timestamp.")
    for key in ("ttft_s", "tpot_s", "output_tokens_per_s"):
        if type(sample.get(key)) not in (int, float) or not math.isclose(
            sample[key], metrics[key], rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("Timing summary disagrees with actual per-token timestamps.")
    if sample.get("decode_intervals_s") != metrics["decode_intervals_s"]:
        raise ValueError("Decode intervals disagree with actual per-token timestamps.")
    distribution([sample["device_span_ms"]])
    return cache["loads"] == 0 and cache["evictions"] == 0


def summarize_samples(samples, cases, preset, warmups, repeats, diagnostics):
    validate_cases(cases)
    if warmups < 2 or repeats < 5:
        raise ValueError("Require at least two warmups and five measured requests.")
    expected_cases = {f"p{p}-o{o}" for p, o in cases}
    if set(diagnostics) != expected_cases or len(samples) != len(cases) * (1 + warmups + repeats):
        raise ValueError("Incomplete or extra diagnostic/sample matrix.")
    groups = []
    for p, o in cases:
        case = f"p{p}-o{o}"
        pair = diagnostics[case]
        if len(pair) != 2 or pair[0]["tokens"] != pair[1]["tokens"] or pair[0]["prompt"] != pair[1]["prompt"]:
            raise ValueError("Each case needs two same-preset diagnostic records.")
        prompt, tokens = pair[0]["prompt"], pair[0]["tokens"]
        if len(prompt) != p or len(tokens) != o or any(r.get("preset") != preset for r in pair):
            raise ValueError("Diagnostic case/preset mismatch.")
        measured = []
        for kind, count in zip(KINDS, (1, warmups, repeats)):
            rows = [s for s in samples if s.get("case") == case and s.get("kind") == kind]
            if (
                len(rows) != count
                or any(type(r.get("repeat")) is not int for r in rows)
                or sorted(r["repeat"] for r in rows) != list(range(count))
            ):
                raise ValueError("Missing or duplicate benchmark sample indices.")
            for row in rows:
                validate_sample(row, prompt, tokens, preset)
            if kind == "measured":
                measured = rows
        metrics = {k: distribution([s[k] for s in measured]) for k in ("ttft_s", "tpot_s", "e2e_s")}
        hot = all(validate_sample(s, prompt, tokens, preset) for s in measured)
        groups.append(
            {
                "case": case,
                "preset": preset,
                "n": repeats,
                **{k: v["median"] for k, v in metrics.items()},
                "distributions": metrics,
                "hot_cache_verified": hot,
                "cache_loads": sum(s["cache_delta"]["loads"] for s in measured),
                "cache_evictions": sum(s["cache_delta"]["evictions"] for s in measured),
                "interpretation": "resident packed expert observations"
                if hot
                else "includes cache misses/evictions; not isolated resident performance",
            }
        )
    return groups


def verify_report(report, library, model, cases, preset, warmups, repeats):
    """Pure CPU revalidation for the supervisor; never trusts a PASS flag alone."""
    import torch
    from safetensors.torch import load_file

    from vllm_ascend.quantization.vq2a8_ascendc_v2 import validate_build_manifest

    identity = validate_build_manifest(library["path"], digest(Path(library["path"])))
    if identity != library or report.get("library") != identity:
        raise ValueError("Benchmark library identity changed or differs from preflight.")
    if (
        report.get("status") != "PASS"
        or report.get("performance_measurement_verified") is not True
        or report.get("implementation") != "ascendc_v2"
        or report.get("scope") != SCOPE
        or report.get("preset") != preset
        or report.get("cases") != [list(c) for c in cases]
        or report.get("warmups") != warmups
        or report.get("repeats") != repeats
        or report.get("model") != model_identity(model)
        or report.get("python_source_sha256") != source_hashes()
        or report.get("library_unchanged") is not True
        or report.get("quality_verified") is not False
        or report.get("serving_verified") is not False
        or report.get("performance_target_met") is not None
    ):
        raise ValueError("Invalid benchmark report identity/scope/status.")
    env = report.get("measurement_environment", {})
    require_measurement_environment(
        {
            "ASCEND_LAUNCH_BLOCKING": env.get("ASCEND_LAUNCH_BLOCKING"),
            "ASCEND_RT_VISIBLE_DEVICES": env.get("physical_npu", ""),
        }
    )
    config = json.loads((Path(model) / "config.json").read_text(encoding="utf-8"))
    if config.get("num_hidden_layers") != LAYERS:
        raise ValueError("Benchmark requires the 43-layer model.")
    for pair in report.get("diagnostics", {}).values():
        if len(pair) != 2:
            raise ValueError("Each case needs two independent diagnostic requests.")
        values = []
        if pair[0].get("logits_file") == pair[1].get("logits_file"):
            raise ValueError("Diagnostic runs must have distinct retained artifacts.")
        for record in pair:
            path = Path(record["logits_file"])
            if digest(path) != record["logits_sha256"]:
                raise ValueError("Diagnostic logits artifact changed.")
            stored = load_file(str(path), device="cpu")
            if set(stored) != {"logits"}:
                raise ValueError("Unexpected diagnostic safetensors payload.")
            logits = stored["logits"]
            validate_diagnostic(
                record["evidence"], logits, record["prompt"], record["tokens"], config["vocab_size"], identity
            )
            values.append(logits)
        if not torch.equal(values[0].view(torch.uint8), values[1].view(torch.uint8)):
            raise ValueError("Same-preset v2 diagnostic logits are not bit-exact across requests.")
    summaries = summarize_samples(report["samples"], cases, preset, warmups, repeats, report["diagnostics"])
    if report.get("summaries") != summaries or report.get("hot_cache_verified") != all(
        s["hot_cache_verified"] for s in summaries
    ):
        raise ValueError("Saved benchmark statistics disagree with raw samples.")
    return report


def write_report(output, report):
    write_json(output / "summary.json", report)
    lines = [
        f"VQ2A8_V2_PERFORMANCE={report['status']}",
        f"SCOPE={SCOPE}",
        f"PERFORMANCE_MEASUREMENT_VERIFIED={report.get('performance_measurement_verified', False)}",
        "PERFORMANCE_TARGET_MET=null NO_SPEED_THRESHOLD",
        f"HOT_CACHE_VERIFIED={report.get('hot_cache_verified', False)}",
        "QUALITY=NOT_REQUESTED SERVING=NOT_REQUESTED",
    ]
    for row in report.get("summaries", []):
        lines.append(
            f"CASE={row['case']} PRESET={row['preset']} n={row['n']} "
            f"TTFT_S={row['ttft_s']:.6f} TPOT_S={row['tpot_s']:.6f} E2E_S={row['e2e_s']:.6f} "
            f"HOT_CACHE={row['hot_cache_verified']}"
        )
    if report.get("error"):
        lines.append(f"ERROR={report['error']}")
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "RUNNING",
        "implementation": "ascendc_v2",
        "scope": SCOPE,
        "preset": args.preset,
        "cases": [list(c) for c in args.cases],
        "warmups": args.warmups,
        "repeats": args.repeats,
        "samples": [],
        "diagnostics": {},
        "performance_measurement_verified": False,
        "performance_target_met": None,
        "quality_verified": False,
        "serving_verified": False,
        "native_instruction_verified": False,
        "on_chip_decode_verified": False,
        "timing_scope": "host wall time from request submission to first token; TPOT averages later token intervals",
        "startup_excluded_from_ttft": True,
        "first_sample_scope": "first timed request after two diagnostic requests; not cold-cache TTFT",
        "excluded": ["HTTP/client latency", "concurrency throughput", "quality", "v1-v2 exactness", "context >128"],
    }
    write_report(output, report)
    try:
        report["measurement_environment"] = require_measurement_environment()
        from tools.validate_vq2a8_ascendc import require_hardware_runtime
        from tools.validate_vq2a8_v026_environment import check_scheduler_apis, require_v026_stack

        require_v026_stack()
        require_hardware_runtime()
        library = checked_model_preflight(args.library, args.preflight, args.model)
        report.update(library=library, model=model_identity(args.model), python_source_sha256=source_hashes())
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

        report["environment"] = environment_report()
        report["scheduler"] = check_scheduler_apis()
        report["device"] = _initialize_device(torch.device("npu:0"))
        require_hardware_runtime()
        root = args.model.resolve(strict=True)
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if config.get("num_hidden_layers") != LAYERS:
            raise ValueError("This benchmark requires the 43-layer VQ2A8 model.")
        tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        seed = tokenizer.encode(
            "The answer to 1 + 1 is. Read the following short example. ", add_special_tokens=False
        ).ids
        if not seed:
            raise ValueError("Empty benchmark seed.")
        prompts = {}
        for length, _ in args.cases:
            prefix = [config["bos_token_id"]] if type(config.get("bos_token_id")) is int else []
            prompts[length] = prefix + (seed * (length // len(seed) + 1))[: length - len(prefix)]
        run_preflight(torch.device("npu:0"), config, prompt_tokens=10)
        run_sas_preflight(torch.device("npu:0"), config, prompt_tokens=10)
        options = offline_engine_options(
            root,
            root / "experts_vq_ascend_v2",
            execution_policy="ascendc_v2",
            root_linear_mode="bf16",
            ascendc_v2_library=library["path"],
            ascendc_v2_sha256=library["sha256"],
        )
        options.update(max_model_len=MAX_CONTEXT, max_num_batched_tokens=MAX_CONTEXT)
        report["engine_options"] = options
        engine_started = time.perf_counter()
        llm = LLM(**options)
        report.update(
            startup_s=time.perf_counter() - started, engine_init_profile_kv_s=time.perf_counter() - engine_started
        )
        print(f"PERF_V2_ENGINE_READY startup_s={report['startup_s']:.3f}", flush=True)
        for length, count in args.cases:
            case, prompt = f"p{length}-o{count}", prompts[length]
            records, values = [], []
            report["diagnostics"][case] = records
            for index in range(2):
                record, logits = diagnostic(
                    llm,
                    prompt,
                    count,
                    args.preset,
                    output / f"{case}-diagnostic-{index}.safetensors",
                    config["vocab_size"],
                    library,
                )
                records.append(record)
                values.append(logits)
                write_report(output, report)
            if records[0]["tokens"] != records[1]["tokens"] or not torch.equal(
                values[0].view(torch.uint8), values[1].view(torch.uint8)
            ):
                raise ValueError("Same-preset diagnostic repeat is not bit-exact; timing not started.")
            for kind, repeats in zip(KINDS, (1, args.warmups, args.repeats)):
                for index in range(repeats):
                    configure(llm, measurement=True, compact=False, optimization=args.preset)
                    sample = timed_request(llm, prompt, count, f"{case}-{kind}-{index}")
                    sample.update(
                        case=case,
                        preset=args.preset,
                        kind=kind,
                        repeat=index,
                        tokens_exact=sample["tokens"] == records[0]["tokens"],
                    )
                    report["samples"].append(sample)
                    write_report(output, report)
                    validate_sample(sample, prompt, records[0]["tokens"], args.preset)
                    print(
                        "PERF_V2_SAMPLE "
                        + json.dumps(
                            {
                                k: sample[k]
                                for k in (
                                    "case",
                                    "preset",
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
        report["summaries"] = summarize_samples(
            report["samples"], args.cases, args.preset, args.warmups, args.repeats, report["diagnostics"]
        )
        report["hot_cache_verified"] = all(s["hot_cache_verified"] for s in report["summaries"])
        report["library_unchanged"] = digest(args.library) == library["sha256"]
        report.update(status="PASS", performance_measurement_verified=True)
        verify_report(report, library, args.model, args.cases, args.preset, args.warmups, args.repeats)
    except Exception as exc:
        report.update(
            status="FAIL", performance_measurement_verified=False, error=str(exc), traceback=traceback.format_exc()
        )
        raise
    finally:
        write_report(output, report)
        print(f"VQ2A8_V2_PERFORMANCE={report['status']} REPORT={output / 'summary.json'}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "library", "preflight", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--preset", choices=("fast", "batched"), default="batched")
    parser.add_argument("--cases", default="10:4")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        args.cases = validate_cases([tuple(map(int, value.split(":"))) for value in args.cases.split(",")])
        if args.warmups < 2 or args.repeats < 5:
            raise ValueError("Require >=2 warmups and >=5 measured requests.")
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    return args


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
