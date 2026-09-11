#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated v1 references / exact v3 model gates / per-token event+wall timing.

No full-model graph or 20 ms result is implied. Residency is mandatory for v3;
policy-budget failure is not proof that the physical device cannot fit the model.
"""

from __future__ import annotations

# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import math
import time
import traceback
from pathlib import Path

from tools.validate_vq2a8_ascendc_v3 import checked_model_preflight, model_identity, python_source_hashes, sha256
from tools.vq2a8_perf_report import MAX_CONTEXT, distribution, token_metrics, validate_cases

LAYERS = 43
SCOPE = "TP1_B1_OFFLINE_CONTEXT_LE_128_EAGER"


def parse_cases(value):
    return validate_cases([tuple(map(int, case.split(":"))) for case in value.split(",")])


def validate_options(args):
    parse_cases(args.cases)
    if args.reference_only and args.correctness_only:
        raise ValueError("Choose reference-only or correctness-only, not both")
    if args.profile and (args.reference_only or args.correctness_only):
        raise ValueError("--profile requires the performance mode")
    if not args.reference_only and not args.preflight:
        raise ValueError("V3 model allocation requires --preflight")
    if not args.reference_only and not args.reference_report:
        raise ValueError("V3 model gates require a hash-bound v1 --reference-report")
    if args.warmups < 2 or args.repeats < 5:
        raise ValueError("Require >=2 warmups and >=5 measured requests")
    if (
        not math.isfinite(args.cache_budget_gib)
        or args.cache_budget_gib < 0
        or not math.isfinite(args.cache_reserve_gib)
        or args.cache_reserve_gib < 1
        or not math.isfinite(args.memory_fraction)
        or not 0 < args.memory_fraction <= 1
    ):
        raise ValueError("Budget >=0, reserve >=1 GiB, memory fraction in (0,1] must be finite")
    if args.target_tpot_ms is not None and (not math.isfinite(args.target_tpot_ms) or args.target_tpot_ms <= 0):
        raise ValueError("Target TPOT must be finite and positive")


def configuration(args):
    return dict(
        cache_budget_gib=args.cache_budget_gib,
        cache_reserve_gib=args.cache_reserve_gib,
        memory_fraction=args.memory_fraction,
        preset="batched",
        root_linear_mode="bf16",
        max_context=MAX_CONTEXT,
        tensor_parallel_size=1,
        max_num_seqs=1,
    )


def write_report(output, report):
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        f"VQ2A8_V3={report['status']} MODE={report['mode']}",
        f"BASELINE_EXACT={report.get('baseline_exact', False)} REPEAT_EXACT={report.get('repeat_exact', False)}",
        f"PERFORMANCE_MEASUREMENT_VERIFIED={report.get('performance_measurement_verified', False)}",
        f"PERFORMANCE_TARGET_MET={report.get('performance_target_met')}",
        "FULL_MODEL_GRAPH_VERIFIED=False DEFAULT_BACKEND=UNCHANGED",
        f"PROFILE_STATUS={report.get('profile', {}).get('status', 'NOT_REQUESTED')}",
    ]
    for row in report.get("summaries", []):
        lines.append(
            f"CASE={row['case']} TTFT_MS={row['ttft_ms']:.4f} TPOT_MS={row['tpot_ms']:.4f} "
            f"DECODE_TOKEN_P95_MS={row['decode_token_ms']['p95_nearest_rank']:.4f} "
            f"DEVICE_TOKEN_P95_MS={row['device_decode_token_ms']['p95_nearest_rank']:.4f}"
        )
    if report.get("error"):
        lines.append("ERROR=" + report["error"])
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_diagnostic(record, logits, library, policy, vocab):
    import torch

    prompt, tokens, evidence = record["prompt"], record["tokens"], record["evidence"]
    validate_cases([(len(prompt), len(tokens))])
    if any(type(t) is not int or not 0 <= t < vocab for t in prompt + tokens):
        raise ValueError("Invalid diagnostic token IDs")
    expected = [{"tokens": len(prompt), "positions": list(range(len(prompt)))}]
    expected += [{"tokens": 1, "positions": [len(prompt) + i]} for i in range(len(tokens) - 1)]
    backend = evidence.get("expert_backend", {})
    if (
        evidence.get("steps") != expected
        or backend.get("policy") != policy
        or backend.get("fallback_enabled") is not False
        or backend.get("library") != library
        or evidence.get("root_fp8", {}).get("mode") != "bf16"
    ):
        raise ValueError("Diagnostic must use exact pinned policy/library and actual prefill+decode steps")
    layers = backend.get("layers", [])
    if len(layers) != LAYERS or {r.get("layer") for r in layers} != set(range(LAYERS)):
        raise ValueError("Missing 43-layer native coverage")
    for layer in layers:
        if len(layer.get("steps", [])) != len(tokens):
            raise ValueError("Missing per-layer decode steps")
        for step, expected_step in zip(layer["steps"], expected):
            if (
                step.get("tokens") != expected_step["tokens"]
                or type(step.get("projection_calls")) is not int
                or step["projection_calls"] < 2
                or type(step.get("kernel_launches")) is not int
                or step["kernel_launches"] < 2
            ):
                raise ValueError("Missing per-step native projection/launch evidence")
    cache = evidence.get("cache", {})
    if {int(k): v for k, v in cache.get("layer_calls", {}).items()} != {i: len(tokens) for i in range(LAYERS)}:
        raise ValueError("Every generated step must execute every layer")
    if evidence.get("load", {}).get("registered_parameters_loaded", 0) <= 0:
        raise ValueError("Missing strict loading evidence")
    if (
        logits.device.type != "cpu"
        or logits.dtype != torch.float32
        or logits.shape != (len(tokens), vocab)
        or not bool(torch.isfinite(logits).all())
        or not torch.equal(logits[torch.arange(len(tokens)), torch.tensor(tokens)], logits.max(-1).values)
    ):
        raise ValueError("Invalid finite FP32 logits / greedy agreement")


def diagnostic(llm, prompt, count, target, library, policy, vocab):
    from safetensors.torch import save_file
    from vllm import SamplingParams

    from tools.benchmark_vq2a8_offline import configure, snapshot
    from tools.validate_vq2a8_tp1_offline import capture_worker_trace, reset_worker_trace, single_worker_result

    configure(llm, measurement=False, compact=False, optimization="batched")
    if single_worker_result(llm.collective_rpc(reset_worker_trace)) != os.getpid():
        raise ValueError("Worker must stay in supervised process")
    before = snapshot(llm)
    result = llm.generate(
        [{"prompt_token_ids": prompt}],
        SamplingParams(temperature=0, max_tokens=count, ignore_eos=True, detokenize=False),
        use_tqdm=False,
    )
    if len(result) != 1 or not result[0].finished or len(result[0].outputs) != 1:
        raise ValueError("Expected one finished diagnostic request")
    tokens = list(result[0].outputs[0].token_ids)
    evidence = single_worker_result(llm.collective_rpc(capture_worker_trace))
    logits = evidence.pop("logits")
    observed = snapshot(llm)
    if len(tokens) != count or observed["finite"] is not True:
        raise ValueError("Diagnostic count/finite failure")
    save_file({"logits": logits.contiguous()}, str(target))
    record = dict(
        prompt=prompt,
        tokens=tokens,
        evidence=evidence,
        logits_file=str(target),
        logits_sha256=sha256(target),
        v3=observed.get("v3", {}),
        v3_before=before.get("v3", {}),
    )
    validate_diagnostic(record, logits, library, policy, vocab)
    target.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record, logits


def load_diagnostic(record, library, policy, vocab):
    from safetensors.torch import load_file

    path = Path(record["logits_file"])
    if sha256(path) != record["logits_sha256"]:
        raise ValueError("Retained logits hash mismatch")
    payload = load_file(str(path), device="cpu")
    if set(payload) != {"logits"}:
        raise ValueError("Unexpected logits payload")
    validate_diagnostic(record, payload["logits"], library, policy, vocab)
    return payload["logits"]


def validate_reference(report, args, cases):
    if (
        report.get("schema_version") != 1
        or report.get("status") != "PASS"
        or report.get("mode") != "reference"
        or report.get("implementation") != "ascendc"
        or report.get("repeat_exact") is not True
        or report.get("model") != model_identity(args.model)
        or report.get("python_source_sha256") != python_source_hashes()
        or report.get("configuration") != configuration(args)
        or report.get("physical_npu") != os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        or report.get("cases") != [list(c) for c in cases]
        or not str(report.get("soc", "")).startswith("Ascend950")
    ):
        raise ValueError("v1 reference report stale, incomplete, or configuration/model/device/cases differ")
    library = report.get("library", {})
    if sha256(Path(library["path"])) != library.get("sha256"):
        raise ValueError("v1 reference library changed")
    return report


def timed_request(llm, prompt, count, request_id):
    """Record one real NPU event per delivered token; never divide a bulk timer."""
    import torch
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    from tools.benchmark_vq2a8_offline import snapshot

    before = snapshot(llm)
    begin = torch.npu.Event(enable_timing=True)
    token_events = [torch.npu.Event(enable_timing=True) for _ in range(count)]
    torch.npu.synchronize()
    started = time.perf_counter()
    begin.record()
    engine = llm.llm_engine
    engine.add_request(
        request_id,
        {"prompt_token_ids": prompt},
        SamplingParams(
            temperature=0, max_tokens=count, ignore_eos=True, detokenize=False, output_kind=RequestOutputKind.CUMULATIVE
        ),
    )
    ready, events, tokens, finished = [], [], [], False
    for _ in range(count + 4):
        for result in engine.step():
            if result.request_id != request_id or len(result.outputs) != 1:
                raise ValueError("Unexpected multiplexing in B1 benchmark")
            current = list(result.outputs[0].token_ids)
            if current[: len(tokens)] != tokens or len(current) > len(tokens) + 1:
                raise ValueError("Expected single-token ordered outputs, no speculative batching")
            if len(current) > len(tokens):
                event = token_events[len(events)]
                event.record()
                events.append(event)
                ready.append(time.perf_counter() - started)
            tokens, finished = current, result.finished
        if finished:
            break
    torch.npu.synchronize()
    elapsed = time.perf_counter() - started
    if not finished or len(tokens) != count or engine.has_unfinished_requests():
        raise ValueError("Incomplete benchmark generation")
    after = snapshot(llm)
    event_ready = [begin.elapsed_time(event) for event in events]
    return {
        **token_metrics(ready, elapsed, count),
        "token_ready_s": ready,
        "tokens": tokens,
        "token_event_ms": event_ready,
        "device_span_ms": event_ready[-1],
        "device_decode_intervals_ms": [b - a for a, b in zip(event_ready, event_ready[1:])],
        "event_scope": "current NPU stream event after each engine token; includes host submission gaps",
        "finite": after["finite"],
        "forwards": after["forwards"],
        "cache_delta": {k: after["cache"][k] - before["cache"][k] for k in ("loads", "hits", "evictions")},
        "native_calls": after["native_calls"] - before["native_calls"],
        "native_launches": after["native_launches"] - before["native_launches"],
        "expert_payload_h2d_bytes": after["h2d_bytes"] - before["h2d_bytes"],
        "memory": {k: v for k, v in after.items() if "bytes" in k},
        "v3_before": before.get("v3", {}),
        "v3_after": after.get("v3", {}),
    }


def validate_residency(before, after, decode_count):
    expected_layers = {str(i) for i in range(LAYERS)}
    if set(after) != expected_layers or (before is not None and set(before) != expected_layers):
        raise ValueError("Missing full 43-layer resident device-route evidence")
    for layer in expected_layers:
        current = after[layer]
        if (
            current.get("ready") is not True
            or current.get("full_model_graph_verified") is not False
            or current.get("route_host_reads") != 0
            or current.get("descriptor_h2d_bytes") != 0
            or type(current.get("decode_calls")) is not int
        ):
            raise ValueError("Resident layer not ready or decode used host routing/descriptor H2D")
        previous = 0 if before is None else before[layer].get("decode_calls")
        if type(previous) is not int or current["decode_calls"] - previous < decode_count:
            raise ValueError("Resident device path did not cover each real decode token")
        if before is not None and current["decode_calls"] - previous != decode_count:
            raise ValueError("Unexpected dummy or extra resident decode calls during timing")


def validate_sample(sample, count, expected_tokens):
    if (
        sample.get("tokens") != expected_tokens
        or sample.get("tokens_exact") is not True
        or sample.get("finite") is not True
        or sample.get("forwards") != count
        or sample.get("expert_payload_h2d_bytes") != 0
        or sample.get("cache_delta", {}).get("loads") != 0
        or sample.get("cache_delta", {}).get("evictions") != 0
    ):
        raise ValueError("Measured v3 request must be exact, finite, fully resident with zero loads/evictions/H2D")
    validate_residency(sample.get("v3_before", {}), sample.get("v3_after", {}), count - 1)
    for key in ("native_calls", "native_launches"):
        if type(sample.get(key)) is not int or sample[key] < 2 * LAYERS * count:
            raise ValueError("Missing native work coverage")
    metrics = token_metrics(sample["token_ready_s"], sample["e2e_s"], count)
    if any(
        not math.isclose(sample[k], metrics[k], rel_tol=1e-9, abs_tol=1e-12)
        for k in ("ttft_s", "tpot_s", "output_tokens_per_s")
    ):
        raise ValueError("Summary differs from raw per-token wall time")
    events = sample.get("token_event_ms", [])
    if len(events) != count:
        raise ValueError("Missing per-token NPU events")
    distribution(events)
    intervals = [b - a for a, b in zip(events, events[1:])]
    distribution(intervals)
    if intervals != sample.get("device_decode_intervals_ms") or metrics["decode_intervals_s"] != sample.get(
        "decode_intervals_s"
    ):
        raise ValueError("Raw token intervals disagree")


def summarize_samples(samples, cases, warmups, repeats, diagnostics):
    if len(samples) != len(cases) * (warmups + repeats):
        raise ValueError("Incomplete/extra timing sample matrix")
    rows = []
    for p, o in cases:
        case = f"p{p}-o{o}"
        measured = []
        for kind, count in (("warmup", warmups), ("measured", repeats)):
            group = [s for s in samples if s.get("case") == case and s.get("kind") == kind]
            if len(group) != count or sorted(s["repeat"] for s in group) != list(range(count)):
                raise ValueError("Missing or duplicated timing repeats")
            for sample in group:
                validate_sample(sample, o, diagnostics[case][0]["tokens"])
            if kind == "measured":
                measured = group
        rows.append(
            dict(
                case=case,
                n=repeats,
                ttft_ms=distribution([s["ttft_s"] * 1000 for s in measured])["median"],
                tpot_ms=distribution([s["tpot_s"] * 1000 for s in measured])["median"],
                decode_token_ms=distribution([v * 1000 for s in measured for v in s["decode_intervals_s"]]),
                device_decode_token_ms=distribution([v for s in measured for v in s["device_decode_intervals_ms"]]),
                hot_resident_verified=True,
            )
        )
    return rows


def collect_profile(llm, prompt, count, expected_tokens, output):
    """Extra request after timing; never a performance sample or ISA claim."""
    import torch_npu

    from tools.benchmark_vq2a8_offline import configure

    target = output / f"profile-v3-p{len(prompt)}-o{count}"
    target.mkdir(exist_ok=False)
    result = dict(
        status="RUNNING",
        scope="untimed_profile_not_performance_sample",
        native_instruction_verified=False,
        directory=str(target),
    )
    try:
        print(f"PERF_V3_STAGE=untimed_profile CASE=p{len(prompt)}-o{count}", flush=True)
        configure(llm, measurement=True, compact=False, optimization="v3", profile=True)
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,
            with_stack=True,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(target)),
        ) as profiler:
            sample = timed_request(llm, prompt, count, "untimed-profile-v3")
            sample["tokens_exact"] = sample["tokens"] == expected_tokens
            profiler.step()
        validate_sample(sample, count, expected_tokens)
        artifacts = {p.relative_to(target).as_posix(): sha256(p) for p in target.rglob("*") if p.is_file()}
        if not artifacts:
            raise ValueError("Profiler produced no artifacts")
        result.update(status="PASS", artifacts_sha256=artifacts)
    except Exception as exc:
        result.update(status="FAIL", error=str(exc))
    (target / "profile-status.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def verify_report(report, args):
    """Recompute retained logits and timing evidence; exit=0 alone never passes."""
    import torch

    cases = parse_cases(args.cases)
    expected_mode = "reference" if args.reference_only else "correctness" if args.correctness_only else "performance"
    if (
        report.get("schema_version") != 1
        or report.get("status") != "PASS"
        or report.get("mode") != expected_mode
        or report.get("implementation") != ("ascendc" if args.reference_only else "ascendc_v3")
        or report.get("scope") != SCOPE
        or report.get("model") != model_identity(args.model)
        or report.get("python_source_sha256") != python_source_hashes()
        or report.get("configuration") != configuration(args)
        or report.get("cases") != [list(c) for c in cases]
        or report.get("repeat_exact") is not True
        or report.get("full_model_graph_verified") is not False
        or report.get("physical_npu") != os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    ):
        raise ValueError("Invalid model report status/configuration/identity")
    policy = "ascendc" if args.reference_only else "ascendc_v3"
    library = report["library"]
    if library["path"] != str(args.library.resolve()) or sha256(args.library) != library["sha256"]:
        raise ValueError("Library changed after execution")
    if not args.reference_only:
        identity = checked_model_preflight(args.library, args.preflight, args.model)
        if identity != library:
            raise ValueError("Preflight library identity mismatch")
        if (
            report.get("reference_report_sha256") != sha256(args.reference_report)
            or report.get("baseline_exact") is not True
        ):
            raise ValueError("Missing exact hash-bound baseline comparison")
        reference = validate_reference(json.loads(args.reference_report.read_text(encoding="utf-8")), args, cases)
    vocab = json.loads((args.model / "config.json").read_text(encoding="utf-8"))["vocab_size"]
    if set(report.get("diagnostics", {})) != {f"p{p}-o{o}" for p, o in cases}:
        raise ValueError("Incomplete diagnostic cases")
    for case, pair in report["diagnostics"].items():
        if len(pair) != 2 or pair[0]["logits_file"] == pair[1]["logits_file"]:
            raise ValueError("Two independent retained diagnostics required")
        left, right = [load_diagnostic(r, library, policy, vocab) for r in pair]
        if (
            pair[0]["prompt"] != pair[1]["prompt"]
            or pair[0]["tokens"] != pair[1]["tokens"]
            or not torch.equal(left.view(torch.uint8), right.view(torch.uint8))
        ):
            raise ValueError("Model repeat is not bit-exact")
        if not args.reference_only:
            for record in pair:
                validate_residency(record.get("v3_before", {}), record.get("v3", {}), len(record["tokens"]) - 1)
            refpair = reference.get("diagnostics", {}).get(case, [])
            if len(refpair) != 2:
                raise ValueError("Missing baseline case")
            refs = [load_diagnostic(r, reference["library"], "ascendc", vocab) for r in refpair]
            if any(r["prompt"] != pair[0]["prompt"] or r["tokens"] != pair[0]["tokens"] for r in refpair) or any(
                not torch.equal(left.view(torch.uint8), v.view(torch.uint8)) for v in refs
            ):
                raise ValueError(f"V3 differs from v1 strict per-step logits/tokens: {case}")
    if expected_mode == "performance":
        if (
            report.get("warmups") != args.warmups
            or report.get("repeats") != args.repeats
            or report.get("target_tpot_ms") != args.target_tpot_ms
        ):
            raise ValueError("Timing configuration differs from requested matrix/target")
        if report.get("measurement_launch_blocking") not in (None, "0"):
            raise ValueError("Timing ran with launch blocking")
        summaries = summarize_samples(report["samples"], cases, args.warmups, args.repeats, report["diagnostics"])
        target = (
            all(r["tpot_ms"] <= args.target_tpot_ms for r in summaries) if args.target_tpot_ms is not None else None
        )
        if (
            report.get("summaries") != summaries
            or report.get("performance_target_met") is not target
            or report.get("performance_measurement_verified") is not True
        ):
            raise ValueError("Statistics/target disagree with raw measurements")
    elif (
        report.get("performance_measurement_verified") is not False or report.get("performance_target_met") is not None
    ):
        raise ValueError("Non-performance run may not claim a timing target")
    return report


def run(args):
    validate_options(args)
    cases = parse_cases(args.cases)
    policy = "ascendc" if args.reference_only else "ascendc_v3"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        schema_version=1,
        status="RUNNING",
        mode="reference" if args.reference_only else "correctness" if args.correctness_only else "performance",
        implementation=policy,
        scope=SCOPE,
        configuration=configuration(args),
        cases=[list(c) for c in cases],
        diagnostics={},
        samples=[],
        baseline_exact=False,
        repeat_exact=False,
        performance_measurement_verified=False,
        performance_target_met=None,
        full_model_graph_verified=False,
        quality_verified=False,
        serving_verified=False,
        target_tpot_ms=args.target_tpot_ms,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    write_report(output, report)
    try:
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
        if not visible.isdecimal():
            raise ValueError("Select exactly one physical NPU")
        if report["mode"] == "performance" and os.environ.get("ASCEND_LAUNCH_BLOCKING") not in (None, "0"):
            raise ValueError("Timing requires ASCEND_LAUNCH_BLOCKING unset or 0")
        report.update(
            physical_npu=visible,
            measurement_launch_blocking=os.environ.get("ASCEND_LAUNCH_BLOCKING"),
            model=model_identity(args.model),
            python_source_sha256=python_source_hashes(),
        )
        from tools.validate_vq2a8_ascendc import require_hardware_runtime
        from tools.validate_vq2a8_v026_environment import require_v026_stack

        require_hardware_runtime()
        require_v026_stack()
        if args.reference_only:
            from tools.validate_vq2a8_ascendc import library_evidence

            evidence = library_evidence(args.library.resolve())
            library = {"path": evidence["path"], "sha256": evidence["sha256"]}
        else:
            library = checked_model_preflight(args.library, args.preflight, args.model)
            reference = validate_reference(json.loads(args.reference_report.read_text(encoding="utf-8")), args, cases)
            report["reference_report_sha256"] = sha256(args.reference_report)
        report["library"] = library
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        import torch
        import torch_npu  # noqa: F401
        from tokenizers import Tokenizer
        from vllm import LLM

        from tools.benchmark_vq2a8_offline import configure, snapshot
        from tools.validate_vq2a8_qli_metadata import run_preflight
        from tools.validate_vq2a8_sas_attention import run_sas_preflight
        from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device, environment_report
        from vllm_ascend.quantization.vq2a8_offline import offline_engine_options

        report["device"] = _initialize_device(torch.device("npu:0"))
        report["soc"] = torch.npu.get_device_name(0)
        report["environment"] = environment_report()
        require_hardware_runtime()
        if not args.reference_only and reference["soc"] != report["soc"]:
            raise ValueError("Baseline/v3 exact SoC mismatch")
        config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
        if config.get("num_hidden_layers") != LAYERS:
            raise ValueError("Expected 43-layer VQ2 model")
        print("PERF_V3_STAGE=attention_preflight", flush=True)
        run_preflight(torch.device("npu:0"), config, prompt_tokens=10)
        run_sas_preflight(torch.device("npu:0"), config, prompt_tokens=10)
        if args.reference_only:
            from tools.validate_vq2a8_phase4_kernel import (
                compare,
                same_fp8_oracle,
                synthetic_dense_oracle,
                synthetic_inputs,
            )
            from vllm_ascend.quantization.vq2a8_ascendc import load_pinned_library, vq2a8_ascendc

            load_pinned_library(library["path"], library["sha256"])
            values = synthetic_inputs(1, 64, 512, 2)
            report["baseline_operator_preflight"] = compare(
                same_fp8_oracle(values[:3], synthetic_dense_oracle(*values[3:])),
                vq2a8_ascendc(*(v.to("npu:0") for v in values)),
            )
        kwargs = (
            {"ascendc_library": library["path"], "ascendc_sha256": library["sha256"]}
            if args.reference_only
            else {"ascendc_v3_library": library["path"], "ascendc_v3_sha256": library["sha256"]}
        )
        options = offline_engine_options(
            args.model,
            args.model / "experts_vq_ascend_v2",
            execution_policy=policy,
            cache_budget_gib=args.cache_budget_gib,
            cache_reserve_gib=args.cache_reserve_gib,
            root_linear_mode="bf16",
            **kwargs,
        )
        options.update(
            max_model_len=MAX_CONTEXT, max_num_batched_tokens=MAX_CONTEXT, gpu_memory_utilization=args.memory_fraction
        )
        report["engine_options"] = options
        report["before_model_memory"] = dict(
            free_total_bytes=list(torch.npu.mem_get_info()),
            allocated_bytes=torch.npu.memory_allocated(),
            reserved_bytes=torch.npu.memory_reserved(),
        )
        torch.npu.reset_peak_memory_stats()
        print(
            f"PERF_V3_STAGE=model_load POLICY={policy} FULL_RESIDENT_REQUIRED={not args.reference_only} "
            f"CONFIG={json.dumps(configuration(args))}",
            flush=True,
        )
        write_report(output, report)
        started = time.perf_counter()
        llm = LLM(**options)
        report["startup_s"] = time.perf_counter() - started
        report["startup_memory"] = snapshot(llm)
        print(f"PERF_V3_ENGINE_READY STARTUP_S={report['startup_s']:.3f}", flush=True)
        tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
        seed = tokenizer.encode("The answer to 1 + 1 is. Read this short example. ", add_special_tokens=False).ids
        if not seed:
            raise ValueError("Empty prompt seed")
        for p, o in cases:
            case = f"p{p}-o{o}"
            prefix = [config["bos_token_id"]] if type(config.get("bos_token_id")) is int else []
            prompt = prefix + (seed * (p // len(seed) + 1))[: p - len(prefix)]
            records, values = [], []
            report["diagnostics"][case] = records
            for index in range(2):
                print(f"PERF_V3_CASE_START={case} KIND=diagnostic REPEAT={index + 1}/2", flush=True)
                record, logits = diagnostic(
                    llm,
                    prompt,
                    o,
                    output / f"{case}-diagnostic-{index}.safetensors",
                    library,
                    policy,
                    config["vocab_size"],
                )
                records.append(record)
                values.append(logits)
                write_report(output, report)
            if records[0]["tokens"] != records[1]["tokens"] or not torch.equal(
                values[0].view(torch.uint8), values[1].view(torch.uint8)
            ):
                raise ValueError("V3/reference self-repeat is not exact")
            if not args.reference_only:
                for reference_record in reference["diagnostics"][case]:
                    golden = load_diagnostic(reference_record, reference["library"], "ascendc", config["vocab_size"])
                    if (
                        reference_record["prompt"] != prompt
                        or reference_record["tokens"] != records[0]["tokens"]
                        or not torch.equal(golden.view(torch.uint8), values[0].view(torch.uint8))
                    ):
                        raise ValueError(f"v1/v3 exact logits or token comparison failed: {case}")
            print(f"PERF_V3_CASE_EXACT={case} REPEAT_EXACT=True BASELINE_EXACT={not args.reference_only}", flush=True)
            if report["mode"] == "performance":
                for kind, count in (("warmup", args.warmups), ("measured", args.repeats)):
                    for index in range(count):
                        print(f"PERF_V3_CASE_START={case} KIND={kind} REPEAT={index + 1}/{count}", flush=True)
                        configure(llm, measurement=True, compact=False, optimization="batched")
                        sample = timed_request(llm, prompt, o, f"v3-{case}-{kind}-{index}")
                        sample.update(
                            case=case, kind=kind, repeat=index, tokens_exact=sample["tokens"] == records[0]["tokens"]
                        )
                        validate_sample(sample, o, records[0]["tokens"])
                        report["samples"].append(sample)
                        write_report(output, report)
                        print(f"PERF_V3_SAMPLE={case} KIND={kind} TPOT_MS={sample['tpot_s'] * 1000:.4f}", flush=True)
        report.update(repeat_exact=True, baseline_exact=not args.reference_only)
        if report["mode"] == "performance":
            report["summaries"] = summarize_samples(
                report["samples"], cases, args.warmups, args.repeats, report["diagnostics"]
            )
            report["performance_measurement_verified"] = True
            if args.target_tpot_ms is not None:
                report["performance_target_met"] = all(r["tpot_ms"] <= args.target_tpot_ms for r in report["summaries"])
            if args.profile:
                write_report(output, report)
                report["profile"] = collect_profile(llm, prompt, o, records[0]["tokens"], output)
        report["status"] = "PASS"
        verify_report(report, args)
    except Exception as exc:
        report.update(
            status="FAIL",
            performance_measurement_verified=False,
            performance_target_met=None,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        write_report(output, report)
        print((output / "summary.txt").read_text(encoding="utf-8"), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "library", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--profile", action="store_true", help="One extra untimed CPU/NPU trace after measurements")
    parser.add_argument("--cases", default="10:4", help="Also supports 10:64,32:64; total context <=128")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cache-budget-gib", type=float, default=0.0)
    parser.add_argument("--cache-reserve-gib", type=float, default=16.0)
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    parser.add_argument(
        "--target-tpot-ms", type=float, help="Optional observation target; missing it never fails correctness"
    )
    args = parser.parse_args(argv)
    try:
        validate_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    if args.plan_only:
        print(
            json.dumps(
                dict(
                    scope="plan_only_no_device_execution",
                    cases=parse_cases(args.cases),
                    configuration=configuration(args),
                    full_model_graph_verified=False,
                ),
                indent=2,
            )
        )
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
