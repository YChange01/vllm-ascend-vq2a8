#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One fresh TP1 worker: V4 residency + V1 batched math, no V3 execution.

Default acceptance checks repeatability, not agreement with a measured V1
reference. Use the independent supervisor's --compare-v1 for that comparison.
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
import subprocess
import time
import traceback
from pathlib import Path

from tools.profile_vq2a8_ascendc import digest, write_json
from tools.vq2a8_baseline import capture_input_identity
from tools.vq2a8_optimization_report import numerical_gate
from tools.vq2a8_perf_report import MAX_CONTEXT, distribution, token_metrics, validate_cases

REPO = Path(__file__).resolve().parents[1]
LAYERS = 43
V4_POLICY = "ascendc_v4"
SCOPE = "TP1_B1_EAGER_CONTEXT_LE_128_V1_BATCHED_ARITHMETIC"
SOURCE_NAMES = (
    "vllm_ascend/quantization/vq2a8_execution.py",
    "vllm_ascend/quantization/vq2a8_execution_v4.py",
    "vllm_ascend/quantization/vq2a8_v4_device_route.py",
    "vllm_ascend/quantization/vq2a8_optimization.py",
    "vllm_ascend/quantization/vq2a8_activation.py",
    "vllm_ascend/quantization/vq2a8_moe.py",
    "vllm_ascend/quantization/vq2a8_offline.py",
    "vllm_ascend/patch/worker/vq2a8_offline_model.py",
    "tools/benchmark_vq2a8_v4.py",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/home/g00872988/vq2a8"))
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-npu", type=int, default=1)
    parser.add_argument("--cache-reserve-gib", type=float, default=8.0)
    parser.add_argument("--cache-budget-gib", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=MAX_CONTEXT)
    parser.add_argument("--kv-cache-mib", type=int, default=1024)
    parser.add_argument("--memory-fraction", type=float, default=None, help="Independent expert budget fraction")
    parser.add_argument("--engine-memory-fraction", type=float, default=0.9)
    parser.add_argument("--cases", default="10:4")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--reference-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--device-route-decode",
        action="store_true",
        help="Compare batched baseline and device-route decode in one V4 resident engine (requires rebuilt library)",
    )
    args = parser.parse_args(argv)
    try:
        args.cases = validate_cases([tuple(map(int, case.split(":"))) for case in args.cases.split(",")])
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.physical_npu < 0 or args.warmups < 2 or args.repeats < 5:
        parser.error("Require one nonnegative physical NPU, >=2 warmups and >=5 measured requests.")
    if not 1 <= args.max_model_len <= MAX_CONTEXT or any(sum(case) > args.max_model_len for case in args.cases):
        parser.error("Each case must fit --max-model-len, which must be in [1,128].")
    if args.kv_cache_mib <= 0:
        parser.error("--kv-cache-mib must be a positive integer.")
    if any(
        not math.isfinite(value) or not 0 < value <= 1
        for value in (args.memory_fraction, args.engine_memory_fraction)
        if value is not None
    ):
        parser.error("Memory fractions must be finite and in (0,1].")
    if (
        not math.isfinite(args.cache_reserve_gib)
        or args.cache_reserve_gib < max(1.0, args.kv_cache_mib / 1024)
        or not math.isfinite(args.cache_budget_gib)
        or args.cache_budget_gib < 0
    ):
        parser.error("Require finite reserve >=1 GiB covering KV, and budget >=0 GiB; no automatic reserve reduction.")
    if args.reference_only and args.reference_report:
        parser.error("A reference worker cannot consume another reference report.")
    if args.reference_only and args.device_route_decode:
        parser.error("Device-route decode is V4-only; the V1 reference must keep batched execution.")
    args.artifact = args.model / "experts_vq_ascend_v2"
    return args


def request_schedule(warmups, repeats, *, device_route_decode=False):
    """Warm both paths before alternating AB/BA; never rebuild/reload the engine."""
    modes = ("batched", "device_route_decode") if device_route_decode else ("batched",)
    for kind, count in (("warmup", warmups), ("measured", repeats)):
        for index in range(count):
            order = modes if index % 2 == 0 else tuple(reversed(modes))
            for optimization in order:
                yield kind, index, optimization


def measured_metrics(samples, case, optimization):
    measured = [
        sample
        for sample in samples
        if sample["case"] == case and sample["kind"] == "measured" and sample["optimization"] == optimization
    ]
    return {
        key: distribution([sample[key] for sample in measured])
        for key in ("ttft_s", "tpot_s", "e2e_s", "output_tokens_per_s", "device_span_ms")
    }


def check_device_route_activity(before, after, prompt_tokens, output_tokens):
    """Verify per-request counter deltas, not stale activity from a warmup."""
    expected_singletons = output_tokens - 1 + int(prompt_tokens == 1)
    expected_prefills = int(prompt_tokens > 1)
    if set(before) != {str(index) for index in range(LAYERS)} or set(after) != set(before):
        raise ValueError("Missing per-layer device-route activity evidence.")
    expected = {
        "singleton_forwards": expected_singletons,
        "batched_prefill_forwards": expected_prefills,
        "route_host_reads": expected_prefills,
        "device_select_calls": 2 * expected_singletons,
        "singleton_route_host_reads": 0,
        "singleton_descriptor_h2d_bytes": 0,
    }
    for layer, current in after.items():
        previous = before[layer]
        if current.get("preset") != "device_route_decode" or previous.get("preset") != "device_route_decode":
            raise ValueError("Device-route sample executed a different optimization preset.")
        for key, wanted in expected.items():
            left, right = previous.get(key), current.get(key)
            if any(type(value) is not int or value < 0 for value in (left, right)) or right - left != wanted:
                raise ValueError(f"Layer {layer} device-route counter {key} did not match this request.")
            if key.startswith("singleton_") and key != "singleton_forwards" and (left or right):
                raise ValueError(f"Layer {layer} device-route decode recorded a host transfer.")
    return {
        "layers": LAYERS,
        "per_layer_request_delta": expected,
        "transfer_scope": "runtime_path_counters_not_profiler_measured_DMA",
    }


def require_idle_device(physical_npu, log):
    """Read-only occupancy snapshot, not an exclusive device reservation."""
    from tools.diagnose_vq2a8_tp1_startup import parse_snapshot

    try:
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=20, check=True)
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
        state = parse_snapshot(result.stdout, physical_npu)
    except (OSError, subprocess.SubprocessError) as exc:
        log.write_text(str(exc), encoding="utf-8")
        state = "unknown"
    print(f"V4_DEVICE_SNAPSHOT physical_npu={physical_npu} state={state} LOG={log}", flush=True)
    if state != "idle":
        raise RuntimeError(
            f"Selected NPU {physical_npu} is {state}; no device work started. "
            "Stop only your own previous job and verify occupancy; no automatic kill or shared-device timing."
        )
    return {"physical_npu": physical_npu, "state": state, "log": str(log), "exclusive_reservation": False}


def check_no_payload_transfer(before, after):
    """Counters cover expert payload only, not routing metadata or all DMA."""
    for key in ("loads", "evictions"):
        values = before["cache"].get(key), after["cache"].get(key)
        if any(type(value) is not int or value < 0 for value in values) or values[0] != values[1]:
            raise ValueError(f"V4 request changed expert {key}; full residency was not maintained.")
    values = before.get("h2d_bytes"), after.get("h2d_bytes")
    if any(type(value) is not int or value < 0 for value in values) or values[0] != values[1]:
        raise ValueError("V4 request transferred expert payload to the device.")
    if before["cache"]["resident_packed_bytes"] != after["cache"]["resident_packed_bytes"]:
        raise ValueError("V4 resident payload size changed during the request.")


def check_residency(snapshot):
    from vllm_ascend.quantization.vq2a8_offline import validate_v4_residency_evidence

    validate_v4_residency_evidence(snapshot["cache"], snapshot.get("v4", {}), LAYERS)


def validate_sample(sample, tokens, count, *, resident):
    if sample.get("finite") is not True or sample.get("tokens") != tokens or sample.get("forwards") != count:
        raise ValueError("Request changed tokens, finite status or real forward count.")
    for key in ("native_calls", "native_launches"):
        if type(sample.get(key)) is not int or sample[key] <= 0:
            raise ValueError("Missing real native projection activity.")
    derived = token_metrics(sample["token_ready_s"], sample["e2e_s"], count)
    if any(
        not math.isclose(sample[key], derived[key], rel_tol=1e-9, abs_tol=1e-12)
        for key in ("ttft_s", "tpot_s", "output_tokens_per_s")
    ):
        raise ValueError("Reported performance disagrees with observed token timestamps.")
    distribution([sample["device_span_ms"]])
    if resident:
        for value in (
            sample["cache_delta"]["loads"],
            sample["cache_delta"]["evictions"],
            sample["expert_payload_h2d_bytes"],
        ):
            if type(value) is not int or value != 0:
                raise ValueError("V4 measured request loaded, evicted or transferred an expert payload.")


def diagnostic(llm, prompt, count, target, *, policy, vocab, library_sha256, optimization="batched"):
    import torch
    from safetensors.torch import save_file
    from vllm import SamplingParams

    from tools.benchmark_vq2a8_offline import configure, snapshot
    from tools.validate_vq2a8_tp1_offline import capture_worker_trace, reset_worker_trace, single_worker_result

    configure(llm, measurement=False, compact=True, optimization=optimization)
    before = snapshot(llm)
    if policy == V4_POLICY:
        check_residency(before)
    single_worker_result(llm.collective_rpc(reset_worker_trace))
    result = llm.generate(
        [{"prompt_token_ids": prompt}],
        SamplingParams(temperature=0, max_tokens=count, ignore_eos=True, detokenize=False),
        use_tqdm=False,
    )
    evidence = single_worker_result(llm.collective_rpc(capture_worker_trace))
    after = snapshot(llm)
    if policy == V4_POLICY:
        check_residency(after)
        check_no_payload_transfer(before, after)
    route_activity = None
    if optimization == "device_route_decode":
        route_activity = check_device_route_activity(before["optimization"], after["optimization"], len(prompt), count)
    if len(result) != 1 or not result[0].finished or len(result[0].outputs) != 1:
        raise ValueError("Incomplete or multiplexed diagnostic request.")
    tokens = list(result[0].outputs[0].token_ids)
    logits = evidence.pop("logits")
    if len(tokens) != count or logits.shape != (count, vocab) or after["finite"] is not True:
        raise ValueError("Missing diagnostic logits/tokens or deferred validity.")
    if any(type(token) is not int or not 0 <= token < vocab for token in tokens) or not bool(
        torch.isfinite(logits).all()
    ):
        raise ValueError("Invalid diagnostic token/logit values.")
    selected = logits[torch.arange(count), torch.tensor(tokens)]
    if not torch.equal(selected, logits.max(dim=1).values):
        raise ValueError("Greedy tokens disagree with captured logits.")
    expected = [{"tokens": len(prompt), "positions": list(range(len(prompt)))}]
    expected += [{"tokens": 1, "positions": [len(prompt) + index]} for index in range(count - 1)]
    backend = evidence["expert_backend"]
    calls = {int(index): value for index, value in evidence["cache"]["layer_calls"].items()}
    records = backend.get("layers", [])
    if (
        evidence["steps"] != expected
        or calls != dict.fromkeys(range(LAYERS), count)
        or backend.get("policy") != policy
        or backend.get("library", {}).get("sha256") != library_sha256
        or backend.get("fallback_enabled") is not False
        or len(records) != LAYERS
        or {record["layer"] for record in records} != set(range(LAYERS))
        or evidence["load"].get("moe_layers") != LAYERS
        or evidence["load"].get("registered_parameters_loaded", 0) <= 0
        or evidence.get("root_fp8", {}).get("mode") != "bf16"
    ):
        raise ValueError("Diagnostic lacks strict model load, real positions or complete policy/layer coverage.")
    for record in records:
        if len(record["steps"]) != count:
            raise ValueError("Missing native per-step coverage.")
        for actual, wanted in zip(record["steps"], expected):
            if (
                any(
                    type(actual.get(key)) is not int
                    for key in ("tokens", "expert_calls", "projection_calls", "projection_rows", "kernel_launches")
                )
                or actual["tokens"] != wanted["tokens"]
                or actual["expert_calls"] < 1
                or actual["projection_calls"] != 2 * actual["expert_calls"]
                or actual["projection_rows"] < 2 * wanted["tokens"]
                or actual["projection_rows"] % 2
                or not 2 <= actual["kernel_launches"] <= actual["projection_calls"]
                or actual["projection_calls"] > 6 * actual["kernel_launches"]
                or actual["kernel_launches"] % 2
            ):
                raise ValueError("Incomplete native gate/up/down execution.")
    save_file({"logits": logits.contiguous()}, str(target))
    record = {"tokens": tokens, "evidence": evidence, "logits_sha256": digest(target), "logits_file": target.name}
    if route_activity is not None:
        record["device_route_activity"] = route_activity
    write_json(target.with_suffix(".json"), record)
    return record, logits


def comparison_reference(path, identity):
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "PASS"
        or report.get("execution_policy") != "ascendc"
        or report.get("optimization") != "batched"
        or report.get("performance_measurement_verified") is not True
        or report.get("identity") != identity
    ):
        raise ValueError("Reference must be a completed same-input/library/device V1 batched run.")
    return report


def run(args):
    from tools.benchmark_vq2a8_offline import configure, preparation_preflight, snapshot, timed_request
    from tools.validate_vq2a8_ascendc import checked_model_preflight, require_hardware_runtime
    from tools.validate_vq2a8_tp1_acceptance import acceptance_environment
    from tools.validate_vq2a8_v023_environment import check_runtime_environment

    if os.environ.get("ASCEND_LAUNCH_BLOCKING") not in (None, "0"):
        raise ValueError("Timing requires ASCEND_LAUNCH_BLOCKING unset or 0.")
    environment = acceptance_environment(REPO, args.physical_npu, "npu:0")
    environment.update(ASCEND_LAUNCH_BLOCKING="0", VLLM_ENABLE_V1_MULTIPROCESSING="0")
    os.environ.clear()
    os.environ.update(environment)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "summary.json"
    policy = "ascendc" if args.reference_only else V4_POLICY
    optimization = "device_route_decode" if args.device_route_decode else "batched"
    report = dict(
        status="RUNNING",
        execution_policy=policy,
        optimization=optimization,
        scope=SCOPE,
        performance_measurement_verified=False,
        numerical_scope="repeatability_not_independent_model_accuracy",
        v1_comparison="NOT_RUN",
        preload_scope="startup_snapshot_contains_per_layer_preload_cost_not_request_timing",
        expert_transfer_scope="expert_payload_only_not_all_DMA_or_router_metadata",
        cases={},
        samples=[],
        serving_verified=False,
        quality_verified=False,
        full_model_graph_verified=False,
        performance_target_met=None,
        device_route_comparison="NOT_RUN",
    )
    write_json(path, report)
    try:
        report["device_snapshot"] = require_idle_device(args.physical_npu, output / "npu-before-init.log")
        report["environment"] = check_runtime_environment()
        print("VQ2A8_V023_ENVIRONMENT " + json.dumps(report["environment"]), flush=True)
        if report["environment"]["errors"]:
            raise RuntimeError(" ".join(report["environment"]["errors"]))
        require_hardware_runtime()
        library = checked_model_preflight(args.library, args.preflight)
        import torch
        import torch_npu  # noqa: F401
        from safetensors.torch import load_file
        from tokenizers import Tokenizer
        from vllm import LLM

        from tools.validate_vq2a8_qli_metadata import run_preflight
        from tools.validate_vq2a8_sas_attention import run_sas_preflight
        from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device
        from vllm_ascend.quantization.vq2a8_offline import offline_engine_options

        report["device"] = _initialize_device(torch.device("npu:0"))
        if torch.npu.get_device_name(0) != library["build"]["soc"]:
            raise ValueError("Library build SoC differs from the selected physical NPU.")
        root, artifact = args.model.resolve(strict=True), args.artifact.resolve(strict=True)
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if config["num_hidden_layers"] != LAYERS:
            raise ValueError("V4 tools require the complete 43-layer model.")
        identity = dict(
            inputs=capture_input_identity(root, artifact),
            model=str(root),
            artifact=str(artifact),
            library_sha256=library["sha256"],
            physical_npu=args.physical_npu,
            soc=library["build"]["soc"],
            sources={name: digest(REPO / name) for name in SOURCE_NAMES},
        )
        report["identity"], report["library"] = identity, library
        reference = comparison_reference(args.reference_report, identity) if args.reference_report else None
        tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        seed = tokenizer.encode(
            "The answer to 1 + 1 is. Read the following short example. ", add_special_tokens=False
        ).ids
        if not seed:
            raise ValueError("Empty benchmark seed.")
        prompts = {}
        for length, _ in args.cases:
            prefix = [config["bos_token_id"]] if isinstance(config.get("bos_token_id"), int) else []
            prompts[length] = prefix + (seed * (length // len(seed) + 1))[: length - len(prefix)]
        report["prompts"] = prompts
        report["preparation_preflight"] = preparation_preflight()
        run_preflight(torch.device("npu:0"), config, prompt_tokens=10)
        run_sas_preflight(torch.device("npu:0"), config, prompt_tokens=10)
        options = offline_engine_options(
            root,
            artifact,
            execution_policy=policy,
            cache_budget_gib=args.cache_budget_gib,
            cache_reserve_gib=args.cache_reserve_gib,
            ascendc_library=library["path"],
            ascendc_sha256=library["sha256"],
            **({"v4_device_route_decode": True} if args.device_route_decode else {}),
            **({"cache_memory_fraction": args.memory_fraction} if args.memory_fraction is not None else {}),
        )
        options.update(
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_model_len,
            kv_cache_memory_bytes=args.kv_cache_mib * 1024**2,
            gpu_memory_utilization=args.engine_memory_fraction,
        )
        report["engine_options"] = options
        write_json(path, report)
        started = time.perf_counter()
        llm = LLM(**options)
        report["engine_init_profile_kv_s"] = time.perf_counter() - started
        # Same point as V1: retain its original startup dummy geometry.
        configure(llm, measurement=False, compact=True, optimization="batched")
        report["startup_snapshot"] = snapshot(llm)
        if policy == V4_POLICY:
            check_residency(report["startup_snapshot"])
            resident = list(report["startup_snapshot"]["v4"].values())
            report["preload"] = {
                "layer_elapsed_sum_s": sum(layer["preload_elapsed_s"] for layer in resident),
                "loads": sum(layer["preload_loads"] for layer in resident),
                "expert_payload_h2d_bytes": sum(layer["preload_h2d_bytes"] for layer in resident),
                "evictions": sum(layer["preload_evictions"] for layer in resident),
                "scope": "sum_of_observed_layer_preload_costs_excludes_root_load_and_worker_profile_KV",
            }
        print(
            "V4_ENGINE_READY "
            + json.dumps(dict(policy=policy, engine_init_profile_kv_s=report["engine_init_profile_kv_s"])),
            flush=True,
        )
        write_json(path, report)
        for length, count in args.cases:
            case, prompt = f"p{length}-o{count}", prompts[length]
            entry = {"status": "RUNNING", "v1_comparison": "NOT_RUN", "prompt": prompt}
            report["cases"][case] = entry
            baseline_record = baseline_logits = None
            if args.device_route_decode:
                baseline_record, baseline_logits = diagnostic(
                    llm,
                    prompt,
                    count,
                    output / f"{case}-batched.safetensors",
                    policy=policy,
                    vocab=config["vocab_size"],
                    library_sha256=library["sha256"],
                    optimization="batched",
                )
                entry["baseline_diagnostic"] = baseline_record
            record, logits = diagnostic(
                llm,
                prompt,
                count,
                output / f"{case}.safetensors",
                policy=policy,
                vocab=config["vocab_size"],
                library_sha256=library["sha256"],
                optimization=optimization,
            )
            entry["diagnostic"] = record
            if args.device_route_decode:
                entry["device_route_comparison"] = numerical_gate(
                    baseline_record["tokens"], baseline_logits, record["tokens"], logits
                )
                if not entry["device_route_comparison"]["accepted"]:
                    raise ValueError(
                        "Device-route decode differs from same-engine batched logits/tokens; timing stopped."
                    )
            repeated, repeat_logits = diagnostic(
                llm,
                prompt,
                count,
                output / f"{case}-repeat.safetensors",
                policy=policy,
                vocab=config["vocab_size"],
                library_sha256=library["sha256"],
                optimization=optimization,
            )
            entry["repeat"] = numerical_gate(record["tokens"], logits, repeated["tokens"], repeat_logits)
            if not entry["repeat"]["accepted"]:
                raise ValueError("Repeated diagnostic tokens/logits are not bit-exact.")
            if reference is not None:
                previous = reference.get("cases", {}).get(case, {})
                if previous.get("status") != "PASS" or previous.get("prompt") != prompt:
                    raise ValueError("Reference lacks the same passing prompt/case.")
                old = previous["diagnostic"]
                old_path = args.reference_report.parent / f"{case}.safetensors"
                if digest(old_path) != old["logits_sha256"]:
                    raise ValueError("Reference logits changed after collection.")
                entry["v1_comparison"] = numerical_gate(
                    old["tokens"], load_file(str(old_path))["logits"], record["tokens"], logits
                )
                if not entry["v1_comparison"]["accepted"]:
                    raise ValueError("V4 differs from measured V1 batched logits/tokens; timing stopped.")
            write_json(path, report)
            for kind, index, sample_optimization in request_schedule(
                args.warmups, args.repeats, device_route_decode=args.device_route_decode
            ):
                configure(llm, measurement=True, compact=True, optimization=sample_optimization)
                before = snapshot(llm)
                if policy == V4_POLICY:
                    check_residency(before)
                sample = timed_request(llm, prompt, count, f"{policy}-{case}-{sample_optimization}-{kind}-{index}")
                sample.update(
                    case=case, kind=kind, repeat=index, execution_policy=policy, optimization=sample_optimization
                )
                report["samples"].append(sample)
                write_json(path, report)
                after = snapshot(llm)
                if policy == V4_POLICY:
                    check_residency(after)
                    check_no_payload_transfer(before, after)
                validate_sample(sample, record["tokens"], count, resident=policy == V4_POLICY)
                if sample_optimization == "device_route_decode":
                    sample["device_route_activity"] = check_device_route_activity(
                        sample["optimization_before"], sample["optimization_after"], len(prompt), count
                    )
                print(
                    "V4_SAMPLE "
                    + json.dumps(
                        {
                            key: sample[key]
                            for key in (
                                "case",
                                "kind",
                                "repeat",
                                "execution_policy",
                                "optimization",
                                "ttft_s",
                                "tpot_s",
                                "e2e_s",
                                "cache_delta",
                                "expert_payload_h2d_bytes",
                            )
                        }
                    ),
                    flush=True,
                )
            entry["metrics"] = measured_metrics(report["samples"], case, optimization)
            if args.device_route_decode:
                entry["baseline_metrics"] = measured_metrics(report["samples"], case, "batched")
                entry["tpot_ratio_vs_same_engine_batched"] = (
                    entry["metrics"]["tpot_s"]["median"] / entry["baseline_metrics"]["tpot_s"]["median"]
                )
            entry["status"] = "PASS"
            write_json(path, report)
        if (
            capture_input_identity(root, artifact) != identity["inputs"]
            or digest(args.library) != library["sha256"]
            or any(digest(REPO / name) != value for name, value in identity["sources"].items())
        ):
            raise ValueError("Inputs/library/source changed during the run.")
        report.update(
            status="PASS",
            performance_measurement_verified=True,
            v1_comparison="PASS" if reference is not None else "NOT_RUN",
            device_route_comparison="PASS" if args.device_route_decode else "NOT_RUN",
        )
        return 0
    except KeyboardInterrupt:
        report.update(status="INTERRUPTED", error="Interrupted by user.", performance_measurement_verified=False)
        raise
    except Exception as exc:
        report.update(
            status="FAIL", error=str(exc), traceback=traceback.format_exc(), performance_measurement_verified=False
        )
        print(f"V4_ERROR={exc}", flush=True)
        return 1
    finally:
        write_json(path, report)
        print(
            f"V4_BENCHMARK_STATUS={report['status']} POLICY={policy} "
            f"V1_COMPARISON={report['v1_comparison']} REPORT={path}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
