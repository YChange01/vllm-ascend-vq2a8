#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded real-model eager/decoder-graph correctness test, not a timing result.

Runs on one idle NPU in a new worker. Exercises 3->4, 7->8, 11->12 compressor
boundaries, changing token IDs and repeated request/cache-slot reuse. Both modes
use the *same* resident payload. Graph-only candidates retain their independent
eager reference, including the original Torch SwiGLU for candidate I.
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
from tools.vq2a8_candidate_options import add_candidate_arguments, validate_candidate_args

CASES = ((1, 4), (3, 4), (7, 4), (11, 4), (12, 4), (1, 15))
REUSE_ROUNDS = 2
SWIGLU_REFERENCE = "deepseek_v4_swiglu_reference"
SWIGLU_SCOPE = "graph_build_only_eager_and_prefill_torch"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, help="expert artifact; defaults to MODEL/experts_vq_ascend_v2")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--physical-npu", type=int, default=1)
    parser.add_argument("--compute-backend", choices=("v1", "v2"), default="v2")
    parser.add_argument(
        "--activation-reorder",
        choices=("scalar", "vectorized", "row_reuse", "chunk_reuse2", "chunk_reuse4"),
        default="scalar",
    )
    parser.add_argument(
        "--activation-preparation",
        choices=("rowwise", "rowwise_packed", "sign_fused", "sign_fused_strided", "sign_fused_direct", "fused"),
        default="rowwise",
    )
    parser.add_argument(
        "--decoder-metadata-mode",
        choices=("recursive", "planned", "planned_fast", "position_template"),
        default="recursive",
    )
    parser.add_argument("--validity-mode", choices=("torch", "fused", "fused_vectorized"), default="torch")
    add_candidate_arguments(parser)
    parser.add_argument("--route-mapping", choices=("torch", "fused"), default="torch")
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
    try:
        validate_candidate_args(args, graph_mode="decoder", device_route=True)
    except ValueError as error:
        parser.error(str(error))
    if args.route_mapping == "fused" and args.compute_backend != "v2":
        parser.error("Fused route mapping requires --compute-backend v2.")
    if args.validity_mode in ("fused", "fused_vectorized") and (
        args.compute_backend != "v2"
        or args.activation_preparation not in ("sign_fused", "sign_fused_strided", "sign_fused_direct")
    ):
        parser.error("Fused validity requires V4 v2 native sign preparation.")
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
    model = worker.get_model()
    model.set_v4_decoder_input_enabled(enabled)
    return model.set_v4_graph_enabled(enabled)


def graph_report(worker):
    return worker.get_model().v4_graph_report()


def input_switch(worker, enabled):
    worker.get_model().set_v4_decoder_input_enabled(enabled)


def set_template_verification(worker, enabled):
    return worker.get_model().set_v4_position_template_verification(enabled)


def require_template_evidence(report):
    evidence = report.get("decoder", {}).get("position_template", {})
    expected = {p + step for p, n in CASES for step in range(n - 1)}
    if (
        evidence.get("reference_verification_enabled") is not True
        or evidence.get("reference_checks", 0) < REUSE_ROUNDS * sum(n - 1 for _, n in CASES)
        or not expected.issubset(set(evidence.get("reference_positions", [])))
        or evidence.get("original_builder_skips", 0) < REUSE_ROUNDS * sum(n - 1 for _, n in CASES)
    ):
        raise AssertionError("Missing position-template original-builder metadata equivalence evidence.")


def swiglu_eager_evidence(before, after):
    """Require an actual original-Torch eager pass, not startup counter reuse."""
    for report in (before, after):
        evidence = report.get("swiglu_candidates", {})
        if (
            report.get("swiglu_mode") != "fused_select_sign"
            or report.get("effective_graph_mode") != "none"
            or evidence.get("scope") != SWIGLU_SCOPE
            or evidence.get("eager_reference") != SWIGLU_REFERENCE
            or evidence.get("counters_prove_device_execution") is not False
            or any(
                type(evidence.get(key)) is not int or evidence[key] < 0
                for key in ("graph_build_calls", "reference_calls")
            )
            or type(report.get("decoder", {}).get("replays")) is not int
            or report["decoder"]["replays"] < 0
        ):
            raise AssertionError("Missing I original-Torch eager-reference report with graphs disabled.")
    deltas = {
        key: after["swiglu_candidates"][key] - before["swiglu_candidates"][key]
        for key in ("graph_build_calls", "reference_calls")
    }
    replays = after["decoder"]["replays"] - before["decoder"]["replays"]
    if deltas["reference_calls"] < 1 or deltas["graph_build_calls"] != 0 or replays != 0:
        raise AssertionError("I reference pass must call original Torch SwiGLU without graph build or replay.")
    return {
        "implementation": SWIGLU_REFERENCE,
        "graph_disabled": True,
        **deltas,
        "decoder_replays": replays,
    }


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


def verify_template_serving_path(llm, prompt, params, reference, unwrap):
    """Compare a real producer-skip run; shadow calls must not mask side effects."""
    unwrap(llm.collective_rpc(set_template_verification, args=(False,)))
    before = unwrap(llm.collective_rpc(graph_report))["decoder"]
    actual = llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)[0].outputs[0]
    after = unwrap(llm.collective_rpc(graph_report))["decoder"]
    delta = after["replays"] - before["replays"]
    skips = after["position_template"]["original_builder_skips"] - before["position_template"]["original_builder_skips"]
    if delta != len(reference.token_ids) - 1 or skips != delta:
        raise AssertionError("Template serving-path replay did not skip the original builder.")
    error = compare_outputs(reference, actual)
    unwrap(llm.collective_rpc(set_template_verification, args=(True,)))
    return error


def verify_input_serving_path(llm, prompt, params, reference, unwrap, template=False):
    """Use the same decoder graph for general -> packed -> general requests.

    Disable template shadow calls: their side effects must not repair a broken
    candidate. Each request changes/reuses live KV rows through normal vLLM.
    """
    rows, maximum_error = [], 0.0
    if template:
        unwrap(llm.collective_rpc(set_template_verification, args=(False,)))
    try:
        for name, enabled in (("general_before", False), ("packed", True), ("general_after", False)):
            unwrap(llm.collective_rpc(input_switch, args=(enabled,)))
            before = unwrap(llm.collective_rpc(graph_report))["decoder"]
            actual = llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)[0].outputs[0]
            after = unwrap(llm.collective_rpc(graph_report))["decoder"]
            delta = after["replays"] - before["replays"]
            counters = {
                key: after["decoder_input"][key] - before["decoder_input"][key]
                for key in ("fastpath_calls", "packed_uploads", "grouped_slot_calls")
            }
            if delta != len(reference.token_ids) - 1 or any(
                value != (delta if enabled else 0) for value in counters.values()
            ):
                raise AssertionError("Input serving-path gate did not execute the requested general/packed path.")
            if template:
                skips = (
                    after["position_template"]["original_builder_skips"]
                    - before["position_template"]["original_builder_skips"]
                )
                if skips != delta:
                    raise AssertionError("Input serving-path gate unexpectedly executed the shadow builder.")
            maximum_error = max(maximum_error, compare_outputs(reference, actual))
            rows.append({"mode": name, "replays": delta, **counters, "template_shadow_enabled": False})
    finally:
        unwrap(llm.collective_rpc(input_switch, args=(True,)))
        if template:
            unwrap(llm.collective_rpc(set_template_verification, args=(True,)))
    return maximum_error, rows


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
        v4_validity_mode=args.validity_mode,
        v4_route_mapping=args.route_mapping,
        v4_runtime_guard=args.runtime_guard,
        v4_select_sign=args.select_sign,
        v4_activation_tail=args.activation_tail,
        v4_b1_schedule=args.b1_schedule,
        v4_swiglu_mode=getattr(args, "swiglu_mode", "torch"),
        v4_decode_graph="decoder",
        v4_graph_replay_stream="caller",
        v4_decoder_metadata_mode=args.decoder_metadata_mode,
        v4_decoder_input_mode=args.decoder_input_mode,
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
    if args.decoder_metadata_mode == "position_template":
        single_worker_result(llm.collective_rpc(set_template_verification, args=(True,)))
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
                eager_before = single_worker_result(llm.collective_rpc(graph_switch, args=(False,)))
                reference = llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)[0].outputs[0]
                swiglu_reference = None
                if getattr(args, "swiglu_mode", "torch") == "fused_select_sign":
                    eager_after = single_worker_result(llm.collective_rpc(graph_report))
                    swiglu_reference = swiglu_eager_evidence(eager_before, eager_after)
            with stage(f"round{round_id}_p{prompt_length}_o{output_length}_decoder"):
                before = single_worker_result(llm.collective_rpc(graph_switch, args=(True,)))
                candidate = llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)[0].outputs[0]
                after = single_worker_result(llm.collective_rpc(graph_report))
                delta = after["decoder"]["replays"] - before["decoder"]["replays"]
                if delta != output_length - 1 or len(candidate.token_ids) != output_length:
                    raise AssertionError("The requested output count or actual decoder replay count differs.")
                error = compare_outputs(reference, candidate)
                input_hits = 0
                input_comparison = []
                if args.decoder_input_mode == "b1_packed":
                    input_hits = (
                        after["decoder"]["decoder_input"]["fastpath_calls"]
                        - before["decoder"]["decoder_input"]["fastpath_calls"]
                    )
                    if input_hits < 1:
                        raise AssertionError("Packed decoder input candidate was requested but never executed.")
                if args.decoder_metadata_mode == "position_template":
                    # The shadow builder could conceal a missing side effect.
                    # Also exercise actual serving behavior without calling it.
                    with stage(f"round{round_id}_p{prompt_length}_o{output_length}_template_no_shadow"):
                        error = max(
                            error, verify_template_serving_path(llm, prompt, params, reference, single_worker_result)
                        )
                if args.decoder_input_mode == "b1_packed":
                    with stage(f"round{round_id}_p{prompt_length}_o{output_length}_input_general_packed_general"):
                        input_error, input_comparison = verify_input_serving_path(
                            llm,
                            prompt,
                            params,
                            reference,
                            single_worker_result,
                            template=args.decoder_metadata_mode == "position_template",
                        )
                        error = max(error, input_error)
                rows.append(
                    {
                        "round": round_id,
                        "prompt": prompt_length,
                        "output": output_length,
                        "max_logprob_error": error,
                        "decoder_replays": delta,
                        "swiglu_eager_reference": swiglu_reference,
                        "input_fastpath_calls": input_hits,
                        "input_serving_comparison": input_comparison,
                    }
                )
    report = single_worker_result(llm.collective_rpc(graph_report))
    if args.decoder_metadata_mode == "position_template":
        require_template_evidence(report)
    receipt = {
        "status": "PASS",
        "scope": "real_model_eager_vs_position_specialized_decoder",
        "hardware_execution_verified": True,
        "timing_valid": False,
        "cases": rows,
        "graph": report,
        "validity_mode": args.validity_mode,
        "route_mapping": args.route_mapping,
        "runtime_guard": args.runtime_guard,
        "select_sign": args.select_sign,
        "activation_tail": args.activation_tail,
        "activation_reorder": args.activation_reorder,
        "b1_schedule": args.b1_schedule,
        "swiglu_mode": getattr(args, "swiglu_mode", "torch"),
        "swiglu_reference": SWIGLU_REFERENCE,
        "projection_reference": "vectorized_baseline"
        if args.activation_reorder in ("chunk_reuse2", "chunk_reuse4") or args.b1_schedule != "baseline"
        else "selected_backend",
        "decoder_input_mode": args.decoder_input_mode,
        "decoder_metadata_mode": args.decoder_metadata_mode,
        "library_sha256": options["additional_config"]["vq2a8_offline"]["ascendc_sha256"],
    }
    validate_receipt(args, receipt)
    (output / "summary.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("V4_DECODER_GRAPH=PASS SUMMARY=" + str(output / "summary.json"), flush=True)
    emit("v4_decoder_graph", "CASE_PASS", scope=receipt["scope"])


def validate_receipt(args, receipt):
    """A successful subprocess must prove the requested modes and all replays."""
    case_replays = REUSE_ROUNDS * sum(count - 1 for _, count in CASES)
    expected_replays = case_replays
    if args.decoder_metadata_mode == "position_template":
        expected_replays *= 2  # Shadow equivalence plus actual producer-skip path.
    if getattr(args, "decoder_input_mode", "general") == "b1_packed":
        expected_replays += 3 * case_replays
    if (
        receipt.get("status") != "PASS"
        or receipt.get("hardware_execution_verified") is not True
        or receipt.get("route_mapping") != args.route_mapping
        or receipt.get("graph", {}).get("route_mapping") != args.route_mapping
        or any(
            receipt.get(name) != getattr(args, name) or receipt.get("graph", {}).get(name) != getattr(args, name)
            for name in ("runtime_guard", "select_sign", "activation_tail", "validity_mode")
        )
        or [(row.get("round"), row.get("prompt"), row.get("output")) for row in receipt.get("cases", [])]
        != [(round_id, prompt, output) for round_id in range(REUSE_ROUNDS) for prompt, output in CASES]
        or receipt.get("graph", {}).get("decoder", {}).get("replays") != expected_replays
    ):
        raise ValueError("Device probe receipt does not confirm all requested modes and real decoder replays.")
    if args.decoder_metadata_mode == "position_template":
        require_template_evidence(receipt.get("graph", {}))
    if args.runtime_guard == "native":
        native_calls = receipt.get("graph", {}).get("decoder", {}).get("runtime_guard_native_calls")
        if type(native_calls) is not int or native_calls != expected_replays:
            raise ValueError("Missing one native runtime guard check per decoder replay.")
    input_mode = getattr(args, "decoder_input_mode", "general")
    if input_mode != "general":
        inputs = receipt.get("graph", {}).get("decoder", {}).get("decoder_input") or {}
        if (
            receipt.get("decoder_input_mode") != input_mode
            or receipt.get("graph", {}).get("decoder_input_mode") != input_mode
            or inputs.get("mode") != input_mode
            or inputs.get("enabled") is not True
            or inputs.get("scope") != "block_table_upload_and_grouped_slot_mapping"
            or inputs.get("general_prepare_inputs_preserved") is not True
            or inputs.get("dynamic_rows_cached") is not False
            or inputs.get("packed_uploads", 0) < 1
            or inputs.get("grouped_slot_calls", 0) < 1
            or any(
                type(row.get("input_fastpath_calls")) is not int or row["input_fastpath_calls"] < 1
                for row in receipt.get("cases", [])
            )
        ):
            raise ValueError("Missing executed packed decoder input evidence against general-input reference.")
        for row in receipt["cases"]:
            comparisons = row.get("input_serving_comparison", [])
            if len(comparisons) != 3:
                raise ValueError("Missing general/packed/general serving comparisons.")
            for evidence, (name, enabled) in zip(
                comparisons, (("general_before", False), ("packed", True), ("general_after", False))
            ):
                expected = row["output"] - 1
                if (
                    evidence.get("mode") != name
                    or evidence.get("replays") != expected
                    or evidence.get("template_shadow_enabled") is not False
                    or any(
                        type(evidence.get(key)) is not int or evidence[key] != (expected if enabled else 0)
                        for key in ("fastpath_calls", "packed_uploads", "grouped_slot_calls")
                    )
                ):
                    raise ValueError("Incomplete general/packed/general serving-path evidence.")
    if getattr(args, "activation_reorder", "scalar") == "row_reuse":
        if (
            receipt.get("activation_reorder") != "row_reuse"
            or receipt.get("graph", {}).get("activation_reorder") != "row_reuse"
        ):
            raise ValueError("Missing requested row_reuse activation reorder evidence.")
    if (
        getattr(args, "activation_reorder", "scalar") in ("chunk_reuse2", "chunk_reuse4")
        or getattr(args, "b1_schedule", "baseline") != "baseline"
    ):
        graph = receipt.get("graph", {})
        evidence = graph.get("projection_candidates", {})
        if (
            receipt.get("activation_reorder") != args.activation_reorder
            or graph.get("activation_reorder") != args.activation_reorder
            or receipt.get("b1_schedule") != args.b1_schedule
            or graph.get("b1_schedule") != args.b1_schedule
            or receipt.get("projection_reference") != "vectorized_baseline"
            or evidence.get("scope") != "graph_build_only_eager_and_prefill_vectorized"
            or evidence.get("counters_prove_device_execution") is not False
            or any(
                type(evidence.get(key)) is not int or evidence[key] < 1
                for key in ("graph_build_calls", "reference_calls")
            )
        ):
            raise ValueError("Missing J/K graph candidate and independent vectorized reference evidence.")
    swiglu_mode = getattr(args, "swiglu_mode", "torch")
    if (
        receipt.get("swiglu_mode", "torch") != swiglu_mode
        or receipt.get("graph", {}).get("swiglu_mode", "torch") != swiglu_mode
    ):
        raise ValueError("Missing requested I SwiGLU mode evidence.")
    if swiglu_mode == "fused_select_sign":
        graph = receipt.get("graph", {})
        evidence = graph.get("swiglu_candidates", {})
        if (
            any(
                graph.get(key) != value
                for key, value in (
                    ("requested_graph_mode", "decoder"),
                    ("compute_backend", "v2"),
                    ("activation_preparation", "sign_fused_direct"),
                    ("select_sign", "fused"),
                    ("activation_tail", "torch"),
                    ("activation_reorder", "vectorized"),
                    ("b1_schedule", "baseline"),
                )
            )
            or receipt.get("swiglu_reference") != SWIGLU_REFERENCE
            or evidence.get("eager_reference") != SWIGLU_REFERENCE
            or evidence.get("scope") != SWIGLU_SCOPE
            or evidence.get("counters_prove_device_execution") is not False
            or any(
                type(evidence.get(key)) is not int or evidence[key] < 1
                for key in ("graph_build_calls", "reference_calls")
            )
        ):
            raise ValueError("Missing I graph candidate and independent original-Torch reference evidence.")
        for row in receipt["cases"]:
            reference = row.get("swiglu_eager_reference") or {}
            if (
                reference.get("implementation") != SWIGLU_REFERENCE
                or reference.get("graph_disabled") is not True
                or type(reference.get("reference_calls")) is not int
                or reference["reference_calls"] < 1
                or any(
                    type(reference.get(key)) is not int or reference[key] != 0
                    for key in ("graph_build_calls", "decoder_replays")
                )
                or type(row.get("decoder_replays")) is not int
                or row["decoder_replays"] != row["output"] - 1
            ):
                raise ValueError("Missing per-case I independent original-Torch eager/decoder replay evidence.")
        if (
            sum(row["swiglu_eager_reference"]["reference_calls"] for row in receipt["cases"])
            > evidence["reference_calls"]
        ):
            raise ValueError("I per-case reference evidence exceeds recorded original-Torch calls.")


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
                    "b1_schedule": args.b1_schedule,
                    "swiglu_mode": getattr(args, "swiglu_mode", "torch"),
                    "activation_preparation": args.activation_preparation,
                    "validity_mode": args.validity_mode,
                    "route_mapping": args.route_mapping,
                    "runtime_guard": args.runtime_guard,
                    "select_sign": args.select_sign,
                    "activation_tail": args.activation_tail,
                    "decoder_metadata_mode": args.decoder_metadata_mode,
                    "decoder_input_mode": args.decoder_input_mode,
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
    validate_receipt(args, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
