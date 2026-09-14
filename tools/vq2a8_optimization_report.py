# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered optimization probes. No speed/quality/serving or ISA claims."""

import json
import traceback
from pathlib import Path
from types import SimpleNamespace

from tools.profile_vq2a8_ascendc import digest, write_json
from tools.vq2a8_perf_report import distribution


def graph_captures(report):
    return sum(layer.get("graph", {}).get("captures", 0) for layer in report.values())


def numerical_gate(left_tokens, left, right_tokens, right):
    import torch

    compatible = left.shape == right.shape and left.dtype == right.dtype
    finite = bool(torch.isfinite(left).all() & torch.isfinite(right).all())
    exact = compatible and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
    return {
        "accepted": finite and exact and left_tokens == right_tokens,
        "tokens_exact": left_tokens == right_tokens,
        "logits_bit_exact": exact,
        "finite": finite,
        "max_abs_error": float((left - right).abs().max()) if compatible and finite else None,
        "independent_reference": False,
        "contract": "same_library_baseline_exact_no_implicit_tolerance",
    }


def candidate_preflight(presets, output, library):
    """Exercise new entries before loading the model; persist each finished case.

    Pipeline must match baseline exactly. FWHT differences are recorded, then
    subjected to the model's strict logits gate. Graph replay must match its
    eager FWHT implementation even after expert metadata/input changes.
    """
    import torch

    from tools.validate_vq2a8_phase4_kernel import bitwise_equal, synthetic_inputs
    from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
    from vllm_ascend.quantization.vq2a8_activation_fast import BatchedFWHTPreparation, PreparationGraph
    from vllm_ascend.quantization.vq2a8_ascendc import (
        grouped_projection,
        grouped_projection_pipeline,
        load_library,
    )

    load_library(library)
    path = output / "optimization-preflight.json"
    report = {"status": "RUNNING", "cases": [], "device_execution_verified": False}
    write_json(path, report)
    try:
        if any(name in presets for name in ("pipeline", "prepare_graph")):
            for n, k in ((64, 512), (64, 2048), (64, 4096), (4096, 2048), (2048, 4096)):
                inputs = [
                    tuple(t.to("npu:0") for t in synthetic_inputs(m, n, k, tiles))
                    for m, tiles in ((1, 1), (2, 3), (15, 32), (16, 256), (17, 3), (32, 32))
                ]
                baseline = grouped_projection(inputs)
                for repeat in range(3):
                    ordered = inputs if repeat % 2 == 0 else list(reversed(inputs))
                    actual = grouped_projection_pipeline(ordered)
                    expected = baseline if repeat % 2 == 0 else list(reversed(baseline))
                    if not all(bitwise_equal(a, b) for a, b in zip(actual, expected)):
                        raise ValueError(f"Pipeline mismatch K={k}, N={n}, repeat={repeat}")
                report["cases"].append(dict(stage="pipeline", n=n, k=k, jobs=6, repeat_permutation_exact=True))
                write_json(path, report)
        if any(name in presets for name in ("fwht", "prepare_graph")):
            for width in (512, 2048, 4096):
                valid = []
                fwht = BatchedFWHTPreparation(validity=valid.append)
                graph = PreparationGraph(fwht)
                reference = RowwiseVQ2A8Preparation(compact=True)
                for rows in ((1, 1), (1, 2, 15, 16, 17, 32)):
                    requests = []
                    with torch.device("cpu"):
                        generator = torch.Generator().manual_seed(width)
                        for i, m in enumerate(rows):
                            x = torch.randn(m, width - 32, generator=generator).bfloat16().to("npu:0")
                            p = dict(
                                weight_scale=torch.randn(width, generator=generator).to("npu:0"),
                                weight_bias=torch.randn(width, generator=generator).to("npu:0"),
                                rht_sign=torch.where((torch.arange(width) + i) % 3 == 0, -1, 1)
                                .to(torch.int8)
                                .to("npu:0"),
                            )
                            requests.append(
                                (x, p, SimpleNamespace(columns=width, rht_true_columns=width - 32, rht_block_size=128))
                            )
                    expected, actual = reference.many(requests), fwht.many(requests)
                    exact = all(bitwise_equal(a, b) for lhs, rhs in zip(expected, actual) for a, b in zip(lhs, rhs))
                    if not all(bool(torch.isfinite(t.float()).all()) for result in actual for t in result):
                        raise ValueError("FWHT produced nonfinite output")
                    if "prepare_graph" in presets and all(m == 1 for m in rows):
                        for iteration in range(3):
                            # Same geometry but changing BOTH activations and
                            # per-expert metadata: catch stale captured weights.
                            changed = [
                                (x + iteration, requests[-1 - i][1], spec) for i, (x, _, spec) in enumerate(requests)
                            ]
                            eager, replay = fwht.many(changed), graph.many(changed)
                            if not all(
                                bitwise_equal(a, b) for lhs, rhs in zip(eager, replay) for a, b in zip(lhs, rhs)
                            ):
                                raise ValueError("Preparation graph differs after input/expert metadata change")
                    if not all(bool(flag) for flag in valid):
                        raise ValueError("FWHT deferred validity failed")
                    report["cases"].append(
                        dict(stage="fwht", width=width, rows=rows, baseline_bit_exact=exact, graph=graph.report())
                    )
                    write_json(path, report)
        report.update(status="PASS", device_execution_verified=bool(report["cases"]))
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        write_json(path, report)
    return report


def validate_sample(sample, tokens, count, *, measured):
    if sample["finite"] is not True or sample["tokens"] != tokens or sample["forwards"] != count:
        raise ValueError("Optimization measurement changed tokens/finite/forward coverage")
    if measured and graph_captures(sample["optimization_after"]) != graph_captures(sample["optimization_before"]):
        raise ValueError("Graph capture contaminated a measured sample; extend warmup")


def collect_profile(llm, prompt, count, preset, output):
    import torch_npu

    from tools.benchmark_vq2a8_offline import configure, timed_request

    target = output / f"profile-{preset}-p{len(prompt)}-o{count}"
    target.mkdir(exist_ok=False)
    result = {"status": "RUNNING", "scope": "separate_untimed_profile_not_performance_sample"}
    try:
        configure(llm, measurement=True, compact=False, optimization=preset, profile=True)
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,
            with_stack=True,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(target)),
        ) as profiler:
            sample = timed_request(llm, prompt, count, f"profile-{preset}")
            profiler.step()
        if sample["finite"] is not True:
            raise ValueError("Nonfinite profile request")
        result.update(status="PASS", native_launches=sample["native_launches"])
    except Exception as exc:
        result.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
    finally:
        configure(llm, measurement=True, compact=False, optimization=preset)
        write_json(target / "profile-status.json", result)
    return result


def run_cases(llm, args, manifest, prompts, output):
    from tools.benchmark_vq2a8_offline import configure, diagnostic, timed_request

    path = output / "summary.json"
    manifest.update(
        optimization_presets=args.optimization_presets,
        candidates=[],
        profiles={},
        performance_target_met=None,
        quality="NOT_REQUESTED",
        serving="NOT_REQUESTED",
        full_model_graph_verified=False,
        native_instruction_verified=False,
    )
    repo = Path(__file__).resolve().parents[1]
    source_names = [
        "vllm_ascend/quantization/" + name
        for name in (
            "vq2a8_optimization.py",
            "vq2a8_activation.py",
            "vq2a8_activation_fast.py",
            "vq2a8_activation_triton.py",
            "vq2a8_execution.py",
            "vq2a8_moe.py",
            "vq2a8_ascendc.py",
        )
    ] + [
        "tools/vq2a8_optimization_report.py",
        "tools/benchmark_vq2a8_offline.py",
        "vllm_ascend/patch/worker/vq2a8_offline_model.py",
    ]
    manifest["optimization_source_sha256"] = {name: digest(repo / name) for name in source_names}
    for length, count in args.cases:
        case = f"p{length}-o{count}"
        prompt = prompts[length]
        left_tokens, left = diagnostic(llm, prompt, count, False, output / f"{case}-baseline.safetensors")
        for preset in args.optimization_presets:
            print(f"OPTIMIZATION_PHASE case={case} preset={preset} stage=exact_regression", flush=True)
            right_tokens, right = diagnostic(
                llm, prompt, count, False, output / f"{case}-{preset}.safetensors", optimization=preset
            )
            gate = numerical_gate(left_tokens, left, right_tokens, right)
            entry = dict(case=case, preset=preset, numerical=gate, status="RUNNING", samples=[], distributions={})
            manifest["candidates"].append(entry)
            write_json(path, manifest)
            if not gate["accepted"]:
                entry["status"] = "REJECTED_NUMERICAL"
                print(f"OPTIMIZATION_REJECTED case={case} preset={preset} reason=baseline_logits_not_exact", flush=True)
                write_json(path, manifest)
                continue  # report other candidates; never present rejected speed as accepted
            # Verify repeat exact separately, outside timed/measurement mode.
            repeat_tokens, repeated = diagnostic(
                llm, prompt, count, False, output / f"{case}-{preset}-repeat.safetensors", optimization=preset
            )
            entry["repeat"] = numerical_gate(right_tokens, right, repeat_tokens, repeated)
            if not entry["repeat"]["accepted"]:
                raise ValueError(f"Non-deterministic optimization {preset}")
            for kind, repeats in (("warmup", args.warmups), ("measured", args.repeats)):
                for index in range(repeats):
                    for variant in (None, preset) if index % 2 == 0 else (preset, None):
                        configure(llm, measurement=True, compact=False, optimization=variant)
                        label = variant or "baseline"
                        sample = timed_request(llm, prompt, count, f"{case}-{preset}-{label}-{kind}-{index}")
                        sample.update(variant=label, kind=kind, repeat=index)
                        entry["samples"].append(sample)
                        validate_sample(sample, left_tokens, count, measured=kind == "measured")
                        write_json(path, manifest)
                        print(
                            "OPTIMIZATION_SAMPLE "
                            + json.dumps(
                                dict(
                                    case=case,
                                    preset=preset,
                                    **{
                                        key: sample[key]
                                        for key in ("variant", "kind", "repeat", "ttft_s", "tpot_s", "cache_delta")
                                    },
                                )
                            ),
                            flush=True,
                        )
            for label in ("baseline", preset):
                rows = [s for s in entry["samples"] if s["kind"] == "measured" and s["variant"] == label]
                entry["distributions"][label] = {
                    metric: distribution([s[metric] for s in rows])
                    for metric in ("ttft_s", "tpot_s", "e2e_s", "output_tokens_per_s")
                }
            entry["status"] = "PASS"
            # One actual prefill+decode trace per passing preset, outside timing.
            if args.profile_optimization:
                manifest["profiles"][f"{case}/{preset}"] = collect_profile(llm, prompt, count, preset, output)
            write_json(path, manifest)
    expected = len(args.cases) * len(args.optimization_presets)
    manifest["optimization_sources_unchanged"] = all(
        digest(repo / name) == original for name, original in manifest["optimization_source_sha256"].items()
    )
    if not manifest["optimization_sources_unchanged"]:
        raise ValueError("Optimization sources changed during the measurement matrix")
    passed = len(manifest["candidates"]) == expected and all(c["status"] == "PASS" for c in manifest["candidates"])
    profiles_ok = all(p["status"] == "PASS" for p in manifest["profiles"].values())
    manifest.update(
        status="PASS" if passed and profiles_ok else "REVIEW_REQUIRED", performance_measurement_verified=passed
    )
    lines = [
        f"OPTIMIZATION_STATUS={manifest['status']}",
        "SCOPE=TP1_OFFLINE_CONTEXT_LE_128",
        "PERFORMANCE_TARGET_MET=null NO_SPEED_THRESHOLD",
        "FULL_MODEL_GRAPH_VERIFIED=False",
        "QUALITY=NOT_REQUESTED SERVING=NOT_REQUESTED",
    ]
    for entry in manifest["candidates"]:
        lines.append(f"CASE={entry['case']} PRESET={entry['preset']} STATUS={entry['status']}")
        for label, dist in entry["distributions"].items():
            lines.append(
                f"  {label}: TTFT_S={dist['ttft_s']['median']:.6f} "
                f"TPOT_S={dist['tpot_s']['median']:.6f} E2E_S={dist['e2e_s']['median']:.6f}"
            )
    # Artifact output, not source editing.
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
