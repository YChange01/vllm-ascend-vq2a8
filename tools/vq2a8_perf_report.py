# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only report contracts for bounded offline performance measurements."""

from __future__ import annotations

import math
import statistics

DEFAULT_CASES = ((10, 4), (32, 32), (96, 32))
MAX_CONTEXT = 128


def validate_cases(cases):
    if not cases or len(cases) > 12 or len(set(cases)) != len(cases):
        raise ValueError("Require 1..12 distinct offline cases.")
    for prompt, output in cases:
        if type(prompt) is not int or type(output) is not int or min(prompt, output) < 2:
            raise ValueError("Prompt/output lengths must be integers >=2.")
        if prompt + output > MAX_CONTEXT:
            raise ValueError("This offline runner supports at most 128 total tokens; no serving/long-context claim.")
    return cases


def distribution(values):
    if not values or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Timing samples must be finite and positive.")
    ordered = sorted(values)
    return {
        "n": len(values),
        "min": ordered[0],
        "median": statistics.median(values),
        "p95_nearest_rank": ordered[math.ceil(0.95 * len(values)) - 1],
        "max": ordered[-1],
        "tail_sample_warning": len(values) < 100,
    }


def token_metrics(ready_s, elapsed_s, output_tokens):
    if len(ready_s) != output_tokens or output_tokens < 2:
        raise ValueError("Expected one observed delivery timestamp per output token.")
    if not all(math.isfinite(v) and v > 0 for v in [*ready_s, elapsed_s]):
        raise ValueError("Invalid token timing.")
    if ready_s != sorted(ready_s) or ready_s[-1] > elapsed_s or ready_s[-1] <= ready_s[0]:
        raise ValueError("Non-monotonic token timing.")
    return {
        "ttft_s": ready_s[0],
        "tpot_s": (ready_s[-1] - ready_s[0]) / (output_tokens - 1),
        "e2e_s": elapsed_s,
        "output_tokens_per_s": output_tokens / elapsed_s,
        "decode_intervals_s": [b - a for a, b in zip(ready_s, ready_s[1:])],
    }


def summarize_performance(samples, cases, repeats, warmups, regressions):
    validate_cases(cases)
    if repeats < 5 or warmups < 2:
        raise ValueError("Require >=2 warmups and >=5 measured requests per variant/case.")
    groups = []
    for prompt, output in cases:
        case = f"p{prompt}-o{output}"
        for variant in ("baseline", "compact"):
            rows = [s for s in samples if s["case"] == case and s["variant"] == variant and s["kind"] == "measured"]
            warm = [s for s in samples if s["case"] == case and s["variant"] == variant and s["kind"] == "warmup"]
            if len(rows) != repeats or len(warm) != warmups:
                raise ValueError(f"Incomplete sample matrix: {case}/{variant}.")
            if sorted(r["repeat"] for r in rows) != list(range(repeats)) or sorted(r["repeat"] for r in warm) != list(
                range(warmups)
            ):
                raise ValueError("Repeated/missing sample indices cannot fill the measurement matrix.")
            for row in rows + warm:
                if row.get("finite") is not True or row.get("tokens_exact") is not True:
                    raise ValueError(f"Invalid numerical result: {case}/{variant}.")
                if len(row.get("tokens", [])) != output or row.get("forwards") != output:
                    raise ValueError("Missing tokens or unexpected dummy/chunked forward.")
                if len(row.get("token_ready_s", [])) != output:
                    raise ValueError("Missing per-token observations.")
                derived = token_metrics(row["token_ready_s"], row["e2e_s"], output)
                if any(
                    not math.isclose(row[k], derived[k], rel_tol=1e-9, abs_tol=1e-12)
                    for k in ("ttft_s", "tpot_s", "output_tokens_per_s")
                ):
                    raise ValueError("Reported metrics disagree with raw token observations.")
            groups.append(
                {
                    "case": case,
                    "variant": variant,
                    "metrics": {
                        key: distribution([s[key] for s in rows])
                        for key in ("ttft_s", "tpot_s", "e2e_s", "output_tokens_per_s", "device_span_ms")
                    },
                    "cache_loads": sum(s["cache_delta"]["loads"] for s in rows),
                    "evictions": sum(s["cache_delta"]["evictions"] for s in rows),
                }
            )
    expected_cases = {f"p{p}-o{o}" for p, o in cases}
    if (
        len(regressions) != len(cases)
        or {r.get("case") for r in regressions} != expected_cases
        or any(r.get("logits_exact") is not True or r.get("tokens_exact") is not True for r in regressions)
    ):
        raise ValueError("Every case requires a passing baseline/compact diagnostic logits regression.")
    ratios = []
    for case in (f"p{p}-o{o}" for p, o in cases):
        left, right = [g for g in groups if g["case"] == case]
        resident = not any(g["cache_loads"] or g["evictions"] for g in (left, right))
        ratios.append(
            {
                "case": case,
                "warm_resident_comparison": resident,
                "baseline_over_compact_median": left["metrics"]["e2e_s"]["median"]
                / right["metrics"]["e2e_s"]["median"],
                "interpretation": "same-process alternating resident observations"
                if resident
                else "includes cache misses/evictions; not an isolated compute speedup",
            }
        )
    return {
        "status": "PASS",
        "performance_measurement_verified": True,
        "performance_target_met": None,
        "scope": "tp1_offline_single_request_max_context_128_not_serving",
        "groups": groups,
        "ratios": ratios,
        "quality_verified": False,
        "serving_verified": False,
        "excluded": ["HTTP/client latency", "concurrency", "context >128", "quality", "long-running stability"],
    }
