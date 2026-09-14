#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only phase-1 validation benchmark; no model construction or NPU access."""

from __future__ import annotations

# Direct scripts must not put tools/bisect ahead of the stdlib bisect module.
# ruff: noqa: E402
import os as _bootstrap_os
import sys as _bootstrap_sys

if not __package__:
    _bootstrap_sys.path[0] = _bootstrap_os.path.dirname(
        _bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))
    )

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from vllm_ascend.quantization.vq2a8_artifact import VQ2_CODEBOOK_SIZE
from vllm_ascend.quantization.vq2a8_repack import unpack_repacked_indices, validate_repacked_matrix
from vllm_ascend.quantization.vq2a8_runtime import open_vq2a8_tp1_artifact


def legacy_validation_replay(payload, spec):
    """Replay the removed full-grid work in addition to all retained checks.

    This is a validation microbenchmark, not execution of an old git checkout
    or a claim about cold disk throughput. The four-bit range test is retained
    here solely to measure the previous redundant work.
    """
    validate_repacked_matrix(payload, spec)
    unpacked = unpack_repacked_indices(payload["packed_indices"], spec.columns)
    if bool((unpacked >= VQ2_CODEBOOK_SIZE).any()):
        raise ValueError("Invalid code")


def benchmark_projection(artifact, layer, expert, kind, repeats):
    read = {}
    # Both validators below remain mandatory. Disable only the loader's
    # duplicate validation so each can be timed over the same CPU payload.
    payload, spec = artifact.load_expert(layer, expert, kind, validate_payload=False, timings=read)
    before = {key: value.view(torch.uint8).clone() for key, value in payload.items()}
    checks = {"legacy_replay": legacy_validation_replay, "current": validate_repacked_matrix}
    for check in checks.values():
        check(payload, spec)
    timings = {name: [] for name in checks}
    for repeat in range(repeats):
        order = list(checks) if repeat % 2 == 0 else list(reversed(checks))
        for name in order:
            start = time.perf_counter()
            checks[name](payload, spec)
            timings[name].append(time.perf_counter() - start)
    if not all(torch.equal(value.view(torch.uint8), before[key]) for key, value in payload.items()):
        raise AssertionError("Validation mutated the payload")
    return {
        "projection": f"{layer}:{expert}:{kind}",
        "payload_unchanged": True,
        "read_s": read["host_read_s"],
        "validation_median_s": {name: statistics.median(values) for name, values in timings.items()},
        "validation_samples_s": timings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--probes", default="0:0,3:0,42:255")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, help="New file; existing reports are refused.")
    args = parser.parse_args()
    if args.repeats < 1 or (args.output is not None and args.output.exists()):
        parser.error("Repeats must be positive and output must be a new path.")
    probes = []
    for probe in args.probes.split(","):
        pieces = probe.split(":")
        if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
            parser.error("Probes must be layer:expert pairs.")
        probes.append(tuple(map(int, pieces)))
    artifact = open_vq2a8_tp1_artifact(args.artifact or args.model / "experts_vq_ascend_v2", args.model / "config.json")
    report = {
        "scope": "cpu_validation_microbenchmark_warm_payload_not_npu_or_cold_disk_performance",
        "cpu_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "results": [],
    }
    print("HOST_BENCHMARK_START " + json.dumps(report), flush=True)
    for layer, expert in probes:
        for kind in ("gate_up", "down"):
            print(f"HOST_PROBE_START={layer}:{expert}:{kind}", flush=True)
            result = benchmark_projection(artifact, layer, expert, kind, args.repeats)
            report["results"].append(result)
            print("HOST_PROBE_RESULT " + json.dumps(result), flush=True)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
    print("HOST_VALIDATION_CHECK=PASS NPU_PERFORMANCE_VERIFIED=False", flush=True)


if __name__ == "__main__":
    main()
