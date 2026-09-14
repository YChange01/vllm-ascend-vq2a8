#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Start standard vllm serve with VQ2A8 V3 defaults for short B1 experiments.

No compilation, acceptance reports, numerical preflights or benchmark loop.
The server keeps its model loaded for repeated /v1/completions requests.
"""

import argparse
import hashlib
import json
import math
import os
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAX_CONTEXT = 128
KV_BYTES = 1024**3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--library", type=Path, default=REPO / "build/vq2a8-ascendc-v3/libvq2a8_ascendc_v3.so")
    parser.add_argument("--physical-npu", type=int, help="TP1 device (default: 0)")
    parser.add_argument("--physical-npus", help="two distinct physical NPU IDs for TP2, e.g. 0,1")
    parser.add_argument("--tensor-parallel-size", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--artifact", type=Path, help="default: MODEL/experts_vq_ascend_v2 (TP1) or experts_vq_tp2_zn (TP2)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preparation", choices=("eager", "fused"), default="eager")
    parser.add_argument("--decode-graph", choices=("none", "moe"), default="none")
    parser.add_argument("--memory-fraction", type=float, default=1.0)
    parser.add_argument("--engine-memory-fraction", type=float, default=0.98)
    parser.add_argument("--reserve-gib", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true", help="Print the vllm serve command without starting it")
    args = parser.parse_args(argv)
    if (args.physical_npu is not None and args.physical_npu < 0) or not 1 <= args.port <= 65535:
        parser.error("Require a nonnegative NPU index and port in [1,65535]")
    if args.tensor_parallel_size == 2:
        if args.physical_npu is not None:
            parser.error("TP2 uses --physical-npus, not --physical-npu")
        try:
            selection = "0,1" if args.physical_npus is None else args.physical_npus
            devices = tuple(int(part) for part in selection.split(","))
        except ValueError:
            parser.error("--physical-npus must contain two distinct nonnegative integers")
        if len(devices) != 2 or len(set(devices)) != 2 or min(devices) < 0:
            parser.error("--physical-npus must contain two distinct nonnegative integers")
        args.physical_npus = ",".join(map(str, devices))
        if args.decode_graph != "none":
            parser.error("TP2 requires --decode-graph none")
    elif args.physical_npus is not None:
        parser.error("--physical-npus requires --tensor-parallel-size 2")
    if args.physical_npu is None:
        args.physical_npu = 0
    if any(not math.isfinite(v) or not 0 < v <= 1 for v in (args.memory_fraction, args.engine_memory_fraction)):
        parser.error("Memory fractions must be in (0,1]")
    if not math.isfinite(args.reserve_gib) or args.reserve_gib < KV_BYTES / 1024**3:
        parser.error("Reserve must include the 1 GiB KV cache")
    return args


def build_command(args):
    model, library = args.model.resolve(strict=True), args.library.resolve(strict=True)
    default_artifact = "experts_vq_tp2_zn" if args.tensor_parallel_size == 2 else "experts_vq_ascend_v2"
    artifact = (args.artifact or model / default_artifact).resolve(strict=True)
    if not model.is_dir() or not artifact.is_dir():
        raise ValueError(f"--model and the {default_artifact} artifact must be directories")
    if library.suffix != ".so" or not library.is_file():
        raise ValueError("--library must point to the compiled V3 .so")
    # Runtime's pinned-library loader needs this digest, but no build manifest
    # or acceptance receipt is required for a quick serving experiment.
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    additional = {
        "enable_flashcomm1": False,
        "mix_placement": False,
        "multistream_dsv4_dsa_overlap": False,
        "vq2a8_offline": {
            "enabled": True,
            "artifact": str(artifact),
            "execution_policy": "ascendc_v3",
            "ascendc_v3_library": str(library),
            "ascendc_v3_sha256": digest,
            "cache_experts": 256,
            "token_chunk": 2,
            "cache_memory_fraction": args.memory_fraction,
            "cache_reserve_gib": args.reserve_gib,
            "root_linear_mode": "bf16",
            "v3_preparation": args.preparation,
            "v3_decode_graph": args.decode_graph,
            "v3_serving": True,
        },
    }
    overrides = {
        "architectures": [f"VQ2A8TP{args.tensor_parallel_size}OfflineForCausalLM"],
        "quantization_config": None,
    }
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model),
        "--served-model-name",
        "vq2a8",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--load-format",
        "safetensors",
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--pipeline-parallel-size",
        "1",
        "--distributed-executor-backend",
        "mp" if args.tensor_parallel_size == 2 else "uni",
        "--enforce-eager",
        "--compilation-config",
        json.dumps({"mode": 0, "cudagraph_mode": "NONE"}),
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--max-num-seqs",
        "1",
        "--max-model-len",
        str(MAX_CONTEXT),
        "--max-num-batched-tokens",
        str(MAX_CONTEXT),
        "--block-size",
        "128",
        "--kv-cache-memory-bytes",
        str(KV_BYTES),
        "--gpu-memory-utilization",
        str(args.engine_memory_fraction),
        "--stream-interval",
        "1",
        "--seed",
        "0",
        "--disable-log-stats",
        "--generation-config",
        "vllm",
        "--hf-overrides",
        json.dumps(overrides),
        "--additional-config",
        json.dumps(additional),
    ]


def server_environment(args):
    environment = os.environ.copy()
    for key in (
        "ASCEND_VISIBLE_DEVICES",
        "NPU_VISIBLE_DEVICES",
        "ASCEND_DEVICE_ID",
        "DEVICE_ID",
        "RANK_ID",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
    ):
        environment.pop(key, None)
    environment["ASCEND_RT_VISIBLE_DEVICES"] = (
        args.physical_npus if args.tensor_parallel_size == 2 else str(args.physical_npu)
    )
    environment["ASCEND_LAUNCH_BLOCKING"] = "0"
    # AsyncLLM's standard server uses an engine-core process, unlike the offline
    # LLM benchmark's in-process worker. Keep all NPU work inside that process.
    environment["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    environment["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    environment["PYTHONPATH"] = str(REPO) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    return environment


def main():
    args = parse_args()
    try:
        command = build_command(args)
        if args.dry_run:
            print(shlex.join(command), flush=True)
        else:
            print(
                f"Starting vllm serve at http://{args.host}:{args.port} "
                f"(V3 TP={args.tensor_parallel_size}, preparation={args.preparation}, "
                f"decode_graph={args.decode_graph})",
                flush=True,
            )
            os.execvpe(command[0], command, server_environment(args))
        return 0
    except (OSError, ValueError) as exc:
        print(f"V3 server: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
