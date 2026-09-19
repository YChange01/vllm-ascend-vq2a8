#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Start vLLM HTTP serving with V4 TP1 full residency and a selected compute backend.

No build, repack, version audit, acceptance receipt or numerical preflight.
Confirm the selected NPU is available before starting this persistent server.
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
import shlex
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAX_CONTEXT = 128
DECODER_GRAPH_MAX_CONTEXT = 16
KV_BYTES = 1024**3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/home/g00872988/vq2a8"))
    parser.add_argument(
        "--artifact",
        type=Path,
        help="Expert artifact directory; defaults to MODEL/experts_vq_ascend_v2. Prepacked experts require backend v2",
    )
    parser.add_argument("--library", type=Path, default=REPO / "build/vq2a8-ascendc-v023-v1/libvq2a8_ascendc.so")
    parser.add_argument(
        "--compute-backend",
        choices=("v1", "v2"),
        default="v1",
        help="V4 compute kernel: v1 preserves the baseline; v2 needs libvq2a8_ascendc_v4_v2.so (not old v2/v3)",
    )
    parser.add_argument("--physical-npu", type=int, default=1)
    parser.add_argument(
        "--activation-reorder",
        choices=("scalar", "vectorized", "row_reuse"),
        default="scalar",
        help="V4 v2 FP8-byte reorder: scalar, vectorized, or experimental M1 row_reuse (new library required)",
    )
    parser.add_argument(
        "--activation-preparation",
        choices=("rowwise", "rowwise_packed", "sign_fused", "sign_fused_strided", "sign_fused_direct", "fused"),
        default="rowwise",
        help="V4 v2 activation candidate; strided/direct require a new library and separate numerical acceptance",
    )
    parser.add_argument("--validity-mode", choices=("torch", "fused", "fused_vectorized"), default="torch")
    parser.add_argument("--runtime-guard", choices=("signature", "planned", "native"), default="signature")
    parser.add_argument("--decoder-input-mode", choices=("general", "b1_packed"), default="general")
    parser.add_argument("--select-sign", choices=("separate", "fused"), default="separate")
    parser.add_argument("--activation-tail", choices=("torch", "fused_reorder"), default="torch")
    parser.add_argument(
        "--route-mapping",
        choices=("torch", "fused"),
        default="torch",
        help="Opt-in exact integer expert-ID to resident-slot mapping; requires rebuilt V4 v2 library",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=MAX_CONTEXT,
        help="Total input plus output token limit, in [1,128]; also bounds startup profile tokens",
    )
    parser.add_argument(
        "--kv-cache-mib",
        type=int,
        default=KV_BYTES // 1024**2,
        help="Explicit KV cache budget in MiB (positive integer; default: 1024)",
    )
    parser.add_argument("--memory-fraction", type=float, default=0.9, help="Expert residency budget fraction")
    parser.add_argument("--engine-memory-fraction", type=float, default=0.9)
    parser.add_argument("--reserve-gib", type=float, default=8.0)
    parser.add_argument(
        "--device-route-decode",
        action="store_true",
        help="Opt in to V4 single-token device routing; requires the rebuilt native library (prefill stays batched)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the command without importing or running vLLM/NPU"
    )
    parser.add_argument(
        "--decode-graph",
        choices=("none", "moe", "decoder"),
        default="none",
        help="Opt-in single-token graph: moe or position-specialized decoder (max length 16); prefill stays eager",
    )
    parser.add_argument(
        "--graph-replay-stream",
        choices=("owner", "caller"),
        default="owner",
        help="MoE graph replay stream: owner preserves the baseline; caller removes per-layer event bridges",
    )
    parser.add_argument(
        "--decoder-metadata-mode",
        choices=("recursive", "planned", "planned_fast", "position_template"),
        default="recursive",
        help="Decoder metadata baseline, compiled checks, or opt-in position-owned producer templates",
    )
    parser.add_argument(
        "--host-profile",
        action="store_true",
        help="Diagnostic CPU ranges/counters without device fences; disable for latency benchmarks",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="Opt-in torch-NPU profiler export directory, controlled by /start_profile and /stop_profile",
    )
    args = parser.parse_args(argv)
    if (
        args.runtime_guard != "signature"
        or args.select_sign != "separate"
        or args.activation_tail != "torch"
        or args.decoder_input_mode != "general"
    ) and (args.compute_backend != "v2" or not args.device_route_decode):
        parser.error("ABCD candidates require V4 v2 device-route decode.")
    if args.runtime_guard in ("planned", "native") and args.decode_graph == "none":
        parser.error("Planned runtime guard requires MoE or decoder graphs.")
    if args.decoder_input_mode != "general" and args.decode_graph != "decoder":
        parser.error("Packed decoder input requires --decode-graph decoder.")
    if (
        args.select_sign == "fused" or args.activation_tail == "fused_reorder"
    ) and args.activation_preparation != "sign_fused_direct":
        parser.error("Select/sign and tail candidates require sign_fused_direct preparation.")
    if args.activation_tail == "fused_reorder" and args.activation_reorder != "vectorized":
        parser.error("Fused tail requires vectorized activation reorder.")
    if args.decode_graph != "none" and not args.device_route_decode:
        parser.error("--decode-graph moe/decoder requires --device-route-decode.")
    if args.graph_replay_stream == "caller" and args.decode_graph == "none":
        parser.error("--graph-replay-stream caller requires --decode-graph moe or decoder.")
    if args.decode_graph == "decoder" and (
        args.graph_replay_stream != "caller" or args.max_model_len > DECODER_GRAPH_MAX_CONTEXT
    ):
        parser.error("--decode-graph decoder requires --graph-replay-stream caller and --max-model-len <=16.")
    if args.decoder_metadata_mode != "recursive" and args.decode_graph != "decoder":
        parser.error("Non-recursive metadata requires --decode-graph decoder.")
    if args.validity_mode in ("fused", "fused_vectorized") and (
        args.compute_backend != "v2"
        or not args.device_route_decode
        or args.activation_preparation not in ("sign_fused", "sign_fused_strided", "sign_fused_direct")
    ):
        parser.error("Fused validity requires V4 v2 device-route decode with native sign preparation.")
    if args.route_mapping == "fused" and (args.compute_backend != "v2" or not args.device_route_decode):
        parser.error("Fused route mapping requires --compute-backend v2 and --device-route-decode.")
    if args.host_profile and args.decode_graph != "decoder":
        parser.error("--host-profile requires --decode-graph decoder.")
    if (
        args.activation_preparation in ("rowwise_packed", "sign_fused", "sign_fused_strided", "sign_fused_direct")
        and not args.device_route_decode
    ):
        parser.error("Packed decode preparation requires --device-route-decode; prefill keeps rowwise arithmetic.")
    if (
        args.activation_reorder != "scalar" or args.activation_preparation != "rowwise"
    ) and args.compute_backend != "v2":
        parser.error("Activation optimizations require --compute-backend v2.")
    if args.physical_npu < 0 or not 1 <= args.port <= 65535:
        parser.error("Require a nonnegative physical NPU and port in [1,65535].")
    if not args.host or any(character.isspace() for character in args.host):
        parser.error("Require a nonempty host without whitespace.")
    if not 1 <= args.max_model_len <= MAX_CONTEXT:
        parser.error(f"--max-model-len must be in [1,{MAX_CONTEXT}].")
    if args.kv_cache_mib <= 0:
        parser.error("--kv-cache-mib must be a positive integer.")
    if any(
        not math.isfinite(value) or not 0 < value <= 1 for value in (args.memory_fraction, args.engine_memory_fraction)
    ):
        parser.error("Memory fractions must be finite and in (0,1].")
    if not math.isfinite(args.reserve_gib) or args.reserve_gib < max(1.0, args.kv_cache_mib / 1024):
        parser.error(
            "Reserve must be finite, at least 1 GiB and no smaller than the requested KV cache; "
            "it is never reduced automatically."
        )
    return args


def build_command(args):
    model, library = args.model.resolve(strict=True), args.library.resolve(strict=True)
    artifact = (args.artifact if args.artifact is not None else model / "experts_vq_ascend_v2").resolve(strict=True)
    if not model.is_dir() or not artifact.is_dir():
        raise ValueError("Model and expert artifact must be directories.")
    if library.suffix != ".so" or not library.is_file():
        raise ValueError("--library must be an existing native .so file for the selected V4 compute backend.")
    if args.compute_backend == "v2" and library.name != "libvq2a8_ascendc_v4_v2.so":
        raise ValueError(
            "--compute-backend v2 requires libvq2a8_ascendc_v4_v2.so; old V2/V3 libraries are not compatible."
        )
    if args.compute_backend == "v1" and library.name == "libvq2a8_ascendc_v4_v2.so":
        raise ValueError("Select --compute-backend v2 for libvq2a8_ascendc_v4_v2.so.")
    # Pin the file used by the runtime, without requiring a build manifest or
    # an acceptance receipt. Actual ABI/weight/device checks remain in loader.
    sha256 = hashlib.sha256(library.read_bytes()).hexdigest()
    additional = {
        "enable_flashcomm1": False,
        "mix_placement": False,
        "multistream_dsv4_dsa_overlap": False,
        "vq2a8_offline": {
            "enabled": True,
            "artifact": str(artifact),
            "execution_policy": "ascendc_v4",
            "ascendc_library": str(library),
            "ascendc_sha256": sha256,
            "cache_experts": 256,
            "token_chunk": 2,
            "cache_memory_fraction": args.memory_fraction,
            "cache_reserve_gib": args.reserve_gib,
            "root_linear_mode": "bf16",
            "v4_serving": True,
        },
    }
    if args.device_route_decode:
        additional["vq2a8_offline"]["v4_device_route_decode"] = True
    if args.compute_backend != "v1":
        additional["vq2a8_offline"]["v4_compute_backend"] = args.compute_backend
    if args.activation_reorder != "scalar":
        additional["vq2a8_offline"]["v4_activation_reorder"] = args.activation_reorder
    if args.activation_preparation != "rowwise":
        additional["vq2a8_offline"]["v4_activation_preparation"] = args.activation_preparation
    if args.validity_mode != "torch":
        additional["vq2a8_offline"]["v4_validity_mode"] = args.validity_mode
    if args.route_mapping != "torch":
        additional["vq2a8_offline"]["v4_route_mapping"] = args.route_mapping
    for name, default in (("runtime_guard", "signature"), ("select_sign", "separate"), ("activation_tail", "torch")):
        if getattr(args, name) != default:
            additional["vq2a8_offline"]["v4_" + name] = getattr(args, name)
    if args.decode_graph != "none":
        additional["vq2a8_offline"]["v4_decode_graph"] = args.decode_graph
        additional["vq2a8_offline"]["v4_graph_replay_stream"] = args.graph_replay_stream
    if args.decoder_metadata_mode != "recursive":
        additional["vq2a8_offline"]["v4_decoder_metadata_mode"] = args.decoder_metadata_mode
    if args.decoder_input_mode != "general":
        additional["vq2a8_offline"]["v4_decoder_input_mode"] = args.decoder_input_mode
    if args.host_profile:
        additional["vq2a8_offline"]["v4_host_profile"] = True
    overrides = {"architectures": ["VQ2A8TP1OfflineForCausalLM"], "quantization_config": None}
    command = [
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
        "1",
        "--pipeline-parallel-size",
        "1",
        "--distributed-executor-backend",
        "uni",
        "--enforce-eager",
        "--compilation-config",
        json.dumps({"mode": 0, "cudagraph_mode": "NONE"}),
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--max-num-seqs",
        "1",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_model_len),
        "--block-size",
        "128",
        "--kv-cache-memory-bytes",
        str(args.kv_cache_mib * 1024**2),
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
    if args.profile_dir is not None:
        command.extend(
            [
                "--profiler-config",
                json.dumps(
                    {
                        "profiler": "torch",
                        "torch_profiler_dir": str(args.profile_dir.resolve()),
                        "torch_profiler_with_stack": False,
                    }
                ),
            ]
        )
    return command


def server_environment(args, environ=None):
    environment = dict(os.environ if environ is None else environ)
    for key in (
        "ASCEND_VISIBLE_DEVICES",
        "NPU_VISIBLE_DEVICES",
        "ASCEND_DEVICE_ID",
        "DEVICE_ID",
        "RANK_ID",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(key, None)
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.physical_npu)
    environment["ASCEND_LAUNCH_BLOCKING"] = "0"
    # Standard AsyncLLM owns an engine-core child; do not inherit the offline
    # benchmark's in-process worker setting.
    environment["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    environment["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = str(REPO) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    return environment


def main(argv=None):
    args = parse_args(argv)
    try:
        command = build_command(args)
        if args.dry_run:
            print(f"V4_SERVER_DRY_RUN physical_npu={args.physical_npu} no_device_execution=True", flush=True)
            print(shlex.join(command), flush=True)
        else:
            print(
                f"Starting vllm serve at http://{args.host}:{args.port} "
                f"(V4 TP1, device selector {args.physical_npu}, "
                f"compute_backend={args.compute_backend}, "
                f"activation_reorder={args.activation_reorder}, activation_preparation={args.activation_preparation}, "
                f"decode={'device_route_decode' if args.device_route_decode else 'batched'}, "
                f"decode_graph={args.decode_graph}, graph_replay_stream={args.graph_replay_stream}, full residency). "
                f"metadata_mode={args.decoder_metadata_mode}, validity_mode={args.validity_mode}, "
                f"route_mapping={args.route_mapping}, "
                f"runtime_guard={args.runtime_guard}, select_sign={args.select_sign}, "
                f"activation_tail={args.activation_tail}, decoder_input_mode={args.decoder_input_mode}, "
                f"host_profile={args.host_profile}. "
                "Confirm this card is available; no other jobs are stopped.",
                flush=True,
            )
            os.execvpe(command[0], command, server_environment(args))
        return 0
    except (OSError, ValueError) as exc:
        print(f"V4 server: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
