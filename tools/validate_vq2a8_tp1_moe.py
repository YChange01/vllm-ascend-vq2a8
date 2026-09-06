#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One isolated real-weight MoE layer check; launch via acceptance.py --stage moe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.validate_vq2a8_tp1_packed_kernel import (
    _comparison_summary,
    _initialize_device,
    _synchronize,
    activation_case,
    environment_report,
)
from vllm_ascend.quantization.vq2a8_moe import VQ2TP1MoE
from vllm_ascend.quantization.vq2a8_runtime import open_vq2a8_tp1_artifact
from vllm_ascend.quantization.vq2a8_validation import audit_model_storage


@torch.inference_mode()
def run_case(cpu: VQ2TP1MoE, runtime: VQ2TP1MoE, count: int, case: str, warmups: int, repeats: int) -> dict:
    label = f"layer{runtime.layer_index} case={case}:m={count}"
    # Keep the same amplitude as the accepted expert cases; do not weaken
    # the MoE gate by silently shrinking inputs at the integration boundary.
    hidden = torch.cat(
        [activation_case(cpu.config.hidden_size, row + runtime.layer_index, 0, case) for row in range(count)]
    )
    tokens = torch.arange(count, dtype=torch.int64) % cpu.config.vocab_size
    device_hidden = hidden.to(runtime.device)
    device_tokens = tokens.to(runtime.device)
    print(f"MOE {label} stage=router", flush=True)
    expected_weights, expected_ids = cpu.route(hidden, tokens)
    actual_weights, actual_ids = runtime.route(device_hidden, device_tokens)
    if not torch.equal(expected_ids, actual_ids.cpu()):
        raise AssertionError(f"Router ID mismatch: cpu={expected_ids.tolist()} device={actual_ids.cpu().tolist()}")
    router_comparison = _comparison_summary(expected_weights, actual_weights, rtol=1e-4, atol=1e-6)
    print(
        "ROUTER_RESULT "
        + json.dumps(
            {
                "case": label,
                "ids": actual_ids.cpu().tolist(),
                "weights": actual_weights.cpu().tolist(),
                "comparison": router_comparison,
            }
        ),
        flush=True,
    )
    print(f"MOE {label} stage=cpu_oracle", flush=True)
    expected = cpu.forward(hidden, tokens)
    print(f"MOE {label} stage=packed_forward", flush=True)
    actual = runtime.forward(device_hidden, device_tokens)
    _synchronize(runtime.device)
    comparison = _comparison_summary(expected, actual, rtol=0.03, atol=0.05)
    print(f"MOE {label} stage=token_chunk_invariance", flush=True)
    singles = torch.cat(
        [runtime.forward(device_hidden[row : row + 1], device_tokens[row : row + 1]) for row in range(count)]
    )
    chunk_comparison = _comparison_summary(actual, singles, rtol=0.03, atol=0.05)
    print(f"MOE {label} stage=repeat", flush=True)
    determinism = None
    for _ in range(warmups + repeats):
        repeated = runtime.forward(device_hidden, device_tokens)
        _synchronize(runtime.device)
        determinism = _comparison_summary(actual, repeated, rtol=0, atol=0)
    backend = torch.npu if runtime.device.type == "npu" else torch.cuda
    result = {
        "layer": runtime.layer_index,
        "case": f"{case}:m={count}",
        "token_count": count,
        "comparison": comparison,
        "router_comparison": router_comparison,
        "token_chunk_comparison": chunk_comparison,
        "determinism": determinism,
        "repeats_checked": repeats,
        "shared_experts": runtime.config.num_shared,
        "routed_scale": runtime.config.routed_scale,
        "routed_scale_owner": "mix_vq2a8_routes",
        "tp_reductions": 0,
        "cache": runtime.cache_stats(),
        "cache_expert_limit": runtime.cache_experts,
        "token_chunk_limit": runtime.token_chunk,
        "device_dense_expert_weight_materialized": False,
        "peak_device_allocated_bytes": backend.max_memory_allocated(),
        "peak_device_reserved_bytes": backend.max_memory_reserved(),
        "native_fp8_dot": runtime.device.type == "cuda",
        "serving_integration_verified": False,
    }
    print("MOE_RESULT " + json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--token-counts", type=int, nargs="+", default=[1, 3])
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["deterministic", "zero"],
        choices=["deterministic", "zero", "impulse", "small", "large"],
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--allow-partial-artifact", action="store_true")
    parser.add_argument("--audit-model", action="store_true")
    parser.add_argument("--verify-tensor-hashes", action="store_true")
    args = parser.parse_args()
    if args.layer < 0 or args.warmups < 0 or args.repeats < 1 or any(count < 1 for count in args.token_counts):
        parser.error("Invalid layer, token count, warmup or repeat count.")
    print("ENVIRONMENT " + json.dumps(environment_report()), flush=True)
    device = torch.device(args.device)
    print("DEVICE " + json.dumps(_initialize_device(device)), flush=True)
    artifact = open_vq2a8_tp1_artifact(
        args.artifact,
        args.model / "config.json",
        require_complete=not args.allow_partial_artifact,
        require_reference_identity=True,
        verify_tensor_hashes=args.verify_tensor_hashes,
    )
    print(
        "ARTIFACT_RESULT "
        + json.dumps(
            {
                "root": str(artifact.root),
                "layer": args.layer,
                "complete": artifact.manifest["complete"],
                "format": artifact.manifest["format"],
                "tensor_hashes_verified": args.verify_tensor_hashes,
            }
        ),
        flush=True,
    )
    if args.audit_model:
        print("MODEL_AUDIT " + json.dumps(audit_model_storage(artifact)), flush=True)
    cpu = VQ2TP1MoE(artifact, args.layer, "cpu", cache_experts=2, token_chunk=2)
    runtime = VQ2TP1MoE(artifact, args.layer, device, cache_experts=2, token_chunk=2)
    for case in args.cases:
        for count in args.token_counts:
            run_case(cpu, runtime, count, case, args.warmups, args.repeats)
    print(
        "VQ2A8_TP1_MOE_GATE=PASS "
        + json.dumps(
            {
                "layer": args.layer,
                "cases": args.cases,
                "token_counts": args.token_counts,
                "native_fp8_dot": device.type == "cuda",
                "serving_integration_verified": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
