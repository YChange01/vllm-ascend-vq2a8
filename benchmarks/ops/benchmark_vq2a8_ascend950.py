# SPDX-License-Identifier: Apache-2.0
"""Validate and benchmark production-shape VQ2A8 operators on Ascend 950."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm_ascend.quantization.vq2a8_method import load_repacked_layer
from vllm_ascend.quantization.vq2a8_ops import (
    custom_vq2a8_down_reduce_available,
    custom_vq2a8_gate_up_available,
    custom_vq2a8_prepare_debug_available,
    reference_vq2a8_direct_down_reduce,
    reference_vq2a8_direct_gate_up,
    reference_vq2a8_prepare,
    vq2a8_down_reduce,
    vq2a8_gate_up,
    vq2a8_prepare_debug,
)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[min(int((len(ordered) - 1) * quantile), len(ordered) - 1)]


def _comparison_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    finite = torch.isfinite(actual)
    finite_values = actual[finite]
    diff = actual - expected
    stats: dict[str, object] = {
        "shape": list(actual.shape),
        "finite": bool(finite.all()),
        "nan_count": int(torch.isnan(actual).sum()),
        "inf_count": int(torch.isinf(actual).sum()),
        "first_values": actual.flatten()[:16].tolist(),
    }
    if finite_values.numel():
        stats.update(
            min=float(finite_values.min()),
            max=float(finite_values.max()),
            mean=float(finite_values.mean()),
            checksum=float(finite_values.double().sum()),
            l2=float(torch.linalg.vector_norm(finite_values.double())),
        )
    if bool(torch.isfinite(diff).all()):
        stats.update(
            mae=float(diff.abs().mean()),
            max_abs_error=float(diff.abs().max()),
            cosine=float(
                F.cosine_similarity(
                    actual.double().flatten(),
                    expected.double().flatten(),
                    dim=0,
                    eps=1e-30,
                )
            ),
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=0.08)
    parser.add_argument("--atol", type=float, default=1.0)
    parser.add_argument(
        "--swiglu-limit",
        type=float,
        default=0.0,
        help="Use the model clamp (10.0 for DeepSeek V4) or 0 for legacy fixture goldens.",
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "gate-up", "full"),
        default="full",
        help="Validate gate/up alone first, then use full for gate/up plus down/reduce.",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    import torch_npu  # noqa: F401

    if args.stage == "prepare" and not custom_vq2a8_prepare_debug_available():
        raise RuntimeError("Missing _C_ascend.vq2a8_prepare_debug; rebuild vllm_ascend.")
    if args.stage != "prepare" and not custom_vq2a8_gate_up_available():
        raise RuntimeError("Missing _C_ascend.vq2a8_gate_up.")
    if args.stage == "full" and not custom_vq2a8_down_reduce_available():
        raise RuntimeError("Missing _C_ascend.vq2a8_down_reduce.")
    device = torch.device("npu")
    payload, metadata = load_repacked_layer(
        args.fixture / "ascend",
        args.layer,
        args.tp_size,
        args.tp_rank,
        expected_experts=1,
    )
    payload_cpu = payload
    payload = {name: tensor.to(device) for name, tensor in payload_cpu.items()}
    golden_path = args.fixture / "golden" / f"tp{args.tp_size}_rank{args.tp_rank}.safetensors"
    with safe_open(golden_path, framework="pt", device="cpu") as handle:
        tensor_names = handle.keys()
        golden = {name: handle.get_tensor(name) for name in tensor_names}
    x_cpu = golden["input_bf16"]
    expert_ids_cpu = golden["expert_ids"]
    x = x_cpu.to(device)
    expert_ids = expert_ids_cpu.to(device)
    token_ids = golden["token_ids"].to(device)
    routing_weights = golden["routing_weights"].to(device)
    num_tokens = int(token_ids.max().cpu()) + 1

    expected_prepare = None
    expected_gate_up = None
    expected_output = None
    if args.stage == "prepare":
        expected_prepare = reference_vq2a8_prepare(
            x_cpu,
            expert_ids_cpu,
            payload_cpu["gate_up_weight_scale"],
            payload_cpu["gate_up_weight_bias"],
            payload_cpu["gate_up_rht_sign"],
            metadata["rht_block_size"],
        )
    else:
        expected_gate_up = reference_vq2a8_direct_gate_up(
            x_cpu,
            expert_ids_cpu,
            payload_cpu["gate_up_packed_indices"],
            payload_cpu["gate_up_codebooks"],
            payload_cpu["gate_up_codebook_tile_ids"],
            payload_cpu["gate_up_weight_scale"],
            payload_cpu["gate_up_weight_bias"],
            payload_cpu["gate_up_rht_sign"],
            metadata["rht_block_size"],
            metadata["row_group_size"],
            swiglu_limit=args.swiglu_limit,
        )
        if args.stage == "full":
            expected_output = reference_vq2a8_direct_down_reduce(
                expected_gate_up,
                expert_ids_cpu,
                golden["token_ids"],
                golden["routing_weights"],
                payload_cpu["down_packed_indices"],
                payload_cpu["down_codebooks"],
                payload_cpu["down_codebook_tile_ids"],
                payload_cpu["down_weight_scale"],
                payload_cpu["down_weight_bias"],
                payload_cpu["down_rht_sign"],
                metadata["rht_block_size"],
                metadata["row_group_size"],
                num_tokens,
            )

    def run_once():
        if args.stage == "prepare":
            return vq2a8_prepare_debug(
                x,
                expert_ids,
                payload["gate_up_weight_scale"],
                payload["gate_up_weight_bias"],
                payload["gate_up_rht_sign"],
                metadata["rht_block_size"],
            )
        gate_up = vq2a8_gate_up(
            x,
            expert_ids,
            payload["gate_up_packed_indices"],
            payload["gate_up_codebooks"],
            payload["gate_up_codebook_tile_ids"],
            payload["gate_up_weight_scale"],
            payload["gate_up_weight_bias"],
            payload["gate_up_rht_sign"],
            metadata["rht_block_size"],
            metadata["row_group_size"],
            "silu",
            False,
            swiglu_limit=args.swiglu_limit,
        )
        output = None
        if args.stage == "full":
            output = vq2a8_down_reduce(
                gate_up,
                expert_ids,
                token_ids,
                routing_weights,
                payload["down_packed_indices"],
                payload["down_codebooks"],
                payload["down_codebook_tile_ids"],
                payload["down_weight_scale"],
                payload["down_weight_bias"],
                payload["down_rht_sign"],
                metadata["rht_block_size"],
                metadata["row_group_size"],
                num_tokens,
                False,
            )
        return gate_up, output

    for _ in range(args.warmup):
        run_once()
    result_tensors = run_once()
    torch.npu.synchronize()
    output = None
    if args.stage == "prepare":
        assert expected_prepare is not None
        quantized, activation_scale, bias_correction = result_tensors
        expected_quantized, expected_scale, expected_bias = expected_prepare
        validation = {
            "prepare_dequantized": _comparison_stats(
                quantized.float().cpu() * activation_scale.float().cpu().unsqueeze(-1),
                expected_quantized.float() * expected_scale.unsqueeze(-1),
            ),
            "prepare_scale": _comparison_stats(activation_scale, expected_scale),
            "prepare_bias": _comparison_stats(bias_correction, expected_bias),
        }
    else:
        assert expected_gate_up is not None
        gate_up, output = result_tensors
        gate_up_cpu = gate_up.float().cpu()
        validation = {
            "gate_up": _comparison_stats(
                gate_up_cpu,
                expected_gate_up,
            )
        }
        if output is not None:
            assert expected_output is not None
            output_cpu = output.float().cpu()
            validation["down_reduce"] = _comparison_stats(
                output_cpu,
                expected_output,
            )
    print(json.dumps({"validation": validation}, indent=2), flush=True)
    if args.stage == "prepare":
        torch.testing.assert_close(
            quantized.float().cpu() * activation_scale.float().cpu().unsqueeze(-1),
            expected_quantized.float() * expected_scale.unsqueeze(-1),
            rtol=args.rtol,
            atol=args.atol,
        )
        torch.testing.assert_close(
            activation_scale.float().cpu(),
            expected_scale,
            rtol=args.rtol,
            atol=args.atol,
        )
        torch.testing.assert_close(
            bias_correction.float().cpu(),
            expected_bias,
            rtol=args.rtol,
            atol=args.atol,
        )
    else:
        torch.testing.assert_close(
            gate_up_cpu,
            expected_gate_up,
            rtol=args.rtol,
            atol=args.atol,
        )
        if output is not None:
            assert expected_output is not None
            torch.testing.assert_close(
                output_cpu,
                expected_output,
                rtol=args.rtol,
                atol=args.atol,
            )

    latencies_ms = []
    for _ in range(args.iterations):
        torch.npu.synchronize()
        start = time.perf_counter()
        run_once()
        torch.npu.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000)
    result = {
        "fixture": str(args.fixture),
        "stage": args.stage,
        "tp_rank": args.tp_rank,
        "tokens": num_tokens,
        "route_rows": x.shape[0],
        "iterations": args.iterations,
        "mean_ms": statistics.mean(latencies_ms),
        "p50_ms": statistics.median(latencies_ms),
        "p95_ms": _percentile(latencies_ms, 0.95),
        "tokens_per_second": num_tokens / (statistics.mean(latencies_ms) / 1000),
        "correctness": "passed",
        "validation": validation,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
