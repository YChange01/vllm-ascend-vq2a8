# SPDX-License-Identifier: Apache-2.0
"""Export one production-shape VQ2 expert, TP repacks, and CPU goldens."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm_ascend.quantization.vq2a8_format import (
    load_layer_metadata,
    parse_expert_name,
)
from vllm_ascend.quantization.vq2a8_method import load_repacked_layer
from vllm_ascend.quantization.vq2a8_ops import (
    reference_vq2a8_down_reduce,
    reference_vq2a8_gate_up,
)
from vllm_ascend.quantization.vq2a8_repack import repack_layer


def _extract_expert(source: Path, output: Path, layer_index: int, expert_id: int) -> None:
    metadata = load_layer_metadata(source, layer_index)
    source_metadata = {name: value for name, value in metadata.items() if parse_expert_name(name)[1] == expert_id}
    if len(source_metadata) != 2:
        raise ValueError(f"Expected gate_up and down for expert {expert_id}, found {sorted(source_metadata)}.")
    selected_metadata = {
        name.replace(f".experts.{expert_id}.", ".experts.0."): value for name, value in source_metadata.items()
    }
    canonical_path = output / "canonical"
    canonical_path.mkdir(parents=True)
    source_tensor_path = source / f"experts_vq_layer_{layer_index}.safetensors"
    with safe_open(source_tensor_path, framework="pt", device="cpu") as handle:
        tensor_names = handle.keys()
        tensors = {}
        for source_name in source_metadata:
            output_name = source_name.replace(f".experts.{expert_id}.", ".experts.0.")
            prefix = f"{source_name}."
            for tensor_name in tensor_names:
                if tensor_name.startswith(prefix):
                    output_tensor_name = tensor_name.replace(source_name, output_name, 1)
                    tensors[output_tensor_name] = handle.get_tensor(tensor_name)
        tensor_metadata = handle.metadata()
    stem = f"experts_vq_layer_{layer_index}"
    (canonical_path / f"{stem}.json").write_text(json.dumps(selected_metadata, indent=2) + "\n", encoding="utf-8")
    save_file(
        tensors,
        canonical_path / f"{stem}.safetensors",
        metadata=tensor_metadata,
    )


def _golden_input(tokens: int, top_k: int, hidden_size: int) -> torch.Tensor:
    values = torch.arange(tokens * hidden_size, dtype=torch.float32)
    token_input = ((values.remainder(257) - 128) / 128).reshape(tokens, hidden_size)
    return token_input.to(torch.bfloat16).repeat_interleave(top_k, dim=0)


def _write_goldens(
    output: Path,
    layer_index: int,
    tp_size: int,
    tokens: int,
    top_k: int,
) -> list[dict[str, object]]:
    golden_path = output / "golden"
    golden_path.mkdir()
    summaries = []
    for tp_rank in range(tp_size):
        payload, metadata = load_repacked_layer(
            output / "ascend",
            layer_index,
            tp_size,
            tp_rank,
            expected_experts=1,
        )
        hidden_size = payload["gate_up_codebook_tile_ids"].shape[1]
        expanded_input = _golden_input(tokens, top_k, hidden_size)
        rows = tokens * top_k
        expert_ids = torch.zeros(rows, dtype=torch.int32)
        token_ids = torch.arange(tokens, dtype=torch.int64).repeat_interleave(top_k)
        routing_weights = torch.full((rows,), 1.0 / top_k, dtype=torch.float32)
        gate_up = reference_vq2a8_gate_up(
            expanded_input.float(),
            expert_ids,
            payload["gate_up_packed_indices"],
            payload["gate_up_codebooks"],
            payload["gate_up_codebook_tile_ids"],
            payload["gate_up_weight_scale"],
            payload["gate_up_weight_bias"],
            payload["gate_up_rht_sign"],
            metadata["rht_block_size"],
            metadata["row_group_size"],
        )
        output_reference = reference_vq2a8_down_reduce(
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
            tokens,
        )
        golden_file = golden_path / f"tp{tp_size}_rank{tp_rank}.safetensors"
        save_file(
            {
                "input_bf16": expanded_input,
                "expert_ids": expert_ids,
                "token_ids": token_ids,
                "routing_weights": routing_weights,
                "expected_gate_up_fp32": gate_up,
                "expected_output_fp32": output_reference,
            },
            golden_file,
        )
        summaries.append(
            {
                "tp_rank": tp_rank,
                "input_shape": list(expanded_input.shape),
                "gate_up_shape": list(gate_up.shape),
                "output_shape": list(output_reference.shape),
                "gate_up_checksum": float(gate_up.double().sum()),
                "output_checksum": float(output_reference.double().sum()),
            }
        )
    return summaries


def _write_sha256(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()
    output = args.output.resolve()
    partial = output.with_name(f".{output.name}.partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Output already exists: {output} or {partial}")
    partial.mkdir(parents=True)
    try:
        _extract_expert(args.input.resolve(), partial, args.layer, args.expert)
        repack_layer(partial / "canonical", partial / "ascend", args.layer, args.tp_size)
        summaries = _write_goldens(partial, args.layer, args.tp_size, args.tokens, args.top_k)
        manifest = {
            "format": "vq2a8_ascend950_kernel_fixture_v1",
            "layer": args.layer,
            "source_expert": args.expert,
            "fixture_expert_id": 0,
            "tp_size": args.tp_size,
            "tokens": args.tokens,
            "top_k": args.top_k,
            "goldens": summaries,
        }
        (partial / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        _write_sha256(partial)
        os.replace(partial, output)
    except Exception:
        print(f"partial_output={partial}", flush=True)
        raise
    print(f"kernel_fixture={output}", flush=True)


if __name__ == "__main__":
    main()
