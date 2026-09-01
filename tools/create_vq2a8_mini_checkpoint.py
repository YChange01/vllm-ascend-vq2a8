# SPDX-License-Identifier: Apache-2.0
"""Create a small, end-to-end VQ2A8 checkpoint for Ascend bring-up."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm_ascend.quantization.vq2a8_format import parse_expert_name
from vllm_ascend.quantization.vq2a8_repack import repack_layer

METADATA_FILES = (
    "README.md",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def decoder_layer_index(name: str) -> int | None:
    match = LAYER_PATTERN.search(name)
    return int(match.group(1)) if match else None


def selected_model_weights(weight_map: dict[str, str], num_layers: int) -> dict[str, str]:
    return {
        name: shard
        for name, shard in weight_map.items()
        if (layer_index := decoder_layer_index(name)) is None or layer_index < num_layers
    }


def slice_router_tensor(
    name: str,
    tensor: torch.Tensor,
    source_experts: int,
    target_experts: int,
) -> torch.Tensor:
    if name.endswith(".ffn.gate.weight") or name.endswith(".ffn.gate.bias"):
        if tensor.shape[0] != source_experts:
            raise ValueError(f"Router tensor {name} has {tensor.shape[0]} experts, expected {source_experts}.")
        return tensor[:target_experts].contiguous()
    if name.endswith(".ffn.gate.tid2eid"):
        if tensor.numel() and int(tensor.max()) >= target_experts:
            raise ValueError(
                f"Hash router {name} references expert {int(tensor.max())}, "
                f"outside the requested {target_experts} experts."
            )
    return tensor


def mini_config(source_config: dict[str, Any], num_layers: int, num_experts: int) -> dict[str, Any]:
    config = dict(source_config)
    source_layers = int(config["num_hidden_layers"])
    source_experts = int(config["n_routed_experts"])
    if not 1 <= num_layers <= source_layers:
        raise ValueError(f"num_layers must be in [1, {source_layers}], got {num_layers}.")
    if not 1 <= num_experts <= source_experts:
        raise ValueError(f"num_experts must be in [1, {source_experts}], got {num_experts}.")
    config["num_hidden_layers"] = num_layers
    config["n_routed_experts"] = num_experts
    config["num_experts_per_tok"] = min(int(config["num_experts_per_tok"]), num_experts)
    config["num_hash_layers"] = min(int(config.get("num_hash_layers", 0)), num_layers)
    config["num_nextn_predict_layers"] = 0
    if "compress_ratios" in config:
        config["compress_ratios"] = config["compress_ratios"][:num_layers]
    quantization_config = dict(config.get("quantization_config", {}))
    quantization_config.update(
        {
            "quant_method": "vq2a8",
            "experts_path": "experts_vq",
            "kernel_path": "experts_vq_ascend",
            "allow_reference_fallback": True,
        }
    )
    config["quantization_config"] = quantization_config
    return config


def _write_model_shards(
    source: Path,
    output: Path,
    selected: dict[str, str],
    source_experts: int,
    target_experts: int,
) -> tuple[dict[str, str], int, int]:
    names_by_source_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard_name in selected.items():
        names_by_source_shard[shard_name].append(name)
    source_shards = sorted(names_by_source_shard)
    output_weight_map: dict[str, str] = {}
    total_parameters = 0
    total_size = 0
    for output_index, source_shard_name in enumerate(source_shards, start=1):
        output_shard_name = f"model-{output_index:05d}-of-{len(source_shards):05d}.safetensors"
        print(
            f"model_shard={output_index}/{len(source_shards)} source={source_shard_name}",
            flush=True,
        )
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(source / source_shard_name, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            for name in sorted(names_by_source_shard[source_shard_name]):
                tensor = slice_router_tensor(
                    name,
                    handle.get_tensor(name),
                    source_experts,
                    target_experts,
                )
                tensors[name] = tensor
                output_weight_map[name] = output_shard_name
                total_parameters += tensor.numel()
                total_size += tensor.numel() * tensor.element_size()
        save_file(tensors, output / output_shard_name, metadata=metadata)
        del tensors
    return output_weight_map, total_parameters, total_size


def _write_canonical_experts(
    source: Path,
    output: Path,
    num_layers: int,
    num_experts: int,
) -> dict[int, list[int]]:
    source_experts_path = source / "experts_vq"
    output_experts_path = output / "experts_vq"
    output_experts_path.mkdir()
    selected_by_layer: dict[int, list[int]] = {}
    for layer_index in range(num_layers):
        stem = f"experts_vq_layer_{layer_index}"
        metadata_path = source_experts_path / f"{stem}.json"
        tensor_path = source_experts_path / f"{stem}.safetensors"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        selected_metadata = {
            name: value for name, value in metadata.items() if parse_expert_name(name)[1] < num_experts
        }
        expert_ids = sorted({parse_expert_name(name)[1] for name in selected_metadata})
        if not expert_ids or expert_ids != list(range(expert_ids[-1] + 1)):
            raise ValueError(f"Layer {layer_index} has invalid selected experts: {expert_ids}.")
        selected_by_layer[layer_index] = expert_ids
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            selected_prefixes = tuple(f"{name}." for name in selected_metadata)
            tensor_names = handle.keys()
            tensors = {name: handle.get_tensor(name) for name in tensor_names if name.startswith(selected_prefixes)}
            tensor_metadata = handle.metadata()
        (output_experts_path / f"{stem}.json").write_text(
            json.dumps(selected_metadata, indent=2) + "\n", encoding="utf-8"
        )
        save_file(
            tensors,
            output_experts_path / f"{stem}.safetensors",
            metadata=tensor_metadata,
        )
        print(
            f"canonical_layer={layer_index} experts={len(expert_ids)}",
            flush=True,
        )
    return selected_by_layer


def _sha256_manifest(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        lines.append(f"{digest.hexdigest()}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=4)
    args = parser.parse_args()

    source = args.source_model.resolve()
    output = args.output.resolve()
    partial = output.with_name(f".{output.name}.partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Output already exists: {output} or {partial}")
    source_config = json.loads((source / "config.json").read_text())
    config = mini_config(source_config, args.layers, args.experts)
    source_index = json.loads((source / "model.safetensors.index.json").read_text())
    selected = selected_model_weights(source_index["weight_map"], args.layers)
    partial.mkdir(parents=True)
    try:
        for filename in METADATA_FILES:
            source_file = source / filename
            if source_file.is_file():
                shutil.copy2(source_file, partial / filename)
        (partial / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        output_weight_map, total_parameters, total_size = _write_model_shards(
            source,
            partial,
            selected,
            int(source_config["n_routed_experts"]),
            args.experts,
        )
        output_index = {
            "metadata": {
                "total_parameters": total_parameters,
                "total_size": total_size,
            },
            "weight_map": output_weight_map,
        }
        (partial / "model.safetensors.index.json").write_text(
            json.dumps(output_index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        selected_by_layer = _write_canonical_experts(source, partial, args.layers, args.experts)
        repacked_path = partial / "experts_vq_ascend"
        for layer_index in range(args.layers):
            repack_layer(
                partial / "experts_vq",
                repacked_path,
                layer_index,
                args.tp_size,
            )
        manifest = {
            "format": "vq2a8_mini_checkpoint_v1",
            "source_model": source.name,
            "num_hidden_layers": args.layers,
            "n_routed_experts": args.experts,
            "num_experts_per_tok": config["num_experts_per_tok"],
            "tp_size": args.tp_size,
            "canonical_experts_per_layer": selected_by_layer,
            "intended_use": "Ascend 950 integration and smoke testing only",
            "accuracy_model": False,
        }
        (partial / "VQ2A8_MINI_CHECKPOINT.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        _sha256_manifest(partial)
        os.replace(partial, output)
    except Exception:
        print(f"partial_output={partial}", flush=True)
        raise
    print(f"mini_checkpoint={output}", flush=True)


if __name__ == "__main__":
    main()
