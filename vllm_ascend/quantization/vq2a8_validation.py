# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side diagnostics for disconnected Ascend 950 acceptance runs.

These checks are intentionally outside the serving hot path. Payload checks
may synchronize or read small checkpoint routing tables on CPU.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import torch
from safetensors import safe_open

from vllm_ascend.quantization.vq2a8_runtime import VQ2_TP1_TORCH_DTYPES, VQ2TP1Artifact

_HASH_TABLE = re.compile(r"^(?:model\.)?layers\.(\d+)\.(?:ffn|mlp)\.gate\.tid2eid$")


def validate_tolerances(rtol: float, atol: float) -> None:
    for name, value in (("rtol", rtol), ("atol", atol)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}.")


def error_metrics(expected: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    """Scale-aware metrics supplement elementwise absolute/relative limits."""
    ref = expected.detach().double().cpu().flatten()
    got = actual.detach().double().cpu().flatten()
    diff = got - ref
    norm_ref = float(torch.linalg.vector_norm(ref))
    norm_diff = float(torch.linalg.vector_norm(diff))
    return {
        "relative_l2_error": norm_diff / max(norm_ref, 1e-12),
        "rmse": float(diff.square().mean().sqrt()) if diff.numel() else 0.0,
        "reference_l2": norm_ref,
    }


def tensor_layout(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "storage_offset": tensor.storage_offset(),
        "pointer_mod_32": tensor.data_ptr() % 32,
        "pointer_mod_256": tensor.data_ptr() % 256,
        "bytes": tensor.numel() * tensor.element_size(),
    }


def audit_model_storage(artifact: VQ2TP1Artifact) -> dict[str, Any]:
    """Inventory disk tensors and verify hash routes against stored experts.

    The total is a storage budget, NOT a peak HBM prediction. Root checkpoint
    bytes may include unused prediction layers and may change dtype at load.
    Hash tables are small CPU reads; other tensors are inspected via headers.
    """
    model_root = artifact.model_config_path.parent
    config = json.loads(artifact.model_config_path.read_text(encoding="utf-8"))
    files = sorted(model_root.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No root checkpoint safetensors found in {model_root}.")
    hash_layers = {i for i in artifact.layers if i < artifact.model_layout.num_hash_layers}
    hash_reports: dict[int, dict[str, Any]] = {}
    root_payload_bytes = 0
    root_tensor_names: set[str] = set()
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in sorted(handle.keys()):
                if name in root_tensor_names:
                    raise ValueError(f"Duplicate root checkpoint tensor {name}.")
                root_tensor_names.add(name)
                # Count bytes from header offsets without loading a dense tensor.
                # safe_open verifies the file; this inventory handles all dtypes.
                match = _HASH_TABLE.fullmatch(name)
                if match and int(match[1]) in hash_layers:
                    index = int(match[1])
                    if index in hash_reports:
                        raise ValueError(f"Multiple hash tables for layer {index}.")
                    table = handle.get_tensor(name)
                    expected_shape = (config["vocab_size"], config["num_experts_per_tok"])
                    if table.dtype not in (torch.int32, torch.int64) or tuple(table.shape) != expected_shape:
                        raise ValueError(f"Invalid hash table {name}: {table.dtype}, {tuple(table.shape)}.")
                    route_ids = sorted(int(x) for x in table.unique().tolist())
                    missing = sorted(set(route_ids) - set(artifact.layer(index).expert_ids))
                    if missing:
                        raise ValueError(f"Hash layer {index} routes to missing artifact experts: {missing}.")
                    duplicate_rows = bool((table.sort(dim=-1).values.diff(dim=-1) == 0).any())
                    hash_reports[index] = {
                        "layer": index,
                        "tensor": name,
                        "route_expert_ids": route_ids,
                        "stored_expert_ids": list(artifact.layer(index).expert_ids),
                        "duplicate_ids_within_topk": duplicate_rows,
                    }
        with path.open("rb") as stream:
            header_size = int.from_bytes(stream.read(8), "little")
            header = json.loads(stream.read(header_size))
        root_payload_bytes += sum(
            entry["data_offsets"][1] - entry["data_offsets"][0]
            for name, entry in header.items()
            if name != "__metadata__"
        )
    if set(hash_reports) != hash_layers:
        raise ValueError(f"Missing checkpoint hash tables for layers {sorted(hash_layers - set(hash_reports))}.")

    expert_payload_bytes = 0
    for layer in artifact.layers.values():
        for name, shape in layer.tensor_shapes.items():
            field = next(field for field in VQ2_TP1_TORCH_DTYPES if name.endswith(f"_{field}"))
            expert_payload_bytes += math.prod(shape) * torch.empty((), dtype=VQ2_TP1_TORCH_DTYPES[field]).element_size()
    return {
        "artifact_complete": artifact.manifest["complete"],
        "artifact_expert_tensor_bytes": expert_payload_bytes,
        "root_checkpoint_tensor_bytes": root_payload_bytes,
        "combined_storage_gib": (expert_payload_bytes + root_payload_bytes) / 1024**3,
        "hash_routing": [hash_reports[index] for index in sorted(hash_reports)],
        "peak_hbm_verified": False,
        "excludes": ["KV cache", "workspace", "allocator reserve", "loading copies", "dtype conversion"],
        "serving_integration_verified": False,
    }
