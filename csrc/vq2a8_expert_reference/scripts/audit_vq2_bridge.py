#!/usr/bin/env python3
"""Read-only CPU audit of direct-TP1 safetensors against the expert layout bridge.

This does not repack or overwrite checkpoints, compile a kernel, or import model
or NPU modules. Byte equality is not a floating-point or model accuracy result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from vq2_bridge import audit_direct_contract, bridge_direct_weights, decode_bridge_bytes

PACKED_SUFFIX = ".packed_indices"
ROW_CHUNK = 64


def _key(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def direct_decode_bytes(words: np.ndarray, books: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Independent direct-layout formula, never using the bridge unpacker.

    W[n,k] = books[ids[k], n//32, (word[n//2,k//8] >> (4*(k%8))) & 15, n%2].
    Chunk the output rows to bound temporary index tensors on real matrices.
    """
    n, k = words.shape[0] * 2, ids.size
    result = np.empty((n, k), dtype=np.uint8)
    columns = np.arange(k, dtype=np.int64)
    unsigned = words.view(np.uint32)
    shifts = (columns % 8 * 4).astype(np.uint32)
    for start in range(0, n, ROW_CHUNK):
        rows = np.arange(start, min(start + ROW_CHUNK, n), dtype=np.int64)
        code = (unsigned[rows[:, None] // 2, columns[None, :] // 8] >> shifts[None, :]) & 15
        result[rows] = books[ids[None, :], rows[:, None] // 32, code, rows[:, None] % 2]
    return result


def _codes_equal(words: np.ndarray, packed_zn: np.ndarray, order: np.ndarray) -> bool:
    """Compare pair indices directly at their two different storage addresses."""
    columns = np.arange(order.size, dtype=np.int64)
    unsigned = words.view(np.uint32)
    shifts = (order % 8 * 4).astype(np.uint32)
    for start in range(0, words.shape[0], ROW_CHUNK):
        pairs = np.arange(start, min(start + ROW_CHUNK, words.shape[0]), dtype=np.int64)
        expected = (unsigned[pairs[:, None], order[None, :] // 8] >> shifts[None, :]) & 15
        stored = packed_zn[
            pairs[:, None] // 16, columns[None, :] // 16, columns[None, :] % 16, (pairs[:, None] % 16) // 2
        ]
        actual = (stored >> ((pairs[:, None] % 2) * 4)) & 15
        if not np.array_equal(actual, expected):
            return False
    return True


def _check_mode(words, books, ids, expected, mode: str) -> dict:
    candidate = bridge_direct_weights(words, books, ids, k_order=mode)
    decoded = decode_bridge_bytes(candidate)
    order = candidate.activation_gather
    if order.shape != ids.shape or not np.array_equal(np.sort(order), np.arange(ids.size)):
        raise ValueError(f"{mode} activation gather is not a complete K permutation")
    # Sorting changes floating-point reduction order, but W's source-order bytes
    # must still be recoverable exactly. This does not exercise actual NPU MMAD.
    roundtrip = np.empty_like(decoded)
    roundtrip[:, order] = decoded
    codes_exact = _codes_equal(words, candidate.packed_zn, order)
    weights_exact = np.array_equal(roundtrip, expected)
    if not codes_exact or not weights_exact:
        raise ValueError(f"{mode} bridge mismatch: codes_exact={codes_exact}, weight_bytes_exact={weights_exact}")
    return {
        "status": "passed",
        "codes_exact": True,
        "weight_bytes_exact": True,
        "source_order_roundtrip_exact": True,
        "k_reordered": candidate.k_reordered,
        "fixed_k256_lut_available": candidate.fixed_k256_lut is not None,
        "packed_bytes": candidate.packed_zn.nbytes,
        "decoded_weight_bytes_checked": expected.nbytes,
    }


def audit_matrix(handle, prefix: str, *, batched: bool = False, expert_index: int = 0) -> dict:
    # Lazy CPU-only dependencies: no vllm, vllm_ascend, torch_npu or device code.
    import torch

    names = ("packed_indices", "codebooks", "codebook_tile_ids")
    dtypes = (torch.int32, torch.float8_e4m3fn, torch.uint8)
    ranks = (2, 4, 1)
    tensors = []
    keys = set(handle.keys())
    for name, dtype, rank in zip(names, dtypes, ranks):
        key = f"{prefix}_{name}" if batched else _key(prefix, name)
        if key not in keys:
            raise ValueError(f"missing required tensor {key!r}")
        if batched:
            source = handle.get_slice(key)
            shape = source.get_shape()
            if len(shape) != rank + 1 or not 0 <= expert_index < shape[0]:
                raise ValueError(
                    f"{key}: expected rank {rank + 1} with valid expert-index={expert_index}, found {shape}"
                )
            # A real layer can exceed 1 GiB. Load only this expert, never call
            # get_tensor() on the batched layer weight storage.
            value = source[expert_index]
        else:
            value = handle.get_tensor(key)
        if value.dtype != dtype or value.device.type != "cpu":
            raise TypeError(f"{key}: require CPU {dtype}, found {value.device} {value.dtype}")
        tensors.append(value)
    words = tensors[0].numpy()
    books = tensors[1].view(torch.uint8).numpy()
    ids = tensors[2].numpy()
    contract = audit_direct_contract(words, books, ids)
    expected = direct_decode_bytes(words, books, ids)
    modes = {"preserve": _check_mode(words, books, ids, expected, "preserve")}
    if contract["stable_grouping_can_form_k256"]:
        modes["codebook"] = _check_mode(words, books, ids, expected, "codebook")
    else:
        modes["codebook"] = {
            "status": "not_representable",
            "reason": "tile ID populations cannot form homogeneous K256 blocks; retain per-K IDs",
            "source_order_roundtrip_exact": False,
        }
    return {
        "prefix": prefix,
        "expert_index": expert_index if batched else None,
        "source_layout": "batched_projection" if batched else "direct_matrix",
        "status": "passed",
        "contract": contract,
        "modes": modes,
    }


def audit_files(
    paths: list[Path],
    prefixes: list[str] | None = None,
    *,
    projections: list[str] | None = None,
    expert_index: int = 0,
) -> dict:
    from safetensors import SafetensorError, safe_open

    results = []
    for path in paths:
        record = {"path": str(path), "status": "passed", "matrices": []}
        try:
            if expert_index < 0:
                raise ValueError("expert-index must be non-negative")
            if prefixes is not None and projections is not None:
                raise ValueError("choose --prefix for direct matrices OR --projection for batched layers")
            path = path.resolve(strict=True)
            record["path"] = str(path)
            if not path.is_file() or path.suffix != ".safetensors":
                raise ValueError("provide an explicit .safetensors file, not a checkpoint directory")
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                found = sorted(key[: -len(PACKED_SUFFIX)] for key in keys if key.endswith(PACKED_SUFFIX))
                if "packed_indices" in keys:
                    found.insert(0, "")
                if prefixes is not None:
                    selected = [(prefix, False) for prefix in dict.fromkeys(prefixes)]
                elif projections is not None:
                    selected = [(projection, True) for projection in dict.fromkeys(projections)]
                else:
                    selected = [(prefix, False) for prefix in found]
                    selected += [
                        (projection, True)
                        for projection in ("gate_up", "down")
                        if f"{projection}_packed_indices" in keys
                    ]
                if not selected:
                    raise ValueError("no direct-layout packed_indices tensors found")
                for prefix, batched in selected:
                    try:
                        record["matrices"].append(
                            audit_matrix(handle, prefix, batched=batched, expert_index=expert_index)
                        )
                    except (ValueError, TypeError, RuntimeError, KeyError, IndexError) as error:
                        record["status"] = "failed"
                        record["matrices"].append({"prefix": prefix, "status": "failed", "error": str(error)})
        except (OSError, ValueError, TypeError, RuntimeError, SafetensorError) as error:
            record.update(status="failed", error=str(error))
        results.append(record)
    return {
        "status": "passed" if results and all(value["status"] == "passed" for value in results) else "failed",
        "scope": "cpu_layout_bytes_only",
        "checkpoint_modified": False,
        "kernel_compiled": False,
        "device_execution_verified": False,
        "model_accuracy_verified": False,
        "drop_in_replacement_ready": False,
        "files": results,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="explicit direct-TP1 .safetensors files (read only)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--prefix", action="append", help="direct matrix prefix; may repeat")
    selection.add_argument(
        "--projection", choices=("gate_up", "down"), action="append", help="batched projection; may repeat"
    )
    parser.add_argument(
        "--expert-index", type=int, default=0, help="one expert to slice from each batched layer (default: 0)"
    )
    args = parser.parse_args(argv)
    report = audit_files(args.files, args.prefix, projections=args.projection, expert_index=args.expert_index)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
