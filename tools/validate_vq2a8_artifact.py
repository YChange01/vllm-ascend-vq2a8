# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate canonical VQ2A8 expert artifacts without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

from vllm_ascend.quantization.vq2a8_artifact import (
    VQ2_CONSUMER_REFERENCE_COMMIT,
    VQ2_MATRIX_KINDS,
    inspect_layer_artifact,
    inspect_vq2_directory,
    load_matrix_tensors,
    load_model_layout,
    validate_matrix_payload,
    validate_model_layout,
)


def _parse_ranges(value: str) -> tuple[int, ...]:
    values: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise argparse.ArgumentTypeError("empty item in integer range")
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            try:
                start, end = int(start_text), int(end_text)
            except ValueError as error:
                raise argparse.ArgumentTypeError(f"invalid integer range: {item!r}") from error
            if start < 0 or end < start:
                raise argparse.ArgumentTypeError(f"invalid integer range: {item!r}")
            values.update(range(start, end + 1))
        else:
            try:
                parsed = int(item)
            except ValueError as error:
                raise argparse.ArgumentTypeError(f"invalid integer: {item!r}") from error
            if parsed < 0:
                raise argparse.ArgumentTypeError(f"integer must be non-negative: {item!r}")
            values.add(parsed)
    return tuple(sorted(values))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experts_path", type=Path, help="Canonical experts_vq directory.")
    parser.add_argument(
        "--layers",
        default="all",
        help="Layers to inspect, for example 0,3-5. Default: all.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        help="Optional config.json. If omitted, ../config.json is used when present.",
    )
    producer_group = parser.add_mutually_exclusive_group()
    producer_group.add_argument(
        "--producer-codec",
        type=Path,
        help="Exact producer codec file; its SHA256 is calculated by this tool.",
    )
    producer_group.add_argument(
        "--claimed-producer-codec-sha256",
        help="Externally supplied producer codec SHA256 (recorded as unverified).",
    )
    payload_group = parser.add_mutually_exclusive_group()
    payload_group.add_argument(
        "--payload-experts",
        type=_parse_ranges,
        default=(),
        help="Expert IDs whose payload values should also be checked.",
    )
    payload_group.add_argument(
        "--all-payloads",
        action="store_true",
        help="Check every matrix payload sequentially (potentially slow).",
    )
    return parser


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Producer codec is not a file: {path}.")
    digest = hashlib.sha256()
    with path.open("rb") as codec_file:
        for chunk in iter(lambda: codec_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _build_parser().parse_args()
    if (
        args.claimed_producer_codec_sha256 is not None
        and re.fullmatch(r"[0-9a-fA-F]{64}", args.claimed_producer_codec_sha256) is None
    ):
        raise ValueError("--claimed-producer-codec-sha256 must contain exactly 64 hexadecimal characters.")
    experts_path = args.experts_path.resolve()
    all_layers = args.layers == "all"
    if all_layers:
        summaries = inspect_vq2_directory(experts_path)
    else:
        selected_layers = _parse_ranges(args.layers)
        if not selected_layers:
            raise ValueError("At least one layer must be selected.")
        summaries = [inspect_layer_artifact(experts_path, layer) for layer in selected_layers]

    config_path = args.model_config
    if config_path is None:
        candidate = experts_path.parent / "config.json"
        config_path = candidate if candidate.is_file() else None
    layout = None
    if config_path is not None:
        layout = load_model_layout(config_path)
        validate_model_layout(summaries, layout, require_all_layers=all_layers)

    payload_summaries = []
    for layer_summary in summaries:
        expert_ids = layer_summary.expert_ids if args.all_payloads else args.payload_experts
        for expert_id in expert_ids:
            if expert_id not in layer_summary.expert_ids:
                raise ValueError(f"Layer {layer_summary.layer_index} has no expert {expert_id}.")
            for kind in VQ2_MATRIX_KINDS:
                tensors, spec = load_matrix_tensors(
                    experts_path,
                    layer_summary.layer_index,
                    expert_id,
                    kind,
                )
                payload_summaries.append(validate_matrix_payload(tensors, spec))

    if args.all_payloads:
        validation_scope = "all_payloads"
    elif args.payload_experts:
        validation_scope = "selected_payloads"
    else:
        validation_scope = "header_only"
    expected_payloads = sum(summary.matrix_count for summary in summaries)
    validation_complete = layout is not None and all_layers and args.all_payloads

    producer_codec = None
    if args.producer_codec is not None:
        codec_path = args.producer_codec.resolve()
        producer_codec = {
            "path": str(codec_path),
            "sha256": _sha256_file(codec_path),
            "verified_from_file": True,
        }
    elif args.claimed_producer_codec_sha256 is not None:
        producer_codec = {
            "path": None,
            "sha256": args.claimed_producer_codec_sha256.lower(),
            "verified_from_file": False,
        }

    report = {
        "format": "canonical-vq2a8",
        "consumer_reference_commit": VQ2_CONSUMER_REFERENCE_COMMIT,
        "producer_codec": producer_codec,
        "experts_path": str(experts_path),
        "model_config": str(config_path.resolve()) if config_path is not None else None,
        "validation": {
            "scope": validation_scope,
            "complete": validation_complete,
            "layers_checked": len(summaries),
            "layers_expected": layout.num_hidden_layers if layout is not None else None,
            "matrix_headers_checked": expected_payloads,
            "matrix_payloads_checked": len(payload_summaries),
            "matrix_payloads_in_checked_layers": expected_payloads,
        },
        "layers": [asdict(summary) for summary in summaries],
        "payloads": [asdict(summary) for summary in payload_summaries],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
