# SPDX-License-Identifier: Apache-2.0
"""Repack canonical VQ2 experts into the Ascend TP-local direct layout."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path

from vllm_ascend.quantization.vq2a8_format import ASCEND_VQ2_TP_FORMAT
from vllm_ascend.quantization.vq2a8_repack import repack_layer


def parse_layers(specification: str, available: Iterable[int]) -> list[int]:
    if specification == "all":
        return sorted(available)
    layers: set[int] = set()
    for part in specification.split(","):
        match = re.fullmatch(r"(\d+)-(\d+)", part)
        if match:
            start, end = map(int, match.groups())
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--tp-size", type=int, default=4)
    args = parser.parse_args()
    if args.tp_size < 1:
        parser.error("--tp-size must be positive")

    available = [
        int(match.group(1))
        for path in args.input.glob("experts_vq_layer_*.json")
        if (match := re.search(r"layer_(\d+)\.json$", path.name))
    ]
    selected = parse_layers(args.layers, available)
    if not selected:
        parser.error("no VQ2 layers selected")
    written = []
    for layer_index in selected:
        layer_files = repack_layer(
            args.input, args.output, layer_index, args.tp_size
        )
        written.extend(str(path.relative_to(args.output)) for path in layer_files)
        print(
            f"layer={layer_index} tp_size={args.tp_size} "
            f"files={len(layer_files)}",
            flush=True,
        )
    manifest = {
        "format": ASCEND_VQ2_TP_FORMAT,
        "tp_size": args.tp_size,
        "layers": selected,
        "files": written,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
