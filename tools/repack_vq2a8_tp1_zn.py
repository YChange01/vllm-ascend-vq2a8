#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only canonical experts_vq -> TP1 packed-zN, using the TP2 byte layout.

The full gate/up and down matrices remain on rank0, with full-K activation
quantization and no TP collective. This is NOT the legacy TP1 direct format.
Conversion verifies stored bytes; device/model accuracy requires a separate run.
"""

from __future__ import annotations

# Keep direct execution independent of runtime imports and tools/bisect.
# ruff: noqa: E402
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from tools import repack_vq2a8_tp2 as common


def parse_args(argv: list[str] | None = None) -> Namespace:
    return common.parse_args(argv, description=__doc__)


def run(args: Namespace) -> dict[str, Any]:
    return common.run(args, tp_size=1, entrypoint=Path(__file__))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"VQ2_TP1_ERROR={error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
