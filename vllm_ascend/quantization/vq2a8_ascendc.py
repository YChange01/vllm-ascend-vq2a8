# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicitly loaded native AscendC prototype; no default backend integration.

Only an already-built library can be loaded. There is no JIT, subprocess,
Triton implementation or fallback in the projection path. C++ validates the
tensor metadata even when callers bypass this Python wrapper.
"""

import hashlib
import re
from pathlib import Path

import torch


def load_pinned_library(path: str | Path, expected_sha256: str) -> dict:
    """Load only the exact standalone candidate selected by the offline gate."""
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("AscendC requires an explicit SHA256 library identity.")
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_sha256:
        raise ValueError("AscendC library differs from the regression-tested candidate.")
    load_library(path)
    return {"path": str(path), "sha256": actual}


def load_library(path: str | Path) -> Path:
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.suffix != ".so":
        raise ValueError("Expected the explicitly built libvq2a8_ascendc.so file.")
    # Use Torch's existing registration state; don't maintain a second mutable
    # process-global loader registry in this plugin.
    if str(path) in torch.ops.loaded_libraries:
        _require_loaded()
        return path
    if any(hasattr(torch.ops.vq2a8_ascendc, name) for name in ("projection", "cube_control")):
        raise RuntimeError("A different AscendC prototype is already loaded; use a fresh process.")
    import torch_npu  # noqa: F401 - register NPU before loading the native library

    torch.ops.load_library(str(path))
    for name in ("projection", "cube_control"):
        if not hasattr(torch.ops.vq2a8_ascendc, name):
            raise RuntimeError(f"Library did not register vq2a8_ascendc::{name}.")
    return path


def _require_loaded():
    if not all(hasattr(torch.ops.vq2a8_ascendc, name) for name in ("projection", "cube_control")):
        raise RuntimeError(
            "First build tools/build_vq2a8_ascendc.py, then call load_library(path). No fallback is enabled."
        )


def vq2a8_ascendc(activation, activation_scale, bias_correction, packed_indices, codebooks, codebook_tile_ids):
    _require_loaded()
    return torch.ops.vq2a8_ascendc.projection(
        activation, activation_scale, bias_correction, packed_indices, codebooks, codebook_tile_ids
    )


def cube_control(activation, synthetic_weight, *, bridge=False):
    _require_loaded()
    return torch.ops.vq2a8_ascendc.cube_control(activation, synthetic_weight, bridge)


def grouped_projection(inputs):
    """One native launch for <=6 same-N/K projections; no Python fallback."""
    _require_loaded()
    if not hasattr(torch.ops.vq2a8_ascendc, "grouped_projection"):
        raise RuntimeError("Rebuild the standalone AscendC library for grouped_projection.")
    if not 1 <= len(inputs) <= 6 or any(len(value) != 6 for value in inputs):
        raise ValueError("Grouped projection expects 1..6 six-tensor inputs.")
    return torch.ops.vq2a8_ascendc.grouped_projection(*(list(values) for values in zip(*inputs)))
