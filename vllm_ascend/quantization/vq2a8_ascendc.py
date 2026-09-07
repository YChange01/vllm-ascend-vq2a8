# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicitly loaded native AscendC prototype; no default backend integration.

Only an already-built library can be loaded. There is no JIT, subprocess,
Triton implementation or fallback in the projection path. C++ validates the
tensor metadata even when callers bypass this Python wrapper.
"""

from pathlib import Path

import torch


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
