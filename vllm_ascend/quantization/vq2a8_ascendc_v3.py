# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate V3 namespace; the out ABI is for trusted, owned workspaces only.

Descriptor pointers are not an untrusted user-facing tensor API. Python and
the native binding validate geometry, while the resident runtime owns every
pointer target until completion. No V1/V2 library is substituted on failure.
"""

import hashlib
from pathlib import Path

import regex as re
import torch

MAX_JOBS = 6
JOB_WORDS = 12
CONSTANT_WORDS = 6656


def _require_loaded():
    if not all(
        hasattr(torch.ops.vq2a8_ascendc_v3, name)
        for name in ("projection", "grouped_projection", "grouped_projection_out", "make_constants")
    ):
        raise RuntimeError("Load the explicitly built V3 library first; no V1/V2 fallback is enabled.")


def load_pinned_library(path, expected_sha256):
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("V3 requires an explicit lowercase SHA256 library identity.")
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.suffix != ".so":
        raise ValueError("Expected an explicitly built V3 .so library.")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_sha256:
        raise ValueError("V3 library differs from the pinned candidate.")
    if str(path) not in torch.ops.loaded_libraries:
        if hasattr(torch.ops.vq2a8_ascendc_v3, "projection"):
            raise RuntimeError("Another V3 library is loaded; use a fresh process.")
        import torch_npu  # noqa: F401 - NPU registration before native loading

        torch.ops.load_library(str(path))
    _require_loaded()
    return {"path": str(path), "sha256": actual, "namespace": "vq2a8_ascendc_v3", "abi_version": 1}


def grouped_projection_v3(inputs, *, pipeline=False):
    """V1-compatible eager input ABI, for prefill/preflight, not device decode."""
    _require_loaded()
    if type(pipeline) is not bool or not 1 <= len(inputs) <= MAX_JOBS or any(len(row) != 6 for row in inputs):
        raise ValueError("V3 grouped projection requires 1..6 six-tensor jobs and a boolean pipeline flag.")
    name = "grouped_projection_pipeline" if pipeline else "grouped_projection"
    if not hasattr(torch.ops.vq2a8_ascendc_v3, name):
        raise RuntimeError(f"The loaded V3 library has no {name}; no fallback enabled.")
    return getattr(torch.ops.vq2a8_ascendc_v3, name)(*(list(values) for values in zip(*inputs)))


def make_constants(anchor):
    _require_loaded()
    return torch.ops.vq2a8_ascendc_v3.make_constants(anchor)


def grouped_projection_out(descriptors, constants, owners, *, jobs, m, n, k, tiles, pipeline=False):
    """Launch without constructing/uploading host descriptors or output allocation."""
    _require_loaded()
    if (
        any(type(value) is not int for value in (jobs, m, n, k, tiles))
        or not 1 <= jobs <= MAX_JOBS
        or m != 1
        or not 0 < n <= 65536
        or n % 32
        or not 0 < k <= 65536
        or k % 512
        or not 1 <= tiles <= 256
        or type(pipeline) is not bool
    ):
        raise ValueError("Invalid V3 projection geometry.")
    if descriptors.shape != (jobs, JOB_WORDS) or descriptors.dtype != torch.int64 or not descriptors.is_contiguous():
        raise ValueError("Expected contiguous int64[jobs,12] descriptors.")
    if constants.shape != (CONSTANT_WORDS,) or constants.dtype != torch.int32 or not constants.is_contiguous():
        raise ValueError("Expected contiguous int32[6656] constants.")
    if not owners or any(value.device != descriptors.device for value in (*owners, constants)):
        raise ValueError("All V3 pointer owners must be retained on the descriptor device.")
    return torch.ops.vq2a8_ascendc_v3.grouped_projection_out(
        descriptors, constants, list(owners), jobs, m, n, k, tiles, pipeline
    )
