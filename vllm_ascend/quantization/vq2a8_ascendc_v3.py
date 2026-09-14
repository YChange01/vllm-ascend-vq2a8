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
RESIDENT_JOB_WORDS = 9
RESIDENT_ABI_VERSION = 1
RESIDENT_PROJECTION = 1
FUSED_PREPARATION = 2
RESIDENT_TP2_PROJECTION = 4
TP1_RESIDENT_SHAPES = ((4096, 2048), (4096, 4096))
TP2_RESIDENT_SHAPES = ((2048, 4096), (4096, 2048))


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
    resident_library_capabilities()
    return {
        "path": str(path),
        "sha256": actual,
        "namespace": "vq2a8_ascendc_v3",
        "abi_version": 1,
        "resident_abi_version": RESIDENT_ABI_VERSION,
    }


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


def resident_library_capabilities(*, require_tp2=False):
    """Reject an old V3 binary before constructing any resident workspace."""
    if type(require_tp2) is not bool:
        raise ValueError("require_tp2 must be boolean.")
    namespace = torch.ops.vq2a8_ascendc_v3
    if not all(
        hasattr(namespace, name)
        for name in (
            "resident_abi_version",
            "resident_capabilities",
            "grouped_projection_resident",
            "grouped_projection_resident_out",
        )
    ):
        raise RuntimeError("Rebuild the V3 library: the resident pair-LUT ABI is unavailable.")
    if namespace.resident_abi_version() != RESIDENT_ABI_VERSION:
        raise RuntimeError("V3 resident ABI version mismatch; rebuild the selected library.")
    capabilities = namespace.resident_capabilities()
    if type(capabilities) is not int or capabilities < 0:
        raise RuntimeError("V3 resident capability response must be a nonnegative integer.")
    if capabilities & RESIDENT_PROJECTION != RESIDENT_PROJECTION:
        raise RuntimeError("V3 library does not support resident pair-LUT projection.")
    if require_tp2 and capabilities & RESIDENT_TP2_PROJECTION != RESIDENT_TP2_PROJECTION:
        raise RuntimeError("Rebuild the V3 library: TP2 resident projection capability is unavailable.")
    return capabilities


def grouped_projection_resident(inputs, *, tp_size=1):
    """Grouped prefill using the converted layout in the V3 library."""
    if type(tp_size) is not int or tp_size not in (1, 2):
        raise ValueError("Resident projection tp_size must be integer 1 or 2.")
    if not 1 <= len(inputs) <= MAX_JOBS or any(len(row) != 5 for row in inputs):
        raise ValueError("Resident grouped projection requires 1..6 five-tensor jobs.")
    shapes = TP2_RESIDENT_SHAPES if tp_size == 2 else TP1_RESIDENT_SHAPES
    for x, _, _, packed, _ in inputs:
        if (
            x.ndim != 2
            or packed.ndim != 4
            or not 1 <= x.shape[0] <= 32
            or (packed.shape[0] * 32, x.shape[1]) not in shapes
        ):
            raise ValueError(f"Resident V3 TP{tp_size} requires M1..32 and (N,K) in {shapes}.")
    if tp_size == 2:
        resident_library_capabilities(require_tp2=True)
    return torch.ops.vq2a8_ascendc_v3.grouped_projection_resident(*(list(values) for values in zip(*inputs)))


def grouped_projection_resident_out(descriptors, owners, *, jobs, m, n, k, tp_size=1):
    """Trusted nine-word workspace; no descriptor upload or output allocation."""
    if type(tp_size) is not int or tp_size not in (1, 2):
        raise ValueError("Resident projection tp_size must be integer 1 or 2.")
    shapes = TP2_RESIDENT_SHAPES if tp_size == 2 else TP1_RESIDENT_SHAPES
    if (
        any(type(value) is not int for value in (jobs, m, n, k))
        or not 1 <= jobs <= MAX_JOBS
        or m != 1
        or (n, k) not in shapes
    ):
        raise ValueError(f"Resident V3 TP{tp_size} requires 1..6 M1 jobs with (N,K) in {shapes}.")
    if (
        descriptors.shape != (jobs, RESIDENT_JOB_WORDS)
        or descriptors.dtype != torch.int64
        or not descriptors.is_contiguous()
    ):
        raise ValueError("Expected contiguous int64[jobs,9] resident descriptors.")
    if not owners or any(value.device != descriptors.device for value in owners):
        raise ValueError("All resident pointer owners must share the descriptor device.")
    if tp_size == 2:
        resident_library_capabilities(require_tp2=True)
    return torch.ops.vq2a8_ascendc_v3.grouped_projection_resident_out(descriptors, list(owners), jobs, m, n, k)


def prepare_resident_out(rotated, weight_scale, order, input_bias, quantized, scale, bias, valid):
    """Fuse row scaling, FP8 quantization and byte permutation into owned outputs.

    RHT and bias GEMV retain their existing implementation and run before this
    operation. Native validation and validity flags remain active during replay.
    """
    return torch.ops.vq2a8_ascendc_v3.prepare_out(
        rotated, weight_scale, order, input_bias, quantized, scale, bias, valid
    )
