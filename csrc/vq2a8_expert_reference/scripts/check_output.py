#!/usr/bin/env python3
"""Read-only synthetic kernel output checker; not a model quality gate.

Both tolerances must be supplied explicitly. Acceptance compares actual BF16
values with the FP32 Golden rounded once to BF16. Unrounded FP32 Golden error
metrics are also reported, but are not a second, hidden acceptance threshold.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import ml_dtypes
import numpy as np

SCOPE = "synthetic_kernel_output_not_model_quality"


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _validate_tolerances(rtol: float, atol: float) -> None:
    for name, value in (("rtol", rtol), ("atol", atol)):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be an explicit finite non-negative number")


def _error_metrics(actual: np.ndarray, expected: np.ndarray, *, rtol: float, atol: float) -> dict:
    difference = np.abs(actual - expected)
    magnitude = np.abs(expected)
    nonzero = magnitude != 0
    # Very large explicitly selected tolerances may exceed float64 range.
    with np.errstate(over="ignore"):
        mismatches = difference > (atol + rtol * magnitude)
    return {
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "max_relative_error_nonzero_reference": float((difference[nonzero] / magnitude[nonzero]).max())
        if nonzero.any()
        else None,
        "zero_reference_nonzero_output_count": int(np.count_nonzero((~nonzero) & (difference != 0))),
        "mismatch_count_at_explicit_tolerances": int(np.count_nonzero(mismatches)),
    }


def compare_arrays(output: np.ndarray, golden: np.ndarray, *, rtol: float, atol: float) -> dict:
    """Validate typed arrays and return a JSON-safe, explicitly scoped result."""
    _validate_tolerances(rtol, atol)
    if not isinstance(output, np.ndarray) or output.dtype != np.dtype(ml_dtypes.bfloat16):
        raise TypeError("output must have bfloat16 dtype; implicit conversion is not allowed")
    if not isinstance(golden, np.ndarray) or golden.dtype != np.dtype(np.float32):
        raise TypeError("golden must have float32 dtype; implicit conversion is not allowed")
    if output.ndim != 2 or output.shape != golden.shape or not output.size:
        raise ValueError("output and golden must have the same nonempty two-dimensional shape")
    actual_f32 = output.astype(np.float32)
    if not np.isfinite(actual_f32).all():
        raise ValueError("output contains NaN or infinity")
    if not np.isfinite(golden).all():
        raise ValueError("golden contains NaN or infinity")
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = golden.astype(ml_dtypes.bfloat16)
    rounded_f32 = rounded.astype(np.float32)
    if not np.isfinite(rounded_f32).all():
        raise ValueError("finite FP32 golden overflows when rounded to BF16")
    actual_f64 = actual_f32.astype(np.float64)
    bf16_metrics = _error_metrics(actual_f64, rounded_f32.astype(np.float64), rtol=rtol, atol=atol)
    fp32_metrics = _error_metrics(actual_f64, golden.astype(np.float64), rtol=rtol, atol=atol)
    bit_mismatches = int(np.count_nonzero(output.view(np.uint16) != rounded.view(np.uint16)))
    return {
        "status": "passed" if bf16_metrics["mismatch_count_at_explicit_tolerances"] == 0 else "failed",
        "scope": SCOPE,
        "model_quality_verified": False,
        "acceptance_reference": "bf16_rounded_fp32_golden",
        "acceptance_rule": "abs(output - reference) <= atol + rtol * abs(reference), for every element",
        "tolerances": {"rtol": float(rtol), "atol": float(atol), "source": "explicit_user_selection"},
        "shape": list(output.shape),
        "elements": int(output.size),
        "finite": True,
        "bf16_rounded_golden": {**bf16_metrics, "bit_mismatch_count": bit_mismatches, "bit_exact": bit_mismatches == 0},
        "fp32_golden": {**fp32_metrics, "role": "reported_error_metrics_not_acceptance_reference"},
    }


def _metadata_shape(metadata: object) -> tuple[int, int]:
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    if metadata.get("output_dtype") != "bfloat16" or metadata.get("golden_output_dtype") != "float32":
        raise ValueError("metadata must declare output_dtype=bfloat16 and golden_output_dtype=float32")
    shape = metadata.get("output_shape")
    if not isinstance(shape, list) or len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("metadata output_shape must contain two positive integers")
    for key, expected in (("total_m", shape[0]), ("n", shape[1])):
        value = metadata.get(key)
        if type(value) is not int or value != expected:
            raise ValueError(f"metadata {key} must match output_shape")
    return shape[0], shape[1]


def _read_exact(path: Path, dtype: np.dtype, shape: tuple[int, int]) -> np.ndarray:
    expected_elements = math.prod(shape)
    expected_bytes = expected_elements * dtype.itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"{path}: expected {expected_bytes} bytes, found {actual_bytes}")
    array = np.fromfile(path, dtype=dtype)
    if array.size != expected_elements:
        raise ValueError(f"{path}: file size changed while reading")
    return array.reshape(shape)


def check_files(metadata_path: Path, output_path: Path, golden_path: Path, *, rtol: float, atol: float) -> dict:
    """Read little-endian files written on Ascend's supported AArch64 target."""
    _validate_tolerances(rtol, atol)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shape = _metadata_shape(metadata)
    raw_output = _read_exact(output_path, np.dtype("<u2"), shape)
    output = raw_output.astype(np.uint16, copy=False).view(ml_dtypes.bfloat16)
    golden = _read_exact(golden_path, np.dtype("<f4"), shape).astype(np.float32, copy=False)
    result = compare_arrays(output, golden, rtol=rtol, atol=atol)
    result["files"] = {
        "metadata": str(metadata_path.resolve()),
        "output": str(output_path.resolve()),
        "golden": str(golden_path.resolve()),
        "byte_order": "little_endian",
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="input/metadata.json from the generator")
    parser.add_argument("--output", type=Path, required=True, help="BF16 output/output_c.bin from the expert runner")
    parser.add_argument("--golden", type=Path, required=True, help="FP32 output/golden_c.bin from the generator")
    parser.add_argument(
        "--rtol", type=float, required=True, help="Explicit relative tolerance; no default quality threshold"
    )
    parser.add_argument(
        "--atol", type=float, required=True, help="Explicit absolute tolerance; no default quality threshold"
    )
    try:
        args = parser.parse_args(argv)
        result = check_files(args.metadata, args.output, args.golden, rtol=args.rtol, atol=args.atol)
        exit_code = 0 if result["status"] == "passed" else 1
    except (OSError, ValueError, TypeError, OverflowError) as error:
        result = {"status": "failed", "scope": SCOPE, "model_quality_verified": False, "error": str(error)}
        exit_code = 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
