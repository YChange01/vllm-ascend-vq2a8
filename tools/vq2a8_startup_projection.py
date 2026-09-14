# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-free V3 resident projection cases for the startup diagnostic child.

The caller initializes logical NPU 0 and supplies a stage context manager
which synchronizes after each stage. These cases never open a model/artifact,
capture a graph, or establish full-model correctness or performance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _case_geometry(case):
    if case == "projection_m1":
        return 1, 2048, 1
    if case == "projection_m32":
        return 32, 4096, 1
    if case == "projection_group6":
        return 32, 4096, 6
    raise ValueError(f"Unknown startup projection case: {case}")


def _library_identity(library, runtime_soc):
    """Read build evidence without confusing a matching SoC with compatibility."""
    path = Path(library).resolve(strict=True)
    if not path.is_file() or path.suffix != ".so":
        raise ValueError("Projection diagnostics require an explicitly selected V3 .so library.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    manifest_path = path.parent / "build-manifest.json"
    identity = {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "runtime_soc": runtime_soc,
        "manifest_path": str(manifest_path),
        "manifest_present": manifest_path.is_file(),
        "build_soc": None,
        "soc_check": "UNKNOWN",
        "manifest_hash_check": "UNKNOWN",
        "compatibility_verified": False,
    }
    if not identity["manifest_present"]:
        return identity
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("V3 build manifest must contain a JSON object.")
    build_soc = manifest.get("soc")
    if build_soc is not None and (not isinstance(build_soc, str) or not build_soc):
        raise ValueError("V3 build manifest has an invalid SoC value.")
    identity["build_soc"] = build_soc
    if build_soc is not None:
        identity["soc_check"] = "MATCH" if build_soc == runtime_soc else "MISMATCH"
    if "library_sha256" in manifest:
        identity["manifest_hash_check"] = "MATCH" if manifest["library_sha256"] == identity["sha256"] else "MISMATCH"
    return identity


def run_case(case, stage, library):
    """Run one bounded synthetic case; return only JSON-serializable evidence.

    ``stage(name)`` must emit submission before synchronizing on context exit,
    so a host-call stall can be distinguished from an asynchronous kernel stall.
    Dense FP64 oracle matrices stay on CPU and are released one job at a time.
    """
    rows, reduction, jobs = _case_geometry(case)
    columns = 4096
    with stage("projection_imports"):
        # Lazy imports keep native/optional dependencies inside the child.
        import torch

        from tools.validate_vq2a8_ascendc_v2 import _convert_inputs, _synthetic
        from tools.validate_vq2a8_phase4_kernel import bitwise_equal, same_fp8_oracle, synthetic_dense_oracle
        from vllm_ascend.quantization.vq2a8_ascendc_v3 import grouped_projection_resident, load_pinned_library

    with stage("projection_library"):
        identity = _library_identity(library, torch.npu.get_device_name(0))
        print("STARTUP_PROJECTION_LIBRARY=" + json.dumps(identity, sort_keys=True), flush=True)
        if identity["soc_check"] == "MISMATCH":
            raise ValueError(
                f"V3 library was built for {identity['build_soc']}, but this NPU is {identity['runtime_soc']}; "
                "rebuild for the exact target before running the projection probe."
            )
        if identity["manifest_hash_check"] == "MISMATCH":
            raise ValueError("V3 library SHA256 differs from its build manifest; select a consistent build.")
        loaded = load_pinned_library(identity["path"], identity["sha256"])

    with stage("projection_synthetic_cpu"), torch.device("cpu"):
        synthetic = [_synthetic(rows, reduction, offset) for offset in range(jobs)]

    with stage("projection_prepare_h2d"), torch.device("cpu"):
        # This helper performs CPU zN/pair-LUT conversion, then H2D and the
        # byte-preserving activation gather. It does not load any real weights.
        inputs = [_convert_inputs(values, torch.device("npu:0")) for values in synthetic]

    with stage("projection_native"):
        outputs = grouped_projection_resident(inputs)

    with stage("projection_verify_cpu"):
        if len(outputs) != jobs:
            raise ValueError(f"Resident projection returned {len(outputs)} outputs, expected {jobs}.")
        checks = []
        for index, (values, output) in enumerate(zip(synthetic, outputs)):
            if output.shape != (rows, columns) or output.dtype != torch.bfloat16:
                raise ValueError(f"Projection job {index} has incorrect output shape or dtype.")
            if output.device != inputs[index][0].device:
                raise ValueError(f"Projection job {index} returned output on the wrong device.")
            # Transfers and scalar checks are intentional validation boundaries,
            # never part of a timing claim or a serving hot path.
            actual = output.detach().cpu()
            if not bool(torch.isfinite(actual.float()).all()):
                raise ValueError(f"Projection job {index} returned nonfinite values.")
            with torch.device("cpu"):
                dense = synthetic_dense_oracle(*values[3:])
                expected = same_fp8_oracle(values[:3], dense)
                del dense
            exact = bitwise_equal(actual, expected)
            check = {
                "job": index,
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "finite": True,
                "exact_bitwise": exact,
                "max_abs_error": float((actual.float() - expected.float()).abs().max()),
            }
            print("STARTUP_PROJECTION_CHECK=" + json.dumps(check, sort_keys=True), flush=True)
            if not exact:
                raise ValueError(f"Projection job {index} differs from the exact synthetic FP8 oracle.")
            checks.append(check)

    return {
        "scope": "synthetic_v3_resident_projection_only",
        "case": case,
        "rows": rows,
        "columns": columns,
        "reduction": reduction,
        "jobs": jobs,
        "library": identity,
        "loaded_library": loaded,
        "checks": checks,
        "device_input_bytes": sum(t.numel() * t.element_size() for values in inputs for t in values),
        "device_output_bytes": sum(t.numel() * t.element_size() for t in outputs),
        "single_cpu_dense_oracle_bytes": columns * reduction * 8,
        "model_weights_loaded": False,
        "full_model_verified": False,
        "graph_verified": False,
        "timing_valid": False,
    }
