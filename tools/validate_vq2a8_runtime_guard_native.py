#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only acceptance for the native batched runtime guard.

Use --library for the built V4/V2 extension, or --build-cpu to compile only
runtime_guard_binding.cpp with the installed PyTorch C++ toolchain. CPU success
does not certify NPU graph replay, model integration, or serving performance.
"""

from __future__ import annotations

# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import gc
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "csrc/vq2a8_ascendc_v4_v2/runtime_guard_binding.cpp"
MUTATIONS = (
    "replacement",
    "same_storage_view",
    "pointer",
    "shape",
    "stride",
    "offset",
    "dtype",
    "device",
    "layout",
    "count",
    "order",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--library", type=Path)
    source.add_argument("--build-cpu", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def mutate(values, name):
    import torch

    x = values[0]
    if name == "replacement":
        values[0] = x.clone()
    elif name == "same_storage_view":
        values[0] = x.view_as(x)
    elif name == "pointer":
        x.set_(x.clone())
    elif name == "shape":
        x.resize_(2, 8)
    elif name == "stride":
        x.transpose_(0, 1)
    elif name == "offset":
        x.as_strided_((4, 4), (4, 1), 1)
    elif name == "dtype":
        x.data = x.double()
    elif name == "device":
        values[0] = torch.empty_like(x, device="meta")
    elif name == "layout":
        values[0] = x.to_sparse()
    elif name == "count":
        values.pop()
    elif name == "order":
        values.reverse()
    else:
        raise ValueError(name)


def reject(fn, detail):
    try:
        fn()
    except (RuntimeError, ValueError):
        return
    raise AssertionError(f"Native runtime guard accepted {detail}")


def run_checks(factory):
    import torch

    checked = []
    for mutation in MUTATIONS:
        values = [torch.arange(20, dtype=torch.float32)[:16].reshape(4, 4), torch.arange(8, dtype=torch.int64)]
        plan = factory(values, ["root.weight", "banks.lookup"])
        plan.check(values)
        plan.check(values)  # Repeated valid calls must not inhibit later checks.
        mutate(values, mutation)
        reject(lambda plan=plan, values=values: plan.check(values), mutation)
        checked.append(mutation)

    # Aggregation copies old snapshots; it may not capture changed metadata.
    values = [torch.arange(16).reshape(4, 4)]
    original = factory(values, ["old_snapshot"])
    values[0].transpose_(0, 1)
    combined = factory([], [])
    combined.append(original)
    reject(lambda: combined.check(values), "recaptured metadata during append")
    checked.append("append_preserves_snapshot")
    reject(lambda: combined.append(original), "append after checking")
    checked.append("append_startup_only")

    # Tensor payload updates are deliberately outside this host-only contract.
    values = [torch.zeros(4)]
    stable = factory(values, ["payload_not_read"])
    values[0].fill_(float("nan"))
    stable.check(values)
    checked.append("payload_not_read")

    tensor = torch.zeros(32)
    storage = tensor.untyped_storage()
    weak_storage = storage._weak_ref()
    try:
        retained = factory([tensor], ["strong_owner"])
        del tensor, storage
        gc.collect()
        if torch.UntypedStorage._expired(weak_storage):
            raise AssertionError("Native guard did not strongly retain tensor storage")
        merged = factory([], [])
        merged.append(retained)
        del retained
        gc.collect()
        if torch.UntypedStorage._expired(weak_storage):
            raise AssertionError("Merged native guard did not strongly retain tensor storage")
        del merged
        gc.collect()
        if not torch.UntypedStorage._expired(weak_storage):
            raise AssertionError("Native guard leaked retained tensor storage")
    finally:
        torch.UntypedStorage._free_weak_ref(weak_storage)
    checked.append("strong_owner_and_release")
    tensor = torch.zeros(32)
    storage = tensor.untyped_storage()
    weak_storage = storage._weak_ref()
    try:
        retained = factory([tensor], ["original_storage_after_set"])
        del storage
        tensor.set_(tensor.clone())
        gc.collect()
        if torch.UntypedStorage._expired(weak_storage):
            raise AssertionError("Native guard lost captured storage after in-place set_")
        reject(
            lambda retained=retained, tensor=tensor: retained.check([tensor]), "changed storage on the same TensorImpl"
        )
        del retained
        gc.collect()
        if not torch.UntypedStorage._expired(weak_storage):
            raise AssertionError("Native guard leaked original storage after set_")
    finally:
        torch.UntypedStorage._free_weak_ref(weak_storage)
    checked.append("original_storage_retained_after_set")
    reject(lambda: factory([torch.ones(1)], []), "labels length")
    reject(lambda: factory([torch.empty(1, device="meta")], ["meta"]), "meta capture")
    reject(lambda: factory([torch.eye(2).to_sparse()], ["sparse"]), "sparse capture")
    checked.extend(("labels_length", "meta_capture", "sparse_capture"))
    return checked


def main(argv=None):
    args = parse_args(argv)
    result = {
        "scope": "host_tensor_metadata_only",
        "status": "PLANNED",
        "native_host_verified": False,
        "device_execution_verified": False,
        "graph_verified": False,
        "model_integration_verified": False,
        "performance_verified": False,
        "source": str(SOURCE),
        "library": str(args.library) if args.library is not None else None,
        "build_cpu": args.build_cpu,
    }
    if args.plan_only:
        print(json.dumps(result, indent=2))
        return 0
    try:
        import torch

        if args.build_cpu:
            from torch.utils.cpp_extension import load

            digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:12]
            path = load(
                name="vq2a8_runtime_guard_host_" + digest,
                sources=[str(SOURCE)],
                is_python_module=False,
                verbose=True,
            )
        else:
            path = str(args.library.resolve(strict=True))
            torch.ops.load_library(path)
        version = torch.ops.vq2a8_ascendc_v4_v2.runtime_guard_version()
        if type(version) is not int or version != 1:
            raise RuntimeError("Runtime guard requires independent ABI 1; no fallback")
        result.update(library=str(path), native_abi=version)
        result["checks"] = run_checks(torch.classes.vq2a8_ascendc_v4_v2.RuntimeTensorGuard)
        result.update(status="PASS", native_host_verified=True)
    except Exception as error:
        result.update(status="FAIL", error=f"{type(error).__name__}: {error}")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
