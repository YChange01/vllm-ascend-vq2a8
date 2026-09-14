# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small startup-operation probes; no model, checkpoint, or packed experts.

The caller initializes NPU 0 and supplies a stage context manager which logs
entry, logs SUBMITTED after its body returns, then synchronizes and logs exit.
Imports are deliberately lazy so listing cases does not initialize a device.
These probes isolate execution failures, not TPOT or model numerical quality.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from types import SimpleNamespace

HC_MULT = 4
HC_HIDDEN_SIZE = 4096
HC_MIX = (2 + HC_MULT) * HC_MULT
HC_ITERATIONS = 20
HC_EPS = 1e-6
NORM_EPS = 1e-6
PROFILE_ROWS = 128
PROJECTION_ROWS = 32
MAX_PREPARATION_JOBS = 6
RHT_BLOCK = 128
PREPARATION_WIDTHS = (4096, 2048)
CASES = ("basic", "hc_pre_m2", "hc_pre_m128", "hc_post_m128", "prepare_m32", "prepare_group6")
StageFactory = Callable[[str], AbstractContextManager]


def _check_tensor(torch, tensor, shape, dtype):
    """Check downloaded data only; device work has a separate named stage."""
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
        raise ValueError(f"Unexpected tensor: shape={tuple(tensor.shape)}, dtype={tensor.dtype}; want {shape}, {dtype}")
    if not bool(torch.isfinite(tensor.float()).all()):
        raise ValueError("Non-finite probe output")


def _hc_operator(torch, name, stage):
    with stage("hc.import_utils"):
        from vllm_ascend.utils import bootstrap_custom_op_env

    with stage("hc.bootstrap_custom_ops"):
        bootstrap_custom_op_env()
    with stage("hc.import_extension"):
        # enable_custom_op() intentionally disables registration on A5. Match
        # the existing isolated QLI/SAS probes, without constructing a model.
        extension = importlib.import_module("vllm_ascend.vllm_ascend_C")
        print(
            "STARTUP_HC_LIBRARY="
            + json.dumps({"operator": name, "extension_path": getattr(extension, "__file__", None)}, sort_keys=True),
            flush=True,
        )
    with stage("hc.resolve_operator"):
        op = getattr(torch.ops._C_ascend, name, None)
        if op is None:
            raise RuntimeError(f"Loaded extension does not register _C_ascend.{name}")
    return op, getattr(extension, "__file__", None)


def _basic(torch, device, stage):
    with stage("basic.setup_cpu"):
        x_cpu = torch.ones((PROJECTION_ROWS, RHT_BLOCK), dtype=torch.bfloat16, device="cpu")
        w_cpu = torch.full((RHT_BLOCK, PROJECTION_ROWS), 0.125, dtype=torch.bfloat16, device="cpu")
    with stage("basic.upload"):
        x, weight = x_cpu.to(device), w_cpu.to(device)
    with stage("basic.add"):
        value = x + 1
    with stage("basic.matmul"):
        output = value @ weight
    with stage("basic.download"):
        actual = output.cpu()
    with stage("basic.verify"):
        _check_tensor(torch, actual, (PROJECTION_ROWS, PROJECTION_ROWS), torch.bfloat16)
        if not bool((actual == 32).all()):
            raise ValueError("Basic arithmetic/matmul probe: expected every output to equal 32")
    return {"scope": "basic_tensor_ops", "verification": "shape_dtype_finite_and_constant_result"}


def _hc_pre(torch, device, rows, stage):
    op, extension_path = _hc_operator(torch, "npu_hc_pre_v2", stage)
    with stage("hc_pre.setup_cpu"):
        rng = torch.Generator(device="cpu").manual_seed(1024)
        x_cpu = torch.randn(
            (rows, HC_MULT, HC_HIDDEN_SIZE), generator=rng, dtype=torch.float32, device="cpu"
        ).bfloat16()
        fn_cpu = (
            torch.randn((HC_MIX, HC_MULT * HC_HIDDEN_SIZE), generator=rng, dtype=torch.float32, device="cpu") * 0.01
        )
        scale_cpu = torch.randn(3, generator=rng, dtype=torch.float32, device="cpu") * 0.01
        base_cpu = torch.randn(HC_MIX, generator=rng, dtype=torch.float32, device="cpu") * 0.01
    with stage("hc_pre.upload"):
        x, fn, scale, base = (value.to(device) for value in (x_cpu, fn_cpu, scale_cpu, base_cpu))
    with stage("hc_pre.call"):
        outputs = op(x, fn, scale, base, HC_MULT, HC_ITERATIONS, NORM_EPS, HC_EPS)
    with stage("hc_pre.download"):
        actual = tuple(value.cpu() for value in outputs)
    with stage("hc_pre.verify"):
        expected = (
            ((rows, HC_HIDDEN_SIZE), torch.bfloat16),
            ((rows, HC_MULT), torch.float32),
            ((rows, HC_MULT, HC_MULT), torch.float32),
        )
        if len(actual) != len(expected):
            raise ValueError("HCPre must return y, post, and comb")
        for value, (shape, dtype) in zip(actual, expected, strict=True):
            _check_tensor(torch, value, shape, dtype)
    return {
        "scope": "hc_pre_v2_only",
        "rows": rows,
        "input_shape": [rows, HC_MULT, HC_HIDDEN_SIZE],
        "extension_path": extension_path,
        "verification": "shape_dtype_finite_only",
    }


def _hc_post(torch, device, stage):
    op, extension_path = _hc_operator(torch, "npu_hc_post", stage)
    with stage("hc_post.setup_cpu"):
        rng = torch.Generator(device="cpu").manual_seed(1025)
        x_cpu = torch.randn(
            (1, PROFILE_ROWS, HC_HIDDEN_SIZE), generator=rng, dtype=torch.float32, device="cpu"
        ).bfloat16()
        residual_cpu = torch.randn(
            (1, PROFILE_ROWS, HC_MULT, HC_HIDDEN_SIZE), generator=rng, dtype=torch.float32, device="cpu"
        ).bfloat16()
        post_cpu = torch.rand((1, PROFILE_ROWS, HC_MULT), generator=rng, dtype=torch.float32, device="cpu")
        comb_cpu = (
            torch.rand((1, PROFILE_ROWS, HC_MULT, HC_MULT), generator=rng, dtype=torch.float32, device="cpu") / HC_MULT
        )
    with stage("hc_post.upload"):
        x, residual, post, comb = (value.to(device) for value in (x_cpu, residual_cpu, post_cpu, comb_cpu))
    with stage("hc_post.call"):
        output = op(x, residual, post, comb)
    with stage("hc_post.download"):
        actual = output.cpu()
    with stage("hc_post.verify"):
        _check_tensor(torch, actual, (1, PROFILE_ROWS, HC_MULT, HC_HIDDEN_SIZE), torch.bfloat16)
    return {
        "scope": "hc_post_only_independent_inputs",
        "rows": PROFILE_ROWS,
        "extension_path": extension_path,
        "verification": "shape_dtype_finite_only",
    }


def _prepare(torch, device, jobs, stage):
    with stage("prepare.import"):
        from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation

    results = []
    for width in PREPARATION_WIDTHS:
        prefix = f"prepare.k{width}"
        with stage(f"{prefix}.setup_cpu"):
            rng = torch.Generator(device="cpu").manual_seed(1026 + width)
            spec = SimpleNamespace(columns=width, rht_true_columns=width, rht_block_size=RHT_BLOCK)
            host_requests = []
            for index in range(jobs):
                hidden = torch.randn(
                    (PROJECTION_ROWS, width), generator=rng, dtype=torch.float32, device="cpu"
                ).bfloat16()
                payload = {
                    "weight_scale": torch.full((width,), 1.0 + index / 100, dtype=torch.float32, device="cpu"),
                    "weight_bias": torch.randn(width, generator=rng, dtype=torch.float32, device="cpu") * 0.01,
                    "rht_sign": torch.where(torch.arange(width, device="cpu") % 3 == 0, -1, 1).to(torch.int8),
                }
                host_requests.append((hidden, payload, spec))
            validity = []
            preparation = RowwiseVQ2A8Preparation(compact=True, validity=validity.append)
        with stage(f"{prefix}.upload"):
            requests = [
                (hidden.to(device), {key: value.to(device) for key, value in payload.items()}, current)
                for hidden, payload, current in host_requests
            ]
        with stage(f"{prefix}.hadamard"):
            # Residency normally warms this constant before the dummy forward.
            preparation._ensure_hadamard(device, RHT_BLOCK)
        with stage(f"{prefix}.call_many"):
            outputs = preparation.many(requests)
        with stage(f"{prefix}.download"):
            actual = [
                (q.view(torch.uint8).cpu().view(torch.float8_e4m3fn), scale.cpu(), bias.cpu())
                for q, scale, bias in outputs
            ]
            flags = [flag.cpu() for flag in validity]
        with stage(f"{prefix}.verify"):
            if len(actual) != jobs or len(flags) != 1 or not all(bool(flag) for flag in flags):
                raise ValueError("Preparation output count or input validity failed")
            for quantized, scale, bias in actual:
                _check_tensor(torch, quantized, (PROJECTION_ROWS, width), torch.float8_e4m3fn)
                _check_tensor(torch, scale, (PROJECTION_ROWS,), torch.float32)
                _check_tensor(torch, bias, (PROJECTION_ROWS,), torch.float32)
                if not bool((scale > 0).all()):
                    raise ValueError("Preparation produced a non-positive activation scale")
        results.append({"width": width, "jobs": jobs, "rows_per_job": PROJECTION_ROWS})
    return {
        "scope": "rowwise_eager_preparation_only_no_projection_kernel",
        "geometry": results,
        "assignment_rows_per_width": jobs * PROJECTION_ROWS,
        "verification": "shape_dtype_finite_positive_scale_and_input_validity_only",
    }


def run_case(case: str, stage: StageFactory) -> dict:
    """Run one case on the caller's initialized logical NPU 0.

    ``prepare_group6`` is six 32-row projection jobs, i.e. 192 routed
    assignments, not a claim of 192 unique source tokens. Both preparation
    cases exercise widths 4096 and 2048 with the real eager implementation.
    The caller owns process isolation, timeouts, device visibility and logs.
    """
    if case not in CASES:
        raise ValueError(f"Unknown startup operation case: {case}")
    with stage("case.import_torch"):
        import torch

    device = torch.device("npu:0")
    with torch.inference_mode():
        if case == "basic":
            return _basic(torch, device, stage)
        if case.startswith("hc_pre_"):
            return _hc_pre(torch, device, 2 if case == "hc_pre_m2" else PROFILE_ROWS, stage)
        if case == "hc_post_m128":
            return _hc_post(torch, device, stage)
        return _prepare(torch, device, 1 if case == "prepare_m32" else MAX_PREPARATION_JOBS, stage)
