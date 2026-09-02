# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Opt-in numerical tracing for full-model VQ2A8 bring-up."""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Any

import torch

logger = logging.getLogger(__name__)

VQ2A8_DEBUG_ENV = "VLLM_ASCEND_VQ2A8_DEBUG"
VQ2A8_DEBUG_LAYERS_ENV = "VLLM_ASCEND_VQ2A8_DEBUG_LAYERS"
VQ2A8_DEBUG_MAX_CALLS_ENV = "VLLM_ASCEND_VQ2A8_DEBUG_MAX_CALLS"
VQ2A8_DEBUG_FAIL_FAST_ENV = "VLLM_ASCEND_VQ2A8_DEBUG_FAIL_FAST"
VQ2A8_DEBUG_COMPARE_REFERENCE_ENV = "VLLM_ASCEND_VQ2A8_DEBUG_COMPARE_REFERENCE"

_DEFAULT_LAYERS = frozenset((0, 1, 2, 3, 21, 42))
_debug_call_counts: defaultdict[tuple[str, int | None], int] = defaultdict(int)


def _enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _selected_layers() -> frozenset[int] | None:
    value = os.getenv(VQ2A8_DEBUG_LAYERS_ENV)
    if value is None or not value.strip():
        return _DEFAULT_LAYERS
    if value.strip().lower() in ("*", "all"):
        return None
    try:
        return frozenset(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(
            f"{VQ2A8_DEBUG_LAYERS_ENV} must be a comma-separated list of decoder layers or 'all', got {value!r}."
        ) from exc


def vq2a8_debug_enabled(layer_index: int | None = None) -> bool:
    if not _enabled(VQ2A8_DEBUG_ENV):
        return False
    selected = _selected_layers()
    return layer_index is None or selected is None or layer_index in selected


def vq2a8_debug_compare_reference_enabled(
    layer_index: int | None = None,
) -> bool:
    return vq2a8_debug_enabled(layer_index) and _enabled(VQ2A8_DEBUG_COMPARE_REFERENCE_ENV)


def reserve_vq2a8_debug_call(
    scope: str,
    layer_index: int | None = None,
) -> int | None:
    """Reserve a bounded trace call and return its zero-based call index."""
    if not vq2a8_debug_enabled(layer_index):
        return None
    try:
        maximum = int(os.getenv(VQ2A8_DEBUG_MAX_CALLS_ENV, "1"))
    except ValueError as exc:
        raise ValueError(f"{VQ2A8_DEBUG_MAX_CALLS_ENV} must be an integer.") from exc
    if maximum < 1:
        return None
    key = (scope, layer_index)
    call_index = _debug_call_counts[key]
    if call_index >= maximum:
        return None
    _debug_call_counts[key] += 1
    return call_index


def tensor_summary(tensor: torch.Tensor, max_values: int = 8) -> dict[str, Any]:
    """Synchronously summarize a tensor; intended only for opt-in debugging."""
    result: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
    }
    if tensor.numel() == 0:
        return result

    values = tensor.detach().float().cpu()
    flat = values.flatten()
    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    result.update(
        finite=bool(finite.all()),
        nan_count=int(torch.isnan(flat).sum()),
        inf_count=int(torch.isinf(flat).sum()),
        zero_count=int((flat == 0).sum()),
        first_values=flat[:max_values].tolist(),
    )
    if finite_values.numel():
        result.update(
            min=float(finite_values.min()),
            max=float(finite_values.max()),
            mean=float(finite_values.mean()),
            absmax=float(finite_values.abs().max()),
            l2=float(torch.linalg.vector_norm(finite_values.double())),
        )
    return result


def log_vq2a8_tensor(
    *,
    scope: str,
    stage: str,
    tensor: torch.Tensor,
    call_index: int,
    layer_index: int | None = None,
    tp_rank: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = tensor_summary(tensor)
    record: dict[str, Any] = {
        "scope": scope,
        "stage": stage,
        "call": call_index,
        "layer": layer_index,
        "tp_rank": tp_rank,
        "tensor": summary,
    }
    if extra:
        record["extra"] = extra
    logger.warning("[VQ2A8_DEBUG] %s", json.dumps(record, sort_keys=True))
    if _enabled(VQ2A8_DEBUG_FAIL_FAST_ENV) and not summary.get("finite", True):
        raise RuntimeError(
            f"VQ2A8 produced a non-finite tensor at {scope}.{stage} "
            f"(layer={layer_index}, call={call_index}, tp_rank={tp_rank}, "
            f"shape={summary['shape']}, dtype={summary['dtype']}, "
            f"nan_count={summary.get('nan_count', 0)}, "
            f"inf_count={summary.get('inf_count', 0)}, "
            f"finite_absmax={summary.get('absmax')})."
        )
    return summary


def log_vq2a8_event(
    *,
    scope: str,
    stage: str,
    call_index: int,
    layer_index: int | None = None,
    tp_rank: int | None = None,
    values: dict[str, Any] | None = None,
) -> None:
    record = {
        "scope": scope,
        "stage": stage,
        "call": call_index,
        "layer": layer_index,
        "tp_rank": tp_rank,
        "values": values or {},
    }
    logger.warning("[VQ2A8_DEBUG] %s", json.dumps(record, sort_keys=True))


def comparison_summary(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, Any]:
    """Compare two tensors after synchronously copying them to CPU."""
    result: dict[str, Any] = {
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
    }
    if actual.shape != expected.shape:
        result["shape_match"] = False
        return result
    result["shape_match"] = True
    actual_cpu = actual.detach().float().cpu().reshape(-1)
    expected_cpu = expected.detach().float().cpu().reshape(-1)
    if actual_cpu.numel() == 0:
        result.update(finite=True, numel=0)
        return result

    paired_finite = torch.isfinite(actual_cpu) & torch.isfinite(expected_cpu)
    result.update(
        numel=actual_cpu.numel(),
        finite=bool(paired_finite.all()),
        actual_nan_count=int(torch.isnan(actual_cpu).sum()),
        expected_nan_count=int(torch.isnan(expected_cpu).sum()),
        actual_inf_count=int(torch.isinf(actual_cpu).sum()),
        expected_inf_count=int(torch.isinf(expected_cpu).sum()),
    )
    if not paired_finite.any():
        return result
    actual_finite = actual_cpu[paired_finite].double()
    expected_finite = expected_cpu[paired_finite].double()
    difference = actual_finite - expected_finite
    actual_norm = torch.linalg.vector_norm(actual_finite)
    expected_norm = torch.linalg.vector_norm(expected_finite)
    denominator = actual_norm * expected_norm
    cosine = (
        torch.dot(actual_finite, expected_finite) / denominator
        if denominator > 0
        else torch.tensor(float("nan"), dtype=torch.double)
    )
    result.update(
        max_abs_error=float(difference.abs().max()),
        mean_abs_error=float(difference.abs().mean()),
        rmse=float(torch.sqrt(torch.mean(difference.square()))),
        cosine=float(cosine),
        actual_absmax=float(actual_finite.abs().max()),
        expected_absmax=float(expected_finite.abs().max()),
    )
    return result


def log_vq2a8_comparison(
    *,
    scope: str,
    stage: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    call_index: int,
    layer_index: int | None = None,
    tp_rank: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = comparison_summary(actual, expected)
    if extra:
        summary["extra"] = extra
    log_vq2a8_event(
        scope=scope,
        stage=f"{stage}_comparison",
        call_index=call_index,
        layer_index=layer_index,
        tp_rank=tp_rank,
        values=summary,
    )
    if _enabled(VQ2A8_DEBUG_FAIL_FAST_ENV) and not summary.get("finite", True):
        raise RuntimeError(
            f"VQ2A8 reference comparison found non-finite values at "
            f"{scope}.{stage} (layer={layer_index}, call={call_index}, "
            f"tp_rank={tp_rank})."
        )
    return summary


def log_vq2a8_logits(
    logits: torch.Tensor,
    *,
    call_index: int,
    tp_rank: int | None = None,
    top_k: int = 8,
) -> None:
    summary = log_vq2a8_tensor(
        scope="model",
        stage="logits",
        tensor=logits,
        call_index=call_index,
        tp_rank=tp_rank,
    )
    if logits.numel() == 0:
        return
    last_logits = logits.detach().float().reshape(-1, logits.shape[-1])[-1].cpu()
    finite_logits = torch.nan_to_num(last_logits, nan=-torch.inf)
    count = min(top_k, finite_logits.numel())
    values, indices = torch.topk(finite_logits, count)
    log_vq2a8_event(
        scope="model",
        stage="logits_topk",
        call_index=call_index,
        tp_rank=tp_rank,
        values={
            "ids": indices.tolist(),
            "values": values.tolist(),
            "all_zero": summary.get("zero_count") == summary.get("numel"),
        },
    )
