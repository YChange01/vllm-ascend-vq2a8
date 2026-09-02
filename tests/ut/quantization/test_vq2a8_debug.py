# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch

from vllm_ascend.quantization.vq2a8_debug import (
    VQ2A8_DEBUG_FAIL_FAST_ENV,
    comparison_summary,
    log_vq2a8_tensor,
)


def test_comparison_summary_reports_exact_match() -> None:
    values = torch.tensor([1.0, 2.0, 3.0])

    summary = comparison_summary(values, values.clone())

    assert summary["finite"] is True
    assert summary["cosine"] == 1.0
    assert summary["max_abs_error"] == 0.0


def test_comparison_summary_preserves_nonfinite_evidence() -> None:
    actual = torch.tensor([float("nan"), float("inf"), 1.0])
    expected = torch.tensor([0.0, 0.0, 1.0])

    summary = comparison_summary(actual, expected)

    assert summary["finite"] is False
    assert summary["actual_nan_count"] == 1
    assert summary["actual_inf_count"] == 1
    assert summary["max_abs_error"] == 0.0


def test_comparison_summary_reports_shape_mismatch_without_flattening() -> None:
    summary = comparison_summary(torch.zeros(2), torch.zeros(1, 2))

    assert summary["shape_match"] is False
    assert "max_abs_error" not in summary


def test_fail_fast_exception_includes_numerical_summary(monkeypatch) -> None:
    monkeypatch.setenv(VQ2A8_DEBUG_FAIL_FAST_ENV, "1")

    with pytest.raises(RuntimeError, match=r"nan_count=1, inf_count=1, finite_absmax=2.0"):
        log_vq2a8_tensor(
            scope="moe",
            stage="output",
            tensor=torch.tensor([float("nan"), float("inf"), -2.0]),
            call_index=0,
            layer_index=0,
            tp_rank=3,
        )
