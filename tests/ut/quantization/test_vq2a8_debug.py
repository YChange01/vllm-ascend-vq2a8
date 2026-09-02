# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch

from vllm_ascend.quantization.vq2a8_debug import comparison_summary


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
