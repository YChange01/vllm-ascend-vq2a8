# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Header-only residency accounting; no device execution or peak-fit claim."""

import json
from types import SimpleNamespace as NS

import pytest

from vllm_ascend.quantization import vq2a8_execution_v3 as v3


def model_layer(index, experts):
    shapes, specs = {}, {}
    for kind, k in (("gate_up", 4096), ("down", 2048)):
        n, tiles = 4096, k // 256
        specs[kind] = NS(rows=n, columns=k, rht_true_columns=k, rht_block_size=128)
        shapes.update(
            {
                f"{kind}_packed_indices": (experts, n // 2, k // 8),
                f"{kind}_codebooks": (experts, tiles, n // 32, 16, 2),
                f"{kind}_codebook_tile_ids": (experts, k),
                **{f"{kind}_{field}": (experts, k) for field in v3.TRANSFORM_FIELDS},
            }
        )
    return NS(layer_index=index, expert_ids=tuple(range(experts)), tensor_shapes=shapes, specs=specs)


def read_budget(capsys):
    line = capsys.readouterr().out.strip()
    assert line.startswith("V3_RESIDENCY_BUDGET ")
    return json.loads(line.split(" ", 1)[1])


def test_v3_resident_budget_reports_before_rejecting_without_tensor_allocation(capsys, monkeypatch):
    monkeypatch.setattr(v3.torch, "empty", lambda *args, **kwargs: pytest.fail("header planning must not allocate"))
    layers = [model_layer(0, 1)]
    required = v3.resident_plan(layers, 1 << 40)["planned_bytes"]
    with pytest.raises(ValueError, match="by 512 bytes; no cache fallback"):
        v3.resident_plan(layers, required - 512, report_budget=True)
    report = read_budget(capsys)
    assert report["required_bytes"] == required
    assert report["budget_bytes"] == required - 512
    assert report["shortfall_bytes"] == 512
    assert report["headroom_bytes"] == -512
    assert report["shortfall_gib"] == 512 / v3.GIB
    assert report["fits"] is False


def test_v3_resident_budget_reports_exact_fit_and_preserves_plan(capsys):
    layers = [model_layer(0, 1)]
    required = v3.resident_plan(layers, 1 << 40)["planned_bytes"]
    quiet = v3.resident_plan(layers, required)
    assert capsys.readouterr().out == ""
    assert v3.resident_plan(layers, required, report_budget=True) == quiet
    report = read_budget(capsys)
    assert report["fits"] is True
    assert report["headroom_bytes"] == report["shortfall_bytes"] == 0
    assert report["payload_bytes"] + report["workspace_bytes"] == report["required_bytes"]


def test_v3_zero_available_budget_still_reports_required_bytes(capsys):
    with pytest.raises(ValueError, match="exceeds budget 0"):
        v3.resident_plan([model_layer(0, 1)], 0, report_budget=True)
    report = read_budget(capsys)
    assert report["fits"] is False and report["budget_bytes"] == 0
    assert report["required_bytes"] == report["shortfall_bytes"] > 0


def test_v3_real_geometry_budget_explains_why_point99_is_not_a_fix(capsys):
    layers = [model_layer(i, 1 if i < 3 else 256) for i in range(43)]
    # Previously observed total and roots allocation; not a new hardware run.
    total, allocated, reserve = 86_067_118_080, 15_909_779_456, 3 * v3.GIB
    plan = v3.resident_plan(layers, total)
    # Converted banks replace uint8 tile IDs with int64 activation order.
    assert plan["payload_bytes"] == 66_079_641_600 + 10_243 * (4096 + 2048) * (8 - 1)
    assert plan["workspace_bytes"] == 61_252_096
    assert plan["planned_bytes"] == 66_581_424_640
    with pytest.raises(ValueError, match="no cache fallback"):
        v3.resident_plan(layers, int(total * 0.99) - allocated - reserve, report_budget=True)
    assert read_budget(capsys)["shortfall_bytes"] == 505_982_669
