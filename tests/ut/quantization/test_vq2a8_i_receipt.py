# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU receipt/report contracts for I; these tests do not execute an NPU."""

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.ut.quantization.test_vq2a8_jk_receipt import receipt_fixture as baseline_receipt_fixture
from tests.ut.quantization.test_vq2a8_v4_serving_model import isolated_model
from tools import validate_vq2a8_v4_decoder_graph as probe


def report_fixture():
    return {
        "swiglu_mode": "fused_select_sign",
        "effective_graph_mode": "none",
        "decoder": {"replays": 0},
        "swiglu_candidates": {
            "scope": probe.SWIGLU_SCOPE,
            "eager_reference": probe.SWIGLU_REFERENCE,
            "graph_build_calls": 43,
            "reference_calls": 100,
            "counters_prove_device_execution": False,
        },
    }


def receipt_fixture():
    args, receipt = baseline_receipt_fixture("vectorized", "baseline")
    args.swiglu_mode = "fused_select_sign"
    receipt["swiglu_mode"] = args.swiglu_mode
    receipt["swiglu_reference"] = probe.SWIGLU_REFERENCE
    receipt["graph"]["swiglu_mode"] = args.swiglu_mode
    receipt["graph"].update(
        requested_graph_mode="decoder", compute_backend="v2", activation_preparation="sign_fused_direct"
    )
    receipt["graph"]["swiglu_candidates"] = report_fixture()["swiglu_candidates"]
    before, after = report_fixture(), report_fixture()
    after["swiglu_candidates"]["reference_calls"] += 4
    for row in receipt["cases"]:
        row["decoder_replays"] = row["output"] - 1
        row["swiglu_eager_reference"] = probe.swiglu_eager_evidence(before, after)
    return args, receipt


def test_i_receipt_requires_original_torch_reference_and_actual_decoder_replays():
    probe.validate_receipt(*receipt_fixture())


@pytest.mark.parametrize(
    "field,bad",
    [
        ("requested_graph_mode", "moe"),
        ("compute_backend", "v1"),
        ("activation_preparation", "sign_fused_strided"),
        ("activation_reorder", "chunk_reuse2"),
        ("activation_reorder", "row_reuse"),
        ("b1_schedule", "tile_major"),
    ],
)
def test_i_receipt_must_confirm_exact_supported_graph_configuration(field, bad):
    args, receipt = receipt_fixture()
    receipt["graph"][field] = bad
    with pytest.raises(ValueError, match="I graph candidate"):
        probe.validate_receipt(args, receipt)


def test_i_receipt_per_case_reference_counts_cannot_exceed_total():
    args, receipt = receipt_fixture()
    receipt["graph"]["swiglu_candidates"]["reference_calls"] = 1
    with pytest.raises(ValueError, match="exceeds recorded original-Torch calls"):
        probe.validate_receipt(args, receipt)


@pytest.mark.parametrize("location", ("top", "graph"))
@pytest.mark.parametrize("bad", (None, "torch", "candidate", True))
def test_i_receipt_mode_must_match_at_every_level(location, bad):
    args, receipt = receipt_fixture()
    target = receipt if location == "top" else receipt["graph"]
    if bad is None:
        del target["swiglu_mode"]
    else:
        target["swiglu_mode"] = bad
    with pytest.raises(ValueError, match="I SwiGLU mode"):
        probe.validate_receipt(args, receipt)


@pytest.mark.parametrize("field", ("graph_build_calls", "reference_calls"))
@pytest.mark.parametrize("bad", (None, True, False, 0, -1, 1.0, "1"))
def test_i_receipt_no_startup_only_or_noninteger_counter_evidence(field, bad):
    args, receipt = receipt_fixture()
    receipt["graph"]["swiglu_candidates"][field] = bad
    with pytest.raises(ValueError, match="I graph candidate"):
        probe.validate_receipt(args, receipt)


@pytest.mark.parametrize("field", ("swiglu_reference", "eager_reference"))
@pytest.mark.parametrize("bad", (None, "selected_backend", "fused_select_sign", True))
def test_i_receipt_candidate_cannot_be_its_own_reference(field, bad):
    args, receipt = receipt_fixture()
    target = receipt if field == "swiglu_reference" else receipt["graph"]["swiglu_candidates"]
    target[field] = bad
    with pytest.raises(ValueError, match="independent original-Torch"):
        probe.validate_receipt(args, receipt)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("scope", "eager_and_graph"),
        ("scope", None),
        ("counters_prove_device_execution", True),
        ("counters_prove_device_execution", 0),
        ("counters_prove_device_execution", None),
    ],
)
def test_i_receipt_capture_counters_are_not_device_execution_proof(field, bad):
    args, receipt = receipt_fixture()
    receipt["graph"]["swiglu_candidates"][field] = bad
    with pytest.raises(ValueError, match="I graph candidate"):
        probe.validate_receipt(args, receipt)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("implementation", "fused_select_sign"),
        ("implementation", None),
        ("graph_disabled", False),
        ("graph_disabled", 1),
        ("reference_calls", 0),
        ("reference_calls", -1),
        ("reference_calls", True),
        ("reference_calls", 1.0),
        ("graph_build_calls", 1),
        ("graph_build_calls", False),
        ("graph_build_calls", None),
        ("decoder_replays", 1),
        ("decoder_replays", False),
        ("decoder_replays", None),
    ],
)
def test_i_receipt_each_eager_case_must_use_original_reference(field, bad):
    args, receipt = receipt_fixture()
    receipt["cases"][-1]["swiglu_eager_reference"][field] = bad
    with pytest.raises(ValueError, match="per-case I"):
        probe.validate_receipt(args, receipt)


@pytest.mark.parametrize("bad", (None, {}, False))
def test_i_receipt_startup_totals_cannot_replace_per_case_reference(bad):
    args, receipt = receipt_fixture()
    receipt["cases"][-1]["swiglu_eager_reference"] = bad
    with pytest.raises(ValueError, match="per-case I"):
        probe.validate_receipt(args, receipt)


@pytest.mark.parametrize("bad", (None, 0, True, 3.0, -1))
def test_i_receipt_every_candidate_case_requires_real_replays(bad):
    args, receipt = receipt_fixture()
    receipt["cases"][0]["decoder_replays"] = bad
    with pytest.raises(ValueError, match="per-case I"):
        probe.validate_receipt(args, receipt)


@pytest.mark.parametrize("corruption", ("status", "hardware", "cases", "replays"))
def test_i_counters_do_not_replace_original_model_acceptance(corruption):
    args, receipt = receipt_fixture()
    if corruption == "status":
        receipt["status"] = "FAIL"
    elif corruption == "hardware":
        receipt["hardware_execution_verified"] = False
    elif corruption == "cases":
        receipt["cases"].pop()
    else:
        receipt["graph"]["decoder"]["replays"] -= 1
    with pytest.raises(ValueError, match="real decoder replays"):
        probe.validate_receipt(args, receipt)


def test_i_eager_evidence_tracks_the_specific_pass_not_startup_counts():
    before = report_fixture()
    after = deepcopy(before)
    after["swiglu_candidates"]["reference_calls"] += 7
    assert probe.swiglu_eager_evidence(before, after) == {
        "implementation": probe.SWIGLU_REFERENCE,
        "graph_disabled": True,
        "reference_calls": 7,
        "graph_build_calls": 0,
        "decoder_replays": 0,
    }


@pytest.mark.parametrize("field", ("reference_calls", "graph_build_calls", "replays"))
def test_i_eager_evidence_rejects_missing_reference_or_any_candidate_call(field):
    before = report_fixture()
    after = deepcopy(before)
    after["swiglu_candidates"]["reference_calls"] += 7
    if field == "reference_calls":
        after["swiglu_candidates"][field] = before["swiglu_candidates"][field]
    elif field == "graph_build_calls":
        after["swiglu_candidates"][field] += 1
    else:
        after["decoder"][field] += 1
    with pytest.raises(AssertionError, match="without graph build or replay"):
        probe.swiglu_eager_evidence(before, after)


@pytest.mark.parametrize("which", ("before", "after"))
@pytest.mark.parametrize("mode", ("decoder", None, False))
def test_i_eager_evidence_requires_disabled_graph_for_both_snapshots(which, mode):
    before, after = report_fixture(), report_fixture()
    after["swiglu_candidates"]["reference_calls"] += 1
    (before if which == "before" else after)["effective_graph_mode"] = mode
    with pytest.raises(AssertionError, match="graphs disabled"):
        probe.swiglu_eager_evidence(before, after)


@pytest.mark.parametrize("bad", (None, True, -1, 1.0))
@pytest.mark.parametrize("field", ("reference_calls", "graph_build_calls", "replays"))
def test_i_eager_evidence_rejects_invalid_counters(field, bad):
    before, after = report_fixture(), report_fixture()
    target = before["decoder"] if field == "replays" else before["swiglu_candidates"]
    target[field] = bad
    with pytest.raises(AssertionError, match="original-Torch eager-reference report"):
        probe.swiglu_eager_evidence(before, after)


def test_i_default_torch_keeps_legacy_receipts_compatible():
    args, receipt = baseline_receipt_fixture("vectorized", "baseline")
    probe.validate_receipt(args, receipt)
    args.swiglu_mode = "torch"
    probe.validate_receipt(args, receipt)
    receipt["swiglu_mode"] = "fused_select_sign"
    with pytest.raises(ValueError, match="I SwiGLU mode"):
        probe.validate_receipt(args, receipt)


def test_i_model_report_aggregates_only_host_dispatch_counters():
    model, owner, _, _ = isolated_model()
    assert model._v4_swiglu_mode == "torch"
    model._v4_swiglu_mode = "fused_select_sign"
    for index, layer in owner.layers.items():
        layer.v4_swiglu_graph_build_calls = index + 1
        layer.v4_swiglu_reference_calls = (index + 1) * 2
    report = model.v4_graph_report()
    assert report["swiglu_mode"] == "fused_select_sign"
    assert report["swiglu_candidates"] == {
        "scope": probe.SWIGLU_SCOPE,
        "eager_reference": probe.SWIGLU_REFERENCE,
        "graph_build_calls": 3,
        "reference_calls": 6,
        "counters_prove_device_execution": False,
    }


def test_i_decoder_probe_forwards_mode_to_engine_options():
    tree = ast.parse(Path(probe.__file__).read_text(encoding="utf-8"))
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "offline_engine_options"
    )
    value = next(keyword.value for keyword in call.keywords if keyword.arg == "v4_swiglu_mode")
    assert ast.unparse(value) == "getattr(args, 'swiglu_mode', 'torch')"


def test_i_model_tolerance_remains_original_strict_threshold():
    def result(value):
        return SimpleNamespace(token_ids=[7], logprobs=[{7: SimpleNamespace(logprob=value)}])

    assert probe.compare_outputs(result(0.0), result(1e-4)) == 1e-4
    with pytest.raises(AssertionError, match="beyond 1e-4/1e-5"):
        probe.compare_outputs(result(0.0), result(1.001e-4))
