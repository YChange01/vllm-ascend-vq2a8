# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only fault injection for J/K real-model validation receipts."""

from itertools import product
from types import SimpleNamespace

import pytest

from tests.ut.quantization.test_vq2a8_decoder_input_integration import receipt_fixture as input_receipt_fixture
from tools import validate_vq2a8_v4_decoder_graph as probe

JK_COMBINATIONS = tuple(
    (reorder, schedule)
    for reorder, schedule in product(("vectorized", "chunk_reuse2", "chunk_reuse4"), ("baseline", "tile_major"))
    if reorder != "vectorized" or schedule != "baseline"
)


def receipt_fixture(reorder="chunk_reuse2", schedule="baseline"):
    modes = dict(
        runtime_guard="planned",
        select_sign="fused",
        activation_tail="torch",
        validity_mode="fused_vectorized",
        route_mapping="fused",
        activation_reorder=reorder,
        b1_schedule=schedule,
    )
    args = SimpleNamespace(**modes, decoder_metadata_mode="recursive", decoder_input_mode="general")
    # Same exact case/replay matrix required by the existing ABCD validator.
    receipt = dict(
        status="PASS",
        hardware_execution_verified=True,
        **modes,
        projection_reference="vectorized_baseline",
        cases=[
            {"round": round_id, "prompt": prompt, "output": output}
            for round_id in range(probe.REUSE_ROUNDS)
            for prompt, output in probe.CASES
        ],
        graph={
            **modes,
            "decoder": {"replays": probe.REUSE_ROUNDS * sum(count - 1 for _, count in probe.CASES)},
            "projection_candidates": {
                "scope": "graph_build_only_eager_and_prefill_vectorized",
                "graph_build_calls": 1,
                "reference_calls": 1,
                "counters_prove_device_execution": False,
            },
        },
    )
    return args, receipt


@pytest.mark.parametrize("reorder,schedule", JK_COMBINATIONS)
def test_jk_receipt_accepts_each_independent_and_combined_candidate(reorder, schedule):
    args, receipt = receipt_fixture(reorder, schedule)
    probe.validate_receipt(args, receipt)


@pytest.mark.parametrize("reorder,schedule", JK_COMBINATIONS)
@pytest.mark.parametrize("location", ("top", "graph"))
@pytest.mark.parametrize("field", ("activation_reorder", "b1_schedule"))
@pytest.mark.parametrize("corruption", ("missing", "wrong", "bool"))
def test_jk_receipt_requested_mode_must_match_every_level(reorder, schedule, location, field, corruption):
    args, receipt = receipt_fixture(reorder, schedule)
    target = receipt if location == "top" else receipt["graph"]
    if corruption == "missing":
        del target[field]
    else:
        target[field] = True if corruption == "bool" else "wrong"
    with pytest.raises(ValueError, match="J/K"):
        probe.validate_receipt(args, receipt)


@pytest.mark.parametrize("reorder,schedule", JK_COMBINATIONS)
@pytest.mark.parametrize("field", ("graph_build_calls", "reference_calls"))
@pytest.mark.parametrize("bad", (None, True, False, 0, -1, 1.0, "1"))
def test_jk_receipt_candidate_and_independent_reference_must_really_execute(reorder, schedule, field, bad):
    args, receipt = receipt_fixture(reorder, schedule)
    receipt["graph"]["projection_candidates"][field] = bad
    with pytest.raises(ValueError, match="independent vectorized reference"):
        probe.validate_receipt(args, receipt)


@pytest.mark.parametrize("reorder,schedule", JK_COMBINATIONS)
@pytest.mark.parametrize("field", ("scope", "graph_build_calls", "reference_calls"))
def test_jk_receipt_missing_candidate_evidence_is_failure(reorder, schedule, field):
    args, receipt = receipt_fixture(reorder, schedule)
    del receipt["graph"]["projection_candidates"][field]
    with pytest.raises(ValueError, match="J/K"):
        probe.validate_receipt(args, receipt)


@pytest.mark.parametrize("reorder,schedule", JK_COMBINATIONS)
def test_jk_receipt_missing_projection_candidates_is_failure(reorder, schedule):
    args, receipt = receipt_fixture(reorder, schedule)
    del receipt["graph"]["projection_candidates"]
    with pytest.raises(ValueError, match="J/K"):
        probe.validate_receipt(args, receipt)


@pytest.mark.parametrize("scope", (None, "", "eager_and_graph", "graph_only", True))
def test_jk_receipt_scope_cannot_include_candidate_eager_reference(scope):
    args, receipt = receipt_fixture()
    receipt["graph"]["projection_candidates"]["scope"] = scope
    with pytest.raises(ValueError, match="J/K"):
        probe.validate_receipt(args, receipt)


@pytest.mark.parametrize("reorder,schedule", JK_COMBINATIONS)
@pytest.mark.parametrize("reference", (None, "selected_backend", "row_reuse", "candidate", True))
def test_jk_receipt_same_candidate_cannot_be_its_own_reference(reorder, schedule, reference):
    args, receipt = receipt_fixture(reorder, schedule)
    receipt["projection_reference"] = reference
    with pytest.raises(ValueError, match="independent vectorized reference"):
        probe.validate_receipt(args, receipt)


def test_jk_receipt_reference_label_cannot_be_missing():
    args, receipt = receipt_fixture()
    del receipt["projection_reference"]
    with pytest.raises(ValueError, match="J/K"):
        probe.validate_receipt(args, receipt)


@pytest.mark.parametrize("bad", [None, True, 0, "false"])
def test_jk_build_counters_cannot_claim_device_execution(bad):
    args, receipt = receipt_fixture()
    receipt["graph"]["projection_candidates"]["counters_prove_device_execution"] = bad
    with pytest.raises(ValueError, match="J/K"):
        probe.validate_receipt(args, receipt)


@pytest.mark.parametrize("corruption", ("status", "hardware", "case", "replays"))
def test_jk_receipt_candidate_counters_do_not_replace_model_evidence(corruption):
    args, receipt = receipt_fixture("chunk_reuse4", "tile_major")
    if corruption == "status":
        receipt["status"] = "FAIL"
    elif corruption == "hardware":
        receipt["hardware_execution_verified"] = False
    elif corruption == "case":
        receipt["cases"].pop()
    else:
        receipt["graph"]["decoder"]["replays"] -= 1
    with pytest.raises(ValueError, match="real decoder replays"):
        probe.validate_receipt(args, receipt)


def test_jk_receipt_baseline_still_requires_no_new_candidate_fields():
    args, receipt = receipt_fixture("vectorized", "baseline")
    del args.activation_reorder
    del args.b1_schedule
    for target in (receipt, receipt["graph"]):
        del target["activation_reorder"]
        del target["b1_schedule"]
    del receipt["projection_reference"]
    del receipt["graph"]["projection_candidates"]
    probe.validate_receipt(args, receipt)


@pytest.mark.parametrize("reorder,schedule", JK_COMBINATIONS)
def test_jk_receipt_composes_with_existing_g_input_path_evidence(reorder, schedule):
    args, receipt = input_receipt_fixture()
    args.activation_reorder = reorder
    args.b1_schedule = schedule
    for target in (receipt, receipt["graph"]):
        target["activation_reorder"] = reorder
        target["b1_schedule"] = schedule
    receipt["projection_reference"] = "vectorized_baseline"
    receipt["graph"]["projection_candidates"] = {
        "scope": "graph_build_only_eager_and_prefill_vectorized",
        "graph_build_calls": 86,
        "reference_calls": 172,
        "counters_prove_device_execution": False,
    }
    probe.validate_receipt(args, receipt)
    receipt["graph"]["projection_candidates"]["reference_calls"] = 0
    with pytest.raises(ValueError, match="J/K"):
        probe.validate_receipt(args, receipt)
