# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json

import pytest
import torch

from tools import accept_vq2a8_ascendc_v3 as accept
from tools import benchmark_vq2a8_ascendc_v3 as bench
from tools import build_vq2a8_ascendc_v3 as build
from tools import validate_vq2a8_ascendc_v3 as validate


def record(tokens):
    return dict(prompt=[1, 2], tokens=tokens)


def test_model_observation_reports_token_divergence_and_only_compares_matching_prefixes():
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    actual = expected.clone()
    actual[0, 0] += 0.25
    result = bench.compare_model_logits(record([0, 1, 1]), expected, record([1, 1, 1]), actual)
    assert result["baseline_exact"] is False and result["tokens_exact"] is False
    assert result["model_tolerance"] is None and result["quality_verified"] is False
    assert [step["input_prefix_equal"] for step in result["steps"]] == [True, False, False]
    assert result["steps"][0]["max_abs_error"] == 0.25
    assert result["steps"][0]["relative_l2_error"] == pytest.approx(0.25 / 5**0.5)
    assert result["generation_warning"]


def test_model_observation_distinguishes_signed_zero_without_relaxing_exactness():
    expected, actual = torch.tensor([[0.0, 1.0]]), torch.tensor([[-0.0, 1.0]])
    result = bench.compare_model_logits(record([1]), expected, record([1]), actual)
    assert result["tokens_exact"] is True and result["baseline_exact"] is False
    assert result["steps"][0]["max_abs_error"] == 0
    assert result["steps"][0]["byte_mismatch_count"] == 1


def test_model_observation_zero_reference_uses_null_relative_error_instead_of_infinity():
    result = bench.compare_model_logits(record([0]), torch.zeros(1, 2), record([0]), torch.ones(1, 2))
    assert result["steps"][0]["relative_l2_error"] is None
    assert result["steps"][0]["reference_l2_zero"] is True


def test_empty_model_observation_cannot_be_declared_exact():
    with pytest.raises(ValueError, match="geometry"):
        bench.compare_model_logits(record([]), torch.empty(0, 2), record([]), torch.empty(0, 2))


@pytest.mark.parametrize("record_count,value_count", [(0, 0), (2, 0), (1, 2), (2, 3)])
def test_model_comparison_requires_two_complete_candidate_diagnostics(record_count, value_count):
    with pytest.raises(ValueError, match="Two candidate diagnostics"):
        bench.compare_case({}, "p2-o2", [{}] * record_count, [None] * value_count, 2, baseline_mode="exact")


def test_model_comparison_observe_accepts_difference_but_explicit_exact_rejects_it(monkeypatch):
    reference_record = record([1])
    reference = dict(library={}, diagnostics={"p2-o1": [reference_record, copy.deepcopy(reference_record)]})
    expected = torch.tensor([[1.0, 2.0]])
    monkeypatch.setattr(bench, "load_diagnostic", lambda *_: expected)
    values = [expected + 0.125, expected + 0.125]
    candidate = [record([1]), record([1])]
    result = bench.compare_case(reference, "p2-o1", candidate, values, 2, baseline_mode="observe")
    assert all(item["baseline_exact"] is False and item["quality_verified"] is False for item in result)
    with pytest.raises(ValueError, match="strict per-step"):
        bench.compare_case(reference, "p2-o1", candidate, values, 2, baseline_mode="exact")


def test_resident_accept_cli_keeps_exact_opt_in_and_passes_graph_preparation(tmp_path):
    args = accept.parse_args(
        [
            "--model",
            str(tmp_path),
            "--soc",
            "Ascend950PR_9599",
            "--baseline-mode",
            "exact",
            "--decode-graph",
            "moe",
            "--preparation",
            "fused",
            "--plan-only",
        ]
    )
    steps = accept.commands(args, tmp_path / "report")
    assert steps[-1][0] == "model-exact"
    command = steps[-1][1]
    assert command[command.index("--baseline-mode") + 1] == "exact"
    assert command[command.index("--decode-graph") + 1] == "moe"
    assert command[command.index("--preparation") + 1] == "fused"


def receipt():
    cases = {}
    for key in validate.expected_cases():
        count = int(key.split(":g")[1].split(":")[0]) if key.startswith(("synthetic:", "resident:")) else 1
        exact = key.startswith(("synthetic:", "resident:")) or key.endswith(":zero")
        cases[key] = dict(
            passed=True,
            repeat_exact=True,
            grouped_exact=True,
            current_stream_exact=True,
            oracle_exact_required=exact,
            native_path="grouped_projection_resident",
            resident_out_exact=True,
            descriptor_reuse_exact=True,
            oracle=[
                dict(allclose=True, mismatch_count=0, max_abs_error=0.0, relative_l2_error=0.0) for _ in range(count)
            ],
        )
    return dict(
        schema_version=validate.SCHEMA_VERSION,
        implementation="ascendc_v3",
        status="PASS",
        device_execution_verified=True,
        physical_runtime=True,
        resident_abi_version=1,
        native_resident_capabilities=3,
        preparation_mode="eager",
        layout="zn_pair_lut_k256",
        library={},
        model={},
        python_source_sha256={},
        physical_npu="0",
        soc="Ascend950PR_9599",
        cases=cases,
    )


@pytest.fixture
def resident_receipt(monkeypatch):
    monkeypatch.setattr(validate, "model_identity", lambda _: {})
    monkeypatch.setattr(validate, "python_source_hashes", lambda: {})
    return receipt()


def test_resident_receipt_accepts_additional_capability_bits(resident_receipt):
    validate.validate_receipt(resident_receipt, {}, None, "0")


def test_receipt_pins_the_graph_and_prepare_implementations(monkeypatch):
    monkeypatch.setattr(validate, "sha256", lambda path: str(path))
    sources = validate.python_source_hashes()
    assert "vllm_ascend/quantization/vq2a8_v3_graph.py" in sources
    assert "vllm_ascend/quantization/vq2a8_v3_workspace.py" in sources
    assert "tools/vq2a8_v3_prepare_check.py" in sources


def test_build_manifest_pins_resident_kernel_and_layout(tmp_path, capsys):
    args = build.parse_args(["--soc", "Ascend950PR_9599", "--build-dir", str(tmp_path), "--plan-only"])
    assert build.build(args) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["layout"] == "zn_pair_lut_k256"
    assert manifest["resident_kernel"] == "v2_pipeline_resident"
    library = tmp_path / build.LIBRARY_NAME
    library.write_bytes(b"manifest identity fixture, never loaded")
    manifest.update(status="built", library_sha256=build.sha256(library))
    path = tmp_path / "build-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate.library_identity(library)["resident_abi_version"] == 1
    for key in ("layout", "resident_kernel", "resident_abi_version", "resident_capabilities_required"):
        broken = copy.deepcopy(manifest)
        broken.pop(key)
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError, match="manifest"):
            validate.library_identity(library)


def prepare_receipt():
    from tools.vq2a8_v3_prepare_check import PREPARE_PREFLIGHT_CASES

    return dict(
        passed=True,
        exactbitwise=True,
        device_execution_verified=True,
        scope="fused_preparation_after_rht_and_bias_gemv",
        cases=[dict(name=name, passed=True, exactbitwise=True) for name in PREPARE_PREFLIGHT_CASES],
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "eager",
        "missing_checks",
        "capability",
        "negative_capability",
        "missing_case",
        "inexact",
        "duplicate",
        "malformed",
    ],
)
def test_fused_mode_cannot_reuse_eager_or_incomplete_preflight(resident_receipt, mutation):
    resident_receipt.update(preparation_mode="fused", preparation_checks=prepare_receipt())
    validate.validate_receipt(resident_receipt, {}, None, "0", "fused")
    if mutation == "eager":
        resident_receipt.update(preparation_mode="eager", preparation_checks=None)
    elif mutation == "missing_checks":
        resident_receipt["preparation_checks"] = None
    elif mutation == "capability":
        resident_receipt["native_resident_capabilities"] = 1
    elif mutation == "negative_capability":
        resident_receipt["native_resident_capabilities"] = -1
    elif mutation == "missing_case":
        resident_receipt["preparation_checks"]["cases"].pop()
    elif mutation == "duplicate":
        cases = resident_receipt["preparation_checks"]["cases"]
        cases[0] = cases[1]
    elif mutation == "malformed":
        resident_receipt["preparation_checks"]["cases"][0] = None
    else:
        resident_receipt["preparation_checks"]["cases"][0]["exactbitwise"] = False
    with pytest.raises(ValueError):
        validate.validate_receipt(resident_receipt, {}, None, "0", "fused")


def graph_layers(calls, prefill, captures, replays):
    return {
        str(index): dict(
            ready=True,
            layout="zn_pair_lut_k256",
            resident_abi_version=1,
            full_model_graph_verified=False,
            route_host_reads=0,
            descriptor_h2d_bytes=0,
            decode_calls=calls,
            resident_projection_launches=calls * 2,
            resident_prefill_launches=prefill,
            preparation_mode="fused",
            decode_graph=dict(
                scope="moe_decode",
                captures=captures,
                replays=replays,
                entries=1,
                failed=False,
                full_model_graph_verified=False,
            ),
        )
        for index in range(43)
    }


def test_measured_graph_requires_actual_replays_and_excludes_capture():
    before, after = graph_layers(3, 2, 1, 3), graph_layers(6, 3, 1, 6)
    bench.validate_residency(before, after, 3, measured=True, decode_graph="moe", preparation="fused")
    after["0"]["decode_graph"]["captures"] += 1
    with pytest.raises(ValueError, match="capture"):
        bench.validate_residency(before, after, 3, measured=True, decode_graph="moe", preparation="fused")
    after["0"]["decode_graph"]["captures"] -= 1
    after["0"]["decode_graph"]["replays"] -= 1
    with pytest.raises(ValueError, match="replay each decode"):
        bench.validate_residency(before, after, 3, measured=True, decode_graph="moe", preparation="fused")


def test_correctness_graph_requires_real_capture_and_each_token_replay():
    before, after = graph_layers(0, 0, 0, 0), graph_layers(3, 1, 1, 3)
    for layer in before.values():
        layer["decode_graph"]["entries"] = 0
    bench.validate_residency(before, after, 3, decode_graph="moe", preparation="fused")
    after["0"]["decode_graph"]["replays"] = 0
    with pytest.raises(ValueError, match="replay each decode"):
        bench.validate_residency(before, after, 3, decode_graph="moe", preparation="fused")
    after["0"]["decode_graph"].update(captures=0, entries=0, replays=3)
    with pytest.raises(ValueError, match="completed graph capture"):
        bench.validate_residency(before, after, 3, decode_graph="moe", preparation="fused")


def test_measured_first_capture_cannot_be_counted_as_hot_graph_timing():
    before, after = graph_layers(0, 0, 0, 0), graph_layers(3, 1, 1, 3)
    for layer in before.values():
        layer["decode_graph"]["entries"] = 0
    with pytest.raises(ValueError, match="without capture"):
        bench.validate_residency(before, after, 3, measured=True, decode_graph="moe", preparation="fused")


def test_graph_cannot_change_mode_or_report_work_when_disabled():
    before, after = graph_layers(3, 2, 1, 3), graph_layers(6, 3, 1, 6)
    after["0"]["decode_graph"]["scope"] = "none"
    with pytest.raises(ValueError, match="prior MoE decode graph evidence"):
        bench.validate_residency(before, after, 3)
    before["0"]["decode_graph"]["scope"] = "none"
    with pytest.raises(ValueError, match="Disabled graph"):
        bench.validate_residency(before, after, 3)


@pytest.mark.parametrize("warmups,repeats", [(0, 5), (1, 5), (2, 0), (2, 4)])
def test_standalone_timing_summary_requires_warmup_and_repeat_minimum(warmups, repeats):
    with pytest.raises(ValueError, match="warmups"):
        bench.summarize_samples([], [(2, 2)], warmups, repeats, {})


def test_single_token_prompt_is_outside_benchmark_prefill_contract():
    with pytest.raises(ValueError):
        bench.parse_cases("1:4")
    with pytest.raises(ValueError):
        bench.summarize_samples([], [(1, 4)], 2, 5, {})


def test_graph_and_preparation_report_cannot_replace_requested_modes():
    before, after = graph_layers(3, 2, 1, 3), graph_layers(6, 3, 1, 6)
    with pytest.raises(ValueError, match="preparation mode"):
        bench.validate_residency(before, after, 3, measured=True, decode_graph="moe", preparation="eager")
    with pytest.raises(ValueError, match="graph evidence"):
        bench.validate_residency(before, after, 3, measured=True, decode_graph="none", preparation="fused")


@pytest.mark.parametrize(
    "mutation", ["old_schema", "old_layout", "old_native_path", "capability", "reuse", "tolerance"]
)
def test_old_or_incomplete_native_evidence_cannot_certify_resident_kernel(resident_receipt, mutation):
    case = resident_receipt["cases"]["resident:k4096:g6"]
    if mutation == "old_schema":
        resident_receipt["schema_version"] = 1
    elif mutation == "old_layout":
        resident_receipt["layout"] = "v1"
    elif mutation == "old_native_path":
        case["native_path"] = "grouped_projection"
    elif mutation == "capability":
        resident_receipt["native_resident_capabilities"] = 2
    elif mutation == "reuse":
        case["descriptor_reuse_exact"] = False
    else:
        case["oracle"][0]["relative_l2_error"] = 0.031
    with pytest.raises(ValueError):
        validate.validate_receipt(resident_receipt, {}, None, "0")
