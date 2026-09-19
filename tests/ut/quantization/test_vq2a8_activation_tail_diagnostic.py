# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only diagnostic orchestration contracts; not Ascend compilation proof."""

import copy
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import diagnose_vq2a8_activation_tail as probe


class TailStandIn:
    def __init__(self, *, mismatch=False, instrumentation_drift=False):
        self.mismatch = mismatch
        self.instrumentation_drift = instrumentation_drift

    def activation_tail_diagnostic_version(self):
        return 1

    def stages(self, *inputs):
        values = probe.torch_tail_stages(*inputs)
        if self.mismatch:
            values["divided_scale"] = torch.nextafter(
                values["divided_scale"], torch.full_like(values["divided_scale"], float("inf"))
            )
            values["scale"] = values["divided_scale"].clamp(min=1e-12)
            values["normalized"] = values["transformed"] / values["scale"][:, None]
            values["clamped"] = values["normalized"].clamp(-448, 448)
            values["q"] = values["clamped"].to(torch.float8_e4m3fn)
        invalid = values["valid"] == 0
        if bool(invalid.any()):
            values["q"].view(torch.uint8)[invalid] = 0x7F
            values["scale"][invalid] = float("nan")
        return values

    def activation_tail_diagnostic(self, *inputs):
        values = self.stages(*inputs)
        if self.instrumentation_drift:
            values["scale"] = values["scale"] + 1
        return list(values.values())

    def activation_quantize(self, *inputs):
        values = self.stages(*inputs)
        return values["q"], values["scale"], values["valid"]

    @staticmethod
    def activation_sign(x, weights, biases, signs):
        valid = torch.isfinite(x).all(-1) & torch.isfinite(weights).all(-1) & torch.isfinite(biases).all(-1)
        return x * signs.float(), valid.int()


def inputs(rows=3, width=2048):
    generator = torch.Generator().manual_seed(123)
    return (
        torch.randn(rows, width, generator=generator),
        torch.randn(rows, width, generator=generator),
        torch.zeros(rows),
    )


def test_plan_is_opt_in_diagnostic_and_starts_with_old_failure(tmp_path, capsys):
    directory = tmp_path / "unused"
    assert probe.main(["--plan-only", "--report-dir", str(directory)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "PLANNED"
    assert value["case_order"][0] == "legacy_m32_g6_k2048"
    assert value["max_saved_rows_per_case"] == 2
    for name in ("device_execution_verified", "preparation_verified", "graph_verified", "performance_verified"):
        assert value[name] is False
    assert not directory.exists()


@pytest.mark.parametrize(
    "option,value",
    [
        ("--max-mismatches", 0),
        ("--max-mismatches", 65),
        ("--max-saved-rows", -1),
        ("--max-saved-rows", 5),
        ("--physical-npu", -1),
        ("--timeout-s", 0),
    ],
)
def test_cli_bounds(option, value):
    with pytest.raises(SystemExit):
        probe.parse_args([option, str(value)])


def test_child_mapping_and_task_queue_do_not_override_explicit_policy(tmp_path):
    args = probe.parse_args(["--physical-npu", "1", "--max-mismatches", "3", "--max-saved-rows", "1"])
    command = probe.child_command(args, tmp_path)
    child = probe.parse_args(command[command.index("--child") :])
    assert child.child and child.report_dir == tmp_path
    assert child.max_mismatches == 3 and child.max_saved_rows == 1
    environment = probe.probe_environment(args, {"DEVICE_ID": "5"})
    assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "1"
    assert environment["TASK_QUEUE_ENABLE"] == "1"
    assert "DEVICE_ID" not in environment
    assert probe.probe_environment(args, {"TASK_QUEUE_ENABLE": "0"})["TASK_QUEUE_ENABLE"] == "0"


@pytest.mark.parametrize("version", [True, False, "1", 0, 2])
def test_abi_must_be_exact_integer(version):
    native = TailStandIn()
    native.activation_tail_diagnostic_version = lambda: version
    with pytest.raises(ValueError, match="ABI mismatch"):
        probe.require_diagnostic_abi(native)


def test_absent_abi_has_no_fallback():
    with pytest.raises(ValueError, match="Rebuild"):
        probe.require_diagnostic_abi(SimpleNamespace())


def test_bit_report_signed_zero_nan_and_one_ulp_are_not_hidden():
    expected = torch.tensor([0.0, 1.0, float("nan")], dtype=torch.float32)
    actual = expected.clone()
    actual[0] = -0.0
    actual[1] = torch.nextafter(actual[1], torch.tensor(float("inf")))
    actual.view(torch.int32)[2] = 0x7FC00001
    report = probe.compare_bits(actual, expected, limit=2)
    assert report["equal"] is False and report["unequal_elements"] == 3
    assert len(report["samples"]) == 2
    assert report["samples"][0]["actual_hex"] == "0x80000000"
    assert report["samples"][0]["expected_hex"] == "0x00000000"
    assert report["samples"][1]["ulp_distance"] == 1
    assert json.dumps(report, allow_nan=False)
    nan = probe.compare_bits(actual[2:], expected[2:], limit=1)
    assert nan["samples"][0]["ulp_distance"] is None


def test_fp8_bit_report_and_coordinate_are_exact():
    expected = torch.tensor([[0x00, 0x01], [0x38, 0x7E]], dtype=torch.uint8).view(torch.float8_e4m3fn)
    actual = expected.clone()
    actual.view(torch.uint8)[1, 0] = 0x39
    result = probe.compare_bits(actual, expected, limit=1)
    assert result["unequal_bytes"] == result["unequal_elements"] == 1
    assert result["samples"][0]["index"] == [1, 0]
    assert result["samples"][0]["actual_hex"] == "0x39"
    assert "ulp_distance" not in result["samples"][0]


def test_bit_report_rejects_contract_difference():
    with pytest.raises(ValueError, match="shape/dtype"):
        probe.compare_bits(torch.ones(2), torch.ones(1), limit=2)


def test_equal_case_has_no_saved_artifact(tmp_path):
    result = probe.diagnose_case(TailStandIn(), inputs(), limit=4, save_rows=2, directory=tmp_path)
    assert result["first_mismatch_stage"] is None
    assert result["diagnostic_matches_legacy_native"] is True
    assert result["first_mismatch_attribution_supported"] is False
    assert result["saved_rows"] == []
    assert list(tmp_path.iterdir()) == []


def test_mismatch_reports_first_stage_and_saves_only_bounded_synthetic_rows(tmp_path):
    result = probe.diagnose_case(TailStandIn(mismatch=True), inputs(), limit=3, save_rows=1, directory=tmp_path)
    assert result["first_mismatch_stage"] == "divided_scale"
    assert result["diagnostic_matches_legacy_native"] is True
    assert result["first_mismatch_attribution_supported"] is True
    assert len(result["saved_rows"]) == 1
    saved = result["saved_rows"][0]
    assert saved["bytes"] < 200000
    value = torch.load(saved["path"], weights_only=True)
    assert value["first_mismatch_stage"] == "divided_scale"
    assert value["rotated"].shape == (1, 2048)
    assert all(tensor.device.type == "cpu" for tensor in value.values() if isinstance(tensor, torch.Tensor))
    assert value["actual_scale"].view(torch.int32).item() != value["expected_scale"].view(torch.int32).item()


def test_do_not_attribute_diagnostic_changed_legacy_outputs(tmp_path):
    result = probe.diagnose_case(
        TailStandIn(instrumentation_drift=True), inputs(), limit=2, save_rows=0, directory=tmp_path
    )
    assert result["diagnostic_matches_legacy_native"] is False
    assert result["first_mismatch_attribution_supported"] is False
    assert result["first_mismatch_stage"] == "scale"


def test_diagnostic_cannot_overwrite_saved_rows(tmp_path):
    (tmp_path / "mismatch_row_0.pt").write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        probe.diagnose_case(TailStandIn(mismatch=True), inputs(), limit=1, save_rows=1, directory=tmp_path)
    assert (tmp_path / "mismatch_row_0.pt").read_bytes() == b"preserve"


def test_invalid_case_does_not_misdiagnose_native_poison_as_rounding(tmp_path):
    values = inputs()
    values[2][0] = float("nan")
    with pytest.raises(AssertionError, match="invalid rows"):
        probe.diagnose_case(TailStandIn(), values, limit=1, save_rows=0, directory=tmp_path)


def test_fresh_validity_is_checked_on_same_mutated_input_buffers():
    assert probe.check_fresh_validity(TailStandIn(), torch.device("cpu")) == ["valid", "invalid", "recovered"]


@pytest.mark.parametrize("kind", ["fp8_boundaries", "scale_boundaries"])
@pytest.mark.parametrize("width", [2048, 4096])
def test_boundary_inputs_stay_finite_and_expected_geometry(kind, width):
    values = probe.boundary_tail_inputs(torch.device("cpu"), width, kind)
    assert values[0].shape == (1 if kind == "fp8_boundaries" else 3, width)
    result = probe.torch_tail_stages(*values)
    assert bool(result["valid"].all())
    if kind == "fp8_boundaries":
        assert torch.equal(result["scale"], torch.ones(1))


def test_old_fixture_is_repeated_six_times_with_original_seed_and_independent_row_gemms():
    # Smaller CPU-only width tests fixture orchestration, not the native geometry.
    rotated, weights, bias = probe.legacy_tail_inputs(torch.device("cpu"), TailStandIn(), 128, 32)
    assert rotated.shape == weights.shape == (192, 128)
    assert bias.shape == (192,)
    for group in range(1, 6):
        assert torch.equal(rotated[:32], rotated[group * 32 : (group + 1) * 32])
        assert torch.equal(weights[:32], weights[group * 32 : (group + 1) * 32])


@pytest.mark.parametrize("mismatch", [False, True])
def test_complete_case_orchestration_records_eight_scoped_reports(monkeypatch, tmp_path, capsys, mismatch):
    seen = []

    def legacy(device, native, width, rows):
        seen.append((width, rows))
        return inputs(width=128)

    monkeypatch.setattr(probe, "legacy_tail_inputs", legacy)
    monkeypatch.setattr(probe, "boundary_tail_inputs", lambda *args: inputs(width=128))
    args = probe.parse_args(["--report-dir", str(tmp_path), "--max-mismatches", "1", "--max-saved-rows", "0"])
    cases, fresh = probe.run_diagnostic_cases(torch.device("cpu"), TailStandIn(mismatch=mismatch), nullcontext, args)
    assert seen == [(2048, 32), (4096, 32), (2048, 1), (4096, 1)]
    assert len(cases) == 8 and fresh == ["valid", "invalid", "recovered"]
    assert len(list(tmp_path.glob("*/stages.json"))) == 8
    events = [json.loads(line.removeprefix("STARTUP_PROBE ")) for line in capsys.readouterr().out.splitlines()]
    assert len(events) == 8
    for event, case in zip(events, cases):
        assert event["case"] == probe.CASE
        assert event["event"] == "TAIL_STAGES"
        assert event["diagnostic_case"] == case["case"]
        assert (case["first_mismatch_stage"] is not None) is mismatch


def completed_child(*, mismatch=False):
    cases = []
    for kind in probe.CASE_KINDS:
        for width in (2048, 4096):
            cases.append(
                {
                    "case": f"{kind}_k{width}",
                    "diagnostic_matches_legacy_native": True,
                    "stages": {name: {"equal": not (mismatch and name == "scale")} for name in probe.STAGES},
                }
            )
    final = {
        "case": probe.CASE,
        "event": "CASE_FAIL" if mismatch else "CASE_PASS",
        "diagnostic_complete": True,
        "diagnostic_outcome": "MISMATCH" if mismatch else "ALL_MATCH",
        "device_execution_verified": True,
        "fresh_validity": ["valid", "invalid", "recovered"],
        "cases": cases,
    }
    return {
        "status": "FAIL" if mismatch else "PASS",
        "exit_code": 2 if mismatch else 0,
        "reaped": True,
        "events": [final],
    }


@pytest.mark.parametrize("mismatch", [False, True])
def test_completion_distinguishes_diagnosis_from_numerical_pass(mismatch):
    report = completed_child(mismatch=mismatch)
    assert probe.accept_completed_child(report)["diagnostic_outcome"] == ("MISMATCH" if mismatch else "ALL_MATCH")


@pytest.mark.parametrize(
    "change",
    [
        "timeout",
        "missing_case",
        "duplicate_case",
        "missing_stage",
        "wrong_abi_evidence",
        "bad_flag",
        "missing_recovery",
        "bad_exit",
        "not_reaped",
        "no_marker",
    ],
)
def test_incomplete_or_forged_evidence_is_not_accepted(change):
    result = completed_child()
    final = result["events"][0]
    if change == "timeout":
        result["status"] = "TIMEOUT"
    elif change == "missing_case":
        final["cases"].pop()
    elif change == "duplicate_case":
        final["cases"][-1] = copy.deepcopy(final["cases"][0])
    elif change == "missing_stage":
        final["cases"][0]["stages"].pop("scale")
    elif change == "wrong_abi_evidence":
        final["cases"][0]["diagnostic_matches_legacy_native"] = False
    elif change == "bad_flag":
        final["cases"][0]["stages"]["scale"]["equal"] = 1
    elif change == "missing_recovery":
        final["fresh_validity"].pop()
    elif change == "bad_exit":
        result["exit_code"] = 2
    elif change == "not_reaped":
        result["reaped"] = False
    elif change == "no_marker":
        final["diagnostic_complete"] = False
    assert probe.accept_completed_child(result) is None


def test_native_source_snapshots_preserve_original_sequence_and_own_queue_inputs():
    root = Path(__file__).resolve().parents[3]
    kernel = (root / "csrc/vq2a8_ascendc_v4_v2/activation_diagnostic_kernel.cpp").read_text()
    binding = (root / "csrc/vq2a8_ascendc_v4_v2/activation_diagnostic_binding.cpp").read_text()
    tokens = [
        "Mul(x, x, weight, width_)",
        "ReduceMax(reduced",
        "Divs(reduced",
        "Maxs(reduced",
        "Div(x, x, divisor",
        "Mins(x, x",
        "Maxs(x, x",
        "Cast(quantUb_.Get<fp8_e4m3fn_t>()",
    ]
    positions = [kernel.index(token) for token in tokens]
    assert positions == sorted(positions)
    assert "RoundMode::CAST_RINT" in kernel
    assert "Fence<HardEvent::MTE3_V>()" in kernel
    assert "uint32_t(0x7fc00000)" in kernel and "uint16_t(0x7f7f)" in kernel
    assert "RunOpApiV2" not in binding
    assert binding.index("const auto launchStream = stream.stream()") < binding.index("OpCommand::RunOpApi")
    assert "[launchStream, blocks, rotated, weightScale, rowBias, outputs, rows, width]" in binding
    assert "activation_tail_diagnostic_version() -> int" in binding
