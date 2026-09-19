# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of acceptance orchestration, never a native NPU validation claim."""

import copy
import inspect
import json
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tools import validate_vq2a8_activation_packed as probe


def test_existing_modes_remain_the_only_default_and_child_receives_selection():
    args = probe.parse_args([])
    assert tuple(args.preparation_modes) == ("rowwise_packed", "sign_fused")
    modes = ["sign_fused_direct", "sign_fused_strided"]
    args = probe.parse_args(["--preparation-modes", *modes, "--queue-lifetime"])
    command = probe.child_command(args)
    child = probe.parse_args(command[command.index("--child") :])
    assert child.child and child.queue_lifetime
    assert child.preparation_modes == modes


@pytest.mark.parametrize("modes", [[], ["rowwise"], ["sign_fused_direct", "sign_fused_direct"]])
def test_invalid_selection_cannot_silently_use_existing_mode(modes):
    with pytest.raises(SystemExit):
        probe.parse_args(["--preparation-modes", *modes])


def test_plan_declares_opt_in_abi_and_no_hardware_claim(tmp_path, capsys):
    directory = tmp_path / "report"
    assert (
        probe.main(
            [
                "--plan-only",
                "--preparation-modes",
                "sign_fused_direct",
                "--report-dir",
                str(directory),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["preparation_modes"] == ["sign_fused_direct"]
    assert report["requires_strided_sign_abi"] is True
    assert report["status"] == "PLANNED"
    for flag in ("device_execution_verified", "graph_verified", "performance_verified", "model_integration_verified"):
        assert report[flag] is False
    assert not directory.exists()


def test_old_perf3_abi_is_sufficient_only_for_old_modes():
    old = SimpleNamespace(activation_preparation_version=lambda: 1)
    assert probe.require_mode_abis(old, probe.DEFAULT_PREPARATION_MODES) == {"activation_preparation": 1}
    for mode in probe.STRIDED_PREPARATION_MODES:
        with pytest.raises(ValueError, match="rebuilt strided-sign ABI; no fallback"):
            probe.require_mode_abis(old, [mode])


@pytest.mark.parametrize("value", [True, False, "1", 0, 2])
@pytest.mark.parametrize("abi", ["activation_preparation_version", "activation_sign_strided_version"])
def test_probe_requires_exact_integer_abi_version(abi, value):
    native = SimpleNamespace(activation_preparation_version=lambda: 1, activation_sign_strided_version=lambda: 1)
    setattr(native, abi, lambda: value)
    with pytest.raises(ValueError, match="ABI mismatch"):
        probe.require_mode_abis(native, ["sign_fused_direct"])


@pytest.mark.parametrize("mode", probe.PREPARATION_MODES)
def test_preparation_factory_selects_exact_flags_without_changing_legacy_modes(monkeypatch, mode):
    import vllm_ascend.quantization.vq2a8_activation_packed as implementation

    monkeypatch.setattr(implementation, "PackedRowwiseVQ2A8Preparation", lambda **kwargs: kwargs)
    native = object()
    actual = probe.preparation_for_mode(mode, native)
    expected = {"fuse_sign": mode != "rowwise_packed", "native_ops": native}
    if mode in probe.STRIDED_PREPARATION_MODES:
        expected.update(strided_sign=True, direct_output=mode == "sign_fused_direct")
    assert actual == expected


@pytest.mark.parametrize("dtype", probe.INPUT_DTYPES)
@pytest.mark.parametrize("layout", probe.INPUT_LAYOUTS)
@pytest.mark.parametrize("width", [2048, 4096])
@pytest.mark.parametrize("groups", [1, 6])
def test_input_fixture_really_exercises_dtype_stride_and_offset(dtype, layout, width, groups):
    values, owner = probe.input_layout_values("cpu", width, groups, dtype, layout)
    hidden = values[0]
    assert hidden.shape == (groups, width)
    assert hidden.dtype == {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    assert hidden.stride(1) == 1
    expected = probe.fixture("cpu", width, groups, dtype=hidden.dtype)[0]
    if layout == "expanded":
        # expand(1, K) may keep a nonzero row stride; G>1 proves stride zero.
        assert groups == 1 or hidden.stride(0) == 0
        expected = expected[:1].expand(groups, -1)
    elif layout == "padded":
        assert hidden.stride(0) == width + 2 * probe.PAD_COLUMNS
        assert hidden.storage_offset() == probe.PAD_COLUMNS
        assert hidden.data_ptr() % 32 == 0
        assert bool((owner[:, : probe.PAD_COLUMNS] == 19).all())
        assert bool((owner[:, -probe.PAD_COLUMNS :] == 19).all())
    else:
        assert hidden.is_contiguous()
    assert torch.equal(hidden, expected)


def test_new_numeric_graph_and_queue_case_matrices_are_complete():
    modes = probe.STRIDED_PREPARATION_MODES
    numeric = probe.extended_numeric_cases(modes)
    assert len(numeric) == 2 * 6 * 2 * 3 * 2
    assert len({case[0] for case in numeric}) == len(numeric)
    assert {case[4] for case in numeric} == {"bf16", "fp32"}
    assert {case[5] for case in numeric} == {"contiguous", "expanded", "padded"}
    graph = probe.graph_cases(modes)
    queue = probe.queue_input_cases(modes)
    assert len(graph) == len(queue) == 2 * 2 * 3 * 2
    assert probe.expected_graph_results(modes) == [case[0] for case in graph]
    assert {case[0] for case in graph} == {case[0] for case in queue}
    boundary = probe.bf16_boundary_cases(modes)
    assert len(boundary) == 2 * 3 * 4 * 2
    assert probe.expected_numeric_results(modes)[-len(numeric) - len(boundary) :] == [
        case[0] for case in numeric + boundary
    ]


@pytest.mark.parametrize(
    "boundary,expected_bits",
    [
        ("tiny", [0x0080, 0x8080]),
        ("subnormal_min", [0x0001, 0x8001]),
        ("signed_zero", [0x0000, 0x8000]),
        ("boundary_mixed", [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x8080, 0x0081, 0x8081]),
    ],
)
def test_boundary_patterns_really_include_bf16_subnormals_and_signed_zeros(boundary, expected_bits):
    pattern = probe.bf16_boundary_pattern(boundary)
    assert pattern.dtype == torch.bfloat16 and pattern.device.type == "cpu"
    assert pattern.view(torch.uint16).tolist() == expected_bits
    if boundary == "tiny":
        assert pattern[0].float().item() == torch.finfo(torch.bfloat16).tiny
    if boundary == "subnormal_min":
        assert pattern[0].float().item() == torch.finfo(torch.bfloat16).tiny / 128


def test_boundary_sign_comparison_checks_fp32_bits_before_quantization():
    values = list(probe.fixture("cpu", 2048, 1))
    values[0].copy_(probe.bf16_boundary_pattern("signed_zero").repeat(1024).reshape(1, -1))
    calls = []

    def signed(hidden, scale, bias, signs):
        calls.append((hidden.dtype, hidden.is_contiguous()))
        return hidden.float() * signs.float(), torch.ones(hidden.shape[0], dtype=torch.int32)

    native = SimpleNamespace(activation_sign_strided=signed, activation_sign=signed)
    probe.check_native_sign_boundary(native, values, "cpu_check")
    assert calls == [(torch.bfloat16, True), (torch.float32, True)]

    def zero_sign_loss(hidden, scale, bias, signs):
        result, valid = signed(hidden, scale, bias, signs)
        return result.abs(), valid

    native.activation_sign_strided = zero_sign_loss
    with pytest.raises(AssertionError, match="cpu_check_signed_fp32:.*unequal bytes"):
        probe.check_native_sign_boundary(native, values, "cpu_check")


def test_boundary_orchestration_checks_complete_outputs_and_selected_case_evidence(monkeypatch):
    """CPU native stand-ins test orchestration, not Ascend cast correctness."""

    checked_names = []
    original_outputs = probe.check_outputs

    def checked_outputs(actual, expected, name):
        checked_names.append(name)
        original_outputs(actual, expected, name)

    def signed(hidden, scale, bias, signs):
        return hidden.float() * signs.float(), torch.ones(hidden.shape[0], dtype=torch.int32)

    class Preparation:
        def packed(self, *values, validity):
            validity(torch.tensor(True))
            return probe.reference(values)

    monkeypatch.setattr(probe, "preparation_for_mode", lambda *_: Preparation())
    monkeypatch.setattr(probe, "check_outputs", checked_outputs)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)
    native = SimpleNamespace(activation_sign_strided=signed, activation_sign=signed)
    modes = ["sign_fused_direct"]
    completed = probe.run_bf16_boundary_checks("cpu", native, lambda _: nullcontext(), modes)
    assert completed == checked_names == [case[0] for case in probe.bf16_boundary_cases(modes)]
    assert probe.bf16_boundary_cases(probe.DEFAULT_PREPARATION_MODES) == []


def child_evidence(modes, *, queue_lifetime):
    results = {
        "numeric": probe.expected_numeric_results(modes),
        "graph": probe.expected_graph_results(modes),
        "invalid": True,
    }
    if queue_lifetime:
        results["queue_lifetime"] = {
            "modes": list(modes),
            "iterations": probe.QUEUE_ITERATIONS,
            "explicit_per_iteration_sync": False,
            "requires_runtime_task_queue_enabled": True,
            "task_queue_enable": "1",
            "input_cases": [case[0] for case in probe.queue_input_cases(modes)],
            "input_owners_dropped_before_fence": True,
            "retained_outputs_checked": True,
        }
    return {
        "status": "PASS",
        "events": [
            {
                "case": probe.CASE,
                "event": "CASE_PASS",
                "results": results,
                "library": {"path": "/tmp/libvq2a8_ascendc_v4_v2.so", "sha256": "a" * 64},
                "preparation_modes": list(modes),
                "native_abis": {"activation_preparation": 1, "activation_sign_strided": 1},
                "device_execution_verified": True,
                "graph_verified": True,
                "model_integration_verified": False,
                "performance_verified": False,
            }
        ],
    }


@pytest.mark.parametrize("queue_lifetime", [False, True])
def test_child_evidence_requires_the_selected_mode_matrix(queue_lifetime):
    modes = probe.STRIDED_PREPARATION_MODES
    evidence = child_evidence(modes, queue_lifetime=queue_lifetime)
    assert probe.validate_child_evidence(evidence, modes=modes, queue_lifetime=queue_lifetime)
    incomplete = copy.deepcopy(evidence)
    incomplete["events"][-1]["results"]["numeric"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        probe.validate_child_evidence(incomplete, modes=modes, queue_lifetime=queue_lifetime)
    with pytest.raises(ValueError, match="Incomplete"):
        probe.validate_child_evidence(evidence, modes=["sign_fused_direct"], queue_lifetime=queue_lifetime)


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_cases", []),
        ("input_owners_dropped_before_fence", False),
        ("retained_outputs_checked", False),
    ],
)
def test_new_queue_evidence_must_prove_owner_release_and_all_layouts(field, value):
    modes = ["sign_fused_direct"]
    evidence = child_evidence(modes, queue_lifetime=True)
    evidence["events"][-1]["results"]["queue_lifetime"][field] = value
    with pytest.raises(ValueError, match="strided queue-lifetime"):
        probe.validate_child_evidence(evidence, modes=modes, queue_lifetime=True)


def test_mode_and_abi_evidence_cannot_be_omitted():
    for field in ("preparation_modes", "native_abis"):
        evidence = child_evidence(["sign_fused_direct"], queue_lifetime=False)
        del evidence["events"][-1][field]
        with pytest.raises(ValueError, match="selected-mode or strided ABI"):
            probe.validate_child_evidence(evidence, modes=["sign_fused_direct"], queue_lifetime=False)


def test_boolean_abi_evidence_is_not_an_integer_version():
    evidence = child_evidence(["sign_fused_direct"], queue_lifetime=False)
    evidence["events"][-1]["native_abis"]["activation_sign_strided"] = True
    with pytest.raises(ValueError, match="selected-mode or strided ABI"):
        probe.validate_child_evidence(evidence, modes=["sign_fused_direct"], queue_lifetime=False)


@pytest.mark.parametrize("layout", probe.INPUT_LAYOUTS)
@pytest.mark.parametrize("dtype", probe.INPUT_DTYPES)
@pytest.mark.parametrize("width", [2048, 4096])
@pytest.mark.parametrize("mode", probe.STRIDED_PREPARATION_MODES)
def test_queue_orchestration_releases_temporary_owners_before_fence(monkeypatch, layout, dtype, width, mode):
    """A synchronous CPU stand-in checks Python ownership, not device ordering."""

    submitted_refs = []
    events = []
    calls = 0
    original_fill = torch.Tensor.fill_

    def npu_compatible_fill(tensor, value):
        # CPU fill_ accepts repeated writes to expanded views; the tested NPU
        # ViewCopy tiling rejects them. Keep that restriction in the fixture.
        assert not any(size > 1 and stride == 0 for size, stride in zip(tensor.shape, tensor.stride()))
        return original_fill(tensor, value)

    def output(values):
        return values[0].float().clone(), values[1].sum(-1, keepdim=True), values[2].sum(-1, keepdim=True)

    class Preparation:
        def packed(self, *values, validity):
            nonlocal calls
            calls += 1
            hidden = values[0]
            assert hidden.shape == (6, width)
            assert hidden.dtype == {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
            if layout == "expanded":
                # Do not fix the write by materializing the input and losing
                # coverage of the native stride-zero read path.
                assert hidden.stride() == (0, 1)
            elif layout == "padded":
                assert hidden.stride() == (width + 2 * probe.PAD_COLUMNS, 1)
                assert hidden.storage_offset() == probe.PAD_COLUMNS
                assert bool((hidden._base[:, : probe.PAD_COLUMNS] == 19).all())
                assert bool((hidden._base[:, -probe.PAD_COLUMNS :] == 19).all())
            if calls > 1:
                submitted_refs.extend(weakref.ref(value) for value in values[:4])
                if hidden._base is not None:
                    submitted_refs.append(weakref.ref(hidden._base))
                expected_value = ((calls - 2) % 7 - 3) / 8
                assert torch.equal(hidden, torch.full((6, width), expected_value, dtype=hidden.dtype))
            validity(torch.tensor(True))
            return output(values)

    def synchronize():
        events.append("fence")
        assert calls == 4
        assert all(ref() is None for ref in submitted_refs)

    monkeypatch.setattr(probe, "QUEUE_ITERATIONS", 3)
    monkeypatch.setattr(probe, "preparation_for_mode", lambda *_: Preparation())
    monkeypatch.setattr(probe, "reference", output)
    monkeypatch.setattr(torch.Tensor, "fill_", npu_compatible_fill)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=synchronize), raising=False)
    case = ("cpu_orchestration", mode, width, dtype, layout)
    assert probe.run_strided_queue_case("cpu", None, lambda _: nullcontext(), case) == case[0]
    assert events == ["fence"]


def test_graph_probe_retains_dynamic_input_and_metadata_recovery_and_exact_oracle():
    source = inspect.getsource(probe.run_graph_checks)
    for target in ("hidden", "scale", "bias", "sign"):
        assert f'"invalid_{target}"' in source and f'"recovered_{target}"' in source
    assert "graph.replay()" in source and "check_outputs(outputs, reference(values)" in source
    assert "targets[target][-1, -1] = bad" in source
    assert "activation_quantize" not in source
