# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for packed M=1 preparation; physical NPU needs its probe."""

import gc
import inspect
import json
import weakref
from types import SimpleNamespace

import pytest
import torch

from tools import validate_vq2a8_activation_packed as validate
from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_activation_packed import (
    PackedRowwiseVQ2A8Preparation,
)


class TorchSignOps:
    """Torch arithmetic oracle, not a claim about native device execution."""

    def __init__(self):
        self.calls = []

    def activation_preparation_version(self):
        return 1

    def activation_sign(self, x, scale, bias, signs):
        self.calls.append((tuple(x.shape), x.dtype))
        valid = torch.isfinite(x).all(-1) & torch.isfinite(scale).all(-1) & torch.isfinite(bias).all(-1)
        valid &= ((signs == -1) | (signs == 1)).all(-1)
        return x * signs.float(), valid.int()


def fixture(width, groups, *, dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(width * 10 + groups)
    hidden = torch.randn(groups, width, generator=generator).to(dtype)
    scale = torch.randn(groups, width, generator=generator)
    bias = torch.randn(groups, width, generator=generator)
    signs = torch.where(torch.arange(groups * width).reshape(groups, width) % 3 == 0, -1, 1).to(torch.int8)
    spec = SimpleNamespace(columns=width, rht_true_columns=width, rht_block_size=128)
    return hidden, scale, bias, signs, spec


def reference(hidden, scale, bias, signs, spec):
    requests = [
        (
            hidden[index : index + 1],
            {
                "weight_scale": scale[index],
                "weight_bias": bias[index],
                "rht_sign": signs[index],
            },
            spec,
        )
        for index in range(hidden.shape[0])
    ]
    prepared = RowwiseVQ2A8Preparation(compact=True).many(requests)
    return tuple(torch.cat(values).contiguous() for values in zip(*prepared))


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("width", [2048, 4096])
@pytest.mark.parametrize("groups", range(1, 7))
@pytest.mark.parametrize("fuse_sign", [False, True])
def test_packed_matches_rowwise_oracle_byte_for_byte(width, groups, fuse_sign):
    values = fixture(width, groups)
    native = TorchSignOps()
    actual = PackedRowwiseVQ2A8Preparation(fuse_sign=fuse_sign, native_ops=native).packed(*values)
    expected = reference(*values)
    for got, want in zip(actual, expected):
        assert_bytes(got, want)
        assert got.is_contiguous()
    expected_calls = [((groups, width), torch.float32)] if fuse_sign else []
    assert native.calls == expected_calls


@pytest.mark.parametrize("fuse_sign", [False, True])
@pytest.mark.parametrize("case", ["zero", "small", "impulse", "expanded"])
def test_packed_preserves_boundaries_and_stride_zero_input(fuse_sign, case):
    hidden, scale, bias, signs, spec = fixture(2048, 6)
    if case == "zero":
        hidden.zero_()
    elif case == "small":
        hidden.mul_(1e-15)
    elif case == "impulse":
        hidden.zero_()
        hidden[:, -1] = -1
    else:
        hidden = hidden[:1].expand(6, -1)
        assert not hidden.is_contiguous()
    actual = PackedRowwiseVQ2A8Preparation(fuse_sign=fuse_sign, native_ops=TorchSignOps()).packed(
        hidden, scale, bias, signs, spec
    )
    expected = reference(hidden, scale, bias, signs, spec)
    for got, want in zip(actual, expected):
        assert_bytes(got, want)


@pytest.mark.parametrize("fuse_sign", [False, True])
@pytest.mark.parametrize(
    "target,bad",
    [
        ("hidden", float("nan")),
        ("scale", float("inf")),
        ("bias", -float("inf")),
        ("signs", 0),
        ("signs", -128),
    ],
)
def test_packed_rechecks_every_dynamic_value(fuse_sign, target, bad):
    hidden, scale, bias, signs, spec = fixture(2048, 2)
    flags = []
    preparation = PackedRowwiseVQ2A8Preparation(validity=flags.append, fuse_sign=fuse_sign, native_ops=TorchSignOps())
    preparation.packed(hidden, scale, bias, signs, spec)
    assert bool(flags[-1])
    tensors = {"hidden": hidden, "scale": scale, "bias": bias, "signs": signs}
    tensors[target].view(-1)[-1] = bad
    preparation.packed(hidden, scale, bias, signs, spec)
    assert len(flags) == 2 and not bool(flags[-1])


def test_packed_deferred_validity_does_not_read_device_scalar(monkeypatch):
    flags = []
    preparation = PackedRowwiseVQ2A8Preparation()

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected scalar read in packed preparation")

    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    preparation.packed(*fixture(2048, 3), validity=flags.append)
    assert len(flags) == 1 and flags[0].shape == ()


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda values: (values[0][:0], *values[1:]), "1..6"),
        (
            lambda values: (
                values[0].repeat(7, 1),
                *(value.repeat(7, 1) for value in values[1:4]),
                values[4],
            ),
            "1..6",
        ),
        (lambda values: (values[0][:, :1024], *values[1:]), "2048 or 4096"),
        (
            lambda values: (
                *values[:4],
                SimpleNamespace(columns=2048, rht_true_columns=2000, rht_block_size=128),
            ),
            "padded or mismatched",
        ),
        (
            lambda values: (
                *values[:4],
                SimpleNamespace(columns=2048, rht_true_columns=2048, rht_block_size=3),
            ),
            "rht_block_size",
        ),
        (lambda values: (values[0], values[1].half(), *values[2:]), "weight_scale"),
        (lambda values: (values[0], values[1].t(), *values[2:]), "weight_scale"),
    ],
)
def test_packed_rejects_unsupported_geometry_before_arithmetic(mutation, match):
    with pytest.raises(ValueError, match=match):
        PackedRowwiseVQ2A8Preparation().packed(*mutation(fixture(2048, 1)))


def test_packed_sign_fusion_requires_only_the_existing_sign_operator():
    native = TorchSignOps()
    preparation = PackedRowwiseVQ2A8Preparation(fuse_sign=True, native_ops=native)
    preparation.packed(*fixture(2048, 1))
    assert native.calls
    native.activation_preparation_version = lambda: 2
    with pytest.raises(RuntimeError, match="ABI 2"):
        PackedRowwiseVQ2A8Preparation(fuse_sign=True, native_ops=native)
    native.activation_preparation_version = lambda: True
    with pytest.raises(RuntimeError, match="ABI True"):
        PackedRowwiseVQ2A8Preparation(fuse_sign=True, native_ops=native)
    with pytest.raises(RuntimeError, match="no unfused fallback"):
        PackedRowwiseVQ2A8Preparation(fuse_sign=True, native_ops=SimpleNamespace())


def test_general_many_remains_the_unmodified_rowwise_prefill_path():
    hidden, scale, bias, signs, spec = fixture(2048, 2)
    requests = [
        (
            hidden[index : index + 1],
            {
                "weight_scale": scale[index],
                "weight_bias": bias[index],
                "rht_sign": signs[index],
            },
            spec,
        )
        for index in range(2)
    ]
    native = TorchSignOps()
    actual = PackedRowwiseVQ2A8Preparation(fuse_sign=True, native_ops=native).many(requests)
    expected = RowwiseVQ2A8Preparation().many(requests)
    for got, want in zip(actual, expected):
        for got_field, want_field in zip(got, want):
            assert_bytes(got_field, want_field)
    assert native.calls == []


def test_packed_has_no_python_repacking_pipeline():
    source = inspect.getsource(PackedRowwiseVQ2A8Preparation.packed)
    for forbidden in (".split(", "torch.stack(", "torch.cat(", "payload"):
        assert forbidden not in source


@pytest.mark.parametrize("fuse_sign", [False, True])
def test_outputs_survive_input_release_and_are_not_reused(fuse_sign):
    values = list(fixture(2048, 2))
    references = [weakref.ref(value) for value in values[:4]]
    preparation = PackedRowwiseVQ2A8Preparation(fuse_sign=fuse_sign, native_ops=TorchSignOps())
    first = preparation.packed(*values)
    first_bytes = tuple(value.view(torch.uint8).clone() for value in first)
    del values
    gc.collect()
    assert all(reference() is None for reference in references)
    second = preparation.packed(*fixture(2048, 2))
    assert all(first_value.data_ptr() != second_value.data_ptr() for first_value, second_value in zip(first, second))
    for value, original in zip(first, first_bytes):
        assert torch.equal(value.view(torch.uint8), original)


def test_graph_preparation_freezes_hadamard_geometry():
    preparation = PackedRowwiseVQ2A8Preparation()
    preparation.prepare_for_graph(torch.device("cpu"), 128)
    values = list(fixture(2048, 1))
    values[-1].rht_block_size = 64
    with pytest.raises(RuntimeError, match="geometry changed"):
        preparation.packed(*values)


def test_packed_probe_plan_covers_both_modes_without_claiming_execution(tmp_path, capsys):
    report_dir = tmp_path / "not-created"
    assert (
        validate.main(
            [
                "--physical-npu",
                "4",
                "--timeout-s",
                "90",
                "--queue-lifetime",
                "--report-dir",
                str(report_dir),
                "--plan-only",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["scope"] == "packed_m1_activation_only"
    assert report["device_execution_verified"] is report["graph_verified"] is False
    assert report["model_integration_verified"] is report["performance_verified"] is False
    assert "--queue-lifetime" in report["command"]
    assert report["command"][report["command"].index("--physical-npu") + 1] == "4"
    assert report["queue_lifetime"] == {
        "enabled": True,
        "iterations": validate.QUEUE_ITERATIONS,
        "runtime_queue_slots_measured": False,
        "requires_runtime_task_queue_enabled": True,
        "explicit_per_iteration_synchronize": False,
    }
    assert not report_dir.exists()


def test_packed_probe_defaults_to_user_npu_and_declares_boundary_coverage():
    assert validate.parse_args([]).physical_npu == 1
    numeric = inspect.getsource(validate.run_numeric_checks)
    graph = inspect.getsource(validate.run_graph_checks)
    for token in ('("zero", "tiny", "impulse", "expanded")', "(2048, 4096)", "range(1, 7)"):
        assert token in numeric
    for token in ("(2048, 4096)", "expanded", "invalid_scale", "invalid_bias", "invalid_sign", "recovered_sign"):
        assert token in graph


@pytest.mark.parametrize("value", [None, "0", "1", "2"])
def test_packed_probe_task_queue_environment_defaults_without_overriding(value):
    args = validate.parse_args(["--queue-lifetime"])
    parent = {"UNCHANGED": "yes"}
    if value is not None:
        parent["TASK_QUEUE_ENABLE"] = value
    before = dict(parent)
    child = validate.probe_environment(args, parent)
    assert parent == before and child is not parent
    assert child["TASK_QUEUE_ENABLE"] == ("1" if value is None else value)
    assert child["UNCHANGED"] == "yes"


def test_packed_probe_non_lifetime_does_not_invent_task_queue_policy():
    parent = {"UNCHANGED": "yes"}
    child = validate.probe_environment(validate.parse_args([]), parent)
    assert "TASK_QUEUE_ENABLE" not in child and parent == {"UNCHANGED": "yes"}


def test_packed_lifetime_defers_validity_until_the_final_fence():
    source = inspect.getsource(validate.run_queue_checks)
    loop = source[source.index("for iteration in range") : source.index("torch.npu.synchronize()")]
    assert "validity=flags.append" in loop and "flags.clear()" in loop
    assert "bool(" not in loop and ".item(" not in loop and ".cpu(" not in loop
    assert source.index("final_valid = flags[-1]") < source.index("torch.npu.synchronize()") < source.index("bool(")


@pytest.mark.parametrize("arguments", [["--physical-npu", "-1"], ["--timeout-s", "0"], ["--child", "--plan-only"]])
def test_packed_probe_rejects_invalid_or_conflicting_arguments(arguments):
    with pytest.raises(SystemExit):
        validate.parse_args(arguments)


def test_sign_fused_candidate_never_calls_failed_native_quantizer():
    implementation = inspect.getsource(PackedRowwiseVQ2A8Preparation)
    probe = inspect.getsource(validate.run_numeric_checks)
    assert "activation_quantize" not in implementation
    assert "activation_quantize" not in probe


def child_evidence(*, queue_lifetime=False):
    results = {
        "numeric": validate.expected_numeric_results(),
        "invalid": True,
        "graph": validate.expected_graph_results(),
    }
    if queue_lifetime:
        results["queue_lifetime"] = {
            "modes": ["rowwise_packed", "sign_fused"],
            "iterations": validate.QUEUE_ITERATIONS,
            "explicit_per_iteration_sync": False,
            "requires_runtime_task_queue_enabled": True,
            "task_queue_enable": "1",
        }
    return {
        "status": "PASS",
        "events": [
            {
                "case": validate.CASE,
                "event": "CASE_PASS",
                "results": results,
                "library": {"path": "/tmp/libvq2a8_ascendc_v4_v2.so", "sha256": "a" * 64},
                "device_execution_verified": True,
                "graph_verified": True,
                "model_integration_verified": False,
                "performance_verified": False,
            }
        ],
    }


@pytest.mark.parametrize("queue_lifetime", [False, True])
def test_complete_child_evidence_is_accepted(queue_lifetime):
    final = validate.validate_child_evidence(
        child_evidence(queue_lifetime=queue_lifetime), queue_lifetime=queue_lifetime
    )
    assert final["case"] == validate.CASE


@pytest.mark.parametrize(
    "corruption",
    [
        "status",
        "trailing_event",
        "case",
        "device",
        "graph",
        "numeric",
        "duplicate_numeric",
        "invalid",
        "library",
        "unexpected_lifetime",
    ],
)
def test_incomplete_or_spoofed_child_evidence_is_rejected(corruption):
    result = child_evidence()
    final = result["events"][-1]
    if corruption == "status":
        result["status"] = "FAIL"
    elif corruption == "trailing_event":
        result["events"].append({"case": validate.CASE, "event": "CASE_FAIL"})
    elif corruption == "case":
        final["case"] = "spoof"
    elif corruption == "device":
        final["device_execution_verified"] = False
    elif corruption == "graph":
        final["results"]["graph"] = []
    elif corruption == "numeric":
        final["results"]["numeric"] = []
    elif corruption == "duplicate_numeric":
        final["results"]["numeric"][-1] = final["results"]["numeric"][0]
    elif corruption == "invalid":
        final["results"]["invalid"] = False
    elif corruption == "library":
        final["library"]["sha256"] = "short"
    else:
        final["results"]["queue_lifetime"] = {}
    with pytest.raises(ValueError):
        validate.validate_child_evidence(result, queue_lifetime=False)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("modes", ["rowwise_packed"]),
        ("iterations", 1),
        ("explicit_per_iteration_sync", True),
        ("requires_runtime_task_queue_enabled", False),
        ("task_queue_enable", "0"),
    ],
)
def test_queue_lifetime_evidence_requires_async_task_queue_contract(field, bad):
    result = child_evidence(queue_lifetime=True)
    result["events"][-1]["results"]["queue_lifetime"][field] = bad
    with pytest.raises(ValueError, match="queue-lifetime"):
        validate.validate_child_evidence(result, queue_lifetime=True)


def test_probe_main_writes_summary_marker_and_has_interrupt_exit_code():
    source = inspect.getsource(validate.main)
    assert 'directory / "summary.json"' in source
    assert "V4_PACKED_ACTIVATION=" in source and "SUMMARY=" in source
    assert 'report["status"] = "INTERRUPTED"' in source and "130" in source
