# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU helper/oracle contracts; native hardware execution is a separate gate."""

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.validate_vq2a8_validity_fused import fixture, reference
from vllm_ascend.quantization.vq2a8_validity_fused import FusedLayerValidity, validate_inputs

REPO = Path(__file__).resolve().parents[3]
NATIVE = REPO / "csrc/vq2a8_ascendc_v4_v2"


def native_reference(**changes):
    return SimpleNamespace(**{"layer_validity_version": lambda: 1, "layer_validity": reference, **changes})


@pytest.mark.parametrize("version", [0, 2, True, False, "1", None])
def test_no_implicit_fallback_for_wrong_abi(version):
    with pytest.raises(RuntimeError, match="Unsupported layer validity ABI"):
        FusedLayerValidity(native_reference(layer_validity_version=lambda: version))


@pytest.mark.parametrize("missing", ["layer_validity", "layer_validity_version"])
def test_missing_native_feature_is_rejected(missing):
    native = native_reference()
    delattr(native, missing)
    with pytest.raises(RuntimeError, match="rebuilt ABI 1; no implicit fallback"):
        FusedLayerValidity(native)


@pytest.mark.parametrize("groups", range(1, 7))
@pytest.mark.parametrize("gate,down", [(2048, 2048), (2048, 4096), (4096, 2048), (4096, 4096)])
@pytest.mark.parametrize("flags", [0, 1, 8])
@pytest.mark.parametrize("projected_3d", [False, True])
def test_bounded_geometry_forwards_original_tensor_owners(groups, gate, down, flags, projected_3d):
    values = fixture("cpu", groups, gate, down, flags, projected_3d=projected_3d)
    calls = []

    def operation(statuses, outputs, route_flags):
        assert all(isinstance(value, list) for value in (statuses, outputs, route_flags))
        assert all(
            got is want
            for part, original in zip((statuses, outputs, route_flags), values)
            for got, want in zip(part, original)
        )
        calls.append(True)
        return reference(statuses, outputs, route_flags)

    result = FusedLayerValidity(native_reference(layer_validity=operation))(*values)
    assert result.dtype == torch.bool and result.shape == () and bool(result)
    assert calls == [True]


def test_status_nonzero_semantics_are_not_status_equals_one():
    values = fixture("cpu")
    values[0][0].copy_(torch.tensor([-2147483648, -2, -1, 1, 2, 2147483647], dtype=torch.int32))
    checker = FusedLayerValidity(native_reference())
    assert bool(checker(*values))
    values[0][0][-1] = 0
    assert not bool(checker(*values))
    values[0][0][-1] = 3
    assert bool(checker(*values))


@pytest.mark.parametrize(
    "family,index", [(0, i) for i in range(6)] + [(1, i) for i in range(3)] + [(2, i) for i in range(8)]
)
def test_each_input_contributes_to_fresh_boolean(family, index):
    values = fixture("cpu")
    checker = FusedLayerValidity(native_reference())
    assert bool(checker(*values))
    target = values[family][index]
    target.view(-1)[-1] = float("nan") if family == 1 else 0
    assert not bool(checker(*values))
    target.fill_(0.5 if family == 1 else 1)
    assert bool(checker(*values))


@pytest.mark.parametrize(
    "failure",
    [
        "status_count",
        "output_count",
        "flag_count",
        "status_dtype",
        "status_rows",
        "zero_groups",
        "many_groups",
        "status_scalar",
        "status_stride",
        "output_dtype",
        "output_rows",
        "output_width",
        "output_stride",
        "output_rank",
        "output_middle",
        "combined_rank",
        "combined_rows",
        "combined_width",
        "output_alignment",
        "flag_dtype",
        "flag_rank",
        "not_tensor",
    ],
)
def test_invalid_metadata_rejected_before_native_call(failure):
    statuses, outputs, flags = fixture("cpu")
    if failure == "status_count":
        statuses.pop()
    elif failure == "output_count":
        outputs.pop()
    elif failure == "flag_count":
        flags.append(flags[0])
    elif failure == "status_dtype":
        statuses[0] = statuses[0].long()
    elif failure == "status_rows":
        statuses[-1] = statuses[-1][:-1]
    elif failure == "zero_groups":
        statuses[0] = statuses[0][:0]
    elif failure == "many_groups":
        statuses[0] = torch.ones(7, dtype=torch.int32)
    elif failure == "status_scalar":
        statuses[0] = statuses[0][0]
    elif failure == "status_stride":
        statuses[0] = torch.ones(12, dtype=torch.int32)[::2]
    elif failure == "output_dtype":
        outputs[0] = outputs[0].float()
    elif failure == "output_rows":
        outputs[0] = outputs[0][:-1]
    elif failure == "output_width":
        outputs[0] = torch.ones(6, 1024, dtype=torch.bfloat16)
    elif failure == "output_stride":
        outputs[0] = outputs[0][:, ::2]
    elif failure == "output_rank":
        outputs[0] = outputs[0].flatten()
    elif failure == "output_middle":
        outputs[0] = torch.ones(6, 2, 4096, dtype=torch.bfloat16)
    elif failure == "combined_rank":
        outputs[-1] = outputs[-1].unsqueeze(0)
    elif failure == "combined_rows":
        outputs[-1] = outputs[-1].expand(6, -1)
    elif failure == "combined_width":
        outputs[-1] = outputs[-1][:, :2048]
    elif failure == "output_alignment":
        outputs[0] = torch.ones(6 * 4096 + 1, dtype=torch.bfloat16)[1:].reshape(6, 4096)
    elif failure == "flag_dtype":
        flags[0] = flags[0].int()
    elif failure == "flag_rank":
        flags[0] = flags[0].view(1)
    elif failure == "not_tensor":
        flags[0] = True
    called = []
    checker = FusedLayerValidity(native_reference(layer_validity=lambda *args: called.append(args)))
    with pytest.raises(ValueError):
        checker(statuses, outputs, flags)
    assert not called


def test_python_hot_path_only_validates_metadata_not_values():
    text = inspect.getsource(validate_inputs) + inspect.getsource(FusedLayerValidity.__call__)
    for forbidden in (".item(", ".cpu(", ".all(", ".any(", "torch.stack", "torch.isfinite", ".contiguous("):
        assert forbidden not in text


def test_native_binding_keeps_tensor_owners_and_caller_resolved_stream():
    binding = (NATIVE / "validity_binding.cpp").read_text(encoding="utf-8")
    assert "layer_validity(Tensor[] statuses, Tensor[] outputs, Tensor[] route_flags) -> Tensor" in binding
    assert "layer_validity_version() -> int" in binding
    for name in ("statuses", "outputs", "flags"):
        assert f"std::vector<at::Tensor> {name}(" in binding
    assert "recordStream(tensor.storage().data_ptr(), stream)" in binding
    enqueue = binding.index('OpCommand::RunOpApi("Vq2a8V4V2LayerValidity"')
    assert binding.index("const auto launchStream = stream.stream();") < enqueue
    assert "[launchStream, statuses, outputs, flags, result, groups, gateWidth, downWidth]" in binding
    assert "stream.stream()" not in binding[enqueue:]
    assert "RunOpApiV2" not in binding
    for forbidden in ("aclrtMemcpy", ".cpu(", ".item<", "at::stack", "at::cat"):
        assert forbidden not in binding


def test_native_contract_repeats_bounds_dtype_layout_and_storage_checks():
    binding = (NATIVE / "validity_binding.cpp").read_text(encoding="utf-8")
    for expression in (
        "statusList.size() == kLayerValidityStatuses",
        "outputList.size() == kLayerValidityOutputs",
        "flagList.size() <= kLayerValidityRouteFlags",
        "c10::DeviceType::PrivateUse1",
        "statusList[0].size(0) <= kMaximumGroups",
        "tensor.is_contiguous()",
        "tensor.storage().nbytes()",
        "tensor.storage_offset() >= 0",
        "tensor.data_ptr()) % kOutputAlignment == 0",
        "CheckTensor(status, device, at::kInt)",
        "CheckTensor(tensor, device, at::kBFloat16)",
        "CheckTensor(flag, device, at::kBool)",
        "flag.dim() == 0",
        "at::empty({},",
    ):
        assert expression in binding


def test_one_native_launch_has_fresh_single_writer_and_vector_finite_scan():
    kernel = (NATIVE / "validity_kernel.cpp").read_text(encoding="utf-8")
    assert "<<<1, nullptr, stream>>>" in kernel
    assert "valid_ = true;" in kernel
    assert "status.GetValue(row) != 0" in kernel
    assert "flag.GetValue(0) != 0" in kernel
    assert "Cast(widened, input, RoundMode::CAST_NONE, width)" in kernel
    assert "Compares(mask, scratch, kValidityFiniteMaximum, CMPMODE::LE, width)" in kernel
    assert "allBits == 0xffffffffU" in kernel
    assert "DataCopyPad(result_, resultUb_.Get<uint8_t>(), scalar)" in kernel
    assert "ValidityFence<HardEvent::V_MTE2>();" in kernel
    assert "sizeof(uint8_t)" in kernel
    assert "atomic" not in kernel.lower().replace("no atomics", "")
    for index in range(6):
        assert f"op.CheckStatus(s{index}, groups);" in kernel
    for index in range(8):
        assert f"if (flagCount > {index}) op.CheckFlag(f{index});" in kernel
