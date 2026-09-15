# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vector-gather source and byte-layout contracts, not NPU execution proof."""

import weakref
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from tests.ut.quantization.test_vq2a8_v4_v2_native import function
from tools import validate_vq2a8_v4_v2 as validate

SOURCE = Path(__file__).resolve().parents[3] / "csrc/vq2a8_ascendc_v4_v2"
SELECT_COLUMNS = 256
MAX_K = 4096


def test_vector_prepare_is_separately_selectable_without_changing_the_scalar_abi():
    source = (SOURCE / "torch_binding.cpp").read_text()
    assert '.def("project", &vq2a8_ascendc_v4_v2::ResidentBank::Project<false>)' in source
    assert '.def("project_vectorized", &vq2a8_ascendc_v4_v2::ResidentBank::ProjectVectorized)' in source
    assert "return Project<true>(x, scale, bias, ids)" in function(source, "ProjectVectorized")
    project = function(source, "Project")
    assert "if constexpr (Vectorized)" in project
    assert "LaunchResidentPrepareVectorized(" in project
    assert "LaunchResidentPrepare(" in project
    assert project.count("LaunchGrouped(") == 1
    assert 'm.def("activation_reorder_version() -> int"' in source
    assert "return Project<true, true>(x, scale, bias, ids)" in function(source, "PrepareVectorized")
    assert "if constexpr (!PrepareOnly)" in project
    assert "if constexpr (PrepareOnly) return {reordered, valid}" in project


def test_vector_prepare_retains_queue_lifetime_and_stream_contracts():
    project = function((SOURCE / "torch_binding.cpp").read_text(), "Project")
    before_enqueue, callback = project.split("at_npu::native::OpCommand::RunOpApi(", 1)
    assert "RecordInputs({x, scale, bias, ids}" in before_enqueue
    assert "[state, x, scale, bias, ids, reordered, descriptors, output, valid" in callback
    assert "state->stream" in callback
    assert ".stream(" not in callback
    assert "getCurrentNPUStream(" not in callback
    assert "RunOpApiV2" not in project
    assert not any(token in project for token in (".cpu(", ".item(", ".to(", "at::kCPU", "synchronize"))


def test_vector_gather_uses_ub_byte_gathers_not_scalar_gm_reads_or_numeric_conversion():
    source = (SOURCE / "resident_prepare.cpp").read_text()
    body = function(source, "GatherVectorized")
    assert "DataCopy(input, x_[rowBase], k_)" in body
    assert "DataCopy(orderWords, order[column], kSelectColumns)" in body
    assert "Gather(orderOffsets, orderWords.ReinterpretCast<uint32_t>()" in body
    assert "Gather(gathered, input, orderOffsets, uint32_t(0), kSelectColumns)" in body
    assert ".GetValue(" not in body
    assert ".SetValue(" not in body
    assert "Cast(" not in body
    assert "CreateVecIndex(offsets, int32_t(0), kSelectColumns)" in source
    assert "Muls(offsets, offsets, int32_t(sizeof(int64_t)), kSelectColumns)" in source
    assert body.index("MTE2_V") < body.index("Gather(orderOffsets")
    assert body.index("Gather(gathered") < body.index("V_MTE3") < body.index("DataCopy(reordered_")
    assert body.index("DataCopy(reordered_") < body.index("MTE3_V") < body.index("V_MTE2")


def test_vector_indirect_access_is_guarded_and_invalid_routes_keep_the_existing_descriptor_contract():
    source = (SOURCE / "resident_prepare.cpp").read_text()
    process = function(source, "Process")
    assert process.index("ValidResidentSlot(expert, experts_)") < process.index("static_cast<uint32_t>(expert)")
    assert process.index("if (valid && column < k_)") < process.index("GatherVectorized(")
    assert "else if (!valid && column < n_)" in process
    assert "Duplicate(outputUb_.Get<uint16_t>(), kInvalidBf16" in process
    assert "if (row == 0 && column == 0) WriteDescriptor(route, expert, valid)" in process
    assert "record.SetValue(field, uint64_t(0))" in function(source, "WriteDescriptor")


@pytest.mark.parametrize("k", (2048, 4096))
@pytest.mark.parametrize("pattern", ("reverse", "interleaved", "shuffled"))
def test_gather_byte_offsets_preserve_all_fp8_bit_patterns(k, pattern):
    # Integer layout oracle only. Deliberately retain 0x00/0x80, FP8 NaN bits,
    # and all other encodings; converting through float would hide errors.
    if pattern == "reverse":
        order = torch.arange(k - 1, -1, -1, dtype=torch.int64)
    elif pattern == "interleaved":
        order = torch.arange(k, dtype=torch.int64).reshape(256, k // 256).T.flatten()
    else:
        order = torch.randperm(k, generator=torch.Generator().manual_seed(k))
    values = (torch.arange(k, dtype=torch.int64) % 256).to(torch.uint8)
    order_bytes = order.contiguous().view(torch.uint8)
    selected = []
    for column in range(0, k, SELECT_COLUMNS):
        # Ascend950 is little-endian. The first Gather selects the low u32
        # word at byte offsets 0, 8, ... in a 256-entry int64 order segment.
        segment = order_bytes[column * 8 : (column + SELECT_COLUMNS) * 8]
        byte_offsets = torch.arange(SELECT_COLUMNS, dtype=torch.int64) * 8
        low_words = segment.view(torch.int32)[byte_offsets // 4].long()
        assert torch.equal(low_words, order[column : column + SELECT_COLUMNS])
        selected.append(values[low_words])
    reordered = torch.cat(selected)
    assert torch.equal(reordered, values[order])
    assert torch.bincount(reordered.long(), minlength=256).tolist() == [k // 256] * 256


def test_vector_prepare_has_bounded_ub_only_overhead_and_no_resident_metadata_copy():
    source = (SOURCE / "resident_prepare.cpp").read_text()
    # Maximum: existing 896 bytes + row + int64 order + u32 offsets twice.
    existing_bytes = SELECT_COLUMNS + SELECT_COLUMNS * 2 + 96 + 32
    extra_bytes = MAX_K + SELECT_COLUMNS * (8 + 4 + 4)
    assert existing_bytes + extra_bytes == 9088
    assert "pipe_.InitBuffer(inputUb_, k_)" in source
    constructor = function((SOURCE / "torch_binding.cpp").read_text(), "ResidentBank")
    assert constructor.count("host.to(") == 1
    assert "constexpr uint32_t kBankWords = 8;" in (SOURCE / "resident_layout.h").read_text()


def test_vectorized_validation_flag_reaches_the_child_and_cannot_claim_kernel_only_coverage():
    assert validate.parse_args([]).activation_reorder == "scalar"
    with pytest.raises(SystemExit):
        validate.parse_args(["--activation-reorder", "vectorized"])
    for phase in ("resident", "lifetime", "graph", "all"):
        args = validate.parse_args(["--phase", phase, "--activation-reorder", "vectorized"])
        plan = validate.validation_plan(args)
        command = plan["command"]
        assert command[command.index("--activation-reorder") + 1] == "vectorized"
        assert plan["activation_reorder"] == "vectorized"
        assert plan["activation_reorder_verified"] is False
        assert plan["device_execution_verified"] is False


def test_selected_probe_bank_has_no_scalar_fallback_or_global_owner_retention():
    calls = []

    class Native:
        def project(self, *args):
            pytest.fail("scalar fallback")

        def project_vectorized(self, *args):
            calls.append(("project_vectorized", args))
            return "project_result"

        def prepare_vectorized(self, *args):
            calls.append(("prepare_vectorized", args))
            return "raw_bytes"

        def select(self, ids):
            return ("select", ids)

        def metadata(self):
            return "metadata"

    assert validate.selected_bank_factory(Native, "scalar") is Native
    factory = validate.selected_bank_factory(Native, "vectorized")
    proxy = factory()
    owner = weakref.ref(proxy.bank)
    assert proxy.project(1, 2, 3, 4) == "project_result"
    assert proxy.prepare_vectorized(1, 2, 3, 4) == "raw_bytes"
    assert proxy.select(7) == ("select", 7)
    assert proxy.metadata() == "metadata"
    assert [name for name, _ in calls] == ["project_vectorized", "prepare_vectorized"]
    del proxy
    assert owner() is None
    with pytest.raises(ValueError):
        validate.selected_bank_factory(Native, "typo")


@pytest.mark.parametrize("k", (2048, 4096))
def test_raw_byte_device_probe_oracle_detects_corruption_without_projecting(k):
    experts = [
        {"k": k, "converted": {"activation_order": torch.randperm(k, generator=torch.Generator().manual_seed(i))}}
        for i in range(3)
    ]

    class CpuLayoutDouble:
        corrupt = False

        def prepare_vectorized(self, q, scale, bias, ids):
            # This double tests the DEVICE PROBE'S oracle, not the AscendC
            # implementation; the actual probe invokes the compiled method.
            raw = q.view(torch.uint8)
            got = torch.zeros_like(raw)
            valid = torch.zeros(ids.numel(), dtype=torch.int32)
            for route, expert in enumerate(ids.tolist()):
                if 0 <= expert < len(experts):
                    order = experts[expert]["converted"]["activation_order"]
                    got[route] = raw[route].index_select(-1, order)
                    valid[route] = 1
            if self.corrupt:
                got.reshape(-1)[0] ^= 1
            return got.view(torch.float8_e4m3fn), valid

    native = CpuLayoutDouble()
    completed = validate.run_vectorized_byte_checks("cpu", native, experts, nullcontext)
    assert len(completed) == 9
    native.corrupt = True
    with pytest.raises(AssertionError, match="gathered FP8 bytes differ"):
        validate.run_vectorized_byte_checks("cpu", native, experts, nullcontext)
