# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_ascend.quantization.vq2a8_decoder_input_plan import install_decoder_input_plan


class NativeOracle:
    def __init__(self, tables, slots, sizes, limits):
        self.tables, self.slots, self.sizes, self.limits = tables, slots, sizes, limits
        self.calls = []

    def copy_rows(self, packed):
        self.calls.append("copy")
        offset = 0
        for table in self.tables:
            count = table.shape[1]
            table[0].copy_(packed[offset : offset + count])
            offset += count

    def slot_mapping(self, starts, positions):
        assert starts.tolist() == [0, 1]
        self.calls.append("slots")
        position = int(positions[0])
        for table, slots, size, limit in zip(self.tables, self.slots, self.sizes, self.limits):
            slots[:limit].fill_(-1)
            slots[0] = int(table[0, position // size]) * size + position % size


class Groups:
    def __init__(self):
        self.commits = self.maps = 0
        self.block_tables = []
        for gid in range(6):
            cpu = torch.tensor([[gid + 1, gid + 11, gid + 21, gid + 31], [91, 92, 93, 94]], dtype=torch.int32)
            table = torch.full_like(cpu, 777)
            self.block_tables.append(
                NS(
                    block_table=NS(cpu=cpu, gpu=table),
                    slot_mapping=NS(gpu=torch.full((7,), 888, dtype=torch.int32)),
                    block_size=2 ** (gid % 3 + 1),
                    physical_block_size=8,
                    blocks_per_phys_block=1,
                    use_hybrid_blocks=False,
                    max_num_batched_tokens=5,
                    is_mamba_group=False,
                    pcp_world_size=1,
                    dcp_world_size=1,
                    cp_kv_cache_interleave_size=1,
                )
            )

    def commit_block_table(self, count):
        self.commits += 1
        for block in self.block_tables:
            block.block_table.gpu[:count].copy_(block.block_table.cpu[:count])

    def compute_slot_mapping(
        self, count, query, positions, positions_compressed_list=None, req_indices_compressed_list=None
    ):
        self.maps += 1
        if count == 1 and positions.numel() == 1:
            for block in self.block_tables:
                position = int(positions[0])
                block.slot_mapping.gpu[: block.max_num_batched_tokens].fill_(-1)
                if position // block.block_size < block.block_table.gpu.shape[1]:
                    block.slot_mapping.gpu[0] = (
                        int(block.block_table.gpu[0, position // block.block_size]) * block.block_size
                        + position % block.block_size
                    )


def fixture():
    group = Groups()
    batch = NS(
        block_table=group,
        num_reqs=1,
        req_ids=["A"],
        req_prompt_embeds={},
        num_computed_tokens_cpu=np.array([1], dtype=np.int32),
        num_prompt_tokens=np.array([1], dtype=np.int32),
    )
    runner = NS(
        input_batch=batch,
        vllm_config=NS(
            parallel_config=NS(), scheduler_config=NS(max_num_seqs=1, async_scheduling=False), cache_config=NS()
        ),
        token=101,
        general_calls=0,
        side_effects=[],
    )

    def prepare(scheduler, scheduled):
        runner.general_calls += 1
        # Side effects before, between and after the two intercepted calls
        # demonstrate that the whole original callable still executes.
        runner.side_effects.append(("before", runner.token))
        runner.input_batch.block_table.commit_block_table(batch.num_reqs)
        runner.cpu_mirror = (runner.token, int(batch.num_computed_tokens_cpu[0]))
        query = torch.tensor([0, 1], dtype=torch.int32)
        positions = torch.tensor([runner.cpu_mirror[1]], dtype=torch.int64)
        runner.input_batch.block_table.compute_slot_mapping(batch.num_reqs, query, positions)
        runner.side_effects.append(("after", runner.token))
        return runner.cpu_mirror, None, scheduler.total_num_scheduled_tokens

    runner._prepare_inputs = prepare
    scheduler = NS(total_num_scheduled_tokens=1, num_scheduled_tokens={"A": 1}, scheduled_spec_decode_tokens={})
    return runner, scheduler, np.array([1], dtype=np.int32)


def install(runner):
    return install_decoder_input_plan(
        runner, "b1_packed", native_factory=NativeOracle, upload=lambda rows, device: torch.cat(rows).clone()
    )


def snapshot(runner):
    return [
        (block.block_table.gpu.clone(), block.slot_mapping.gpu.clone())
        for block in runner.input_batch.block_table.block_tables
    ]


def test_input_plan_general_default_does_not_patch_or_load_native():
    runner, _, _ = fixture()
    method = runner._prepare_inputs
    assert install_decoder_input_plan(runner) is None
    assert runner._prepare_inputs is method


@pytest.mark.parametrize("position", [1, 2, 3, 4, 7])
def test_input_plan_matches_general_full_storage_a_b_a_live_kv_and_positions(position):
    reference, ref_sched, scheduled = fixture()
    candidate, cand_sched, _ = fixture()
    plan = install(candidate)
    addresses = [(b.block_table.gpu.data_ptr(), b.slot_mapping.gpu.data_ptr()) for b in plan.blocks]
    for request, token, shift in (("A", 101, 0), ("B", 202, 20), ("A", 303, 0)):
        for runner, sched in ((reference, ref_sched), (candidate, cand_sched)):
            runner.input_batch.req_ids[:] = [request]
            sched.num_scheduled_tokens = {request: 1}
            runner.token = token
            runner.input_batch.num_computed_tokens_cpu[0] = position
            for gid, block in enumerate(runner.input_batch.block_table.block_tables):
                block.block_table.cpu[0] = torch.tensor([gid + shift + j for j in range(4)])
        assert reference._prepare_inputs(ref_sched, scheduled) == candidate._prepare_inputs(cand_sched, scheduled)
        for old, new in zip(snapshot(reference), snapshot(candidate)):
            for expected, actual in zip(old, new):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert addresses == [(b.block_table.gpu.data_ptr(), b.slot_mapping.gpu.data_ptr()) for b in plan.blocks]
    assert candidate.side_effects == reference.side_effects
    assert candidate.general_calls == reference.general_calls == 3
    assert candidate.input_batch.block_table.commits == candidate.input_batch.block_table.maps == 0
    assert plan.report()["fastpath_calls"] == plan.report()["packed_uploads"] == 3


@pytest.mark.parametrize(
    "failure",
    [
        "prefill",
        "chunked_prefill",
        "batch",
        "spec",
        "async",
        "cp",
        "mm",
        "embeds",
        "request_schema",
        "new_table",
        "new_slots",
        "geometry",
        "disabled",
        "past_capacity",
        "prefix_cache",
    ],
)
def test_input_plan_fallback_before_any_native_write(failure):
    runner, sched, scheduled = fixture()
    plan = install(runner)
    if failure == "prefill":
        runner.input_batch.num_computed_tokens_cpu[0] = 0
    elif failure == "chunked_prefill":
        runner.input_batch.num_prompt_tokens[0] = 4
    elif failure == "batch":
        runner.input_batch.num_reqs = 2
    elif failure == "spec":
        sched.scheduled_spec_decode_tokens = {"A": [2]}
    elif failure == "async":
        runner.use_async_scheduling = True
    elif failure == "cp":
        runner.use_cp = True
    elif failure == "mm":
        runner.is_multimodal_model = True
    elif failure == "embeds":
        runner.input_batch.req_prompt_embeds = {0: torch.ones(1)}
    elif failure == "request_schema":
        sched.num_scheduled_tokens = {"B": 1}
    elif failure == "new_table":
        plan.blocks[0].block_table.gpu = plan.blocks[0].block_table.gpu.clone()
    elif failure == "new_slots":
        plan.blocks[0].slot_mapping.gpu = plan.blocks[0].slot_mapping.gpu.clone()
    elif failure == "geometry":
        plan.blocks[0].block_size = 4
    elif failure == "disabled":
        plan.set_enabled(False)
    elif failure == "past_capacity":
        runner.input_batch.num_computed_tokens_cpu[0] = 99
    else:
        runner.vllm_config.cache_config.enable_prefix_caching = True
    runner._prepare_inputs(sched, scheduled)
    assert plan.native.calls == []
    assert runner.input_batch.block_table.commits == runner.input_batch.block_table.maps == 1
    assert plan.fallbacks == 1 and plan.hits == 0


def test_input_plan_toggle_and_detach_are_instance_local():
    runner, sched, scheduled = fixture()
    other, _, _ = fixture()
    original = runner._prepare_inputs
    plan = install(runner)
    for enabled in (False, True, False):
        plan.set_enabled(enabled)
        runner._prepare_inputs(sched, scheduled)
    assert plan.hits == 1 and plan.fallbacks == 2
    assert other.input_batch.block_table.commits == 0
    assert "commit_block_table" not in other.input_batch.block_table.__dict__
    with pytest.raises(ValueError):
        plan.set_enabled(1)
    plan.detach()
    assert runner._prepare_inputs is original
    assert not hasattr(runner, "_vq2a8_decoder_input_plan")
    runner._prepare_inputs(sched, scheduled)
    assert runner.input_batch.block_table.commits == 3


def test_input_plan_failure_never_falls_back_after_partial_work():
    runner, sched, scheduled = fixture()
    plan = install(runner)

    def fail(packed):
        raise RuntimeError("native failure")

    plan.native.copy_rows = fail
    with pytest.raises(RuntimeError, match="native failure"):
        runner._prepare_inputs(sched, scheduled)
    assert not plan.active and not plan.lock.locked()
    assert runner.input_batch.block_table.commits == runner.input_batch.block_table.maps == 0


def test_input_plan_alias_targets_rejected_before_install():
    runner, _, _ = fixture()
    blocks = runner.input_batch.block_table.block_tables
    blocks[1].slot_mapping.gpu = blocks[0].slot_mapping.gpu
    with pytest.raises(ValueError, match="alias"):
        install(runner)
    assert not hasattr(runner, "_vq2a8_decoder_input_plan")


def test_input_plan_no_class_patch_and_native_source_lifetime_contract():
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    native = (root / "csrc/vq2a8_ascendc_v4_v2/input_plan_binding.cpp").read_text()
    assert "launchStream, descriptors, tables, slots, packed, groups" in native
    assert "launchStream, descriptors, tables, slots, query, positions, groups" in native
    assert "RecordOwners(stream)" in native
    assert "CheckInputAlias" in native


@pytest.mark.parametrize("version", [True, 1.0, "1", None, 0, 2])
def test_input_plan_rejects_noninteger_or_wrong_native_abi(monkeypatch, version):
    from vllm_ascend.quantization.vq2a8_decoder_input_plan import _native_factory

    monkeypatch.setattr(torch.ops, "vq2a8_ascendc_v4_v2", NS(decoder_input_plan_version=lambda: version))
    with pytest.raises(RuntimeError, match="ABI"):
        _native_factory([], [], [], [])


def test_input_plan_snapshot_is_independent_of_reused_cpu_rows(monkeypatch):
    from vllm_ascend.quantization.vq2a8_decoder_input_plan import _snapshot_upload

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value: value)
    rows = [torch.tensor([2, 3], dtype=torch.int32), torch.tensor([4, 5], dtype=torch.int32)]
    packed = _snapshot_upload(rows, torch.device("cpu"))
    rows[0].fill_(99)
    rows[1].fill_(101)
    assert packed.tolist() == [2, 3, 4, 5]


def test_input_plan_does_not_convert_a_device_scalar_in_eligibility():
    runner, sched, scheduled = fixture()
    plan = install(runner)

    class ForbiddenScalar:
        def __int__(self):
            raise AssertionError("must not read a device scalar")

    runner.input_batch.num_computed_tokens_cpu = [ForbiddenScalar()]
    assert not plan._eligible(sched, scheduled)
