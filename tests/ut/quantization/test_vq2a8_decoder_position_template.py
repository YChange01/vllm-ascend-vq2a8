# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU protocol tests; payload/real-model equivalence still needs NPU gates."""

import ast
from copy import copy
from dataclasses import field, make_dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.quantization.vq2a8_decoder_position_template import (
    DECODE_FIELDS,
    ROOT_FIELDS,
    PositionTemplateAdapter,
    PositionTemplateBuffers,
    install_position_template,
)


class DeviceTensor(torch.Tensor):
    @property
    def device(self):
        return torch.device("cuda:0")


def device(value, dtype=torch.int32):
    return torch.Tensor._make_subclass(DeviceTensor, torch.tensor(value, dtype=dtype), False)


def schema(name, names):
    result = make_dataclass(
        name,
        [(name, object, field(default=None)) for name in sorted(names)],
        namespace={"__module__": "vllm_ascend.attention.dsa_v1"},
    )
    result.__module__ = "vllm_ascend.attention.dsa_v1"
    return result


Root = schema("AscendDSAMetadata", ROOT_FIELDS)
Decode = schema("AscendDSADecodeMetadata", DECODE_FIELDS)


class State(Enum):
    DecodeOnly = 2


class RopeDataProxy:
    __module__ = "vllm_ascend.ops.rope_dsv4"

    def __init__(self, data, idx):
        self._data, self.idx = data, idx


def metadata(position=3, table=None, slots=None):
    table = device([[2, 0]]) if table is None else table
    slots = device([position + 16]) if slots is None else slots
    query = device([0, 1])
    lengths = device([position + 1])
    data = {"config": {"default": (device([position], torch.float32), device([-position], torch.float32))}}
    cos, sin = RopeDataProxy(data, 0), RopeDataProxy(data, 1)
    decode = Decode(
        input_positions=device([position], torch.int64),
        block_table=table,
        seq_lens=lengths,
        max_seqlen_kv=position + 1,
        max_seqlen_q=1,
        seq_lens_list=[position + 1],
        max_seq_lens=position + 1,
        slot_mapping=slots,
        block_size=8,
        num_compressed_tokens=1,
        query_start_loc=query,
        query_start_loc_cpu=torch.tensor([0, 1]),
        start_pos=device([position]),
        num_reqs_actual=1,
        sas_metadata=device([position] * 1024),
        qli_metadata=device([position] * 1024),
        cos=cos,
        sin=sin,
    )
    root = Root(
        num_actual_tokens=1,
        query_start_loc=query,
        seq_lens=lengths,
        num_decodes=1,
        num_decode_tokens=1,
        num_prefills=0,
        num_input_tokens=1,
        query_lens=torch.tensor([1]),
        head_dim=512,
        attn_state=State.DecodeOnly,
        decode=decode,
        cos=cos,
        sin=sin,
    )
    return {"layer0": root, "layer1": root}


def bound_owner(position=3):
    owner = PositionTemplateBuffers(metadata(position))
    sources = {(0, "block_table"): device([[9, 1]]), (0, "slot_mapping"): device([72 + position])}
    owner.bind({"layer0": (0, 8), "layer1": (0, 8)}, sources)
    return owner, sources


@pytest.mark.parametrize("position", range(16))
def test_position_specialization_refreshes_same_position_a_b_a(position):
    owner, sources = bound_owner(position)
    untouched = owner.tree["layer0"].decode.sas_metadata.clone()
    for block_id in (3, 9, 3):
        sources[(0, "block_table")].fill_(block_id)
        sources[(0, "slot_mapping")].fill_(block_id * 8 + position % 8)
        owner.update(owner.request(sources))
        assert owner.tree["layer0"].decode.block_table.tolist() == [[block_id, block_id]]
        assert owner.tree["layer0"].decode.slot_mapping.tolist() == [block_id * 8 + position % 8]
        torch.testing.assert_close(owner.tree["layer0"].decode.sas_metadata, untouched)
    assert owner.copies == 6
    assert owner.requests == 3
    assert owner.baseline_device_targets > len(owner.bindings)


def test_alias_targets_copy_once_and_cross_group_alias_is_rejected():
    owner, _ = bound_owner()
    assert len(owner.bindings) == 2
    other = PositionTemplateBuffers(metadata())
    sources = {
        (gid, field): device([[2, 0]]) if field == "block_table" else device([19])
        for gid in (0, 1)
        for field in ("block_table", "slot_mapping")
    }
    with pytest.raises(ValueError, match="alias"):
        other.bind({"layer0": (0, 8), "layer1": (1, 8)}, sources)


@pytest.mark.parametrize("failure", ("source_shape", "source_dtype", "static_shape", "static_scalar", "request_tree"))
def test_all_validation_precedes_every_copy(failure):
    owner, sources = bound_owner()
    request = owner.request(sources)
    old = owner.tree["layer0"].decode.block_table.clone()
    if failure == "source_shape":
        sources[(0, "slot_mapping")] = device([1, 2])
    elif failure == "source_dtype":
        sources[(0, "slot_mapping")] = device([1], torch.int64)
    elif failure == "static_shape":
        owner.tree["layer0"].decode.sas_metadata = device([1, 2])
    elif failure == "static_scalar":
        owner.tree["layer0"].decode.max_seq_lens = 99
    else:
        request["layer0"] = metadata()["layer0"]
    with pytest.raises(ValueError):
        owner.update(request)
    torch.testing.assert_close(owner.tree["layer0"].decode.block_table, old)
    assert owner.copies == 0
    assert not request.consumed


@pytest.mark.parametrize(
    "failure",
    (
        "tensor_same_contract",
        "tensor_set",
        "tensor_resize",
        "tensor_stride",
        "cpu_payload",
        "cpu_query_starts",
        "root_replaced",
        "dataclass_replaced",
        "dataclass_type",
        "dataclass_schema",
        "dict_replaced",
        "dict_keys",
        "sequence_replaced",
        "sequence_length",
        "sequence_constant_type",
        "proxy_replaced",
        "proxy_selector",
        "alias_split",
        "alias_merge",
    ),
)
def test_owned_metadata_guards_reject_mutations_before_any_live_copy(failure):
    owner, sources = bound_owner()
    request = owner.request(sources)
    root = owner.tree["layer0"]
    decode = root.decode
    before = decode.block_table.clone()
    if failure == "tensor_same_contract":
        decode.sas_metadata = decode.sas_metadata.clone() + 1
    elif failure == "tensor_set":
        decode.sas_metadata.set_(decode.sas_metadata.clone())
    elif failure == "tensor_resize":
        decode.sas_metadata.resize_(1023)
    elif failure == "tensor_stride":
        decode.sas_metadata.as_strided_((1024,), (0,))
    elif failure == "cpu_payload":
        root.query_lens[0] = 2
    elif failure == "cpu_query_starts":
        decode.query_start_loc_cpu[1] = 2
    elif failure == "root_replaced":
        owner.tree = dict(owner.tree)
    elif failure == "dataclass_replaced":
        root.decode = copy(decode)
    elif failure == "dataclass_type":
        decode.__class__ = schema("AscendDSADecodeMetadata", DECODE_FIELDS)
    elif failure == "dataclass_schema":
        decode.__dataclass_fields__ = {**decode.__dataclass_fields__, "unexpected": object()}
    elif failure == "dict_replaced":
        root.cos._data = dict(root.cos._data)
    elif failure == "dict_keys":
        root.cos._data["unexpected"] = None
    elif failure == "sequence_replaced":
        decode.seq_lens_list = list(decode.seq_lens_list)
    elif failure == "sequence_length":
        decode.seq_lens_list.append(4)
    elif failure == "sequence_constant_type":
        decode.seq_lens_list[0] = float(decode.seq_lens_list[0])
    elif failure == "proxy_replaced":
        root.cos = copy(root.cos)
    elif failure == "proxy_selector":
        root.cos.idx = 1
    elif failure == "alias_split":
        decode.seq_lens = root.seq_lens.clone()
    else:
        decode.sas_metadata = decode.qli_metadata
    with pytest.raises(ValueError, match="Position template"):
        owner.update(request)
    torch.testing.assert_close(decode.block_table, before)
    assert owner.copies == 0
    assert not request.consumed


def test_owned_update_does_not_use_the_generic_source_validation_dag(monkeypatch):
    owner, sources = bound_owner()

    def unexpected(*args, **kwargs):
        raise AssertionError("Generic source validation must not run on owned metadata")

    monkeypatch.setattr(owner, "_validate", unexpected)
    owner.update(owner.request(sources))
    assert owner.copies == 2


def test_owned_cpu_snapshots_do_not_alias_the_exposed_template():
    owner, sources = bound_owner()
    cpu = owner.tree["layer0"].query_lens
    for value in (2, 1, 3, 1):
        cpu.fill_(value)
        if value == 1:
            owner.update(owner.request(sources))
        else:
            with pytest.raises(ValueError, match="CPU payload"):
                owner.update(owner.request(sources))
    assert owner.copies == 4


@pytest.mark.parametrize("replacement", (False, True))
def test_owned_immutable_tensor_pointer_and_field_are_guarded(replacement):
    values = metadata()
    values["layer0"].hadamard = device([1.0], torch.float32)
    owner = PositionTemplateBuffers(values)
    sources = {(0, "block_table"): device([[9, 1]]), (0, "slot_mapping"): device([75])}
    owner.bind({"layer0": (0, 8), "layer1": (0, 8)}, sources)
    if replacement:
        owner.tree["layer0"].hadamard = owner.tree["layer0"].hadamard.clone()
    else:
        owner.tree["layer0"].hadamard.set_(owner.tree["layer0"].hadamard.clone())
    with pytest.raises(ValueError, match="Position template owned"):
        owner.update(owner.request(sources))
    assert owner.copies == 0


def test_request_is_one_use_and_cannot_be_used_for_another_position():
    owner, sources = bound_owner()
    request = owner.request(sources)
    other, _ = bound_owner(4)
    with pytest.raises(ValueError, match="own fresh"):
        other.update(request)
    owner.update(request)
    with pytest.raises(ValueError, match="own fresh"):
        owner.update(request)


@pytest.mark.parametrize("failure", ("unknown_field", "unknown_type", "prefill", "mask", "cp", "padding", "slot_2d"))
def test_closed_schema_rejects_unmodeled_dependencies(failure):
    values = metadata()
    value = values["layer0"]
    if failure == "unknown_field":
        new_type = schema("AscendDSADecodeMetadata", DECODE_FIELDS | {"future_dynamic"})
        value.decode = new_type(**value.decode.__dict__)
    elif failure == "unknown_type":
        value.decode = SimpleNamespace(**value.decode.__dict__)
    elif failure == "prefill":
        value.num_prefills = 1
    elif failure == "mask":
        value.decode.attn_mask = device([1])
    elif failure == "cp":
        value.decode.cp_seq_len = device([1])
    elif failure == "padding":
        value.num_input_tokens = 2
    else:
        value.decode.slot_mapping = device([[2, 3]])
    with pytest.raises(ValueError):
        PositionTemplateBuffers(values)


def test_shadow_reference_checks_every_static_payload_and_live_fields():
    owner, sources = bound_owner()
    original = metadata(3, sources[(0, "block_table")], sources[(0, "slot_mapping")])
    owner.compare_reference(original, sources)
    original["layer0"].decode.sas_metadata[1] += 1
    with pytest.raises(AssertionError):
        owner.compare_reference(original, sources)
    assert owner.copies == 0


@pytest.mark.parametrize("field,last_defined", (("sas_metadata", 899), ("qli_metadata", 863)))
@pytest.mark.parametrize("region", ("first", "last_defined", "first_reserved", "last_reserved"))
def test_shadow_descriptor_mismatch_reports_location_without_skipping_any_word(field, last_defined, region):
    owner, sources = bound_owner()
    original = metadata(3, sources[(0, "block_table")], sources[(0, "slot_mapping")])
    index = {"first": 0, "last_defined": last_defined, "first_reserved": last_defined + 1, "last_reserved": 1023}[
        region
    ]
    getattr(original["layer0"].decode, field)[index] += 7
    with pytest.raises(AssertionError) as failure:
        owner.compare_reference(original, sources)
    message = str(failure.value)
    assert f"fields=['{field}']" in message
    assert f"layer0.decode.{field}" in message
    assert "position=3" in message
    assert "unequal_elements=1/1024" in message
    assert f"first_flat_index={index}" in message
    assert f"last_flat_index={index}" in message
    assert f"({index}, 10, 3)" in message
    assert "rtol=0, atol=0" in message
    assert owner.copies == 0


def test_shadow_descriptor_mismatch_bounds_samples_but_reports_full_range():
    owner, sources = bound_owner()
    original = metadata(3, sources[(0, "block_table")], sources[(0, "slot_mapping")])
    original["layer0"].decode.sas_metadata[900:] += 7
    with pytest.raises(AssertionError) as failure:
        owner.compare_reference(original, sources)
    message = str(failure.value)
    assert "unequal_elements=124/1024" in message
    assert "first_flat_index=900" in message and "last_flat_index=1023" in message
    assert "(907, 10, 3)" in message and "(908, 10, 3)" not in message
    assert owner.copies == 0


def fake_runner(positions=(3, 4)):
    table, slots = device([[2, 0]]), device([19])
    block = SimpleNamespace(
        get_device_tensor=lambda: table,
        slot_mapping=SimpleNamespace(gpu=slots),
        block_size=8,
        physical_block_size=8,
    )
    builder_type = type("AscendDSAMetadataBuilder", (), {"__module__": "vllm_ascend.attention.dsa_v1"})
    builder = builder_type()
    builder.compressor_ratio = 1
    builder.model_config = SimpleNamespace(
        get_head_size=lambda: 512,
        hf_config=SimpleNamespace(
            num_attention_heads=8, index_topk=512, index_n_heads=64, index_head_dim=128, sliding_window=128
        ),
    )
    group = SimpleNamespace(
        layer_names=["layer0", "layer1"],
        kv_cache_spec=SimpleNamespace(block_size=8),
        get_metadata_builder=lambda index: builder,
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(),
            scheduler_config=SimpleNamespace(max_num_seqs=1, async_scheduling=False),
            cache_config=SimpleNamespace(),
        ),
        input_batch=SimpleNamespace(
            block_table=[block],
            num_reqs=1,
            req_ids=["a"],
            req_prompt_embeds={},
            num_computed_tokens_cpu_tensor=torch.tensor([positions[0]]),
        ),
        attn_groups=[[group]],
        # The Ascend platform can normalize disable_cascade_attn to False;
        # a true heuristic capability flag does not mean an active cascade.
        cascade_attn_enabled=True,
        with_prefill=False,
        attn_state=State.DecodeOnly,
        optimistic_seq_lens_cpu=torch.tensor([positions[0] + 1]),
        query_start_loc=SimpleNamespace(cpu=torch.tensor([0, 1])),
        _dsa_positions_cpu_buf=torch.tensor([positions[0]]),
    )
    original_calls = []

    def original(*args, **kwargs):
        original_calls.append((args, kwargs))
        position = runner.optimistic_seq_lens_cpu.tolist()[0] - 1
        return metadata(position, table[:1], slots[:1]), None

    runner._build_attention_metadata = original
    bank = SimpleNamespace(
        model=SimpleNamespace(_v4_graph_enabled=True),
        max_model_len=16,
        entries={position: {"metadata": PositionTemplateBuffers(metadata(position))} for position in positions},
    )
    return runner, bank, original_calls


def build_args(request="a"):
    return dict(num_tokens=1, num_reqs=1, max_query_len=1, num_scheduled_tokens={request: 1})


@pytest.mark.parametrize("capability", (False, True))
def test_cascade_capability_without_active_prefixes_allows_templates(capability):
    runner, bank, calls = fake_runner()
    runner.cascade_attn_enabled = capability
    adapter = install_position_template(runner, bank)
    request, extra = runner._build_attention_metadata(**build_args(), cascade_attn_prefix_lens=None)
    bank.entries[3]["metadata"].update(request)
    assert extra is None
    assert adapter.hits == 1
    assert not calls


@pytest.mark.parametrize("capability", (False, True))
@pytest.mark.parametrize("prefixes", ([], [[0]], [[128]]))
def test_any_explicit_cascade_prefixes_still_fail_before_template_use(capability, prefixes):
    runner, bank, calls = fake_runner()
    runner.cascade_attn_enabled = capability
    adapter = install_position_template(runner, bank)
    with pytest.raises(ValueError, match="alternate decode builders"):
        runner._build_attention_metadata(**build_args(), cascade_attn_prefix_lens=prefixes)
    assert not calls
    assert adapter.hits == 0
    assert all(entry["metadata"].copies == 0 for entry in bank.entries.values())


@pytest.mark.parametrize(
    "feature",
    (
        "use_cp",
        "use_async_scheduling",
        "use_async_spec_decode",
        "is_multimodal_model",
        "enable_prompt_embeds",
        "is_mm_prefix_lm",
        "enable_hamming_sparse",
    ),
)
@pytest.mark.parametrize("during_replay", (False, True))
def test_unsupported_dynamic_features_are_named_and_still_rejected(feature, during_replay):
    runner, bank, calls = fake_runner()
    if during_replay:
        adapter = install_position_template(runner, bank)
    setattr(runner, feature, True)
    with pytest.raises(ValueError, match=f"unsupported dynamic features: {feature}"):
        if during_replay:
            runner._build_attention_metadata(**build_args())
        else:
            install_position_template(runner, bank)
    assert not calls
    assert all(entry["metadata"].copies == 0 for entry in bank.entries.values())
    if during_replay:
        assert adapter.hits == 0
    else:
        assert not hasattr(runner, "_vq2a8_position_template")


def test_instance_adapter_really_skips_original_builder_and_supports_changed_requests():
    runner, bank, original_calls = fake_runner()
    adapter = install_position_template(runner, bank)
    for request_id, position, block_id in (("a", 3, 2), ("b", 4, 7), ("a", 3, 11)):
        runner.input_batch.req_ids = [request_id]
        runner.optimistic_seq_lens_cpu.fill_(position + 1)
        runner.input_batch.num_computed_tokens_cpu_tensor.fill_(position)
        runner._dsa_positions_cpu_buf.fill_(position)
        runner.input_batch.block_table[0].get_device_tensor().fill_(block_id)
        runner.input_batch.block_table[0].slot_mapping.gpu.fill_(block_id * 8 + position)
        value, extra = runner._build_attention_metadata(**build_args(request_id))
        assert extra is None
        bank.entries[position]["metadata"].update(value)
        assert value["layer0"].decode.block_table.tolist() == [[block_id, block_id]]
    assert not original_calls
    assert adapter.report()["original_builder_skips"] == 3
    assert adapter.report()["dynamic_copies"] == 6
    assert not adapter.report()["input_preparation_skipped"]


@pytest.mark.parametrize("prefill,eager", ((True, False), (False, True)))
def test_prefill_and_eager_reference_always_keep_original_builder(prefill, eager):
    runner, bank, calls = fake_runner()
    adapter = install_position_template(runner, bank)
    runner.with_prefill = prefill
    bank.model._v4_graph_enabled = not eager
    runner._build_attention_metadata(**build_args())
    assert len(calls) == 1
    assert adapter.baseline_calls == 1
    assert adapter.hits == 0


@pytest.mark.parametrize(
    "failure",
    (
        "batch",
        "padding",
        "request",
        "position",
        "computed",
        "query",
        "cpu_position",
        "spec",
        "unknown_arg",
        "cp",
        "async",
        "table_owner",
        "slot_owner",
        "block_size",
    ),
)
def test_runtime_guards_fail_without_any_graph_write(failure):
    runner, bank, calls = fake_runner()
    adapter = install_position_template(runner, bank)
    kwargs = build_args()
    if failure == "batch":
        runner.input_batch.num_reqs = 2
    elif failure == "padding":
        kwargs["num_tokens_padded"] = 2
    elif failure == "request":
        kwargs["num_scheduled_tokens"] = {"stale": 1}
    elif failure == "position":
        runner.optimistic_seq_lens_cpu.fill_(17)
    elif failure == "computed":
        runner.input_batch.num_computed_tokens_cpu_tensor.fill_(2)
    elif failure == "query":
        runner.query_start_loc.cpu[1] = 2
    elif failure == "cpu_position":
        runner._dsa_positions_cpu_buf.fill_(4)
    elif failure == "spec":
        kwargs["use_spec_decode"] = True
    elif failure == "unknown_arg":
        kwargs["future_mode"] = 1
    elif failure == "cp":
        runner.use_cp = True
    elif failure == "async":
        runner.vllm_config.scheduler_config.async_scheduling = True
    elif failure == "table_owner":
        runner.input_batch.block_table[0].get_device_tensor = lambda: device([[9, 9]])
    elif failure == "slot_owner":
        runner.input_batch.block_table[0].slot_mapping.gpu = device([99])
    else:
        runner.input_batch.block_table[0].block_size = 16
    with pytest.raises(ValueError):
        runner._build_attention_metadata(**kwargs)
    assert not calls
    assert adapter.hits == 0
    assert all(entry["metadata"].copies == 0 for entry in bank.entries.values())


def test_reference_verification_is_opt_in_and_reports_positions():
    runner, bank, calls = fake_runner()
    adapter = PositionTemplateAdapter(runner, bank)
    adapter.verify_reference = True
    request, _ = adapter.build(**build_args())
    bank.entries[3]["metadata"].update(request)
    assert len(calls) == 1
    assert adapter.report()["reference_checks"] == 1
    assert adapter.report()["reference_positions"] == [3]
    assert adapter.report()["reference_verification_enabled"]
    assert adapter.report()["template_requests"] == 1
    assert adapter.report()["original_builder_skips"] == 0
    assert adapter.report()["original_builder_calls"] == 1
    assert adapter.report()["baseline_builder_calls"] == 0


@pytest.mark.parametrize(
    "name, expected", (("AscendDSAMetadata", ROOT_FIELDS), ("AscendDSADecodeMetadata", DECODE_FIELDS))
)
def test_schema_manifest_matches_the_actual_backend_source(name, expected):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/dsa_v1.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name)
    assert {node.target.id for node in cls.body if isinstance(node, ast.AnnAssign)} == expected


@pytest.mark.parametrize("change", ("group", "layer_names", "builder", "ratio", "heads", "routed_experts"))
def test_changed_builder_configuration_cannot_reuse_old_position_templates(change):
    runner, bank, calls = fake_runner()
    adapter = install_position_template(runner, bank)
    group = runner.attn_groups[0][0]
    builder = group.get_metadata_builder(0)
    if change == "group":
        runner.attn_groups = []
    elif change == "layer_names":
        group.layer_names = ["layer0"]
    elif change == "builder":
        group.get_metadata_builder = lambda index: SimpleNamespace(**builder.__dict__)
    elif change == "ratio":
        builder.compressor_ratio = 128
    elif change == "heads":
        builder.model_config.hf_config.num_attention_heads = 16
    else:
        runner.model_config = SimpleNamespace(enable_return_routed_experts=True)
    with pytest.raises(ValueError):
        runner._build_attention_metadata(**build_args())
    assert not calls
    assert adapter.hits == 0


def test_detach_restores_instance_method_and_drops_cache_owners():
    runner, bank, calls = fake_runner()
    original = runner._build_attention_metadata
    adapter = install_position_template(runner, bank)
    adapter.detach()
    assert runner._build_attention_metadata is original
    assert not hasattr(runner, "_vq2a8_position_template")
    assert adapter.runner is adapter.bank is None
    assert not adapter.block_owners
    runner._build_attention_metadata(**build_args())
    assert len(calls) == 1
