# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU mocked TP2 wiring contracts, not distributed/NPU numerical validation."""

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from tools import serve_vq2a8_v3 as server
from vllm_ascend.quantization import vq2a8_offline as offline


def options(tmp_path, **changes):
    return offline.offline_engine_options(
        tmp_path / "model",
        tmp_path / "artifact",
        execution_policy="ascendc_v3",
        ascendc_v3_library=tmp_path / "candidate.so",
        ascendc_v3_sha256="a" * 64,
        tensor_parallel_size=2,
        **changes,
    )


def config(plan):
    return NS(
        additional_config=plan["additional_config"],
        parallel_config=NS(
            tensor_parallel_size=plan["tensor_parallel_size"],
            pipeline_parallel_size=1,
            data_parallel_size=1,
            distributed_executor_backend=plan["distributed_executor_backend"],
        ),
        model_config=NS(enforce_eager=True, quantization=None, dtype=torch.bfloat16, max_model_len=32),
        quant_config=None,
        scheduler_config=NS(max_num_seqs=1, max_num_batched_tokens=32),
        compilation_config=NS(mode=0, cudagraph_mode=0),
        cache_config=NS(kv_cache_memory_bytes=1024**3),
        load_config=NS(load_format="safetensors"),
    )


@pytest.mark.parametrize("preparation", ["eager", "fused"])
def test_tp2_options_select_new_architecture_and_real_mp_executor(tmp_path, preparation):
    plan = options(tmp_path, v3_preparation=preparation)
    assert plan["hf_overrides"]["architectures"] == ["VQ2A8TP2OfflineForCausalLM"]
    assert plan["tensor_parallel_size"] == 2 and plan["distributed_executor_backend"] == "mp"
    assert offline.validate_offline_config(config(plan)) == plan["additional_config"]["vq2a8_offline"]
    old = offline.offline_engine_options(tmp_path, tmp_path)
    assert old["tensor_parallel_size"] == 1 and old["distributed_executor_backend"] == "uni"
    assert old["hf_overrides"]["architectures"] == ["VQ2A8TP1OfflineForCausalLM"]


@pytest.mark.parametrize("value", [0, 3, True, 2.0, "2"])
def test_tp2_options_reject_invalid_parallel_size(tmp_path, value):
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        offline.offline_engine_options(tmp_path, tmp_path, tensor_parallel_size=value)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("parallel_config", "distributed_executor_backend", "uni"),
        ("parallel_config", "distributed_executor_backend", "ray"),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "data_parallel_size", 2),
        ("parallel_config", "prefill_context_parallel_size", 2),
        ("parallel_config", "decode_context_parallel_size", 2),
        ("parallel_config", "enable_expert_parallel", True),
        ("parallel_config", "use_sequence_parallel_moe", True),
        ("parallel_config", "enable_eplb", True),
        ("model_config", "enforce_eager", False),
        ("model_config", "quantization", "fp8"),
        ("compilation_config", "cudagraph_mode", 1),
        ("scheduler_config", "async_scheduling", True),
    ],
)
def test_tp2_rejects_other_parallel_and_graph_modes(tmp_path, section, key, value):
    cfg = config(options(tmp_path))
    setattr(getattr(cfg, section), key, value)
    with pytest.raises(ValueError):
        offline.validate_offline_config(cfg)


@pytest.mark.parametrize(
    "key,value",
    [("execution_policy", "cached"), ("root_linear_mode", "online_fp8_sm90"), ("v3_decode_graph", "moe")],
)
def test_tp2_rejects_non_v3_bf16_contract(tmp_path, key, value):
    cfg = config(options(tmp_path))
    cfg.additional_config["vq2a8_offline"][key] = value
    with pytest.raises(ValueError, match="TP2"):
        offline.validate_offline_config(cfg)


@pytest.mark.parametrize(
    "key,value",
    [
        ("enable_flashcomm1", True),
        ("enable_dsa_cp", True),
        ("mix_placement", True),
        ("multistream_dsv4_dsa_overlap", True),
        ("finegrained_tp_config", {"oproj_tensor_parallel_size": 1}),
        ("finegrained_tp_config", {"embedding_tensor_parallel_size": 1}),
        ("finegrained_tp_config", None),
    ],
)
def test_tp2_rejects_ascend_parallel_overrides(tmp_path, key, value):
    cfg = config(options(tmp_path))
    cfg.additional_config[key] = value
    with pytest.raises(ValueError, match="TP2"):
        offline.validate_offline_config(cfg)


def root_config():
    return dict(
        hidden_size=8, num_attention_heads=4, head_dim=4, o_groups=2, o_lora_rank=4, q_lora_rank=4, vocab_size=7
    )


def owner_for_root(name, rank):
    owner = offline.OfflineMoEOwner.__new__(offline.OfflineMoEOwner)
    owner.tp_size, owner.tp_rank = 2, rank
    owner.root_config = root_config()
    owner.layers = {}
    owner.inventory = {name: {}}
    return owner


def copy_weight(param, value):
    param.copy_(value)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize(
    "name,shape,axis",
    [
        ("layers.0.attn.wq_b.weight", (16, 4), 0),
        ("layers.0.attn.wo_a.weight", (8, 8), 0),
        ("layers.0.attn.wo_b.weight", (8, 8), 1),
    ],
)
def test_tp2_root_uses_full_checkpoint_and_parameter_shard_loader(rank, name, shape, axis):
    owner = owner_for_root(name, rank)
    source = torch.arange(shape[0] * shape[1]).reshape(shape).to(torch.bfloat16)
    local = source.narrow(axis, rank * (shape[axis] // 2), shape[axis] // 2)
    param = torch.zeros_like(local)
    received = []

    def shard_loader(target, full):
        received.append(full)
        assert full is source, "Must not pre-shard before the standard loader shards again"
        target.copy_(full.narrow(axis, rank * local.shape[axis], local.shape[axis]))

    param.weight_loader = shard_loader
    target = offline.canonical_root_parameter(name)
    loaded, _ = owner.load_root({target: param}, [(name, source)], copy_weight)
    assert loaded == {target} and received == [source]
    assert torch.equal(param, local)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("name", ["embed.weight", "head.weight"])
def test_tp2_root_vocab_loader_preserves_padding(rank, name):
    owner = owner_for_root(name, rank)
    source = torch.arange(7 * 8).reshape(7, 8).to(torch.bfloat16)
    param = torch.full((4, 8), -1, dtype=torch.bfloat16)
    length = 4 if rank == 0 else 3

    def vocab_loader(target, full):
        assert full.shape == (7, 8)
        target.zero_()
        target[:length].copy_(full[rank * 4 : rank * 4 + length])

    param.weight_loader = vocab_loader
    owner.load_root({offline.canonical_root_parameter(name): param}, [(name, source)], copy_weight)
    assert torch.equal(param[:length], source[rank * 4 : rank * 4 + length])
    if rank == 1:
        assert not param[-1].count_nonzero()


@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_root_attention_sink_is_sliced_once(rank):
    name = "layers.0.attn.attn_sink"
    owner = owner_for_root(name, rank)
    source, param = torch.arange(4).float(), torch.zeros(2)
    owner.load_root({offline.canonical_root_parameter(name): param}, [(name, source)], copy_weight)
    assert torch.equal(param, source[rank * 2 : rank * 2 + 2])


@pytest.mark.parametrize("name", ["layers.0.attn.indexer.wq_b.weight", "layers.0.attn_norm.weight", "hc_head_fn"])
def test_tp2_root_non_parallel_families_remain_replicated(name):
    owner = owner_for_root(name, 1)
    source = torch.arange(8).to(torch.bfloat16)
    param = torch.zeros_like(source)
    owner.load_root({offline.canonical_root_parameter(name): param}, [(name, source)], copy_weight)
    assert torch.equal(param, source)


@pytest.mark.parametrize("failure", ["shape", "no_loader", "nan", "dtype", "replicated_shape", "sink_shape"])
def test_tp2_root_loading_remains_fail_closed(failure):
    name = "layers.0.attn.wq_b.weight"
    source = torch.ones(16, 4, dtype=torch.bfloat16)
    param = torch.zeros(8, 4, dtype=torch.bfloat16)
    calls = []
    param.weight_loader = lambda *args: calls.append(args)
    if failure == "shape":
        source = source[:8]
    elif failure == "no_loader":
        del param.weight_loader
    elif failure == "nan":
        source[0, 0] = float("nan")
    elif failure == "dtype":
        source = source.float()
    elif failure == "replicated_shape":
        name = "layers.0.attn.indexer.wq_b.weight"
    elif failure == "sink_shape":
        name, source = "layers.0.attn.attn_sink", torch.ones(4, dtype=torch.bfloat16)
    owner = owner_for_root(name, 0)
    with pytest.raises(ValueError):
        owner.load_root({offline.canonical_root_parameter(name): param}, [(name, source)], copy_weight)
    assert not calls


@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_owner_selects_rank_artifact_and_runtime_without_tp1_reader(tmp_path, monkeypatch, rank):
    from vllm_ascend.quantization import vq2a8_ascendc_v3 as native
    from vllm_ascend.quantization import vq2a8_tp2_runtime as reader

    (tmp_path / "config.json").write_text(json.dumps(root_config()), encoding="utf-8")
    opts = options(tmp_path)["additional_config"]["vq2a8_offline"]
    calls = []
    artifact = NS()
    monkeypatch.setattr(native, "load_pinned_library", lambda *args: NS())
    monkeypatch.setattr(native, "resident_library_capabilities", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms",
        NS(current_platform=NS(visible_device_id_to_physical_device_id=lambda index: index + 3)),
    )
    monkeypatch.setattr(offline, "audit_offline_root", lambda path: {})
    monkeypatch.setattr(offline, "open_vq2a8_tp1_artifact", lambda *a, **kw: pytest.fail("TP1 reader used"))

    def open_tp2(root, model_config, **kwargs):
        calls.append(kwargs)
        return artifact

    monkeypatch.setattr(reader, "open_vq2a8_tp2_artifact", open_tp2)

    class Runtime(offline.CachedVQ2TP1MoE):
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    module = "vllm_ascend.quantization.vq2a8_execution_tp2"
    monkeypatch.setitem(sys.modules, module, NS(AscendCV3VQ2TP2MoE=Runtime))
    group = NS(world_size=2, rank_in_group=rank, all_reduce=lambda tensor: tensor)
    owner = offline.OfflineMoEOwner(tmp_path, opts, NS(type="npu", index=rank), tp_size=2, tp_group=group)
    assert calls == [{"require_tp2": True}, {"tp_rank": rank, "verify_tensor_hashes": True}]
    assert owner.create_layer(0) is owner.layers[0]
    args, kwargs = calls[2]
    assert args[0] is artifact and args[1] == 0
    assert kwargs["tp_group"] is group and kwargs["tp_rank"] == rank
    assert kwargs["projection_kernel"] == "v2" and kwargs["v3_decode_graph"] == "none"
    with pytest.raises(ValueError, match="Duplicate"):
        owner.create_layer(0)


@pytest.mark.parametrize("world,rank", [(1, 0), (3, 0), (2, -1), (2, 2), (2, True), (2.0, 0)])
def test_tp2_owner_rejects_group_mismatch_before_loading(tmp_path, world, rank):
    group = NS(world_size=world, rank_in_group=rank, all_reduce=lambda tensor: tensor)
    with pytest.raises(ValueError, match="TP2 owner"):
        offline.OfflineMoEOwner(tmp_path, {"execution_policy": "ascendc_v3"}, NS(type="npu"), tp_size=2, tp_group=group)


def test_tp2_owner_rejects_old_native_library_before_artifact_hashing(tmp_path, monkeypatch):
    from vllm_ascend.quantization import vq2a8_ascendc_v3 as native
    from vllm_ascend.quantization import vq2a8_tp2_runtime as reader

    (tmp_path / "config.json").write_text(json.dumps(root_config()), encoding="utf-8")
    calls = []
    monkeypatch.setattr(native, "load_pinned_library", lambda *args: calls.append("library"))

    def old_library(**kwargs):
        assert kwargs == {"require_tp2": True}
        calls.append("capability")
        raise RuntimeError("TP2 resident projection capability is unavailable")

    monkeypatch.setattr(native, "resident_library_capabilities", old_library)
    monkeypatch.setattr(reader, "open_vq2a8_tp2_artifact", lambda *a, **k: pytest.fail("hashed before capability"))
    monkeypatch.setattr(offline, "audit_offline_root", lambda *args: pytest.fail("root allocation before capability"))
    group = NS(world_size=2, rank_in_group=0, all_reduce=lambda x: x)
    with pytest.raises(RuntimeError, match="capability"):
        offline.OfflineMoEOwner(
            tmp_path, options(tmp_path)["additional_config"]["vq2a8_offline"], NS(type="npu"), tp_size=2, tp_group=group
        )
    assert calls == ["library", "capability"]


@pytest.mark.parametrize(
    "change",
    [
        {"num_attention_heads": 3},
        {"o_groups": 3},
        {"o_groups": 0},
        {"head_dim": True},
        {"vocab_size": None},
        {"q_lora_rank": "missing"},
    ],
)
def test_tp2_owner_rejects_missing_or_invalid_root_geometry_before_loading(tmp_path, change):
    cfg = {**root_config(), **change}
    if cfg.get("q_lora_rank") == "missing":
        del cfg["q_lora_rank"]
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    group = NS(world_size=2, rank_in_group=0, all_reduce=lambda x: x)
    with pytest.raises(ValueError, match="TP2 root"):
        offline.OfflineMoEOwner(tmp_path, {"execution_policy": "ascendc_v3"}, NS(type="npu"), tp_size=2, tp_group=group)


def test_tp2_architecture_alias_cannot_be_used_as_tp1_or_vice_versa():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/worker/vq2a8_offline_model.py"
    tree = ast.parse(source.read_text("utf-8"))
    names = {"VQ2A8TP1OfflineForCausalLM", "VQ2A8TP2OfflineForCausalLM"}
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    scope = {
        "AscendDeepseekV4ForCausalLM": object,
        "OfflineDecoderModel": object,
        "validate_offline_config": lambda cfg: {},
    }
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), scope)
    for architecture, wrong_tp in (("VQ2A8TP1OfflineForCausalLM", 2), ("VQ2A8TP2OfflineForCausalLM", 1)):
        with pytest.raises(ValueError, match="architecture"):
            scope[architecture](vllm_config=NS(parallel_config=NS(tensor_parallel_size=wrong_tp)))


def server_args(tmp_path, *extra):
    model = tmp_path / "model"
    (model / "experts_vq_tp2_zn").mkdir(parents=True)
    library = tmp_path / "candidate.so"
    library.write_bytes(b"CPU command fixture only")
    return server.parse_args(["--model", str(model), "--library", str(library), "--tensor-parallel-size", "2", *extra])


def test_tp2_server_selects_two_devices_artifact_and_mp(tmp_path, monkeypatch):
    args = server_args(tmp_path, "--physical-npus", "3,5")
    command = server.build_command(args)
    argument = lambda name: command[command.index(name) + 1]
    assert argument("--tensor-parallel-size") == "2" and argument("--distributed-executor-backend") == "mp"
    assert json.loads(argument("--hf-overrides"))["architectures"] == ["VQ2A8TP2OfflineForCausalLM"]
    assert json.loads(argument("--additional-config"))["vq2a8_offline"]["artifact"].endswith("experts_vq_tp2_zn")
    monkeypatch.setenv("WORLD_SIZE", "8")
    environment = server.server_environment(args)
    assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "3,5" and "WORLD_SIZE" not in environment
    assert environment["VLLM_ENABLE_V1_MULTIPROCESSING"] == "1"


def test_tp2_server_explicit_artifact_and_default_device_pair(tmp_path):
    artifact = tmp_path / "separate artifact"
    artifact.mkdir()
    args = server_args(tmp_path, "--artifact", str(artifact))
    command = server.build_command(args)
    additional = json.loads(command[command.index("--additional-config") + 1])
    assert additional["vq2a8_offline"]["artifact"] == str(artifact.resolve())
    assert server.server_environment(args)["ASCEND_RT_VISIBLE_DEVICES"] == "0,1"


@pytest.mark.parametrize(
    "extra",
    [
        ("--physical-npus", "0"),
        ("--physical-npus", "0,0"),
        ("--physical-npus", "-1,0"),
        ("--physical-npus", "a,1"),
        ("--physical-npu", "2"),
        ("--decode-graph", "moe"),
    ],
)
def test_tp2_server_rejects_ambiguous_devices_and_graph(tmp_path, extra):
    with pytest.raises(SystemExit):
        server_args(tmp_path, *extra)
