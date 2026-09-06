# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from safetensors.torch import save_file

from tools import validate_vq2a8_tp1_acceptance as acceptance
from tools import validate_vq2a8_tp1_offline as gate
from vllm_ascend.quantization.vq2a8_offline import (
    OfflineMoEOwner,
    audit_offline_root,
    canonical_root_parameter,
    offline_engine_options,
    validate_offline_config,
    validate_offline_evidence,
)


def config():
    return NS(
        additional_config={"vq2a8_offline": {"enabled": True, "artifact": "/artifact"}},
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        model_config=NS(enforce_eager=True, quantization=None, dtype=torch.bfloat16, max_model_len=32),
        quant_config=None,
        scheduler_config=NS(max_num_seqs=1, max_num_batched_tokens=32),
        compilation_config=NS(mode=0, cudagraph_mode=0),
        cache_config=NS(gpu_memory_utilization=0.9),
        load_config=NS(load_format="safetensors"),
    )


def test_plan_preserves_model_type_and_selects_only_explicit_offline_architecture():
    plan = offline_engine_options(Path("/model"), Path("/artifact"))
    assert plan["hf_overrides"] == {"architectures": ["VQ2A8TP1OfflineForCausalLM"], "quantization_config": None}
    assert plan["load_format"] == "safetensors" and plan["enforce_eager"]
    assert plan["distributed_executor_backend"] == "uni"
    assert plan["max_num_seqs"] == 1 and plan["max_model_len"] == plan["max_num_batched_tokens"] == 32
    assert not plan["enable_prefix_caching"] and not plan["async_scheduling"]
    assert validate_offline_config(config())["enabled"] is True


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("parallel_config", "tensor_parallel_size", 4),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "data_parallel_size", 2),
        ("parallel_config", "prefill_context_parallel_size", 2),
        ("parallel_config", "decode_context_parallel_size", 2),
        ("parallel_config", "enable_expert_parallel", True),
        ("parallel_config", "enable_eplb", True),
        ("model_config", "quantization", "vq2a8"),
        ("model_config", "enforce_eager", False),
        ("model_config", "dtype", torch.float16),
        ("model_config", "max_model_len", 129),
        ("scheduler_config", "max_num_batched_tokens", 129),
        ("scheduler_config", "max_num_seqs", 2),
        ("scheduler_config", "async_scheduling", True),
        ("cache_config", "enable_prefix_caching", True),
        ("cache_config", "cpu_offload_gb", 1),
        ("load_config", "load_format", "dummy"),
        ("compilation_config", "mode", 3),
        ("compilation_config", "cudagraph_mode", 1),
    ],
)
def test_offline_config_rejects_unsupported_execution(section, key, value):
    cfg = config()
    setattr(getattr(cfg, section), key, value)
    with pytest.raises(ValueError):
        validate_offline_config(cfg)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"enabled": False, "artifact": "/a"},
        {"enabled": True, "artifact": "/a", "typo": 1},
        {"enabled": True, "artifact": "/a", "cache_experts": True},
        {"enabled": True, "artifact": "/a", "token_chunk": 0},
    ],
)
def test_offline_config_requires_explicit_bounded_options(options):
    cfg = config()
    cfg.additional_config["vq2a8_offline"] = options
    with pytest.raises(ValueError):
        validate_offline_config(cfg)


@pytest.mark.parametrize(
    "source,target",
    [
        ("head.weight", "lm_head.weight"),
        ("embed.weight", "model.embed_tokens.weight"),
        ("layers.2.attn.indexer.compressor.norm.weight", "model.layers.2.self_attn.indexer.compressor.norm.weight"),
        ("layers.3.attn_norm.weight", "model.layers.3.input_layernorm.weight"),
        ("layers.3.ffn_norm.weight", "model.layers.3.post_attention_layernorm.weight"),
        ("hc_head_fn", "model.hc_head_fn"),
        ("layers.0.ffn.gate.tid2eid", None),
        ("mtp.0.head.weight", None),
    ],
)
def test_canonical_root_names(source, target):
    assert canonical_root_parameter(source) == target


def owner_for_load():
    owner = OfflineMoEOwner.__new__(OfflineMoEOwner)
    owner.layers = {0: NS(root={"gate.weight": torch.ones(1)})}
    owner.inventory = {name: {} for name in ("head.weight", "layers.0.ffn.gate.weight")}
    return owner


def copy_weight(param, value):
    param.data.copy_(value)


def test_strict_root_load_accounts_for_registered_delegated_and_mtp():
    owner = owner_for_load()
    param = torch.zeros(2, 3, dtype=torch.bfloat16)
    loaded, report = owner.load_root(
        {"lm_head.weight": param},
        [
            ("head.weight", torch.ones_like(param)),
            ("layers.0.ffn.gate.weight", torch.ones(1)),
            ("mtp.0.head.weight", torch.ones(1)),
        ],
        copy_weight,
    )
    assert loaded == {"lm_head.weight"} and torch.equal(param, torch.ones_like(param))
    assert report["moe_root_tensors_loaded"] == 1 and report["mtp_tensors_skipped"] == 1


@pytest.mark.parametrize(
    "failure", ["duplicate", "missing_root", "missing_moe", "unknown", "shape", "dtype", "nan", "unloaded_parameter"]
)
def test_strict_root_load_never_silently_skips_bad_weights(failure):
    owner = owner_for_load()
    parameters = {"lm_head.weight": torch.zeros(2, 3, dtype=torch.bfloat16)}
    weights = [("head.weight", torch.ones(2, 3, dtype=torch.bfloat16)), ("layers.0.ffn.gate.weight", torch.ones(1))]
    if failure == "duplicate":
        weights.append(weights[0])
    elif failure == "missing_root":
        weights.pop(0)
    elif failure == "missing_moe":
        weights.pop()
    elif failure == "unknown":
        weights.append(("unexpected", torch.ones(1)))
    elif failure == "shape":
        weights[0] = ("head.weight", torch.ones(3, 2, dtype=torch.bfloat16))
    elif failure == "dtype":
        weights[0] = ("head.weight", torch.ones(2, 3, dtype=torch.float32))
    elif failure == "nan":
        weights[0][1][0, 0] = float("nan")
    elif failure == "unloaded_parameter":
        parameters["extra"] = torch.zeros(1)
    with pytest.raises(ValueError):
        owner.load_root(parameters, weights, copy_weight)


def test_a5_compressor_norm_exact_widening_is_explicit_and_recorded():
    owner = owner_for_load()
    name = "layers.2.attn.indexer.compressor.norm.weight"
    owner.inventory = {name: {}}
    owner.layers = {}
    target = torch.zeros(4, dtype=torch.float32)
    value = torch.tensor([1, 2, 3, 4], dtype=torch.bfloat16)
    _, report = owner.load_root({canonical_root_parameter(name): target}, [(name, value)], copy_weight)
    assert torch.equal(value.float(), target)
    assert report["a5_compressor_norm_bf16_to_fp32"] == [name]


def test_root_header_audit_refuses_fp8_and_duplicate_tensors(tmp_path):
    save_file({"head.weight": torch.ones(4, dtype=torch.bfloat16)}, str(tmp_path / "a.safetensors"))
    assert audit_offline_root(tmp_path)["head.weight"]["dtype"] == "BF16"
    save_file({"head.weight": torch.ones(4)}, str(tmp_path / "b.safetensors"))
    with pytest.raises(ValueError, match="Duplicate"):
        audit_offline_root(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    save_file({"head.weight": torch.ones(4).to(torch.float8_e4m3fn)}, str(other / "a.safetensors"))
    with pytest.raises(ValueError, match="Unsupported root dtype"):
        audit_offline_root(other)


def evidence():
    logits = torch.zeros(4, 8)
    logits[:, 7] = 2
    return {
        "load": {"moe_layers": 2, "registered_parameters_loaded": 4},
        "steps": [
            {"tokens": 3, "positions": [0, 1, 2]},
            {"tokens": 1, "positions": [3]},
            {"tokens": 1, "positions": [4]},
            {"tokens": 1, "positions": [5]},
        ],
        "cache": {"layer_calls": {0: 4, 1: 4}, "resident_experts": 4, "per_layer_cache_limit": 2},
        "logits": logits,
        "peak_allocated_bytes": 1024,
        "peak_reserved_bytes": 2048,
    }


def test_model_evidence_requires_all_layers_real_steps_and_greedy_logits():
    result = validate_offline_evidence(evidence(), [0, 1, 2], [7] * 4, 2, 8)
    assert result["finite_logits"] and result["greedy_logits_agree"]
    assert result["decode_steps"] == 3 and result["layers_executed"] == 2


@pytest.mark.parametrize(
    "failure",
    ["missing_layer", "missing_step", "wrong_position", "cache_overflow", "nan", "shape", "not_loaded", "sampler"],
)
def test_bad_model_evidence_cannot_pass(failure):
    data = evidence()
    if failure == "missing_layer":
        del data["cache"]["layer_calls"][1]
    elif failure == "missing_step":
        data["steps"].pop()
    elif failure == "wrong_position":
        data["steps"][2]["positions"] = [3]
    elif failure == "cache_overflow":
        data["cache"]["resident_experts"] = 5
    elif failure == "nan":
        data["logits"][1, 0] = float("nan")
    elif failure == "shape":
        data["logits"] = data["logits"][:1]
    elif failure == "not_loaded":
        data["load"]["registered_parameters_loaded"] = 0
    elif failure == "sampler":
        data["logits"][1, 0] = 3
    with pytest.raises(ValueError):
        validate_offline_evidence(data, [0, 1, 2], [7] * 4, 2, 8)


def test_model_supervisor_launch_plan_and_compact_failure_safe_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "acceptance",
            "--stage",
            "model",
            "--model",
            str(tmp_path),
            "--artifact",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "report"),
        ],
    )
    calls = []

    def child(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 3600
        record = validate_offline_evidence(evidence(), [0, 1, 2], [7] * 4, 2, 8)
        for run in range(2):
            kwargs["stdout"].write("MODEL_RESULT " + json.dumps({**record, "run": run}) + "\n")
        kwargs["stdout"].write(
            'VQ2A8_TP1_OFFLINE_EXECUTION_GATE=PASS {"native_fp8_dot": false, "repeat_exact": true}\n'
        )
        return NS(returncode=0)

    monkeypatch.setattr(acceptance.subprocess, "run", child)
    assert acceptance.main() == 0
    assert len(calls) == 1 and calls[0][1].endswith("validate_vq2a8_tp1_offline.py")
    assert "--allow-partial-artifact" not in calls[0] and "--probes" not in calls[0]
    text = (tmp_path / "report/summary.txt").read_text()
    assert "STAGE=MODEL_OFFLINE_EXECUTION" in text and "runs=2 layers=2 prefill=3 decode=3" in text
    assert "LOGITS_REFERENCE_VERIFIED=False" in text and "SERVING_VERIFIED=False" in text
    assert len(text) < 1200
    log = tmp_path / "report/probe-full_model.log"
    assert not acceptance.summarize_log(log, -6)["passed"]
    log.write_text("VQ2A8_TP1_MOE_GATE=PASS {}\n")
    assert not acceptance.summarize_log(log, 0, expected_pass="VQ2A8_TP1_OFFLINE_EXECUTION_GATE=PASS")["passed"]


@pytest.mark.parametrize("args", [["--allow-partial-artifact"], ["--device", "cuda:0"]])
def test_model_stage_does_not_certify_partial_or_cuda(monkeypatch, tmp_path, args):
    monkeypatch.setattr(sys, "argv", ["acceptance", "--stage", "model", "--model", str(tmp_path), *args])
    with pytest.raises(SystemExit) as error:
        acceptance.main()
    assert error.value.code == 2 and not list(tmp_path.iterdir())


def test_offline_trace_rpc_requires_one_worker():
    with pytest.raises(ValueError):
        gate.single_worker_result([])
    with pytest.raises(ValueError):
        gate.single_worker_result([1, 2])
    assert gate.single_worker_result([1]) == 1


def test_decoder_hook_preserves_default_and_threads_hash_ids_without_importing_npu():
    # Execute the real forward method with tiny CPU stand-ins. This verifies
    # the integration seam, not the proprietary attention/HC implementation.
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v4.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV2DecoderLayer")
    method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"))
    namespace = {"torch": torch, "IntermediateTensors": object}
    # Postponed annotations and injected stand-ins make no global patches.
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    # Test the actual call seam via AST extraction below; surrounding HC is
    # intentionally not emulated as an independent numerical oracle.
    selection = next(
        n for n in ast.walk(method) if isinstance(n, ast.If) and ast.unparse(n.test) == "input_ids is None"
    )
    calls = []
    stub = NS(mlp=lambda *a, **kw: calls.append((a, kw)) or a[0])
    hidden, ids = torch.ones(3, 4), torch.tensor([0, 1, 0])
    scope = {"self": stub, "hidden_states": hidden, "input_ids": None}
    seam = compile(ast.fix_missing_locations(ast.Module(body=[selection], type_ignores=[])), str(path), "exec")
    exec(seam, scope)
    assert calls[-1] == ((hidden,), {})
    scope["input_ids"] = ids
    exec(seam, scope)
    assert calls[-1][1]["input_ids"] is ids
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4Model")
    assert any(isinstance(n, ast.Assign) and ast.unparse(n) == "requires_moe_input_ids = False" for n in model.body)
    assert "input_ids=input_ids" in ast.unparse(model)


def construct_meta_root(hf_config):
    """Execute model/adapter constructors, substituting device primitives only.

    Usable for real checkpoint HEADER comparison without allocating its weights.
    This does not test CANN, attention execution, or vLLM worker compatibility.
    """
    from enum import Enum

    class DeviceType(Enum):
        A5 = 1

    class Linear(torch.nn.Module):
        def __init__(self, input_size, output_size, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(output_size, input_size, dtype=torch.bfloat16, device="meta"))

    class Embedding(Linear):
        def __init__(self, vocab, hidden, **kwargs):
            super().__init__(hidden, vocab)

    class Norm(torch.nn.Module):
        def __init__(self, hidden, eps=None, has_weight=True, dtype=None):
            super().__init__()
            if has_weight:
                self.weight = torch.nn.Parameter(torch.empty(hidden, dtype=dtype or torch.bfloat16, device="meta"))

    class DevicePrimitive(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class Owner:
        def __init__(self, *args):
            self.layers = {}
            self.calls = {}

        def create_layer(self, index):
            self.layers[index] = NS()
            self.calls[index] = 0
            return self.layers[index]

    class TorchProxy:
        npu = NS(current_device=lambda: 0)

        def __getattr__(self, name):
            return getattr(torch, name)

        def device(self, *args):
            return torch.device("meta")

    def make_layers(count, factory, prefix):
        return 0, count, torch.nn.ModuleList([factory(f"{prefix}.{i}") for i in range(count)])

    scope = {
        "torch": TorchProxy(),
        "nn": torch.nn,
        "Path": Path,
        "json": json,
        "AscendDeviceType": DeviceType,
        "get_ascend_device_type": lambda: DeviceType.A5,
        "get_tensor_model_parallel_world_size": lambda: 1,
        "enable_dsa_cp": lambda: False,
        "extract_dsv4_layer_index": lambda cfg, prefix: int(prefix.split(".")[-2]),
        "get_dsv4_compress_ratio": lambda cfg, index: cfg.compress_ratios[index],
        "get_pp_group": lambda: NS(is_first_rank=True, is_last_rank=True),
        "current_platform": NS(device_type="meta"),
        "make_layers": make_layers,
        "maybe_prefix": lambda prefix, name: prefix + "." + name if prefix else name,
        "support_torch_compile": lambda cls: cls,
        "get_ascend_config": NS,
        "OfflineMoEOwner": Owner,
        "validate_offline_config": validate_offline_config,
        "default_weight_loader": copy_weight,
        "DSAModules": NS,
        "_dsv4_block_sizes": lambda: {128: ([128] * 4,)},
    }
    for name in ("ReplicatedLinear", "ColumnParallelLinear", "RowParallelLinear"):
        scope[name] = Linear
    scope.update(RMSNorm=Norm, VocabParallelEmbedding=Embedding, ParallelLMHead=Embedding)
    for name in (
        "ComplexExpRotaryEmbedding",
        "AscendDeepseekV4SWACache",
        "AscendDeepseekV4IndexerCache",
        "AscendCompressorStateCache",
        "AscendDeepseekSparseAttention",
        "LogitsProcessor",
    ):
        scope[name] = DevicePrimitive
    for name in ("SupportsPP", "DeepseekV2MixtureOfExperts", "SupportsLoRA", "SupportsEagle"):
        scope[name] = type(name, (), {})
    root = Path(__file__).resolve().parents[3]
    selected = {
        "Compressor",
        "Indexer",
        "DeepseekV4Attention",
        "DeepseekV2DecoderLayer",
        "DeepseekV4Model",
        "AscendDeepseekV4ForCausalLM",
    }
    for name, selection in (
        ("vllm_ascend/models/deepseek_v4.py", selected),
        ("vllm_ascend/patch/worker/vq2a8_offline_model.py", None),
    ):
        tree = ast.parse((root / name).read_text())
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and (selection is None or n.name in selection)]
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *classes],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), name, "exec"), scope)
    cfg = config()
    cfg.model_config.hf_config = hf_config
    cfg.model_config.model = "/model"
    cfg.cache_config.block_size = 128
    with torch.device("meta"):
        return scope["VQ2A8TP1OfflineForCausalLM"](vllm_config=cfg)


def test_real_constructor_seams_allocate_no_fused_experts_and_preserve_parameter_names():
    hf = NS(
        vocab_size=8,
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        q_lora_rank=8,
        o_lora_rank=8,
        head_dim=8,
        qk_rope_head_dim=4,
        o_groups=1,
        sliding_window=128,
        rms_norm_eps=1e-6,
        index_topk=4,
        index_n_heads=2,
        index_head_dim=8,
        compress_ratios=[0, 4, 128],
        rope_theta=10000,
        compress_rope_theta=10000,
        rope_parameters={"original_max_position_embeddings": 128, "factor": 1, "beta_fast": 32, "beta_slow": 1},
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        n_routed_experts=4,
        n_shared_experts=1,
    )
    model = construct_meta_root(hf)
    params = dict(model.named_parameters())
    assert "model.layers.1.self_attn.compressor.norm.weight" in params
    assert params["model.layers.1.self_attn.compressor.norm.weight"].dtype == torch.float32
    assert params["model.layers.2.self_attn.wq_b.weight"].shape == (16, 8)
    assert not any(".mlp." in name for name in params)
    assert set(model.model.offline_owner.layers) == {0, 1, 2}
    assert len(model.moe_mlp_layers) == 3 and not model.moe_layers
