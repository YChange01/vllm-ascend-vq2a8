# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Official 0.23 migration contracts; no NPU or serving validation claim."""

import ast
import copy
import functools
import hashlib
import json
import subprocess
import sys
from itertools import islice
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

from tools import validate_vq2a8_v023_environment as environment
from tools.validate_vq2a8_v023_environment import V023_REQUIREMENTS, stack_errors

REPO = Path(__file__).resolve().parents[3]


def packages():
    return {
        "vllm": "0.23.0+empty",
        "vllm-ascend": "0.23.0+vq2a8",
        "torch": "2.10.0+cpu",
        "torch-npu": "2.10.0.post4",
        "transformers": "5.5.4",
        "triton-ascend": "3.2.2",
        "fastapi": "0.123.0",
    }


def test_source_empty_runtime_and_custom_ascend_version_are_accepted():
    assert not stack_errors(packages(), (3, 11, 10), "Linux")


def test_acceptance_retains_migration_environment_record():
    from tools.validate_vq2a8_tp1_acceptance import _RESULT_PREFIXES

    assert "MODEL_V023_ENVIRONMENT " in _RESULT_PREFIXES
    assert "MODEL_V023_SCHEDULER_PREFLIGHT " in _RESULT_PREFIXES


@pytest.mark.parametrize(
    "name,bad",
    [
        ("vllm", "0.26.0+empty"),
        ("vllm", "0.23.1"),
        ("vllm", "0.23.0.dev1"),
        ("vllm-ascend", "0.23.1.dev78+gf9f49e316"),
        ("vllm-ascend", "0.23.0rc1"),
        ("vllm-ascend", "0.26.0rc1"),
        ("vllm-ascend", None),
        ("torch", "2.11.0"),
        ("torch-npu", "2.10.0"),
        ("transformers", "5.14.1"),
        ("triton-ascend", "3.2.2.dev20260729205041"),
        ("fastapi", "0.114.0"),
        ("fastapi", "0.124.0"),
        ("fastapi", "0.137.0"),
    ],
)
def test_mixed_stacks_fail_before_model_loading(name, bad):
    values = packages()
    values[name] = bad
    assert any(name in error for error in stack_errors(values, (3, 11, 10), "Linux"))


@pytest.mark.parametrize("python,system", [((3, 9), "Linux"), ((3, 13), "Linux"), ((3, 11), "Windows")])
def test_unsupported_npu_host_is_not_certified(python, system):
    assert stack_errors(packages(), python, system)


@pytest.mark.parametrize("failure", [OSError("pip unavailable"), subprocess.TimeoutExpired("pip check", 120)])
def test_pip_check_errors_preserve_report_and_skip_runtime_imports(monkeypatch, capsys, failure):
    monkeypatch.setattr(sys, "argv", ["validate_vq2a8_v023_environment.py"])
    monkeypatch.setattr(environment, "environment_report", lambda: {"errors": []})

    def failed_check(*args, **kwargs):
        raise failure

    def unexpected_import():
        pytest.fail("Runtime imports must not run after pip check fails.")

    monkeypatch.setattr(environment.subprocess, "run", failed_check)
    monkeypatch.setattr(environment, "check_runtime_imports", unexpected_import)
    assert environment.main() == 1
    report = json.loads(capsys.readouterr().out.split("VQ2A8_V023_ENVIRONMENT ", 1)[1])
    assert report["status"] == "failed"
    assert "pip check could not complete" in report["errors"][0]


def test_runtime_and_build_pins_match_v023_compatibility_gate():
    requirements = {}
    for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            req = Requirement(line)
            requirements[req.name] = req.specifier
    build_text = (REPO / "pyproject.toml").read_text(encoding="utf-8").split("build-backend", 1)[0]
    for name in ("torch", "torch-npu", "transformers", "triton-ascend"):
        assert str(requirements[name]) == str(Requirement(name + V023_REQUIREMENTS[name]).specifier)
        assert f'"{name}{V023_REQUIREMENTS[name]}"' in build_text
    # vLLM 0.23 supplies the minimum while Ascend's official release caps it.
    # Keep Ascend's own pins unchanged; the combined environment gate must use
    # the intersection, not copy the incompatible 0.26 FastAPI range.
    assert str(requirements["fastapi"]) == "<0.124.0"
    assert '"fastapi<0.124.0"' in build_text
    combined = requirements["fastapi"] & Requirement("fastapi>=0.115.0").specifier
    assert str(combined) == str(Requirement("fastapi" + V023_REQUIREMENTS["fastapi"]).specifier)


@pytest.mark.parametrize(
    "script",
    [
        "build_vq2a8_ascendc.py",
        "profile_vq2a8_ascendc.py",
        "validate_vq2a8_ascendc.py",
        "validate_vq2a8_ascendc_suite.py",
        "validate_vq2a8_tp1_acceptance.py",
        "validate_vq2a8_tp1_offline.py",
        "validate_vq2a8_v023_environment.py",
        "serve_vq2a8_v3.py",
        "run_vq2a8_cpu_tests.py",
    ],
)
def test_direct_cli_does_not_shadow_stdlib_bisect(script, tmp_path):
    # No model, vLLM or torch_npu import is needed for these help commands.
    result = subprocess.run(
        [sys.executable, str(REPO / "tools" / script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout


def test_offline_registration_preserves_all_official_v023_models():
    path = REPO / "vllm_ascend/models/__init__.py"
    calls = []
    registry = NS(register_model=lambda *args: calls.append(args))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom)]
    scope = {"ModelRegistry": registry}
    exec(compile(tree, str(path), "exec"), scope)
    scope["register_model"]()
    names = {name for name, _ in calls}
    assert len(names) == len(calls)
    original = {
        "DeepseekV4ForCausalLM": "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM",
        "DeepSeekV4MTPModel": "vllm_ascend.models.deepseek_v4_mtp:DeepSeekV4MTP",
        "LlamaForCausalLMVwnEagle3": "vllm_ascend.models.llama_eagle3_vwn:Eagle3VwnLlamaForCausalLM",
    }
    assert names == {*original, "VQ2A8TP1OfflineForCausalLM", "VQ2A8TP2OfflineForCausalLM"}
    assert all(dict(calls)[name] == value for name, value in original.items())


@pytest.mark.parametrize("offline", [False, True])
def test_v023_forward_keeps_original_layer_loop_and_real_hash_ids(offline):
    """Execute the actual migrated layer loop, not a copied forward algorithm."""
    path = REPO / "vllm_ascend/models/deepseek_v4.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV4Model")
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    loop = next(node for node in forward.body if isinstance(node, ast.For))
    calls = []

    class Layer:
        def __init__(self, index):
            self.layer_idx = index

        def __call__(self, positions, hidden_states, residual, scaling, **kwargs):
            calls.append(kwargs)
            return hidden_states + 1, residual

    hidden = torch.zeros(2, 4, 8)
    ids = torch.tensor([7, 11])
    owner = NS(
        requires_moe_input_ids=offline,
        layers=[Layer(0), Layer(1), Layer(2)],
        start_layer=0,
        end_layer=3,
    )
    scope = dict(
        self=owner,
        islice=islice,
        hidden_states=hidden,
        input_ids=ids,
        positions=torch.arange(2),
        residual=None,
        llama_4_scaling=None,
    )
    exec(compile(ast.Module(body=[copy.deepcopy(loop)], type_ignores=[]), str(path), "exec"), scope)
    assert len(calls) == 3
    if offline:
        assert all(call.get("input_ids") is ids for call in calls)
    else:
        assert all(not call for call in calls)
    torch.testing.assert_close(scope["hidden_states"], hidden + 3, rtol=0, atol=0)
    # v0.23 keeps the MTP residual buffer after this loop rather than v0.26's
    # auxiliary-hidden-state collection. The unrelated upstream path must remain.
    copies = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "copy_"
        and "_mtp_hidden_buffer" in ast.unparse(node.func.value)
    ]
    assert len(copies) == 2


def _load_runner_layer_kv_specs_method():
    """Execute the real resolver without importing the NPU worker on a CPU host."""
    path = REPO / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method = next(
        node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "_get_layer_kv_cache_specs"
    )

    class AttentionLayerBase:
        pass

    class UniformTypeKVCacheSpecs(NS):
        pass

    class AscendSFAIndexerCacheSpec(NS):
        pass

    scope = {
        "AttentionLayerBase": AttentionLayerBase,
        "UniformTypeKVCacheSpecs": UniformTypeKVCacheSpecs,
        "AscendSFAIndexerCacheSpec": AscendSFAIndexerCacheSpec,
    }
    module = ast.parse("from __future__ import annotations\n")
    module.body.append(copy.deepcopy(method))
    exec(compile(module, str(path), "exec"), scope)
    return scope


@pytest.mark.parametrize("model_block_size,state_block_size", [(32, 2), (64, 4), (128, 8)])
@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("context_on_runner", [False, True])
def test_compressed_kv_specs_survive_scheduler_block_size_change(
    model_block_size, state_block_size, uniform, context_on_runner
):
    scope = _load_runner_layer_kv_specs_method()
    calls = []

    class DSV4Cache(scope["AttentionLayerBase"]):
        def get_kv_cache_spec(self, config):
            calls.append(config.cache_config.block_size)
            # DSV4 supports model block sizes 32/64/128, not the scheduler's
            # minimum compressor block size. This reproduces the reported 8.
            return {32: None, 64: None, 128: None}[config.cache_config.block_size]

    planned = {
        "indexer": NS(block_size=model_block_size, head_size=128, dtype=torch.float8_e4m3fn, scale_dim=1),
        "swa": NS(block_size=model_block_size, head_size=640, dtype=torch.float8_e4m3fn),
        "state_c4": NS(block_size=state_block_size, head_size=512, page_size_padded=132 * model_block_size),
        "state_c128": NS(block_size=2 * state_block_size, head_size=1024, page_size_padded=640 * model_block_size),
    }
    before = copy.deepcopy(planned)
    groups = [
        NS(
            layer_names=[name],
            kv_cache_spec=scope["UniformTypeKVCacheSpecs"](kv_cache_specs={name: spec}) if uniform else spec,
        )
        for name, spec in planned.items()
    ]
    context = NS(static_forward_context={name: DSV4Cache() for name in planned})
    config = NS(cache_config=NS(block_size=model_block_size), compilation_config=context)
    owner = NS(vllm_config=config, use_compress=True)
    if context_on_runner:
        owner.compilation_config = context
    # EngineCore updates the shared config before allocation in the in-process
    # offline engine. Do not reset this scheduler value back to the model size.
    config.cache_config.block_size = min(spec.block_size for spec in planned.values())
    resolve = scope["_get_layer_kv_cache_specs"]
    for _ in range(2):  # Both raw allocation and reshape resolve layer specs.
        actual = resolve(owner, NS(kv_cache_groups=groups))
        assert actual.keys() == planned.keys()
        assert all(actual[name] is spec for name, spec in planned.items())
    assert planned == before
    assert config.cache_config.block_size == state_block_size
    assert calls == []


@pytest.mark.parametrize("use_compress", [False, None])
@pytest.mark.parametrize("uniform", [False, True])
def test_noncompressed_kv_specs_keep_official_v023_planned_objects(use_compress, uniform):
    scope = _load_runner_layer_kv_specs_method()
    planned = {"indexer": NS(block_size=16), "attention": NS(block_size=16)}
    restored = scope["AscendSFAIndexerCacheSpec"](block_size=16, scale_dim=1, cache_sparse_li_c8=True)
    calls = []

    class Layer(scope["AttentionLayerBase"]):
        def __init__(self, spec):
            self.spec = spec

        def get_kv_cache_spec(self, config):
            calls.append(config.cache_config.block_size)
            return self.spec

    owner = NS(
        vllm_config=NS(cache_config=NS(block_size=16)),
        compilation_config=NS(static_forward_context={"indexer": Layer(restored), "attention": Layer(None)}),
    )
    if use_compress is not None:
        owner.use_compress = use_compress
    groups = [
        NS(
            layer_names=[name],
            kv_cache_spec=scope["UniformTypeKVCacheSpecs"](kv_cache_specs={name: spec}) if uniform else spec,
        )
        for name, spec in planned.items()
    ]
    actual = scope["_get_layer_kv_cache_specs"](owner, NS(kv_cache_groups=groups))
    # Official 0.23 reads the scheduled per-layer specs directly for every
    # backend. Do not transplant 0.26's later SFA restoration path here.
    assert actual["indexer"] is planned["indexer"]
    assert actual["attention"] is planned["attention"]
    assert calls == []


def _load_version_gate(monkeypatch, runtime_version, override=None):
    """Run the real version helper without importing the NPU utility module."""
    import vllm

    monkeypatch.setattr(vllm, "__version__", runtime_version, raising=False)
    path = REPO / "vllm_ascend/utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "vllm_version_is")
    scope = dict(
        functools=functools, Version=Version, InvalidVersion=InvalidVersion, envs_ascend=NS(VLLM_VERSION=override)
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope["vllm_version_is"]


@pytest.mark.parametrize(
    "actual,expected",
    [
        ("0.23.0", True),
        ("0.23", True),
        ("0.23.0+empty", True),
        ("0.23.0+empty.vq2a8text1", True),
        ("0.23.0+cu130", True),
        ("0.23.0rc1+empty", False),
        ("0.23.0.dev1+empty", False),
        ("0.23.0.post1+empty", False),
        ("0.23.1+empty", False),
        ("0.26.0+empty", False),
    ],
)
@pytest.mark.parametrize("explicit_override", [False, True])
def test_version_gate_ignores_only_local_build_metadata(monkeypatch, actual, expected, explicit_override):
    gate = _load_version_gate(
        monkeypatch, "0.26.0" if explicit_override else actual, actual if explicit_override else None
    )
    assert gate("0.23.0") is expected
    assert gate("0.23.0") is expected
    assert gate.cache_info().hits == 1


def test_version_gate_override_and_invalid_versions_are_preserved(monkeypatch):
    assert not _load_version_gate(monkeypatch, "0.23.0+empty", "0.26.0")("0.23.0")
    for runtime, override in [("not-a-version", None), ("0.23.0", "not-a-version")]:
        with pytest.raises(ValueError, match="Invalid vllm version"):
            _load_version_gate(monkeypatch, runtime, override)("0.23.0")


def _load_structured_output_patch():
    class Manager:
        def should_advance(self, request):
            return False

        def grammar_init(self, request):
            pass

        def grammar_bitmask(self, *args):
            pass

    path = REPO / "vllm_ascend/patch/platform/patch_structured_output.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module = ast.parse("from __future__ import annotations\n")
    module.body.extend(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        or isinstance(node, ast.FunctionDef)
        and node.name == "_patch_structured_output_manager"
    )
    scope = dict(StructuredOutputManager=Manager)
    exec(compile(module, str(path), "exec"), scope)
    scope["_patch_structured_output_manager"]()
    return Manager


def test_v023_scheduler_keeps_original_request_only_advance_interface():
    manager_type = _load_structured_output_patch()
    manager = manager_type()
    request = NS(use_structured_output=False)
    # Evaluate each real scheduler's call, including its keyword arguments.
    path = REPO / "vllm_ascend/core/recompute_scheduler.py"
    calls = [
        node
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "should_advance"
    ]
    assert calls
    for call in calls:
        assert not call.keywords
        scope = dict(self=NS(structured_output_manager=manager), request=request)
        assert eval(compile(ast.Expression(call), str(path), "eval"), scope) is False
    assert not hasattr(manager, "trim_reasoning_for_advance")
    assert not (REPO / "vllm_ascend/patch/platform/patch_kv_delivery_preemption.py").exists()


def test_scheduler_preflight_rejects_incompatible_manager_before_generation():
    class IncompatibleManager:
        def should_advance(self, request, *, required_extra):
            return False

    with pytest.raises(RuntimeError, match="scheduler/structured-output API mismatch"):
        environment._check_structured_output_manager(IncompatibleManager)


def test_scheduler_preflight_accepts_v023_without_v026_reasoning_backports():
    manager_type = _load_structured_output_patch()
    report = environment._check_structured_output_manager(manager_type)
    assert report["plain_request_checked"] is True
    assert report["npu_kernel_execution"] is False
    assert "new_token_ids" not in report["should_advance"]
    assert not hasattr(manager_type, "trim_reasoning_for_advance")


def test_offline_scheduler_preflight_runs_before_device_and_model_loading():
    path = REPO / "tools/validate_vq2a8_tp1_offline.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = {
        node.func.id: node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert calls["check_scheduler_apis"] < calls["_initialize_device"] < calls["LLM"]


@pytest.mark.parametrize(
    "relative,expected_blob",
    [
        ("vllm_ascend/core/recompute_scheduler.py", "85b2590e98b248700baeb5d4cc5e9954a54f486f"),
        ("vllm_ascend/patch/platform/patch_structured_output.py", "d69af3f620028751816a7c5e8a96913a982cd739"),
        ("vllm_ascend/worker/model_runner_v1.py", "70ef1d79a8d52d79b9d808f16257d39951a3d48b"),
    ],
)
def test_migration_retains_official_v023_framework_files(relative, expected_blob):
    # Git blob IDs from official vllm-ascend v0.23.0, not from the VQ2 fork.
    # Normalize worktree line endings so source archives and Windows run alike.
    # Deliberate future upstream changes must update the documented baseline.
    payload = (REPO / relative).read_bytes().replace(b"\r\n", b"\n")
    blob = b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    assert hashlib.sha1(blob, usedforsecurity=False).hexdigest() == expected_blob


def test_v023_tools_do_not_import_or_require_v026_stack():
    for path in sorted((REPO / "tools").glob("*vq2a8*.py")):
        source = path.read_text(encoding="utf-8")
        assert "validate_vq2a8_v026_environment" not in source, path
        assert "require_v026_stack" not in source, path
    assert not (REPO / "tools/validate_vq2a8_v026_environment.py").exists()


def test_cpu_harness_retains_python_body_but_rejects_every_kernel_launch():
    from tools.run_vq2a8_cpu_tests import CpuContractKernel

    def kernel(*args):
        pytest.fail("The CPU wrapper must not invoke the kernel")

    wrapped = CpuContractKernel(kernel)
    assert wrapped.fn is kernel
    with pytest.raises(RuntimeError, match="cannot compile or launch kernels"):
        wrapped[(1,)]()


def test_cpu_harness_rejects_missing_tests_before_runtime_imports(tmp_path):
    source = """
import importlib.abc, runpy, sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.')
               for name in ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'pytest')):
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-X",
            "utf8",
            "-c",
            source,
            str(REPO / "tools/run_vq2a8_cpu_tests.py"),
            "--repo",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "No VQ2A8 CPU contract tests" in result.stderr
    assert "Unexpected runtime import" not in result.stderr
    assert not list(tmp_path.iterdir())
