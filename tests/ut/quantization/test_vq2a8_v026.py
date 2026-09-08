# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""0.26 migration contracts, without claiming NPU or serving validation."""

import ast
import copy
import json
import subprocess
import sys
from itertools import islice
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from packaging.requirements import Requirement

from tools import validate_vq2a8_v026_environment as environment
from tools.validate_vq2a8_v026_environment import V026_REQUIREMENTS, stack_errors

REPO = Path(__file__).resolve().parents[3]


def packages():
    return {
        "vllm": "0.26.0+empty",
        "vllm-ascend": "0.26.0rc2.dev1+g12345678",
        "torch": "2.10.0+cpu",
        "torch-npu": "2.10.0.post4",
        "transformers": "5.14.1",
        "triton-ascend": "3.2.2",
        "fastapi": "0.136.3",
    }


def test_source_empty_runtime_and_custom_ascend_version_are_accepted():
    assert not stack_errors(packages(), (3, 11, 10), "Linux")


def test_acceptance_retains_migration_environment_record():
    from tools.validate_vq2a8_tp1_acceptance import _RESULT_PREFIXES

    assert "MODEL_V026_ENVIRONMENT " in _RESULT_PREFIXES


@pytest.mark.parametrize(
    "name,bad",
    [
        ("vllm", "0.23.0+empty"),
        ("vllm", "0.26.1"),
        ("vllm", "0.26.0.dev1"),
        ("vllm-ascend", "0.23.1.dev78+gf9f49e316"),
        ("vllm-ascend", "0.26.0rc0"),
        ("vllm-ascend", None),
        ("torch", "2.11.0"),
        ("torch-npu", "2.10.0"),
        ("transformers", "5.5.4"),
        ("triton-ascend", "3.2.2.dev20260729205041"),
        ("fastapi", "0.123.0"),
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
    monkeypatch.setattr(sys, "argv", ["validate_vq2a8_v026_environment.py"])
    monkeypatch.setattr(environment, "environment_report", lambda: {"errors": []})

    def failed_check(*args, **kwargs):
        raise failure

    def unexpected_import():
        pytest.fail("Runtime imports must not run after pip check fails.")

    monkeypatch.setattr(environment.subprocess, "run", failed_check)
    monkeypatch.setattr(environment, "check_runtime_imports", unexpected_import)
    assert environment.main() == 1
    report = json.loads(capsys.readouterr().out.split("VQ2A8_V026_ENVIRONMENT ", 1)[1])
    assert report["status"] == "failed"
    assert "pip check could not complete" in report["errors"][0]


def test_runtime_and_build_pins_match_v026_compatibility_gate():
    requirements = {}
    for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            req = Requirement(line)
            requirements[req.name] = req.specifier
    build_text = (REPO / "pyproject.toml").read_text(encoding="utf-8").split("build-backend", 1)[0]
    for name in ("torch", "torch-npu", "transformers", "triton-ascend", "fastapi"):
        assert str(requirements[name]) == str(Requirement(name + V026_REQUIREMENTS[name]).specifier)
        assert f'"{name}{V026_REQUIREMENTS[name]}"' in build_text


@pytest.mark.parametrize(
    "script",
    [
        "build_vq2a8_ascendc.py",
        "profile_vq2a8_ascendc.py",
        "validate_vq2a8_ascendc.py",
        "validate_vq2a8_ascendc_suite.py",
        "validate_vq2a8_tp1_acceptance.py",
        "validate_vq2a8_tp1_offline.py",
        "validate_vq2a8_v026_environment.py",
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


def test_offline_registration_does_not_remove_new_upstream_models():
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
    assert {"VQ2A8TP1OfflineForCausalLM", "KimiK3ForCausalLM", "DeepseekV4ForCausalLM"} <= names
    assert dict(calls)["DeepseekV4ForCausalLM"] == "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM"


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_v026_forward_keeps_auxiliary_collection_and_real_hash_ids(offline, auxiliary):
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
        aux_hidden_state_layers=(1, 3) if auxiliary else (),
    )
    scope = dict(
        self=owner,
        islice=islice,
        hidden_states=hidden,
        input_ids=ids,
        positions=torch.arange(2),
        residual=None,
        llama_4_scaling=None,
        aux_hidden_states=[],
    )
    exec(compile(ast.Module(body=[copy.deepcopy(loop)], type_ignores=[]), str(path), "exec"), scope)
    assert len(calls) == 3
    if offline:
        assert all(call.get("input_ids") is ids for call in calls)
    else:
        assert all(not call for call in calls)
    torch.testing.assert_close(scope["hidden_states"], hidden + 3, rtol=0, atol=0)
    values = scope["aux_hidden_states"]
    assert len(values) == (2 if auxiliary else 0)
    if auxiliary:
        torch.testing.assert_close(values[0], torch.ones(2, 8), rtol=0, atol=0)
        torch.testing.assert_close(values[1], torch.full((2, 8), 3.0), rtol=0, atol=0)
