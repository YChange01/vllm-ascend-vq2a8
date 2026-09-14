#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run VQ2 CPU contracts: python -X utf8 tools/run_vq2a8_cpu_tests.py [-k EXPR].

No real vLLM imports, NPU initialization, kernel compilation or serving proof.
Namespace packages isolate adapters from engine initialization. Triton wrappers
retain Python bodies for call-contract tests but reject every kernel launch.
Tensor operations use real PyTorch; numeric assertions are never replaced.
"""

from __future__ import annotations

# Do not put tools/bisect ahead of the stdlib bisect module.
# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import importlib.machinery
import types
from pathlib import Path


def package(name, path=None):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=path is not None)
    if path is not None:
        module.__path__ = [str(path)]
        module.__spec__.submodule_search_locations = module.__path__
    sys.modules[name] = module
    if "." in name and name.rsplit(".", 1)[0] in sys.modules:
        parent, child = name.rsplit(".", 1)
        setattr(sys.modules[parent], child, module)
    return module


class CpuContractKernel:
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        raise RuntimeError("CPU contract harness cannot compile or launch kernels")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args, test_args = parser.parse_known_args(argv)
    repo = args.repo.resolve()
    tests = sorted((repo / "tests/ut/quantization").glob("test_vq2a8_*.py"))
    if not tests:
        parser.error(f"No VQ2A8 CPU contract tests found under {repo}.")

    # This is an isolated process, not an initializer for an existing engine.
    # Keep native-only tests skipped even on hosts with accelerator packages.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONUTF8"] = "1"
    sys.modules["torch_npu"] = None
    sys.path.insert(0, str(repo))
    package("vllm_ascend", repo / "vllm_ascend")
    package("vllm_ascend.quantization", repo / "vllm_ascend/quantization")
    package("vllm", repo / "_no_vllm_import")
    shim = package("vllm.triton_utils")
    shim.triton = types.SimpleNamespace(
        jit=CpuContractKernel,
        cdiv=lambda a, b: (a + b - 1) // b,
        next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
    )
    shim.tl = types.SimpleNamespace(constexpr=int, tensor=object)

    import pytest
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    print(f"CPU_CONTRACT_ONLY torch={torch.__version__} python={sys.version}", flush=True)
    return pytest.main(["--noconftest", "-o", "addopts=", "--tb=short", "-q", *map(str, tests), *test_args])


if __name__ == "__main__":
    raise SystemExit(main())
