# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import copy
import hashlib
import importlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import build_vq2a8_ascendc as build
from tools import validate_vq2a8_ascendc as gate

REPO = Path(__file__).resolve().parents[3]


def test_native_host_layout(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Host C++ compiler unavailable; run this test on the development Linux host.")
    executable = tmp_path / "layout"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(REPO),
            str(Path(__file__).with_name("vq2a8_ascendc_layout_test.cpp")),
            "-o",
            str(executable),
        ],
        check=True,
    )
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    assert "ASCENDC_HOST_LAYOUT=PASS" in result.stdout
    assert "DEVICE_EXECUTION_VERIFIED=False" in result.stdout


def test_native_fence_preserves_framework_event_ownership(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Run the host event-model regression on the Linux development host.")
    source = (REPO / "csrc/vq2a8_ascendc/kernel.cpp").read_text()
    fence = re.search(r"template <HardEvent E>\s+__aicore__ inline void Fence\(\) \{.*?\n\}", source, re.S)
    assert fence is not None
    (tmp_path / "fence_under_test.h").write_text(fence.group())
    executable = tmp_path / "sync"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(tmp_path),
            str(Path(__file__).with_name("vq2a8_ascendc_sync_test.cpp")),
            "-o",
            str(executable),
        ],
        check=True,
    )
    result = subprocess.run([str(executable)], capture_output=True, text=True, check=True)
    assert "ASCENDC_HOST_SYNC=PASS DEVICE_EXECUTION_VERIFIED=False" in result.stdout


def good_evidence(stage):
    return {
        "status": "passed",
        "implementation": "ascendc",
        "stage": stage,
        "probe": "0:0",
        "device": {"type": "npu", "soc": 260},
        "library": {"sha256": "abc"},
        "native_instruction_verified": False,
        "on_chip_decode_verified": False,
        "model_integration_verified": False,
        "performance_verified": False,
        "results": [
            {
                "key": key,
                "passed": True,
                "repeat_exact": True,
                "row_chunk_exact": True,
                "oracle": {"allclose": True},
                "accepted_baseline": {"allclose": True},
                "independent_chain": {"allclose": True},
            }
            for key in sorted(gate.expected_keys(stage))
        ],
    }


@pytest.mark.parametrize("stage", ["direct", "bridge", "fused", "expert"])
def test_evidence_is_fail_closed(stage, tmp_path):
    path = tmp_path / "evidence.json"
    original = good_evidence(stage)
    path.write_text(json.dumps(original))
    assert gate.evidence_passed(path, stage, "abc")
    assert not gate.evidence_passed(path, stage, "wrong-binary")
    assert not gate.evidence_passed(path, stage, "abc", "3:255")
    for mutation in (
        {"status": "failed"},
        {"implementation": "triton"},
        {"probe": "3:255"},
        {"device": {"type": "cuda", "soc": 260}},
        {"device": {"type": "npu", "soc": 220}},
        {"results": original["results"][:-1]},
        {"results": original["results"] + original["results"][:1]},
        {"native_instruction_verified": True},
        {"on_chip_decode_verified": True},
        {"performance_verified": True},
        {"model_integration_verified": True},
    ):
        path.write_text(json.dumps({**original, **mutation}))
        assert not gate.evidence_passed(path, stage, "abc")
    fields = ["oracle"] + (["accepted_baseline"] if stage in ("fused", "expert") else [])
    if stage == "expert":
        fields += ["independent_chain"]
    for field in fields:
        changed = copy.deepcopy(original)
        changed["results"][0][field]["allclose"] = False
        path.write_text(json.dumps(changed))
        assert not gate.evidence_passed(path, stage, "abc")


@pytest.mark.parametrize("contents", ["null", "{}", "[]", "not-json"])
def test_malformed_evidence(contents, tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text(contents)
    assert not gate.evidence_passed(path, "direct", "abc")


def test_build_manifest_pins_binary_and_native_sources(tmp_path):
    library = tmp_path / "libvq2a8_ascendc.so"
    library.write_bytes(b"not-a-real-library-host-test-only")
    manifest = {
        "status": "built",
        "source_sha256": build.source_hashes(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }
    path = tmp_path / "build-manifest.json"
    path.write_text(json.dumps(manifest))
    assert gate.library_evidence(library)["sha256"] == manifest["library_sha256"]
    library.write_bytes(b"different")
    with pytest.raises(ValueError, match="hash mismatch"):
        gate.library_evidence(library)
    manifest["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    manifest["source_sha256"]["kernel.cpp"] = "stale"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sources changed"):
        gate.library_evidence(library)


@pytest.mark.parametrize("failure", ["exit", "timeout", "missing", "incomplete"])
def test_supervisor_stops_without_promoting(failure, tmp_path, monkeypatch):
    args = argparse.Namespace(
        library=tmp_path / "native.so",
        output_dir=tmp_path / "report",
        model=None,
        physical_npu=4,
        timeout=1,
        probe="0:0",
    )
    monkeypatch.setattr(gate, "library_evidence", lambda _: {"sha256": "abc"})
    calls = []

    def child(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "4"
        assert kwargs["env"]["ASCEND_LAUNCH_BLOCKING"] == "1"
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        if failure == "incomplete":
            report = good_evidence("direct")
            report["results"].pop()
            Path(command[command.index("--output") + 1]).write_text(json.dumps(report))
        return SimpleNamespace(returncode=1 if failure == "exit" else 0)

    monkeypatch.setattr(gate.subprocess, "run", child)
    assert gate.run_parent(args) == 1
    assert len(calls) == 1
    result = json.loads((args.output_dir / "summary.json").read_text())
    assert result["status"] == "failed"
    assert result["default_model_backend"] == "unchanged"
    assert result["on_chip_decode_verified"] is False
    assert result["model_integration_verified"] is False


def test_wrapper_requires_explicit_load_and_does_not_fallback(monkeypatch):
    module = importlib.import_module("vllm_ascend.quantization.vq2a8_ascendc")
    monkeypatch.setattr(module.torch.ops, "vq2a8_ascendc", SimpleNamespace())
    with pytest.raises(RuntimeError, match="No fallback"):
        module.vq2a8_ascendc(None, None, None, None, None, None)
    with pytest.raises(RuntimeError, match="No fallback"):
        module.cube_control(None, None)


def test_wrapper_loads_once_and_forwards_to_native_dispatch(tmp_path, monkeypatch):
    module = importlib.import_module("vllm_ascend.quantization.vq2a8_ascendc")
    library = tmp_path / "native.so"
    library.touch()
    other = tmp_path / "other.so"
    other.touch()
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    calls = []
    native = SimpleNamespace(projection=lambda *args: ("native", args), cube_control=lambda *args: args)
    ops = SimpleNamespace(loaded_libraries=set(), vq2a8_ascendc=SimpleNamespace())

    def load(path):
        calls.append(path)
        ops.loaded_libraries.add(path)
        ops.vq2a8_ascendc = native

    ops.load_library = load
    monkeypatch.setattr(module.torch, "ops", ops)
    module.load_library(library)
    module.load_library(library)
    assert calls == [str(library.resolve())]
    assert module.vq2a8_ascendc(1, 2, 3, 4, 5, 6) == ("native", (1, 2, 3, 4, 5, 6))
    assert module.cube_control(1, 2, bridge=True) == (1, 2, True)
    with pytest.raises(RuntimeError, match="fresh process"):
        module.load_library(other)


def test_device_tail_rows_use_shared_scalar_helper():
    kernel = (build.SOURCE / "kernel.cpp").read_text()
    # The executed C++ layout test exercises this exact helper for both AIVs,
    # including M<=16 where subtracting 16 without a guard would underflow.
    assert "HalfRows(m_, firstRow)" in kernel
    assert "Min(" not in kernel
    assert "Max(" not in kernel


def test_native_path_has_no_dense_workspace_or_triton():
    kernel = (build.SOURCE / "kernel.cpp").read_text()
    binding = (build.SOURCE / "torch_binding.cpp").read_text()
    assert "Mmad(c, a, b, p)" in kernel
    assert "Get<fp8_e4m3fn_t>()" in kernel
    assert "LoadData2DParamsV2" in kernel
    assert "KERNEL_TYPE_MIX_AIC_1_2" in kernel
    assert "Fixpipe<float, float, kToUb>" in kernel
    assert "CrossCoreWaitFlag<4, PIPE_MTE1>(kReady + kPeer)" in kernel
    assert "CrossCoreWaitFlag<4, PIPE_FIX>(kStored + kPeer)" in kernel
    assert "at::empty({x.size(0), n}" in binding
    assert "TORCH_LIBRARY_IMPL(vq2a8_ascendc, PrivateUse1" in binding
    assert "Mmad" not in (build.SOURCE / "torch_binding.cpp").read_text()  # device computation isn't host C++
    for source in (kernel, binding):
        for forbidden in ("tl.dot", "tl.gather", "torch::matmul", "aclnnMatmul", "aclrtMalloc", ".cpu()"):
            assert forbidden not in source


def test_build_does_not_certify_hardware(tmp_path, monkeypatch):
    monkeypatch.setattr(build.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="Linux Ascend950"):
        build.build(argparse.Namespace(cann=tmp_path, soc="Ascend950PR_957d", jobs=1, build_dir=tmp_path / "build"))
