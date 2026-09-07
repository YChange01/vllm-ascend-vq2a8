# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import copy
import json
import re
import subprocess
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tools.validate_vq2a8_fused_fp8 import case_keys, evidence_passed, run_parent, save_codegen
from tools.validate_vq2a8_phase4_kernel import (
    bitwise_equal,
    compare,
    same_fp8_oracle,
    synthetic_dense_oracle,
    synthetic_inputs,
)
from vllm_ascend.quantization.vq2a8_fused_fp8 import (
    BLOCK_K,
    _vq_decode_cube_kernel,
    launch_cube_control,
    launch_fused_fp8,
    validate_fused_fp8_inputs,
    vq2a8_fused_fp8,
)

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA developer check, not Ascend certification")


@pytest.mark.parametrize("rows", [1, 3, 10, 32])
def test_fused_contract_and_no_cpu_fallback(rows):
    inputs = synthetic_inputs(rows)
    shape = validate_fused_fp8_inputs(*inputs)
    assert (shape.size_n, shape.size_k, shape.column_tiles) == (64, 512, 3)
    with pytest.raises(ValueError, match="accelerator"):
        vq2a8_fused_fp8(*inputs)


@pytest.mark.parametrize("rows", [0, 33])
def test_fused_invalid_batch(rows):
    with pytest.raises(ValueError, match="M in"):
        validate_fused_fp8_inputs(*synthetic_inputs(rows))


@pytest.mark.parametrize("index", range(6))
def test_fused_rejects_wrong_dtypes(index):
    inputs = list(synthetic_inputs())
    inputs[index] = inputs[index].double()
    with pytest.raises(ValueError):
        validate_fused_fp8_inputs(*inputs)


def test_fused_rejects_noncontiguous_rows_and_unsupported_table():
    inputs = list(synthetic_inputs(6))
    inputs[:3] = [t[::2] for t in inputs[:3]]
    with pytest.raises(ValueError, match="contiguous"):
        validate_fused_fp8_inputs(*inputs)
    with pytest.raises(ValueError, match="column tiles"):
        validate_fused_fp8_inputs(*synthetic_inputs(3, tiles=33))


@pytest.mark.parametrize("index", [0, 3, 4, 5])
def test_fused_rejects_unaligned_contiguous_view(index):
    inputs = list(synthetic_inputs(3))
    tensor = inputs[index]
    storage = torch.empty(tensor.numel() + 1, dtype=tensor.dtype)
    view = storage[1:].view(tensor.shape)
    view.copy_(tensor)
    inputs[index] = view
    with pytest.raises(ValueError, match="aligned"):
        validate_fused_fp8_inputs(*inputs)


@GPU
@pytest.mark.parametrize(
    "m,n,k,tiles", [(1, 32, 512, 1), (3, 64, 512, 3), (10, 96, 1024, 7), (32, 64, 1024, 32), (3, 96, 4096, 32)]
)
def test_fused_actual_fp8_dot_decode_oracle_repeat_and_row_chunks(m, n, k, tiles):
    host = synthetic_inputs(m, n, k, tiles)
    inputs = tuple(t.cuda() for t in host)
    actual, compiled = launch_fused_fp8(*inputs)
    expected = same_fp8_oracle(host[:3], synthetic_dense_oracle(*host[3:]))
    assert compare(expected, actual)["allclose"]
    ttir = compiled.asm["ttir"]
    # Inspect the real compiled candidate, not a synthetic stand-in. CUDA
    # FP8 MMA proof does not certify the separately lowered Ascend branch.
    assert "tt.gather" in ttir and "tt.dot" in ttir
    assert "tt.reduce" not in ttir
    assert re.search(r"tt.dot.*?64x128xf8E4M3FN.*?128x32xf8E4M3FN", ttir)
    assert re.search(r"(?:wgmma\.mma_async|mma)\.sync\.aligned[^;]*\.e4m3\.e4m3", compiled.asm["ptx"])
    assert not any("tt.bitcast" in line and "!tt.ptr" in line for line in ttir.splitlines())
    gathers = [line for line in ttir.splitlines() if "tt.gather" in line]
    assert all("tensor<4096xi32>" in line and "xf32>" in line for line in gathers)
    for _ in range(3):
        assert bitwise_equal(actual, vq2a8_fused_fp8(*inputs))
    split = torch.cat(
        [
            vq2a8_fused_fp8(inputs[0][i : i + 1], inputs[1][i : i + 1], inputs[2][i : i + 1], *inputs[3:])
            for i in range(m)
        ]
    )
    assert bitwise_equal(actual, split)


@GPU
@pytest.mark.parametrize("case", ["zero", "impulse", "small", "extreme"])
def test_fused_numerical_boundaries(case):
    host = list(synthetic_inputs(3, 64, 1024, 32))
    if case == "zero":
        host[0] = torch.zeros((3, 1024)).to(torch.float8_e4m3fn)
        host[2].zero_()
    elif case == "impulse":
        x = torch.zeros((3, 1024))
        x[0, 127], x[1, 128], x[2, 1023] = -1, 1, 2
        host[0] = x.to(torch.float8_e4m3fn)
    elif case == "small":
        host[1] *= 1e-9
        host[2] *= 1e-9
    else:
        # Finite E4M3 extremes, signs and subnormals with a safe FP32 sum.
        values = torch.tensor([448.0, -448.0, 0.0, -0.0, 2**-9, -(2**-9), 1.0, -1.0])
        host[0] = values.repeat(3 * 1024 // 8).reshape(3, 1024).to(torch.float8_e4m3fn)
    expected = same_fp8_oracle(host[:3], synthetic_dense_oracle(*host[3:]))
    actual = vq2a8_fused_fp8(*(t.cuda() for t in host))
    assert compare(expected, actual)["allclose"]


@GPU
@pytest.mark.parametrize("m,n,tiles", [(1, 32, 1), (3, 96, 3), (32, 96, 32)])
def test_fused_masked_store_and_canaries(m, n, tiles):
    host = synthetic_inputs(m, n, 512, tiles)
    inputs = tuple(t.cuda() for t in host)
    guard = torch.full((32 + m * n + 32,), 97, device="cuda", dtype=torch.bfloat16)
    output = guard[32:-32].reshape(m, n)
    output.fill_(float("nan"))
    _vq_decode_cube_kernel[(n // 32,)](
        *inputs,
        output,
        M=m,
        N=n,
        K=512,
        COLUMN_TILES=tiles,
        TABLE_TILES=1 << (tiles - 1).bit_length(),
        BK=BLOCK_K,
        BM=64,
        ASCEND=False,
        num_warps=4,
        enable_fp_fusion=False,
    )
    assert torch.all(guard[:32] == 97) and torch.all(guard[-32:] == 97)
    assert compare(same_fp8_oracle(host[:3], synthetic_dense_oracle(*host[3:])), output)["allclose"]


@GPU
@pytest.mark.parametrize("seed", [11, 37, 73])
def test_native_fp8_accumulation_with_prepared_a8_dynamic_range(seed):
    # Small integer-like synthetic values hide imprecise long WGMMA sums.
    # Exercise cancellation and the real A8 range without relaxing tolerances.
    generator = torch.Generator().manual_seed(seed)
    host = list(synthetic_inputs(3, 64, 4096, 32))
    host[0] = (torch.randn((3, 4096), generator=generator) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
    host[4] = (torch.randn(host[4].shape, generator=generator) * 0.25).to(torch.float8_e4m3fn)
    expected = same_fp8_oracle(host[:3], synthetic_dense_oracle(*host[3:]))
    actual = vq2a8_fused_fp8(*(t.cuda() for t in host))
    assert compare(expected, actual)["allclose"]


@GPU
@pytest.mark.parametrize("m", [1, 3, 10, 32])
@pytest.mark.parametrize("bridge", [False, True])
def test_matching_geometry_cube_control(m, bridge):
    a = ((torch.arange(m * 512).reshape(m, 512) % 31 - 15) / 8).to(torch.float8_e4m3fn)
    b = ((torch.arange(32 * 512).reshape(32, 512) % 29 - 14) / 8).to(torch.float8_e4m3fn)
    actual, _ = launch_cube_control(a.cuda(), b.cuda(), bridge=bridge)
    assert compare((a.double() @ b.double().T * (-1 if bridge else 1)).bfloat16(), actual)["allclose"]


def valid_evidence(stage, device="npu:0"):
    records = []
    for key in case_keys(stage, "0:0"):
        records.append(
            {
                "key": key,
                "passed": True,
                "repeat_exact": True,
                "oracle": {"allclose": True},
                "baseline": {"allclose": True},
                "chain_baseline": {"allclose": True},
                "codegen": {
                    "files": [{"path": "kernel.ttir"}],
                    "cuda_native_fp8_mma_detected": device.startswith("cuda"),
                },
                "row_chunk_exact": True,
                "dense_expert_weight_on_device": False,
                "decoded_weight_tile": [32, 128],
                "dot_api": "tl.dot_scaled:e4m3" if device.startswith("npu") else "tl.dot:e4m3",
            }
        )
    return {
        "status": "passed",
        "stage": stage,
        "device": device,
        "probe": "0:0",
        "results": records,
        "npu_execution_verified": device.startswith("npu"),
        "native_instruction_verified": False,
        "on_chip_decode_verified": False,
        "model_integration_verified": False,
        "performance_verified": False,
    }


@pytest.mark.parametrize("stage", ["direct", "bridge", "fused", "expert"])
def test_evidence_requires_complete_stage(stage, tmp_path):
    path = tmp_path / "report.json"
    good = valid_evidence(stage)
    path.write_text(json.dumps(good))
    assert evidence_passed(path, stage, "npu:0", "0:0")
    mutations = [
        dict(status="failed"),
        dict(device="cuda:0"),
        dict(npu_execution_verified=False),
        dict(model_integration_verified=True),
        dict(native_instruction_verified=True),
        dict(results=good["results"][:-1]),
        dict(results=good["results"] + good["results"][:1]),
    ]
    for mutation in mutations:
        path.write_text(json.dumps({**good, **mutation}))
        assert not evidence_passed(path, stage, "npu:0", "0:0")
    for field, value in (("repeat_exact", False), ("oracle", {"allclose": False}), ("codegen", {"files": []})):
        bad = copy.deepcopy(good)
        bad["results"][0][field] = value
        path.write_text(json.dumps(bad))
        assert not evidence_passed(path, stage, "npu:0", "0:0")


def test_save_codegen_retains_verbatim_without_certifying(tmp_path):
    compiled = SimpleNamespace(asm={"ttir": "IR", "cubin": b"\x00\xff", "../../escape": "bad"}, metadata="metadata")
    result = save_codegen(compiled, tmp_path / "codegen")
    assert (tmp_path / "codegen/kernel.ttir").read_text() == "IR"
    assert (tmp_path / "codegen/kernel.cubin").read_bytes() == b"\x00\xff"
    assert len(result["files"]) == 2 and result["reviewed"] is False


def test_save_codegen_rejects_silent_fp16_mma(tmp_path):
    compiled = SimpleNamespace(asm={"ptx": "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32;"}, metadata="metadata")
    with pytest.raises(AssertionError, match="native E4M3"):
        save_codegen(compiled, tmp_path / "codegen")
    assert (tmp_path / "codegen/kernel.ptx").is_file()


def test_direct_retains_codegen_before_an_oracle_failure(tmp_path, monkeypatch):
    import tools.validate_vq2a8_fused_fp8 as driver
    import tools.validate_vq2a8_phase4_kernel as checks
    import tools.validate_vq2a8_tp1_packed_kernel as runtime
    import vllm_ascend.quantization.vq2a8_fused_fp8 as kernel

    compiled = SimpleNamespace(asm={"ttir": "diagnostic IR"}, metadata="unreviewed")
    monkeypatch.setattr(runtime, "environment_report", lambda: {})
    monkeypatch.setattr(runtime, "_initialize_device", lambda device: {"type": "host-test"})
    monkeypatch.setattr(kernel, "launch_cube_control", lambda *args, **kwargs: (None, compiled))

    def reject(*args):
        assert (tmp_path / "direct-codegen/direct-m32/kernel.ttir").read_text() == "diagnostic IR"
        raise AssertionError("oracle mismatch")

    monkeypatch.setattr(checks, "compare", reject)
    args = argparse.Namespace(device="cpu", stage="direct", probe="0:0", output=tmp_path / "direct.json")
    with pytest.raises(AssertionError, match="oracle mismatch"):
        driver.run_child(args)
    report = json.loads(args.output.read_text())
    assert report["status"] == "failed" and report["results"] == []
    assert report["native_instruction_verified"] is False


def test_cuda_report_must_have_actual_native_instruction(tmp_path):
    path = tmp_path / "report.json"
    report = valid_evidence("fused", "cuda:0")
    path.write_text(json.dumps(report))
    assert evidence_passed(path, "fused", "cuda:0", "0:0")
    report["results"][0]["codegen"]["cuda_native_fp8_mma_detected"] = False
    path.write_text(json.dumps(report))
    assert not evidence_passed(path, "fused", "cuda:0", "0:0")


def test_supervisor_success_does_not_promote_model(tmp_path, monkeypatch):
    import tools.validate_vq2a8_fused_fp8 as driver

    stages = []

    def child(command, **kwargs):
        stage = command[command.index("--stage") + 1]
        stages.append(stage)
        from pathlib import Path

        Path(command[command.index("--output") + 1]).write_text(json.dumps(valid_evidence(stage)))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(driver.subprocess, "run", child)
    monkeypatch.setattr(driver, "acceptance_environment", lambda *args: {})
    monkeypatch.setattr(driver, "LiveChildLog", lambda *args: nullcontext())
    args = argparse.Namespace(
        output_dir=tmp_path / "run",
        model=None,
        artifact=None,
        probe="0:0",
        device="npu:0",
        physical_npu=4,
        allow_partial_artifact=False,
        timeout=1,
    )
    assert run_parent(args) == 0
    assert stages == ["direct", "bridge", "fused"]
    result = json.loads((tmp_path / "run/summary.json").read_text())
    assert result["fused_projection_npu_execution_verified"] is True
    assert result["model_integration_verified"] is False and result["native_instruction_verified"] is False


@pytest.mark.parametrize(
    "options",
    [
        ["--allow-partial-artifact"],
        ["--probe", "0:0,3:0"],
        ["--timeout", "0"],
        ["--stage", "expert"],
        ["--stage", "direct"],
    ],
)
def test_cli_rejects_unsafe_or_incomplete_options(options, monkeypatch):
    import tools.validate_vq2a8_fused_fp8 as driver

    monkeypatch.setattr(driver.sys, "argv", ["prototype", *options])
    with pytest.raises(SystemExit) as error:
        driver.main()
    assert error.value.code == 2


@pytest.mark.parametrize("failure", ["exit", "timeout", "evidence"])
def test_supervisor_stops_without_fallback_reset_or_model(failure, tmp_path, monkeypatch):
    import tools.validate_vq2a8_fused_fp8 as driver

    calls = []

    def child(command, **kwargs):
        calls.append(command)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        return SimpleNamespace(returncode=1 if failure == "exit" else 0)

    monkeypatch.setattr(driver.subprocess, "run", child)
    monkeypatch.setattr(driver, "acceptance_environment", lambda *args: {})
    monkeypatch.setattr(driver, "LiveChildLog", lambda *args: nullcontext())
    args = argparse.Namespace(
        output_dir=tmp_path / "run",
        model=None,
        artifact=None,
        probe="0:0",
        device="npu:0",
        physical_npu=4,
        allow_partial_artifact=False,
        timeout=1,
    )
    assert run_parent(args) == 1
    assert len(calls) == 1 and calls[0][calls[0].index("--stage") + 1] == "direct"
    result = json.loads((tmp_path / "run/summary.json").read_text())
    assert result["status"] == "failed" and result["model_integration_verified"] is False


def test_default_runtime_does_not_import_fused_prototype():
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "vllm_ascend"
    for relative in (
        "quantization/vq2a8_moe.py",
        "quantization/vq2a8_execution.py",
        "quantization/vq2a8_triton.py",
        "patch/worker/vq2a8_offline_model.py",
    ):
        assert "vq2a8_fused_fp8" not in (root / relative).read_text()
