# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only build recipe contracts: these tests never claim CANN/NPU execution."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import build_vq2a8_ascendc_v2 as build


def fake_source(tmp_path):
    source = tmp_path / "source"
    for name in build.MANDATORY_SOURCES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("reference source\n")
    (source / "code.txt").write_text("reference source\n")
    return source


def test_manifest_hashes_chat_source_and_native_inputs(tmp_path):
    source = fake_source(tmp_path)
    hashes = build.source_hashes(source, source)
    assert {"csrc/vq2a8_ascendc_v2/" + name for name in build.MANDATORY_SOURCES} <= hashes.keys()
    (source / "code.txt").write_text("corrected reference\n")
    updated = build.source_hashes(source, source)
    assert updated["csrc/vq2a8_expert_reference/code.txt"] != hashes["csrc/vq2a8_expert_reference/code.txt"]
    assert updated["csrc/vq2a8_ascendc_v2/kernel.cpp"] == hashes["csrc/vq2a8_ascendc_v2/kernel.cpp"]


def test_source_hashes_refuses_absent_or_empty_reference(tmp_path):
    source = fake_source(tmp_path)
    (source / "code.txt").write_text("")
    with pytest.raises(ValueError, match="empty"):
        build.source_hashes(source, source)
    (source / "code.txt").unlink()
    with pytest.raises(ValueError, match="missing"):
        build.source_hashes(source, source)


def test_source_hashes_ignores_test_and_cache_artifacts(tmp_path):
    source = fake_source(tmp_path)
    hashes = build.source_hashes(source, source)
    for name in ("tests/tiling.cpp", "build/generated.cpp", "native/__pycache__/unused.cpp"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not a build input")
    assert build.source_hashes(source, source) == hashes


def test_installed_recipe_discovery_never_uses_download_or_path_compiler(tmp_path):
    cann = tmp_path / "cann"
    with pytest.raises(RuntimeError, match="ascendc.cmake"):
        build.discover_cmake(cann)
    recipe = cann / build.ASCENDC_CMAKE_CANDIDATES[1]
    recipe.parent.mkdir(parents=True)
    recipe.write_text("include(...)")
    assert build.discover_cmake(cann) == recipe.resolve()
    compiler = cann / build.COMPILER_PREFIXES[0] / "bisheng"
    compiler.parent.mkdir(parents=True)
    compiler.write_text("not executed")
    assert build.discover_compiler(cann) == compiler.resolve()


def test_exact_soc_lookup_and_c310_guard(tmp_path):
    soc_dir = tmp_path / "aarch64-linux/data/platform_config"
    soc_dir.mkdir(parents=True)
    soc = soc_dir / "Ascend950DT_9574.ini"
    soc.write_text("CCEC_AIC_version=dav-c310-cube\nCCEC_VECTOR_version=dav-c310-vec\ncube_core_cnt=32\n")
    assert build.discover_soc_config(tmp_path, "ascend950dt_9574") == soc.resolve()
    assert build.validate_soc_config(soc)["cube_core_cnt"] == "32"
    with pytest.raises(RuntimeError, match="Exact SoC"):
        build.discover_soc_config(tmp_path, "Ascend950DT_9575")
    soc.write_text("CCEC_AIC_version=dav-c220\n")
    with pytest.raises(RuntimeError, match="not an Ascend950"):
        build.validate_soc_config(soc)
    soc.write_text("cube_core_cnt=28\n")
    assert build.validate_soc_config(soc)["cube_core_cnt"] == "28"
    soc.write_text("cube_core_cnt=0\n")
    with pytest.raises(RuntimeError, match="no AIC cores"):
        build.validate_soc_config(soc)


def test_commands_build_independent_library_without_pip_or_raw_link_flags(tmp_path):
    commands = build.cmake_commands(
        tmp_path / "source",
        tmp_path / "output",
        tmp_path / "cann",
        "Ascend950DT_9574",
        "sdk/ascendc.cmake",
        "torch_npu",
        "torch/cmake",
        4,
    )
    flat = json.dumps(commands)
    assert commands[0][0] == commands[1][0] == "cmake"
    assert "vq2a8_ascendc_v2" in commands[1]
    assert "vq2a8_ascendc" not in commands[1]
    assert "pip" not in flat and "ld.lld" not in flat and "--cce-" not in flat
    assert "-DVQ2A8_ASCENDC_CMAKE=sdk/ascendc.cmake" in commands[0]


@pytest.mark.parametrize(
    "soc,jobs", [("Ascend910B1", 4), ("Ascend950", 4), ("Ascend950;bad", 4), ("Ascend950DT_9574", 0)]
)
def test_validation_rejects_wrong_soc_and_jobs(tmp_path, soc, jobs):
    args = SimpleNamespace(soc=soc, jobs=jobs, build_dir=tmp_path / "out")
    with pytest.raises(ValueError, match="exact Ascend950"):
        build.validate_options(args)


def test_build_directory_must_not_overlap_source_or_foreign_project(tmp_path):
    source = fake_source(tmp_path)
    args = SimpleNamespace(soc="Ascend950DT_9574", jobs=4, build_dir=source / "build")
    with pytest.raises(ValueError, match="out-of-source"):
        build.validate_options(args, source)
    args.build_dir = tmp_path / "output"
    args.build_dir.mkdir()
    (args.build_dir / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=/different/project\n")
    with pytest.raises(ValueError, match="different CMake project"):
        build.validate_options(args, source)


def test_dry_run_writes_nothing_and_never_imports_torch_npu(tmp_path, monkeypatch, capsys):
    source = fake_source(tmp_path)
    monkeypatch.setattr(build, "SOURCE", source)
    monkeypatch.setattr(build, "source_hashes", lambda: {"code.txt": "fake-dry-run-hash"})
    monkeypatch.setattr(build.platform, "system", lambda: "Windows")
    monkeypatch.setattr(build, "toolchain_probe", lambda *args: pytest.fail("must not probe"))
    monkeypatch.setattr(build.importlib.util, "find_spec", lambda *args: pytest.fail("must not import torch_npu"))
    output = tmp_path / "output"
    args = build.parse_args(["--soc", "Ascend950DT_9574", "--build-dir", str(output), "--dry-run"])
    assert build.build(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "planned"
    assert report["abi_version"] == 1
    assert report["implementation"] == "ascendc_v2"
    assert not report["device_execution_verified"]
    assert not report["model_execution_verified"]
    assert not output.exists()


def test_compiler_identification_is_read_only_and_sdk_recipe_is_hashed(tmp_path, monkeypatch):
    recipe = tmp_path / "ascendc.cmake"
    recipe.write_text("ascendc_library c310 KERNEL_TYPE_MIX_AIC_1_2")
    soc = tmp_path / "soc.ini"
    soc.write_text("CCEC_AIC_version=dav-c310-cube\n")
    monkeypatch.setattr(build, "discover_compiler", lambda cann: Path("bisheng"))
    monkeypatch.setattr(build, "discover_soc_config", lambda cann, target: soc)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="bisheng version", stderr="")

    monkeypatch.setattr(build.subprocess, "run", run)
    report = build.toolchain_probe(tmp_path, "Ascend950DT_9574", recipe)
    assert calls == [["bisheng", "--version"], ["bisheng", "--help"]]
    assert report["recipe_sha256"][str(recipe)] == build.sha256(recipe)
    assert report["recipe_mentions_explicit_mix_1_2"]
    assert not report["api_compilation_verified"]
