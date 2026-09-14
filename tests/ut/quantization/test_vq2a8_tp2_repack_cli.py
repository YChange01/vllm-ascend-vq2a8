# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CPU subprocess coverage for the independent canonical-to-TP2 CLI."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools/repack_vq2a8_tp2.py"
FIELD_DTYPES = {
    "packed_zn": "U8",
    "pair_lut": "U8",
    "activation_order": "I64",
    "weight_scale": "F32",
    "weight_bias": "F32",
    "rht_sign": "I8",
}
CPU_ENTRYPOINT = """
import importlib.abc
import runpy
import sys

class RejectRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.')
               for name in ('vllm', 'torch_npu', 'vllm_ascend')):
            raise AssertionError('CPU repacker imported runtime package: ' + fullname)
        return None

sys.meta_path.insert(0, RejectRuntimeImports())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
PUBLISH_CONTRACT_ENTRYPOINT = CPU_ENTRYPOINT.replace(
    "runpy.run_path(sys.argv[0], run_name='__main__')",
    """
namespace = runpy.run_path(sys.argv[0], run_name='_tp2_publish_test')
cli_globals = namespace['_publish_directory'].__globals__
import ctypes
import errno
import os
from types import SimpleNamespace

staging, output = map(cli_globals['Path'], sys.argv[1:3])
case = sys.argv[3]
real_rename, real_cdll = os.rename, ctypes.CDLL
real_cli_os, real_cli_sys = cli_globals['os'], cli_globals['sys']
calls, library_calls = [], []

class MockRenameAt2:
    def __call__(self, *arguments):
        expected = (-100, os.fsencode(staging), -100, os.fsencode(output), 1)
        assert arguments == expected, (arguments, expected)
        calls.append(arguments)
        if case == 'exists':
            ctypes.set_errno(errno.EEXIST)
            return -1
        real_rename(staging, output)
        return 0

rename = MockRenameAt2()
def load_libc(name, *, use_errno):
    assert name is None and use_errno is True
    library_calls.append(name)
    if case == 'missing':
        return SimpleNamespace()
    return SimpleNamespace(renameat2=rename)

def forbid_fallback(*args, **kwargs):
    raise AssertionError('no-replace publication must never fall back to os.rename')

# Substitute only the helper's globals; do not change the real os module or
# pretend the full CPU conversion/torch process is running on another OS.
cli_globals['os'] = SimpleNamespace(name='posix', fsencode=os.fsencode,
                                    strerror=os.strerror, rename=forbid_fallback)
cli_globals['sys'] = SimpleNamespace(platform='darwin' if case == 'unsupported' else 'linux')
ctypes.CDLL = load_libc
try:
    try:
        namespace['_publish_directory'](staging, output)
    except OSError as error:
        assert case != 'success', repr(error)
        if case == 'exists':
            assert isinstance(error, FileExistsError)
            assert error.errno == errno.EEXIST and error.filename == str(output)
        elif case == 'missing':
            assert error.errno == errno.ENOTSUP
            assert 'renameat2' in str(error)
        else:
            assert 'Linux and Windows only' in str(error)
    else:
        assert case == 'success', 'unsupported publication unexpectedly succeeded'
finally:
    ctypes.CDLL = real_cdll
    cli_globals['os'], cli_globals['sys'] = real_cli_os, real_cli_sys

if case in ('success', 'exists'):
    assert len(calls) == 1 and len(library_calls) == 1
    assert rename.argtypes == [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                               ctypes.c_char_p, ctypes.c_uint]
    assert rename.restype is ctypes.c_int
else:
    assert not calls
    assert len(library_calls) == (1 if case == 'missing' else 0)
assert 'torch' not in sys.modules
print('MOCK_LINUX_NOREPLACE_CONTRACT=PASS')
""",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(directory):
    return {path.relative_to(directory).as_posix(): digest(path) for path in directory.rglob("*") if path.is_file()}


def write_model(tmp_path, *, layers=1, present_layers=None, experts=2, invalid_last_permutation=False):
    model = tmp_path / "model"
    source = model / "experts_vq"
    source.mkdir(parents=True)
    config = dict(
        num_hidden_layers=layers,
        num_hash_layers=0,
        n_routed_experts=experts,
        hidden_size=512,
        moe_intermediate_size=256,
        quantization_config={"quant_method": "vq2a8"},
    )
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    selected = tuple(range(layers)) if present_layers is None else present_layers
    for layer in selected:
        metadata, tensors = {}, {}
        # Reverse order verifies that shard grouping follows explicit expert IDs.
        for expert in reversed(range(experts)):
            for kind, columns in (("gate_up", 512), ("down", 256)):
                name = f"{layer}.mlp.experts.{expert}.{kind}"
                rows, row_group, group = 512, 32, 256
                metadata[name] = dict(
                    rows=rows,
                    cols=columns,
                    n_row_tiles=rows // row_group,
                    n_col_tiles=columns // group,
                    row_group_size=row_group,
                    group_size=group,
                    K=16,
                    index_bits=4,
                    vector_len=2,
                    n_vectors=rows * columns // 2,
                    n_elements=rows * columns,
                    orig_shape=[rows, columns],
                    norm_dim=0,
                    enable_perm=True,
                    enable_norm=True,
                    enable_rht=True,
                    rht_block_size=128,
                    rht_true_columns=columns,
                )
                code = 1 + expert + (3 if kind == "down" else 0)
                code_word = sum(code << (4 * nibble) for nibble in range(8))
                if code_word >= 2**31:
                    code_word -= 2**32
                shape = (columns // group, rows // row_group, 16, 2)
                book_bytes = (torch.arange((columns // group) * (rows // row_group) * 32) % 126).to(torch.uint8)
                book_bytes[::32] = 128  # Preserve negative zero in every output-row tile.
                permutation = torch.roll(torch.arange(columns, dtype=torch.int32), shifts=17)
                if invalid_last_permutation and layer == selected[-1] and expert == experts - 1 and kind == "down":
                    permutation.zero_()
                values = dict(
                    packed_indices=torch.full((rows * columns // 16,), code_word, dtype=torch.int32),
                    codebooks=book_bytes.reshape(shape).view(torch.float8_e4m3fn),
                    perm=permutation,
                    weight_scale=torch.linspace(0.5, 1.5, columns) + expert / 8,
                    weight_bias=(torch.arange(columns).float() % 7 - 3) / 16,
                    rht_sign=torch.where(torch.arange(columns) % 2 == 0, 1, -1).to(torch.int8),
                )
                tensors.update({f"{name}.{field}": value for field, value in values.items()})
        (source / f"experts_vq_layer_{layer}.json").write_text(json.dumps(metadata), encoding="utf-8")
        save_file(tensors, source / f"experts_vq_layer_{layer}.safetensors")
    return model, source


def run_cli(source, output, *flags, inject=None):
    assert SCRIPT.is_file(), "The TP2 CLI must exist before running its subprocess contract tests"
    entrypoint = CPU_ENTRYPOINT
    if inject is not None:
        entrypoint = entrypoint.replace(
            "runpy.run_path(sys.argv[0], run_name='__main__')",
            "namespace = runpy.run_path(sys.argv[0], run_name='_tp2_cli_test')\n"
            "cli_globals = namespace['main'].__globals__\n" + inject + "\nraise SystemExit(namespace['main']())",
        )
    command = [
        sys.executable,
        "-I",
        "-X",
        "utf8",
        "-c",
        entrypoint,
        str(SCRIPT),
        "--input",
        str(source),
        "--output",
        str(output),
        *flags,
    ]
    return subprocess.run(command, cwd=REPO, capture_output=True, text=True, encoding="utf-8", timeout=90)


def require_success(result):
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def assert_no_staging(output):
    assert not output.exists()
    assert list(output.parent.glob(f".{output.name}.partial-*")) == []


def validate_artifact(output, *, layers, experts, experts_per_shard, complete):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "vq2a8_zn_tp2_v1"
    assert manifest["tp_size"] == 2 and manifest["tp_ranks"] == [0, 1]
    assert manifest["runtime_compatible"] is False
    assert manifest["complete"] is complete
    shards = manifest["shards"]
    expected = {
        (rank, layer, tuple(range(start, min(start + experts_per_shard, experts))))
        for rank in range(2)
        for layer in layers
        for start in range(0, experts, experts_per_shard)
    }
    assert {(entry["rank"], entry["layer"], tuple(entry["expert_ids"])) for entry in shards} == expected
    assert len(shards) == len(expected)
    for shard in shards:
        ids, rank, layer = shard["expert_ids"], shard["rank"], shard["layer"]
        stem = f"tp2/rank{rank}/layer_{layer:03d}/experts_{ids[0]:04d}_{ids[-1] + 1:04d}"
        assert shard["file"] == stem + ".safetensors"
        assert shard["metadata_file"] == stem + ".json"
        tensor_file, metadata_file = output / shard["file"], output / shard["metadata_file"]
        assert digest(tensor_file) == shard["sha256"]
        assert digest(metadata_file) == shard["metadata_sha256"]
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        matrices = metadata["matrices"]
        assert {(item["expert_id"], item["kind"]) for item in matrices} == {
            (expert, kind) for expert in ids for kind in ("gate_up", "down")
        }
        assert len(matrices) == 2 * len(ids)
        expected_keys = {
            f"{expert}.{kind}.{field}" for expert in ids for kind in ("gate_up", "down") for field in FIELD_DTYPES
        }
        payload_bytes = 0
        with safe_open(tensor_file, framework="pt", device="cpu") as handle:
            assert set(handle.keys()) == expected_keys
            for matrix in matrices:
                expert, kind = matrix["expert_id"], matrix["kind"]
                assert matrix["name"] == f"{layer}.mlp.experts.{expert}.{kind}"
                assert matrix["runtime_supported"] is False
                assert set(matrix["tensors"]) == set(FIELD_DTYPES)
                for field, dtype in FIELD_DTYPES.items():
                    entry = matrix["tensors"][field]
                    key = f"{expert}.{kind}.{field}"
                    assert entry["key"] == key and entry["dtype"] == dtype
                    tensor_slice = handle.get_slice(key)
                    assert tensor_slice.get_dtype() == dtype
                    assert list(tensor_slice.get_shape()) == entry["shape"]
                    value = handle.get_tensor(key)
                    payload_bytes += value.numel() * value.element_size()
                    if field == "activation_order":
                        assert value.dtype == torch.int64
                    if field == "pair_lut":
                        assert bool((value == 128).any())
        assert shard["payload_bytes"] == payload_bytes
    assert not list(output.rglob("*.partial"))
    return manifest


@pytest.mark.parametrize("experts_per_shard", [1, 2, 32])
def test_tp2_repack_cli_publishes_two_ranks_with_reloaded_tensors_and_hashes(tmp_path, experts_per_shard):
    model, source = write_model(tmp_path)
    before = snapshot(model)
    output = tmp_path / "tp2 output"
    require_success(run_cli(source, output, "--experts-per-shard", str(experts_per_shard)))
    validate_artifact(output, layers=[0], experts=2, experts_per_shard=experts_per_shard, complete=True)
    assert snapshot(model) == before


def test_tp2_repack_cli_partial_selection_is_not_complete(tmp_path):
    model, source = write_model(tmp_path, layers=2, present_layers=(0,), experts=1)
    before = snapshot(model)
    output = tmp_path / "partial"
    require_success(run_cli(source, output, "--layers", "0"))
    validate_artifact(output, layers=[0], experts=1, experts_per_shard=32, complete=False)
    assert snapshot(model) == before


def test_tp2_repack_cli_default_all_rejects_missing_configured_layer(tmp_path):
    model, source = write_model(tmp_path, layers=2, present_layers=(0,), experts=1)
    before = snapshot(model)
    output = tmp_path / "incomplete"
    assert run_cli(source, output).returncode != 0
    assert_no_staging(output)
    assert snapshot(model) == before


def test_tp2_repack_cli_dry_run_never_creates_output_or_parent(tmp_path):
    model, source = write_model(tmp_path, experts=1)
    before = snapshot(model)
    output = tmp_path / "not-created" / "tp2"
    result = run_cli(source, output, "--dry-run")
    require_success(result)
    assert result.stdout.strip()
    assert not output.parent.exists()
    assert snapshot(model) == before


def test_tp2_repack_cli_explicit_model_config_is_used(tmp_path):
    model, source = write_model(tmp_path, experts=1)
    custom = tmp_path / "separate-config.json"
    custom.write_bytes((model / "config.json").read_bytes())
    (model / "config.json").write_text("{}", encoding="utf-8")
    before = snapshot(model)
    output = tmp_path / "explicit-config"
    require_success(run_cli(source, output, "--model-config", str(custom)))
    validate_artifact(output, layers=[0], experts=1, experts_per_shard=32, complete=True)
    assert snapshot(model) == before


@pytest.mark.parametrize("target", ["same", "descendant", "ancestor"])
def test_tp2_repack_cli_refuses_overlapping_source_output_trees(tmp_path, target):
    model, source = write_model(tmp_path, experts=1)
    before = snapshot(model)
    output = {"same": source, "descendant": source / "output", "ancestor": model}[target]
    assert run_cli(source, output).returncode != 0
    assert snapshot(model) == before
    assert not (source / "output").exists()


def test_tp2_repack_cli_existing_output_is_preserved(tmp_path):
    model, source = write_model(tmp_path, experts=1)
    before = snapshot(model)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.bin").write_bytes(b"existing user data")
    output_before = snapshot(output)
    assert run_cli(source, output).returncode != 0
    assert snapshot(output) == output_before and snapshot(model) == before


@pytest.mark.parametrize(
    "flags",
    [
        ("--threads", "0"),
        ("--threads", "-1"),
        ("--experts-per-shard", "0"),
        ("--experts-per-shard", "-1"),
        ("--layers", "1"),
        ("--layers", "1-0"),
        ("--layers", "0,"),
        ("--resume",),
    ],
)
def test_tp2_repack_cli_invalid_options_never_publish(tmp_path, flags):
    model, source = write_model(tmp_path, experts=1)
    before = snapshot(model)
    output = tmp_path / "invalid"
    assert run_cli(source, output, *flags).returncode != 0
    assert_no_staging(output)
    assert snapshot(model) == before


def test_tp2_repack_cli_bad_late_payload_cannot_publish_partial_output(tmp_path):
    model, source = write_model(tmp_path, invalid_last_permutation=True)
    before = snapshot(model)
    output = tmp_path / "bad-payload"
    result = run_cli(source, output, "--experts-per-shard", "1")
    assert result.returncode != 0
    assert "perm" in (result.stdout + result.stderr).lower()
    assert_no_staging(output)
    assert snapshot(model) == before


def test_tp2_repack_cli_layer_ranges_are_sorted_and_deduplicated(tmp_path):
    model, source = write_model(tmp_path, layers=2, experts=1)
    before = snapshot(model)
    output = tmp_path / "range-selection"
    require_success(run_cli(source, output, "--layers", "1,0-1"))
    manifest = validate_artifact(output, layers=[0, 1], experts=1, experts_per_shard=32, complete=True)
    assert manifest["layers_selected"] == [0, 1]
    assert snapshot(model) == before


def test_tp2_repack_cli_insufficient_disk_fails_before_conversion(tmp_path):
    model, source = write_model(tmp_path, experts=1)
    before = snapshot(model)
    output = tmp_path / "no-disk"
    result = run_cli(
        source,
        output,
        inject="""
real_disk_usage = cli_globals['shutil'].disk_usage
cli_globals['shutil'].disk_usage = lambda path: real_disk_usage(path)._replace(free=0)
""",
    )
    assert result.returncode != 0
    assert "Insufficient output disk space" in result.stderr
    assert "VQ2_TP2_STAGE=convert_start" not in result.stdout
    assert_no_staging(output)
    assert snapshot(model) == before


def test_tp2_repack_cli_second_shard_write_failure_removes_first_shard(tmp_path):
    model, source = write_model(tmp_path, experts=1)
    before = snapshot(model)
    output = tmp_path / "failed-write"
    result = run_cli(
        source,
        output,
        inject="""
import errno
real_write_shard = cli_globals['_write_shard']
written = []
def write_failure(*args, **kwargs):
    if written:
        assert (args[0] / written[0]['file']).is_file()
        raise OSError(errno.ENOSPC, 'injected second shard write failure')
    result = real_write_shard(*args, **kwargs)
    written.append(result)
    return result
cli_globals['_write_shard'] = write_failure
""",
    )
    assert result.returncode != 0
    assert "injected second shard write failure" in result.stderr
    assert "VQ2_TP2_STAGE=partial_output_removed" in result.stdout
    assert_no_staging(output)
    assert snapshot(model) == before


def test_tp2_repack_cli_source_change_during_conversion_refuses_publication(tmp_path):
    model, source = write_model(tmp_path, experts=1)
    before = snapshot(model)
    output = tmp_path / "changed-source"
    result = run_cli(
        source,
        output,
        inject="""
real_write_shard = cli_globals['_write_shard']
source = cli_globals['Path'](sys.argv[sys.argv.index('--input') + 1])
changed = []
def mutate_source_after_write(*args, **kwargs):
    result = real_write_shard(*args, **kwargs)
    if not changed:
        path = source / 'experts_vq_layer_0.json'
        path.write_bytes(path.read_bytes() + b'\\n')
        changed.append(path)
    return result
cli_globals['_write_shard'] = mutate_source_after_write
""",
    )
    assert result.returncode != 0
    assert "Source layer 0 changed during conversion" in result.stderr
    assert_no_staging(output)
    after = snapshot(model)
    changed_file = "experts_vq/experts_vq_layer_0.json"
    assert after.pop(changed_file) != before.pop(changed_file)
    assert after == before  # Only the explicitly injected external mutation occurred.


def test_tp2_repack_cli_completed_layer_change_during_later_layer_refuses_publication(tmp_path):
    model, source = write_model(tmp_path, layers=2, experts=1)
    before = snapshot(model)
    output = tmp_path / "changed-completed-layer"
    result = run_cli(
        source,
        output,
        inject="""
real_write_shard = cli_globals['_write_shard']
source = cli_globals['Path'](sys.argv[sys.argv.index('--input') + 1])
changed = []
def mutate_completed_layer(*args, **kwargs):
    result = real_write_shard(*args, **kwargs)
    if args[1] == 1 and not changed:
        assert (args[0] / 'tp2/rank1/layer_000/experts_0000_0001.safetensors').is_file()
        path = source / 'experts_vq_layer_0.json'
        path.write_bytes(path.read_bytes() + b'\\n')
        changed.append(path)
    return result
cli_globals['_write_shard'] = mutate_completed_layer
""",
    )
    assert result.returncode != 0
    assert "Source layer 0 changed" in result.stderr
    assert "VQ2_TP2_STAGE=layer_done layer=1" in result.stdout
    assert "VQ2_TP2_STAGE=partial_output_removed" in result.stdout
    assert_no_staging(output)
    after = snapshot(model)
    changed_file = "experts_vq/experts_vq_layer_0.json"
    assert after.pop(changed_file) != before.pop(changed_file)
    assert after == before


def test_tp2_repack_cli_metadata_geometry_change_after_planning_refuses_conversion(tmp_path):
    model, source = write_model(tmp_path, experts=1)
    before = snapshot(model)
    output = tmp_path / "changed-planned-metadata"
    result = run_cli(
        source,
        output,
        inject="""
real_convert_layer = cli_globals['_convert_layer']
def mutate_planned_metadata(*args, **kwargs):
    path = args[0] / 'experts_vq_layer_0.json'
    metadata = cli_globals['json'].loads(path.read_text(encoding='utf-8'))
    entry = metadata['0.mlp.experts.0.down']
    for field in ('rows', 'n_row_tiles', 'n_elements', 'n_vectors'):
        entry[field] *= 2
    entry['orig_shape'][0] *= 2
    path.write_text(cli_globals['json'].dumps(metadata), encoding='utf-8')
    artifact, converter = cli_globals['_cpu_modules']()
    # The changed JSON is a valid matrix spec, not malformed JSON/geometry.
    spec = artifact.load_layer_specs(args[0], 0)['0.mlp.experts.0.down']
    converter.validate_tp2_spec(spec)
    return real_convert_layer(*args, **kwargs)
cli_globals['_convert_layer'] = mutate_planned_metadata
""",
    )
    assert result.returncode != 0
    error = result.stderr.lower()
    assert "metadata" in error and "changed" in error and "plan" in error
    assert "VQ2_TP2_STAGE=write_verify" not in result.stdout
    assert "VQ2_TP2_STAGE=partial_output_removed" in result.stdout
    assert_no_staging(output)
    after = snapshot(model)
    changed_file = "experts_vq/experts_vq_layer_0.json"
    assert after.pop(changed_file) != before.pop(changed_file)
    assert after == before


def test_tp2_repack_cli_help_stays_independent_of_runtime_imports():
    result = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", "-c", CPU_ENTRYPOINT, str(SCRIPT), "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    require_success(result)
    assert "--experts-per-shard" in result.stdout and "--dry-run" in result.stdout
    assert "--resume" not in result.stdout


@pytest.mark.parametrize("case", ["success", "exists", "missing", "unsupported"])
def test_tp2_repack_cli_mock_linux_publish_syscall_contract(tmp_path, case):
    """Mock libc boundary only; this is not a Linux filesystem integration test."""
    staging, output = tmp_path / "阶段 staging", tmp_path / "结果 output"
    staging.mkdir()
    (staging / "candidate.bin").write_bytes(b"unpublished candidate")
    if case == "exists":
        output.mkdir()
        (output / "keep.bin").write_bytes(b"concurrent user data")
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-X",
            "utf8",
            "-c",
            PUBLISH_CONTRACT_ENTRYPOINT,
            str(SCRIPT),
            str(staging),
            str(output),
            case,
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    require_success(result)
    assert "MOCK_LINUX_NOREPLACE_CONTRACT=PASS" in result.stdout
    if case == "success":
        assert not staging.exists()
        assert (output / "candidate.bin").read_bytes() == b"unpublished candidate"
    else:
        assert (staging / "candidate.bin").read_bytes() == b"unpublished candidate"
        if case == "exists":
            assert snapshot(output) == {"keep.bin": hashlib.sha256(b"concurrent user data").hexdigest()}
        else:
            assert not output.exists()
