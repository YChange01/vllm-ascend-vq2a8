#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -Eeuo pipefail

usage() {
    echo "Usage: $0 MODEL_PATH OUTPUT_PATH [LAYERS=0] [PHYSICAL_NPU=4]" >&2
    echo "OUTPUT_PATH must not exist and its parent directory must already exist." >&2
    echo "Example (gate): $0 /path/to/model /path/to/model/experts_vq_ascend_v2_layer0 0 4" >&2
    echo "Example (full): $0 /path/to/model /path/to/model/experts_vq_ascend_v2 all 4" >&2
}

if (( $# < 2 || $# > 4 )); then
    usage
    exit 2
fi

model_path=$(realpath -- "$1")
if [[ ! -d "$(dirname -- "$2")" ]]; then
    echo "Output parent directory must already exist: $(dirname -- "$2")" >&2
    exit 1
fi
output_parent=$(realpath -- "$(dirname -- "$2")")
output_name=$(basename -- "$2")
if [[ -z "$output_name" || "$output_name" == "." || "$output_name" == ".." ]]; then
    echo "Output path must name a new directory below an existing parent: $2" >&2
    exit 1
fi
output_path="${output_parent}/${output_name}"
layers=${3:-0}
physical_npu=${4:-4}
python_bin=${PYTHON_BIN:-/usr/local/python3.11.10/bin/python3}
script_path=$(realpath -- "${BASH_SOURCE[0]}")
repo_path=$(dirname -- "$(dirname -- "$script_path")")
source_path="${model_path}/experts_vq"
config_path="${model_path}/config.json"

if [[ ! -x "$python_bin" ]]; then
    echo "Python is not executable: $python_bin" >&2
    exit 1
fi
if [[ ! -d "$source_path" || ! -f "$config_path" ]]; then
    echo "Missing canonical experts_vq or config.json under: $model_path" >&2
    exit 1
fi
if [[ -e "$output_path" || -L "$output_path" ]]; then
    echo "Output already exists; refusing to overwrite: $output_path" >&2
    exit 1
fi
if [[ ! -f "$repo_path/tools/repack_vq2a8_tp1.py" ]]; then
    echo "Run this script from a complete vllm-ascend-vq2a8 checkout." >&2
    exit 1
fi

wrapper_workspace=""
cleanup() {
    if [[ -n "$wrapper_workspace" && -d "$wrapper_workspace" ]]; then
        rm -rf -- "$wrapper_workspace"
    fi
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

repack_python_path=$repo_path
if [[ -n ${PYTHONPATH:-} ]]; then
    repack_python_path="${repo_path}:${PYTHONPATH}"
fi

echo "========== REPACK PLAN =========="
echo "repo:       $repo_path"
echo "git:        $(git -C "$repo_path" rev-parse HEAD)"
echo "model:      $model_path"
echo "source:     $source_path"
echo "output:     $output_path"
echo "layers:     $layers"
echo "device map: physical $physical_npu -> logical npu:0 (gate only)"
du -sh -- "$source_path"
df -h -- "$output_parent"

echo
echo "========== ASCEND 950 GATE =========="
env \
    -u ASCEND_VISIBLE_DEVICES \
    -u NPU_VISIBLE_DEVICES \
    -u ASCEND_DEVICE_ID \
    -u DEVICE_ID \
    -u RANK_ID \
    -u LOCAL_RANK \
    -u RANK \
    -u WORLD_SIZE \
    -u PYTHONOPTIMIZE \
    ASCEND_RT_VISIBLE_DEVICES="$physical_npu" \
    PYTHONPATH="$repack_python_path" \
    "$python_bin" - <<'PY'
from pathlib import Path

import torch
import torch_npu


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


require(torch.npu.is_available(), "torch-npu reports no available device")
require(
    torch.npu.device_count() == 1,
    "physical device must map to exactly one logical NPU",
)
torch.npu.set_device(0)
soc = torch_npu.npu.get_soc_version()
name = torch.npu.get_device_name(0)
require(soc == 260, f"expected Ascend 950 SoC 260, got {soc}")
require("Ascend950" in name, f"expected Ascend950 device, got {name!r}")
from vllm_ascend.utils import get_ascend_device_type

device_type = get_ascend_device_type()
require(
    device_type.name == "A5",
    f"expected vLLM Ascend A5, got {device_type}",
)

meminfo = {}
for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    key, value = line.split(":", maxsplit=1)
    meminfo[key] = int(value.strip().split()[0]) * 1024
available_memory = meminfo.get("MemAvailable", 0)
required_memory = 8 * 1024**3
require(
    available_memory >= required_memory,
    "TP1 repack requires at least 8 GiB available host memory; "
    f"found {available_memory / 1024**3:.2f} GiB",
)
print(f"ASCEND950_GATE=PASS soc={soc} name={name} device_type={device_type.name}")
print(f"HOST_MEMORY_GATE=PASS available_gib={available_memory / 1024**3:.2f}")
PY

echo
echo "========== CPU REPACK =========="
echo "The conversion is CPU/I/O work; it does not consume NPU HBM."
wrapper_workspace=$(mktemp -d -- "${output_parent}/.${output_name}.wrapper-XXXXXX")
conversion_path="${wrapper_workspace}/artifact"
env \
    -u ASCEND_VISIBLE_DEVICES \
    -u NPU_VISIBLE_DEVICES \
    -u ASCEND_DEVICE_ID \
    -u DEVICE_ID \
    -u RANK_ID \
    -u LOCAL_RANK \
    -u RANK \
    -u WORLD_SIZE \
    -u PYTHONOPTIMIZE \
    ASCEND_RT_VISIBLE_DEVICES="$physical_npu" \
    PYTHONPATH="$repack_python_path" \
    "$python_bin" "$repo_path/tools/repack_vq2a8_tp1.py" \
        --input "$source_path" \
        --output "$conversion_path" \
        --model-config "$config_path" \
        --layers "$layers" \
        --require-reference-identity

echo
echo "========== OUTPUT HASH GATE =========="
env -u PYTHONOPTIMIZE PYTHONPATH="$repack_python_path" \
    "$python_bin" - "$conversion_path" "$layers" "$repo_path/tools/repack_vq2a8_tp1.py" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


root = Path(sys.argv[1]).resolve()
requested_layers = sys.argv[2]
repacker_path = Path(sys.argv[3]).resolve()
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
require(manifest["format"] == "vq2a8_direct_tp1_v1", "unexpected artifact format")
require(
    manifest["tp_size"] == 1 and manifest["tp_ranks"] == [0],
    "artifact is not TP1 rank0",
)
require(manifest["output"]["artifact_root"] == ".", "artifact root must be relative")
require(
    manifest["communication"] == {
        "multi_rank_owner": "moe_runner",
        "reduction_required": False,
        "tp_reduction_count": 0,
        "tp_reduction_owner": "none_for_tp1",
    },
    "unexpected TP1 communication contract",
)
if requested_layers == "all":
    require(manifest["complete"] is True, "full repack did not produce a complete artifact")
require(
    manifest["producer_evidence"]["reference_identity_match"] is True,
    "input does not match the externally verified reference artifact",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


require(
    manifest["repacker"] == {
        "contract_revision": 1,
        "tool": "tools/repack_vq2a8_tp1.py",
        "tool_sha256": sha256(repacker_path),
    },
    "repacker identity does not match the current checkout",
)

for layer in manifest["layers"]:
    for path_key, digest_key in (
        ("tensor_file", "tensor_sha256"),
        ("metadata_file", "metadata_sha256"),
    ):
        path = (root / layer[path_key]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"artifact path escapes output root: {path}") from error
        require(path.is_file(), f"missing artifact file: {path}")
        actual = sha256(path)
        require(
            actual == layer[digest_key],
            f"artifact hash mismatch for {path}: {actual} != {layer[digest_key]}",
        )

print(
    f"OUTPUT_HASH_GATE=PASS format={manifest['format']} "
    f"layers={manifest['layers_selected']} complete={manifest['complete']}"
)
PY

echo
echo "========== NO-CLOBBER PUBLISH =========="
if [[ -e "$output_path" || -L "$output_path" ]]; then
    echo "Output appeared during repack; refusing to overwrite: $output_path" >&2
    exit 1
fi
mv --no-clobber --no-target-directory -- "$conversion_path" "$output_path"
if [[ -e "$conversion_path" || -L "$conversion_path" ]]; then
    echo "Atomic publish was blocked; output may have appeared concurrently: $output_path" >&2
    exit 1
fi
rmdir -- "$wrapper_workspace"
wrapper_workspace=""
"$python_bin" - "$output_parent" <<'PY'
import os
import sys

directory = sys.argv[1]
descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY

echo
echo "TP1_REPACK=PASS output=$output_path"
if [[ "$layers" != "all" ]]; then
    echo "This is a partial gate artifact. Use a NEW output path with LAYERS=all for the full artifact."
fi
