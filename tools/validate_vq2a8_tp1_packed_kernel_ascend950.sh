#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

repo_path="${REPO:-/home/g00872988/vllm-ascend-vq2a8}"
model_path="${MODEL:-/home/g00872988/DeepSeek-V4-Flash-VQ2A8-32x256}"
artifact_path="${ARTIFACT:-${model_path}/experts_vq_ascend_v2}"
python_bin="${PYTHON_BIN:-}"
physical_npu="${PHYSICAL_NPU:-4}"
probes="${PROBES:-0:0,3:0,3:127,3:255,42:255}"
warmups="${WARMUPS:-1}"
repeats="${REPEATS:-3}"

if [[ -z "$python_bin" ]]; then
    python_bin="$(command -v python3 || true)"
fi

repo_path="$(readlink -f -- "$repo_path")"
model_path="$(readlink -f -- "$model_path")"
artifact_path="$(readlink -f -- "$artifact_path")"

for required_file in \
    "$repo_path/tools/validate_vq2a8_tp1_packed_kernel.py" \
    "$model_path/config.json" \
    "$artifact_path/manifest.json"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Missing required file: $required_file" >&2
        exit 2
    fi
done
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    echo "Cannot find an executable Python. Set PYTHON_BIN to the torch_npu Python." >&2
    exit 2
fi
if [[ ! "$physical_npu" =~ ^[0-9]+$ ]]; then
    echo "PHYSICAL_NPU must be one non-negative physical device ID, got: $physical_npu" >&2
    exit 2
fi
if [[ ! "$warmups" =~ ^[0-9]+$ || ! "$repeats" =~ ^[1-9][0-9]*$ ]]; then
    echo "WARMUPS must be non-negative and REPEATS must be positive integers." >&2
    exit 2
fi

echo "========== VQ2A8 TP1 M=1 NATIVE E4M3 PACKED KERNEL GATE =========="
echo "repo:       $repo_path"
echo "git:        $(git -C "$repo_path" rev-parse HEAD)"
echo "model:      $model_path"
echo "artifact:   $artifact_path"
echo "python:     $python_bin"
echo "device map: physical $physical_npu -> logical npu:0"
echo "compute:    E4M3 x E4M3, FP32 accumulate (tl.dot)"
echo "CV profile: CANN dynamic Vector -> Cube -> Vector pipeline"
echo "memory:     SSBuffer defaults; no explicit scope blacklist"
echo "probes:     $probes"
echo "warmups:    $warmups"
echo "repeats:    $repeats"

runtime_python_path="$repo_path${PYTHONPATH:+:$PYTHONPATH}"
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
    ASCEND_LAUNCH_BLOCKING=1 \
    PYTHONPATH="$runtime_python_path" \
    "$python_bin" \
        "$repo_path/tools/validate_vq2a8_tp1_packed_kernel.py" \
        --model "$model_path" \
        --artifact "$artifact_path" \
        --device npu:0 \
        --probes "$probes" \
        --warmups "$warmups" \
        --repeats "$repeats"
