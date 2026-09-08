#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-free QLI metadata preflight; not attention/model correctness acceptance."""

from __future__ import annotations

# Direct scripts must not put tools/bisect ahead of the stdlib bisect module.
# ruff: noqa: E402
import os as _bootstrap_os
import sys as _bootstrap_sys

if not __package__:
    _bootstrap_sys.path[0] = _bootstrap_os.path.dirname(
        _bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))
    )

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

# Shared C++ ABI: vllm_quant_lightning_indexer_metadata.h. The 160-word
# reserved tail is not consumed and is deliberately excluded from comparisons.
QLI_WORDS = 1024
LI_CORES, LD_CORES, WORDS_PER_CORE = 36, 72, 8
DEFINED_WORDS = (LI_CORES + LD_CORES) * WORDS_PER_CORE


def check_metadata(values):
    """Validate the single-request, no-flash-decode split schedule and ABI."""
    if len(values) != QLI_WORDS:
        raise ValueError("QLI metadata must contain 1024 int32 words.")
    last_end, enabled, disabled = [0, 0, 0], 0, False
    for core in range(LI_CORES):
        row = values[core * WORDS_PER_CORE : (core + 1) * WORDS_PER_CORE]
        if row[0] == 0:
            disabled = True
            if any(row):
                raise ValueError("QLI inactive LI slot is uninitialized; rebuild custom AICPU ops.")
        elif row[0] == 1 and not disabled and row[1:4] == last_end and row[4:7] > last_end:
            enabled += 1
            last_end = row[4:7]
        else:
            raise ValueError("QLI split intervals must be enabled consecutively, without gaps or overlaps.")
    if not enabled or last_end != [1, 0, 0]:
        raise ValueError("QLI metadata did not cover the complete single request.")
    if any(values[LI_CORES * WORDS_PER_CORE : DEFINED_WORDS]):
        raise ValueError("QLI no-FD metadata must leave every LD slot disabled and zeroed.")
    return enabled


def run_preflight(device, config, prompt_tokens=10):
    # This function is also called inside the supervised offline process,
    # before LLM construction. No production attention hot-path is changed.
    import torch

    from vllm_ascend.utils import bootstrap_custom_op_env

    if str(device) != "npu:0" or not 2 <= prompt_tokens <= 28:
        raise ValueError("QLI preflight requires isolated npu:0 and the bounded offline prompt.")
    bootstrap_custom_op_env()
    extension = importlib.import_module("vllm_ascend.vllm_ascend_C")
    package = Path(extension.__file__).resolve().parent
    files = [Path(extension.__file__).resolve()]
    vendor = package / "_cann_ops_custom"
    for name in ("libcust_opapi.so", "libtransformer_aicpu_kernels.so"):
        files.extend(sorted(vendor.rglob(name)))
    print(
        "QLI_ENV "
        + json.dumps(
            {
                "device": str(device),
                "physical_npu": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
                "package_files": [
                    {"path": str(p), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in files
                ],
            }
        ),
        flush=True,
    )
    cases = [("prefill", prompt_tokens, prompt_tokens)]
    cases += [(f"decode{i}", 1, prompt_tokens + i) for i in range(1, 4)]
    results = []
    for name, query_len, key_len in cases:
        kwargs = dict(
            num_heads_q=config["index_n_heads"],
            num_heads_k=1,
            head_dim=config["index_head_dim"],
            query_quant_mode=0,
            key_quant_mode=0,
            batch_size=1,
            max_seqlen_q=query_len,
            max_seqlen_k=key_len,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=config["index_topk"],
            sparse_mode=3,
            pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1,
            cmp_ratio=4,
            device=str(device),
        )
        print("QLI_START " + json.dumps({"case": name, "kwargs": kwargs}), flush=True)
        query = torch.tensor([query_len], dtype=torch.int32, device=device)
        key = torch.tensor([key_len], dtype=torch.int32, device=device)
        torch.npu.synchronize()
        start, previous, active = time.perf_counter(), None, None
        for repeat in range(3):
            result = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(
                actual_seq_lengths_query=query.clone(), actual_seq_lengths_key=key.clone(), **kwargs
            )
            torch.npu.synchronize()
            if result.dtype != torch.int32 or result.shape != (QLI_WORDS,) or result.device != device:
                raise ValueError("QLI returned the wrong dtype, shape or device.")
            values = result.cpu().tolist()
            active = check_metadata(values)
            current = values[:DEFINED_WORDS]
            if repeat and current != previous:
                raise ValueError("QLI defined metadata is not exactly repeatable.")
            previous = current
        record = dict(case=name, passed=True, active_li_cores=active, repeats=3, elapsed_s=time.perf_counter() - start)
        results.append(record)
        print("QLI_RESULT " + json.dumps(record), flush=True)
    print("QLI_METADATA_PREFLIGHT=PASS MODEL_VERIFIED=False", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-npu", type=int, default=4)
    parser.add_argument("--model", type=Path, help="Read config.json only; never load model weights.")
    args = parser.parse_args()
    if args.physical_npu < 0:
        parser.error("Physical NPU must be non-negative.")
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from tools.validate_vq2a8_tp1_acceptance import acceptance_environment

    child_env = acceptance_environment(repo, args.physical_npu, "npu:0")
    for key in set(os.environ) - set(child_env):
        del os.environ[key]
    os.environ.update(child_env)
    import torch
    import torch_npu  # noqa: F401

    from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device

    device = torch.device("npu:0")
    print("DEVICE " + json.dumps(_initialize_device(device)), flush=True)
    config = (
        json.loads((args.model / "config.json").read_text())
        if args.model
        else {"index_n_heads": 64, "index_head_dim": 128, "index_topk": 512}
    )
    run_preflight(device, config)


if __name__ == "__main__":
    main()
