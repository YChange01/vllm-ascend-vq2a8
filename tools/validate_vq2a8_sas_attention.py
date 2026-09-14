#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-free A5 SWA preflight: metadata -> packed KV scatter -> attention.

This checks the layer-0 execution path, not full DSA, FP8 roots or model quality.
"""

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
import json
import os
import sys
import time
from pathlib import Path

SAS_WORDS = 1024
FA_CORES, FD_CORES = 36, 72
FA_WORDS, FD_WORDS = 9, 8  # KV-quant SAS ABI; not QLI's 8/8-word ABI.
DEFINED_WORDS = FA_CORES * FA_WORDS + FD_CORES * FD_WORDS
BLOCK_SIZE, CACHE_PADDING = 128, 128


def check_sas_metadata(values, heads):
    """Check a short, single-request SWA schedule, with no flash-decode tasks."""
    if len(values) != SAS_WORDS or heads not in (64, 128):
        raise ValueError("SAS preflight expects 1024 int32 words and 64/128 query heads.")
    pair = 2 if heads == 128 else 1
    last_end, enabled, disabled = [0, 0, 0], 0, False
    for core in range(0, FA_CORES, pair):
        row = values[core * FA_WORDS : (core + 1) * FA_WORDS]
        if pair == 2 and row != values[(core + 1) * FA_WORDS : (core + 2) * FA_WORDS]:
            raise ValueError("SAS N128 paired Cube metadata must be identical.")
        if row[0] == 0:
            disabled = True
            # N128 idle pairs use word 8 for their cross-core barrier loop.
            if any(row[:8]) or (pair == 1 and row[8] != 0):
                raise ValueError("SAS disabled FA slot is uninitialized; rebuild custom AICPU ops.")
        elif row[0] == 1 and not disabled and row[1:4] == last_end and row[4:7] > last_end:
            last_end = row[4:7]
            enabled += pair
        else:
            raise ValueError("SAS FA intervals must be consecutive, with no gaps or overlaps.")
        if row[7] != 0:
            raise ValueError("SAS short-probe FA workspace index must be zero (no FD tasks).")
        if pair == 2 and not 0 <= row[8] <= BLOCK_SIZE:
            raise ValueError("SAS short-probe barrier loop count is invalid.")
    if not enabled or last_end != [1, 0, 0]:
        raise ValueError("SAS metadata does not cover the whole single request.")
    if any(values[FA_CORES * FA_WORDS : DEFINED_WORDS]):
        raise ValueError("SAS short-probe FD slots must all be disabled and initialized.")
    return enabled


def expected_swa_output(query_len, key_len, heads, head_dim):
    """Analytic oracle: zero Q, constant +/-0.5 V, one zero-logit sink.

    The values are exactly representable by the FP8 KV path. With zero query,
    all visible keys have logit zero, so the sink changes the divisor to L+1.
    """
    import torch

    if not 1 <= query_len <= key_len <= BLOCK_SIZE:
        raise ValueError("SAS analytic probe must fit one sliding-window block.")
    value = (torch.arange(head_dim) % 2).float() - 0.5
    visible = torch.arange(key_len - query_len + 1, key_len + 1).float()
    return (visible / (visible + 1))[:, None, None] * value[None, None, :].expand(query_len, heads, head_dim)


def run_sas_preflight(device, config, prompt_tokens=10):
    import importlib

    import torch

    from vllm_ascend.utils import bootstrap_custom_op_env

    heads, dim = config["num_attention_heads"], config["head_dim"]
    if (
        str(device) != "npu:0"
        or not 2 <= prompt_tokens <= 28
        or heads not in (64, 128)
        or dim != 512
        or config["qk_rope_head_dim"] != 64
        or config["sliding_window"] != 128
    ):
        raise ValueError("SAS preflight supports isolated TP1 A5, D=512, RoPE=64, SWA=128 and a short prompt.")
    bootstrap_custom_op_env()
    extension = importlib.import_module("vllm_ascend.vllm_ascend_C")
    print(
        "SAS_ENV "
        + json.dumps(
            {
                "extension": extension.__file__,
                "heads": heads,
                "head_dim": dim,
                "device": str(device),
                "defined_metadata_words": DEFINED_WORDS,
            }
        ),
        flush=True,
    )
    cases = [("prefill", prompt_tokens, prompt_tokens)]
    cases += [(f"decode{i}", 1, prompt_tokens + i) for i in range(1, 4)]
    results = []
    for name, qlen, klen in cases:
        print(f"SAS_START case={name} stage=prepare qlen={qlen} klen={klen}", flush=True)
        cuq = torch.tensor([0, qlen], dtype=torch.int32, device=device)
        cuk = torch.tensor([0, klen], dtype=torch.int32, device=device)
        usedq = torch.tensor([qlen], dtype=torch.int32, device=device)
        usedk = torch.tensor([klen], dtype=torch.int32, device=device)
        q = torch.zeros((qlen, heads, dim), dtype=torch.bfloat16, device=device)
        sinks = torch.zeros(heads, dtype=torch.float32, device=device)
        # Block 0 is reserved; exercise the same nonzero block-table indirection
        # and flat slot mapping used by A5DeviceAdaptor.dsa_kv_compress_scatter.
        table = torch.tensor([[1]], dtype=torch.int32, device=device)
        slots = torch.arange(BLOCK_SIZE, BLOCK_SIZE + klen, dtype=torch.int64, device=device)
        cache = torch.zeros((2, BLOCK_SIZE, 1, dim + CACHE_PADDING), dtype=torch.uint8, device=device)
        cache = cache.view(torch.float8_e4m3fn)
        value = (torch.arange(dim) % 2).float() - 0.5
        kv = value.expand(klen, dim).contiguous().to(dtype=torch.bfloat16, device=device)
        print(f"SAS_START case={name} stage=kv_scatter", flush=True)
        torch.ops._C_ascend.kv_compress_epilog(
            kv_compress_cache=cache.view(-1, 1, cache.shape[-1]),
            x=kv,
            slot_mapping=slots,
            quant_group_size=64,
            quant_mode=2,
            round_scale_flag=True,
            layout=1,
        )
        torch.npu.synchronize()
        expected = expected_swa_output(qlen, klen, heads, dim)
        previous_meta, previous_output = None, None
        max_error, active = 0.0, None
        start = time.perf_counter()
        for repeat in range(3):
            print(f"SAS_START case={name} repeat={repeat} stage=metadata", flush=True)
            metadata = torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata(
                num_heads_q=heads,
                num_heads_kv=1,
                head_dim=dim,
                kv_quant_mode=1,
                cu_seqlens_q=cuq,
                cu_seqlens_ori_kv=cuk,
                seqused_q=usedq,
                seqused_kv=usedk,
                batch_size=1,
                max_seqlen_q=qlen,
                max_seqlen_kv=klen,
                cmp_ratio=1,
                ori_mask_mode=4,
                ori_win_left=127,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=True,
                has_cmp_kv=False,
                device=str(device),
            )
            torch.npu.synchronize()
            if metadata.dtype != torch.int32 or metadata.shape != (SAS_WORDS,) or metadata.device != device:
                raise ValueError("SAS metadata has the wrong shape, dtype or device.")
            words = metadata.cpu().tolist()
            active = check_sas_metadata(words, heads)
            current_meta = words[:DEFINED_WORDS]  # reserved 124 words have no semantics
            if repeat and current_meta != previous_meta:
                raise ValueError("SAS defined metadata is not exactly repeatable.")
            previous_meta = current_meta
            print(f"SAS_START case={name} repeat={repeat} stage=attention active_fa_cores={active}", flush=True)
            # Match the production A5 PA_ND call: compute consumes seqused_kv
            # and the block table. Unlike metadata above, its tiling rejects
            # cu_seqlens_ori_kv (and cu_seqlens_cmp_kv / ori_sparse_indices).
            result = torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv(
                q,
                kv_quant_mode=1,
                ori_kv=cache,
                ori_block_table=table,
                cu_seqlens_q=cuq,
                seqused_kv=usedk,
                sinks=sinks,
                metadata=metadata,
                tile_size=64,
                rope_head_dim=64,
                softmax_scale=dim**-0.5,
                cmp_ratio=1,
                ori_mask_mode=4,
                ori_win_left=127,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
            )[0]
            torch.npu.synchronize()
            if result.shape != q.shape or result.dtype != q.dtype or result.device != device:
                raise ValueError("SAS attention has the wrong shape, dtype or device.")
            actual = result.float().cpu()
            error = float((actual - expected).abs().max())
            max_error = max(max_error, error)
            if not torch.isfinite(actual).all() or not torch.allclose(actual, expected, atol=0.002, rtol=0.005):
                raise ValueError(f"SAS analytic SWA/sink oracle failed: max_abs_error={error}.")
            if repeat and not torch.equal(actual, previous_output):
                raise ValueError("SAS attention output is not exactly repeatable.")
            previous_output = actual
        record = dict(
            case=name,
            passed=True,
            active_fa_cores=active,
            repeats=3,
            max_abs_error=max_error,
            elapsed_s=time.perf_counter() - start,
        )
        results.append(record)
        print("SAS_RESULT " + json.dumps(record), flush=True)
    print("SAS_ATTENTION_PREFLIGHT=PASS SCOPE=SWA_ONLY MODEL_VERIFIED=False", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-npu", type=int, default=4)
    parser.add_argument("--model", type=Path, required=True, help="Read config.json only; never load weights.")
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

    from tools.validate_vq2a8_qli_metadata import run_preflight
    from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device

    device = torch.device("npu:0")
    print("DEVICE " + json.dumps(_initialize_device(device)), flush=True)
    config = json.loads((args.model / "config.json").read_text())
    run_preflight(device, config)  # also records loaded extension/vendor library hashes
    run_sas_preflight(device, config)


if __name__ == "__main__":
    main()
