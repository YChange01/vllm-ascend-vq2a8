#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-NPU vLLM/HCCL + optional pinned V3 local-projection smoke; no model load."""

from __future__ import annotations

# Prevent direct execution from allowing tools/bisect to shadow the stdlib.
# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import hashlib
import json
import re
import shlex
import threading
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCAL_CASES = (("gate_up", 2048, 4096, 4096), ("down", 4096, 2048, 1024))


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=REPO / "build/vq2a8-ascendc-v3/libvq2a8_ascendc_v3.so")
    parser.add_argument("--library-sha256", help="required exact lowercase SHA-256 unless --communication-only")
    parser.add_argument("--communication-only", action="store_true", help="do not load any compiled library")
    parser.add_argument(
        "--timeout-s", type=positive_integer, default=120, help="whole-process watchdog and HCCL/Gloo timeout"
    )
    parser.add_argument(
        "--report-dir", type=Path, help="optional new rank_0.json/rank_1.json; existing reports are never overwritten"
    )
    parser.add_argument(
        "--plan-only", action="store_true", help="print the torchrun command without importing torch/vLLM/NPU"
    )
    args = parser.parse_args(argv)
    if args.library_sha256 is not None and re.fullmatch(r"[0-9a-f]{64}", args.library_sha256) is None:
        parser.error("--library-sha256 must be 64 lowercase hexadecimal characters")
    if not args.communication_only and not args.plan_only and args.library_sha256 is None:
        parser.error("--library-sha256 is required for native projection smoke")
    return args


def build_command(args):
    command = [
        "torchrun",
        "--standalone",
        "--nproc-per-node=2",
        str(Path(__file__).resolve()),
        "--timeout-s",
        str(args.timeout_s),
    ]
    if args.communication_only:
        command.append("--communication-only")
    else:
        command.extend(
            ("--library", str(args.library), "--library-sha256", args.library_sha256 or "<SHA256_OF_SELECTED_V3_SO>")
        )
    if args.report_dir is not None:
        command.extend(("--report-dir", str(args.report_dir)))
    return command


def launch_environment(environ):
    values = {}
    for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK"):
        raw = environ.get(key, "")
        if re.fullmatch(r"0|[1-9][0-9]*", raw) is None:
            raise ValueError(f"Missing/invalid torchrun {key}; launch with --standalone --nproc-per-node=2.")
        values[key] = int(raw)
    if values["WORLD_SIZE"] != 2 or values["LOCAL_WORLD_SIZE"] != 2:
        raise ValueError("This smoke requires exactly two ranks on one node.")
    if values["RANK"] not in (0, 1) or values["LOCAL_RANK"] != values["RANK"]:
        raise ValueError("Expected rank == local_rank in {0,1} for standalone TP2.")
    visible = environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if re.fullmatch(r"(?:0|[1-9][0-9]*),(?:0|[1-9][0-9]*)", visible) is None:
        raise ValueError("Set ASCEND_RT_VISIBLE_DEVICES to exactly two distinct physical NPU IDs, e.g. 0,1.")
    physical = tuple(int(value) for value in visible.split(","))
    if physical[0] == physical[1]:
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES must select two distinct devices.")
    if not environ.get("MASTER_ADDR") or not environ.get("MASTER_PORT"):
        raise ValueError("torchrun must provide MASTER_ADDR and MASTER_PORT.")
    return {
        "rank": values["RANK"],
        "local_rank": values["LOCAL_RANK"],
        "world_size": 2,
        "visible_devices": visible,
        "physical_npu": physical[values["LOCAL_RANK"]],
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage(name, rank, **fields):
    print("TP2_SMOKE_STAGE=" + name + " " + json.dumps({"rank": rank, **fields}, sort_keys=True), flush=True)


@contextmanager
def deadline(seconds, rank):
    def abort():
        print(f"TP2_SMOKE_STATUS=TIMEOUT rank={rank} timeout_s={seconds}", file=sys.stderr, flush=True)
        os._exit(124)

    timer = threading.Timer(seconds, abort)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


@contextmanager
def tp_environment(launch, timeout_s, *, config_module, parallel_state):
    """Initialize vLLM 0.23 TP without constructing or downloading a model."""
    parallel = config_module.ParallelConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        distributed_executor_backend="external_launcher",
        distributed_timeout_seconds=timeout_s,
        cpu_distributed_timeout_seconds=timeout_s,
        disable_custom_all_reduce=True,
        rank=launch["rank"],
    )
    # vLLM 0.23 intentionally leaves model_config's default None unvalidated.
    # Passing None explicitly fails Pydantic's non-Optional ModelConfig field;
    # constructing ModelConfig() instead can trigger a model config download.
    config = config_module.VllmConfig(parallel_config=parallel)
    with config_module.set_current_vllm_config(config):
        try:
            parallel_state.init_distributed_environment(
                world_size=2,
                rank=launch["rank"],
                local_rank=launch["local_rank"],
                distributed_init_method="env://",
                backend="hccl",
                timeout=timedelta(seconds=timeout_s),
            )
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=2, pipeline_model_parallel_size=1, backend="hccl"
            )
            yield parallel_state.get_tp_group()
        finally:
            try:
                parallel_state.destroy_model_parallel()
            finally:
                parallel_state.destroy_distributed_environment()


def validate_group(group, launch, *, backend, current_device):
    if (group.world_size, group.rank, group.local_rank, group.rank_in_group, list(group.ranks)) != (
        2,
        launch["rank"],
        launch["local_rank"],
        launch["rank"],
        [0, 1],
    ):
        raise RuntimeError("vLLM TP group rank mapping is not the requested standalone TP2 mapping.")
    if str(backend).lower() != "hccl":
        raise RuntimeError(f"Expected actual HCCL backend, not {backend!r}; Gloo fallback cannot pass this smoke.")
    if (
        group.device.type != "npu"
        or group.device.index != launch["local_rank"]
        or current_device != launch["local_rank"]
    ):
        raise RuntimeError("TP group/device mapping does not match torchrun LOCAL_RANK.")


def _exact(torch, actual, expected, label, *, dtype):
    if actual.dtype != dtype or tuple(actual.shape) != tuple(expected.shape):
        raise RuntimeError(f"{label}: output shape/dtype mismatch.")
    host = actual.detach().cpu()
    if not bool(torch.isfinite(host).all()) or not torch.equal(host, expected):
        raise RuntimeError(f"{label}: finite/exact numeric comparison failed.")


def communication_smoke(torch, group, device, rank):
    partial = torch.full((1, 8), float(3 * (rank + 1)), dtype=torch.float32, device=device)
    # This is the sole routed SUM. Shared output is added only AFTER it.
    routed = group.all_reduce(partial)
    _exact(torch, routed, torch.full((1, 8), 9.0, device="cpu"), "routed FP32 SUM", dtype=torch.float32)
    final = routed + 5.0
    _exact(torch, final, torch.full((1, 8), 14.0, device="cpu"), "shared added once", dtype=torch.float32)
    return {
        "routed_all_reduce_calls": 1,
        "dtype": "float32",
        "rank_partial": 3 * (rank + 1),
        "routed_sum": 9,
        "shared_value": 5,
        "shared_added_after_reduce": True,
        "final_value": 14,
        "exact": True,
    }


def synthetic_projection(torch, kind, rank):
    """Prepared FP8 input and analytic oracle; no RHT/A8-preparation claim."""
    _, n, k, logical_k = next(case for case in LOCAL_CASES if case[0] == kind)
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError("Synthetic rank must be integer 0/1.")
    x = torch.zeros((1, k), dtype=torch.float32, device="cpu")
    x[0, :logical_k:256] = rank + 1
    x[0, 127:logical_k:256] = rank + 1
    x = x.to(torch.float8_e4m3fn)
    scale, bias = (
        torch.tensor([0.125], dtype=torch.float32, device="cpu"),
        torch.tensor([(rank + 1) / 2], dtype=torch.float32, device="cpu"),
    )
    packed = torch.full((n // 32, k // 16, 16, 8), 0x21, dtype=torch.uint8, device="cpu")
    lut_rows = (torch.arange(n // 32, device="cpu") % 3 - 1).float() / 2
    entries = torch.arange(32, device="cpu")
    values = (entries // 2 - 8).float() / 4 + (entries % 2).float() / 4
    lut = (
        (lut_rows[:, None] + values[None, :])
        .expand(k // 256, -1, -1)
        .contiguous()
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    rows = torch.arange(n, device="cpu")
    codes = 1 + (rows // 2) % 2  # 0x21 stores alternating codeword 1/2 per output pair.
    weight = (codes - 8).float() / 4 + ((rows // 32) % 3 - 1).float() / 2 + (rows % 2).float() / 4
    expected = (weight[None, :] * (2 * (logical_k // 256) * (rank + 1)) * scale[:, None] + bias[:, None]).to(
        torch.bfloat16
    )
    return (x, scale, bias, packed, lut), expected


def projection_smoke(torch, group, device, rank, projector):
    records = []
    for kind, n, k, logical_k in LOCAL_CASES:
        _stage("native_" + kind, rank, n=n, k=k, logical_k=logical_k)
        host, expected = synthetic_projection(torch, kind, rank)
        inputs = tuple(tensor.to(device=device) for tensor in host)
        actual = projector([inputs], tp_size=2)[0]
        _exact(torch, actual, expected, kind, dtype=torch.bfloat16)
        repeated = projector([inputs], tp_size=2)[0]
        _exact(torch, repeated, expected, kind + " repeat", dtype=torch.bfloat16)
        record = {
            "kind": kind,
            "n": n,
            "k": k,
            "logical_k": logical_k,
            "padding_k": k - logical_k,
            "native_calls": 2,
            "local_exact": True,
            "repeat_exact": True,
            "tp_sum_exact": None,
        }
        if kind == "down":
            reduced = group.all_reduce(actual.float())
            golden = sum(synthetic_projection(torch, kind, other)[1].float() for other in (0, 1))
            _exact(torch, reduced, golden, "down FP32 TP SUM", dtype=torch.float32)
            record.update(tp_sum_exact=True, down_all_reduce_calls=1)
        records.append(record)
    return records


def validate_rank_reports(records, *, communication_only):
    if (
        len(records) != 2
        or any(not isinstance(record, dict) or type(record.get("rank")) is not int for record in records)
        or {record["rank"] for record in records} != {0, 1}
    ):
        raise RuntimeError("Missing/duplicate rank completion evidence.")
    for record in records:
        if record.get("communication_verified") is not True or record.get("native_projection_verified") is not (
            not communication_only
        ):
            raise RuntimeError("One rank did not complete the requested smoke checks.")


def run(args, launch):
    _stage("imports", launch["rank"])
    import torch
    import torch_npu
    import vllm
    import vllm.config as config_module
    import vllm.envs as vllm_envs
    from vllm.distributed import parallel_state
    from vllm.platforms import current_platform

    if not torch.npu.is_available() or torch.npu.device_count() != 2:
        raise RuntimeError("Two visible NPUs are required; no CPU/mock fallback can pass.")
    if current_platform.device_type != "npu":
        raise RuntimeError("vLLM Ascend platform plugin is not active.")
    if vllm_envs.VLLM_DISTRIBUTED_USE_SPLIT_GROUP:
        raise RuntimeError(
            "This pinned HCCL smoke requires vLLM's default new_group path; unset VLLM_DISTRIBUTED_USE_SPLIT_GROUP."
        )
    if not torch.distributed.is_backend_available("hccl"):
        raise RuntimeError("HCCL backend unavailable; refusing vLLM's Gloo fallback.")
    torch.npu.set_device(launch["local_rank"])
    device = torch.device(f"npu:{launch['local_rank']}")
    _stage("device", launch["rank"], physical_npu=launch["physical_npu"], logical_npu=launch["local_rank"])
    library, capabilities, wrapper_sha = None, None, None
    projector = None
    if not args.communication_only:
        from vllm_ascend.quantization import vq2a8_ascendc_v3 as wrapper

        _stage("pinned_library", launch["rank"])
        library = wrapper.load_pinned_library(args.library, args.library_sha256)
        capabilities = wrapper.resident_library_capabilities(require_tp2=True)
        wrapper_sha = _sha256(wrapper.__file__)
        projector = wrapper.grouped_projection_resident
    _stage("distributed_init", launch["rank"], timeout_s=args.timeout_s)
    with tp_environment(launch, args.timeout_s, config_module=config_module, parallel_state=parallel_state) as group:
        validate_group(
            group,
            launch,
            backend=torch.distributed.get_backend(group.device_group),
            current_device=torch.npu.current_device(),
        )
        _stage("routed_sum", launch["rank"])
        communication = communication_smoke(torch, group, device, launch["rank"])
        native = [] if args.communication_only else projection_smoke(torch, group, device, launch["rank"], projector)
        completion = {
            "rank": launch["rank"],
            "communication_verified": True,
            "native_projection_verified": not args.communication_only,
        }
        records = [None, None]
        # CPU completion coordination is separate from the one routed device SUM.
        torch.distributed.all_gather_object(records, completion, group=group.cpu_group)
        validate_rank_reports(records, communication_only=args.communication_only)
        torch.npu.synchronize()
        _stage("checks_complete", launch["rank"])
    return {
        "schema_version": 1,
        "status": "PASS",
        "scope": "two_npu_communication_only"
        if args.communication_only
        else "two_npu_communication_and_synthetic_local_projection",
        **launch,
        "timeout_s": args.timeout_s,
        "communication_verified": True,
        "native_projection_verified": not args.communication_only,
        "communication": communication,
        "native_cases": native,
        "rank_completion": records,
        "library": library,
        "resident_capabilities": capabilities,
        "software": {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "vllm": vllm.__version__},
        "source_observations": {
            "tool_sha256": _sha256(__file__),
            "vllm_parallel_state_sha256": _sha256(parallel_state.__file__),
            "wrapper_sha256": wrapper_sha,
        },
        "local_rht_a8_preparation_verified": False,
        "model_integration_verified": False,
        "model_quality_verified": False,
        "full_model_graph_verified": False,
        "performance_verified": False,
    }


def main(argv=None):
    args = parse_args(argv)
    if args.plan_only:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY",
                    "command": "ASCEND_RT_VISIBLE_DEVICES=0,1 " + shlex.join(build_command(args)),
                    "communication_verified": False,
                    "native_projection_verified": False,
                    "model_loaded": False,
                },
                sort_keys=True,
            )
        )
        return 0
    try:
        launch = launch_environment(os.environ)
        report_path = None if args.report_dir is None else args.report_dir / f"rank_{launch['rank']}.json"
        if report_path is not None and (report_path.exists() or report_path.is_symlink()):
            raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")
        with deadline(args.timeout_s, launch["rank"]):
            report = run(args, launch)
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, sort_keys=True)
                stream.write("\n")
        print("TP2_SMOKE_RESULT=" + json.dumps(report, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        print(
            "TP2_SMOKE_STATUS=FAIL " + json.dumps({"rank": os.environ.get("RANK"), "error": str(error)}),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
