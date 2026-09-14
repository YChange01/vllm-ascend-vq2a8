#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate TP1 startup operators with synthetic tensors, never model weights.

Each case gets a fresh child, a disk log, periodic Python stacks and a deadline.
Stop on the first failed/expired case; never reset a device or kill another job.
"""

from __future__ import annotations

# Keep tools/bisect from shadowing the standard library in direct execution.
# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import faulthandler
import json
import re
import signal
import subprocess
import tempfile
import time
import traceback
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from tools.vq2a8_live_log import LiveChildLog

REPO = Path(__file__).resolve().parents[1]
CASES = (
    "basic",
    "hc_pre_m2",
    "hc_pre_m128",
    "hc_post_m128",
    "prepare_m32",
    "prepare_group6",
    "projection_m1",
    "projection_m32",
    "projection_group6",
)
PREFIX = "STARTUP_PROBE "
SCOPE = "synthetic_component_probes_only_no_model_or_artifact_load"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-npu", type=int, default=1)
    parser.add_argument("--library", type=Path, default=REPO / "build/vq2a8-ascendc-v3/libvq2a8_ascendc_v3.so")
    parser.add_argument("--cases", default=",".join(CASES), help="comma-separated subset, in execution order")
    parser.add_argument(
        "--timeout-s", type=int, default=120, help="whole-child deadline for EACH case, including imports"
    )
    parser.add_argument(
        "--allow-busy", action="store_true", help="allow a known occupied device; may affect other jobs"
    )
    parser.add_argument(
        "--launch-blocking", choices=("0", "1"), default="1", help="child only; stages always synchronize"
    )
    parser.add_argument("--report-dir", type=Path, help="new output directory; default: unique directory under /tmp")
    parser.add_argument("--plan-only", action="store_true", help="no imports of torch/vLLM, files or device operations")
    parser.add_argument("--child", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.plan_only and args.child:
        parser.error("--plan-only cannot be combined with the internal --child option")
    args.cases = args.cases.split(",")
    if args.physical_npu < 0 or args.timeout_s <= 0:
        parser.error("Require --physical-npu >= 0 and --timeout-s > 0")
    if len(set(args.cases)) != len(args.cases) or any(case not in CASES for case in args.cases):
        parser.error("--cases requires unique names from: " + ",".join(CASES))
    return args


def child_environment(args, environ=None):
    result = dict(os.environ if environ is None else environ)
    for key in (
        "ASCEND_VISIBLE_DEVICES",
        "NPU_VISIBLE_DEVICES",
        "ASCEND_DEVICE_ID",
        "DEVICE_ID",
        "RANK_ID",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        result.pop(key, None)
    result["ASCEND_RT_VISIBLE_DEVICES"] = str(args.physical_npu)
    result["ASCEND_LAUNCH_BLOCKING"] = args.launch_blocking
    result["PYTHONUNBUFFERED"] = "1"
    result["PYTHONPATH"] = str(REPO) + (os.pathsep + result["PYTHONPATH"] if result.get("PYTHONPATH") else "")
    return result


def child_command(args, case):
    return [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--child",
        case,
        "--physical-npu",
        str(args.physical_npu),
        "--library",
        str(args.library.resolve()),
        "--timeout-s",
        str(args.timeout_s),
        "--launch-blocking",
        args.launch_blocking,
    ]


def emit(case, event, **fields):
    print(PREFIX + json.dumps({"case": case, "event": event, **fields}, ensure_ascii=False), flush=True)


def stage_recorder(case, synchronize):
    @contextmanager
    def stage(name):
        started = time.monotonic()
        emit(case, "BEGIN", stage=name)
        try:
            yield
            # Distinguish an op/API that does not return from a subsequent wait.
            emit(case, "SUBMITTED", stage=name, elapsed_s=round(time.monotonic() - started, 6))
            synchronize()
        except BaseException as exc:
            emit(case, "FAIL", stage=name, error=str(exc), elapsed_s=round(time.monotonic() - started, 6))
            raise
        emit(case, "PASS", stage=name, elapsed_s=round(time.monotonic() - started, 6))

    return stage


def read_events(log):
    events = []
    with Path(log).open(encoding="utf-8", errors="replace") as source:
        for line in source:
            if line.startswith(PREFIX):
                try:
                    record = json.loads(line[len(PREFIX) :])
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    events.append(record)
    return events


def parse_snapshot(text, physical_npu):
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if "NPU ID" in line and "Process id" in line), None)
    if start is None:
        return "unknown"
    table = "\n".join(lines[start:])
    idle = {int(x) for x in re.findall(r"No running processes found in NPU\s+(\d+)\s*\|", table)}
    busy = {int(x) for x in re.findall(r"(?m)^\s*\|\s*(\d+)\s*\|\s*\d+\s*\|", table)}
    if physical_npu in idle and physical_npu not in busy:
        return "idle"
    if physical_npu in busy and physical_npu not in idle:
        return "busy"
    return "unknown"


def terminate_child(process):
    """Bounded cleanup of our child/session only; no device reset or broad pkill."""
    if process.poll() is not None:
        return True
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=3)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            return process.poll() is not None
    except ProcessLookupError:
        return process.poll() is not None
    return True


def run_child(command, environment, log, timeout_s):
    started = time.monotonic()
    timed_out, reaped = False, True
    with Path(log).open("x", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            env=environment,
            cwd=REPO,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        with LiveChildLog(Path(log), "tp1-startup"):
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                reaped = terminate_child(process)
            except BaseException:
                terminate_child(process)
                raise
    events = read_events(log)
    completed = any(event.get("event") == "CASE_PASS" for event in events)
    status = "TIMEOUT" if timed_out else "PASS" if process.returncode == 0 and completed else "FAIL"
    return {
        "status": status,
        "exit_code": process.returncode,
        "reaped": reaped,
        "elapsed_s": round(time.monotonic() - started, 3),
        "log": str(log),
        "last_event": events[-1] if events else None,
        "events": events,
    }


def run_case_child(args):
    # This branch is the ONLY location where NPU dependencies are imported.
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.physical_npu):
        raise ValueError("Child device mapping differs from the selected physical NPU")
    faulthandler.enable()
    faulthandler.dump_traceback_later(min(30, max(1, args.timeout_s / 2)), repeat=True)

    def sync():
        pass

    stage = stage_recorder(args.child, lambda: sync())
    try:
        with stage("imports"):
            import torch
            import torch_npu  # noqa: F401

        with stage("device_init"):
            if torch.npu.device_count() != 1:
                raise RuntimeError("Expected exactly one visible device; refusing ambiguous device mapping")
            torch.set_num_threads(4)
            torch.npu.set_device(0)
            torch.npu.config.allow_internal_format = False
            sync = torch.npu.synchronize
        with stage("device_info"):
            packages = {}
            for name in ("torch", "torch-npu", "vllm", "vllm-ascend"):
                try:
                    packages[name] = version(name)
                except PackageNotFoundError:
                    packages[name] = "not-installed"
            emit(
                args.child,
                "INFO",
                physical_npu=args.physical_npu,
                logical_npu=0,
                soc=torch.npu.get_device_name(0),
                packages=packages,
                launch_blocking=os.environ.get("ASCEND_LAUNCH_BLOCKING"),
                free_total_bytes=list(torch.npu.mem_get_info()),
                allow_internal_format=False,
            )
        if args.child.startswith("projection_"):
            from tools.vq2a8_startup_projection import run_case

            metrics = run_case(args.child, stage, args.library)
        else:
            from tools.vq2a8_startup_ops import run_case

            metrics = run_case(args.child, stage)
        with stage("final_sync"):
            pass
        emit(
            args.child, "CASE_PASS", metrics=metrics, peak_allocated_bytes=torch.npu.max_memory_allocated(), scope=SCOPE
        )
        return 0
    except Exception as exc:
        traceback.print_exc()
        emit(args.child, "CASE_FAIL", error=str(exc))
        return 1
    finally:
        faulthandler.cancel_dump_traceback_later()


def main(argv=None):
    args = parse_args(argv)
    if args.child:
        return run_case_child(args)
    plan = {
        "scope": SCOPE,
        "physical_npu": args.physical_npu,
        "cases": args.cases,
        "timeout_s_per_case": args.timeout_s,
        "launch_blocking": args.launch_blocking,
        "commands": [child_command(args, case) for case in args.cases],
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    if os.name != "posix":
        raise RuntimeError("NPU probes require Linux; use --plan-only on other platforms")
    if args.report_dir is None:
        report_dir = Path(tempfile.mkdtemp(prefix="vq2-tp1-startup-"))
    else:
        report_dir = args.report_dir.resolve()
        report_dir.mkdir(parents=True, exist_ok=False)
    print(f"REPORT={report_dir}", flush=True)
    print(
        "Synthetic tensors only; NPU runtime still uses HBM. Timings are diagnostic, not model throughput.", flush=True
    )
    report = {**plan, "status": "RUNNING", "results": []}
    try:
        for case in args.cases:
            snapshot_path = report_dir / f"{case}-npu.log"
            try:
                snapshot = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=20, check=True)
                snapshot_path.write_text(snapshot.stdout + snapshot.stderr, encoding="utf-8")
                state = parse_snapshot(snapshot.stdout, args.physical_npu)
            except (OSError, subprocess.SubprocessError) as exc:
                snapshot_path.write_text(str(exc), encoding="utf-8")
                state = "unknown"
            if state == "unknown" or (state == "busy" and not args.allow_busy):
                report["results"].append({"case": case, "status": "BLOCKED", "device_state": state})
                report["status"] = "BLOCKED"
                print(f"STOP={case} device_state={state}; --allow-busy only overrides known occupancy.", flush=True)
                break
            if state == "busy":
                print("WARNING: shared NPU; the probe may affect existing jobs or fail from contention.", flush=True)
            result = run_child(
                child_command(args, case), child_environment(args), report_dir / f"{case}.log", args.timeout_s
            )
            report["results"].append({"case": case, "device_state": state, **result})
            print(f"CASE={case} STATUS={result['status']} LOG={result['log']}", flush=True)
            if result["status"] != "PASS":
                report["status"] = result["status"]
                print(
                    "STOP: no further NPU work. Timeout does not prove kernel cancellation or device recovery.",
                    flush=True,
                )
                break
        else:
            report["status"] = "PASS"
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
    except Exception as exc:
        report["status"], report["error"] = "FAIL", str(exc)
        traceback.print_exc()
    finally:
        (report_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"SUMMARY={report_dir / 'summary.json'} STATUS={report['status']}", flush=True)
    return 0 if report["status"] == "PASS" else 130 if report["status"] == "INTERRUPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
