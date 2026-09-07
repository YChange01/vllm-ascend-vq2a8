#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect one small fused AscendC simulator trace from a pinned library.

No rebuild, model loading, numerical campaign or global CANN config changes.
Use a matching standalone suite, or explicitly select a diagnostic build that
has not yet passed that suite. Both modes require the native build manifest.
Simulation instruction/dataflow evidence requires review; it is not a hardware
benchmark. Vendor environment variables here are child-only profiler controls,
not new vLLM runtime settings.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.validate_vq2a8_ascendc import library_evidence  # noqa: E402
from tools.vq2a8_live_log import LiveChildLog  # noqa: E402

KERNEL_PREFIX = "vq2a8_ascendc_fused"
MAX_CSV_FILES = 32
MAX_CSV_BYTES = 128 * 1024 * 1024
MAX_SAMPLES_PER_KIND = 6
MAX_INVENTORY_FILES = 80
MAX_TIMEOUT_MINUTES = 60
PROGRESS_SECONDS = 30
PROGRESS_TAIL_BYTES = 4096


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def flush_levels(value):
    if isinstance(value, dict):
        return [v for k, v in value.items() if k == "flush_level"] + [
            level for k, v in value.items() if k != "flush_level" for level in flush_levels(v)
        ]
    if isinstance(value, list):
        return [level for item in value for level in flush_levels(item)]
    return []


def prepare_config(source, directory):
    """Patch a private copy, preserving all other simulator settings verbatim."""
    text = source.read_text()
    levels = flush_levels(json.loads(text))
    if not levels or any(type(v) is not int or v not in (2, 3) for v in levels):
        raise ValueError("Require a JSON simulator config with numeric flush_level 2 or 3.")
    updated, count = re.subn(r'("flush_level"\s*:\s*)[23](?=\s*[,}])', r"\g<1>2", text)
    if count != len(levels) or flush_levels(json.loads(updated)) != [2] * len(levels):
        raise ValueError("Could not isolate every flush_level setting safely.")
    directory.mkdir(mode=0o700)
    target = directory / "config.json"
    target.write_text(updated)
    target.chmod(0o600)
    return {
        "source": str(source),
        "source_sha256": digest(source),
        "private": str(target),
        "private_sha256": digest(target),
        "original_flush_levels": levels,
    }


def checked_suite(path, library):
    suite = json.loads(path.read_text())
    if suite.get("status") != "completed_review_pending" or suite.get("library", {}).get("sha256") != library["sha256"]:
        raise ValueError("Need a completed standalone suite for this exact library; do not rebuild for profiling.")
    if not suite.get("results") or any(row.get("status") != "passed" for row in suite["results"]):
        raise ValueError("Standalone suite contains missing/failed steps.")
    return {"path": str(path), "sha256": digest(path), "library_sha256": library["sha256"]}


def simulator_runtime_paths(maps):
    return sorted(
        {line.split()[-1] for line in maps.splitlines() if "/" in line and "libruntime_camodel.so" in line.split()[-1]}
    )


def run_application(args):
    # LD_PRELOAD must already have loaded the simulator at process startup.
    # Refuse an accidental bare-Python invocation before importing torch/NPU.
    mapped = simulator_runtime_paths(Path("/proc/self/maps").read_text())
    if not mapped:
        raise RuntimeError("No mapped libruntime_camodel.so; refusing to run a hardware fallback.")
    config_dir = os.environ.get("CAMODEL_CONFIG_PATH")
    if not config_dir:
        raise RuntimeError("Profiler did not preserve/provide CAMODEL_CONFIG_PATH.")
    config = (Path(config_dir) / "config.json").resolve(strict=True)
    if not config.is_relative_to(args.application_report.parent.resolve()):
        raise RuntimeError("Effective simulator config is outside this private report directory.")
    levels = flush_levels(json.loads(config.read_text()))
    if not levels or any(type(v) is not int or v != 2 for v in levels):
        raise RuntimeError("Effective simulator config must have flush_level=2.")
    library = library_evidence(args.library)
    if library["sha256"] != args.expected_library_sha256:
        raise ValueError("Library changed between profiler parent and application.")
    report = {
        "status": "starting",
        "library_sha256": library["sha256"],
        "simulator_runtime_paths": mapped,
        "config_path": str(config),
        "config_sha256": digest(config),
        "flush_levels": levels,
        "shape": {"m": 32, "n": 32, "k": 512, "tiles": 3},
        "kernel_prefix": KERNEL_PREFIX,
        "projection_calls": 0,
        "physical_npu_execution_claimed": False,
    }
    write_json(args.application_report, report)
    print("ASCENDC_SIM_APPLICATION " + json.dumps(report), flush=True)
    import torch
    import torch_npu  # noqa: F401

    from tools.validate_vq2a8_phase4_kernel import synthetic_inputs
    from vllm_ascend.quantization.vq2a8_ascendc import load_library, vq2a8_ascendc

    torch.set_num_threads(4)
    torch.npu.set_device(0)  # simulator logical device, not a physical-device selection
    load_library(args.library)
    # CPU construction, then six transfers. No preparation/oracle/repeat kernels.
    inputs = tuple(t.to("npu:0") for t in synthetic_inputs(32, 32, 512, 3))
    torch.npu.synchronize()
    output = vq2a8_ascendc(*inputs)  # exactly one native fused call, blockDim=1
    torch.npu.synchronize()
    # Do not print/read device values in simulator mode. Metadata only.
    report.update(
        status="completed",
        projection_calls=1,
        output_shape=list(output.shape),
        library_unchanged=digest(args.library) == library["sha256"],
    )
    write_json(args.application_report, report)
    print("ASCENDC_SIM_APPLICATION " + json.dumps(report), flush=True)
    return 0


def collect_instruction_csv(root):
    """Keep full CSVs on disk and print bounded samples including operands.

    Mnemonic/pipe matches are selection hints, never automated FP8 acceptance.
    CSV rows often aggregate repeated instructions, so row counts are not calls.
    """
    evidence = []
    for path in sorted(root.rglob("*instr_exe*.csv"))[:MAX_CSV_FILES]:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            continue
        entry = {"path": str(path), "bytes": path.stat().st_size, "rows": 0}
        evidence.append(entry)
        if entry["bytes"] > MAX_CSV_BYTES:
            entry["status"] = "too_large_for_summary"
            continue
        entry["sha256"] = digest(path)
        histogram, samples, counts, seen = Counter(), [], Counter(), set()
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            entry["columns"] = reader.fieldnames
            if not {"instr", "pipe", "detail"}.issubset(reader.fieldnames or []):
                entry["status"] = "unrecognized_columns"
                continue
            for row in reader:
                instr, pipe, detail = (row.get(k) or "" for k in ("instr", "pipe", "detail"))
                if not instr.strip():
                    continue
                entry["rows"] += 1
                histogram[(pipe, instr)] += 1
                kind = (
                    "sync"
                    if re.search(r"flag|event|barrier|wait|sync", instr, re.I)
                    else "matrix"
                    if pipe.upper() == "CUBE" or re.fullmatch(r"MMAD\w*", instr, re.I)
                    else "transfer"
                    if re.search(r"mte|copy|load|store|fixpipe", instr + " " + pipe, re.I)
                    else "other"
                )
                # One example per mnemonic/pipe: repeated PC/detail variants of
                # the first DMA must not hide later UB->L1 or output transfers.
                identity = (instr, pipe)
                if counts[kind] < MAX_SAMPLES_PER_KIND and identity not in seen:
                    seen.add(identity)
                    counts[kind] += 1
                    samples.append(
                        {
                            "kind": kind,
                            **{k: (str(v)[:2000] if v is not None else None) for k, v in row.items() if k is not None},
                        }
                    )
        entry.update(
            status="instruction_rows_collected" if entry["rows"] else "empty",
            histogram=[{"pipe": p, "instr": i, "rows": n} for (p, i), n in histogram.most_common(24)],
            samples=samples,
        )
    return evidence


def inspect_profiler_log(path):
    """Profiler exit 0 may mean partial traces parsed after an application abort."""
    result = {
        "application_timeout_reported": False,
        "runtime_error_count": 0,
        "errors": [],
        "shutdown_notice_seen": False,
    }
    with path.open(errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            if "SigIntHandler received signal" in line or "Model is terminating" in line:
                result["shutdown_notice_seen"] = True
            if "The timeout has reached" in line or "application will be forcibly killed" in line:
                result["application_timeout_reported"] = True
                result["shutdown_notice_seen"] = True
            if re.search(r"\[error\]", line, re.I):
                result["runtime_error_count"] += 1
                if len(result["errors"]) < 12:
                    result["errors"].append(
                        {
                            "line": number,
                            "text": line.strip()[:800],
                            "after_shutdown_notice": result["shutdown_notice_seen"],
                        }
                    )
    return result


def instruction_progress(directory):
    """Bounded raw-tail observation, not an automatic liveness or ISA verdict.

    CCU logs can grow while waiting, so observe instruction logs separately.
    A buffered log can also stay unchanged while simulation advances. File
    order is not necessarily time order; tails are not last-executed PCs.
    """
    rows = []
    for core in ("cubecore0", "veccore0", "veccore1"):
        matches = instruction_dump_paths(directory, core)
        if not matches:
            continue
        path = matches[-1]
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            continue
        try:
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(max(0, size - PROGRESS_TAIL_BYTES))
                tail = stream.read(PROGRESS_TAIL_BYTES).decode("utf-8", errors="replace").splitlines()
            rows.append({"core": core, "bytes": size, "tail": [line[:600] for line in tail[-2:]]})
        except OSError:
            continue  # profiler may not have created/flushed the dump yet
    return rows


def instruction_dump_paths(directory, core):
    """Find both pre-export nested and post-export flat instruction dumps."""
    if core not in ("cubecore0", "veccore0", "veccore1"):
        raise ValueError("Expected core0's Cube or one of its two Vector cores.")
    return sorted(
        p
        for p in (directory / "profile").glob(f"OPPROF*/**/dump/core0.{core}.instr_log*.dump")
        if re.fullmatch(rf"core0\.{core}\.instr_log(?:\.\d+)?\.dump", p.name)
        and p.is_file()
        and not p.is_symlink()
        and p.resolve().is_relative_to(directory.resolve())
    )


def profiler_command(msprof, args, directory, library_sha):
    return [
        msprof,
        "op",
        "simulator",
        f"--soc-version={args.soc}",
        f"--output={directory / 'profile'}",
        f"--kernel-name={KERNEL_PREFIX}",
        "--launch-count=1",
        f"--timeout={args.timeout_minutes}",
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--application-report",
        str(directory / "application.json"),
        "--library",
        str(args.library),
        "--expected-library-sha256",
        library_sha,
    ]


def run_profiler(command, directory, environment, timeout_seconds):
    log = directory / "profiler.log"
    # A deadline covers parsing too; kill only this newly created process group.
    with log.open("w") as stream, LiveChildLog(log, "ascendc-simulator"):
        child = subprocess.Popen(
            command, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        started = time.monotonic()
        deadline = started + timeout_seconds
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout_seconds)
                try:
                    return {"exit": child.wait(timeout=min(PROGRESS_SECONDS, remaining)), "timeout": False}
                except subprocess.TimeoutExpired:
                    # Lost console/dump access must not abandon the live child
                    # or alter the bounded execution/cleanup contract.
                    with contextlib.suppress(OSError):
                        print(
                            "ASCENDC_SIM_PROGRESS "
                            + json.dumps(
                                {
                                    "elapsed_s": round(time.monotonic() - started, 1),
                                    "instruction_logs": instruction_progress(directory),
                                    "note": "raw observations; log growth alone does not prove forward progress",
                                }
                            ),
                            flush=True,
                        )
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            return {"exit": child.returncode, "timeout": True}


def run(args):
    if sys.platform != "linux":
        raise RuntimeError("Run on the Linux CANN installation, not the development host.")
    args.library = args.library.resolve(strict=True)
    library = library_evidence(args.library)
    # A rebuilt diagnostic candidate cannot inherit the old library's suite.
    suite = None if args.diagnostic_build else checked_suite(args.suite_report / "summary.json", library)
    msprof = shutil.which("msprof")
    if not msprof:
        raise RuntimeError("msprof is not on PATH; source the existing CANN environment first.")
    config = (args.cann / "tools/simulator" / args.soc / "lib/config.json").resolve(strict=True)
    config_sha = digest(config)
    directory = Path(tempfile.mkdtemp(prefix="vq2a8-ascendc-sim-")).resolve()
    print(f"ASCENDC_SIM_REPORT={directory}", flush=True)
    report = {
        "status": "running",
        "scope": "single_synthetic_fused_simulation",
        "library": library,
        "suite": suite,
        "diagnostic_build": args.diagnostic_build,
        "matching_standalone_suite_supplied": suite is not None,
        "harness_sha256": digest(Path(__file__)),
        "native_instruction_verified": False,
        "on_chip_decode_verified": False,
        "performance_verified": False,
        "model_integration_verified": False,
        "default_model_backend": "unchanged",
    }
    manifest = directory / "summary.json"
    write_json(manifest, report)
    try:
        report["config"] = prepare_config(config, directory / "config")
        version = subprocess.run(
            [msprof, "op", "simulator", "--version"], capture_output=True, text=True, timeout=30, check=False
        )
        report["profiler_version"] = {"exit": version.returncode, "output": (version.stdout + version.stderr)[:4000]}
        command = profiler_command(msprof, args, directory, library["sha256"])
        report["command"] = command
        print("ASCENDC_SIM_COMMAND " + json.dumps(command), flush=True)
        # Vendor profiler control; no process-global or toolkit file mutation.
        environment = {**os.environ, "CAMODEL_CONFIG_PATH": str(directory / "config")}
        write_json(manifest, report)
        report["profiler"] = run_profiler(command, directory, environment, args.timeout_minutes * 60 + 120)
        report["profiler_log_review"] = inspect_profiler_log(directory / "profiler.log")
        application = directory / "application.json"
        report["application"] = json.loads(application.read_text()) if application.is_file() else {}
        report["instruction_csv"] = collect_instruction_csv(directory / "profile")
        report["profile_files"] = [
            {"path": str(p.relative_to(directory)), "bytes": p.stat().st_size}
            for p in sorted((directory / "profile").rglob("*"))
            if p.is_file() and not p.is_symlink()
        ][:MAX_INVENTORY_FILES]
        app = report["application"]
        collected = [r for r in report["instruction_csv"] if r["status"] == "instruction_rows_collected"]
        # Require at least an AIC and two AIV report paths, but never infer dataflow
        # correctness or complete coverage from file counts alone.
        report["core_reports_present"] = (
            any("cubecore" in r["path"] for r in collected)
            and len({str(Path(r["path"]).parent) for r in collected if "veccore" in r["path"]}) >= 2
        )
        complete = (
            report["profiler"] == {"exit": 0, "timeout": False}
            and not report["profiler_log_review"]["application_timeout_reported"]
            and not report["profiler_log_review"]["runtime_error_count"]
            and app.get("status") == "completed"
            and app.get("projection_calls") == 1
            and app.get("library_sha256") == library["sha256"]
            and app.get("library_unchanged") is True
            and report["core_reports_present"]
            and all(r["status"] == "instruction_rows_collected" for r in report["instruction_csv"])
        )
        report["status"] = "collected_review_pending" if complete else "incomplete_review_pending"
    except (OSError, ValueError, RuntimeError, csv.Error, subprocess.TimeoutExpired) as exc:
        report.update(status="incomplete_review_pending", error=str(exc))
    finally:
        try:
            report["global_config_unchanged"] = digest(config) == config_sha
        except OSError:
            report["global_config_unchanged"] = False
        if not report["global_config_unchanged"]:
            report["status"] = "incomplete_review_pending"
        write_json(manifest, report)
    for entry in report.get("instruction_csv", []):
        print("ASCENDC_SIM_CSV " + json.dumps(entry), flush=True)
    print("ASCENDC_SIM_FILES " + json.dumps(report.get("profile_files", [])), flush=True)
    print("ASCENDC_SIM_LOG_REVIEW " + json.dumps(report.get("profiler_log_review", {})), flush=True)
    print(
        f"ASCENDC_SIM={report['status']} GLOBAL_CONFIG_UNCHANGED={report['global_config_unchanged']} REPORT={directory}"
    )
    if report.get("error"):
        print("ERROR=" + report["error"])
    print("NATIVE_INSTRUCTION_VERIFIED=False ON_CHIP_DECODE_VERIFIED=False MODEL_INTEGRATION_VERIFIED=False")
    return 0 if report["status"] == "collected_review_pending" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    evidence = parser.add_mutually_exclusive_group()
    evidence.add_argument("--suite-report", type=Path, help="Existing successful standalone suite directory.")
    evidence.add_argument(
        "--diagnostic-build",
        action="store_true",
        help="Trace a freshly built candidate without claiming prior numerical acceptance.",
    )
    parser.add_argument("--library", type=Path, default=REPO / "build/vq2a8-ascendc/libvq2a8_ascendc.so")
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann-9.1.0"))
    parser.add_argument("--soc", default="Ascend950PR_957d")
    parser.add_argument(
        "--timeout-minutes",
        type=int,
        default=5,
        help="Simulator limit, 1..60 minutes (default: 5); scalar decode may need a longer explicit limit.",
    )
    parser.add_argument("--application-report", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-library-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.timeout_minutes <= MAX_TIMEOUT_MINUTES or not re.fullmatch(r"Ascend950[A-Za-z0-9_]+", args.soc):
        parser.error(f"Require Ascend950 SOC and 1..{MAX_TIMEOUT_MINUTES} simulator timeout minutes.")
    if args.application_report:
        return run_application(args)
    if not args.suite_report and not args.diagnostic_build:
        parser.error("Require --suite-report or explicit --diagnostic-build; no implicit acceptance bypass.")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
