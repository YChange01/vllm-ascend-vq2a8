#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One supervised run for work packages 2 and 4 ONLY: offline perf + native evidence.

No rebuild, installation, repack, HTTP listener, quality suite or device security
change. Missing evidence never becomes PASS. Use --plan-only before execution.
"""

from __future__ import annotations

# ruff: noqa: E402
import os as _bootstrap_os
import sys as _bootstrap_sys

if not __package__:
    _bootstrap_sys.path[0] = _bootstrap_os.path.dirname(
        _bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))
    )

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from tools.profile_vq2a8_ascendc import digest, write_json
from tools.validate_vq2a8_ascendc import library_evidence
from tools.validate_vq2a8_tp1_acceptance import acceptance_environment
from tools.vq2a8_baseline import capture_input_identity
from tools.vq2a8_live_log import LiveChildLog
from tools.vq2a8_native_review import apply_review, index_native_reports
from tools.vq2a8_perf_report import summarize_performance, validate_cases

REPO = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "torch-npu", "vllm", "vllm-ascend", "triton-ascend", "numpy", "transformers", "safetensors")


def plan(args):
    cases = validate_cases([tuple(map(int, v.split(":"))) for v in args.cases.split(",")])
    if (
        not math.isfinite(args.cache_budget_gib)
        or args.cache_budget_gib < 0
        or not math.isfinite(args.cache_reserve_gib)
        or args.cache_reserve_gib < 1
    ):
        raise ValueError("Cache budget must be finite >=0 and reserve finite >=1 GiB.")
    if (
        args.repeats < 5
        or args.warmups < 2
        or not 1 <= args.simulator_timeout_minutes <= 10
        or args.timeout_seconds < 60
    ):
        raise ValueError("Require >=2 warmups, >=5 repeats, 1..10 simulator minutes and >=60 seconds stage deadline.")
    return {
        "work_packages": [2, 4],
        "scope": "TP1 offline BF16 roots, single request, <=128 context tokens",
        "cases": cases,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "performance_target_met": None,
        "physical_npu": args.physical_npu,
        "cache_budget_gib": args.cache_budget_gib,
        "cache_reserve_gib": args.cache_reserve_gib,
        "simulator_timeout_minutes": args.simulator_timeout_minutes,
        "stage_timeout_seconds": args.timeout_seconds,
        "stages": [
            "environment/device",
            "short native regression",
            "paired offline measurement",
            "binary collection",
            "fused trace",
            "grouped trace",
            "evidence review/summary",
        ],
        "not_requested": ["cross-version/quality evaluation", "HTTP serving", "concurrency", "stability"],
        "baseline": "same library, original preparation, measurement mode; not old-server diagnostic logs",
        "native_review": "complete traces + exact hash-bound source/dataflow review; no regex-only approval",
    }


def source_identity():
    files = set()
    for folder, pattern in (("tools", "*.py"), ("vllm_ascend", "*.py"), ("csrc/vq2a8_ascendc", "*")):
        files.update(p for p in (REPO / folder).rglob(pattern) if p.is_file() and not p.is_symlink())
    return {str(p.relative_to(REPO)): digest(p) for p in sorted(files)}


def fingerprint(args, requested, library):
    root = args.model.resolve(strict=True)
    artifact = root / "experts_vq_ascend_v2"
    payload = list(root.glob("*.safetensors")) + list(artifact.rglob("*.safetensors"))
    packages = {}
    for name in PACKAGES:
        dist = importlib.metadata.distribution(name)
        packages[name] = {"version": dist.version, "direct_url": dist.read_text("direct_url.json")}
    return {
        "plan": requested,
        "library": library,
        "model": str(root),
        "inputs": capture_input_identity(root, artifact),
        "payload_stat_only": {
            str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(payload)
        },
        "source_sha256": source_identity(),
        "packages": packages,
        "python": sys.executable,
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("ASCEND", "CANN", "VLLM"))
            or k in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH", "OMP_NUM_THREADS")
        },
    }


def tree_hashes(directory):
    return {
        str(p.relative_to(directory)): digest(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and not p.is_symlink()
    }


def reusable(stage, identity_sha):
    if stage.get("status") != "PASS" or stage.get("identity_sha256") != identity_sha:
        return False
    directory = Path(stage["directory"])
    return directory.is_dir() and bool(stage.get("artifacts")) and tree_hashes(directory) == stage["artifacts"]


def performance_passed(output, library_sha, requested=None):
    path = output / "summary.json"
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("status") != "PASS"
        or data.get("library", {}).get("sha256") != library_sha
        or data.get("library_unchanged") is not True
        or data.get("performance_measurement_verified") is not True
    ):
        return False
    if requested is not None and (
        data.get("cases") != [list(v) for v in requested["cases"]]
        or data.get("warmups") != requested["warmups"]
        or data.get("repeats") != requested["repeats"]
    ):
        return False
    checked = summarize_performance(
        data["samples"], [tuple(v) for v in data["cases"]], data["repeats"], data["warmups"], data["regressions"]
    )
    return all(data.get(key) == checked[key] for key in ("groups", "ratios", "performance_target_met", "scope"))


class SummaryChildLog(LiveChildLog):
    """Keep raw child logs on disk; bounded console progress instead of layer spam."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._screen_pending = ""
        self._screen_updated = time.monotonic()

    def _emit(self, text):
        lines = (self._screen_pending + text).split("\n")
        self._screen_pending = lines[-1][-65536:]
        for line in lines[:-1]:
            if line.startswith(("PERF_", "PERFORMANCE=", "PROBE_WAIT=", "ASCENDC_SIM_REPORT=", "ERROR=", "Traceback")):
                super()._emit(line[:2000] + "\n")
                self._screen_updated = time.monotonic()
        if time.monotonic() - self._screen_updated >= 30:
            super()._emit(f"PROBE_WAIT={self.probe} LOG={self.path}\n")
            self._screen_updated = time.monotonic()


def supervise(command, log, environment, timeout):
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream, SummaryChildLog(log, log.stem):
        child = subprocess.Popen(
            command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        timed_out = False
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise
    return {
        "exit": child.returncode,
        "timeout": timed_out,
        "elapsed_s": time.monotonic() - started,
        "command": command,
        "log": str(log),
    }


def summary(root, report, review_path=None):
    latest = {row["name"]: dict(row) for row in report["stages"]}
    for row in latest.values():
        if row["status"] == "PASS" and not reusable(row, report["identity_sha256"]):
            row.update(status="FAIL", error="Saved stage evidence changed after collection.")
    perf = latest.get("performance", {})
    dirs = {
        k: Path(latest[f"native-{k}"]["directory"]) / "result" for k in ("fused", "grouped") if f"native-{k}" in latest
    }
    try:
        native = index_native_reports(root, report["identity"]["library"]["sha256"], dirs)
    except (OSError, ValueError) as exc:
        native = {
            "status": "REVIEW_REQUIRED",
            "error": str(exc),
            "kernels": [],
            "library_sha256": report["identity"]["library"]["sha256"],
            "native_fp8_instruction_observed": False,
            "native_instruction_verified": False,
            "on_chip_decode_verified": False,
        }
    if any(latest.get(f"native-{k}", {}).get("status") != "PASS" for k in ("fused", "grouped")):
        native["native_fp8_instruction_observed"] = False
    if review_path:
        native = apply_review(native, json.loads(review_path.read_text(encoding="utf-8")), root)
        native["review_receipt_sha256"] = digest(review_path)
    result = {
        "work_packages": [2, 4],
        "performance_measurement_verified": perf.get("status") == "PASS",
        "performance_target_met": None,
        "native": native,
        "quality": "NOT_REQUESTED",
        "serving": "NOT_REQUESTED",
        "stages": list(latest.values()),
        "identity_sha256": report["identity_sha256"],
    }
    required = [latest.get(n, {}).get("status") for n in ("environment", "preflight", "performance")]
    result["status"] = (
        "FAIL"
        if report.get("error") or any(s == "FAIL" for s in required)
        else "BLOCKED"
        if any(s != "PASS" for s in required)
        else native["status"]
    )
    if report.get("error"):
        result.update(error=report["error"], performance_measurement_verified=False)
    lines = [
        f"VQ2A8_WORK_PACKAGES=2,4 STATUS={result['status']}",
        f"PERFORMANCE_MEASUREMENT_VERIFIED={result['performance_measurement_verified']}",
        "PERFORMANCE_TARGET_MET=null (no speed threshold)",
        f"NATIVE_FP8_INSTRUCTION_OBSERVED={native['native_fp8_instruction_observed']}",
        f"NATIVE_INSTRUCTION_VERIFIED={native['native_instruction_verified']}",
        f"ON_CHIP_DECODE_VERIFIED={native['on_chip_decode_verified']}",
        "QUALITY=NOT_REQUESTED SERVING=NOT_REQUESTED",
        "SCOPE=TP1_OFFLINE_SINGLE_REQUEST_CONTEXT_LE_128",
    ]
    if perf.get("status") == "PASS":
        data = json.loads((Path(perf["directory"]) / "result/summary.json").read_text(encoding="utf-8"))
        result["performance"] = data
        for group in data["groups"]:
            metrics = group["metrics"]
            lines.append(
                f"PERF {group['case']} {group['variant']} n={metrics['e2e_s']['n']} "
                f"ttft_s={metrics['ttft_s']['median']:.6f} tpot_s={metrics['tpot_s']['median']:.6f} "
                f"e2e_s={metrics['e2e_s']['median']:.6f} tokens_s={metrics['output_tokens_per_s']['median']:.6f}"
            )
    for stage in latest.values():
        lines.append(f"STAGE={stage['name']} STATUS={stage['status']} REPORT={stage['directory']}")
    write_json(root / "summary.json", result)
    (root / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return result


def worker(args):
    from tools.validate_vq2a8_ascendc import require_hardware_runtime, run_model_preflight
    from tools.validate_vq2a8_ascendc_suite import collect_binary_evidence

    if args.worker == "preflight":
        run_model_preflight(args.library.resolve(), args.model.resolve(), args.physical_npu, args.output_dir)
    elif args.worker == "binary":
        collect_binary_evidence(library_evidence(args.library), args.output_dir)
    else:
        from tools.validate_vq2a8_v026_environment import check_runtime_imports, require_v026_stack

        environment = require_v026_stack()
        environment["runtime"] = check_runtime_imports()
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True, timeout=120)
        require_hardware_runtime()
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)
        if (torch.ones(1, device="npu:0") + 1).cpu().item() != 2:
            raise ValueError("Physical NPU smoke failed.")
        library = library_evidence(args.library)
        name = torch.npu.get_device_name(0)
        if name != library["build"]["soc"]:
            raise ValueError(
                f"Build SOC {library['build']['soc']} != actual device {name}; rebuild only this operator."
            )
        properties = torch.npu.get_device_properties(0)
        environment.update(
            status="PASS",
            device_name=name,
            device_properties=str(properties),
            hardware_runtime=require_hardware_runtime(),
            library_sha256=library["sha256"],
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir / "summary.json", environment)
    return 0


def run(args):
    requested = plan(args)
    if args.plan_only:
        print(json.dumps(requested, indent=2), flush=True)
        return 0
    if sys.platform != "linux":
        raise RuntimeError("Execute measurements on the Linux Ascend server. --plan-only works on the PC.")
    if args.model is None or args.library is None:
        raise ValueError("Require --model and --library.")
    library = library_evidence(args.library)
    identity = fingerprint(args, requested, library)
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if args.resume:
        root = args.resume.resolve(strict=True)
        report = json.loads((root / "journal.json").read_text(encoding="utf-8"))
        if report["identity_sha256"] != identity_sha:
            raise ValueError("Inputs/source/library/environment/plan changed; use a NEW report directory.")
        if digest(root / "source.zip") != report.get("source_archive_sha256"):
            raise ValueError("Saved source archive changed; preserve the report and use a NEW directory.")
        if report.get("error"):
            report.setdefault("previous_errors", []).append(report.pop("error"))
    else:
        root = (
            args.output_dir
            or REPO / "reports" / datetime.now(timezone.utc).strftime("vq2a8-perf-native-%Y%m%dT%H%M%SZ")
        ).resolve()
        root.mkdir(parents=True, exist_ok=False)
        report = {"identity": identity, "identity_sha256": identity_sha, "stages": []}
        write_json(root / "plan.json", requested)
        # Include untracked authored Python files too; git diff alone cannot
        # reconstruct an uncommitted delivery after those files are changed.
        with zipfile.ZipFile(root / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, expected in identity.get("source_sha256", {}).items():
                source_path = REPO / name
                if digest(source_path) != expected:
                    raise ValueError("Source changed while freezing the evidence snapshot.")
                archive.write(source_path, arcname=name)
        report["source_archive_sha256"] = digest(root / "source.zip")
        for name in ("kernel.cpp", "layout.h", "launch.h", "torch_binding.cpp", "CMakeLists.txt"):
            (root / name).write_bytes((REPO / "csrc/vq2a8_ascendc" / name).read_bytes())
        diff = subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=REPO, capture_output=True, check=True)
        (root / "source.diff").write_bytes(diff.stdout)
        write_json(root / "journal.json", report)
    child_env = acceptance_environment(REPO, args.physical_npu, "npu:0")
    child_env["ASCEND_LAUNCH_BLOCKING"] = "0"
    child_env["OMP_NUM_THREADS"] = "4"
    script = str(Path(__file__).resolve())

    def stage(name, build_command, accepted, timeout, *, reuse=True):
        previous = [s for s in report["stages"] if s["name"] == name]
        if reuse and previous and reusable(previous[-1], identity_sha):
            print(f"REUSE={name} REPORT={previous[-1]['directory']}", flush=True)
            return previous[-1]
        directory = root / f"{name}-attempt-{len(previous) + 1:03d}"
        directory.mkdir()
        output = directory / "result"
        command = build_command(output)
        record = {"name": name, "status": "RUNNING", "directory": str(directory), "identity_sha256": identity_sha}
        report["stages"].append(record)
        write_json(root / "journal.json", report)
        try:
            record.update(supervise(command, directory / "child.log", child_env, timeout))
            record["status"] = "PASS" if record["exit"] == 0 and not record["timeout"] and accepted(output) else "FAIL"
        except Exception as exc:
            record.update(status="FAIL", error=str(exc))
        finally:
            record["artifacts"] = tree_hashes(directory)
            write_json(root / "journal.json", report)
        return record

    def helper(kind, output):
        return [
            sys.executable,
            "-u",
            script,
            "--worker",
            kind,
            "--model",
            str(args.model.resolve()),
            "--library",
            library["path"],
            "--physical-npu",
            str(args.physical_npu),
            "--output-dir",
            str(output),
        ]

    def has_status(output, values):
        path = output / "summary.json"
        return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("status") in values

    try:
        # Even on resume re-check current device/driver, rather than trusting an old PASS.
        env = stage(
            "environment", lambda o: helper("environment", o), lambda o: has_status(o, ["PASS"]), 300, reuse=False
        )
        if env["status"] != "PASS":
            return 1
        device = json.loads((Path(env["directory"]) / "result/summary.json").read_text())
        device_key = {k: device[k] for k in ("device_name", "device_properties")}
        if report.get("device_identity", device_key) != device_key:
            raise ValueError("Resume device identity changed; use a new report directory.")
        report["device_identity"] = device_key
        preflight = stage(
            "preflight",
            lambda o: helper("preflight", o),
            lambda o: (o / "preflight.json").is_file()
            and json.loads((o / "preflight.json").read_text())["status"] == "passed",
            1900,
        )
        if preflight["status"] != "PASS":
            return 1
        receipt = Path(preflight["directory"]) / "result/preflight.json"
        stage(
            "performance",
            lambda o: [
                sys.executable,
                "-u",
                str(REPO / "tools/benchmark_vq2a8_offline.py"),
                "--model",
                str(args.model.resolve()),
                "--library",
                library["path"],
                "--preflight",
                str(receipt),
                "--output-dir",
                str(o),
                "--cases",
                args.cases,
                "--warmups",
                str(args.warmups),
                "--repeats",
                str(args.repeats),
                "--cache-budget-gib",
                str(args.cache_budget_gib),
                "--cache-reserve-gib",
                str(args.cache_reserve_gib),
            ],
            lambda o: performance_passed(o, library["sha256"], requested),
            args.timeout_seconds,
        )
        stage("binary", lambda o: helper("binary", o), lambda o: has_status(o, ["collected_review_pending"]), 300)
        for kind in ("fused", "grouped"):
            stage(
                f"native-{kind}",
                lambda o, kind=kind: [
                    sys.executable,
                    "-u",
                    str(REPO / "tools/profile_vq2a8_ascendc.py"),
                    "--diagnostic-build",
                    "--library",
                    library["path"],
                    "--cann",
                    library["build"]["cann"],
                    "--soc",
                    device["device_name"],
                    "--projection-path",
                    kind,
                    "--output-dir",
                    str(o),
                    "--timeout-minutes",
                    str(args.simulator_timeout_minutes),
                ],
                lambda o: has_status(o, ["collected_review_pending"]),
                args.simulator_timeout_minutes * 60 + 180,
            )
        if fingerprint(args, requested, library_evidence(args.library)) != identity:
            raise ValueError("Run inputs/source/environment changed; evidence cannot be accepted.")
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(root / "journal.json", report)
        result = summary(root, report, args.review_json)
        print(f"REPORT={root}", flush=True)
    return 0 if result["status"] == "PASS" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--physical-npu", type=int, default=0)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--output-dir", type=Path)
    target.add_argument("--resume", type=Path)
    target.add_argument("--summarize", type=Path)
    parser.add_argument(
        "--review-json", type=Path, help="Optional human native-dataflow attestation tied to this evidence."
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--cases", default="10:4,32:32,96:32")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--simulator-timeout-minutes", type=int, default=5)
    parser.add_argument("--cache-budget-gib", type=float, default=0.0)
    parser.add_argument("--cache-reserve-gib", type=float, default=16.0)
    parser.add_argument("--worker", choices=["environment", "preflight", "binary"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.physical_npu < 0:
        parser.error("Physical NPU ID must be nonnegative.")
    if args.summarize:
        root = args.summarize.resolve(strict=True)
        report = json.loads((root / "journal.json").read_text(encoding="utf-8"))
        return 0 if summary(root, report, args.review_json)["status"] == "PASS" else 2
    return worker(args) if args.worker else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
