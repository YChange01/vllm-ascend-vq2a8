#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-command standalone AscendC campaign; never switches the model backend.

Shared controls run once, then all layers are sampled in isolated processes.
Numerical failures continue; device/runtime failures stop subsequent NPU work.
Native binary evidence and resident-projection timings are collected for review,
not automatically promoted to native-instruction/on-chip/performance acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import validate_vq2a8_ascendc as gate  # noqa: E402
from tools.vq2a8_live_log import LiveChildLog  # noqa: E402


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def select_probes(config, requested=None):
    """Plan from checkpoint metadata without importing Torch in the supervisor."""
    layers, hashed, experts = (config[k] for k in ("num_hidden_layers", "num_hash_layers", "n_routed_experts"))
    if not all(type(v) is int for v in (layers, hashed, experts)) or not (
        0 <= hashed <= layers and layers > 0 and experts > 0
    ):
        raise ValueError("Invalid checkpoint layer/expert counts.")
    if requested is not None:
        probes = []
        for value in requested.split(","):
            if not re.fullmatch(r"[0-9]+:[0-9]+", value):
                raise ValueError("--probes must be comma-separated layer:expert pairs.")
            layer, expert = map(int, value.split(":"))
            if not 0 <= layer < layers or not 0 <= expert < (1 if layer < hashed else experts):
                raise ValueError(f"Probe outside the frozen artifact: {value}")
            canonical = f"{layer}:{expert}"
            if canonical in probes:
                raise ValueError("Duplicate probes are not allowed.")
            probes.append(canonical)
        return probes
    # One real stored expert in EVERY layer; varied routed IDs. The first
    # num_hash_layers store only expert 0 in this frozen export.
    pairs = {(layer, 0 if layer < hashed else ((layer - hashed) * 73) % experts) for layer in range(layers)}
    if hashed < layers:
        for layer in {hashed, hashed + (layers - hashed) // 2, layers - 1}:
            for expert in {0, (experts - 1) // 2, min(135, experts - 1), experts - 1}:
                pairs.add((layer, expert))
    return [f"{layer}:{expert}" for layer, expert in sorted(pairs)]


def make_plan(probes):
    plan = [{"id": s, "stage": s, "probe": probes[0]} for s in ("direct", "bridge", "fused", "boundaries")]
    plan += [{"id": "expert-" + p.replace(":", "-"), "stage": "expert", "probe": p} for p in probes]
    # First/middle/last selected experts cover distinct layer positions while
    # keeping timing overhead bounded. Every timing target first passes gates.
    timed = list(dict.fromkeys((probes[0], probes[len(probes) // 2], probes[-1])))
    plan += [{"id": "timing-" + p.replace(":", "-"), "stage": "timing", "probe": p} for p in timed]
    return plan


def discover_device_objects(build_dir):
    """Read only this native target's generated objects, never a broad tree."""
    roots = [build_dir / "auto_gen/vq2a8_ascendc_kernel"]
    roots += sorted(build_dir.glob("vq2a8_ascendc_kernel_*device-prefix"))
    found, seen = [], set()
    for root in roots:
        if not root.is_dir() or not root.resolve().is_relative_to(build_dir):
            continue
        for path in sorted(root.rglob("*.o")):
            if path.is_symlink() or not path.resolve().is_relative_to(build_dir):
                continue
            with path.open("rb") as source:
                if source.read(4) != b"\x7fELF":
                    continue  # preprocessed text can also have an .o suffix
            digest = sha256(path)
            if digest not in seen:
                seen.add(digest)
                found.append({"path": str(path), "sha256": digest, "bytes": path.stat().st_size})
    return found


def inspect_instruction_text(path):
    """Classify objdump text, not the executable ISA or its correctness.

    CANN's public objdump can deliberately print only symbols and offsets.
    Empty address lines, raw bytes and unknown opcodes are NOT instructions.
    Even mnemonic text and FP8 keyword hits still require human ISA review.
    """
    addresses, empty, unknown, instructions, hits = 0, 0, 0, 0, 0
    symbols, excerpt = [], []
    with path.open(errors="replace") as source:
        for line in source:
            symbol = re.match(r"\s*[0-9a-f]+\s+<(.+)>:\s*$", line, re.I)
            if symbol:
                symbols.append(symbol.group(1))
                continue
            address = re.match(r"\s*[0-9a-f]+:\s*(.*)$", line, re.I)
            if not address:
                continue
            addresses += 1
            body = address.group(1).strip()
            if not body:
                empty += 1
                continue
            # Discard any leading encoding bytes/words. This is deliberately
            # conservative: a raw .word or <unknown> is not an ISA decode.
            tokens = body.split()
            while tokens and re.fullmatch(r"(?:[0-9a-f]{2}|[0-9a-f]{4}|[0-9a-f]{8}|[0-9a-f]{16})", tokens[0], re.I):
                tokens.pop(0)
            if not tokens or not re.fullmatch(r"[a-z_][a-z0-9_.]*", tokens[0], re.I) or tokens[0].lower() == "unknown":
                unknown += 1
                continue
            instructions += 1
            if re.search(
                r"\b(?:mad\w*|mmad\w*|fp8|e4m3\w*|copy_ubuf\w*|copy_cbuf\w*|set_intra\w*|wait_intra\w*)\b",
                " ".join(tokens),
                re.I,
            ):
                hits += 1
                if len(excerpt) < 12:
                    excerpt.append(line.strip()[:400])
    return {
        "address_lines": addresses,
        "empty_address_lines": empty,
        "undecoded_address_lines": unknown,
        "instruction_lines": instructions,
        "signal_line_count": hits,
        "symbol_count": len(symbols),
        "fused_symbols": [s for s in symbols if "vq2a8_ascendc_fused" in s],
        "excerpt": excerpt,
    }


def collect_binary_evidence(library, directory):
    """Disassemble bounded native-target artifacts; opcode hits are NOT proof."""
    directory.mkdir()
    build_dir = Path(library["path"]).parent.resolve()
    cann = Path(library["build"]["cann"])
    candidates = [
        cann / prefix / "bin/llvm-objdump"
        for prefix in ("aarch64-linux/ccec_compiler", "ccec_compiler", "compiler/ccec_compiler")
    ]
    tool = next((p for p in candidates if p.is_file()), None)
    objects = discover_device_objects(build_dir)
    report = {
        "library_sha256": library["sha256"],
        "objects": objects,
        "tool": str(tool) if tool else None,
        "native_instruction_verified": False,
        "on_chip_decode_verified": False,
        "object_linkage_to_loaded_library_verified": False,
        "review": "Inspect fused operand types, UB/L1/L0 transfers and link provenance; text hits are insufficient.",
    }
    if tool:
        report["tool_sha256"] = sha256(tool)
    for index, obj in enumerate(objects):
        if not tool:
            obj["status"] = "tool_missing"
            continue
        log = directory / f"object-{index:03d}.asm.txt"
        command = [str(tool), "-d", "--demangle", obj["path"]]
        obj.update(command=command, disassembly=str(log))
        print(f"ASCENDC_BINARY_START object={index} FILE={obj['path']}", flush=True)
        try:
            with log.open("w") as stream:
                child = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=60, check=False)
            obj.update(exit=child.returncode, status="disassembled" if child.returncode == 0 else "disassembly_failed")
        except (OSError, subprocess.TimeoutExpired) as exc:
            obj.update(status="disassembly_failed", error=str(exc))
        obj.update(inspect_instruction_text(log), disassembly_sha256=sha256(log))
        if obj["status"] == "disassembled":
            if obj["address_lines"] and obj["empty_address_lines"] == obj["address_lines"]:
                obj["status"] = "symbols_and_offsets_only"
                obj["note"] = "CANN public objdump does not expose assembly; do not count offsets as instructions."
            elif not obj["instruction_lines"]:
                obj["status"] = "no_instructions"
            elif obj["empty_address_lines"] or obj["undecoded_address_lines"]:
                obj["status"] = "partial_instruction_text"
        print("ASCENDC_BINARY_RESULT " + json.dumps(obj), flush=True)
    report["status"] = (
        "collected_review_pending"
        if objects and all(o["status"] == "disassembled" for o in objects)
        else "incomplete_review_pending"
    )
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_step(args, step, directory, digest):
    evidence, log = directory / f"{step['id']}.json", directory / f"{step['id']}.log"
    command = [
        sys.executable,
        "-u",
        str(REPO / "tools/validate_vq2a8_ascendc.py"),
        "--library",
        str(args.library),
        "--model",
        str(args.model),
        "--stage",
        step["stage"],
        "--probe",
        step["probe"],
        "--output",
        str(evidence),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
    ]
    env = gate.acceptance_environment(REPO, args.physical_npu, "npu:0")
    if step["stage"] in ("timing", "grouped"):
        env["ASCEND_LAUNCH_BLOCKING"] = "0"  # child only; never change the parent
    result = {
        **step,
        "status": "failed",
        "exit": None,
        "timeout": False,
        "log": str(log),
        "evidence": str(evidence),
        "command": command,
    }
    print(f"ASCENDC_SUITE_STEP_START={step['id']} LOG={log}", flush=True)
    with log.open("w") as stream, LiveChildLog(log, step["id"]):
        try:
            child = subprocess.run(
                command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout, check=False
            )
            result["exit"] = child.returncode
        except subprocess.TimeoutExpired:
            result["timeout"] = True
        except OSError as exc:
            result["error"] = str(exc)
    result["passed"] = result["exit"] == 0 and gate.evidence_passed(evidence, step["stage"], digest, step["probe"])
    if result["passed"]:
        result["status"] = "passed"
        return result
    try:
        child_evidence = json.loads(evidence.read_text())
        error = child_evidence.get("error", "")
        if not isinstance(error, str):
            error = ""
    except (OSError, ValueError, AttributeError):
        error = ""
    result["error"] = result.get("error", error or "Exit/timeout/missing or incomplete evidence")
    # Continue only positively identified Python numerical assertions. An NPU
    # runtime fault, timeout, signal exit or unknown failure stops device work.
    numerical = result["exit"] == 1 and not result["timeout"] and error.startswith("AssertionError:")
    if numerical:
        with log.open(errors="replace") as source:
            numerical = not any(
                re.search(r"507015|aicore exception|acl api failed|DDR address|device.*fault", line, re.I)
                for line in source
            )
    result["failure_kind"] = "numerical" if numerical else "runtime_or_unknown"
    for line in gate.error_excerpt(log):
        print(line, flush=True)
    return result


def write_summary(directory, report):
    """A compact final report alongside full per-stage evidence, updated live."""
    rows = report["results"]
    numeric = [r for r in rows if r["stage"] != "timing"]
    timed = [r for r in rows if r["stage"] == "timing"]
    report["standalone_numerics_verified"] = len(numeric) == report["planned_numeric_steps"] and all(
        r["status"] == "passed" for r in numeric
    )
    report["timings_collected"] = (
        bool(timed) and len(timed) == report["planned_timing_steps"] and all(r["status"] == "passed" for r in timed)
    )
    lines = [
        f"ASCENDC_SUITE={report['status'].upper()} SCOPE=STANDALONE",
        f"PROBES={len(report['probes'])} LAYERS_COVERED={report['layers_covered']} "
        f"ALL_LAYERS_SAMPLED={report['all_layers_sampled']} ALL_EXPERTS_TESTED=False",
        f"STEPS={len(rows)}/{len(report['plan'])} NUMERICS_VERIFIED={report['standalone_numerics_verified']} "
        f"TIMINGS_COLLECTED={report['timings_collected']}",
    ]
    for row in rows:
        lines.append(
            f"STEP={row['id']} STATUS={row['status']}" + (f" REASON={row['error']}" if row.get("error") else "")
        )
        if row["stage"] == "timing" and row["status"] == "passed":
            evidence = json.loads(Path(row["evidence"]).read_text())
            for item in evidence["results"]:
                t = item["timings"]
                lines.append(
                    f"TIMING probe={row['probe']} case={item['key']} "
                    f"candidate_wall_ms={t['candidate']['wall_ms']['median']:.6f} "
                    f"baseline_wall_ms={t['accepted_baseline']['wall_ms']['median']:.6f} "
                    f"candidate_event_ms={t['candidate']['event_ms']['median']:.6f} "
                    f"baseline_event_ms={t['accepted_baseline']['event_ms']['median']:.6f} "
                    f"observed_wall_ratio={t['wall_median_ratio_baseline_over_candidate']:.3f}"
                )
    lines += [
        f"BINARY_EVIDENCE={report.get('binary', {}).get('status', 'not_collected')}",
        "NATIVE_INSTRUCTION_VERIFIED=False ON_CHIP_DECODE_VERIFIED=False PERFORMANCE_VERIFIED=False",
        "DEFAULT_MODEL_BACKEND=UNCHANGED MODEL_INTEGRATION_VERIFIED=False "
        "QUALITY_VERIFIED=False SERVING_VERIFIED=False",
        f"REPORT={directory}",
    ]
    (directory / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (directory / "summary.txt").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def run(args):
    library = gate.library_evidence(args.library)
    config = json.loads((args.model / "config.json").read_text())
    probes = select_probes(config, args.probes)
    plan = make_plan(probes)
    directory = args.output_dir or Path(tempfile.mkdtemp(prefix="vq2a8-ascendc-suite-"))
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Use an empty --output-dir; old evidence is never reused implicitly.")
    report = {
        "status": "running",
        "implementation": "ascendc",
        "library": library,
        "probes": probes,
        "plan": plan,
        "results": [],
        "layers_covered": len({p.split(":")[0] for p in probes}),
        "all_layers_sampled": len({p.split(":")[0] for p in probes}) == config["num_hidden_layers"],
        "all_experts_tested": False,
        "model_config_sha256": sha256(args.model / "config.json"),
        "harness_sha256": {
            p: sha256(REPO / p)
            for p in (
                "tools/validate_vq2a8_ascendc.py",
                "tools/validate_vq2a8_ascendc_suite.py",
                "vllm_ascend/quantization/vq2a8_ascendc.py",
            )
        },
        "planned_numeric_steps": sum(p["stage"] != "timing" for p in plan),
        "planned_timing_steps": sum(p["stage"] == "timing" for p in plan),
        "physical_npu": args.physical_npu,
        "default_model_backend": "unchanged",
        "native_instruction_verified": False,
        "on_chip_decode_verified": False,
        "performance_verified": False,
        "model_integration_verified": False,
        "quality_verified": False,
        "serving_verified": False,
    }
    print(f"ASCENDC_SUITE_REPORT_DIR={directory}", flush=True)
    print(
        "ASCENDC_SUITE_PLAN "
        + json.dumps(
            {
                "probes": probes,
                "steps": len(plan),
                "numerical_cases": sum(len(gate.expected_keys(p["stage"])) for p in plan if p["stage"] != "timing"),
            }
        ),
        flush=True,
    )
    write_summary(directory, report)
    try:
        report["binary"] = collect_binary_evidence(library, directory / "binary")
    except (OSError, ValueError, KeyError) as exc:
        report["binary"] = {"status": "incomplete_review_pending", "error": str(exc)}
    stop, failed_experts = None, set()
    for step in plan:
        reason = stop
        if step["stage"] == "timing" and step["probe"] in failed_experts:
            reason = "This expert failed numerical acceptance; timing is not eligible."
        if reason:
            result = {**step, "status": "skipped", "passed": False, "error": reason}
        else:
            result = run_step(args, step, directory, library["sha256"])
            if not result["passed"]:
                if step["stage"] not in ("expert", "timing") or result["failure_kind"] != "numerical":
                    stop = f"Device work stopped after {step['id']}: {result['failure_kind']}"
                if step["stage"] == "expert":
                    failed_experts.add(step["probe"])
        report["results"].append(result)
        write_summary(directory, report)
        print(f"ASCENDC_SUITE_STEP_DONE={step['id']} STATUS={result['status']}", flush=True)
    report["status"] = (
        "completed_review_pending" if all(r["status"] == "passed" for r in report["results"]) else "failed"
    )
    print(write_summary(directory, report), flush=True)
    return 0 if report["status"] == "completed_review_pending" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=REPO / "build/vq2a8-ascendc/libvq2a8_ascendc.so")
    parser.add_argument("--physical-npu", type=int, default=4)
    parser.add_argument(
        "--probes", help="Optional explicit subset; default samples every layer plus routed-ID boundaries."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per isolated NPU child, in seconds.")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.physical_npu < 0 or args.timeout <= 0 or args.warmups < 3 or args.repeats < 10:
        parser.error("Require physical NPU >=0, timeout >0, warmups >=3 and repeats >=10.")
    args.model, args.library = args.model.resolve(), args.library.resolve()
    if args.output_dir:
        args.output_dir = args.output_dir.resolve()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
