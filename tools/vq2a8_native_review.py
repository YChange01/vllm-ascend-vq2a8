# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hash-bound native evidence index. Text hits are observations, not approval."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from tools.profile_vq2a8_ascendc import MAX_CSV_BYTES, digest

REVIEW_CLAIMS = ("fp8_operands", "packed_decode_in_ub", "ub_l1_l0_cube", "no_decoded_weight_gm_roundtrip")


def scan_instructions(path):
    result = {"fp8_mmad_calls": 0, "transfers": [], "examples": []}
    if path.stat().st_size > MAX_CSV_BYTES:
        raise ValueError("Instruction CSV exceeds bounded review reader size.")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"instr", "pipe", "detail", "call_count", "addr"}.issubset(reader.fieldnames or []):
            raise ValueError("Missing instruction operands/count/PC columns.")
        for number, row in enumerate(reader, 2):
            instr, pipe, detail = row["instr"].strip(), row["pipe"].strip(), row["detail"]
            count = int(row["call_count"] or "0", 0)
            if count <= 0:
                continue
            # A scalar MADD, zero-call static row or untyped matrix is NOT FP8.
            if instr == "MMAD" and pipe == "CUBE" and re.search(r"dtype:\s*E4M3E4M3(?:XD|[\s,;]|$)", detail):
                result["fp8_mmad_calls"] += count
                if len(result["examples"]) < 8:
                    result["examples"].append({"line": number, "pc": row["addr"], "detail": detail})
            if re.search(r"Src:(?:UB|L1|L0[A-C]|OUT),Dst:(?:UB|L1|L0[A-C]|OUT)", detail):
                if len(result["transfers"]) < 32:
                    result["transfers"].append({"line": number, "pc": row["addr"], "instr": instr, "detail": detail})
    return result


def index_native_reports(root, library_sha, directories=None):
    entries = []
    for kind in ("fused", "grouped"):
        directory = Path(directories[kind]) if directories and kind in directories else root / f"native-{kind}"
        summary = directory / "summary.json"
        entry = {"kernel": kind, "complete": False, "fp8_mmad_observed": False, "files": {}}
        entries.append(entry)
        if not summary.is_file():
            entry["error"] = "No completed collector report."
            continue
        report = json.loads(summary.read_text(encoding="utf-8"))
        app = report.get("application", {})
        jobs = 2 if kind == "grouped" else 1
        entry["complete"] = (
            report.get("status") == "collected_review_pending"
            and report.get("profiler") == {"exit": 0, "timeout": False}
            and report.get("global_config_unchanged") is True
            and report.get("profiler_log_review", {}).get("runtime_error_count") == 0
            and report.get("profiler_log_review", {}).get("application_timeout_reported") is False
            and app.get("status") == "completed"
            and (root / "kernel.cpp").is_file()
            and digest(root / "kernel.cpp")
            == report.get("library", {}).get("build", {}).get("source_sha256", {}).get("kernel.cpp")
            and report.get("library", {}).get("sha256") == library_sha
            and app.get("loaded_library", {}).get("sha256") == library_sha
            and app.get("library_unchanged") is True
            and app.get("kernel_prefix") == f"vq2a8_ascendc_{kind}"
            and app.get("logical_projections") == jobs
            and app.get("output_shapes") == ([[17, 32], [1, 32]] if jobs == 2 else [[17, 32]])
        )
        observed = []
        for csv_entry in report.get("instruction_csv", []):
            path = Path(csv_entry["path"])
            if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
                raise ValueError("Instruction evidence must remain within this report directory.")
            if not path.is_file() or digest(path) != csv_entry.get("sha256"):
                raise ValueError("Instruction evidence changed after collection.")
            scanned = scan_instructions(path)
            name = str(path.relative_to(root))
            entry["files"][name] = digest(path)
            observed.append({"path": name, **scanned})
        entry["instructions"] = observed
        entry["fp8_mmad_observed"] = entry["complete"] and any(r["fp8_mmad_calls"] > 0 for r in observed)
        entry["files"][str(summary.relative_to(root))] = digest(summary)
    return {
        "library_sha256": library_sha,
        "kernel_source_sha256": digest(root / "kernel.cpp") if (root / "kernel.cpp").is_file() else None,
        "kernels": entries,
        "native_fp8_instruction_observed": all(e["fp8_mmad_observed"] for e in entries),
        "native_instruction_verified": False,
        "on_chip_decode_verified": False,
        "status": "REVIEW_REQUIRED",
        "note": "Simulator observations are not hardware timings; address/dataflow review is still required.",
    }


def apply_review(index, review, root):
    """Accept an explicit human attestation, never infer dataflow from regex.

    Receipts identify a reviewer, explain every claim for BOTH entry points,
    and cite immutable local evidence plus source locations. This validates
    provenance/coverage, not the truth of a reviewer's semantic assessment.
    """
    if review.get("library_sha256") != index["library_sha256"] or not review.get("reviewer"):
        raise ValueError("Review must identify reviewer and exact final library SHA256.")
    if not index["native_fp8_instruction_observed"]:
        raise ValueError("Cannot approve incomplete traces or missing executed FP8 operands.")
    for entry in index["kernels"]:
        reviewed = review.get("kernels", {}).get(entry["kernel"], {})
        for claim in REVIEW_CLAIMS:
            item = reviewed.get(claim, {})
            if (
                item.get("accepted") is not True
                or not isinstance(item.get("explanation"), str)
                or not item["explanation"].strip()
            ):
                raise ValueError(f"Missing reasoned review: {entry['kernel']}/{claim}.")
            refs = item.get("references", [])
            if not refs:
                raise ValueError("Review claims need source/trace references, not just a boolean.")
            for ref in refs:
                path = root / ref["path"]
                if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not ref.get("location"):
                    raise ValueError("Review references must be local evidence with explicit locations.")
                if digest(path) != ref.get("sha256"):
                    raise ValueError("Stale native review reference.")
            if not any(ref["path"] in entry["files"] and ref["sha256"] == entry["files"][ref["path"]] for ref in refs):
                raise ValueError("Every claim must cite this kernel's collected trace/report.")
            if claim != "fp8_operands" and not any(
                ref["path"] == "kernel.cpp" and ref["sha256"] == index.get("kernel_source_sha256") for ref in refs
            ):
                raise ValueError("Dataflow claims also need the pinned kernel source location.")
    return {
        **index,
        "native_instruction_verified": True,
        "on_chip_decode_verified": True,
        "status": "PASS",
        "verification_method": "explicit_hash_bound_human_attestation",
        "review": review,
    }
