#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read a bounded summary of an existing full-model startup trace; no NPU work."""

from __future__ import annotations

# Keep tools/bisect from shadowing the standard library in direct execution.
# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import re
from collections import deque
from pathlib import Path

DEFAULT_ROOT = Path("/tmp")
REPORT_PATTERN = "vq2-full-startup-*"
MAX_EVENTS = 12
MAX_STACK_LINES = 24
MAX_LINE_CHARS = 500
MAX_EVENT_BYTES = 1024 * 1024
MAX_STACK_BYTES = 2 * 1024 * 1024
EVENT_TYPES = frozenset(("READY", "BEGIN", "SUBMITTED", "PASS", "FAIL"))
THREAD_HEADER = re.compile(r"^(?:Current thread|Thread)\s+(0x[0-9a-fA-F]+)\b")


def _line(value, limit=MAX_LINE_CHARS):
    text = " ".join(str(value).split())
    text = "".join(character if character.isprintable() else "?" for character in text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def select_report(report_dir=None, root=DEFAULT_ROOT):
    if report_dir is not None:
        selected = Path(report_dir).resolve(strict=True)
        if not selected.is_dir():
            raise ValueError(f"Not a report directory: {selected}")
        return selected
    candidates = []
    for path in Path(root).glob(REPORT_PATTERN):
        if path.is_dir():
            candidates.append((path.stat().st_mtime_ns, path.name, path))
    if not candidates:
        raise FileNotFoundError(f"No {REPORT_PATTERN} report directories under {root}; specify a report directory.")
    # Do not require events.jsonl to exist/nonempty: doing that would silently
    # select an older run when the newest worker is only beginning its trace.
    return max(candidates)[2].resolve(strict=True)


def read_events(path):
    result = {
        "count": 0,
        "ignored": 0,
        "partial_tail": False,
        "recent": deque(maxlen=MAX_EVENTS),
        "active": {},
        "last_error": None,
        "deepest_error": None,
        "error_ancestors": set(),
        "main_thread_id": None,
        "has_span_ids": False,
        "ready": False,
        "missing": False,
    }
    try:
        source = Path(path).open("rb")  # noqa: SIM115 - closed by the following with; only open may be missing.
    except FileNotFoundError:
        result["missing"] = True
        return result
    with source:
        # Snapshot the byte boundary once: a running worker cannot make this
        # reader chase newly appended events indefinitely.
        remaining = os.fstat(source.fileno()).st_size
        oversized = False
        while remaining:
            raw = source.readline(min(remaining, MAX_EVENT_BYTES))
            if not raw:
                result["partial_tail"] = True
                break
            remaining -= len(raw)
            if not raw.endswith(b"\n"):
                if not remaining:
                    result["partial_tail"] = True
                oversized = True
                continue
            if oversized:
                result["ignored"] += 1
                oversized = False
                continue
            try:
                event = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                result["ignored"] += 1
                continue
            if (
                not isinstance(event, dict)
                or not isinstance(event.get("event"), str)
                or event["event"] not in EVENT_TYPES
            ):
                result["ignored"] += 1
                continue
            result["count"] += 1
            result["recent"].append(event)
            main_thread_id = event.get("main_thread_id")
            if type(main_thread_id) is int and main_thread_id > 0:
                result["main_thread_id"] = main_thread_id
            if event["event"] == "READY":
                result["ready"] = True
                continue
            span_id = event.get("span_id")
            if event["event"] == "FAIL":
                result["last_error"] = event
                # An exception is emitted inner-to-outer while spans unwind.
                # Preserve the innermost failure of this chain, but reset for
                # a newer independent failure instead of retaining old errors.
                if type(span_id) is not int or span_id not in result["error_ancestors"]:
                    result["deepest_error"] = event
                    ancestors = set()
                    parent_id = event.get("parent_id")
                    while type(parent_id) is int and parent_id > 0 and parent_id not in ancestors:
                        ancestors.add(parent_id)
                        parent = result["active"].get(parent_id)
                        parent_id = parent["begin"].get("parent_id") if parent else None
                    result["error_ancestors"] = ancestors
                else:
                    result["error_ancestors"].discard(span_id)
            if type(span_id) is not int or span_id < 1:
                continue
            result["has_span_ids"] = True
            active = result["active"]
            if event["event"] == "BEGIN":
                parent_id = event.get("parent_id")
                parent = active.get(parent_id) if type(parent_id) is int else None
                active[span_id] = {
                    "begin": event,
                    "latest": event,
                    "depth": parent["depth"] + 1 if parent else 0,
                    "order": result["count"],
                }
            elif event["event"] == "SUBMITTED":
                if span_id in active:
                    active[span_id]["latest"] = event
            else:
                active.pop(span_id, None)
    return result


def read_main_stack(path, main_thread_id):
    try:
        source = Path(path).open("rb")  # noqa: SIM115 - closed by the following with; only open may be missing.
    except FileNotFoundError:
        return "MISSING", []
    with source:
        size = os.fstat(source.fileno()).st_size
        start = max(0, size - MAX_STACK_BYTES)
        source.seek(start)
        raw = source.read(size - start)
    if start:
        raw = raw.partition(b"\n")[2]
    lines = raw.decode("utf-8", errors="replace").splitlines()
    boundaries = [index for index, text in enumerate(lines) if text.startswith("Timeout (")]
    if not boundaries:
        return "NO_TIMEOUT_DUMP_YET", []
    # Never fall back to an older complete dump while the latest is being
    # written: its main-thread block may not have reached the file yet.
    latest = lines[boundaries[-1] + 1 :]
    headers = [(index, THREAD_HEADER.match(text)) for index, text in enumerate(latest)]
    headers = [(index, match) for index, match in headers if match]
    if type(main_thread_id) is not int:
        return "MAIN_THREAD_ID_UNKNOWN", []
    for offset, (begin, match) in enumerate(headers):
        if int(match.group(1), 16) != main_thread_id:
            continue
        end = headers[offset + 1][0] if offset + 1 < len(headers) else len(latest)
        block = []
        for text in latest[begin:end]:
            if not text.strip() or text.startswith("Extension modules:"):
                break
            block.append(_line(text))
            if len(block) == MAX_STACK_LINES:
                break
        return "LATEST_MAIN_THREAD_DUMP", block
    return "LATEST_DUMP_MAIN_THREAD_NOT_YET_PRESENT", []


def event_line(event):
    # Only known scalar fields are rendered; arbitrary metadata/Tensor values
    # never enter the compact summary.
    fields = (
        ("seq", event.get("seq")),
        ("time", event.get("time")),
        ("event", event.get("event")),
        ("stage", event.get("stage")),
        ("span", event.get("span_id")),
        ("parent", event.get("parent_id")),
        ("elapsed_s", event.get("elapsed_s")),
        ("device", event.get("device_completion")),
    )
    return _line(
        " ".join(f"{name}={_line(value, 180)}" for name, value in fields if isinstance(value, (str, int, float)))
    )


def summarize(report_dir):
    report = Path(report_dir)
    events = read_events(report / "events.jsonl")
    lines = [f"REPORT={_line(report)}", "READ_ONLY=True NPU_INITIALIZED=False"]
    lines.append(f"EVENTS={events['count']} IGNORED={events['ignored']} PARTIAL_TAIL={events['partial_tail']}")
    if not events["count"]:
        detail = "missing" if events["missing"] else "empty or no complete recognized events"
        lines.append(f"TRACE_NOT_READY=events.jsonl is {detail}; no older report was substituted.")
    else:
        latest = events["recent"][-1]
        lines.append(_line(f"WORKER_PID={latest.get('pid', 'UNKNOWN')} MODE={latest.get('mode', 'UNKNOWN')}"))
    if events["active"]:
        pending = max(events["active"].values(), key=lambda entry: (entry["depth"], entry["order"]))
        latest = pending["latest"]
        lines.append(_line(f"DEEPEST_OPEN depth={pending['depth']} {event_line(latest)}"))
        if latest.get("event") == "SUBMITTED" and latest.get("mode") == "sync":
            lines.append("OPEN_STATE=call returned; synchronization has not completed in this snapshot.")
        else:
            lines.append("OPEN_STATE=span has no terminal event; this alone does not prove a CPU or NPU deadlock.")
    else:
        if events["has_span_ids"]:
            lines.append("DEEPEST_OPEN=NONE")
        elif events["ready"]:
            lines.append("DEEPEST_OPEN=NONE (hooks ready; no model span yet)")
            lines.append(
                "STARTUP_STATE=hooks installed; awaiting first traced model span, not verified device completion."
            )
        else:
            lines.append("DEEPEST_OPEN=UNKNOWN (no span IDs yet)")
    if events["last_error"]:
        failed = events["last_error"]
        lines.append(
            _line(f"LAST_ERROR stage={failed.get('stage')} {failed.get('error_type', '')}: {failed.get('error', '')}")
        )
        deepest = events["deepest_error"]
        if deepest is not failed:
            lines.append(
                _line(
                    f"DEEPEST_ERROR stage={deepest.get('stage')} "
                    f"{deepest.get('error_type', '')}: {deepest.get('error', '')}"
                )
            )
    lines.append(f"RECENT_EVENTS (up to {MAX_EVENTS}):")
    lines.extend(event_line(event) for event in events["recent"])
    stack_status, stack = read_main_stack(report / "stacks.log", events["main_thread_id"])
    lines.append(f"STACK_STATUS={stack_status}")
    lines.extend(stack)
    lines.append("NOTE=async PASS means the call returned, not verified device completion; this is not a TPOT report.")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", nargs="?", type=Path, help="default: newest /tmp/vq2-full-startup-* directory")
    args = parser.parse_args(argv)
    try:
        report = select_report(args.report_dir)
        for line in summarize(report):
            print(line)
        return 0
    except (OSError, ValueError) as exc:
        print(_line(f"ERROR={type(exc).__name__}: {exc}"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
