#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only, stdlib-only extraction of Ascend serving profiles.

No torch/NPU imports or connection to the serving process. JSON events are read
incrementally in two passes; only the selected short windows are retained.
This is diagnostic attribution, not a benchmark or an automatic request parser.
"""

from __future__ import annotations

# tools/bisect must not shadow the standard library when invoked directly.
import os as _bootstrap_os
import sys as _bootstrap_sys

if not __package__:
    _bootstrap_sys.path[0] = _bootstrap_os.path.dirname(
        _bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))
    )

import argparse  # noqa: E402
import csv  # noqa: E402
import gzip  # noqa: E402
import heapq  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

CHUNK_BYTES = 1024 * 1024
GRAPH_API = "aclmdlRIExecuteAsync"
SYNC_API = "aclrtSynchronizeStreamWithTimeout"
SCALAR_OP = "aten::_local_scalar_dense"


class JSONStream:
    """Incremental raw_decode with strict array/object separators and EOF."""

    def __init__(self, stream):
        self.stream, self.buffer, self.position, self.eof = stream, "", 0, False
        self.decoder = json.JSONDecoder()

    def fill(self):
        self.buffer = self.buffer[self.position :]
        self.position = 0
        block = self.stream.read(CHUNK_BYTES)
        self.buffer += block
        self.eof = not block

    def peek(self):
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof:
                return ""
            self.fill()

    def expect(self, character):
        if self.peek() != character:
            raise ValueError(f"Invalid/truncated trace JSON: expected {character!r}.")
        self.position += 1

    def value(self):
        if not self.peek():
            raise ValueError("Truncated trace JSON value.")
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
                # A number may continue as 1e+3 or 1.25 across block boundaries.
                partial_number = (
                    isinstance(value, (int, float)) and end < len(self.buffer) and self.buffer[end] in ".eE+-0123456789"
                )
                if not self.eof and (end == len(self.buffer) or partial_number):
                    self.fill()
                    continue
                self.position = end
                return value
            except json.JSONDecodeError as error:
                if self.eof:
                    raise ValueError(f"Invalid/truncated trace JSON: {error}") from error
                self.fill()

    def array(self):
        self.expect("[")
        if self.peek() == "]":
            self.expect("]")
            return
        while True:
            yield self.value()
            if self.peek() == "]":
                self.expect("]")
                return
            self.expect(",")


def trace_events(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        reader = JSONStream(stream)
        if reader.peek() == "[":
            yield from reader.array()
        else:
            reader.expect("{")
            found = False
            if reader.peek() != "}":
                while True:
                    key = reader.value()
                    if not isinstance(key, str):
                        raise ValueError("Trace JSON object key must be a string.")
                    reader.expect(":")
                    if key == "traceEvents":
                        if found:
                            raise ValueError("Duplicate traceEvents field.")
                        found = True
                        yield from reader.array()
                    else:
                        reader.value()
                    if reader.peek() == "}":
                        break
                    reader.expect(",")
            reader.expect("}")
            if not found:
                raise ValueError("Trace JSON has no traceEvents array.")
        if reader.peek():
            raise ValueError("Trailing data after trace JSON.")


def finite_number(value):
    if isinstance(value, bool):
        raise ValueError("Boolean is not a timestamp/duration.")
    result = float(str(value).strip())
    if not math.isfinite(result):
        raise ValueError("Non-finite timestamp/duration.")
    return result


def event_span(event):
    if not isinstance(event, dict) or event.get("ph") != "X":
        return None
    try:
        start, duration = finite_number(event["ts"]), finite_number(event["dur"])
    except (KeyError, ValueError, TypeError):
        return None
    return (start, start + duration) if duration >= 0 else None


def overlap(span, bounds):
    return max(0.0, min(span[1], bounds[1]) - max(span[0], bounds[0]))


def interval_union(intervals):
    end, total = -math.inf, 0.0
    for start, stop in sorted(intervals):
        if stop > max(start, end):
            total += stop - max(start, end)
        end = max(end, stop)
    return total


def clipped_intervals(events, bounds):
    return [
        (max(span[0], bounds[0]), min(span[1], bounds[1]))
        for event in events
        if (span := event_span(event)) is not None and overlap(span, bounds)
    ]


def largest_gaps(intervals, bounds, top=5):
    cursor, gaps = bounds[0], []
    for start, stop in sorted(intervals):
        start, stop = max(start, bounds[0]), min(stop, bounds[1])
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, stop)
    if cursor < bounds[1]:
        gaps.append((cursor, bounds[1]))
    return [
        {"start_us": start, "end_us": stop, "duration_ms": round((stop - start) / 1000, 6)}
        for start, stop in sorted(gaps, key=lambda pair: pair[1] - pair[0], reverse=True)[:top]
    ]


def short_event(event):
    start, stop = event_span(event)
    return {
        "name": event.get("name", ""),
        "pid": event.get("pid"),
        "tid": event.get("tid"),
        "start_us": start,
        "end_us": stop,
        "duration_ms": round((stop - start) / 1000, 6),
    }


def scan_trace(path):
    graphs, scalars, synchronizations, metadata = [], [], [], []
    categories, invalid, complete, phases = Counter(), 0, 0, Counter()
    for event in trace_events(path):
        if not isinstance(event, dict):
            invalid += 1
            continue
        phases[str(event.get("ph", ""))] += 1
        if event.get("ph") == "M":
            metadata.append(event)
        span = event_span(event)
        if span is None:
            invalid += int(event.get("ph") == "X")
            continue
        complete += 1
        categories[str(event.get("cat", ""))] += 1
        name = str(event.get("name", ""))
        if GRAPH_API in name:
            graphs.append(event)
        if name == SCALAR_OP:
            scalars.append(event)
        if SYNC_API in name:
            synchronizations.append(event)
    for events in (graphs, scalars, synchronizations):
        events.sort(key=lambda event: event_span(event)[0])
    return {
        "complete_events": complete,
        "invalid_complete_events": invalid,
        "categories": dict(categories),
        "phases": dict(phases),
        "metadata": metadata,
        "graphs": graphs,
        "scalars": scalars,
        "synchronizations": synchronizations,
    }


def find_cycles(scan, *, steps_per_request=0, expected_requests=0, main_tid=None, scalar_min_ms=5.0):
    graphs = scan["graphs"]
    if main_tid is not None:
        graphs = [event for event in graphs if str(event.get("tid")) == str(main_tid)]
    lanes = {(str(event.get("pid")), str(event.get("tid"))) for event in graphs}
    if len(lanes) != 1:
        raise ValueError(f"Need one graph API lane, found {len(lanes)}; select --main-tid or one rank profile.")
    if steps_per_request and len(graphs) % steps_per_request:
        raise ValueError("Graph count is not divisible by --decode-steps-per-request; request grouping is unsafe.")
    if expected_requests and (not steps_per_request or len(graphs) != expected_requests * steps_per_request):
        raise ValueError("Graph count does not match expected requests * decode steps; do not force grouping.")
    tid = str(graphs[0].get("tid"))
    scalars = [event for event in scan["scalars"] if str(event.get("tid")) == tid]
    if len({str(event.get("pid")) for event in scalars}) > 1:
        raise ValueError("Ambiguous CPU processes with the same thread ID; use a single-worker trace.")
    cycles = []
    for index, (graph, following) in enumerate(zip(graphs, graphs[1:])):
        start, submitted = event_span(graph)
        stop = event_span(following)[0]
        if stop <= submitted:
            continue
        boundary = bool(steps_per_request and (index + 1) % steps_per_request == 0)
        candidates = [
            event
            for event in scalars
            if event_span(event)[0] >= submitted
            and event_span(event)[1] <= stop
            and event_span(event)[1] - event_span(event)[0] >= scalar_min_ms * 1000
        ]
        scalar = short_event(candidates[0]) if len(candidates) == 1 else None
        cycles.append(
            {
                "graph_index": index,
                "start_us": start,
                "end_us": stop,
                "duration_ms": round((stop - start) / 1000, 6),
                "graph_api": short_event(graph),
                "request_scope": "cross_request_by_user_grouping"
                if boundary
                else ("within_request_by_user_grouping" if steps_per_request else "request_boundary_unknown"),
                "scalar_candidates": len(candidates),
                "long_scalar": scalar,
                "graph_to_scalar_return_ms": None if scalar is None else round((scalar["end_us"] - start) / 1000, 6),
                "scalar_return_to_next_graph_ms": None
                if scalar is None
                else round((stop - scalar["end_us"]) / 1000, 6),
            }
        )
    return graphs, cycles


def csv_rows(path, required):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name}: missing columns {set(required) - set(reader.fieldnames or [])}.")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path.name}: incomplete CSV fields near line {reader.line_num}.")
            yield row


def ranked(groups, top):
    return sorted(groups, key=lambda row: row.get("time_ms", 0), reverse=True)[:top]


def kernel_summary(rows, bounds, top):
    intervals, streams, types, shapes = [], defaultdict(list), {}, {}
    start, stop = bounds
    for row in rows:
        span = (row["start"], row["end"])
        duration = overlap(span, bounds)
        if not duration:
            continue
        clip = (max(span[0], start), min(span[1], stop))
        intervals.append(clip)
        streams[(row["device"], row["stream"])].append(clip)
        for target, key in (
            (types, (row["type"], row["core"])),
            (shapes, (row["type"], row["core"], row["shape"], row["dtype"], row["format"], row["output_shape"])),
        ):
            if target is shapes and "matmul" not in (row["type"] + row["name"]).lower():
                continue
            if key not in target:
                target[key] = {"type": row["type"], "count": 0, "time_ms": 0.0}
                target[key]["core"] = row["core"]
                if target is shapes:
                    target[key].update({field: row[field] for field in ("shape", "dtype", "format", "output_shape")})
            target[key]["count"] += 1
            target[key]["time_ms"] += duration / 1000
    for target in (types, shapes):
        for value in target.values():
            value["time_ms"] = round(value["time_ms"], 6)
    union = interval_union(intervals)
    return {
        "task_count_intersecting": len(intervals),
        "task_duration_sum_ms": round(sum(b - a for a, b in intervals) / 1000, 6),
        "task_active_union_ms": round(union / 1000, 6),
        "no_recorded_task_ms": round((stop - start - union) / 1000, 6),
        "per_stream_union_ms": [
            {"device": device, "stream": stream, "time_ms": round(interval_union(parts) / 1000, 6)}
            for (device, stream), parts in streams.items()
        ],
        "types_top": ranked(list(types.values()), top),
        "matmul_shapes_top": ranked(list(shapes.values()), top),
        "largest_no_recorded_task_gaps": largest_gaps(intervals, bounds),
    }


def load_kernels(path, windows):
    rows, groups, devices, count, invalid = [], {}, set(), 0, 0
    for row in csv_rows(path, ("Name", "Type", "Start Time(us)", "Duration(us)")):
        count += 1
        try:
            start = finite_number(row["Start Time(us)"])
            duration = finite_number(row["Duration(us)"])
            if duration < 0:
                raise ValueError("negative duration")
        except (ValueError, TypeError):
            invalid += 1
            continue
        shape = row.get("Input Shapes", "").strip() or "<missing>"
        dtype = row.get("Input Data Types", "").strip() or "<missing>"
        core = row.get("Accelerator Core", "unknown").strip()
        input_format = row.get("Input Formats", "").strip() or "<missing>"
        output_shape = row.get("Output Shapes", "").strip() or "<missing>"
        device = row.get("Device_id", "unknown")
        devices.add(device)
        if "matmul" in (row["Name"] + row["Type"]).lower():
            key = (row["Type"], core, shape, dtype, input_format, output_shape)
            value = groups.setdefault(
                key,
                {
                    "type": row["Type"],
                    "core": core,
                    "shape": shape,
                    "dtype": dtype,
                    "format": input_format,
                    "output_shape": output_shape,
                    "count": 0,
                    "time_ms": 0.0,
                    "max_us": 0.0,
                },
            )
            value["count"] += 1
            value["time_ms"] += duration / 1000
            value["max_us"] = max(value["max_us"], duration)
        if any(overlap((start, start + duration), window) for window in windows):
            rows.append(
                {
                    "name": row["Name"],
                    "type": row["Type"],
                    "start": start,
                    "end": start + duration,
                    "shape": shape,
                    "dtype": dtype,
                    "device": device,
                    "stream": row.get("Stream ID", "unknown"),
                    "core": core,
                    "format": input_format,
                    "output_shape": output_shape,
                }
            )
    if len(devices) > 1:
        raise ValueError("kernel_details.csv contains multiple devices; supply a single-rank/device profile.")
    for value in groups.values():
        value["avg_us"] = round(value["time_ms"] * 1000 / value["count"], 6)
        value["time_ms"] = round(value["time_ms"], 6)
    return rows, {
        "rows": count,
        "invalid_rows": invalid,
        "devices": sorted(devices),
        "matmul_shapes_top": ranked(list(groups.values()), 12),
    }


def cpu_exclusive(events, bounds, top):
    """Innermost recorded cpu_op time, not CPU busy time. Never mix API PIDs.

    Sweep gives non-double-counted attribution for nested spans. Crossing spans
    are counted and explicitly make that attribution ambiguous.
    """
    lanes = defaultdict(list)
    for event in events:
        if event.get("cat") == "cpu_op" and (span := event_span(event)) and overlap(span, bounds):
            lanes[(str(event.get("pid")), str(event.get("tid")))].append(event)
    output = []
    for (pid, tid), lane in lanes.items():
        lane.sort(key=lambda event: (event_span(event)[0], -event_span(event)[1]))
        points, stack, crossings = [], [], 0
        for index, event in enumerate(lane):
            start, stop = event_span(event)
            while stack and stack[-1] <= start:
                stack.pop()
            if stack and stop > stack[-1]:
                crossings += 1
            stack.append(stop)
            points.extend(((max(start, bounds[0]), 1, index), (min(stop, bounds[1]), -1, index)))
        points.sort()
        active, heap, times, last = set(), [], defaultdict(float), bounds[0]
        for timestamp, action, index in points:
            while heap and heap[0][2] not in active:
                heapq.heappop(heap)
            if heap and timestamp > last:
                times[lane[heap[0][2]].get("name", "")] += timestamp - last
            if action == 1:
                active.add(index)
                start, stop = event_span(lane[index])
                heapq.heappush(heap, (stop - start, -start, index))
            else:
                active.discard(index)
            last = timestamp
        parts = clipped_intervals(lane, bounds)
        output.append(
            {
                "pid": pid,
                "tid": tid,
                "recorded_cpu_op_union_ms": round(interval_union(parts) / 1000, 6),
                "crossing_spans": crossings,
                "exclusive_recorded_ops_top": ranked(
                    [{"name": name, "time_ms": round(duration / 1000, 6)} for name, duration in times.items()], top
                ),
                "largest_uninstrumented_gaps": largest_gaps(parts, bounds),
            }
        )
    return sorted(output, key=lambda row: row["recorded_cpu_op_union_ms"], reverse=True)


def selected_trace(path, windows, slice_path=None, max_events=100000):
    events, copied = [], 0
    with slice_path.open("x", encoding="utf-8") if slice_path else _null_writer() as writer:
        if writer:
            writer.write('{"displayTimeUnit":"us","vq2_scope":"window_only_flows_may_be_incomplete","traceEvents":[')
        for event in trace_events(path):
            if not isinstance(event, dict):
                continue
            span = event_span(event)
            keep = bool(span and any(overlap(span, window) for window in windows))
            if keep:
                events.append(event)
                if len(events) > max_events:
                    raise ValueError(
                        "Selected windows exceed --max-window-events. "
                        "Choose a shorter window or explicitly raise the limit."
                    )
            if writer:
                point = event.get("ts")
                try:
                    point = finite_number(point)
                    in_window = any(a <= point <= b for a, b in windows)
                except (ValueError, TypeError):
                    in_window = False
                # Do not write orphan Begin/End scopes. Flow incompleteness is explicit.
                if keep or event.get("ph") == "M" or (not span and in_window and event.get("ph") not in ("B", "E")):
                    if copied:
                        writer.write(",")
                    writer.write(json.dumps(event, ensure_ascii=False, allow_nan=False))
                    copied += 1
        if writer:
            writer.write("]}\n")
    return events


class _null_writer:
    def __enter__(self):
        return None

    def __exit__(self, *_):
        return False


def api_top(events, bounds, top):
    groups = defaultdict(lambda: [0, 0.0])
    for event in events:
        name = str(event.get("name", ""))
        if "AscendCL@" not in name or not (span := event_span(event)):
            continue
        duration = overlap(span, bounds)
        if duration:
            groups[name][0] += 1
            groups[name][1] += duration / 1000
    return ranked(
        [{"name": name, "count": count, "time_ms": round(duration, 6)} for name, (count, duration) in groups.items()],
        top,
    )


def host_matmul_samples(events, bounds, limit=8):
    samples, seen = [], set()
    for event in events:
        if event.get("cat") != "cpu_op" or not event_span(event) or not overlap(event_span(event), bounds):
            continue
        if str(event.get("name", "")) not in ("aten::matmul", "aten::mm", "aten::bmm", "aten::linear", "aten::addmm"):
            continue
        args = event.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        picked = {
            key: value
            for key, value in args.items()
            if any(part in key.lower() for part in ("shape", "dim", "type", "stack", "correlation", "external id"))
        }
        signature = (event.get("name"), json.dumps(picked, sort_keys=True))
        if signature in seen:
            continue
        seen.add(signature)
        samples.append({**short_event(event), "args_excerpt": str(json.dumps(picked, ensure_ascii=False))[:1800]})
        if len(samples) >= limit:
            break
    return samples


def csv_top(path, time_field, top):
    if not path.exists():
        return {"available": False}
    rows = list(csv_rows(path, (time_field,)))
    rows.sort(key=lambda row: finite_number(row[time_field]), reverse=True)
    return {"available": True, "rows": rows[:top]}


def host_scope_samples(events, bounds, pid, tid, top):
    """Existing annotations may identify work outside recorded cpu_op ranges."""
    keywords = ("sampl", "schedul", "prepare", "metadata", "bookkeep", "output", "execute_model", "replay")
    samples = []
    for event in events:
        if str(event.get("pid")) != str(pid) or str(event.get("tid")) != str(tid):
            continue
        if not (span := event_span(event)) or not overlap(span, bounds):
            continue
        if any(word in str(event.get("name", "")).lower() for word in keywords):
            samples.append(
                {**short_event(event), "cat": event.get("cat"), "time_ms": round(overlap(span, bounds) / 1000, 6)}
            )
    return ranked(samples, top)


def build_report(profile, output, args):
    trace = profile / "trace_view.json"
    if not trace.is_file():
        trace = profile / "trace_view.json.gz"
    if not trace.is_file():
        raise ValueError("No trace_view.json or trace_view.json.gz in --profile-dir.")
    print("EXTRACT_STAGE=trace_inventory", flush=True)
    scan = scan_trace(trace)
    graphs, cycles = find_cycles(
        scan,
        steps_per_request=args.decode_steps_per_request,
        expected_requests=args.expected_requests,
        main_tid=args.main_tid,
    )
    candidates = [cycle for cycle in cycles if cycle["request_scope"] != "cross_request_by_user_grouping"]
    if args.cycle_index is not None:
        candidates = [cycle for cycle in candidates if cycle["graph_index"] == args.cycle_index]
    chosen = candidates[-args.cycles :]
    if not chosen:
        raise ValueError("No eligible adjacent replay window. Check graph count/grouping/--cycle-index.")
    if any(cycle["duration_ms"] > args.max_window_ms for cycle in chosen):
        raise ValueError(
            "Selected adjacent interval exceeds --max-window-ms; check request boundaries or choose --cycle-index."
        )
    windows = [(cycle["start_us"], cycle["end_us"]) for cycle in chosen]
    print("EXTRACT_STAGE=kernel_csv", flush=True)
    kernels, kernel_inventory = load_kernels(profile / "kernel_details.csv", windows)
    print("EXTRACT_STAGE=selected_trace_windows", flush=True)
    events = selected_trace(
        trace,
        windows,
        output / "trace_slice.json" if args.write_trace_slice else None,
        max_events=args.max_window_events,
    )
    warnings = [
        "This is a profiled replay-to-replay timeline, not HTTP TPOT; do not rescale it to 72 ms.",
        "No automatic prefill/decode/request inference. User grouping assumes sequential requests, "
        "complete capture, one decoder replay per decode.",
        "Kernel sums can overlap. Task union is recorded device-task coverage, "
        "not hardware utilization or a proven critical path.",
        "CPU-op exclusive spans include blocking waits; uninstrumented gaps do not prove CPU idle time.",
        "Scalar reads, stream synchronizations and device work overlap; never add them as independent costs.",
        "Shape grouping is not source attribution. "
        "Graph replay kernels may have no corresponding replay-time Python stack.",
        "Absolute timestamps are assumed microseconds on aligned host/device axes; "
        "do not use CSV/trace with different runs.",
    ]
    if not args.decode_steps_per_request:
        warnings.append(
            "REQUEST BOUNDARIES UNKNOWN: selected adjacent graphs may belong to different requests or prefill."
        )
    if not (profile / "analyse.done").exists():
        warnings.append("analyse.done is absent; export may be incomplete.")
    if scan["invalid_complete_events"] or kernel_inventory["invalid_rows"]:
        warnings.append("Invalid events/CSV timing rows were excluded; inspect inventory counts.")
    if scan["phases"].get("B") or scan["phases"].get("E"):
        warnings.append(
            "B/E scopes are not analyzed and are omitted from the optional slice; only complete X spans are counted."
        )
    if not kernels:
        warnings.append(
            "No kernel overlaps selected windows. Check timestamp alignment, profile completeness and paths."
        )
    sections = []
    for cycle in chosen:
        spans = [("full_replay_interval", (cycle["start_us"], cycle["end_us"]))]
        scalar = cycle["long_scalar"]
        if scalar:
            spans += [
                ("graph_to_scalar_return", (cycle["start_us"], scalar["end_us"])),
                ("post_scalar_to_next_graph", (scalar["end_us"], cycle["end_us"])),
            ]
        for name, bounds in spans:
            sections.append(
                {
                    "graph_index": cycle["graph_index"],
                    "name": name,
                    "start_us": bounds[0],
                    "end_us": bounds[1],
                    "wall_ms": round((bounds[1] - bounds[0]) / 1000, 6),
                    "device": kernel_summary(kernels, bounds, args.top),
                    "cpu_lanes": cpu_exclusive(events, bounds, args.top),
                    "native_api_inclusive_top": api_top(events, bounds, args.top),
                    "host_matmul_samples": host_matmul_samples(events, bounds),
                    "host_scope_samples": host_scope_samples(
                        events, bounds, scalar["pid"] if scalar else None, cycle["graph_api"]["tid"], args.top
                    ),
                }
            )
    return {
        "schema_version": 1,
        "profile_dir": str(profile),
        "trace_bytes": trace.stat().st_size,
        "analyse_done": (profile / "analyse.done").exists(),
        "warnings": warnings,
        "grouping": {
            "decode_steps_per_request": args.decode_steps_per_request,
            "expected_requests": args.expected_requests,
            "source": "user_assertion_not_request_ids",
            "graph_index_zero_based": True,
        },
        "trace_inventory": {
            key: value
            for key, value in scan.items()
            if key not in ("graphs", "scalars", "synchronizations", "metadata")
        },
        "track_metadata": scan["metadata"],
        "kernel_inventory": kernel_inventory,
        "graph_apis": [short_event(event) for event in graphs],
        "long_scalar_top": [
            short_event(event)
            for event in sorted(
                scan["scalars"], key=lambda event: event_span(event)[1] - event_span(event)[0], reverse=True
            )[:12]
        ],
        "synchronization_top": [
            short_event(event)
            for event in sorted(
                scan["synchronizations"], key=lambda event: event_span(event)[1] - event_span(event)[0], reverse=True
            )[:12]
        ],
        "adjacent_cycles": cycles,
        "selected_windows": sections,
        "global_op_top": csv_top(profile / "op_statistic.csv", "Total Time(us)", args.top),
        "global_api_top": csv_top(profile / "api_statistic.csv", "Time(us)", args.top),
    }


def render_text(report):
    lines = ["VQ2_PROFILE_EXTRACT v1", "PROFILE_DIR=" + report["profile_dir"]]

    def emit(label, value):
        lines.append(label + " " + json.dumps(value, ensure_ascii=False, separators=(",", ":")))

    emit(
        "INVENTORY",
        {
            **report["trace_inventory"],
            "kernel_rows": report["kernel_inventory"]["rows"],
            "analyse_done": report["analyse_done"],
        },
    )
    emit("GROUPING", report["grouping"])
    for warning in report["warnings"]:
        lines.append("NOTE " + warning)
    for row in report["adjacent_cycles"][-24:]:
        emit("CYCLE", {key: value for key, value in row.items() if key not in ("graph_api", "long_scalar")})
    for row in report["graph_apis"][-24:]:
        emit("GRAPH_API", row)
    for row in report["long_scalar_top"]:
        emit("LONG_SCALAR", row)
    for row in report["kernel_inventory"]["matmul_shapes_top"]:
        emit("GLOBAL_MATMUL_MIXED_PHASES", row)
    for section in report["selected_windows"]:
        emit("WINDOW", {key: section[key] for key in ("graph_index", "name", "start_us", "end_us", "wall_ms")})
        device = section["device"]
        emit(
            "DEVICE_COVERAGE",
            {
                key: device[key]
                for key in (
                    "task_count_intersecting",
                    "task_duration_sum_ms",
                    "task_active_union_ms",
                    "no_recorded_task_ms",
                )
            },
        )
        emit("DEVICE_STREAMS", device["per_stream_union_ms"])
        emit("DEVICE_UNCOVERED_GAPS", device["largest_no_recorded_task_gaps"][:3])
        for row in device["types_top"]:
            emit("DEVICE_TYPE", row)
        for row in device["matmul_shapes_top"]:
            emit("WINDOW_MATMUL", row)
        cycle = next(row for row in report["adjacent_cycles"] if row["graph_index"] == section["graph_index"])
        main_tid = str(cycle["graph_api"]["tid"])
        lanes = section["cpu_lanes"]
        main_lanes = [lane for lane in lanes if lane["tid"] == main_tid]
        for lane in main_lanes:
            emit(
                "MAIN_CPU_COVERAGE",
                {key: lane[key] for key in ("pid", "tid", "recorded_cpu_op_union_ms", "crossing_spans")},
            )
            for row in lane["exclusive_recorded_ops_top"]:
                emit("MAIN_CPU_RECORDED_EXCLUSIVE", row)
            emit("MAIN_CPU_UNINSTRUMENTED_GAPS", lane["largest_uninstrumented_gaps"][:3])
        emit(
            "OTHER_CPU_LANES",
            [
                {key: lane[key] for key in ("pid", "tid", "recorded_cpu_op_union_ms")}
                for lane in lanes
                if lane not in main_lanes
            ][:4],
        )
        for row in section["native_api_inclusive_top"]:
            emit("API_INCLUSIVE_NOT_ADDITIVE", row)
        for row in section["host_scope_samples"]:
            emit("HOST_SCOPE_NOT_ADDITIVE", row)
        # One copy of samples suffices; child windows overlap the full interval.
        if section["name"] == "full_replay_interval":
            for row in section["host_matmul_samples"]:
                emit("HOST_MATMUL_SAMPLE", row)
    lines.append("END_VQ2_PROFILE_EXTRACT")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True, help="One rank's ASCEND_PROFILER_OUTPUT directory.")
    parser.add_argument(
        "--output-dir", type=Path, help="New directory, never overwritten; default sibling timestamped directory."
    )
    parser.add_argument(
        "--decode-steps-per-request",
        type=int,
        default=0,
        help="Explicit grouping assumption, e.g. 3 for 4 output tokens; default unknown.",
    )
    parser.add_argument(
        "--expected-requests",
        type=int,
        default=0,
        help="Check graph count against the asserted sequential request count.",
    )
    parser.add_argument("--main-tid", help="Select one graph API thread if more than one is present.")
    parser.add_argument("--cycles", type=int, default=1, help="Last eligible adjacent cycles to inspect, 1..4.")
    parser.add_argument(
        "--cycle-index", type=int, help="Explicit zero-based start graph index; cross-request edges are excluded."
    )
    parser.add_argument("--top", type=int, default=8, help="Rows per ranking, 1..30.")
    parser.add_argument("--max-window-ms", type=float, default=2000, help="Reject unexpectedly long adjacent windows.")
    parser.add_argument(
        "--max-window-events",
        type=int,
        default=100000,
        help="Bound retained complete events; exceeding fails explicitly.",
    )
    parser.add_argument(
        "--write-trace-slice",
        action="store_true",
        help="Optional short trace; outside-window flow endpoints are not included.",
    )
    args = parser.parse_args(argv)
    if (
        args.decode_steps_per_request < 0
        or args.expected_requests < 0
        or not 1 <= args.cycles <= 4
        or not 1 <= args.top <= 30
    ):
        parser.error("Invalid grouping/cycles/top range.")
    if not math.isfinite(args.max_window_ms) or args.max_window_ms <= 0 or args.max_window_events <= 0:
        parser.error("Window limits must be positive and finite.")
    profile = args.profile_dir.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output_dir or profile.parent / ("vq2_extract_" + stamp)).resolve()
    if output == profile or output in profile.parents:
        parser.error("Output must not replace the profile directory or an ancestor.")
    try:
        output.mkdir(parents=True, exist_ok=False)
        report = build_report(profile, output, args)
        with (output / "summary.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        text = render_text(report)
        with (output / "paste_summary.txt").open("x", encoding="utf-8") as stream:
            stream.write(text)
        print(text, end="")
        print(f"EXTRACT_STATUS=PASS OUTPUT={output}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        print(f"EXTRACT_STATUS=FAIL ERROR={error} OUTPUT={output}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
