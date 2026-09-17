# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone stdlib tests: no pytest, torch, vLLM, or NPU required."""

import contextlib
import csv
import gzip
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[3] / "tools" / "extract_vq2a8_profile.py"
SPEC = importlib.util.spec_from_file_location("vq2_profile_extract", SCRIPT)
extract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extract)


def event(name, start, duration, *, pid=10, tid=10, cat="cpu_op", **fields):
    return {"ph": "X", "name": name, "ts": start, "dur": duration, "pid": pid, "tid": tid, "cat": cat, **fields}


def fixture_events():
    events = [{"ph": "M", "pid": 10, "name": "process_name", "args": {"name": "EngineCore"}}]
    for start in (0, 80000, 160000, 500000, 580000, 660000):
        events.extend(
            [
                event("AscendCL@aclmdlRIExecuteAsync", start, 30, pid=999, cat="acl"),
                event("aten::item", start + 1000, 39000),
                event("aten::_local_scalar_dense", start + 1000, 39000),
                event("AscendCL@aclrtSynchronizeStreamWithTimeout", start + 1010, 38900, pid=999, cat="acl"),
                event("sample_token", start + 41000, 1000, cat="user_annotation"),
                event("aten::matmul", start + 42000, 2000, args={"Input Dims": [[1, 128], [128, 128]]}),
                event("aten::mm", start + 42100, 1700),
            ]
        )
    return events


class ExtractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_trace(self, data, *, compressed=False):
        path = self.root / ("trace_view.json.gz" if compressed else "trace_view.json")
        text = json.dumps(data, ensure_ascii=False)
        if compressed:
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write(text)
        else:
            path.write_text(text, encoding="utf-8")
        return path

    def write_kernels(self, entries=None):
        entries = entries or [
            [
                "1",
                "MatMul_test",
                "MatMulV3",
                "AI_CORE",
                "581000\n",
                "35000",
                "4",
                '"1,128;128,128"',
                "FLOAT",
                "ND",
                "1,128",
            ],
            [
                "1",
                "MatMul_test",
                "MatMulV3",
                "MIX_AIC",
                "582000",
                "30000",
                "5",
                '"1,128;128,128"',
                "FLOAT",
                "ND",
                "1,128",
            ],
            ["1", "Cast_test", "Cast", "AI_VECTOR_CORE", "625000", "1000", "4", "1,128", "FLOAT", "ND", "1,128"],
        ]
        path = self.root / "kernel_details.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "Device_id",
                    "Name",
                    "Type",
                    "Accelerator Core",
                    "Start Time(us)",
                    "Duration(us)",
                    "Stream ID",
                    "Input Shapes",
                    "Input Data Types",
                    "Input Formats",
                    "Output Shapes",
                ]
            )
            writer.writerows(entries)
        return path

    def test_stream_formats_and_tiny_chunk_boundaries(self):
        events = [event('escaped \\" \u4f60\u597d', 12, 3, args={"text": "brackets ] } and commas ,"})]
        for chunk in (1, 2, 3, 7, 31):
            for data in (events, {"other": [1, 2], "traceEvents": events, "tail": True}):
                with self.subTest(chunk=chunk, data=type(data)), patch.object(extract, "CHUNK_BYTES", chunk):
                    self.assertEqual(list(extract.trace_events(self.write_trace(data))), events)

    def test_stream_scientific_numbers_crossing_chunks(self):
        path = self.root / "numbers.json"
        for number in ("1e+30", "-1.25e-10", "1.25", "123456"):
            path.write_text('{"other":' + number + ',"traceEvents":[]}', encoding="utf-8")
            with patch.object(extract, "CHUNK_BYTES", 1):
                self.assertEqual(list(extract.trace_events(path)), [])

    def test_gzip_trace(self):
        events = fixture_events()
        self.assertEqual(list(extract.trace_events(self.write_trace({"traceEvents": events}, compressed=True))), events)

    def test_malformed_trace_rejected(self):
        for text in (
            '{"traceEvents":[',
            '[{"ph":"X"}',
            '{"x":[]}',
            "[{},]",
            "{} trailing",
            '{"traceEvents":[],"traceEvents":[]}',
        ):
            with self.subTest(text=text):
                path = self.root / "bad.json"
                path.write_text(text)
                with self.assertRaises(ValueError):
                    list(extract.trace_events(path))

    def test_inventory_and_synthetic_api_pid(self):
        scan = extract.scan_trace(self.write_trace({"traceEvents": fixture_events()}))
        graphs, cycles = extract.find_cycles(scan, steps_per_request=3, expected_requests=2)
        self.assertEqual(len(graphs), 6)
        self.assertEqual(cycles[2]["request_scope"], "cross_request_by_user_grouping")
        self.assertEqual(cycles[-1]["long_scalar"]["pid"], 10)
        self.assertEqual(cycles[-1]["graph_to_scalar_return_ms"], 40)
        self.assertEqual(cycles[-1]["scalar_return_to_next_graph_ms"], 40)

    def test_no_request_grouping_is_not_invented(self):
        scan = extract.scan_trace(self.write_trace(fixture_events()))
        _, cycles = extract.find_cycles(scan)
        self.assertTrue(all(row["request_scope"] == "request_boundary_unknown" for row in cycles))

    def test_count_mismatch_rejected(self):
        scan = extract.scan_trace(self.write_trace(fixture_events()))
        for steps, requests in ((4, 0), (3, 3), (0, 2)):
            with self.subTest(steps=steps, requests=requests), self.assertRaises(ValueError):
                extract.find_cycles(scan, steps_per_request=steps, expected_requests=requests)

    def test_multiple_graph_lanes_require_selection(self):
        events = fixture_events() + [event("aclmdlRIExecuteAsync", 5, 1, pid=999, tid=20)]
        scan = extract.scan_trace(self.write_trace(events))
        with self.assertRaises(ValueError):
            extract.find_cycles(scan)
        graphs, _ = extract.find_cycles(scan, main_tid="10")
        self.assertEqual(len(graphs), 6)

    def test_ambiguous_long_scalars_are_not_picked(self):
        events = fixture_events() + [event("aten::_local_scalar_dense", 45000, 10000)]
        scan = extract.scan_trace(self.write_trace(events))
        _, cycles = extract.find_cycles(scan)
        self.assertEqual(cycles[0]["scalar_candidates"], 2)
        self.assertIsNone(cycles[0]["long_scalar"])

    def test_interval_union_overlap(self):
        self.assertEqual(extract.interval_union([(1, 5), (2, 4), (4, 8), (10, 12)]), 9)
        self.assertEqual(extract.interval_union([]), 0)

    def test_cpu_exclusive_does_not_double_count(self):
        events = [
            event("parent", 0, 100),
            event("child", 20, 60),
            event("grandchild", 30, 20),
            event("AscendCL@wait", 10, 70, pid=999, cat="acl"),
        ]
        lane = extract.cpu_exclusive(events, (0, 100), 10)[0]
        values = {row["name"]: row["time_ms"] for row in lane["exclusive_recorded_ops_top"]}
        self.assertEqual(values, {"parent": 0.04, "child": 0.04, "grandchild": 0.02})
        self.assertEqual(lane["recorded_cpu_op_union_ms"], 0.1)
        self.assertEqual(lane["crossing_spans"], 0)

    def test_cpu_window_clipping_and_thread_separation(self):
        events = [event("parent", 0, 100), event("child", 20, 60), event("other", 10, 100, tid=20)]
        lanes = extract.cpu_exclusive(events, (50, 90), 10)
        self.assertEqual(len(lanes), 2)
        main = next(lane for lane in lanes if lane["tid"] == "10")
        self.assertEqual(
            {row["name"]: row["time_ms"] for row in main["exclusive_recorded_ops_top"]}, {"child": 0.03, "parent": 0.01}
        )

    def test_crossing_cpu_spans_are_flagged(self):
        lane = extract.cpu_exclusive([event("a", 0, 60), event("b", 40, 60)], (0, 100), 10)[0]
        self.assertEqual(lane["crossing_spans"], 1)
        self.assertAlmostEqual(sum(row["time_ms"] for row in lane["exclusive_recorded_ops_top"]), 0.1)

    def test_kernel_csv_multiline_and_core_separation(self):
        rows, inventory = extract.load_kernels(self.write_kernels(), [(580000, 660000)])
        self.assertEqual(inventory["rows"], 3)
        self.assertEqual(len(inventory["matmul_shapes_top"]), 2)
        summary = extract.kernel_summary(rows, (580000, 660000), 8)
        self.assertEqual(summary["task_duration_sum_ms"], 66)
        self.assertEqual(summary["task_active_union_ms"], 36)
        self.assertEqual(summary["no_recorded_task_ms"], 44)
        self.assertEqual(len(summary["matmul_shapes_top"]), 2)

    def test_bad_csv_field_count_rejected(self):
        path = self.root / "bad.csv"
        path.write_text("A,B\n1\n")
        with self.assertRaises(ValueError):
            list(extract.csv_rows(path, ("A", "B")))

    def test_slice_preserves_complete_events_but_omits_orphan_scopes(self):
        events = [
            event("parent", 0, 200),
            {"ph": "B", "ts": 70, "name": "orphan"},
            {"ph": "M", "name": "process_name", "pid": 1},
            {"ph": "s", "ts": 80, "id": 1},
        ]
        path = self.write_trace(events)
        output = self.root / "slice.json"
        selected = extract.selected_trace(path, [(50, 100)], output)
        self.assertEqual(selected, [events[0]])
        result = json.loads(output.read_text())
        self.assertNotIn("B", [row["ph"] for row in result["traceEvents"]])
        self.assertEqual(result["traceEvents"][0]["dur"], 200)
        self.assertIn("incomplete", result["vq2_scope"])

    def test_window_event_limit_fails_explicitly(self):
        path = self.write_trace([event("a", 0, 100), event("b", 0, 100)])
        with self.assertRaisesRegex(ValueError, "max-window-events"):
            extract.selected_trace(path, [(0, 100)], max_events=1)

    def test_none_args_are_safe(self):
        values = extract.host_matmul_samples([event("aten::matmul", 0, 100, args=None)], (0, 100))
        self.assertEqual(values[0]["args_excerpt"], "{}")

    def test_cli_end_to_end_and_never_overwrite(self):
        trace = self.write_trace({"traceEvents": fixture_events()})
        self.write_kernels()
        (self.root / "analyse.done").touch()
        before = trace.read_bytes()
        output = self.root / "extract"
        command = [
            "--profile-dir",
            str(self.root),
            "--output-dir",
            str(output),
            "--decode-steps-per-request",
            "3",
            "--expected-requests",
            "2",
            "--write-trace-slice",
        ]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(extract.main(command), 0)
            self.assertEqual(extract.main(command), 1)
        self.assertEqual(trace.read_bytes(), before)
        report = json.loads((output / "summary.json").read_text())
        self.assertEqual(report["selected_windows"][0]["graph_index"], 4)
        self.assertEqual(len(report["selected_windows"]), 3)
        self.assertIn("post_scalar_to_next_graph", (output / "paste_summary.txt").read_text())
        self.assertEqual(report["selected_windows"][0]["device"]["task_active_union_ms"], 36)

    def test_cli_rejects_long_window(self):
        self.write_trace(fixture_events())
        self.write_kernels()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = extract.main(["--profile-dir", str(self.root), "--max-window-ms", "10"])
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
