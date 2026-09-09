"""CPU source-contract checks; these do not compile or execute CANN code."""

import unittest
from pathlib import Path


def _braced_body(source: str, marker: str) -> tuple[int, int, str]:
    """Extract one balanced C++ block from the fixed, reviewed host fixture."""
    start = source.index("{", source.index(marker))
    depth = 1
    for end in range(start + 1, len(source)):
        if source[end] == "{":
            depth += 1
        elif source[end] == "}":
            depth -= 1
            if depth == 0:
                return start, end, source[start + 1 : end]
    raise AssertionError(f"Unbalanced block after {marker!r}")


class HostTimingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (Path(__file__).resolve().parents[1] / "mat_fp4_host.cc").read_text(encoding="utf-8")
        _, _, cls.run_test = _braced_body(cls.source, "void RunTest(")

    def test_events_enclose_entire_timed_loop(self):
        begin, end, loop = _braced_body(self.run_test, "cycle < test_cycles")
        before, after = self.run_test[:begin], self.run_test[end + 1 :]
        self.assertEqual(self.run_test.count("start_event.Record(stream);"), 1)
        self.assertEqual(self.run_test.count("end_event.Record(stream);"), 1)
        self.assertIn("start_event.Record(stream);", before)
        self.assertIn("end_event.Record(stream);", after)
        self.assertEqual(loop.count("LaunchGroupPass("), 1)
        self.assertNotIn(".Record(", loop)

    def test_timed_loop_uses_resident_inputs(self):
        _, _, loop = _braced_body(self.run_test, "cycle < test_cycles")
        self.assertNotIn("CopyHostToDevice", loop)
        self.assertNotIn("rtMemcpy", loop)
        self.assertNotIn("Synchronize", loop)
        warmup_sync = self.run_test.index('stream.Synchronize("after warmup kernels")')
        start_event = self.run_test.index("start_event.Record(stream);")
        self.assertLess(warmup_sync, start_event)

    def test_elapsed_checked_after_batch_completion(self):
        end_event = self.run_test.index("end_event.Record(stream);")
        final_sync = self.run_test.index('stream.Synchronize("after timed kernels")')
        elapsed = self.run_test.index("start_event.ElapsedMillisecondsTo(end_event)")
        guard = self.run_test.index("!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0")
        average = self.run_test.index("elapsed_ms / static_cast<double>(test_cycles)")
        self.assertLess(end_event, final_sync)
        self.assertLess(final_sync, elapsed)
        self.assertLess(elapsed, guard)
        self.assertLess(guard, average)
        self.assertIn("if (test_cycles == 0)", self.run_test)
        self.assertIn("#include <cmath>", self.source)

    def test_report_does_not_claim_end_to_end_or_accuracy_pass(self):
        self.assertIn("resident kernel batch average", self.run_test)
        self.assertIn("excludes H2D/D2H and model preparation, not E2E", self.run_test)
        self.assertIn("accuracy verification: NOT PERFORMED", self.run_test)
        self.assertIn("not per-group measured latency", self.run_test)


if __name__ == "__main__":
    unittest.main()
