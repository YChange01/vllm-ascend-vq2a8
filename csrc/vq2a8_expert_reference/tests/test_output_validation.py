"""CPU tests for synthetic output acceptance; no NPU execution is implied."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import ml_dtypes
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_output  # noqa: E402


class ArrayValidationTests(unittest.TestCase):
    def test_exact_bf16_rounding_and_fp32_metrics_are_separate(self):
        golden = np.array([[1.001, -2.003, 0.0]], np.float32)
        output = golden.astype(ml_dtypes.bfloat16)
        report = check_output.compare_arrays(output, golden, rtol=0, atol=0)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["bf16_rounded_golden"]["bit_exact"])
        self.assertGreater(report["fp32_golden"]["max_abs_error"], 0)
        self.assertGreater(report["fp32_golden"]["mismatch_count_at_explicit_tolerances"], 0)
        self.assertFalse(report["model_quality_verified"])

    def test_actual_mismatch_fails_even_when_golden_rounds(self):
        golden = np.array([[1.001, 0.0]], np.float32)
        output = np.array([[2.0, 0.0]], dtype=ml_dtypes.bfloat16)
        report = check_output.compare_arrays(output, golden, rtol=0, atol=0)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["bf16_rounded_golden"]["mismatch_count_at_explicit_tolerances"], 1)

    def test_explicit_relative_plus_absolute_boundary(self):
        golden = np.array([[2.0, 0.0]], np.float32)
        output = np.array([[2.25, 0.125]], dtype=ml_dtypes.bfloat16)
        self.assertEqual(check_output.compare_arrays(output, golden, rtol=0.0625, atol=0.125)["status"], "passed")
        self.assertEqual(check_output.compare_arrays(output, golden, rtol=0, atol=0.125)["status"], "failed")

    def test_invalid_tolerances_rejected(self):
        golden = np.zeros((1, 2), np.float32)
        output = golden.astype(ml_dtypes.bfloat16)
        for tolerance in (-1, float("nan"), float("inf"), True):
            for key in ("rtol", "atol"):
                values = {"rtol": 0.0, "atol": 0.0, key: tolerance}
                with self.subTest(key=key, tolerance=tolerance), self.assertRaises(ValueError):
                    check_output.compare_arrays(output, golden, **values)

    def test_nonfinite_output_and_golden_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            golden = np.array([[0.0, 1.0]], np.float32)
            output = golden.astype(ml_dtypes.bfloat16)
            for location in ("output", "golden"):
                test_output, test_golden = output.copy(), golden.copy()
                (test_output if location == "output" else test_golden)[0, 0] = value
                with self.subTest(value=value, location=location), self.assertRaises(ValueError):
                    check_output.compare_arrays(test_output, test_golden, rtol=0, atol=0)

    def test_fp32_overflow_to_bf16_rejected(self):
        golden = np.array([[np.finfo(np.float32).max]], np.float32)
        output = np.zeros((1, 1), dtype=ml_dtypes.bfloat16)
        with self.assertRaisesRegex(ValueError, "overflows"):
            check_output.compare_arrays(output, golden, rtol=1, atol=1)

    def test_dtype_and_shape_are_not_implicitly_coerced(self):
        golden = np.ones((1, 2), np.float32)
        output = golden.astype(ml_dtypes.bfloat16)
        for bad_output, bad_golden in ((golden, golden), (output, golden.astype(np.float64))):
            with self.assertRaises(TypeError):
                check_output.compare_arrays(bad_output, bad_golden, rtol=0, atol=0)
        for bad_output, bad_golden in (
            (output.reshape(-1), golden.reshape(-1)),
            (output, golden.T),
            (output[:, :0], golden[:, :0]),
        ):
            with self.assertRaises(ValueError):
                check_output.compare_arrays(bad_output, bad_golden, rtol=0, atol=0)

    def test_signed_zero_reports_numeric_and_bitwise_distinction(self):
        output = np.array([[-0.0]], dtype=ml_dtypes.bfloat16)
        golden = np.array([[0.0]], np.float32)
        report = check_output.compare_arrays(output, golden, rtol=0, atol=0)
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["bf16_rounded_golden"]["bit_exact"])
        self.assertIsNone(report["bf16_rounded_golden"]["max_relative_error_nonzero_reference"])
        json.dumps(report, allow_nan=False)


class FileValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vq2-output-check-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.metadata = self.directory / "metadata.json"
        self.output = self.directory / "output_c.bin"
        self.golden = self.directory / "golden_c.bin"
        self.metadata_object = {
            "output_dtype": "bfloat16",
            "golden_output_dtype": "float32",
            "output_shape": [1, 2],
            "total_m": 1,
            "n": 2,
        }
        self.metadata.write_text(json.dumps(self.metadata_object), encoding="utf-8")
        golden = np.array([[1.001, 0.0]], np.float32)
        golden.astype("<f4").tofile(self.golden)
        golden.astype(ml_dtypes.bfloat16).view(np.uint16).astype("<u2").tofile(self.output)

    def _args(self):
        return ["--metadata", str(self.metadata), "--output", str(self.output), "--golden", str(self.golden)]

    def _main(self, args):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = check_output.main(args)
        return code, json.loads(stdout.getvalue())

    def test_valid_files_are_read_only_and_scoped(self):
        paths = (self.metadata, self.output, self.golden)
        original = {path: path.read_bytes() for path in paths}
        code, report = self._main(self._args() + ["--rtol", "0", "--atol", "0"])
        self.assertEqual(code, 0)
        self.assertEqual(report["scope"], "synthetic_kernel_output_not_model_quality")
        self.assertEqual(original, {path: path.read_bytes() for path in paths})

    def test_missing_explicit_tolerances_is_json_failure(self):
        code, report = self._main(self._args())
        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "failed")

    def test_missing_and_wrong_size_files_fail(self):
        for path in (self.output, self.golden):
            original = path.read_bytes()
            for bad in (original[:-1], original + b"\x00"):
                path.write_bytes(bad)
                with self.assertRaisesRegex(ValueError, "bytes"):
                    check_output.check_files(self.metadata, self.output, self.golden, rtol=0, atol=0)
            path.write_bytes(original)
        self.output.unlink()
        code, report = self._main(self._args() + ["--rtol", "0", "--atol", "0"])
        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "failed")

    def test_malformed_metadata_and_contract_mismatch_fail(self):
        for change in (
            {"output_shape": [True, 2]},
            {"output_shape": [0, 2]},
            {"total_m": 2},
            {"n": 3},
            {"output_dtype": "float16"},
            {"golden_output_dtype": "float64"},
        ):
            self.metadata.write_text(json.dumps({**self.metadata_object, **change}), encoding="utf-8")
            with self.subTest(change=change), self.assertRaises(ValueError):
                check_output.check_files(self.metadata, self.output, self.golden, rtol=0, atol=0)
        self.metadata.write_text("not json", encoding="utf-8")
        code, report = self._main(self._args() + ["--rtol", "0", "--atol", "0"])
        self.assertEqual(code, 2)
        self.assertIn("error", report)

    def test_mismatch_and_nonfinite_files_return_nonzero(self):
        for value, expected_exit in ((2.0, 1), (float("nan"), 2)):
            np.array([[value, 0]], dtype=ml_dtypes.bfloat16).view(np.uint16).astype("<u2").tofile(self.output)
            code, report = self._main(self._args() + ["--rtol", "0", "--atol", "0"])
            self.assertEqual(code, expected_exit)
            self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
