"""Generated safetensors fixtures for the read-only CPU bridge audit."""

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_vq2_bridge as audit  # noqa: E402


def fixture(prefix="gate_up", *, mixed=True):
    rng = np.random.default_rng(91)
    codes = rng.integers(0, 16, (16, 512), dtype=np.uint32)
    words = np.zeros((16, 64), dtype=np.uint32)
    for lane in range(8):
        words |= codes[:, lane::8] << (4 * lane)
    books = rng.integers(0, 127, (2, 1, 16, 2), dtype=np.uint8)
    books[0, 0, 0] = [128, 0]  # Signed zero bytes must survive.
    ids = np.tile(np.arange(2, dtype=np.uint8), 256) if mixed else np.repeat(np.arange(2, dtype=np.uint8), 256)
    name = lambda value: f"{prefix}.{value}" if prefix else value
    tensors = {
        name("packed_indices"): torch.from_numpy(words.view(np.int32)),
        name("codebooks"): torch.from_numpy(books).view(torch.float8_e4m3fn),
        name("codebook_tile_ids"): torch.from_numpy(ids),
    }
    return tensors


def batched_fixture():
    result = {}
    for projection in ("gate_up", "down"):
        for key, value in fixture(projection).items():
            batched = torch.stack((value, value))
            result[key.replace(f"{projection}.", f"{projection}_")] = batched
    return result


class BridgeAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="vq2-bridge-audit-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "expert.safetensors"

    def save(self, tensors=None):
        save_file(fixture() if tensors is None else tensors, str(self.path))
        return self.path

    def test_mixed_tile_ids_both_modes_roundtrip_and_input_untouched(self):
        self.save()
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        report = audit.audit_files([self.path])
        self.assertEqual(report["status"], "passed")
        matrix = report["files"][0]["matrices"][0]
        self.assertEqual(matrix["contract"]["mixed_k256_blocks"], 2)
        self.assertFalse(matrix["modes"]["preserve"]["fixed_k256_lut_available"])
        self.assertTrue(matrix["modes"]["codebook"]["k_reordered"])
        for result in matrix["modes"].values():
            self.assertTrue(result["codes_exact"])
            self.assertTrue(result["weight_bytes_exact"])
            self.assertTrue(result["source_order_roundtrip_exact"])
        self.assertEqual(before, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        for field in (
            "checkpoint_modified",
            "kernel_compiled",
            "device_execution_verified",
            "model_accuracy_verified",
            "drop_in_replacement_ready",
        ):
            self.assertFalse(report[field])

    def test_select_prefix_and_discover_multiple_matrices(self):
        self.save(fixture("gate_up") | fixture("down", mixed=False))
        report = audit.audit_files([self.path])
        self.assertEqual([value["prefix"] for value in report["files"][0]["matrices"]], ["down", "gate_up"])
        selected = audit.audit_files([self.path], ["down"])["files"][0]["matrices"]
        self.assertEqual(len(selected), 1)
        self.assertFalse(selected[0]["modes"]["codebook"]["k_reordered"])

    def test_root_tensor_keys_supported(self):
        self.save(fixture(""))
        self.assertEqual(audit.audit_files([self.path])["files"][0]["matrices"][0]["prefix"], "")

    def test_batched_layer_auto_discovery_and_explicit_expert(self):
        self.save(batched_fixture())
        report = audit.audit_files([self.path], expert_index=1)
        self.assertEqual(report["status"], "passed")
        matrices = report["files"][0]["matrices"]
        self.assertEqual([value["prefix"] for value in matrices], ["gate_up", "down"])
        self.assertTrue(all(value["expert_index"] == 1 for value in matrices))
        selected = audit.audit_files([self.path], projections=["down"])
        self.assertEqual(len(selected["files"][0]["matrices"]), 1)
        self.assertEqual(selected["files"][0]["matrices"][0]["expert_index"], 0)

    def test_batched_load_never_reads_whole_layer(self):
        from safetensors import safe_open

        self.save(batched_fixture())
        with safe_open(str(self.path), framework="pt", device="cpu") as handle:
            wrapper = mock.Mock()
            wrapper.keys.side_effect = handle.keys
            wrapper.get_slice.side_effect = handle.get_slice
            wrapper.get_tensor.side_effect = AssertionError("whole batched layer must not be read")
            result = audit.audit_matrix(wrapper, "gate_up", batched=True, expert_index=1)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(wrapper.get_slice.call_count, 3)
        wrapper.get_tensor.assert_not_called()

    def test_batched_invalid_expert_and_conflicting_selectors(self):
        self.save(batched_fixture())
        for kwargs in ({"expert_index": -1}, {"expert_index": 2}, {"prefixes": ["down"], "projections": ["down"]}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(audit.audit_files([self.path], **kwargs)["status"], "failed")

    def test_exact_source_dtypes_required(self):
        for name, dtype in [
            ("packed_indices", torch.int64),
            ("codebooks", torch.float32),
            ("codebook_tile_ids", torch.int32),
        ]:
            tensors = fixture()
            tensors[f"gate_up.{name}"] = tensors[f"gate_up.{name}"].to(dtype)
            self.save(tensors)
            with self.subTest(name=name):
                report = audit.audit_files([self.path])
                self.assertEqual(report["status"], "failed")
                self.assertIn("require CPU", report["files"][0]["matrices"][0]["error"])

    def test_missing_tensor_and_unknown_prefix_rejected(self):
        tensors = fixture()
        del tensors["gate_up.codebooks"]
        self.save(tensors)
        self.assertEqual(audit.audit_files([self.path])["status"], "failed")
        self.assertEqual(audit.audit_files([self.path], ["missing"])["status"], "failed")

    def test_malformed_tile_ids_and_nan_rejected(self):
        for kind in ("range", "shape", "nan"):
            tensors = fixture()
            if kind == "range":
                tensors["gate_up.codebook_tile_ids"][0] = 2
            elif kind == "shape":
                tensors["gate_up.codebook_tile_ids"] = tensors["gate_up.codebook_tile_ids"][:511]
            else:
                tensors["gate_up.codebooks"].view(torch.uint8)[0, 0, 0, 0] = 127
            self.save(tensors)
            with self.subTest(kind=kind):
                self.assertEqual(audit.audit_files([self.path])["status"], "failed")

    def test_non_groupable_contract_keeps_preserve_mode(self):
        tensors = fixture()
        tensors["gate_up.codebook_tile_ids"][0] = 1
        self.save(tensors)
        report = audit.audit_files([self.path])
        self.assertEqual(report["status"], "passed")
        matrix = report["files"][0]["matrices"][0]
        self.assertFalse(matrix["contract"]["stable_grouping_can_form_k256"])
        self.assertEqual(matrix["modes"]["codebook"]["status"], "not_representable")
        self.assertTrue(matrix["modes"]["preserve"]["weight_bytes_exact"])

    def test_corrupt_bridge_decode_is_detected(self):
        self.save()
        actual_decoder = audit.decode_bridge_bytes

        def bad_decoder(candidate):
            output = actual_decoder(candidate)
            output[0, 0] ^= 1
            return output

        with mock.patch.object(audit, "decode_bridge_bytes", bad_decoder):
            report = audit.audit_files([self.path])
        self.assertEqual(report["status"], "failed")
        self.assertIn("weight_bytes_exact=False", report["files"][0]["matrices"][0]["error"])

    def test_multiple_files_continue_after_missing_file(self):
        self.save()
        report = audit.audit_files([self.path.parent / "missing.safetensors", self.path])
        self.assertEqual(report["status"], "failed")
        self.assertEqual([value["status"] for value in report["files"]], ["failed", "passed"])

    def test_corrupt_safetensors_and_directory_fail_with_json_result(self):
        self.path.write_bytes(b"invalid safetensors fixture")
        for path in (self.path, self.path.parent):
            with self.subTest(path=path):
                report = audit.audit_files([path])
                self.assertEqual(report["status"], "failed")
                self.assertIn("error", report["files"][0])

    def test_n_groups_across_independent_decode_chunks(self):
        tensors = fixture()
        tensors["gate_up.packed_indices"] = tensors["gate_up.packed_indices"].repeat(3, 1)
        books = tensors["gate_up.codebooks"].view(torch.uint8).repeat(1, 3, 1, 1)
        books[:, 1] = 17
        books[:, 2] = 42
        tensors["gate_up.codebooks"] = books.view(torch.float8_e4m3fn)
        self.save(tensors)
        report = audit.audit_files([self.path])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["files"][0]["matrices"][0]["contract"]["n"], 96)

    def test_cli_json_and_exit_status(self):
        self.save()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = audit.main([str(self.path), "--prefix", "gate_up"])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "passed")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = audit.main([str(self.path), "--prefix", "missing"])
        self.assertEqual(result, 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "failed")

    def test_direct_scalar_formula_at_distinct_rows_and_columns(self):
        tensors = fixture()
        words = tensors["gate_up.packed_indices"].numpy()
        books = tensors["gate_up.codebooks"].view(torch.uint8).numpy()
        ids = tensors["gate_up.codebook_tile_ids"].numpy()
        decoded = audit.direct_decode_bytes(words, books, ids)
        for row in (0, 1, 15, 16, 30, 31):
            for column in (0, 1, 7, 8, 255, 256, 511):
                code = ((int(words[row // 2, column // 8]) & 0xFFFFFFFF) >> (4 * (column % 8))) & 15
                self.assertEqual(decoded[row, column], books[int(ids[column]), row // 32, code, row % 2])


if __name__ == "__main__":
    unittest.main()
