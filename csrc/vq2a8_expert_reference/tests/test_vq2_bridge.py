"""CPU-only layout/math tests. No CCE build, NPU execution or model acceptance."""

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import ml_dtypes
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vq2_bridge as bridge  # noqa: E402


def fixture():
    rng = np.random.default_rng(830)
    indices = rng.integers(0, 16, (32, 1024), dtype=np.uint8)
    words = np.zeros((32, 128), dtype=np.uint32)
    for nibble in range(8):
        words |= indices[:, nibble::8].astype(np.uint32) << np.uint32(4 * nibble)
    ids = np.tile(np.arange(4, dtype=np.uint8), 256)
    # Arbitrary non-Cartesian pairs, >4 distinct finite values; retain sign bits.
    books = rng.integers(0, 256, (4, 2, 16, 2), dtype=np.uint8) & np.uint8(0xFE)
    return indices, words.view(np.int32), books, ids


def literal_decode(indices, books, ids):
    result = np.empty((indices.shape[0] * 2, indices.shape[1]), np.uint8)
    for pair in range(indices.shape[0]):
        for k in range(indices.shape[1]):
            result[2 * pair : 2 * pair + 2, k] = books[ids[k], pair // 16, indices[pair, k]]
    return result


class VQ2BridgeTests(unittest.TestCase):
    def setUp(self):
        self.indices, self.words, self.books, self.ids = fixture()

    def test_direct_unpack_includes_negative_int32_words(self):
        self.assertTrue((self.words < 0).any())
        np.testing.assert_array_equal(bridge.unpack_direct_words(self.words), self.indices)

    def test_pack_address_and_roundtrip(self):
        packed = bridge.pack_pairs_zn(self.indices)
        self.assertEqual(packed.shape, (2, 64, 16, 8))
        self.assertEqual(packed.nbytes, self.words.nbytes)
        for n in range(0, 64, 4):
            for k in (0, 15, 16, 255, 256, 1023):
                expected = int(self.indices[n // 2, k]) | (int(self.indices[n // 2 + 1, k]) << 4)
                self.assertEqual(int(packed[n // 32, k // 16, k % 16, (n % 32) // 4]), expected)
        np.testing.assert_array_equal(bridge.unpack_pairs_zn(packed), self.indices)

    def test_preserve_mode_keeps_all_weight_bytes_and_tile_ids(self):
        result = bridge.bridge_direct_weights(self.words, self.books, self.ids)
        np.testing.assert_array_equal(
            bridge.decode_bridge_bytes(result), literal_decode(self.indices, self.books, self.ids)
        )
        np.testing.assert_array_equal(result.codebook_tile_ids, self.ids)
        self.assertIsNone(result.fixed_k256_lut)
        self.assertFalse(result.k_reordered)

    def test_arbitrary_vq_not_four_scalar_levels(self):
        self.assertGreater(np.unique(self.books[0, 0]).size, 4)
        result = bridge.bridge_direct_weights(self.words, self.books, self.ids)
        np.testing.assert_array_equal(result.codebook_bytes, self.books)

    def test_codebook_order_lossless_weight_roundtrip(self):
        result = bridge.bridge_direct_weights(self.words, self.books, self.ids, k_order="codebook")
        order = np.argsort(self.ids, kind="stable")
        np.testing.assert_array_equal(result.activation_gather, order)
        np.testing.assert_array_equal(result.codebook_tile_ids, np.repeat(np.arange(4), 256))
        np.testing.assert_array_equal(result.fixed_k256_lut, self.books.reshape(4, 2, 32))
        restored = bridge.decode_bridge_bytes(result)[:, np.argsort(order)]
        np.testing.assert_array_equal(restored, literal_decode(self.indices, self.books, self.ids))

    def test_uniform_physical_blocks_do_not_require_sort(self):
        ids = np.repeat(np.array([2, 0, 3, 1], np.uint8), 256)
        result = bridge.bridge_direct_weights(self.words, self.books, ids)
        np.testing.assert_array_equal(result.fixed_k256_lut, self.books[[2, 0, 3, 1]].reshape(4, 2, 32))
        self.assertFalse(result.k_reordered)

    def test_fixed_lut_expert_indexing_matches_source_weight_bytes(self):
        result = bridge.bridge_direct_weights(self.words, self.books, self.ids, k_order="codebook")
        indices = bridge.unpack_pairs_zn(result.packed_zn)
        fixed = result.fixed_k256_lut.reshape(4, 2, 16, 2)
        actual = np.empty((64, 1024), np.uint8)
        for pair in range(32):
            for k in range(1024):
                actual[2 * pair : 2 * pair + 2, k] = fixed[k // 256, pair // 16, indices[pair, k]]
        expected = literal_decode(self.indices, self.books, self.ids)[:, result.activation_gather]
        np.testing.assert_array_equal(actual, expected)

    def test_mutated_bridge_plan_rejected_before_numpy_gather(self):
        result = bridge.bridge_direct_weights(self.words, self.books, self.ids, k_order="codebook")
        for bad_index in (-1, 1024, int(result.activation_gather[1])):
            order = result.activation_gather.copy()
            order[0] = bad_index
            with self.assertRaises(ValueError):
                bridge.decode_bridge_bytes(replace(result, activation_gather=order))
        fixed = result.fixed_k256_lut.copy()
        fixed[0, 0, 0] ^= np.uint8(1)
        with self.assertRaises(ValueError):
            bridge.decode_bridge_bytes(replace(result, fixed_k256_lut=fixed))
        with self.assertRaises(ValueError):
            bridge.bridge_prepared_activation(
                np.zeros((1, 1024), np.uint8),
                np.ones(1, np.float32),
                np.zeros(1, np.float32),
                replace(result, activation_gather=np.zeros(1024, np.int64)),
            )

    def test_prepared_fp8_gather_preserves_quantization_and_row_metadata(self):
        result = bridge.bridge_direct_weights(self.words, self.books, self.ids, k_order="codebook")
        source = np.tile(np.arange(1024, dtype=np.uint16) % 126, (3, 1)).astype(np.uint8)
        scale = np.array([0.03, 1.3, 0.07], np.float32)  # not E8M0 powers of two
        bias = np.array([-0.2, 0.25, 0.1], np.float32)
        target, mx, kept_scale, kept_bias = bridge.bridge_prepared_activation(source, scale, bias, result)
        np.testing.assert_array_equal(target[:, np.argsort(result.activation_gather)], source)
        np.testing.assert_array_equal(mx, np.full((3, 32), 127, np.uint8))
        np.testing.assert_array_equal(kept_scale.view(np.uint8), scale.view(np.uint8))
        np.testing.assert_array_equal(kept_bias.view(np.uint8), bias.view(np.uint8))

    def test_algebra_after_preparation_gather_not_before_rht(self):
        # Here inputs are already prepared FP8 bytes. The bridge intentionally
        # accepts no raw activations/rht_sign: a gather BEFORE RHT128 is invalid.
        result = bridge.bridge_direct_weights(self.words, self.books, self.ids, k_order="codebook")
        fp8 = ml_dtypes.float8_e4m3fn
        source = np.random.default_rng(9).choice([-1, 0, 1], (2, 1024)).astype(fp8)
        scale, bias = np.array([0.3, 0.7], np.float32), np.array([0.1, -0.4], np.float32)
        target, _, _, _ = bridge.bridge_prepared_activation(source.view(np.uint8), scale, bias, result)
        original_weight = literal_decode(self.indices, self.books, self.ids).view(fp8).astype(np.float64)
        target_weight = bridge.decode_bridge_bytes(result).view(fp8).astype(np.float64)
        # FP64 and bounded dyadic fixtures prove algebra, NOT NPU FP32 bit-exactness.
        np.testing.assert_array_equal(
            source.astype(np.float64) @ original_weight.T, target.view(fp8).astype(np.float64) @ target_weight.T
        )

    def test_fp32_epilogue_must_precede_bf16_rounding(self):
        accumulator = np.array([[1.00390625]], np.float32)
        scale, bias = np.array([3], np.float32), np.array([0], np.float32)
        correct = bridge.fp32_epilogue(accumulator, scale, bias)
        wrong = bridge.fp32_epilogue(accumulator.astype(ml_dtypes.bfloat16).astype(np.float32), scale, bias)
        self.assertEqual(correct.dtype, np.float32)
        self.assertNotEqual(correct.astype(ml_dtypes.bfloat16).item(), wrong.astype(ml_dtypes.bfloat16).item())

    def test_input_arrays_not_modified_or_shared(self):
        sources = [self.words, self.books, self.ids]
        before = [x.copy() for x in sources]
        result = bridge.bridge_direct_weights(*sources, k_order="codebook")
        result.packed_zn.fill(0)
        result.codebook_bytes.fill(0)
        result.codebook_tile_ids.fill(0)
        for source, snapshot in zip(sources, before):
            np.testing.assert_array_equal(source, snapshot)

    def test_audit_fail_closed_and_actual_shape_constraint(self):
        report = bridge.audit_direct_contract(self.words, self.books, self.ids)
        self.assertEqual(report["mixed_k256_blocks"], 4)
        self.assertEqual(report["max_codebooks_per_k256"], 4)
        self.assertTrue(report["stable_grouping_can_form_k256"])
        self.assertFalse(report["original_host_shape_supported"])  # N64 is only a CPU fixture
        self.assertFalse(report["drop_in_replacement_ready"])
        self.assertFalse(report["device_execution_verified"])

    def test_strict_types_before_cast(self):
        for value in (self.indices.astype(np.int16), self.indices.astype(np.float32), self.indices.tolist()):
            with self.subTest(dtype=type(value)), self.assertRaises(TypeError):
                bridge.pack_pairs_zn(value)
        with self.assertRaises(TypeError):
            bridge.bridge_direct_weights(self.words.astype(np.int64), self.books, self.ids)
        with self.assertRaises(TypeError):
            bridge.bridge_direct_weights(self.words, self.books.astype(np.int16), self.ids)

    def test_pair_range_and_shapes(self):
        bad = self.indices.copy()
        bad[0, 0] = 16
        with self.assertRaises(ValueError):
            bridge.pack_pairs_zn(bad)
        for value in (self.indices[:31], self.indices[:, :1000], np.empty((0, 1024), np.uint8)):
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                bridge.pack_pairs_zn(value)
        with self.assertRaises(ValueError):
            bridge.unpack_pairs_zn(np.zeros((2, 64, 16, 7), np.uint8))

    def test_codebook_ids_and_nan_rejected(self):
        for ids in (self.ids[:-1], np.full(1024, 4, np.uint8)):
            with self.assertRaises(ValueError):
                bridge.bridge_direct_weights(self.words, self.books, ids)
        for nan in (127, 255):
            books = self.books.copy()
            books[0, 0, 0, 0] = nan
            with self.assertRaises(ValueError):
                bridge.bridge_direct_weights(self.words, books, self.ids)

    def test_nonuniform_population_requires_per_column_ids(self):
        ids = self.ids.copy()
        ids[0] = 1  # 255 / 257 / 256 / 256
        bridge.bridge_direct_weights(self.words, self.books, ids)
        with self.assertRaises(ValueError):
            bridge.bridge_direct_weights(self.words, self.books, ids, k_order="codebook")
        with self.assertRaises(ValueError):
            bridge.bridge_direct_weights(self.words, self.books, self.ids, k_order="unknown")

    def test_activation_and_epilogue_reject_invalid_contracts(self):
        result = bridge.bridge_direct_weights(self.words, self.books, self.ids)
        a = np.zeros((1, 1024), np.uint8)
        scale, bias = np.ones(1, np.float32), np.zeros(1, np.float32)
        for bad_scale in (np.array([np.nan], np.float32), np.zeros(1, np.float32)):
            with self.assertRaises(ValueError):
                bridge.bridge_prepared_activation(a, bad_scale, bias, result)
        a[0, 0] = 255
        with self.assertRaises(ValueError):
            bridge.bridge_prepared_activation(a, scale, bias, result)
        with self.assertRaises(TypeError):
            bridge.fp32_epilogue(np.ones((1, 1), np.float16), scale, bias)


if __name__ == "__main__":
    unittest.main()
