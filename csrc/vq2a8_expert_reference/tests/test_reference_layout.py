"""CPU checks of restored Python helpers; NOT execution of the CCE kernel."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import ml_dtypes
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gen_data  # noqa: E402
import w2_layout as layout  # noqa: E402


class ReferenceLayoutTests(unittest.TestCase):
    def test_byte_bit_order(self):
        codes = np.array([[0, 1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(layout.pack_2bit_codes(codes), [[0xE4]])
        np.testing.assert_array_equal(layout.unpack_2bit_codes(np.array([[0xE4]], dtype=np.uint8)), codes)

    def test_packed_zn_roundtrip(self):
        rng = np.random.default_rng(5)
        for shape in [(32, 16), (64, 512), (3, 96, 1024), (2, 3, 32, 256)]:
            with self.subTest(shape=shape):
                codes = rng.integers(0, 4, shape, dtype=np.uint8)
                packed = layout.pack_zn_2bit_codes(codes)
                self.assertEqual(packed.shape, shape[:-2] + (shape[-2] // 32, shape[-1] // 16, 16, 8))
                self.assertEqual(packed.nbytes, codes.nbytes // 4)
                np.testing.assert_array_equal(layout.unpack_zn_2bit_codes(packed), codes)

    def test_zn_offsets_against_independent_scalar_pack(self):
        codes = np.random.default_rng(17).integers(0, 4, (64, 32), dtype=np.uint8)
        packed = layout.pack_zn_2bit_codes(codes)
        for n1 in range(2):
            for k1 in range(2):
                for k0 in range(16):
                    for byte in range(8):
                        expected = sum(int(codes[n1 * 32 + byte * 4 + i, k1 * 16 + k0]) << (2 * i) for i in range(4))
                        self.assertEqual(int(packed[n1, k1, k0, byte]), expected)

    def test_codebook_axis_order_and_roundtrip(self):
        codes = np.random.default_rng(6).integers(0, 4, (2, 64, 512), dtype=np.uint8)
        grouped = layout.reshape_codes_by_codebook(codes)
        self.assertEqual(grouped.shape, (2, 2, 2, 32, 256))
        np.testing.assert_array_equal(grouped[1, 1, 0], codes[1, :32, 256:])
        np.testing.assert_array_equal(layout.restore_codes_from_codebook_groups(grouped), codes)

    def test_four_scalar_levels_pair_order(self):
        levels = np.array([3, 17, 29, 47], dtype=np.uint8)
        book = layout.build_pair_lut(levels)
        self.assertEqual(book.shape, (16, 2))
        for index in range(16):
            np.testing.assert_array_equal(book[index], [levels[index & 3], levels[index >> 2]])

    def test_arbitrary_vq_pairs_are_not_restricted_to_four_scalar_levels(self):
        # This is an independent BYTE-level model of the shown AIV lookup, not CCE execution.
        # 16 arbitrary pairs with >4 distinct values cannot be a four-level Cartesian product.
        book = np.arange(32, dtype=np.uint8).reshape(16, 2)
        pair_codes = np.random.default_rng(19).integers(0, 16, (16, 256), dtype=np.uint8)
        carriers = np.empty((32, 256), dtype=np.uint8)
        carriers[0::2] = pair_codes & 3
        carriers[1::2] = pair_codes >> 2
        packed = layout.pack_zn_2bit_codes(carriers)
        flat = packed.reshape(-1)
        decoded = np.stack((book[flat & 15, 0], book[flat & 15, 1], book[flat >> 4, 0], book[flat >> 4, 1]), axis=-1)
        decoded = decoded.reshape(1, 16, 16, 32).transpose(0, 3, 1, 2).reshape(32, 256)
        expected = book[pair_codes].transpose(0, 2, 1).reshape(32, 256)
        np.testing.assert_array_equal(decoded, expected)
        self.assertGreater(np.unique(book).size, 4)

    def test_invalid_layouts_rejected(self):
        for codes in [np.full((32, 16), 4, np.uint8), np.zeros((31, 16), np.uint8), np.zeros((32, 20), np.uint8)]:
            with self.subTest(shape=codes.shape), self.assertRaises(ValueError):
                layout.pack_zn_2bit_codes(codes)
        with self.assertRaises(ValueError):
            layout.unpack_zn_2bit_codes(np.zeros((1, 1, 16, 7), np.uint8))
        with self.assertRaises(ValueError):
            layout.build_pair_lut(np.zeros(5, np.uint8))

    def test_group_list_validation(self):
        np.testing.assert_array_equal(gen_data.parse_group_list("2,0,1", 3), [2, 0, 1])
        np.testing.assert_array_equal(gen_data.parse_group_list(None, 3), [1, 1, 1])
        for text in ["1,2", "1,,2", "1,a,2"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                gen_data.parse_group_list(text, 3)
        for values in [[0, 0, 0], [1, -1, 1]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                gen_data.validate_shape(3, 1024, 1024, np.array(values, np.int64))
        with self.assertRaises(ValueError):
            gen_data.validate_shape(1, 512, 1024, np.ones(1, np.int64))

    def test_mx_scales_zero_and_known_exponents(self):
        source = np.concatenate([np.zeros(32), np.full(32, 448), np.full(32, 896), np.full(32, 224)]).astype(
            np.float32
        )[None]
        quantized, codes, values = gen_data.quantize_mxfp8_e4m3(source, ml_dtypes.float8_e4m3fn)
        np.testing.assert_array_equal(codes, [[127, 127, 128, 126]])
        np.testing.assert_array_equal(values, [[1, 1, 2, 0.5]])
        restored = quantized.astype(np.float32).reshape(1, 4, 32) * values[..., None]
        np.testing.assert_array_equal(restored.reshape(source.shape), source)

    def test_mx_byte_representation(self):
        source = np.random.default_rng(7).normal(size=(3, 128)).astype(np.float32)
        quantized, codes, values = gen_data.quantize_mxfp8_e4m3(source, ml_dtypes.float8_e4m3fn)
        self.assertEqual(quantized.dtype.itemsize, 1)
        self.assertEqual(codes.dtype, np.uint8)
        self.assertEqual(codes.shape, (3, 4))
        self.assertTrue(np.isfinite(quantized.astype(np.float32)).all())
        self.assertTrue((codes < 255).all())
        np.testing.assert_array_equal(values, np.exp2(codes.astype(np.int32) - 127))

    def test_k_permutation_requires_more_than_a_byte_repack(self):
        # Current direct layout can have mixed codebook IDs inside a physical K256 tile.
        canonical_ids = np.repeat(np.arange(2, dtype=np.uint8), 256)
        permutation = np.arange(512).reshape(2, 256).T.reshape(-1)
        physical_ids = canonical_ids[permutation]
        self.assertEqual(np.unique(physical_ids[:256]).size, 2)
        self.assertEqual(np.unique(canonical_ids[:256]).size, 1)

    def test_generator_bins_and_golden(self):
        # Tiny synthetic fixture only, never operate inside user checkpoint directories.
        previous = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="vq2-expert-cpu-") as temporary:
            try:
                os.chdir(temporary)
                with contextlib.redirect_stdout(io.StringIO()):
                    gen_data.generate(3, 1024, 1024, np.array([1, 0, 2], np.int64), 9, True)
                metadata = json.loads(Path("input/metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["group_list"], [1, 0, 2])
                self.assertEqual(Path("input/input_b.bin").stat().st_size, 3 * 1024 * 1024 // 4)
                self.assertEqual(Path("input/input_table.bin").stat().st_size, 3 * 4 * 32 * 32)
                a = np.fromfile("input/input_a.bin", dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
                scales = np.fromfile("input/input_a_scale.bin", dtype=np.uint8).reshape(3, 32)
                a = (a.reshape(3, 32, 32) * np.exp2(scales.astype(np.int32) - 127)[..., None]).reshape(3, 1024)
                b = np.fromfile("output/golden_b_fp8.bin", dtype=np.uint8).view(ml_dtypes.float8_e4m3fn)
                b = b.astype(np.float32).reshape(3, 1024, 1024)
                expected = np.concatenate([a[:1] @ b[0].T, a[1:] @ b[2].T])
                golden = np.fromfile("output/golden_c.bin", dtype=np.float32).reshape(3, 1024)
                np.testing.assert_allclose(golden, expected, rtol=1e-5, atol=2e-4)
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
