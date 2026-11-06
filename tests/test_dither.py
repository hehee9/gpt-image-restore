import cv2
import numpy as np

from gptimg_restore import dither


def test_shipped_blue_noise_thresholds_match_generator_exactly():
    shipped = dither._blue_noise_thresholds()
    generated = dither._generate_blue_noise_thresholds()

    assert shipped.dtype == generated.dtype
    np.testing.assert_array_equal(shipped, generated)


def test_strength_zero_and_16_bit_are_noops():
    rgb = np.linspace(0, 255, 48, dtype=np.float32).reshape(4, 4, 3)
    original = rgb + 1
    cel_fill_mask = np.ones(rgb.shape[:2], dtype=bool)

    zero, zero_report = dither.apply(rgb, original, strength=0, bit_depth=8,
                                     cel_fill_mask=cel_fill_mask)
    sixteen, sixteen_report = dither.apply(rgb, original, strength=1, bit_depth=16,
                                           cel_fill_mask=cel_fill_mask)

    assert zero is rgb
    assert sixteen is rgb
    np.testing.assert_array_equal(zero, rgb)
    np.testing.assert_array_equal(sixteen, rgb)
    assert zero_report == {
        "applied": False,
        "mean_weight": 0.0,
        "mean_flatness": 0.0,
        "weighted_fraction": 0.0,
        "seed": 0,
    }
    assert sixteen_report["applied"] is False


def test_unchanged_pixels_remain_identical_after_rounding():
    rgb = np.full((8, 8, 3), 128, dtype=np.float32)
    original = rgb.copy()
    rgb[2:6, 2:6] += 0.25
    output, report = dither.apply(rgb, original, strength=1, bit_depth=8, seed=12)

    np.testing.assert_array_equal(np.rint(output[0:2]), np.rint(rgb[0:2]))
    np.testing.assert_array_equal(np.rint(output[6:]), np.rint(rgb[6:]))
    assert report["applied"] is True
    assert report["weighted_fraction"] == 0.25


def test_dithering_is_deterministic_for_a_seed():
    rgb = np.full((12, 16, 3), 128, dtype=np.float32)
    original = rgb - 1

    first, _ = dither.apply(rgb, original, strength=1, bit_depth=8, seed=42)
    second, _ = dither.apply(rgb, original, strength=1, bit_depth=8, seed=42)

    np.testing.assert_array_equal(first, second)


def test_corrected_text_like_edge_and_neighbors_get_no_dither():
    rgb = np.full((32, 32, 3), 128, dtype=np.float32)
    rgb[:, 16] = 32
    original = rgb - 1

    output, report = dither.apply(rgb, original, strength=1, bit_depth=8, seed=9)

    np.testing.assert_array_equal(output[:, 15:18], rgb[:, 15:18])
    assert np.any(output[:, :12] != rgb[:, :12])
    assert report["mean_flatness"] < 1.0


def test_dithering_reduces_smooth_ramp_banding_after_rounding():
    ramp = (128 + np.linspace(0, 3, 512, dtype=np.float32))[None, :, None]
    rgb = np.repeat(ramp, 3, axis=2)
    original = rgb - 1
    dithered, _ = dither.apply(rgb, original, strength=1, bit_depth=8, seed=7)

    plain_8bit = np.rint(rgb[0, :, 0])
    dithered_8bit = np.rint(dithered[0, :, 0])
    target = rgb[0, :, 0]
    window = np.full(16, 1 / 16, dtype=np.float64)
    plain_error = np.max(np.abs(np.convolve(plain_8bit, window, mode="valid")
                                - np.convolve(target, window, mode="valid")))
    dithered_error = np.max(np.abs(np.convolve(dithered_8bit, window, mode="valid")
                                    - np.convolve(target, window, mode="valid")))

    assert dithered_error < plain_error
    assert np.max(np.abs(dithered_8bit - target)) <= 1.5


def test_document_mask_dithers_narrow_gaps_without_touching_glyph_margin():
    rgb = np.full((32, 48, 3), 128.3, np.float32)
    glyphs = np.zeros(rgb.shape[:2], np.uint8)
    glyphs[:, 2::7] = 1
    rgb[glyphs > 0] = 12
    protected = cv2.dilate(glyphs, np.ones((3, 3), np.uint8)) > 0
    document_mask = ~protected
    original = rgb - 0.5
    legacy, _ = dither.apply(rgb, original, strength=1, bit_depth=8, seed=5)
    output, _ = dither.apply(rgb, original, strength=1, bit_depth=8, seed=5,
                             document_mask=document_mask)
    interior = document_mask.copy()
    interior[:, :5] = False
    interior[:, -5:] = False
    assert np.array_equal(legacy[interior], rgb[interior])
    assert np.any(output[interior] != rgb[interior])
    assert np.array_equal(output[protected], rgb[protected])
    assert np.max(np.abs(output - rgb)) <= 1


def test_empty_masks_are_byte_identical_to_existing_dither():
    rng = np.random.default_rng(4)
    rgb = rng.uniform(120, 120.2, (32, 48, 3)).astype(np.float32)
    original = rgb - rng.uniform(0.1, 0.7, rgb.shape).astype(np.float32)
    existing, existing_report = dither.apply(rgb, original, strength=1, bit_depth=8, seed=9)
    none_mask, none_report = dither.apply(rgb, original, strength=1, bit_depth=8, seed=9,
                                          cel_fill_mask=None)
    empty_mask, empty_report = dither.apply(
        rgb, original, strength=1, bit_depth=8, seed=9,
        cel_fill_mask=np.zeros(rgb.shape[:2], dtype=bool))
    empty_document, empty_document_report = dither.apply(
        rgb, original, strength=1, bit_depth=8, seed=9,
        document_mask=np.zeros(rgb.shape[:2], dtype=bool))

    np.testing.assert_array_equal(none_mask, existing)
    np.testing.assert_array_equal(empty_mask, existing)
    np.testing.assert_array_equal(empty_document, existing)
    assert none_report == existing_report
    assert empty_report == existing_report
    assert empty_document_report == existing_report


def test_cel_fill_mask_uses_unbiased_blue_noise_quantization():
    height = width = 128
    rgb = np.empty((height, width, 3), dtype=np.float32)
    rgb[:] = (120.3, 140.5, 160.7)
    original = rgb - 1
    cel_fill_mask = np.ones((height, width), dtype=bool)

    blue, _ = dither.apply(rgb, original, strength=1, bit_depth=8, seed=17,
                           cel_fill_mask=cel_fill_mask)
    repeated, _ = dither.apply(rgb, original, strength=1, bit_depth=8, seed=17,
                               cel_fill_mask=cel_fill_mask)
    tpdf, _ = dither.apply(rgb, original, strength=1, bit_depth=8, seed=17)
    error = blue - rgb

    tiled = dither._blue_noise_thresholds()[
        np.arange(height)[:, None] % 64,
        np.arange(width)[None, :] % 64,
    ]
    expected = np.floor(rgb + 1.0 - tiled[..., None])
    np.testing.assert_array_equal(blue, expected)
    np.testing.assert_array_equal(blue, repeated)
    assert np.max(np.abs(error)) < 1
    assert np.max(np.abs(error.mean(axis=(0, 1)))) < 0.001

    blue_low_frequency_rms = np.sqrt(np.mean(cv2.GaussianBlur(error, (0, 0), 3) ** 2))
    tpdf_error = tpdf - rgb
    tpdf_low_frequency_rms = np.sqrt(
        np.mean(cv2.GaussianBlur(tpdf_error, (0, 0), 3) ** 2))
    assert blue_low_frequency_rms < tpdf_low_frequency_rms * 0.5


def test_unmasked_pixels_match_the_existing_tpdf_path():
    rgb = np.full((64, 96, 3), (120.3, 140.5, 160.7), dtype=np.float32)
    original = rgb - 1
    cel_fill_mask = np.zeros(rgb.shape[:2], dtype=bool)
    cel_fill_mask[::2, ::3] = True

    existing, _ = dither.apply(rgb, original, strength=1, bit_depth=8, seed=21)
    mixed, _ = dither.apply(rgb, original, strength=1, bit_depth=8, seed=21,
                            cel_fill_mask=cel_fill_mask)

    np.testing.assert_array_equal(mixed[~cel_fill_mask], existing[~cel_fill_mask])
