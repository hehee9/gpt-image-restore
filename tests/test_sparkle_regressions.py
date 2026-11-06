import cv2
import numpy as np
import pytest

from gptimg_restore import sparkle


def test_local_inverse_text_protection_matches_full_inverse():
    rng = np.random.default_rng(164)
    rgb = rng.uniform(-0.8, 0.8, (192, 320, 3)).astype(np.float32)
    rgb += np.array([221, 203, 177], np.float32)
    alpha = np.zeros(rgb.shape[:2], np.uint8)
    for y in (45, 95, 145):
        cv2.putText(alpha, 'dark rows 123', (12, y), cv2.FONT_HERSHEY_COMPLEX,
                    0.7, 255, 1, cv2.LINE_AA)
    rgb -= alpha[..., None] * np.array([0.68, 0.62, 0.55], np.float32)
    surface = sparkle._median_surface(rgb)
    luma = rgb @ sparkle._LUMA
    residual = np.maximum(surface @ sparkle._LUMA - luma,
                          cv2.morphologyEx(luma, cv2.MORPH_CLOSE,
                                           np.ones((31, 31), np.uint8)) - luma)
    ys, xs = np.mgrid[12:180:7, 12:308:7].reshape(2, -1)
    expected = sparkle._text_points(255 - rgb, 255 - surface, 255 - luma,
                                    residual, ys, xs)
    actual = sparkle._text_points(rgb, surface, luma, residual, ys, xs, inverse=True)
    assert np.any(expected)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('bit_depth', [8, 16])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_saved_view_in_place_temporaries_preserve_rounding(bit_depth, dtype):
    values = np.random.default_rng(831).uniform(-1, 256, (17, 23, 3)).astype(dtype)
    original = values.copy()
    scale = 1 if bit_depth == 8 else 257
    expected = np.rint(np.clip(values, 0, 255) * scale) / scale
    np.testing.assert_array_equal(sparkle._saved_view(values, bit_depth), expected)
    np.testing.assert_array_equal(values, original)
