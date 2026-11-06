"""@description 선택적 잉크 렌더러의 float 연산 및 전체 윤곽선 동등성 검사."""

import numpy as np
import pytest
from scipy.special import ndtr

from gptimg_restore import celfill

native = celfill._native_ink
pytestmark = pytest.mark.skipif(native is None, reason="Optional extension must be built")


def test_native_small_array_operations_are_bit_exact():
    rng = np.random.default_rng(41)
    points = rng.normal(100, 80, (83, 2)).astype(np.float32)
    normal = rng.normal(0, 1, (83, 2)).astype(np.float32)
    depths = np.arange(-12, 12.01, 0.5, dtype=np.float32)
    expected = points[:, None, :] + normal[:, None, :] * depths[None, :, None]
    np.testing.assert_array_equal(native._coordinates(points, normal, depths), expected)
    profile = rng.uniform(0, 255, (83, 49, 3)).astype(np.float32)
    direction = rng.normal(0, 1, (83, 3)).astype(np.float32)
    expected = np.sum(profile * direction[:, None, :], axis=2)
    np.testing.assert_array_equal(native._projection(profile, direction), expected)
    low = rng.uniform(0, 255, (83, 3)).astype(np.float32)
    vector = rng.normal(0, 100, (83, 3)).astype(np.float32)
    contrast = np.sum(vector * vector, axis=1)
    contrast[:4] = [0, 0.5, 1, 1.5]
    expected = np.sum((profile - low[:, None, :]) * vector[:, None, :], axis=2)
    expected /= np.maximum(contrast[:, None], 1)
    np.testing.assert_array_equal(native._alpha(profile, low, vector, contrast), expected)


def test_native_accumulation_preserves_each_contour_rounding():
    rng = np.random.default_rng(72)
    total = rng.uniform(0, 10000, (31, 29, 3)).astype(np.float32)
    mass = rng.uniform(0, 100, (31, 29)).astype(np.float32)
    expected_total, expected_mass = total.copy(), mass.copy()
    for _ in range(3):
        flat = rng.choice(31 * 29, 250, replace=False)
        yy, xx = np.unravel_index(flat, mass.shape)
        value = rng.uniform(0, 255, (250, 3))
        confidence = rng.uniform(0.05, 12, 250)
        expected_total[yy, xx] += value * confidence[:, None]
        expected_mass[yy, xx] += confidence
        native._accumulate(total, mass, yy, xx, value, confidence)
    np.testing.assert_array_equal(total, expected_total)
    np.testing.assert_array_equal(mass, expected_mass)


@pytest.mark.parametrize("kind", ["curve", "ring", "masked", "empty"])
def test_native_complete_contours_match_python(monkeypatch, kind):
    y, x = np.mgrid[:192, :256].astype(np.float32)
    if kind == "ring":
        depth = np.abs(np.hypot(x - 128, y - 96) - 60) - 4
    else:
        depth = x - (120 + 15 * np.sin(y / 40))
    coverage = ndtr(depth / 1.2)
    source = np.array([12, 20, 25], np.float32) + coverage[..., None] * np.array(
        [218, 180, 155], np.float32
    )
    zero = np.zeros(source.shape[:2], bool)
    detail = zero.copy()
    if kind == "masked":
        detail[45:100] = True
    support = np.full(source.shape[:2], kind != "empty", bool)
    actual, actual_mask = native.render_ink_profiles(
        source, source.copy(), zero, detail, zero, support
    )
    monkeypatch.setattr(celfill, "_native_ink", None)
    expected, expected_mask = celfill._render_ink_profiles(
        source, source.copy(), zero, detail, zero, support
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_mask, expected_mask)
    if kind != "empty":
        assert actual_mask.any()
