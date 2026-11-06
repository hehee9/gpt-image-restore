"""@description 선택적 채색 적합 가속과 Python 경로의 비트 일치 검증."""

import numpy as np
import pytest

from gptimg_restore import celfill


native = pytest.importorskip("gptimg_restore._native_celfill_fit")


@pytest.mark.parametrize("seed", [1, 7, 32])
def test_faint_profile_matches_python(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    plane = rng.uniform(50, 240, (73, 87)).astype(np.float32)
    points = rng.uniform([0, 0], [86, 72], (64, 2)).astype(np.float32)
    tangent = np.array([np.cos(seed), np.sin(seed)], np.float32)
    monkeypatch.setattr(celfill, "_native_fit", None)
    expected = celfill._faint_profile(plane, points, tangent)
    actual = native._faint_profile(plane, points, tangent)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]


def test_direction_candidates_match_numpy():
    rng = np.random.default_rng(15)
    candidate = rng.random((39, 71)) > 0.5
    nx, ny, contrast, noise = rng.normal(size=(4, 39, 71)).astype(np.float32)
    cutoff = np.cos(np.deg2rad(25))
    for angle in np.arange(0, np.pi, np.pi / 6):
        tangent = np.array([np.cos(angle), np.sin(angle)], np.float32)
        alignment = abs(-nx * tangent[1] + ny * tangent[0])
        expected = candidate & (alignment > cutoff) & (
            abs(contrast) >= np.maximum(1.5, 2.5 * noise)
        )
        np.testing.assert_array_equal(
            native._ridge_pixels(candidate, nx, ny, contrast, noise, tangent, cutoff),
            expected,
        )


@pytest.mark.parametrize("count", [6, 11, 31])
@pytest.mark.parametrize("constant", [False, True])
def test_consistency_percentile_boundary_matches_python(count, constant, monkeypatch):
    monkeypatch.setattr(celfill, "_native_fit", None)
    for below in range(count + 1):
        data = np.full((1, count, 3), np.nextafter(np.float32(6), np.float32(7)), np.float32)
        data[:, :below] = np.nextafter(np.float32(6), np.float32(5))
        model = np.zeros_like(data)
        if constant:
            model = np.broadcast_to(np.zeros(3, np.float32), data.shape)
        interior = np.ones(data.shape[:2], bool)
        assert native._region_is_consistent(data, model, interior, count) == (
            celfill._region_is_consistent(data, model, interior, count)
        )


def test_complete_faint_and_fit_match_python(monkeypatch):
    y, x = np.mgrid[:144, :192].astype(np.float32)
    source = np.full((144, 192, 3), 205, np.float32)
    source += (3 * np.sin((x + y) * 2 * np.pi / 8))[..., None]
    source -= (5 * np.exp(-((y - 64 - 0.1 * x) / 1.2) ** 2))[..., None]
    source += (12 * (x / 192) ** 2)[..., None]
    core = np.zeros(source.shape[:2], bool)
    core[:, 95:98] = True
    source[core] = 20
    monkeypatch.setattr(celfill, "_native_fit", None)
    expected_faint = celfill._faint_strokes(source, core)
    actual_faint = native._faint_strokes(source, core)
    np.testing.assert_array_equal(actual_faint[0], expected_faint[0])
    assert actual_faint[1] == expected_faint[1]
    expected = celfill._fit_regions(source, core, np.zeros_like(core), expected_faint[0])
    actual = native._fit_regions(source, core, np.zeros_like(core), actual_faint[0])
    for name in ("values", "selected", "labels", "coefficients", "limits", "centers", "scales", "gradient"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert actual.regions == expected.regions
