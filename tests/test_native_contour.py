"""@description 선택형 윤곽 가속 경로와 Python 경로의 비트 동일성 검사."""

import numpy as np
import pytest

from gptimg_restore import contour


native = contour._native
pytestmark = pytest.mark.skipif(native is None, reason="Contour extension module missing or disabled")


@pytest.mark.parametrize("seed", [7, 29])
def test_nearest_refinement_matches_numpy(seed):
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 100, (150, 2)).astype(np.float32)
    normal = rng.normal(size=(150, 2)).astype(np.float32)
    xx = rng.integers(0, 100, 900, dtype=np.intp)
    yy = rng.integers(0, 100, 900, dtype=np.intp)
    index = rng.integers(0, len(points), 900, dtype=np.int32)
    actual = index.copy()
    positions = np.column_stack((xx + 3, yy + 5))
    displacement = positions - points[index]
    shift = np.rint(displacement[:, 0] * normal[index, 1]
                    - displacement[:, 1] * normal[index, 0]).astype(int)
    index = np.clip(index + shift, 0, len(points) - 1)
    nearest_squared = np.sum((positions - points[index]) ** 2, axis=1)
    for offset in (-1, 1):
        candidate = np.clip(index + offset, 0, len(points) - 1)
        squared = np.sum((positions - points[candidate]) ** 2, axis=1)
        closer = squared < nearest_squared
        index[closer] = candidate[closer]
        nearest_squared[closer] = squared[closer]
    native.refine_index(actual, xx, yy, 3, 5, points, normal)
    np.testing.assert_array_equal(actual, index)


@pytest.mark.parametrize("dtype", [np.int32, np.intp])
def test_profile_scatter_matches_numpy(dtype):
    rng = np.random.default_rng(72)
    rgb = rng.uniform(-5, 260, (45, 64, 3)).astype(np.float32)
    protected = (rng.random((45, 64)) < 0.2).astype(np.uint8)
    points = rng.uniform(0, 50, (30, 2)).astype(np.float32)
    normal = rng.uniform(-1, 1, (30, 2)).astype(np.float32)
    weight = rng.random((30, 31))
    model = rng.uniform(0, 255, (30, 31, 3)).astype(np.float32)
    confidence = rng.random((45, 64)).astype(np.float32)
    correction = np.zeros_like(rgb)
    expected_confidence = confidence.copy()
    expected_correction = correction.copy()
    yy, xx = np.nonzero(np.ones((35, 50), np.uint8))
    index = rng.integers(0, len(points), len(xx), dtype=dtype)
    expected_x, expected_y = xx + 3, yy + 4
    displacement = np.column_stack((expected_x, expected_y)) - points[index]
    distance = np.linalg.norm(displacement, axis=1)
    depth = np.sum(displacement * normal[index], axis=1)
    valid = (distance < 14) & (protected[expected_y, expected_x] == 0)
    expected_x, expected_y, index_valid, depth, distance = (
        value[valid] for value in (expected_x, expected_y, index, depth, distance))
    sample_depth = np.clip(depth + 15, 0, 30).astype(np.float32)
    lower = sample_depth.astype(np.int32)
    upper = np.minimum(lower + 1, 30)
    fraction = sample_depth - lower
    pixel_weight = weight[index_valid, lower] * (1 - fraction) + weight[index_valid, upper] * fraction
    pixel_weight *= np.clip((14 - distance) / 4, 0, 1)
    use = pixel_weight > expected_confidence[expected_y, expected_x]
    expected_x, expected_y, pixel_weight = expected_x[use], expected_y[use], pixel_weight[use]
    mix = fraction[use, None]
    target = model[index_valid[use], lower[use]] * (1 - mix) + model[index_valid[use], upper[use]] * mix
    expected_confidence[expected_y, expected_x] = pixel_weight
    expected_correction[expected_y, expected_x] = np.clip(rgb[expected_y, expected_x] - target, -10, 10) * pixel_weight[:, None]
    accepted = native.apply_profile(rgb, protected, confidence, correction,
                                    points, normal, weight, model, index, xx, yy, 3, 4)
    assert accepted == bool(len(expected_x))
    np.testing.assert_array_equal(confidence.view(np.uint32), expected_confidence.view(np.uint32))
    np.testing.assert_array_equal(correction.view(np.uint32), expected_correction.view(np.uint32))


@pytest.mark.parametrize("size", [256, 512])
def test_full_contour_matches_fallback(monkeypatch, size):
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    angle = np.arctan2((yy - size / 2) / .65, xx - size / 2)
    radius = np.sqrt((xx - size / 2) ** 2 + ((yy - size / 2) / .65) ** 2)
    depth = radius - size * .34 - .5 * np.sin(angle * 48)
    gray = 65 + 95 / (1 + np.exp(np.clip(depth / 1.4, -60, 60)))
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    actual, report, confidence = contour._restore(rgb, 1.0)
    monkeypatch.setattr(contour, "_native", None)
    expected, expected_report, expected_confidence = contour._restore(rgb, 1.0)
    assert report["accepted_contours"] > 0
    assert report == expected_report
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(confidence.view(np.uint32), expected_confidence.view(np.uint32))
