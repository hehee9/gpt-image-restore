import cv2
import numpy as np
import pytest

from gptimg_restore import lattice
from gptimg_restore._gpu import GPUBackend, GPUUnavailableError
from gptimg_restore._gpu_lattice import _directional_lines, _initial_fields, _periodic_weight


@pytest.fixture(scope="module")
def backend():
    try:
        return GPUBackend()
    except GPUUnavailableError as error:
        pytest.skip(str(error))


def test_dense_fields_match_cpu(backend):
    rng = np.random.default_rng(3)
    rgb = rng.uniform(0, 255, (131, 143, 3)).astype(np.float32)
    smooth, residual, fine, edges = _initial_fields(backend, rgb)
    expected = cv2.GaussianBlur(rgb, (21, 21), 2.5)
    np.testing.assert_allclose(smooth, expected, atol=0.0001)
    np.testing.assert_allclose(residual, rgb - expected, atol=0.0001)
    np.testing.assert_allclose(fine, np.mean((rgb - expected) ** 2, axis=2), atol=0.02)
    gx = cv2.Sobel(rgb, cv2.CV_32F, 1, 0, scale=0.125)
    gy = cv2.Sobel(rgb, cv2.CV_32F, 0, 1, scale=0.125)
    np.testing.assert_array_equal(edges, np.any(gx * gx + gy * gy > 16, axis=2))


def test_directional_lines_match_opencv(backend):
    rng = np.random.default_rng(4)
    candidate = (rng.random((103, 117)) > 0.5).astype(np.uint8)
    cv2.line(candidate, (1, 1), (90, 90), 1, 2)
    expected = np.zeros_like(candidate)
    for angle in range(0, 180, 15):
        direction = np.deg2rad(angle)
        dx, dy = int(round(5 * np.cos(direction))), int(round(5 * np.sin(direction)))
        kernel = np.zeros((11, 11), np.uint8)
        cv2.line(kernel, (5 - dx, 5 - dy), (5 + dx, 5 + dy), 1)
        expected |= cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    np.testing.assert_array_equal(_directional_lines(backend, candidate), expected)


def test_periodic_weight_matches_cpu(backend):
    rng = np.random.default_rng(5)
    residual = rng.normal(size=(201, 233, 3)).astype(np.float32)
    x = np.arange(233, dtype=np.float32)
    residual += 4 * np.sin(x[None, :, None] * 0.9)
    np.testing.assert_allclose(_periodic_weight(backend, residual),
                               lattice._periodic_texture_weight(residual), atol=0.00001)
