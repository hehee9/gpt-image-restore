import cv2
import numpy as np
import pytest

from gptimg_restore import textgap
from gptimg_restore._gpu import GPUBackend, GPUUnavailableError
from gptimg_restore._gpu_textgap import _segmented_background_gpu


def test_segmented_background_matches_cpu():
    try:
        backend = GPUBackend()
    except GPUUnavailableError as error:
        pytest.skip(str(error))
    rng = np.random.default_rng(19)
    rgb = rng.uniform(100, 200, (128, 160, 3)).astype(np.float32)
    samples = (rng.random((128, 160)) > 0.2).astype(np.float32)
    mask = np.ones((128, 160), np.uint8)
    mask[:, 80] = 0
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    region = np.ones_like(mask)
    background, mass, determinant = _segmented_background_gpu(
        backend, rgb, samples, labels, stats, region)
    for label in (1, 2):
        left, top, width, height, _ = stats[label]
        box = np.s_[top:top + height, left:left + width]
        component = labels[box] == label
        expected = textgap._background_model(rgb[box], samples[box] * component)
        for actual, reference, tolerance, relative in zip(
                (background[box], mass[box], determinant[box]), expected,
                (0.005, 0.00001, 0.1), (0, 0, 1e-5)):
            np.testing.assert_allclose(actual[component], reference[component], atol=tolerance, rtol=relative)
        for actual, reference, threshold in ((mass[box], expected[1], 0.02),
                                             (determinant[box], expected[2], 1)):
            assert np.count_nonzero((actual[component] < threshold) !=
                                    (reference[component] < threshold)) == 0
