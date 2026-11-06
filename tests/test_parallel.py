from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from gptimg_restore import _parallel


@pytest.fixture(params=(1, 2, 3, 4, 8))
def worker_count(request, monkeypatch):
    workers = request.param
    monkeypatch.setattr(_parallel, "_WORKERS", workers)
    if workers == 1:
        monkeypatch.setattr(_parallel, "_POOL", None)
        yield workers
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            monkeypatch.setattr(_parallel, "_POOL", pool)
            yield workers


@pytest.mark.parametrize("ksize", (3, 9, 31))
@pytest.mark.parametrize("shape", ((17, 23, 3), (127, 2065), (128, 2049), (129, 683, 3)))
def test_median_blur_matches_opencv_exactly(shape, ksize, worker_count):
    seed = sum(shape) + ksize
    source = np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)

    expected = cv2.medianBlur(source, ksize)
    actual = _parallel._median_blur(source, ksize)

    np.testing.assert_array_equal(actual, expected)
