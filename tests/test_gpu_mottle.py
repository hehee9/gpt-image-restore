"""얼룩 GPU 블록과 CPU 기준 비교."""

import numpy as np
import pytest

from gptimg_restore import mottle
from gptimg_restore._gpu import GPUBackend, GPUUnavailableError
from gptimg_restore._gpu_mottle import _linear_box_axis


@pytest.fixture(scope="module")
def backend():
    try:
        return GPUBackend()
    except GPUUnavailableError as error:
        pytest.skip(str(error))


@pytest.mark.parametrize("radius", [1, 13])
@pytest.mark.parametrize("axis", [0, 1])
def test_linear_box_axis(backend, radius, axis):
    rng = np.random.default_rng(23)
    rgb = rng.uniform(0, 255, (64, 70, 3)).astype(np.float32)
    allowed = rng.random((64, 70)) > 0.1
    expected = mottle._linear_box_axis(rgb, allowed, radius, axis)
    actual = backend.cp.asnumpy(_linear_box_axis(backend.cp.asarray(rgb),
                                 backend.cp.asarray(allowed), radius, axis, backend))
    np.testing.assert_allclose(actual, expected, atol=0.0001, rtol=0)


def test_estimate(backend):
    rng = np.random.default_rng(42)
    rgb = rng.uniform(120, 125, (80, 90, 3)).astype(np.float32)
    rgb[25:50, 40:42] = 40
    expected_band, expected_weight = mottle._estimate(rgb)
    actual_band, actual_weight = mottle._estimate(rgb, backend=backend)
    np.testing.assert_allclose(actual_band, expected_band, atol=0.0002, rtol=0)
    np.testing.assert_allclose(actual_weight, expected_weight, atol=0.0002, rtol=0)
