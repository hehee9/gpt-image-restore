import numpy as np
import pytest
from scipy import ndimage

from gptimg_restore import periodic


@pytest.mark.parametrize("shape,height,width", [((1, 1), 193, 217), ((3, 5), 701, 1254), ((7, 3), 1672, 513)])
def test_gain_expansion_matches_grid_mode_zoom(shape, height, width):
    gain = np.random.default_rng(47).uniform(0, 2, shape).astype(np.float32)
    expected = ndimage.zoom(gain, 256, order=1, mode="nearest", grid_mode=True)[:height, :width]
    actual = periodic._expand_gain(gain, height, width, 256)
    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("seed", [11, 73])
def test_gain_expansion_preserves_corner_order_across_exponents(seed):
    random = np.random.default_rng(seed)
    exponents = random.choice([0, -24, -60, -120], size=(3, 5))
    gain = (random.uniform(0.5, 1, (3, 5)) * np.exp2(exponents)).astype(np.float32)
    gain[0, 0], gain[-1, -1] = 0, 2
    expected = ndimage.zoom(gain, 256, order=1, mode="nearest", grid_mode=True)[:701, :1254]
    actual = periodic._expand_gain(gain, 701, 1254, 256)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("fx,fy,phase", [(0.25, -0.25, 0), (0.5, 0, 0), (0, 0.5, 0.7), (0.2389471, -0.192371, 1.3)])
def test_separable_carrier_matches_direct_basis(fx, fy, phase):
    height, width = 941, 1255
    yy, xx = np.ogrid[:height, :width]
    direct_phase = 2 * np.pi * (fx * xx + fy * yy) + phase
    cosine, sine = periodic._carrier_basis(fx, fy, phase, height, width)
    assert cosine.dtype == np.float32
    np.testing.assert_allclose(cosine, np.cos(direct_phase).astype(np.float32), rtol=0, atol=6e-8)
    if periodic._self_conjugate(fx, fy):
        assert sine is None
    else:
        assert sine.dtype == np.float32
        np.testing.assert_allclose(sine, np.sin(direct_phase).astype(np.float32), rtol=0, atol=6e-8)
