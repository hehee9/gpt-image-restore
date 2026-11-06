import numpy as np
import pytest
from scipy import fft, ndimage

from gptimg_restore import periodic
from gptimg_restore import _gpu_periodic as gpu_periodic
from gptimg_restore._gpu import GPUBackend, GPUUnavailableError


@pytest.fixture(scope="module")
def gpu():
    try:
        return GPUBackend()
    except GPUUnavailableError as error:
        pytest.skip(str(error))


@pytest.mark.parametrize("sigma", [0.8, 1.2, 2.0])
def test_gpu_gaussian_preserves_scipy_precision(gpu, sigma):
    values = np.random.default_rng(27).uniform(0, 255, (193, 217, 3)).astype(np.float32)
    expected = ndimage.gaussian_filter(values, (sigma, sigma, 0))
    actual = gpu.cp.asnumpy(gpu_periodic._gaussian(gpu.cp.asarray(values), sigma, gpu))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape,height,width", [((1, 1), 193, 217), ((3, 5), 701, 1254), ((7, 3), 1672, 513)])
def test_gpu_gain_expansion_matches_cpu(gpu, shape, height, width):
    random = np.random.default_rng(47)
    gain = random.uniform(0, 2, shape).astype(np.float32)
    expected = periodic._expand_gain(gain, height, width, 256)
    actual = gpu.cp.asnumpy(gpu_periodic._expand_gain(gpu.cp.asarray(gain), height, width, 256, gpu))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("fx,fy,phase", [(0.25, -0.25, 0), (0.5, 0, 0), (0, 0.5, 0.7), (0.2389471, -0.192371, 1.3)])
def test_gpu_carrier_basis_matches_cpu(gpu, fx, fy, phase):
    expected = periodic._carrier_basis(fx, fy, phase, 941, 1255)
    actual = gpu_periodic._carrier_basis(fx, fy, phase, 941, 1255, gpu)
    for a, b in zip(actual, expected):
        if b is None:
            assert a is None
        else:
            np.testing.assert_allclose(gpu.cp.asnumpy(a), b, rtol=0, atol=6e-8)


def _wave_image():
    y, x = np.mgrid[:384, :513]
    base = 100 + 20 * x / 513 + 10 * y / 384
    wave = 1.3 * np.cos(2 * np.pi * (0.25 * x - 0.25 * y) + 0.71)
    image = np.repeat((base + wave)[..., None], 3, axis=2).astype(np.float32)
    image[40:70, 80:100] = 20
    image[290:350, 430:490] = 255
    return image


def test_gpu_preprocessing_matches_cpu(gpu):
    image = _wave_image()
    mask, clipped, residual, power, maxima = gpu_periodic._prepare(image, gpu, 3.0, 18.0, 16, 0.02)
    cpu_mask, cpu_clipped = periodic._background_mask(image, 3.0)
    np.testing.assert_array_equal(mask, cpu_mask)
    np.testing.assert_array_equal(clipped, cpu_clipped)
    opponent = periodic._opponent(image)
    cpu_residual = opponent - ndimage.gaussian_filter(opponent, (1.2, 1.2, 0))
    np.testing.assert_allclose(residual, cpu_residual, rtol=0, atol=5e-5)
    window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1])).astype(np.float32) * cpu_mask
    values = (cpu_residual - np.median(cpu_residual[cpu_mask], axis=0)) * window[..., None]
    cpu_power = abs(fft.fft2(values, axes=(0, 1)))
    cpu_power /= window.sum()
    cpu_power **= 2
    np.testing.assert_allclose(power, cpu_power, rtol=0.01, atol=2e-8)
    expected = periodic._candidates(cpu_power, 12, 18, 16, 0.02)
    actual = gpu_periodic._candidates(power, maxima, 12, 18, 16)
    assert [(v["family"], v["channel"]) for v in actual] == [(v["family"], v["channel"]) for v in expected]


@pytest.mark.parametrize("spatial_gain", [False, True])
def test_gpu_periodic_output_and_decisions_match_cpu(gpu, spatial_gain):
    image = _wave_image()
    expected, cpu_report = periodic.apply(image, spatial_gain=spatial_gain)
    actual, report = periodic.apply(image, backend=gpu, spatial_gain=spatial_gain)
    assert report["status"] == cpu_report["status"]
    assert report["path"] == cpu_report["path"]
    assert report["candidate_count"] == cpu_report["candidate_count"]
    assert len(report["components"]) == len(cpu_report["components"])
    for a, b in zip(report["components"], cpu_report["components"]):
        assert a["eighth_grid_vector"] == b["eighth_grid_vector"]
        assert [v["accepted"] for v in a["channels"].values()] == [v["accepted"] for v in b["channels"].values()]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0.002)
    assert np.abs(np.rint(actual).astype(np.int16) - np.rint(expected).astype(np.int16)).max() <= 1


@pytest.mark.parametrize("shape,value", [((16, 25), 128), ((128, 192), 255), ((96, 192), 0)])
def test_gpu_periodic_early_returns(gpu, shape, value):
    image = np.full((*shape, 3), value, np.float32)
    actual, report = periodic.apply(image, backend=gpu)
    expected, cpu_report = periodic.apply(image)
    np.testing.assert_array_equal(actual, expected)
    assert report == cpu_report
