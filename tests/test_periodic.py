import json

import numpy as np
import pytest
from scipy import ndimage

from gptimg_restore import periodic


def _smooth(width, height):
    y, x = np.mgrid[:height, :width]
    base = 90 + 30 * x / width + 15 * y / height
    low_frequency = 2 * np.sin(2 * np.pi * x / width)
    return np.stack([base + 10 + low_frequency, base, base - 12], axis=-1).astype(np.float32)


def _wave(width, height, nx=2, ny=-2, amplitude=1.0, phase=0.71):
    y, x = np.mgrid[:height, :width]
    sx, sy = 16 * (width // 16) / width, 16 * (height // 16) / height
    resize_phase = np.pi * ((nx / 8) * (sx - 1) + (ny / 8) * (sy - 1))
    value = amplitude * np.cos(2 * np.pi * (nx * sx / 8 * x + ny * sy / 8 * y) + phase + resize_phase)
    return value.astype(np.float32)


@pytest.mark.parametrize("width,height", [(1536, 1024), (1254, 1254), (1672, 941)])
def test_native_wave_and_parity_removal(width, height):
    base = _smooth(width, height)
    diagonal = _wave(width, height, amplitude=1.6)
    parity = _wave(width, height, nx=4, ny=0, amplitude=0.65, phase=0)
    noise = diagonal[..., None] * np.array([1.0, 0.7, 0.45], np.float32)
    noise += parity[..., None]
    image = base + noise
    restored, report = periodic.apply(image)
    assert restored.shape == image.shape
    assert restored.dtype == np.float32
    assert report["native_size_model_matched"]
    assert len(report["components"]) >= 2
    assert np.sqrt(np.mean((restored - base) ** 2)) < np.sqrt(np.mean(noise ** 2)) * 0.08
    assert np.max(np.abs(np.mean(restored - base, axis=(0, 1)))) < 0.003
    json.dumps(report, allow_nan=False)


def test_resized_wave_uses_discovery():
    base = _smooth(1536, 1024)
    wave = _wave(1536, 1024, amplitude=2.0)[..., None]
    image = base + wave
    scale = (0.83, 0.83, 1)
    resized_base = ndimage.zoom(base, scale, order=3).astype(np.float32)
    resized = ndimage.zoom(image, scale, order=3).astype(np.float32)
    restored, report = periodic.apply(resized)
    before = np.sqrt(np.mean((resized - resized_base) ** 2))
    after = np.sqrt(np.mean((restored - resized_base) ** 2))
    assert report["path"] in ("discovered", "mixed")
    assert after < before * 0.12


@pytest.mark.parametrize("width,height", [(1025, 512), (1024, 513)])
def test_native_bin_does_not_pin_nearby_off_grid_frequency(width, height):
    y, x = np.mgrid[:height, :width]
    wave = np.cos(2 * np.pi * (0.25 * x - 0.25 * y) + 0.63)
    image = np.repeat((128 + wave)[..., None], 3, axis=2).astype(np.float32)
    restored, report = periodic.apply(image, spatial_gain=False)
    assert report["native_size_model_matched"]
    assert np.sqrt(np.mean((restored - 128) ** 2)) < 0.02
    assert any(periodic._frequency_distance(component["frequency"], (0.25, -0.25), width, height) < 0.01
               for component in report["components"])


def test_strength_zero_and_linear_scaling():
    base = _smooth(256, 192)
    image = base + _wave(256, 192, amplitude=1.2)[..., None]
    unchanged, report = periodic.apply(image, strength=0)
    assert np.array_equal(unchanged, image)
    assert unchanged is not image
    assert report["status"] == "disabled"
    full, _ = periodic.apply(image)
    half, _ = periodic.apply(image, strength=0.5)
    assert np.allclose(half, image - (image - full) * 0.5, atol=1e-5)


def test_clean_gradient_has_no_false_components():
    image = _smooth(384, 256)
    report = periodic.analyze(image)
    restored, _ = periodic.apply(image)
    assert report["components"] == []
    assert report["status"] == "no_significant_component"
    assert np.array_equal(restored, image)


def test_repeated_high_contrast_strokes_are_preserved():
    image = np.full((384, 512, 3), 235, np.float32)
    for top in range(24, 360, 24):
        for left in range(24, 480, 16):
            image[top:top+12, left:left+3] = 20
            image[top:top+3, left:left+9] = 20
    restored, report = periodic.apply(image)
    assert report["components"] == []
    assert np.array_equal(restored, image)


def test_fully_clipped_fields_are_unchanged():
    image = np.full((128, 192, 3), 255, np.float32)
    restored, report = periodic.apply(image)
    assert np.array_equal(restored, image)
    assert report["status"] == "insufficient_background"


def test_optional_slow_gain_tracks_known_modulation():
    width, height = 1024, 768
    base = _smooth(width, height)
    gain = 0.2 + 1.4 * (np.arange(width, dtype=np.float32) + 0.5) / width
    wave = _wave(width, height, amplitude=1.5) * gain[None, :]
    image = base + wave[..., None]
    fixed, _ = periodic.apply(image, spatial_gain=False)
    adaptive, report = periodic.apply(image, spatial_gain=True)
    error_fixed = np.sqrt(np.mean((fixed - base) ** 2))
    error_adaptive = np.sqrt(np.mean((adaptive - base) ** 2))
    assert error_adaptive < error_fixed * 0.6
    assert report["settings"]["spatial_gain"]


def test_near_axis_conjugates_are_one_resolvable_component():
    assert periodic._frequency_distance((0, 0.4976), (0.00005, -0.49762), 1254, 1254) < 0.1
    assert periodic._frequency_distance((0.4976, 0.4976), (0.4976, -0.4976), 1254, 1254) > 5


def test_weak_unmodeled_chroma_carrier_is_left_alone():
    image = _smooth(384, 256)
    y, x = np.mgrid[:256, :384]
    image[..., 0] += (0.025 * np.cos(2 * np.pi * (0.4374 * x + 0.0001 * y))).astype(np.float32)
    restored, report = periodic.apply(image)
    assert report["components"] == []
    assert np.array_equal(restored, image)


def test_weak_native_chroma_carrier_can_still_be_removed():
    image = _smooth(384, 256)
    image[..., 0] += _wave(384, 256, amplitude=0.03)
    restored, report = periodic.apply(image)
    base = _smooth(384, 256)
    before = (image[..., 0] - image[..., 1]) - (base[..., 0] - base[..., 1])
    after = (restored[..., 0] - restored[..., 1]) - (base[..., 0] - base[..., 1])
    assert report["native_size_model_matched"]
    assert after.std() < before.std() * 0.1


def test_slow_gain_does_not_turn_masked_dc_into_amplitude():
    y, x = np.mgrid[:256, :256]
    carrier = np.cos(2 * np.pi * (x * 0.23 - y * 0.24))
    selected = carrier > 0.1
    values = carrier[selected]
    target = np.zeros((len(values), 3))
    target[:, 0] = values + 0.4
    model = {"shape": (256, 256), "matrix": values[:, None],
             "coefficients": np.array([[1.0, 0, 0]]), "target": target,
             "weights": np.ones_like(target), "noise": np.full(3, 0.1),
             "x": x[selected], "y": y[selected], "clipped": np.zeros((256, 256), bool)}
    gain = periodic._slow_gain(model, {"columns": (0, 1)}, 0)
    assert np.max(np.abs(gain - 1)) < 1e-5


def test_subresolution_nyquist_offsets_do_not_create_a_sine_basis():
    assert periodic._snap_frequency((0.49999997, 0.000000016), 1536, 1024) == (0.5, 0.0)
    frequency = (0.497607656, 0.0)
    assert periodic._snap_frequency(frequency, 1254, 1254) == frequency


def test_exact_parity_at_odd_delivery_size_has_bounded_amplitude():
    base = _smooth(1255, 941)
    wave = 0.8 * (-1.0) ** np.arange(1255)
    image = base + wave[None, :, None].astype(np.float32)
    restored, report = periodic.apply(image, spatial_gain=False)
    assert report["components"]
    assert np.sqrt(np.mean((restored - base) ** 2)) < 0.025
    assert max(channel["amplitude"] for c in report["components"] for channel in c["channels"].values()) < 1.0
