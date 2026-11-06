import numpy as np
import pytest

from gptimg_restore import dots
from gptimg_restore._native import load_native


native = load_native("_native_dots")
pytestmark = pytest.mark.skipif(native is None, reason="Optional dots extension not built or disabled")


@pytest.mark.parametrize("mode", ["conservative", "strict"])
@pytest.mark.parametrize("collect_targets", [False, True])
def test_native_detector_matches_python(monkeypatch, mode, collect_targets):
    rng = np.random.default_rng(438)
    image = np.full((160, 192, 3), 80, dtype=np.float32)
    image += rng.normal(0, 0.08, image.shape).astype(np.float32)
    image[35:37, 30:32] = 220
    image[50, 65] = 160
    image[51, 66] = 160
    image[100:104, 120:124] = 200
    image[120, 155] = [180, 80, 80]
    image = image[:, ::-1]
    monkeypatch.setattr(dots, "_native", native)
    actual = dots._detect(image, mode, dots._MODES[mode], collect_targets=collect_targets)
    monkeypatch.setattr(dots, "_native", None)
    expected = dots._detect(image, mode, dots._MODES[mode], collect_targets=collect_targets)
    np.testing.assert_array_equal(actual[0], expected[0])
    if expected[1] is None:
        assert actual[1] is None
    else:
        np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]


def test_native_component_maximum_uses_first_row_major_tie():
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[10, 10:12] = 1
    labels[11, 10] = 1
    residual = np.zeros((20, 20), dtype=np.float32)
    residual[10, 10:12] = 32
    residual[11, 10] = 32
    stats = np.array([[0, 0, 20, 20, 397], [10, 10, 2, 2, 3]], dtype=np.int32)
    assert native._component_candidates(labels, stats, residual) == [(10, 10, 32.0)]


def test_native_half_height_core_is_eight_connected_and_row_major():
    luma = np.full((48, 48), 80, dtype=np.float32)
    luma[24, 24] = 180
    luma[25, 25] = 140
    luma[23, 23] = 140
    backgrounds = np.array([80], dtype=np.float32)
    mads = np.zeros(1, dtype=np.float32)
    distance = np.full(luma.shape, 20, dtype=np.float32)
    result = native._validate_luma(
        [(24, 24, 100)], luma, backgrounds, mads, distance,
        dots._MODES["conservative"], lambda *args: 0.0,
    )
    assert len(result) == 1
    np.testing.assert_array_equal(result[0][2], [23, 24, 25])
    np.testing.assert_array_equal(result[0][3], [23, 24, 25])
    assert result[0][4] == 100.0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_mad_buffer_preserves_precision_at_rejection_threshold(dtype):
    luma = np.full((48, 48), 80, dtype=np.float32)
    luma[24, 24] = 200
    backgrounds = np.array([80], dtype=np.float32)
    mads = np.array([np.nextafter(dtype(2), dtype(np.inf))], dtype=dtype)
    distance = np.full(luma.shape, 20, dtype=np.float32)
    result = native._validate_luma(
        [(24, 24, 120)], luma, backgrounds, mads, distance,
        dots._MODES["conservative"], lambda *args: 0.0,
    )
    assert result == []


def test_native_contrast_preserves_numpy_scalar_promotion():
    background = np.float32(80.12345)
    luma = np.full((48, 48), background, dtype=np.float32)
    luma[24, 24] = 220
    backgrounds = np.array([background], dtype=np.float32)
    mads = np.zeros(1, dtype=dots._MAD_DTYPE)
    distance = np.full(luma.shape, 20, dtype=np.float32)
    result = native._validate_luma(
        [(24, 24, 140)], luma, backgrounds, mads, distance,
        dots._MODES["conservative"], lambda *args: 0.0,
    )
    assert len(result) == 1
    assert result[0][4] == float(luma[24, 24] - float(background))


def test_native_overlapping_rgb_targets_keep_float32_accumulation_order():
    rgb = np.full((32, 32, 3), 255, dtype=np.float32)
    ys, xs = np.array([15], dtype=np.intp), np.array([15], dtype=np.intp)
    batch = [(15, 15, ys, xs, 100.0)] * 3
    clean = np.array([[80.00001, 1, 0.1], [120.2, 2, 0.2], [70.7, 3, 0.3]], dtype=np.float32)
    mask = np.zeros(rgb.shape[:2], dtype=bool)
    contrasts = []
    sums, counts = native._accumulate_rgb(
        batch, rgb, clean, mask.view(np.uint8), None, None, contrasts, 6.0, True,
    )
    expected = np.zeros_like(rgb)
    for value in clean:
        expected[ys, xs] += value
    np.testing.assert_array_equal(sums, expected)
    assert counts[15, 15] == 3
    assert mask.sum() == 1
    assert contrasts == [100.0] * 3
