import cv2
import numpy as np

from gptimg_restore.surface import apply


def test_surface_reduces_noise_and_preserves_strokes():
    rng = np.random.default_rng(4)
    h = w = 256
    x = np.arange(w, dtype=np.float32)
    base = np.broadcast_to(100 + x[None, :] * 0.08, (h, w)).copy()
    noise = cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (0, 0), 1.0)
    noise *= 4 / noise.std()
    image = np.stack([base + noise, base + noise * 0.7, base + noise * 1.2], axis=2)
    image[65:190, 122:127] = 25
    output, report = apply(image)
    flat = np.s_[20:230, 20:90]
    assert np.std(output[flat] - base[flat][..., None]) < np.std(image[flat] - base[flat][..., None]) * 0.9
    assert np.max(np.abs(output[70:185, 122:127] - image[70:185, 122:127])) < 0.5
    assert output.shape == image.shape
    assert output.dtype == np.float32
    assert report["masked_fraction"] > 0


def test_zero_strength_is_exact_copy():
    image = np.random.default_rng(7).uniform(0, 255, (32, 40, 3)).astype(np.float32)
    output, report = apply(image, strength=0)
    assert np.array_equal(output, image)
    assert output is not image
    assert report["mean_abs_change"] == [0.0, 0.0, 0.0]


def test_clipped_white_is_untouched():
    image = np.full((64, 64, 3), 255, dtype=np.float32)
    output, _ = apply(image)
    assert np.array_equal(output, image)


def test_isoluminant_colored_stroke_is_preserved():
    image = np.full((96, 96, 3), 100, dtype=np.float32)
    image[:, 46:50] = [200, 49.063, 100]
    output, _ = apply(image)
    assert np.max(np.abs(output[:, 44:52] - image[:, 44:52])) < 0.5


def test_nonuniform_near_white_region_is_untouched():
    image = np.full((64, 64, 3), 244, dtype=np.float32)
    image[:, 24:40] = 250
    image[:, 40:48] = 255
    output, _ = apply(image)
    assert np.array_equal(output[:, 24:48], image[:, 24:48])
