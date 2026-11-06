import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gptimg_restore import dots


def _flat_impulse(peak=255.0):
    image = np.full((128, 128, 3), 80.0, dtype=np.float32)
    image[63:65, 63:65] = peak
    return image


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


@pytest.mark.parametrize("peak", [255.0, 120.0])
def test_flat_two_by_two_impulse_is_replaced(peak):
    image = _flat_impulse(peak)
    original = image.copy()

    restored, report = dots.apply(image)

    assert report["candidate_count"] == 1
    assert report["masked_pixels"] == 4
    np.testing.assert_array_equal(restored[63:65, 63:65], 80.0)
    outside_core = np.ones(image.shape[:2], dtype=bool)
    outside_core[63:65, 63:65] = False
    assert np.array_equal(restored[outside_core], original[outside_core])
    assert np.array_equal(image, original)
    assert restored.shape == image.shape
    assert restored.dtype == np.float32
    json.dumps(report, allow_nan=False)


def test_analyze_reports_the_conservative_candidate():
    report = dots.analyze(_flat_impulse())

    assert report["stage"] == "dots"
    assert report["mode"] == "conservative"
    assert report["candidate_count"] == 1
    assert report["masked_pixels"] == 4
    assert report["parameters"]["texture_max_dn"] == 2.0
    json.dumps(report, allow_nan=False)


def test_strength_zero_is_an_exact_copy_without_detection(monkeypatch):
    image = _flat_impulse()
    original = image.copy()

    def fail_if_detected(*args, **kwargs):
        raise AssertionError("strength=0 must bypass detection")

    monkeypatch.setattr(dots, "_detect", fail_if_detected)
    restored, report = dots.apply(image, strength=0)

    assert np.array_equal(restored, original)
    assert restored is not image
    assert np.array_equal(image, original)
    assert report["detection_measured"] is False
    assert report["candidate_count"] is None


def test_half_strength_blends_only_the_core_and_preserves_input():
    image = _flat_impulse()
    original = image.copy()

    restored, report = dots.apply(image, strength=0.5)

    np.testing.assert_array_equal(restored[63:65, 63:65], 167.5)
    assert np.array_equal(restored[:63], original[:63])
    assert np.array_equal(restored[65:], original[65:])
    assert report["changed_pixels"] == 4
    assert image.dtype == restored.dtype == np.float32
    assert image.shape == restored.shape
    assert np.array_equal(image, original)


def test_texture_associated_star_like_dot_is_kept_by_conservative_mode():
    image = _flat_impulse()
    offsets_y, offsets_x = np.mgrid[-9:10, -9:10]
    ring = (np.hypot(offsets_x, offsets_y) >= 6) & (np.hypot(offsets_x, offsets_y) <= 9)
    ring_y, ring_x = np.nonzero(ring)
    for index, (row, column) in enumerate(zip(ring_y, ring_x)):
        image[64 + row - 9, 64 + column - 9] = 76.0 if index % 2 else 84.0
    image[63:65, 63:65] = 255.0

    conservative = dots.analyze(image)
    strict = dots.analyze(image, mode="strict")

    assert conservative["candidate_count"] == 0
    assert strict["candidate_count"] == 1


def test_real_text_pages_match_refined_mode_counts():
    reference_dir = os.environ.get("GPTIMG_RESTORE_REFERENCE_DIR")
    if not reference_dir:
        pytest.skip("set GPTIMG_RESTORE_REFERENCE_DIR to the reference work directory")

    root = Path(reference_dir)
    images_dir = root / "images" / "generated_text"
    counts_path = root / "analysis_dots" / "detector_refined_counts.json"
    filenames = ("txt_01_white.png", "txt_03_blue.png")
    required_files = [counts_path, *(images_dir / filename for filename in filenames)]
    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        pytest.skip("reference files are missing: " + ", ".join(map(str, missing_files)))
    expected_images = json.loads(counts_path.read_text(encoding="utf-8"))["images"]

    for filename in filenames:
        image = _read_rgb(images_dir / filename)
        expected = expected_images.get(filename, {"strict": 0, "conservative": 0})
        for mode in ("conservative", "strict"):
            report = dots.analyze(image, mode=mode)
            assert report["candidate_count"] == expected[mode], (
                filename,
                mode,
                report["candidate_count"],
                expected[mode],
            )
        conservative_output, conservative_report = dots.apply(image)
        assert conservative_report["candidate_count"] == expected["conservative"]
        if expected["conservative"] == 0:
            assert np.array_equal(conservative_output, image)


def test_smooth_reference_fields_remain_exactly_unchanged():
    reference_dir = os.environ.get("GPTIMG_RESTORE_REFERENCE_DIR")
    if not reference_dir:
        pytest.skip("set GPTIMG_RESTORE_REFERENCE_DIR to the reference work directory")

    images_dir = Path(reference_dir) / "images" / "generated"
    filenames = (
        "gen_01_flat_gray.png",
        "gen_02_blue_sky.png",
        "gen_03_white_studio.png",
        "gen_04_navy_gradient.png",
        "gen_10_pink_gradient.png",
    )
    required_files = [images_dir / filename for filename in filenames]
    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        pytest.skip("reference files are missing: " + ", ".join(map(str, missing_files)))

    for filename in filenames:
        image = _read_rgb(images_dir / filename)
        for mode in ("conservative", "strict"):
            restored, report = dots.apply(image, mode=mode)
            assert report["candidate_count"] == 0, (filename, mode, report["candidate_count"])
            assert np.array_equal(restored, image), (filename, mode)
