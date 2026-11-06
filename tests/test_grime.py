import json

import cv2
import numpy as np
import pytest

from gptimg_restore.grime import analyze, apply
from gptimg_restore import grime


@pytest.mark.parametrize("angle", [30, 37, 45])
@pytest.mark.parametrize("period", [3, 4.5, 6])
@pytest.mark.parametrize("chroma", [False, True])
def test_oriented_hatching_reduction(angle, period, chroma):
    y, x = np.mgrid[:192, :192].astype(np.float32)
    base = np.stack([100 + x * 0.03, 130 + y * 0.02, 155 + x * 0.01], axis=-1)
    phase = (y * np.cos(np.radians(angle)) - x * np.sin(np.radians(angle))) / period
    noise = 3 * np.sin(2 * np.pi * phase)
    color = np.array([1, -0.299 / 0.587, 0] if chroma else [1, 1, 1], dtype=np.float32)
    image = (base + noise[..., None] * color).astype(np.float32)
    original = image.copy()
    output, report = apply(image)
    inside = np.s_[16:-16, 16:-16]
    before = np.sqrt(np.mean((image[inside] - base[inside]) ** 2))
    after = np.sqrt(np.mean((output[inside] - base[inside]) ** 2))
    assert after < before * 0.65
    assert np.array_equal(image, original)
    assert output.shape == image.shape
    assert output.dtype == np.float32
    json.dumps(report)


@pytest.mark.parametrize("color", [[20, 20, 20], [200, 49.063, 100]])
def test_text_strokes_and_edge_neighbors_are_preserved(color):
    image = np.full((192, 192, 3), 100, dtype=np.float32)
    glyphs = np.zeros((192, 192), dtype=np.uint8)
    cv2.putText(glyphs, "TEXT", (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 255, 2)
    image[glyphs > 0] = color
    image[110:170, 40:42] = color
    y, x = np.mgrid[:192, :192]
    image += (1.5 * np.sin(2 * np.pi * (x - y) / 5))[..., None].astype(np.float32)
    output, _ = apply(image)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, scale=0.125)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, scale=0.125)
    edges = np.max(gx * gx + gy * gy, axis=-1) > 100
    adjacent = cv2.dilate(edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert np.max(np.abs(output[adjacent] - image[adjacent])) == 0


def test_regular_fabric_preserved_at_high_contrast():
    y, x = np.mgrid[:192, :192].astype(np.float32)
    fabric = 100 + 25 * np.sin(2 * np.pi * (x - y) / (5 * np.sqrt(2)))
    image = np.repeat(fabric[..., None], 3, axis=-1).astype(np.float32)
    output, _ = apply(image)
    assert np.sqrt(np.mean((output - image) ** 2)) < 0.05


def test_low_contrast_regular_texture_has_same_tradeoff_as_grime():
    y, x = np.mgrid[:192, :192].astype(np.float32)
    fabric = 100 + 2 * np.sin(2 * np.pi * (x - y) / (5 * np.sqrt(2)))
    image = np.repeat(fabric[..., None], 3, axis=-1).astype(np.float32)
    output, _ = apply(image)
    # 같은 주파수와 대비를 가진 실제 직물 무늬도 줄어드는 절충점 명시.
    assert output[16:-16, 16:-16].std() < image[16:-16, 16:-16].std() * 0.65


def test_zero_strength_exact_copy_and_clipped_white():
    image = np.random.default_rng(31).uniform(0, 255, (37, 53, 3)).astype(np.float32)
    output, report = apply(image, strength=0)
    assert np.array_equal(image, output)
    assert image is not output
    assert report["mean_abs_change"] == [0.0, 0.0, 0.0]
    image[5:15, 10:30] = [250, 253, 255]
    output, _ = apply(image)
    assert np.array_equal(output[5:15, 10:30], image[5:15, 10:30])


def test_analyze_score_and_angle_respond_to_hatching():
    y, x = np.mgrid[:192, :192].astype(np.float32)
    base = np.full((192, 192, 3), 100, dtype=np.float32)
    hatch = 3 * np.sin(2 * np.pi * (y * np.cos(np.radians(37)) - x * np.sin(np.radians(37))) / 4.5)
    report = analyze((base + hatch[..., None]).astype(np.float32))
    clean = analyze(base)
    assert report["score"] > clean["score"] + 10
    assert abs(report["dominant_angle_degrees"] - 37) < 10
    assert abs(report["dominant_period_pixels"] - 4.5) < 0.6
    json.dumps(report)


def test_analyze_does_not_report_text_edges_as_grime():
    glyphs = np.zeros((192, 192), dtype=np.uint8)
    for y in [45, 90, 135, 180]:
        cv2.putText(glyphs, "TEXT", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    image = np.full((192, 192, 3), 100, dtype=np.float32)
    image[glyphs > 0] = [200, 49.063, 100]
    report = analyze(image)
    assert report["status"] == "ok"
    assert report["score"] < 2


def test_analyze_marks_missing_background_instead_of_zero_severity():
    image = np.random.default_rng(32).uniform(30, 220, (192, 192, 3)).astype(np.float32)
    report = analyze(image)
    assert report["status"] == "insufficient_background"
    assert report["score"] is None
    assert report["sampled_tiles"] == 0
    json.dumps(report)


@pytest.mark.parametrize("ink", [[20, 20, 20], [200, 49.063, 100], [240, 240, 240]])
@pytest.mark.parametrize("chroma", [False, True])
def test_dense_text_gaps_cleaned_without_changing_antialiased_strokes(ink, chroma):
    y, x = np.mgrid[:192, :240].astype(np.float32)
    base = np.stack([100 + x * 0.03, 100 + x * 0.03, 100 + x * 0.03], axis=-1)
    alpha = np.zeros((192, 240), dtype=np.uint8)
    for row in range(22, 190, 24):
        cv2.putText(alpha, "DENSE TEXT 123", (2, row), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, 255, 2, cv2.LINE_AA)
    blend = alpha.astype(np.float32) / 255
    clean = base * (1 - blend[..., None]) + np.array(ink, np.float32) * blend[..., None]
    wave = 3 * np.sin(2 * np.pi * (y * np.cos(np.radians(37)) - x * np.sin(np.radians(37))) / 4.5)
    color = np.array([1, -0.299 / 0.587, 0] if chroma else [1, 1, 1], np.float32)
    image = (clean + wave[..., None] * color).astype(np.float32)
    output, report = apply(image)
    distance = cv2.distanceTransform((alpha == 0).astype(np.uint8), cv2.DIST_L2, 5)
    gaps = (distance > 4.5) & (distance < 9)
    assert np.count_nonzero(gaps) > 500
    assert np.sqrt(np.mean((output[gaps] - clean[gaps]) ** 2)) < np.sqrt(np.mean((image[gaps] - clean[gaps]) ** 2)) * 0.65
    assert np.array_equal(output[alpha > 0], image[alpha > 0])
    assert report["text_gap_fraction"] > 0.05


def test_clean_gradient_between_glyphs_has_no_visible_halo_or_color_fringe():
    y, x = np.mgrid[:192, :240].astype(np.float32)
    image = np.stack([100 + x * 0.03, 120 + x * 0.02, 150 - x * 0.01], axis=-1)
    alpha = np.zeros((192, 240), dtype=np.uint8)
    for row in range(22, 190, 24):
        cv2.putText(alpha, "DENSE TEXT 123", (2, row), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, 255, 2, cv2.LINE_AA)
    blend = alpha.astype(np.float32) / 255
    image = image * (1 - blend[..., None]) + 20 * blend[..., None]
    output, _ = apply(image)
    assert np.max(np.abs(output - image)) < 0.5


@pytest.mark.parametrize("chroma", [False, True])
def test_cloudy_checker_interiors_cleaned_without_color_boundary_bleed(chroma):
    y, x = np.mgrid[:216, :216].astype(np.float32)
    checker = ((x // 36 + y // 36) % 2).astype(bool)
    clean = np.where(checker[..., None], np.array([45, 210, 90], np.float32),
                     np.array([7, 147, 43], np.float32))
    cloud = (1.7 * np.sin(2 * np.pi * x / 15)
             + 1.5 * np.cos(2 * np.pi * y / 19)
             + 0.7 * np.sin(2 * np.pi * (x - y) / 5))
    color = np.array([1, -0.299 / 0.587, 0] if chroma else [1, 1, 1], np.float32)
    image = clean + cloud[..., None] * color
    output, report = apply(image)
    inside = (x > 36) & (x < 180) & (y > 36) & (y < 180)
    inside &= (x % 36 > 8) & (x % 36 < 28) & (y % 36 > 8) & (y % 36 < 28)
    before = np.sqrt(np.mean((image[inside] - clean[inside]) ** 2))
    after = np.sqrt(np.mean((output[inside] - clean[inside]) ** 2))
    assert after < before * 0.6
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, scale=0.125)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, scale=0.125)
    edges = np.max(gx * gx + gy * gy, axis=-1) > 100
    ring = cv2.dilate(edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert np.array_equal(output[ring], image[ring])
    assert report["flat_interior_fraction"] > 0.25


def test_enclosed_clean_gradient_keeps_its_trend():
    y, x = np.mgrid[:144, :144].astype(np.float32)
    image = np.full((144, 144, 3), 20, np.float32)
    gradient = np.stack([100 + x * 0.08, 120 + y * 0.04, 150 - x * 0.03], axis=-1)
    image[24:120, 24:120] = gradient[24:120, 24:120]
    output, report = apply(image)
    assert np.max(np.abs(output - image)) < 0.5
    assert report["flat_interior_fraction"] > 0.25


def test_wobbly_antialiased_checker_edges_and_halos_are_reduced(monkeypatch):
    y, x = np.mgrid[:216, :216].astype(np.float32)

    def _coverage(position):
        distance = np.sin(np.pi * (position + 0.5) / 36) * (36 / np.pi)
        return (np.clip(distance + 0.5, 0, 1)
                + 0.07 * np.sign(distance) * np.exp(-((np.abs(distance) - 1.5) ** 2) / 0.5))

    ax = _coverage(x + 0.45 * np.sin(2 * np.pi * y / 17))
    ay = _coverage(y + 0.35 * np.sin(2 * np.pi * x / 23))
    alpha = ax * ay + (1 - ax) * (1 - ay)
    light, dark = np.array([45, 210, 90], np.float32), np.array([7, 147, 43], np.float32)
    image = dark + alpha[..., None] * (light - dark)
    output, _ = apply(image)
    monkeypatch.setattr(grime, "_flat_boundary_estimate", lambda rgb, labels, eligible:
                        (np.zeros_like(rgb), np.zeros(rgb.shape[:2], np.float32)))
    interior_only, _ = apply(image)

    def _boundary_metrics(array):
        alpha = (array[44:64, 68:77, 1] - dark[1]) / (light[1] - dark[1])
        crossings = []
        for row in alpha:
            index = np.flatnonzero((row[:-1] >= 0.5) & (row[1:] < 0.5))[0]
            crossings.append(index + (row[index] - 0.5) / (row[index] - row[index + 1]))
        halo = np.mean(np.maximum(np.maximum(-alpha, alpha - 1), 0))
        return np.std(crossings), halo

    before_jitter, before_halo = _boundary_metrics(interior_only)
    after_jitter, after_halo = _boundary_metrics(output)
    assert after_jitter < before_jitter * 0.65
    assert after_halo < before_halo * 0.65
    outside = (x < 36) | (x > 180) | (y < 36) | (y > 180)
    assert np.array_equal(output[outside], interior_only[outside])


def test_boundary_cleanup_is_noop_when_only_one_side_is_flat(monkeypatch):
    y, x = np.mgrid[:192, :192].astype(np.float32)
    alpha = np.clip(x - 95.5 + 0.45 * np.sin(2 * np.pi * y / 17) + 0.5, 0, 1)
    first, second = np.array([45, 210, 90], np.float32), np.array([7, 147, 43], np.float32)
    pixels = first + alpha[..., None] * (second - first)
    pixels += alpha[..., None] * (10 * np.sin(2 * np.pi * x / 11))[..., None]
    image = np.full((192, 192, 3), 20, np.float32)
    image[24:168, 24:168] = pixels[24:168, 24:168]
    output, report = apply(image)
    monkeypatch.setattr(grime, "_flat_boundary_estimate", lambda rgb, labels, eligible:
                        (np.zeros_like(rgb), np.zeros(rgb.shape[:2], np.float32)))
    interior_only, _ = apply(image)
    assert report["flat_interior_fraction"] > 0.1
    assert np.array_equal(output, interior_only)
