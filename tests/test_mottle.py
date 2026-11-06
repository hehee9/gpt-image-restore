import json

import cv2
import numpy as np
import pytest

from gptimg_restore.mottle import analyze, apply
from gptimg_restore.cli import main
from gptimg_restore.io_utils import load_image, save_image


@pytest.mark.parametrize("chroma", [False, True])
def test_blotches_reduced_and_gradient_and_text_preserved(chroma):
    y, x = np.mgrid[:256, :320].astype(np.float32)
    clean = np.stack([100 + x * 0.07, 130 + y * 0.04, 160 - x * 0.03], axis=-1)
    noise = (2 * np.sin(2 * np.pi * x / 13) + 1.7 * np.cos(2 * np.pi * y / 23)
             + np.sin(2 * np.pi * (x + y) / 17))
    color = np.array([1, -0.299 / 0.587, 0] if chroma else [1, 1, 1], np.float32)
    image = clean + noise[..., None] * color
    glyph = np.zeros(image.shape[:2], np.uint8)
    cv2.putText(glyph, "TEXT", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 255, 3, cv2.LINE_AA)
    coverage = glyph[..., None].astype(np.float32) / 255
    image = image * (1 - coverage) + 20 * coverage
    original = image.copy()
    output, report = apply(image)
    distance = cv2.distanceTransform((glyph == 0).astype(np.uint8), cv2.DIST_L2, 5)
    selected = (distance > 8) & (x > 16) & (x < 304) & (y > 16) & (y < 240)
    assert np.sqrt(np.mean((output[selected] - clean[selected]) ** 2)) < np.sqrt(np.mean((image[selected] - clean[selected]) ** 2)) * 0.35
    assert np.array_equal(output[glyph > 0], image[glyph > 0])
    assert np.array_equal(image, original)
    assert output.dtype == np.float32
    assert report["eligible_fraction"] > 0.5
    coordinates = np.stack([np.ones_like(x), x, y], axis=-1)[selected]
    before = np.linalg.lstsq(coordinates, image[selected], rcond=None)[0]
    after = np.linalg.lstsq(coordinates, output[selected], rcond=None)[0]
    assert np.max(np.abs(before[1:] - after[1:])) < 0.001
    json.dumps(report, allow_nan=False)


def test_real_fine_texture_preserved():
    y, x = np.mgrid[:192, :256].astype(np.float32)
    texture = 8 * np.sin(2 * np.pi * (x - y) / 5) + 5 * np.cos(2 * np.pi * y / 3)
    image = np.repeat((120 + texture)[..., None], 3, axis=-1)
    output, _ = apply(image)
    assert np.array_equal(image, output)


def test_clean_linear_gradient_no_halos():
    y, x = np.mgrid[:192, :240].astype(np.float32)
    image = np.stack([100 + x * 0.1, 140 + y * 0.06, 160 - x * 0.03], axis=-1)
    glyph = np.zeros(image.shape[:2], np.uint8)
    for top in [45, 85, 125, 165]:
        cv2.putText(glyph, "GRADIENT", (8, top), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, 255, 2, cv2.LINE_AA)
    coverage = glyph[..., None].astype(np.float32) / 255
    image = image * (1 - coverage) + 20 * coverage
    output, _ = apply(image)
    assert np.max(np.abs(output - image)) < 0.01


@pytest.mark.parametrize("transpose", [False, True])
def test_finite_stroke_between_flat_colors_adds_no_fringe(transpose):
    image = np.full((256, 320, 3), 100, np.float32)
    image[:, 160:] = 110
    image[32:224, 158:162] = 0
    if transpose:
        image = image.transpose(1, 0, 2).copy()
    output, _ = apply(image)
    assert np.max(np.abs(output - image)) < 0.001


def test_local_support_cannot_cross_a_stroke_even_when_region_reconnects():
    image = np.full((256, 320, 3), 100, np.float32)
    image[:, 160:] = 103
    image[32:224, 158:162] = 0
    output, _ = apply(image)
    assert np.max(np.abs(output[80:176] - image[80:176])) < 0.001


def test_zero_strength_and_constant_noop():
    image = np.random.default_rng(40).uniform(0, 255, (31, 47, 3)).astype(np.float32)
    output, report = apply(image, strength=0)
    assert np.array_equal(image, output)
    assert output is not image
    assert report["mean_abs_change"] == [0.0] * 3
    image[:] = 128
    output, _ = apply(image)
    assert np.max(np.abs(output - image)) < 1e-4
    assert analyze(image)["score"] < 1e-4
    json.dumps(analyze(image), allow_nan=False)


def test_analysis_reports_missing_smooth_background():
    image = np.random.default_rng(41).uniform(0, 255, (96, 96, 3)).astype(np.float32)
    assert analyze(image)["status"] == "insufficient_background"


def test_cli_mottle_preserves_16_bit_gray_samples_and_alpha(tmp_path):
    y, x = np.mgrid[:96, :128].astype(np.float32)
    gray = 110 + x * 0.04 + 2 * np.sin(2 * np.pi * x / 17) + np.cos(2 * np.pi * y / 23)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    alpha = x / 127 * 255
    source = tmp_path / "gray_alpha.png"
    save_image(source, rgb, alpha, bit_depth=16)
    loaded = load_image(source)
    outdir = tmp_path / "out"
    assert main([str(source), "-o", str(outdir), "--dots", "0", "--periodic", "0",
                 "--grime", "0", "--mottle", "1", "--surface", "0", "--bit-depth", "16", "--report"]) == 0
    output = load_image(outdir / "gray_alpha_restored.png")
    assert output.bit_depth == 16
    assert np.array_equal(output.alpha, loaded.alpha)
    assert np.array_equal(output.rgb[..., 0], output.rgb[..., 1])
    assert np.array_equal(output.rgb[..., 1], output.rgb[..., 2])
    assert np.mean(np.abs(output.rgb - loaded.rgb)) > 0.25
    assert np.any(np.abs(output.rgb - np.rint(output.rgb)) > 0.01)
    report = json.loads((outdir / "gray_alpha_report.json").read_text(encoding="utf-8"))
    assert report["stages"]["mottle"]["strength"] == 1
