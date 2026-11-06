import json

import cv2
import numpy as np
import pytest

from gptimg_restore import textgap


def _text_page(ink):
    y, x = np.mgrid[:256, :288].astype(np.float32)
    background = np.stack([80 + x * 0.025, 25 + y * 0.02, 150 - x * 0.015], axis=-1)
    alpha = np.zeros((256, 288), np.uint8)
    for row in range(22, 255, 24):
        cv2.putText(alpha, "DENSE TEXT 123", (2, row), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, 255, 2, cv2.LINE_AA)
    blend = alpha.astype(np.float32) / 255
    image = background * (1 - blend[..., None]) + np.array(ink, np.float32) * blend[..., None]
    return image, background, alpha


@pytest.mark.parametrize("ink", [[5, 5, 5], [240, 235, 250]])
@pytest.mark.parametrize("pattern", ["diagonal", "chevron"])
def test_interline_pattern_removed_and_antialias_pixels_exact(ink, pattern):
    clean, background, alpha = _text_page(ink)
    y, x = np.mgrid[:256, :288].astype(np.float32)
    phase = x - y if pattern == "diagonal" else x - np.abs((y % 12) - 6)
    wave = 3 * np.sin(2 * np.pi * phase / 7)
    noise = wave[..., None] * np.array([1, 0.7, -0.5], np.float32)
    image = clean + noise * (alpha == 0)[..., None]
    original = image.copy()
    output, report = textgap.apply(image)
    distance = cv2.distanceTransform((alpha == 0).astype(np.uint8), cv2.DIST_L2, 5)
    gaps = (distance >= 2) & (distance <= 6) & (x < 195) & (y > 40) & (y < 220)
    before = np.sqrt(np.mean((image[gaps] - background[gaps]) ** 2))
    after = np.sqrt(np.mean((output[gaps] - background[gaps]) ** 2))
    assert after < before * 0.25
    assert np.array_equal(output[alpha > 0], image[alpha > 0])
    crossings = cv2.dilate((alpha >= 26).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert np.array_equal(output[crossings], image[crossings])
    assert np.array_equal(image, original)
    assert output.dtype == np.float32
    assert report["eligible_fraction"] > 0.1
    json.dumps(report)


def test_exterior_glow_removed_without_thinning_the_strokes():
    clean, background, alpha = _text_page([240, 235, 250])
    distance = cv2.distanceTransform((alpha == 0).astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.mgrid[:256, :288].astype(np.float32)
    glow = (5 + 3 * np.cos(2 * np.pi * x / 19)) * np.exp(-distance / 3)
    image = clean + glow[..., None] * np.array([0.6, 0.4, 1], np.float32) * (alpha == 0)[..., None]
    output, _ = textgap.apply(image)
    gaps = (distance >= 2) & (distance < 6) & (x < 195) & (y > 40) & (y < 220)
    assert np.std(output[gaps] - background[gaps]) < np.std(image[gaps] - background[gaps]) * 0.35
    assert np.array_equal(output[alpha > 0], image[alpha > 0])


def test_clean_gradient_and_its_antialias_are_preserved():
    image, _, _ = _text_page([240, 235, 250])
    output, report = textgap.apply(image)
    assert report["eligible_fraction"] > 0.1
    assert np.max(np.abs(output - image)) < 0.02


@pytest.mark.parametrize("texture", ["fabric", "random"])
def test_non_text_texture_is_bit_identical(texture):
    y, x = np.mgrid[:256, :288].astype(np.float32)
    if texture == "fabric":
        values = 120 + 22 * np.sin(2 * np.pi * (x - y) / 6) + 5 * np.cos(2 * np.pi * y / 17)
        image = np.repeat(values[..., None], 3, axis=2)
    else:
        image = np.random.default_rng(45).uniform(20, 230, (256, 288, 3)).astype(np.float32)
    output, report = textgap.apply(image)
    assert np.array_equal(output, image)
    assert report["eligible_fraction"] == 0


def test_disabled_textgap_is_an_exact_copy():
    image, _, _ = _text_page([240, 235, 250])
    output, _ = textgap.apply(image, strength=0)
    assert np.array_equal(output, image)
    assert output is not image


def _gray_text_page():
    image = np.full((256, 384, 3), 220, np.float32)
    alpha = np.zeros((256, 384), np.uint8)
    for row in range(22, 255, 24):
        for left in (2, 194):
            cv2.putText(alpha, "DENSE TEXT", (left, row), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, 255, 2, cv2.LINE_AA)
    return image, alpha


def test_adjacent_flat_panels_keep_their_boundary_and_colors():
    image, alpha = _gray_text_page()
    image[:, :192] = 200
    image *= 1 - alpha[..., None] / 255
    output, report = textgap.apply(image)
    assert report["eligible_fraction"] > 0.1
    assert np.array_equal(output[:, 190:194], image[:, 190:194])
    assert np.max(np.abs(output - image)) < 0.02


def test_faint_horizontal_rule_between_text_lines_is_exact():
    image, alpha = _gray_text_page()
    image *= 1 - alpha[..., None] / 255
    image[124, 12:370] = 208
    output, report = textgap.apply(image)
    assert report["eligible_fraction"] > 0.1
    assert np.array_equal(output[124, 12:370], image[124, 12:370])


@pytest.mark.parametrize("filled", [False, True])
def test_faint_icon_is_exact(filled):
    image, alpha = _gray_text_page()
    image *= 1 - alpha[..., None] / 255
    icon = np.zeros(alpha.shape, np.uint8)
    cv2.circle(icon, (180, 124), 7, 255, -1 if filled else 1, cv2.LINE_AA)
    blend = icon.astype(np.float32) / 255
    image = image * (1 - blend[..., None]) + 208 * blend[..., None]
    output, report = textgap.apply(image)
    assert report["eligible_fraction"] > 0.1
    assert np.array_equal(output[icon > 0], image[icon > 0])


def test_gradient_across_internal_tile_boundaries_stays_exact():
    y, x = np.mgrid[:640, :768].astype(np.float32)
    background = np.stack([90 + x * 0.025, 30 + y * 0.02, 160 - x * 0.015], axis=-1)
    alpha = np.zeros(x.shape, np.uint8)
    for row in range(22, 639, 24):
        for left in (2, 194, 386, 578):
            cv2.putText(alpha, "DENSE TEXT", (left, row), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, 255, 2, cv2.LINE_AA)
    blend = alpha.astype(np.float32) / 255
    image = background * (1 - blend[..., None]) + 240 * blend[..., None]
    output, report = textgap.apply(image)
    assert report["eligible_fraction"] > 0.1
    assert np.max(np.abs(output - image)) < 0.02


def _textured_row(large=False, blotches=True, shadow=False):
    height, width = (320, 900) if large else (160, 520)
    y, x = np.mgrid[:height, :width].astype(np.float32)
    values = 42 + 0.025 * x + 0.015 * y + 0.8 * np.sin(x / 35) * np.cos(y / 28)
    background = values[..., None] * np.array([1, 0.62, 0.38], np.float32)
    alpha = np.zeros((height, width), np.uint8)
    cv2.putText(alpha, "GLYPH RING", (30, 185 if large else 88), cv2.FONT_HERSHEY_SIMPLEX,
                3.1 if large else 1.05, 255, 6 if large else 2, cv2.LINE_AA)
    distance = cv2.distanceTransform((alpha == 0).astype(np.uint8), cv2.DIST_L2, 5)
    blend = alpha.astype(np.float32) / 255
    image = background * (1 - blend[..., None]) + np.array([248, 236, 212], np.float32) * blend[..., None]
    if shadow:
        image -= (5 * np.exp(-distance / 3) * (alpha == 0))[..., None]
    clean = image.copy()
    if blotches:
        pattern = (6 * np.sin(x / 5.3 + y / 9) + 4 * np.cos(x / 10 - y / 4))
        envelope = np.clip((11 - distance) / 4, 0, 1) * (distance >= 2)
        image += (pattern * envelope)[..., None]
    return image, clean, alpha, distance


@pytest.mark.parametrize("large", [False, True])
def test_single_textured_row_and_large_headline_ring_restoration(large):
    image, clean, alpha, distance = _textured_row(large=large)
    output, report = textgap.apply(image)
    ring = (distance >= 3) & (distance <= 7)
    before = np.sqrt(np.mean((image[ring] - clean[ring]) ** 2))
    after = np.sqrt(np.mean((output[ring] - clean[ring]) ** 2))
    assert report["glyph_ring"]["rows"] > 0
    assert after < 0.8 * before
    assert np.array_equal(output[alpha > 0], image[alpha > 0])
    assert np.array_equal(output[(alpha > 0) & (alpha < 26)], image[(alpha > 0) & (alpha < 26)])
    assert np.array_equal(output[distance > 18], image[distance > 18])


def test_uniform_contour_shadow_preserved_on_textured_row():
    image, _, alpha, distance = _textured_row(blotches=False, shadow=True)
    output, report = textgap.apply(image)
    assert report["glyph_ring"]["rows"] > 0
    ring = (distance >= 3) & (distance <= 7)
    assert np.sqrt(np.mean((output[ring] - image[ring]) ** 2)) < 0.35
    assert np.array_equal(output[alpha > 0], image[alpha > 0])


@pytest.mark.parametrize("sigma, quantized", [(0.3, False), (0.45, False), (1.0, False), (1.0, True)])
def test_soft_text_edges_keep_core_and_bound_faint_alpha_change(sigma, quantized):
    y, x = np.mgrid[:180, :520].astype(np.float32)
    values = 42 + 0.025 * x + 0.015 * y + 0.8 * np.sin(x / 35) * np.cos(y / 28)
    background = values[..., None] * np.array([1, 0.62, 0.38], np.float32)
    alpha = np.zeros(x.shape, np.uint8)
    cv2.putText(alpha, "GLYPH RING", (25, 105), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, 255, 3, cv2.LINE_AA)
    alpha = (cv2.GaussianBlur(alpha, (0, 0), sigma).astype(np.float32) / 255 if quantized
             else cv2.GaussianBlur(alpha.astype(np.float32) / 255, (0, 0), sigma))
    image = background * (1 - alpha[..., None]) + np.array([248, 236, 212], np.float32) * alpha[..., None]
    image += (5 * np.sin(x / 5.3 + y / 9) * (alpha == 0))[..., None]
    output, report = textgap.apply(image)
    assert report["glyph_ring"]["rows"] > 0
    core = cv2.dilate((alpha >= 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert np.array_equal(output[core], image[core])
    maximum_change = 1 if sigma >= 1 else 2
    assert np.max(np.abs(output[alpha > 0] - image[alpha > 0])) <= maximum_change
    faint = (alpha > 0) & (alpha < 0.01)
    assert np.count_nonzero(faint) > 0
    assert np.max(np.abs(output[faint] - image[faint])) <= maximum_change
    assert np.count_nonzero(output[alpha == 0] != image[alpha == 0]) > 0


@pytest.mark.parametrize("shape", ["rectangle", "circle"])
def test_repeated_outlined_shapes_are_not_text(shape):
    y, x = np.mgrid[:180, :520].astype(np.float32)
    values = 42 + 0.025 * x + 0.015 * y + 0.8 * np.sin(x / 35) * np.cos(y / 28)
    image = values[..., None] * np.array([1, 0.62, 0.38], np.float32)
    for left in range(20, 460, 40):
        if shape == "rectangle":
            cv2.rectangle(image, (left, 65), (left + 22, 102), (248, 236, 212), 3)
        else:
            cv2.circle(image, (left + 11, 84), 16, (248, 236, 212), 3)
    output, report = textgap.apply(image)
    assert report["glyph_ring"]["rows"] == 0
    assert np.array_equal(output, image)


def test_korean_row_with_short_jamo_fragments_is_detected():
    # '달콤하고 고소한 조화'의 19×175 이진 획 마스크.
    packed = bytes.fromhex(
        "003c00000003e000000000000000000f8000000001e0003c0fff0003c0000000000000000f8f00000003a1e3ff301ffe"
        "0fe38000e000007007801f8e00007807e1c7fee0001c0fc703ffe003ffe007800cdc003ff803930700c07ff8026e07fb"
        "c007ffc00e03fff8003de07ff60e03f9ffe3fffc0007000003003807ff70000780ffcc1c07f1f9c7ffb8000e00000e00"
        "fc00f0fc001e003c1838fe007b807870001c00001c03fe07f9fc007f00fe327fdc00f603f8fc1f38000e380e1f0e73f0"
        "01df838e7efc1841ffef39f81e60000e60398e18e7000f4f871dfc0031ffffdc33801cc0001cc1e70c39ce0079c30ff3"
        "807fe1f80038e7003b800039830f003f1c00e3c00fc700ffc067f079ce0076000072000e0018380007000f1e000380ff"
        "e07f1c00c08000c00018307060000e049ffc07ff00e9c07c38ffffc0fffffffff0e0c01fffffff9807fc0183800071ff"
        "ff81ffffffffe0c0003ffffff0300e0403070000c1c00001c0007c0001818018000000e01ffc07fe0001800000000000"
        "000003ff8000000001c03ff80ffc0003000000000000000007fe000000000380"
    )
    glyphs = np.unpackbits(np.frombuffer(packed, np.uint8))[:19 * 175].reshape(19, 175)
    _, _, stats, _ = cv2.connectedComponentsWithStats(glyphs)
    assert np.any(stats[1:, cv2.CC_STAT_HEIGHT] == 11)
    assert np.any(stats[1:, cv2.CC_STAT_HEIGHT] == 19)
    alpha = cv2.GaussianBlur(np.pad(glyphs, 20).astype(np.float32), (0, 0), 0.35)
    y, x = np.mgrid[:alpha.shape[0], :alpha.shape[1]].astype(np.float32)
    background = (40 + 0.015 * x + 0.02 * y)[..., None] * np.array([1, 0.62, 0.38], np.float32)
    image = background * (1 - alpha[..., None]) + np.array([248, 236, 212], np.float32) * alpha[..., None]
    rows = textgap.glyphring._rows_at_scale(image, 1)
    assert any(left <= 20 and right >= 195 and height >= 18 for left, _, right, _, height, _ in rows)


@pytest.mark.parametrize("large", [False, True])
def test_first_glyph_left_of_detected_row_keeps_its_full_core_support(monkeypatch, large):
    image, _, alpha, _ = _textured_row(large=large)
    left, top, right, bottom, height, paint = textgap.glyphring._text_rows(image)[0]
    partial = (left + round(0.95 * height), top, right, bottom, height, paint)
    monkeypatch.setattr(textgap.glyphring, "_text_rows", lambda _: [partial])
    output, report = textgap.glyphring.apply(image, 1)
    core = cv2.dilate((alpha >= 128).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    core[:, int(left + 1.5 * height):] = False
    assert report["rows"] > 0
    assert np.array_equal(output[core], image[core])


def test_glyph_ring_outer_taper_has_no_boundary_step():
    image, _, _, distance = _textured_row()
    output, _ = textgap.apply(image)
    change = np.max(np.abs(output - image), axis=2)
    assert np.max(change[(distance >= 8.8) & (distance < 9.2)]) < 0.15
    assert np.array_equal(output[distance >= 10], image[distance >= 10])


def test_neighboring_text_rows_preserve_each_others_full_antialias():
    y, x = np.mgrid[:180, :420].astype(np.float32)
    background = np.stack([45 + x * 0.025, 25 + y * 0.01, 17 + x * 0.012], axis=2)
    alpha = np.zeros(x.shape, np.uint8)
    for baseline in (78, 110):
        cv2.putText(alpha, "GLYPH RING", (30, baseline), cv2.FONT_HERSHEY_SIMPLEX,
                    1.05, 255, 2, cv2.LINE_AA)
    blend = alpha.astype(np.float32) / 255
    image = background * (1 - blend[..., None]) + 245 * blend[..., None]
    image += (4 * np.sin(x / 6) * (alpha == 0))[..., None]
    output, report = textgap.apply(image)
    assert report["glyph_ring"]["rows"] >= 2
    assert np.array_equal(output[alpha > 0], image[alpha > 0])


def test_curved_bokeh_behind_text_has_only_negligible_correction():
    image, _, alpha, distance = _textured_row(blotches=False)
    y, x = np.mgrid[:image.shape[0], :image.shape[1]].astype(np.float32)
    bokeh = 100 * np.exp(-((x - 118) ** 2 + (y - 83) ** 2) / (2 * 18 ** 2))
    image += (bokeh * (1 - alpha.astype(np.float32) / 255))[..., None]
    output, _ = textgap.apply(image)
    ring = (distance >= 2) & (distance <= 9)
    assert np.sqrt(np.mean((output[ring] - image[ring]) ** 2)) < 0.5
    assert np.array_equal(output[alpha > 0], image[alpha > 0])


def test_reduced_resolution_plane_keeps_masked_foreground_out_of_samples():
    y, x = np.mgrid[:80, :900].astype(np.float32)
    background = np.stack([60 + x * 0.03, 30 + y * 0.02, 15 + x * 0.01], axis=2)
    samples = np.ones(x.shape, np.float32)
    samples[25:55, (np.arange(900) % 20) < 6] = 0
    image = background.copy()
    image[samples == 0] = 240
    model, mass, determinant = textgap.glyphring._local_plane(image, samples, 12)
    supported = (mass > 0.02) & (determinant > 1) & (x > 20) & (x < 879) & (y > 10) & (y < 69)
    assert np.max(np.abs(model[supported] - background[supported])) < 0.05


@pytest.mark.parametrize("kind", ["skin", "fur", "bokeh"])
def test_photo_like_textures_without_text_stay_exact(kind):
    y, x = np.mgrid[:256, :384].astype(np.float32)
    rng = np.random.default_rng(314)
    values = 110 + 12 * np.sin(x / 70) + 10 * np.cos(y / 60)
    if kind == "skin":
        spots = np.zeros(values.shape, np.float32)
        for px, py in rng.integers([0, 0], [384, 256], (65, 2)):
            cv2.circle(spots, (int(px), int(py)), 2, -12, -1)
        values += cv2.GaussianBlur(spots, (0, 0), 0.6)
    elif kind == "fur":
        values += 27 * np.sin(x * 0.6 + 3 * np.sin(y / 22)) + 8 * np.cos((x + y) / 15)
    else:
        values = values.copy()
        for px, py in rng.integers([0, 0], [384, 256], (20, 2)):
            values += 65 * np.exp(-((x - px) ** 2 + (y - py) ** 2) / 140)
    image = values[..., None] * np.array([1, 0.73, 0.5], np.float32)
    output, report = textgap.apply(image)
    assert np.array_equal(output, image)
    assert report["eligible_fraction"] == 0


@pytest.mark.parametrize("ink", [[5, 5, 5], [240, 235, 250]])
def test_document_background_removes_broad_noise_and_keeps_glyph_margin(ink):
    clean, background, alpha = _text_page(ink)
    y, x = np.mgrid[:256, :288].astype(np.float32)
    noise = 3 * np.cos(2 * np.pi * x / 144) * np.cos(2 * np.pi * y / 128)
    noise += np.sin((x - y) / 2)
    image = clean + (noise * (alpha == 0))[..., None]
    output, report = textgap._restore_document_background(image)
    protected = cv2.dilate((alpha >= 26).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    background_pixels = cv2.dilate((alpha > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0
    before = np.sqrt(np.mean((image[background_pixels] - background[background_pixels]) ** 2))
    after = np.sqrt(np.mean((output[background_pixels] - background[background_pixels]) ** 2))
    assert report["accepted"]
    assert after < before * 0.2
    assert np.array_equal(output[protected], image[protected])
    assert np.array_equal(output[alpha > 0], image[alpha > 0])
    assert output.dtype == np.float32
    json.dumps(report)


@pytest.mark.parametrize("kind", ["no_text", "textured_panel"])
def test_document_background_rejects_missing_text_or_distributed_surface(kind):
    image, background, _ = _text_page([5, 5, 5])
    if kind == "no_text":
        y, x = np.mgrid[:256, :288].astype(np.float32)
        image = background + (2 * np.sin(x / 30) * np.cos(y / 40))[..., None]
    else:
        image[-64:] = np.random.default_rng(37).uniform(0, 255, image[-64:].shape)
    output, report = textgap._restore_document_background(image)
    assert not report["accepted"]
    assert np.array_equal(output, image)


def test_document_background_keeps_float_gradient_without_internal_quantization():
    image, background, alpha = _text_page([5, 5, 5])
    output, report = textgap._restore_document_background(image)
    safe = cv2.dilate((alpha > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0
    assert report["accepted"]
    assert np.max(np.abs(output[safe] - background[safe])) < 0.01
    assert np.count_nonzero(np.abs(output[safe] - np.rint(output[safe])) > 0.1) > 1000
    assert np.ptp(output[..., 0][safe]) > 6


@pytest.mark.parametrize("strength", [0.5, 1.0])
def test_document_background_strong_gate_preserves_existing_paths(monkeypatch, strength):
    image, _, _ = _text_page([5, 5, 5])
    def unexpected(_):
        pytest.fail("Document background restoration must run only at strength >= 1.1")
    monkeypatch.setattr(textgap, "_restore_document_background", unexpected)
    _, report = textgap.apply(image, strength=strength)
    assert "document_background" not in report
    output, document_report = textgap.restore_document_background(image, strength=strength)
    assert np.array_equal(output, image)
    assert not document_report["accepted"]


def test_document_background_strong_route_retains_faint_independent_rule():
    image, _, _ = _text_page([5, 5, 5])
    image[124, 214:278] = [75, 18, 145]
    output, report = textgap._restore_document_background(image)
    assert report["accepted"]
    assert np.array_equal(output[124, 214:278], image[124, 214:278])


def test_document_background_dither_mask_excludes_glyph_and_one_pixel_margin():
    image, _, alpha = _text_page([5, 5, 5])
    mask = np.zeros(image.shape[:2], bool)
    _, report = textgap._restore_document_background(image, document_mask=mask)
    protected = cv2.dilate((alpha >= 26).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert report["accepted"]
    assert np.count_nonzero(mask) > image.shape[0] * image.shape[1] * 0.3
    assert not np.any(mask[protected])
    assert not np.any(mask[alpha > 0])


def test_document_background_certifies_repeated_text_above_1600_pixels():
    image, _, alpha = _text_page([5, 5, 5])
    image = np.tile(image, (7, 7, 1))
    alpha = np.tile(alpha, (7, 7))
    output, report = textgap._restore_document_background(image)
    protected = cv2.dilate((alpha >= 26).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert max(image.shape[:2]) > 1600
    assert report["accepted"]
    assert np.array_equal(output[protected], image[protected])


@pytest.mark.parametrize("contrast", [3, 4, 6])
@pytest.mark.parametrize("thickness", [1, 2])
def test_document_background_preserves_very_faint_rules(contrast, thickness):
    image, background, alpha = _text_page([5, 5, 5])
    shape = np.zeros(alpha.shape, np.float32)
    shape[124:124 + thickness, 214:278] = 1
    image -= contrast * shape[..., None]
    output, report = textgap._restore_document_background(image)
    assert report["accepted"]
    assert np.max(np.abs(output[shape > 0] - image[shape > 0])) <= 0.5
    assert abs(np.mean(output[shape > 0] - background[shape > 0]) + contrast) <= 0.5


@pytest.mark.parametrize("kind", ["dashed", "box", "watermark_hard", "watermark_soft"])
def test_document_background_keeps_faint_coherent_shapes(kind):
    image, background, alpha = _text_page([5, 5, 5])
    shape = np.zeros(alpha.shape, np.float32)
    contrast = 5 if kind.startswith("watermark") else 4
    if kind == "dashed":
        for left in range(214, 278, 10):
            shape[124, left:left + 6] = 1
    elif kind == "box":
        cv2.rectangle(shape, (212, 92), (280, 152), 1, 1, cv2.LINE_AA)
    else:
        shape[92:152, 212:280] = 1
        if kind == "watermark_soft":
            shape = cv2.GaussianBlur(shape, (0, 0), 3.5)
    image += contrast * shape[..., None]
    output, report = textgap._restore_document_background(image)
    content = (shape >= 0.1) & (alpha == 0)
    assert report["accepted"]
    assert np.max(np.abs(output[content] - image[content])) <= 0.5
    assert np.max(np.abs(output[alpha > 0] - image[alpha > 0])) == 0
    outside = (shape < 0.001) & (alpha == 0)
    outside &= cv2.dilate((shape > 0.001).astype(np.uint8), np.ones((33, 33), np.uint8)) == 0
    outside &= cv2.dilate((alpha > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) == 0
    assert np.sqrt(np.mean((output[outside] - background[outside]) ** 2)) < 0.1


@pytest.mark.parametrize("top", [126, 238])
def test_document_background_keeps_light_gray_table_row_shading(top):
    image, alpha = _gray_text_page()
    image *= 1 - alpha[..., None] / 255
    row = np.zeros(alpha.shape, bool)
    row[top:top + 18] = True
    row &= alpha == 0
    image[row] -= 4
    output, report = textgap._restore_document_background(image)
    assert report["accepted"]
    assert np.max(np.abs(output[row] - image[row])) <= 0.5
    assert abs(np.mean(output[row]) - 216) <= 0.5


def test_document_background_still_flattens_smooth_gpt_like_blobs():
    image, background, alpha = _text_page([5, 5, 5])
    y, x = np.mgrid[:256, :288].astype(np.float32)
    blob = 3 * np.exp(-((x - 233) ** 2 / (2 * 22 ** 2) + (y - 125) ** 2 / (2 * 35 ** 2)))
    blob -= 2.5 * np.exp(-((x - 81) ** 2 + (y - 145) ** 2) / (2 * 28 ** 2))
    image += (blob * (alpha == 0))[..., None]
    output, report = textgap._restore_document_background(image)
    safe = cv2.dilate((alpha > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) == 0
    before = np.std((image - textgap._document_plane(image, safe))[safe])
    after = np.std((output - textgap._document_plane(output, safe))[safe])
    assert report["accepted"]
    assert after < before * 0.25


@pytest.mark.parametrize("kind", ["text", "diagonal_rule"])
def test_document_background_keeps_faint_repeated_text_and_oriented_rules(kind):
    image, _, alpha = _text_page([5, 5, 5])
    shape = np.zeros(alpha.shape, np.uint8)
    if kind == "text":
        cv2.putText(shape, "COPY TEXT", (212, 128), cv2.FONT_HERSHEY_SIMPLEX,
                    0.28, 255, 1, cv2.LINE_AA)
    else:
        cv2.line(shape, (218, 99), (270, 151), 255, 1, cv2.LINE_AA)
    blend = shape.astype(np.float32) / 255
    image += (5 if kind == "text" else 3) * blend[..., None]
    output, report = textgap._restore_document_background(image)
    assert report["accepted"]
    assert np.max(np.abs(output[shape >= 26] - image[shape >= 26])) <= 0.5


@pytest.mark.parametrize("thickness", [-1, 2])
def test_document_background_preserves_faint_circular_logo(thickness):
    image, _, alpha = _text_page([5, 5, 5])
    shape = np.zeros(alpha.shape, np.float32)
    cv2.circle(shape, (245, 124), 24, 1, thickness)
    image += 4 * shape[..., None]
    output, report = textgap._restore_document_background(image)
    assert report["accepted"]
    assert np.array_equal(output[shape > 0], image[shape > 0])


@pytest.mark.parametrize("direction", ["horizontal", "vertical", "diagonal"])
def test_document_background_flattens_smooth_sine_bands(direction):
    image, _, alpha = _text_page([5, 5, 5])
    y, x = np.mgrid[:256, :288].astype(np.float32)
    coordinate = y if direction == "horizontal" else x if direction == "vertical" else x + y
    noise = 4 * np.sin(coordinate / 7)
    image += (noise * (alpha == 0))[..., None]
    output, report = textgap._restore_document_background(image)
    safe = cv2.dilate((alpha > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
    strong = safe & (np.abs(noise) > 2)
    assert report["accepted"]
    assert np.mean(np.max(np.abs(output[strong] - image[strong]), axis=1) < 0.01) < 0.01
    assert np.std((output - textgap._document_plane(output, safe))[safe]) < 0.1


def test_document_background_flattens_independent_pixel_noise():
    image, _, alpha = _text_page([5, 5, 5])
    noise = np.random.default_rng(714).normal(0, 2, image.shape).astype(np.float32)
    image += noise * (alpha == 0)[..., None]
    output, report = textgap._restore_document_background(image)
    safe = cv2.dilate((alpha > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
    assert report["accepted"]
    assert np.mean(np.max(np.abs(output[safe] - image[safe]), axis=1) < 0.01) < 0.01
    assert np.std((output - textgap._document_plane(output, safe))[safe]) < 0.1


def test_faint_content_protector_ignores_large_independent_noise():
    import time

    noise = np.random.default_rng(715).normal(0, 2, (2048, 2048, 3)).astype(np.float32)
    start = time.perf_counter()
    protected = textgap._faint_document_details(noise, np.zeros((2048, 2048), bool))
    elapsed = time.perf_counter() - start
    assert np.mean(protected) < 0.01
    assert elapsed < 15


def _repeated_frame_row(styles):
    y, x = np.mgrid[:180, :520].astype(np.float32)
    values = 42 + 0.025 * x + 0.015 * y + 0.8 * np.sin(x / 35) * np.cos(y / 28)
    image = values[..., None] * np.array([1, 0.62, 0.38], np.float32)
    image += (5 * np.sin(x / 5.3 + y / 9) + 4 * np.cos(x / 10 - y / 4))[..., None]
    for left, style in zip(range(20, 460, 40), styles):
        cv2.rectangle(image, (left, 65), (left + 22, 102), (248, 236, 212), 3)
        if style == 1:
            cv2.line(image, (left + 11, 65), (left + 11, 102), (248, 236, 212), 3)
        elif style == 2:
            cv2.line(image, (left, 83), (left + 22, 83), (248, 236, 212), 3)
    return image


@pytest.mark.parametrize("period", [2, 3, 4])
def test_varied_repeated_grille_is_not_text(period):
    image = _repeated_frame_row([int(index % period == 0) for index in range(11)])
    output, report = textgap.glyphring.apply(image, 1)
    assert report["rows"] == 0
    assert np.array_equal(output, image)


def test_repeated_ui_icons_with_three_templates_are_not_text():
    image = _repeated_frame_row([index % 3 for index in range(11)])
    output, report = textgap.glyphring.apply(image, 1)
    assert report["rows"] == 0
    assert np.array_equal(output, image)
