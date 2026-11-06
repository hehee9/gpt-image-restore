"""@description 윤곽 잔물결 감소와 문자/선화/사진 보호의 회귀 검사."""

from io import BytesIO

import cv2
import numpy as np
import pytest

from gptimg_restore import contour
from gptimg_restore.cli import PRESET_STRENGTHS


def _ellipse(ripple=0.0, notch=False, size=512):
    y, x = np.mgrid[:size, :size].astype(np.float32)
    angle = np.arctan2((y - size / 2) / 0.65, x - size / 2)
    radius = np.sqrt((x - size / 2) ** 2 + ((y - size / 2) / 0.65) ** 2)
    boundary = size * 0.34 + ripple * np.sin(angle * 48)
    if notch:
        boundary += 16 * np.exp(-((angle - 0.55) / 0.055) ** 2)
    depth = radius - boundary
    gray = 65 + 95 / (1 + np.exp(np.clip(depth / 1.4, -60, 60)))
    gray += 20 * np.exp(-((depth + 3) / 2) ** 2)
    return np.repeat(gray[..., None], 3, axis=2).astype(np.float32), depth


def test_smooth_ellipse_ripple_is_reduced_without_overshoot():
    clean, depth = _ellipse()
    source, _ = _ellipse(0.5)
    restored, report = contour.apply(source)
    rim = np.abs(depth) < 6
    before = np.sqrt(np.mean((source[rim] - clean[rim]) ** 2))
    after = np.sqrt(np.mean((restored[rim] - clean[rim]) ** 2))
    assert after < 0.85 * before
    assert restored.dtype == np.float32
    assert restored.min() >= source.min()
    assert restored.max() <= source.max()
    assert report["accepted_contours"] > 0


def test_genuine_notch_keeps_shape_and_contrast():
    source, depth = _ellipse(0.4, notch=True)
    restored, _ = contour.apply(source)
    y, x = np.mgrid[:512, :512]
    angle = np.arctan2((y - 256) / 0.65, x - 256)
    notch = (np.abs(angle - 0.55) < 0.09) & (np.abs(depth) < 5)
    assert np.max(np.abs(restored[notch] - source[notch])) < 2
    assert np.ptp(restored[notch]) > 0.95 * np.ptp(source[notch])


def test_dense_parallel_hair_lines_stay_exact():
    source = np.full((256, 384, 3), 100, np.float32)
    source[:, 2::5] = 210
    restored, _ = contour.apply(source)
    np.testing.assert_array_equal(restored, source)


def test_text_core_and_antialiasing_stay_exact():
    source = np.full((220, 620, 3), 55, np.uint8)
    cv2.putText(source, "SMOOTH TEXT", (12, 135), cv2.FONT_HERSHEY_SIMPLEX,
                2.1, (240, 240, 240), 4, cv2.LINE_AA)
    core_and_aa = np.any(source != 55, axis=2)
    restored, _ = contour.apply(source.astype(np.float32))
    np.testing.assert_array_equal(restored[core_and_aa], source[core_and_aa])


def test_broad_title_and_sparse_light_chat_glyphs_keep_one_pixel_margin():
    source = np.full((960, 910, 3), 38, np.uint8)
    cv2.putText(source, "MAZE", (90, 210), cv2.FONT_HERSHEY_SIMPLEX,
                4.7, (225, 225, 225), 7, cv2.LINE_AA)
    for text, xy, gray in (("OK", (185, 480), 245), ("09:25", (650, 550), 155),
                           ("Call", (320, 720), 190), ("?", (230, 870), 220)):
        cv2.putText(source, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                    .85, (gray, gray, gray), 2, cv2.LINE_AA)
    glyph = np.any(source != 38, axis=2).astype(np.uint8)
    mask = cv2.dilate(glyph, np.ones((3, 3), np.uint8)) > 0
    restored, _ = contour.apply(source.astype(np.float32))
    np.testing.assert_array_equal(restored[mask], source[mask])


@pytest.mark.parametrize("light_on_dark", [False, True])
def test_dense_document_preserves_all_glyphs_and_one_pixel_margin(light_on_dark):
    background, ink = (40, 225) if light_on_dark else (215, 30)
    source = np.full((640, 800, 3), background, np.uint8)
    for y in range(30, 625, 28):
        cv2.putText(source, "Document 123 AB: letters and symbols!", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (ink, ink, ink), 1, cv2.LINE_AA)
    glyph = np.any(source != background, axis=2).astype(np.uint8)
    mask = cv2.dilate(glyph, np.ones((3, 3), np.uint8)) > 0
    restored, report = contour.apply(source.astype(np.float32))
    assert report["protected_text_fraction"] > .5
    np.testing.assert_array_equal(restored[mask], source[mask])


def _line_art():
    source = np.full((512, 512, 3), 238, np.uint8)
    cv2.rectangle(source, (70, 240), (450, 510), (250, 205, 60), -1)
    cv2.rectangle(source, (70, 240), (450, 510), (20, 20, 20), 4, cv2.LINE_AA)
    cv2.ellipse(source, (260, 180), (105, 150), 0, 0, 360, (240, 175, 135), -1)
    cv2.ellipse(source, (260, 180), (105, 150), 0, 0, 360, (15, 15, 15), 3, cv2.LINE_AA)
    for x in (205, 300):
        cv2.line(source, (x - 23, 172), (x + 20, 185), (10, 10, 10), 2, cv2.LINE_AA)
        cv2.ellipse(source, (x, 190), (18, 7), 0, 0, 180, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.line(source, (248, 253), (282, 257), (35, 35, 35), 1, cv2.LINE_AA)
    cv2.polylines(source, [np.array([[120, 330], [170, 355], [130, 430]], np.int32)],
                  False, (30, 30, 30), 2, cv2.LINE_AA)
    return source


def test_flat_colored_line_art_and_thin_facial_strokes_stay_exact():
    source = _line_art()
    restored, report = contour.apply(source.astype(np.float32))
    np.testing.assert_array_equal(restored, source)
    assert report["protected_line_art"] is True
    assert report["masked_fraction"] == 0


def test_sparse_illustration_is_protected_relative_to_its_ink_region():
    source = np.full((1280, 1024, 3), 250, np.uint8)
    source[700:1212, 290:802] = _line_art()
    restored, report = contour.apply(source.astype(np.float32))
    assert report["protected_line_art"] is True
    np.testing.assert_array_equal(restored, source)


def test_small_avatar_icons_and_short_chat_text_are_not_global_line_art():
    source = np.full((1024, 768, 3), 40, np.uint8)
    avatar = cv2.resize(_line_art(), (76, 76), interpolation=cv2.INTER_AREA)
    for y in (20, 250, 480, 710):
        source[y:y + 76, 20:96] = avatar
        cv2.putText(source, "OK!  19:30", (140, y + 45), cv2.FONT_HERSHEY_SIMPLEX,
                    .75, (220, 220, 220), 2, cv2.LINE_AA)
    rgb = source.astype(np.float32)
    protected, _ = contour._line_art_context(rgb, rgb @ np.array([.299, .587, .114], np.float32))
    assert protected is False


def test_repeated_comic_frames_are_protected_without_disabling_separate_plate():
    source = np.full((1024, 768, 3), 200, np.float32)
    drawn = np.zeros(source.shape[:2], np.uint8)
    rng = np.random.default_rng(81)
    for top in (20, 265):
        source[top:top + 220, 20:748] = 90 + rng.normal(0, 4, (220, 728, 3))
        cv2.rectangle(source, (20, top), (747, top + 219), (12, 12, 12), 4)
        for x in (100, 280, 460, 640):
            for y in (top + 65, top + 160):
                for target, color in ((source, (18, 18, 18)), (drawn, 255)):
                    cv2.ellipse(target, (x, y), (22, 30), 0, 20, 340, color, 2)
                    cv2.polylines(target, [np.array([[x - 10, y - 27], [x - 18, y - 38],
                                                     [x - 3, y - 30]], np.int32)], False, color, 2)
                    cv2.line(target, (x - 10, y - 6), (x - 4, y - 4), color, 2)
                    cv2.line(target, (x + 4, y - 4), (x + 10, y - 6), color, 2)
                    cv2.ellipse(target, (x, y + 8), (9, 5), 0, 0, 180, color, 2)
    plate, depth = _ellipse(.5)
    source[512:, 128:640] = plate
    restored, report = contour.apply(source)
    assert report["protected_line_art"] is False
    assert 0 < report["protected_illustration_fraction"] < .35
    ink = cv2.dilate(drawn[:490], np.ones((3, 3), np.uint8)) > 0
    np.testing.assert_array_equal(restored[:490][ink], source[:490][ink])
    assert np.max(abs(restored[512:, 128:640][abs(depth) < 6]
                      - source[512:, 128:640][abs(depth) < 6])) > 1


@pytest.mark.parametrize("grain", [0, 2])
def test_framed_photo_plate_stays_on_correction_path_for_all_consumers(grain):
    from gptimg_restore import celfill, defocus

    source = np.full((1024, 1024, 3), 200, np.float32)
    if grain:
        source += np.random.default_rng(12).normal(0, grain, source.shape).astype(np.float32)
    for top, bottom in ((30, 480), (530, 995)):
        cv2.rectangle(source, (35, top), (988, bottom), (12, 12, 12), 5)
    plate, depth = _ellipse(.5, size=384)
    source[65:449, 320:704] = plate
    restored, report = contour.apply(source)
    rim_change = np.max(abs(restored[65:449, 320:704] - plate), axis=2)[abs(depth) < 6]
    assert report["protected_line_art"] is False
    assert np.mean(rim_change) > 1
    assert np.count_nonzero(rim_change > 1) > 500
    _, defocus_report = defocus.apply(source)
    assert defocus_report["protected_line_art"] is False
    cel_output, cel_report = celfill.apply(source)
    assert cel_report["illustration_context"] is False
    np.testing.assert_array_equal(cel_output, source)


def test_dark_ceramic_rim_inside_photo_frame_is_still_corrected():
    from gptimg_restore import celfill, defocus

    source = np.full((1024, 1024, 3), 200, np.float32)
    for top, bottom in ((30, 480), (530, 995)):
        cv2.rectangle(source, (35, top), (988, bottom), (12, 12, 12), 5)
    plate, depth = _ellipse(.5, size=384)
    plate -= (100 * np.exp(-0.5 * (depth / 1.7) ** 2))[..., None]
    source[65:449, 320:704] = plate
    source += np.random.default_rng(91).normal(0, 2, source.shape).astype(np.float32)
    restored, report = contour.apply(source)
    rim_change = np.max(abs(restored[65:449, 320:704] - source[65:449, 320:704]), axis=2)[abs(depth) < 6]
    assert report["protected_line_art"] is False
    assert report["protected_illustration_fraction"] == 0
    assert np.mean(rim_change) > 1
    assert np.count_nonzero(rim_change > 1) > 500
    assert defocus.apply(source)[1]["protected_line_art"] is False
    assert celfill.apply(source)[1]["illustration_context"] is False


def test_grid_of_twelve_closed_product_rims_is_not_an_ink_network():
    source = np.full((1024, 1024, 3), 190, np.float32)
    rim = np.zeros(source.shape[:2], bool)
    y, x = np.mgrid[:160, :160].astype(np.float32)
    angle = np.arctan2((y - 80) / .65, x - 80)
    radius = np.sqrt((x - 80) ** 2 + ((y - 80) / .65) ** 2)
    depth = radius - (54.4 + .5 * np.sin(angle * 18))
    gray = (190 - 35 / (1 + np.exp(np.clip(depth / 1.4, -60, 60)))
            - 110 * np.exp(-.5 * (depth / 1.7) ** 2))
    plate = np.repeat(gray[..., None], 3, axis=2)
    for top in (40, 200, 360):
        for left in (70, 300, 530, 760):
            source[top:top + 160, left:left + 160] = plate
            rim[top:top + 160, left:left + 160] = abs(depth) < 6
    for top, bottom in ((20, 535), (580, 1000)):
        cv2.rectangle(source, (35, top), (988, bottom), (12, 12, 12), 5)
    source += np.random.default_rng(29).normal(0, 2, source.shape).astype(np.float32)
    restored, report = contour.apply(source)
    rim_change = np.max(abs(restored - source), axis=2)[rim]
    assert report["protected_line_art"] is False
    assert report["protected_illustration_fraction"] == 0
    assert np.mean(rim_change) > 1
    assert np.count_nonzero(rim_change > 1) > 1000


@pytest.mark.parametrize("scale", [.25, .35, .5, .75, 1, 1.5, 2])
def test_line_art_is_protected_after_ordinary_resizing(scale):
    source = cv2.resize(_line_art(), (1024, 1024))
    source = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    restored, report = contour.apply(source.astype(np.float32))
    assert report["protected_line_art"] is True
    np.testing.assert_array_equal(restored, source)


@pytest.mark.parametrize("quality", [85, 90, 95, 100])
def test_line_art_is_protected_after_jpeg_round_trip(quality):
    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(_line_art()).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    source = np.array(Image.open(buffer).convert("RGB"), np.float32)
    restored, report = contour.apply(source)
    assert report["protected_line_art"] is True
    np.testing.assert_array_equal(restored, source)


def test_black_menu_headline_keeps_plate_ripple_correction_enabled():
    plate, depth = _ellipse(.5, size=768)
    plain = np.full((1024, 768, 3), 200, np.float32)
    plain[256:] = plate
    menu = np.rint(plain).astype(np.uint8)
    cv2.putText(menu, "CAFE MENU", (90, 150), cv2.FONT_HERSHEY_SIMPLEX,
                2.5, (10, 10, 10), 5, cv2.LINE_AA)
    restored, report = contour.apply(menu.astype(np.float32))
    rim = np.abs(depth) < 6
    assert report["protected_line_art"] is False
    assert report["accepted_contours"] > 0
    assert np.max(abs(restored[256:][rim] - menu[256:][rim])) > 1


def test_distributed_dark_text_is_excluded_from_ink_network_evidence():
    source = np.full((768, 1024, 3), 210, np.uint8)
    for y in (100, 290, 480, 690):
        for x in (30, 560):
            cv2.putText(source, "CAFE MENU", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1.4, (15, 15, 15), 3, cv2.LINE_AA)
    rgb = source.astype(np.float32)
    protected, _ = contour._line_art_context(rgb, rgb @ np.array([.299, .587, .114], np.float32))
    assert protected is False


def test_text_and_photo_poster_cannot_combine_remote_flatness_and_ink_evidence():
    from gptimg_restore import celfill, defocus

    rng = np.random.default_rng(41)
    # celfill의 사진 대조군을 생성한 뒤와 같은 난수 상태 재현.
    rng.normal(145, 12, (360, 480, 3))
    source = np.full((360, 480, 3), 220, np.uint8)
    for y in range(28, 350, 24):
        cv2.putText(source, "TEXT 123 AB", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .7, (25, 25, 25), 1, cv2.LINE_AA)
    source[80:260, 290:460] = rng.normal(100, 25, (180, 170, 3))
    source = source.astype(np.float32)
    _, report = contour.apply(source)
    assert report["protected_line_art"] is False
    assert defocus.apply(source)[1]["protected_line_art"] is False
    cel_output, cel_report = celfill.apply(source)
    assert cel_report["illustration_context"] is False
    np.testing.assert_array_equal(cel_output, source)


def test_photo_like_texture_does_not_trigger_line_art_guard():
    rng = np.random.default_rng(7)
    y, x = np.mgrid[:512, :768].astype(np.float32)
    texture = 120 + 24 * np.sin(x / 4) * np.cos(y / 5) + 12 * np.sin((x + y) / 2)
    source = np.repeat(texture[..., None], 3, axis=2) + rng.normal(0, 5, (512, 768, 3))
    for xx in range(30, 760, 80):
        cv2.line(source, (xx, 0), (xx - 50, 511), (25, 25, 25), 3)
    source = source.astype(np.float32)
    protected, _ = contour._line_art_context(source, source @ np.array([.299, .587, .114], np.float32))
    assert protected is False


def test_flat_one_sided_step_is_not_mistaken_for_ink():
    y, x = np.mgrid[:256, :512].astype(np.float32)
    clean = 45 + 140 / (1 + np.exp(np.clip((y - 128) / 1.4, -60, 60)))
    noisy = 45 + 140 / (1 + np.exp(np.clip((y - 128 - .5 * np.sin(x * 2 * np.pi / 24)) / 1.4, -60, 60)))
    source = np.repeat(noisy[..., None], 3, axis=2)
    restored, report = contour.apply(source)
    rim = np.abs(y - 128) < 7
    assert report["protected_line_art"] is False
    assert np.mean((restored[..., 0][rim] - clean[rim]) ** 2) < np.mean((noisy[rim] - clean[rim]) ** 2)


def test_resizing_textured_surfaces_does_not_trigger_line_art_guard():
    rng = np.random.default_rng(42)
    source = (180 + rng.normal(0, 1.2, (1024, 1024, 3))).astype(np.float32)
    cv2.circle(source, (512, 512), 350, (25, 25, 25), 2)
    for rgb in (source, cv2.resize(source, (2048, 2048))):
        protected, _ = contour._line_art_context(rgb, rgb @ np.array([.299, .587, .114], np.float32))
        assert protected is False


def test_zero_strength_and_presets():
    source, _ = _ellipse(0.5)
    restored, report = contour.apply(source, strength=0)
    np.testing.assert_array_equal(restored, source)
    assert report["enabled"] is False
    assert PRESET_STRENGTHS["safe"]["contour"] == 0
    assert PRESET_STRENGTHS["balanced"]["contour"] == 0
    assert PRESET_STRENGTHS["strong"]["contour"] == 1


@pytest.mark.parametrize("strength", [-1, float("nan"), float("inf")])
def test_invalid_strength(strength):
    with pytest.raises(ValueError):
        contour.apply(np.zeros((8, 8, 3), np.float32), strength=strength)
