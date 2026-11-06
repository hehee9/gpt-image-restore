import json

import cv2
import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from gptimg_restore import sparkle
from gptimg_restore.io_utils import load_image, save_image


def _texture():
    rng = np.random.default_rng(50)
    noise = rng.uniform(-4, 4, (128, 128, 1)).astype(np.float32)
    return np.repeat(noise, 3, axis=2) + np.array([45, 35, 25], np.float32)


def test_textured_neutral_flecks_and_feathered_halo():
    clean = _texture()
    image = clean.copy()
    image[40:42, 45:47] += 100
    image[80:82, 76:79] += 80
    image[39:43, 44:48] += 32
    original = image.copy()
    output, report = sparkle.apply(image)
    assert np.mean(output[40:42, 45:47] - clean[40:42, 45:47]) < 12
    assert np.mean(output[80:82, 76:79] - clean[80:82, 76:79]) < 12
    assert np.mean(np.abs(output - clean)) < 0.05
    assert np.mean(output[39, 44:48] - clean[39, 44:48]) < 16
    np.testing.assert_array_equal(image, original)
    assert output.dtype == np.float32
    assert report['candidate_count'] == 2
    json.dumps(report, allow_nan=False)


def test_elongated_highlight_line_preserved():
    image = _texture()
    image[30:80, 62:64] += 100
    output, _ = sparkle.apply(image)
    np.testing.assert_array_equal(output, image)


def test_regular_rivet_grid_preserved():
    image = _texture()
    for y in range(24, 110, 20):
        for x in range(24, 110, 20):
            image[y:y + 2, x:x + 2] += 100
    output, report = sparkle.apply(image)
    np.testing.assert_array_equal(output, image)
    assert report['regular_highlights_protected'] == 25


def test_white_text_dots_on_colored_background_preserved():
    image = _texture() + np.array([40, 0, 90], np.float32)
    image[30:32, 40:42] = 255
    image[35:60, 40:42] = 255
    image[60:62, 80:82] = 255
    output, _ = sparkle.apply(image)
    np.testing.assert_array_equal(output, image)


def test_strength_zero_skips_detection(monkeypatch):
    image = _texture()
    def fail(*args):
        raise AssertionError('zero must skip detection')
    monkeypatch.setattr(sparkle, '_detect', fail)
    output, report = sparkle.apply(image, strength=0)
    np.testing.assert_array_equal(output, image)
    assert output is not image
    assert report['detection_measured'] is False


def test_half_strength_scales_correction_linearly():
    image = _texture()
    image[60:62, 60:63] += 100
    full, _ = sparkle.apply(image)
    half, _ = sparkle.apply(image, strength=0.5)
    np.testing.assert_allclose(image - half, (image - full) * 0.5, atol=8e-6)


def test_io_alpha_16_bit_and_gray_contract(tmp_path):
    gray = np.rint((_texture()[..., 0] / 255) * 65535).astype(np.uint16)
    gray[60:62, 60:62] += 25700
    source = tmp_path / 'gray16.png'
    Image.fromarray(gray).save(source)
    loaded = load_image(source)
    assert loaded.bit_depth == 16
    output, _ = sparkle.apply(loaded.rgb)
    np.testing.assert_array_equal(output[..., 0], output[..., 1])
    np.testing.assert_array_equal(output[..., 1], output[..., 2])
    alpha = np.linspace(0, 255, 128, dtype=np.float32)[None].repeat(128, axis=0)
    target = tmp_path / 'alpha16.png'
    save_image(target, output, alpha, bit_depth=16)
    result = load_image(target)
    np.testing.assert_allclose(result.alpha, alpha, atol=1 / 257)
    np.testing.assert_allclose(result.rgb, output, atol=1 / 257)


def test_round_dark_iris_catchlights_preserved():
    image = _texture() + 65
    yy, xx = np.mgrid[:128, :128]
    iris = (xx - 64) ** 2 + (yy - 64) ** 2 <= 17 ** 2
    image[iris] -= 70
    image[56:58, 58:60] += 130
    image[61:63, 68:70] += 120
    output, report = sparkle.apply(image)
    np.testing.assert_array_equal(output[iris], image[iris])
    assert report['eye_catchlights_protected'] == 2


@pytest.mark.parametrize('size', [64, 128, 256, 512])
def test_scaled_round_iris_catchlight_preserved(size):
    noise = np.random.default_rng(50).uniform(-4, 4, (size, size, 1)).astype(np.float32)
    image = np.repeat(noise, 3, axis=2) + np.array([110, 100, 90], np.float32)
    yy, xx = np.mgrid[:size, :size]
    iris = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 <= (17 * size / 128) ** 2
    image[iris] -= 70
    y, x = round(size * 0.44), round(size * 0.46)
    image[y:y + 2, x:x + 2] += 130
    output, report = sparkle.apply(image)
    np.testing.assert_array_equal(output[iris], image[iris])
    assert report['eye_catchlights_protected'] == 1


def test_small_dark_round_button_fleck_removed():
    image = _texture() + 65
    yy, xx = np.mgrid[:128, :128]
    button = (xx - 64) ** 2 + (yy - 64) ** 2 <= 4 ** 2
    image[button] -= 70
    image[63:65, 63:65] += 120
    output, report = sparkle.apply(image)
    assert np.mean(image[63:65, 63:65] - output[63:65, 63:65]) > 80
    assert report['eye_catchlights_protected'] == 0


def test_mid_gray_texture_fleck_still_removed_with_eye_veto():
    image = _texture() + 35
    image[60:62, 60:62] += 100
    output, report = sparkle.apply(image)
    assert np.mean(image[60:62, 60:62] - output[60:62, 60:62]) > 85
    assert report['eye_catchlights_protected'] == 0


def test_elongated_dark_armor_strip_fleck_still_removed():
    image = _texture() + 65
    image[15:115, 54:76] -= 70
    image[60:62, 62:64] += 120
    output, report = sparkle.apply(image)
    assert np.mean(image[60:62, 62:64] - output[60:62, 62:64]) > 100
    assert report['eye_catchlights_protected'] == 0


@pytest.mark.parametrize('thickness', [1, 2])
@pytest.mark.parametrize('green', [150, 190])
def test_dark_serif_counters_and_background_are_not_darkened(thickness, green):
    alpha = np.zeros((128, 256), np.uint8)
    for baseline in (35, 65, 95):
        cv2.putText(alpha, 'parks a e o', (12, baseline), cv2.FONT_HERSHEY_COMPLEX,
                    0.65, 255, thickness, cv2.LINE_AA)
    noise = np.random.default_rng(5).normal(0, 1.4, (128, 256, 1)).astype(np.float32)
    background = np.array([65, green, 73], np.float32) + noise
    image = background - alpha[..., None].astype(np.float32) / 255 * 60
    output, _ = sparkle.apply(image)
    counters = ndimage.binary_fill_holes(alpha >= 40) & (alpha < 25)
    band = (ndimage.distance_transform_edt(alpha < 128) < 4) & (alpha < 25)
    assert counters.sum() > 20
    for mask in (counters, band):
        assert np.min(output[mask] - background[mask]) > -6
        assert np.mean(output[mask] - background[mask]) > -1
    np.testing.assert_array_equal(output[alpha >= 25], image[alpha >= 25])


def test_textured_fleck_near_bright_surface_still_removed():
    noise = np.random.default_rng(19).uniform(-35, 35, (256, 256, 1)).astype(np.float32)
    image = np.repeat(noise, 3, axis=2) + np.array([100, 90, 80], np.float32)
    image[60:80, 65:85] = [220, 210, 200]
    image[65:67, 55:57] += 120
    output, _ = sparkle.apply(image)
    assert np.mean(image[65:67, 55:57] - output[65:67, 55:57]) > 80


@pytest.mark.parametrize('x', [129, 134, 145])
@pytest.mark.parametrize('increment', [110, 120, 130, 140, 150, 160, 170])
def test_dark_surface_fleck_beside_unrelated_bright_panel_removed(x, increment):
    noise = np.random.default_rng(50).uniform(-4, 4, (256, 256, 1)).astype(np.float32)
    image = np.repeat(noise + 80, 3, axis=2)
    image[:, :128] += 140
    image[127:129, x:x + 2] += increment
    output, _ = sparkle.apply(image)
    assert np.mean(image[127:129, x:x + 2] - output[127:129, x:x + 2]) > increment * 0.85


@pytest.mark.parametrize('groups', [1, 2, 5])
@pytest.mark.parametrize('slope', [-1, -0.3, 0, 0.1, 0.3, 1])
def test_small_light_hangul_strokes_and_detached_ticks_preserved(groups, slope):
    alpha = np.zeros((512, 256), np.uint8)
    for baseline in (180, 260, 340):
        for x in (25, 53, 90, 126, 174)[:groups]:
            y = baseline + round((x - 25) * slope)
            cv2.line(alpha, (x, y), (x + 8, y), 255, 1, cv2.LINE_AA)
            cv2.line(alpha, (x + 8, y), (x + 8, y + 10), 255, 1, cv2.LINE_AA)
            cv2.line(alpha, (x + 13, y), (x + 13, y + 10), 255, 1, cv2.LINE_AA)
            alpha[y + 4:y + 6, x + 18:x + 20] = 255
    noise = np.random.default_rng(31).uniform(-2, 2, (512, 256, 1)).astype(np.float32)
    background = noise + np.array([50, 28, 16], np.float32)
    coverage = alpha[..., None].astype(np.float32) / 255
    image = background * (1 - coverage) + np.array([240, 215, 175], np.float32) * coverage
    image[500:502, 215:217] += 120
    output, _ = sparkle.apply(image)
    np.testing.assert_array_equal(output[alpha > 0], image[alpha > 0])
    assert np.mean(image[500:502, 215:217] - output[500:502, 215:217]) > 90


def test_sparse_aligned_fabric_filaments_do_not_protect_nearby_fleck():
    noise = np.random.default_rng(82).uniform(-2, 2, (160, 256, 1)).astype(np.float32)
    image = noise + np.array([50, 35, 25], np.float32)
    for x, width in ((40, 20), (82, 6), (145, 10)):
        points = np.array([[x, 50], [x + width - 1, 63], [x + 1, 76],
                           [x + width - 2, 89]], np.int32)
        cv2.polylines(image, [points], False, (180, 165, 155), 1, cv2.LINE_8)
    image[71:73, 76:78] += 120
    output, _ = sparkle.apply(image)
    assert np.mean(image[71:73, 76:78] - output[71:73, 76:78]) > 90


@pytest.mark.parametrize('textured', [False, True])
def test_dark_specks_and_halo_removed_on_consistent_surface(textured):
    """평탄면 실측 잡음 1.3~4.8 DN에 맞춘 약한 점/후광 제거, 강한 평탄면 표식은 보존."""
    noise = np.random.default_rng(6).uniform(-2, 2, (128, 128, 1)).astype(np.float32)
    clean = np.zeros((128, 128, 3), np.float32) + [70, 80, 90]
    clean = clean.astype(np.float32) + (noise if textured else 0)
    image = clean.copy()
    image[39:43, 44:48] -= 8 if textured else 0.5
    image[40:42, 45:47] -= 35 if textured else 4
    output, report = sparkle.apply(image)
    assert np.mean(clean[40:42, 45:47] - output[40:42, 45:47]) < (5 if textured else 1)
    assert report['dark_candidate_count'] == 1
    assert report['dark_masked_pixels'] > 0
    np.testing.assert_array_equal(output[:30], image[:30])
    half, _ = sparkle.apply(image, strength=0.5)
    np.testing.assert_allclose(half - image, (output - image) * 0.5, atol=8e-6)


def test_two_dn_dark_point_on_flat_green_removed():
    image = np.zeros((128, 128, 3), np.float32) + [3, 120, 3]
    image[60, 60, 1] -= 2
    output, _ = sparkle.apply(image)
    assert output[60, 60, 1] > 119.5


def test_dark_periods_commas_and_tittles_preserved():
    alpha = np.zeros((160, 256), np.uint8)
    cv2.putText(alpha, 'i, j. ij', (20, 85), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, 255, 2, cv2.LINE_AA)
    coverage = alpha[..., None].astype(np.float32) / 255
    image = 90 * (1 - coverage) + np.zeros((160, 256, 3), np.float32) + 28 * coverage
    output, _ = sparkle.apply(image)
    np.testing.assert_array_equal(output[alpha > 0], image[alpha > 0])


@pytest.mark.parametrize('size', [64, 128, 256])
def test_dark_pupil_and_eye_details_preserved(size):
    image = np.full((size, size, 3), 190, np.float32)
    yy, xx = np.mgrid[:size, :size]
    iris = (xx - size // 2) ** 2 + (yy - size // 2) ** 2 < (size * 0.13) ** 2
    image[iris] = [65, 72, 80]
    cv2.circle(image, (size // 2, size // 2), 2, (5, 5, 5), -1)
    output, _ = sparkle.apply(image)
    np.testing.assert_array_equal(output[iris], image[iris])


def test_dark_regular_polka_dots_and_line_preserved():
    image = np.full((128, 128, 3), 75, np.float32)
    for y in range(24, 110, 20):
        for x in range(24, 110, 20):
            image[y:y + 2, x:x + 2] -= 30
    image[10:110, 12:14] -= 25
    output, report = sparkle.apply(image)
    np.testing.assert_array_equal(output, image)
    assert report['dark_regular_points_protected'] == 25


def test_skin_pores_freckles_mole_and_nostril_preserved():
    image = np.zeros((128, 128, 3), np.float32) + [160, 112, 89]
    image[40:42, 40:42] -= 20
    image[55:58, 56:59] -= 50
    image[80:82, 72:75] -= 75
    image = image.astype(np.float32)
    output, _ = sparkle.apply(image)
    np.testing.assert_array_equal(output, image)


def test_dark_path_keeps_bright_correction_bit_identical():
    image = _texture()
    image[40:42, 45:47] += 100
    image[80:82, 76:79] -= 20
    weight, surface, _ = sparkle._detect(image)
    expected = image - np.maximum(image - surface, 0) * (weight * 0.95)[..., None]
    output, _ = sparkle.apply(image)
    np.testing.assert_array_equal(output[weight > 0], expected[weight > 0])
    np.testing.assert_array_equal(output[output < image], expected[output < image])


def test_thick_dark_caption_and_price_antialiasing_preserved():
    alpha = np.zeros((180, 512), np.uint8)
    cv2.putText(alpha, 'No public API', (12, 65), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, 255, 5, cv2.LINE_AA)
    cv2.putText(alpha, '48  6,900', (40, 140), cv2.FONT_HERSHEY_SIMPLEX,
                1.8, 255, 4, cv2.LINE_AA)
    coverage = alpha[..., None].astype(np.float32) / 255
    image = np.full((180, 512, 3), 220, np.float32) * (1 - coverage) + 20 * coverage
    output, _ = sparkle.apply(image)
    np.testing.assert_array_equal(output[alpha > 0], image[alpha > 0])


def test_irregularly_spaced_dark_stitching_preserved():
    """박음질 보존과 평탄면 실측 잡음 1.3~4.8 DN 범위의 떨어진 약한 점 제거."""
    image = np.full((160, 160, 3), 200, np.float32)
    stitches = np.zeros((160, 160), bool)
    for y in (24, 35, 49, 66, 85, 102):
        x = 50 + round(y * 0.15)
        image[y:y + 5, x:x + 2] -= 30
        stitches[y:y + 5, x:x + 2] = True
    image[120:122, 120:122] -= 4
    output, report = sparkle.apply(image)
    np.testing.assert_array_equal(output[stitches], image[stitches])
    assert np.mean(output[120:122, 120:122] - image[120:122, 120:122]) > 3.5
    assert report['dark_stroke_points_protected'] == 6


def test_dark_antialiased_fragment_near_strong_stroke_preserved():
    """강한 획 옆 조각 보존과 평탄면 실측 잡음 1.3~4.8 DN 범위의 원거리 약한 점 제거."""
    image = np.full((160, 160, 3), 200, np.float32)
    image[20:120, 78:98] = 20
    image[60:62, 72:75] -= 40
    image[60:62, 30:32] -= 4
    output, _ = sparkle.apply(image)
    np.testing.assert_array_equal(output[60:62, 72:75], image[60:62, 72:75])
    assert np.mean(output[60:62, 30:32] - image[60:62, 30:32]) > 3.5


@pytest.mark.parametrize('background', [150, 230])
@pytest.mark.parametrize('contrast', [30, 70, 120])
@pytest.mark.parametrize('radius', [1, 2, 3])
@pytest.mark.parametrize('source_dtype', [np.float32, np.uint8])
@pytest.mark.parametrize('restore', [sparkle.apply, sparkle.apply_final])
def test_strong_dark_chart_markers_on_flat_surround_preserved(
        background, contrast, radius, source_dtype, restore):
    """평탄면 잡음보다 대비가 큰 표식 보존, 8비트 안티앨리어싱과 float 원형 포함."""
    image = np.full((128, 128, 3), background, source_dtype)
    color = (background - contrast,) * 3
    cv2.circle(image, (64, 64), radius, color, -1, cv2.LINE_AA)
    image = image.astype(np.float32)
    output, _ = restore(image)
    np.testing.assert_array_equal(output, image)


def test_final_pass_catches_rounding_exposed_dark_point_without_rounding_buffer():
    image = np.full((128, 128, 3), 80.51, np.float32)
    image[60, 60] = 79.49
    original = image.copy()
    early, _ = sparkle.apply(image)
    np.testing.assert_array_equal(early, image)
    output, report = sparkle.apply_final(image, bit_depth=8)
    assert np.all(np.rint(output[60, 60]) == 81)
    np.testing.assert_array_equal(output[:30], image[:30])
    np.testing.assert_array_equal(image, original)
    assert output.dtype == np.float32
    assert report['dark_candidate_count'] == 1
    assert np.any(output != np.rint(output))


def test_final_pass_uses_selected_16_bit_precision():
    image = np.full((128, 128, 3), 80.51, np.float32)
    image[60, 60] = 79.49
    output, report = sparkle.apply_final(image, bit_depth=16)
    np.testing.assert_array_equal(output, image)
    assert report['bit_depth'] == 16


def test_final_pass_reduces_both_polarities_with_existing_bright_gates():
    image = _texture() + np.array([-20, 0, 20], np.float32)
    clean = image.copy()
    image[40:42, 45:47] += 100
    image[80:82, 76:78] -= 20
    output, report = sparkle.apply_final(image)
    assert np.mean(output[40:42, 45:47] - clean[40:42, 45:47]) < 12
    assert np.mean(clean[80:82, 76:78] - output[80:82, 76:78]) < 5
    assert report['candidate_count'] == 1
    assert report['dark_candidate_count'] == 1


def test_zero_final_strength_skips_detection_and_rounding(monkeypatch):
    image = np.full((32, 32, 3), 80.51, np.float32)

    def fail(*args):
        raise AssertionError('zero must skip final detection')

    monkeypatch.setattr(sparkle, '_detect', fail)
    output, report = sparkle.apply_final(image, strength=0)
    np.testing.assert_array_equal(output, image)
    assert output is not image
    assert report['detection_measured'] is False


def test_final_pass_preserves_warm_skin_glint():
    image = _texture() + np.array([55, 35, 25], np.float32)
    image[60:62, 60:62] += 40
    output, _ = sparkle.apply_final(image)
    np.testing.assert_array_equal(output, image)


def test_final_pass_preserves_bright_fragment_next_to_retained_edge():
    noise = np.random.default_rng(82).uniform(-2, 2, (160, 160, 1)).astype(np.float32)
    image = np.repeat(noise + 80, 3, axis=2)
    image[20:120, 78:98] = 20
    image[60:62, 72:75] += 100
    image[60:62, 30:32] += 100
    output, _ = sparkle.apply_final(image)
    np.testing.assert_array_equal(output[60:62, 72:75], image[60:62, 72:75])
    assert np.mean(image[60:62, 30:32] - output[60:62, 30:32]) > 80


@pytest.mark.parametrize('base', [[45, 35, 25], [25, 35, 45]])
def test_final_protected_footprint_shortcut_matches_full_detector(base):
    image = np.random.default_rng(50).normal(0, 20, (192, 192, 1)).astype(np.float32)
    image = image + np.array(base, np.float32)
    decision = sparkle._saved_view(image, 8)
    weight, surface, _ = sparkle._detect(decision)
    background = surface @ sparkle._LUMA
    red, green, blue = surface[..., 0], surface[..., 1], surface[..., 2]
    skin = ((background >= 25) & (red - green >= 5) & (green - blue >= 2)
            & (red <= 3 * np.maximum(green, 1)) & (blue >= 0.1 * np.maximum(red, 1)))
    structure = cv2.morphologyEx(background, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) >= 16
    protected = skin | (cv2.dilate(structure.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0)
    weight[protected] = 0
    expected = image - np.maximum(image - surface, 0) * (weight * 0.95)[..., None]
    decision = sparkle._saved_view(expected, 8)
    dark_weight, target, _ = sparkle._dark_points(decision, sparkle._median_surface(decision), weight)
    expected += np.maximum(target - expected, 0) * (dark_weight * 0.95)[..., None]
    output, _ = sparkle.apply_final(image)
    np.testing.assert_array_equal(output, expected)
