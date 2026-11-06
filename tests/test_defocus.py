import cv2
import numpy as np
import pytest
from PIL import Image

from gptimg_restore import defocus
from gptimg_restore.cli import PRESET_STRENGTHS, STAGE_ORDER, _parser, main, resolve_strengths


def _basin_field(size=256):
    """@description 부드러운 잎 모양과 반복되는 어두운 골을 가진 합성 비초점 면."""
    y, x = np.mgrid[:size, :size].astype(np.float32)
    gray = (80 + 5 * np.sin(x / 13) * np.cos(y / 17)
            - 8 * (0.5 + 0.5 * np.cos(x * 2 * np.pi / 11))
            * (0.5 + 0.5 * np.cos(y * 2 * np.pi / 11)))
    return np.repeat(gray[..., None], 3, axis=2)


def _dark_rms(rgb):
    """@description 고정된 골 크기 대역에서 어두운 잔차의 RMS 계산."""
    y = rgb @ defocus._LUMA
    band = cv2.GaussianBlur(y, (0, 0), 2.6) - cv2.GaussianBlur(y, (0, 0), 0.8)
    return np.sqrt(np.mean(np.maximum(band, 0) ** 2))


def test_defocused_dark_basins_become_shallower_without_color_shift():
    source = _basin_field()
    restored, report = defocus.apply(source)
    assert _dark_rms(restored) < 0.85 * _dark_rms(source)
    assert restored.dtype == np.float32
    assert report["eligible_fraction"] > 0.5
    assert report["max_abs_change"] > 1
    assert np.all(restored >= source)
    np.testing.assert_array_equal(restored[..., 0], restored[..., 1])
    np.testing.assert_array_equal(restored[..., 1], restored[..., 2])


@pytest.mark.parametrize("value", [0, 80, 100.125, 255])
def test_uniform_float_and_saved_pixels_stay_exact(value):
    source = np.full((128, 160, 3), value, np.float32)
    restored, report = defocus.apply(source)
    np.testing.assert_array_equal(restored, source)
    np.testing.assert_array_equal(np.rint(restored).astype(np.uint8),
                                  np.rint(source).astype(np.uint8))
    assert report["max_abs_change"] == 0


def test_zero_strength_keeps_every_float_sample_and_returns_a_copy():
    source = np.random.default_rng(91).uniform(0, 255, (61, 53, 3)).astype(np.float32)
    restored, report = defocus.apply(source, 0)
    np.testing.assert_array_equal(restored, source)
    assert not np.shares_memory(restored, source)
    assert report["enabled"] is False


def test_focused_fine_texture_stays_exact():
    source = np.full((256, 256, 3), 80, np.float32)
    source[::2] = 130
    source[:, ::3] = 30
    restored, _ = defocus.apply(source)
    np.testing.assert_array_equal(restored, source)


def test_skin_and_eye_like_disc_stay_exact_without_external_masks():
    source = np.full((256, 320, 3), (155, 110, 80), np.float32)
    cv2.ellipse(source, (160, 128), (45, 22), 0, 0, 360, (190, 190, 190), -1)
    cv2.circle(source, (160, 128), 17, (45, 45, 45), -1)
    cv2.circle(source, (160, 128), 8, (12, 12, 12), -1)
    cv2.circle(source, (155, 122), 3, (220, 220, 220), -1)
    restored, _ = defocus.apply(source)
    np.testing.assert_array_equal(restored, source)


def test_stitches_and_text_stay_exact():
    source = np.full((256, 384, 3), 180, np.uint8)
    for x in range(16, 370, 11):
        cv2.line(source, (x, 195), (x + 4, 188), (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(source, "FIELD TEXT", (18, 95), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (35, 35, 35), 2, cv2.LINE_AA)
    source = source.astype(np.float32)
    restored, _ = defocus.apply(source)
    np.testing.assert_array_equal(restored, source)


def test_small_eligible_island_is_rejected():
    source = np.full((168, 168, 3), (150, 110, 80), np.float32)
    field = _basin_field(40)
    source[64:104, 64:104] = field
    _, weight, rejected = defocus._field_weight(source)
    assert rejected > 0
    assert not np.any(weight)
    restored, _ = defocus.apply(source)
    np.testing.assert_array_equal(restored, source)


def test_illustration_context_skips_eligible_basin_field(monkeypatch):
    source = _basin_field()
    monkeypatch.setattr(defocus, "_line_art_context", lambda rgb, luma: (True, 0.75))
    restored, report = defocus.apply(source)
    np.testing.assert_array_equal(restored, source)
    assert report["protected_line_art"] is True
    assert report["eligible_fraction"] == 0


def test_sharp_dark_object_and_highlight_keep_their_boundary_on_blurred_texture():
    source = _basin_field(320)
    cv2.circle(source, (160, 160), 23, (12, 12, 12), -1)
    cv2.circle(source, (153, 151), 5, (220, 220, 220), -1)
    y, x = np.mgrid[:320, :320]
    subject = (x - 160) ** 2 + (y - 160) ** 2 <= 25 ** 2
    restored, _ = defocus.apply(source)
    np.testing.assert_array_equal(restored[subject], source[subject])
    assert np.any(restored[~subject] != source[~subject])


@pytest.mark.parametrize("strength", [-1, np.inf, np.nan])
def test_invalid_strength_is_rejected(strength):
    with pytest.raises(ValueError, match="defocus"):
        defocus.apply(_basin_field(32), strength)


def test_preset_flag_order_and_analysis_contract():
    assert PRESET_STRENGTHS["safe"]["defocus"] == 0
    assert PRESET_STRENGTHS["balanced"]["defocus"] == 0
    assert PRESET_STRENGTHS["strong"]["defocus"] == 1
    assert STAGE_ORDER[-2:] == ("contour", "defocus")
    assert _parser().parse_args(["x.png", "--defocus", "0.4"]).defocus == 0.4
    assert resolve_strengths("strong", None, None, None, None, defocus=0)["defocus"] == 0
    assert defocus.analyze(_basin_field(64))["method"] == "defocused_negative_band"


@pytest.mark.parametrize("preset", ["safe", "balanced"])
def test_safe_and_balanced_outputs_match_explicit_disabled_stage(tmp_path, preset):
    source = tmp_path / "field.png"
    Image.fromarray(np.uint8(np.rint(_basin_field(128)))).save(source)
    default = tmp_path / "default"
    disabled = tmp_path / "disabled"
    assert main([str(source), "-o", str(default), "--preset", preset]) == 0
    assert main([str(source), "-o", str(disabled), "--preset", preset, "--defocus", "0"]) == 0
    assert (default / "field_restored.png").read_bytes() == (disabled / "field_restored.png").read_bytes()
