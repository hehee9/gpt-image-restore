"""@description 선화 채색 모델, 경계와 보호 영역의 회귀 검사."""

import cv2
import numpy as np
import pytest

from gptimg_restore import celfill


def _illustration():
    """@description 잡음과 알려진 색 기울기를 가진 외곽선 일러스트 생성."""
    clean = np.full((512, 512, 3), 238, np.uint8)
    cv2.rectangle(clean, (55, 240), (460, 511), (240, 205, 60), -1)
    cv2.rectangle(clean, (55, 240), (460, 530), (20, 20, 20), 4, cv2.LINE_AA)
    cv2.ellipse(clean, (260, 180), (105, 150), 0, 0, 360, (240, 175, 135), -1)
    cv2.ellipse(clean, (260, 180), (105, 150), 0, 0, 360, (15, 15, 15), 3, cv2.LINE_AA)
    for x in (205, 300):
        cv2.ellipse(clean, (x, 190), (18, 7), 0, 0, 360, (250, 250, 250), -1)
        cv2.ellipse(clean, (x, 190), (18, 7), 0, 0, 360, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.circle(clean, (x, 190), 3, (20, 20, 20), -1)
    cv2.line(clean, (248, 253), (282, 257), (35, 35, 35), 1, cv2.LINE_AA)
    cv2.polylines(
        clean,
        [np.array([[120, 330], [170, 355], [130, 430]], np.int32)],
        False,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.ellipse(clean, (235, 92), (23, 13), 0, 0, 360, (255, 245, 220), -1)
    clean = clean.astype(np.float32)
    y, x = np.mgrid[:512, :512].astype(np.float32)
    body = np.all(clean == (240, 205, 60), axis=2)
    clean[body] += (6 * (x[body] / 512) ** 2 + 4 * y[body] / 512)[:, None]
    core, _, _ = celfill._ink_masks(clean)
    detail = celfill._protected_details(clean)
    noise = 1.4 * np.sin((x + 0.84 * y) * 2 * np.pi / 7) + 1.6 * np.sin(
        (x - 0.9 * y) * 2 * np.pi / 26
    )
    source = clean + noise[..., None]
    source[core | detail] = clean[core | detail]
    return clean, source, body


@pytest.fixture(scope="module")
def restored_illustration():
    clean, source, body = _illustration()
    mask = np.ones(source.shape[:2], bool)
    output, report = celfill.apply(source, context_rgb=clean, corrected_mask=mask)
    return clean, source, body, output, report, mask


def test_flat_fills_and_border_noise_are_reconstructed(restored_illustration):
    clean, source, body, output, report, _ = restored_illustration
    interior = cv2.erode(body.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    border = interior & (np.indices(body.shape)[0] >= 495)
    assert report["illustration_context"] is True
    for region in (interior, border):
        before = np.sqrt(np.mean((source[region] - clean[region]) ** 2))
        after = np.sqrt(np.mean((output[region] - clean[region]) ** 2))
        assert after < before * 0.3
    assert report["changed_pixels"] > 10000
    assert output.dtype == np.float32


def test_ink_highlights_eyes_and_mask_are_exact(restored_illustration):
    _, source, _, output, report, mask = restored_illustration
    core, _, _ = celfill._ink_masks(source)
    detail = celfill._protected_details(source)
    np.testing.assert_array_equal(output[core | detail], source[core | detail])
    np.testing.assert_array_equal(mask, np.any(output != source, axis=2))
    assert not mask[core | detail].any()
    assert report["ink_core_changed_pixels"] == 0
    assert report["detail_changed_pixels"] == 0
    assert report["core_changed"] == 0


@pytest.mark.parametrize("strength", [0, -1, float("nan"), float("inf")])
def test_strength_contract(strength):
    source = np.full((24, 32, 3), 120.25, np.float32)
    mask = np.ones(source.shape[:2], bool)
    if strength == 0:
        output, report = celfill.apply(source, strength, corrected_mask=mask)
        np.testing.assert_array_equal(output, source)
        assert not mask.any()
        assert report["enabled"] is False
    else:
        with pytest.raises(ValueError):
            celfill.apply(source, strength, corrected_mask=mask)


@pytest.mark.parametrize("kind", ["photo", "document", "poster"])
def test_nonillustration_is_byte_exact_and_clears_mask(kind):
    rng = np.random.default_rng(41)
    source = rng.normal(145, 12, (360, 480, 3)).astype(np.float32)
    if kind != "photo":
        source = np.full((360, 480, 3), 220, np.uint8)
        for y in range(28, 350, 24):
            cv2.putText(
                source,
                "TEXT 123 AB",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (25, 25, 25),
                1,
                cv2.LINE_AA,
            )
        if kind == "poster":
            source[80:260, 290:460] = rng.normal(100, 25, (180, 170, 3))
        source = source.astype(np.float32)
    mask = np.ones(source.shape[:2], bool)
    output, report = celfill.apply(source, corrected_mask=mask)
    assert report["illustration_context"] is False
    np.testing.assert_array_equal(output, source)
    assert not mask.any()


def test_original_context_gates_current_buffer(monkeypatch):
    source = np.full((32, 48, 3), 137.25, np.float32)
    context = np.full_like(source, 42)
    seen = []

    def _context(rgb, luma):
        seen.append(rgb)
        return False, 0.2

    monkeypatch.setattr(celfill, "_line_art_context", _context)
    output, _ = celfill.apply(source, context_rgb=context)
    np.testing.assert_array_equal(seen[0], context)
    np.testing.assert_array_equal(output, source)


def test_fractional_strength_and_original_geometry_contract(monkeypatch):
    source = np.full((12, 16, 3), 120.25, np.float32)
    context = np.full_like(source, 110)
    core = np.zeros(source.shape[:2], bool)
    detail = np.zeros_like(core)
    core[0, 0] = True
    detail[0, 1] = True
    seen = []

    def _restore(rgb, geometry_rgb=None):
        seen.append(geometry_rgb)
        masks = {
            "core": core,
            "detail": detail,
            "accepted": np.ones_like(core),
            "aa": np.zeros_like(core),
        }
        return rgb + 4, {"regions": []}, masks

    monkeypatch.setattr(celfill, "_line_art_context", lambda rgb, luma: (True, 1))
    monkeypatch.setattr(celfill, "_restore", _restore)
    mask = np.ones_like(core)
    output, report = celfill.apply(
        source, 0.5, context_rgb=context, corrected_mask=mask
    )
    np.testing.assert_array_equal(seen[0], context)
    np.testing.assert_array_equal(
        output[~(core | detail)], source[~(core | detail)] + 2
    )
    np.testing.assert_array_equal(output[core | detail], source[core | detail])
    np.testing.assert_array_equal(mask, ~(core | detail))
    assert report["strength"] == 0.5


def test_empty_eligible_regions_are_noop(monkeypatch):
    monkeypatch.setattr(celfill, "_line_art_context", lambda rgb, luma: (True, 1.0))
    source = np.full((64, 64, 3), 20, np.float32)
    mask = np.ones(source.shape[:2], bool)
    output, report = celfill.apply(source, corrected_mask=mask)
    np.testing.assert_array_equal(output, source)
    assert report["regions"] == []
    assert not mask.any()


def test_quadratic_trend_is_retained_instead_of_flattened():
    y, x = np.mgrid[:96, :144].astype(np.float32)
    trend = 160 + 22 * (x / 144) ** 2 + 10 * (y / 96) ** 2
    noise = 1.5 * np.sin((x + y) * 2 * np.pi / 8)
    source = np.repeat((trend + noise)[..., None], 3, axis=2)
    fit = celfill._fit_regions(
        source, np.zeros(trend.shape, bool), np.zeros(trend.shape, bool)
    )
    assert any(region["degree"] >= 2 for region in fit.regions)
    assert (
        np.sqrt(np.mean((fit.values[fit.selected, 0] - trend[fit.selected]) ** 2)) < 0.3
    )
    assert np.ptp(fit.values[fit.selected, 0]) > 25


def test_narrow_and_small_fills_are_fitted():
    source = np.full((100, 120, 3), 230, np.float32)
    core = np.zeros(source.shape[:2], bool)
    core[:, 40:44] = True
    core[:, 54:58] = True
    core[40:44] = True
    core[59:63] = True
    source[core] = 20
    source[44:59, 44:54] = 155
    fit = celfill._fit_regions(source, core, np.zeros_like(core))
    assert fit.selected[48:55, 47:51].all()
    assert any(region["area"] < 200 for region in fit.regions)


def test_shading_boundary_remains_monotone_and_soft():
    y, x = np.mgrid[:120, :180].astype(np.float32)
    depth = x - 80 - 0.25 * y
    coverage = 0.5 + 0.5 * np.tanh(depth / 2)
    source = np.repeat((155 + 35 * coverage)[..., None], 3, axis=2)
    core = np.zeros(depth.shape, bool)
    fit = celfill._fit_regions(source, core, core)
    boundary = celfill._render_color_boundaries(source, fit, core)
    profile = boundary.output[60, :, 0]
    transition = (profile > 157) & (profile < 188)
    assert transition.sum() >= 4
    assert np.min(np.diff(profile[65:115])) > -0.1
    assert np.max(abs(profile - source[60, :, 0])) < 5


def test_color_boundary_coverage_does_not_reintroduce_adjacent_fill_blotch():
    y, x = np.mgrid[:200, :260].astype(np.float32)
    coverage = 0.5 + 0.5 * np.tanh((x - 130) / 0.8)
    clean = 160 + 20 * coverage
    stain = 5 * np.exp(-((x - 116) ** 2 + (y - 100) ** 2) / (2 * 4**2))
    source = np.repeat((clean + stain)[..., None], 3, axis=2)
    core = np.zeros(clean.shape, bool)
    fit = celfill._fit_regions(source, core, core)
    boundary = celfill._render_color_boundaries(source, fit, core)
    assert fit.selected[100, 116]
    assert boundary.color_mask[100, 116]
    assert abs(boundary.output[100, 116, 0] - 160) < 0.5
    assert np.max(abs(boundary.output[30, 125:136, 0] - clean[30, 125:136])) < 0.5


def test_unmodeled_thin_color_highlight_is_not_completed_as_neighbor_fill():
    source = np.full((120, 180, 3), (185, 47, 38), np.float32)
    cv2.line(source, (25, 15), (140, 100), (249, 173, 162), 2, cv2.LINE_AA)
    highlight = source[..., 1] > 130
    output, _, _ = celfill._restore(source)
    assert np.max(abs(output[highlight] - source[highlight])) < 15


def test_isolated_neutral_residual_is_completed_without_a_dark_notch():
    source = np.full((100, 140, 3), 193, np.float32)
    source[49:51, 69:71] = 151
    output, report, _ = celfill._restore(source)
    np.testing.assert_allclose(output[49:51, 69:71], 193, atol=0.1)
    assert report["isolated_completed_pixels"] == 4


def test_oblique_color_boundary_keeps_subpixel_crossing_smoothness():
    y, x = np.mgrid[:160, :200].astype(np.float32)
    position = 65 + 0.33 * y + 0.12 * np.sin(y * 0.9)
    coverage = 0.5 + 0.5 * np.tanh((x - position) / 1.8)
    red = np.array([185, 47, 38], np.float32)
    pink = np.array([249, 173, 162], np.float32)
    source = red + coverage[..., None] * (pink - red)
    output, _, _ = celfill._restore(source)

    def _crossings(image):
        profiles = image[12:-12, :, 1]
        return np.array([np.interp(110, row, np.arange(200)) for row in profiles])

    original_crossing = _crossings(source)
    new_crossing = _crossings(output)
    before = np.sqrt(np.mean(np.diff(original_crossing, n=2) ** 2))
    after = np.sqrt(np.mean(np.diff(new_crossing, n=2) ** 2))
    assert after <= before
    assert np.max(abs(new_crossing - original_crossing)) < 0.25


def test_curved_mouth_and_short_strokes_survive_without_safety_cap():
    source = np.full((180, 240, 3), (245, 205, 190), np.uint8)
    cv2.ellipse(source, (112, 88), (29, 18), 0, 0, 180, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.line(source, (84, 88), (140, 88), (20, 20, 20), 1, cv2.LINE_AA)
    cv2.ellipse(source, (112, 91), (24, 8), 0, 0, 180, (250, 250, 250), -1)
    cv2.line(source, (45, 130), (58, 142), (30, 30, 30), 1, cv2.LINE_AA)
    cv2.line(source, (173, 135), (191, 152), (30, 30, 30), 1, cv2.LINE_AA)
    source = source.astype(np.float32)
    dark = source.max(2) < 90
    output, _, masks = celfill._restore(source)
    assert masks["core"][dark].all()
    np.testing.assert_array_equal(output[dark], source[dark])
    teeth = source.min(2) > 248
    np.testing.assert_allclose(output[teeth], source[teeth], atol=3)


def test_broad_colored_shadow_survives_multiscale_segmentation():
    y, x = np.mgrid[:128, :2400].astype(np.float32)
    depth = x - 1200 - 0.15 * y
    trough = np.exp(-((depth / 80) ** 2))
    source = np.empty((128, 2400, 3), np.float32)
    source[:] = (185, 47, 38)
    source -= trough[..., None] * np.array([154, 44, 36], np.float32)
    output, _, _ = celfill._restore(source)
    center = abs(depth) < 6
    assert np.max(abs(output[center] - source[center])) < 8


def test_safety_cap_keeps_colored_stroke_with_alternating_edge_softness(monkeypatch):
    y, x = np.mgrid[:256, :800].astype(np.float32)
    width = 3 + 45 * np.exp(-(((y - 128) / 20) ** 2))
    center = 380 + 0.2 * y
    trough = np.exp(-(((x - center) / width) ** 2))
    source = np.empty((256, 800, 3), np.float32)
    source[:] = (185, 47, 38)
    source -= trough[..., None] * np.array([154, 44, 36], np.float32)
    monkeypatch.setattr(celfill, "_line_art_context", lambda rgb, luma: (True, 1))
    output, report = celfill.apply(source)
    dark = (abs(x - center) < 2) & (abs(y - 128) < 12)
    assert np.max(abs(output[dark] - source[dark])) < 8
    assert report["safety_trigger_pixels"] > 0


def test_change_cap_restores_entire_seeded_components_and_their_margin():
    source = np.full((60, 80, 3), 150, np.float32)
    proposed = source.copy()
    proposed[20:30, 20:30] += 16
    proposed[23:26, 23:26] += 50
    proposed[40:45, 60:65] += 16
    output, guarded, report = celfill._preserve_large_changes(source, proposed)
    np.testing.assert_array_equal(output[18:32, 18:32], source[18:32, 18:32])
    np.testing.assert_array_equal(output[40:45, 60:65], proposed[40:45, 60:65])
    assert guarded[18:32, 18:32].all()
    assert report["safety_guarded_components"] == 1
    assert report["safety_reverted_pixels"] == 100
    assert np.max(abs(output - source)) <= 24


def test_change_cap_bounds_tiny_discrepancy_without_a_reverted_island():
    source = np.full((40, 60, 3), 150, np.float32)
    proposed = source.copy()
    proposed[20, 30] += 60
    output, guarded, report = celfill._preserve_large_changes(source, proposed)
    assert not guarded.any()
    assert 23.9 < float((output - source).max()) <= 24
    assert report["safety_bounded_components"] == 1
    assert report["safety_bounded_pixels"] == 1


def test_change_cap_connects_fragments_and_fills_component_holes():
    source = np.full((60, 80, 3), 150, np.float32)
    proposed = source.copy()
    proposed[20:40, 20:40] += 16
    proposed[26:34, 26:34] = source[26:34, 26:34]
    proposed[22:24, 22:24] += 50
    proposed[20:40, 29] = source[20:40, 29]
    output, guarded, report = celfill._preserve_large_changes(source, proposed)
    assert guarded[20:40, 20:40].all()
    np.testing.assert_array_equal(output[20:40, 20:40], source[20:40, 20:40])
    assert report["safety_guarded_components"] == 1


def _faint_stroke_case(amplitude=4, width=2, curved=False, colored=True):
    """@description 대각 해칭 위의 고립된 옅은 선 생성."""
    y, x = np.mgrid[:160, :256].astype(np.float32)
    color = (185, 140, 130) if colored else (185, 185, 185)
    source = np.empty((160, 256, 3), np.float32)
    source[:] = color
    noise = 0.8 * np.sin((x + 0.84 * y) * 2 * np.pi / 7) + 0.6 * np.sin(
        (x - 0.7 * y) * 2 * np.pi / 23
    )
    source += noise[..., None]
    stroke = np.zeros((160, 256), bool)
    for column in range(60, 180):
        center = 75 + (4 * np.sin((column - 60) * np.pi / 120) if curved else 0)
        first = int(round(center)) - width // 2
        stroke[first : first + width, column] = True
    source[stroke] -= amplitude
    return source, stroke


@pytest.mark.parametrize("amplitude", [3, 4, 6, 8])
@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("curved", [False, True])
@pytest.mark.parametrize("colored", [False, True])
def test_long_faint_stroke_is_protected(amplitude, width, curved, colored):
    source, stroke = _faint_stroke_case(amplitude, width, curved, colored)
    core, _, _ = celfill._ink_masks(source)
    protected, _ = celfill._faint_strokes(source, core)
    assert protected[stroke].all()
    assert protected.sum() < 1200


@pytest.mark.parametrize("angle", range(0, 180, 15))
def test_diagonal_faint_stroke_is_protected_at_every_orientation(angle, monkeypatch):
    y, x = np.mgrid[:256, :256].astype(np.float32)
    noise = 0.8 * np.sin((x + 0.84 * y) * 2 * np.pi / 7) + 0.6 * np.sin(
        (x - 0.7 * y) * 2 * np.pi / 23
    )
    source = np.full((256, 256, 3), 170, np.float32) + noise[..., None]
    tangent = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    start = tuple(np.rint(np.array([128, 128]) - 60 * tangent).astype(int))
    stop = tuple(np.rint(np.array([128, 128]) + 60 * tangent).astype(int))
    stroke = np.zeros((256, 256), np.uint8)
    cv2.line(stroke, start, stop, 1, 1)
    stroke = stroke > 0
    source[stroke] -= 4
    core, _, _ = celfill._ink_masks(source)
    protected, _ = celfill._faint_strokes(source, core)
    assert protected[stroke].all()
    monkeypatch.setattr(celfill, "_line_art_context", lambda rgb, luma: (True, 1))
    output, _ = celfill.apply(source)
    np.testing.assert_array_equal(output[stroke], source[stroke])


@pytest.mark.parametrize("period", [5, 7, 12, 20, 28])
def test_parallel_diagonal_hatching_is_not_protected_as_faint_art(period):
    y, x = np.mgrid[:240, :320].astype(np.float32)
    phase = (x + 0.84 * y) * 2 * np.pi / period
    source = np.full((240, 320, 3), 170, np.float32)
    source += (2.5 * np.sin(phase) + 1.5 * np.sin((x - 0.84 * y) * 2 * np.pi / period))[
        ..., None
    ]
    core, _, _ = celfill._ink_masks(source)
    protected, _ = celfill._faint_strokes(source, core)
    assert not protected.any()


def test_faint_stroke_stays_exact_while_surrounding_texture_is_removed(monkeypatch):
    source, stroke = _faint_stroke_case(3, 1, True, True)
    monkeypatch.setattr(celfill, "_line_art_context", lambda rgb, luma: (True, 1))
    output, report = celfill.apply(source)
    np.testing.assert_array_equal(output[stroke], source[stroke])
    assert report["faint_stroke_pixels"] > 0
    region = np.s_[10:55, 20:235]
    assert np.std(output[region][..., 0]) < 0.3 * np.std(source[region][..., 0])
