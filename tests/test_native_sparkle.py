import cv2
import numpy as np
import pytest

from gptimg_restore import sparkle


@pytest.fixture
def native():
    if sparkle._native is None:
        pytest.skip('Optional sparkle extension is disabled')
    return sparkle._native


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('inverse', [False, True])
def test_component_colors_match_numpy(native, dtype, inverse):
    rng = np.random.default_rng(840)
    rgb = rng.uniform(-2, 256, (53, 61, 3)).astype(dtype)
    residual = rng.uniform(-20, 100, (53, 61)).astype(dtype)
    binary = (rng.random((53, 61)) > 0.65).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    ids = np.arange(1, len(stats), dtype=np.int64)[::2]
    colors = []
    dark_colors, contrasts = [], []
    for label in ids:
        pixels = rgb[labels == label]
        colors.append(np.max(255 - pixels if inverse else pixels, axis=0))
        dark_colors.append(np.min(pixels, axis=0))
        contrasts.append(np.max(residual[labels == label]))
    np.testing.assert_array_equal(native.component_colors(rgb, labels, stats, ids, inverse),
                                  np.asarray(colors, np.float32))
    actual = native.component_colors(rgb, labels, stats, ids, residual=residual)
    np.testing.assert_array_equal(actual[0], np.asarray(dark_colors, np.float32))
    np.testing.assert_array_equal(actual[1], np.asarray(contrasts, np.float32))


def test_eye_contour_bounds_match_opencv(native):
    rng = np.random.default_rng(462)
    for _ in range(20):
        mask = np.zeros((161, 151), np.uint8)
        for _ in range(20):
            x, y = rng.integers(0, 160, 2)
            radius = int(rng.integers(2, 60))
            cv2.circle(mask, (int(x), int(y)), radius, 1, 1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cy, cx = rng.integers(0, 151, 2)
        expected = False
        for contour in contours:
            left, top, bw, bh = cv2.boundingRect(contour)
            if (min(bw, bh) >= 12 and max(bw, bh) <= 128
                    and max(bw, bh) <= 1.7 * min(bw, bh)
                    and left > 0 and top > 0 and left + bw < 151 and top + bh < 161
                    and left < cx + 13 and left + bw > cx - 12
                    and top < cy + 13 and top + bh > cy - 12):
                expected = True
                break
        assert native.possible_eye(contours, 161, 151, cy, cx) == expected


def test_row_geometry_matches_numpy(native):
    rng = np.random.default_rng(225)
    for n in (1, 3, 17, 80):
        bx, by, bw, bh = rng.integers(0, 80, (4, n)).astype(np.float32)
        bh += 1
        paint = rng.uniform(80, 160, (n, 3)).astype(np.float32)
        cy = by + bh * 0.5
        aligned = ((np.abs(cy[:, None] - cy) <= 0.35 * np.maximum(bh[:, None], bh))
                   & (bh[:, None] >= 0.8 * bh) & (bh[:, None] <= 1.25 * bh)
                   & (np.max(np.abs(paint[:, None] - paint), axis=2) <= 24))
        np.testing.assert_array_equal(native.aligned_rows(by, bh, paint), aligned)
        for subset in (np.arange(n), np.arange(n)[::2]):
            selected = aligned[subset]
            ah = bh[subset]
            x, y = 45, 48
            dx = np.maximum(np.maximum(bx - x, x - (bx + bw)), 0)
            dy = np.maximum(np.maximum(by - y, y - (by + bh)), 0)
            distances = np.hypot(dx, dy)
            left = np.min(np.where(selected, bx, np.inf), axis=1)
            right = np.max(np.where(selected, bx + bw, -np.inf), axis=1)
            top = np.min(np.where(selected, by, np.inf), axis=1)
            bottom = np.max(np.where(selected, by + bh, -np.inf), axis=1)
            near = np.min(np.where(selected, distances, np.inf), axis=1)
            rows = ((np.sum(selected, axis=1) >= 3) & (right - left >= 2 * ah)
                    & (x >= left - ah) & (x <= right + ah)
                    & (y >= top - 2) & (y <= bottom + 2) & (near <= ah))
            actual = native.row_bounds(selected.view(np.uint8), bx, by, bw, bh, distances, ah, x, y)
            for a, b in zip(actual, (left, right, top, bottom, rows)):
                np.testing.assert_array_equal(a, b)


def test_ordered_candidate_selection_matches_python(native):
    rng = np.random.default_rng(823)
    ys, xs = rng.integers(8, 56, (2, 500), dtype=np.int64)
    expected = np.zeros((64, 64), bool)
    selected = []
    for index, (y, x) in enumerate(zip(ys, xs)):
        if not expected[y, x]:
            expected[y - 2:y + 3, x - 2:x + 3] = True
            selected.append(index)
    actual = np.zeros_like(expected)
    assert native.select_distinct(ys, xs, actual.view(np.uint8)) == selected
    np.testing.assert_array_equal(actual, expected)
    blocked = rng.integers(0, 2, (500, 13, 13), dtype=np.uint8)
    expected[:] = False
    actual[:] = False
    selected = []
    for i, (y, x) in enumerate(zip(ys, xs)):
        if not expected[y, x]:
            expected[y - 6:y + 7, x - 6:x + 7] |= blocked[i].astype(bool)
            selected.append(i)
    found = []
    for start in range(0, 500, 93):
        found.extend(native.select_bright(ys, xs, blocked[start:start + 93], actual.view(np.uint8), start))
    assert found == selected
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_dark_scatter_preserves_order_and_ties(native, dtype):
    rng = np.random.default_rng(822)
    ys, xs = rng.integers(8, 56, (2, 80), dtype=np.int64)
    feather = (rng.integers(0, 3, (80, 13, 13)) * 0.5).astype(np.float32)
    reference = rng.uniform(0, 255, (80, 3)).astype(dtype)
    expected_weight = np.zeros((64, 64), np.float32)
    expected_target = rng.uniform(0, 255, (64, 64, 3)).astype(np.float32)
    actual_weight, actual_target = expected_weight.copy(), expected_target.copy()
    for index, (y, x) in enumerate(zip(ys, xs)):
        region = np.s_[y - 6:y + 7, x - 6:x + 7]
        stronger = feather[index] > expected_weight[region]
        expected_weight[region] = np.maximum(expected_weight[region], feather[index])
        expected_target[region] = np.where(stronger[..., None], reference[index], expected_target[region])
    native.scatter_dark(ys, xs, feather, reference, actual_weight, actual_target)
    np.testing.assert_array_equal(actual_weight, expected_weight)
    np.testing.assert_array_equal(actual_target, expected_target)


@pytest.mark.parametrize('final', [False, True])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_native_pipeline_matches_python(native, monkeypatch, final, dtype):
    rng = np.random.default_rng(620)
    rgb = rng.uniform(-3, 3, (192, 320, 1)).astype(dtype)
    rgb = np.repeat(rgb, 3, axis=2) + np.array([55, 45, 35], dtype)
    for y, x in zip(*rng.integers([15, 15], [175, 303], (60, 2)).T):
        rgb[y:y + 2, x:x + 2] += 70
    text = np.zeros(rgb.shape[:2], np.uint8)
    cv2.putText(text, 'native 123', (15, 115), cv2.FONT_HERSHEY_COMPLEX,
                0.8, 255, 1, cv2.LINE_AA)
    rgb += text[..., None].astype(dtype) * 0.7
    apply = sparkle.apply_final if final else sparkle.apply
    actual, actual_report = apply(rgb)
    monkeypatch.setattr(sparkle, '_native', None)
    expected, expected_report = apply(rgb)
    np.testing.assert_array_equal(actual, expected)
    assert actual_report == expected_report
