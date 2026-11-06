"""선택적 문자 확장과 원본 계산의 비트 동일성 확인."""
import cv2
import numpy as np
import pytest

from gptimg_restore import glyphring, textgap
from gptimg_restore._native import load_native


@pytest.fixture
def native():
    extension = load_native("_native_text")
    if extension is None:
        pytest.skip("Text extension was not built or is disabled")
    return extension


def _page():
    y, x = np.mgrid[:256, :384].astype(np.float32)
    image = np.stack([100 + x * 0.015, 50 + y * 0.01, 130 - x * 0.01], axis=-1)
    alpha = np.zeros((256, 384), np.uint8)
    for row in range(24, 255, 26):
        cv2.putText(alpha, "TEXT 123 abc DEF", (4, row), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, 255, 2, cv2.LINE_AA)
    blend = alpha.astype(np.float32)[..., None] / 255
    image = image * (1 - blend) + np.array([245, 240, 250], np.float32) * blend
    image += (2 * np.sin((x - y) * (2 * np.pi / 7)))[..., None]
    return image


def _assert_rows(first, second):
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert a[:5] == b[:5]
        np.testing.assert_array_equal(a[5], b[5])


def test_disabled_loader(monkeypatch):
    monkeypatch.setenv("GPTIMG_RESTORE_PURE_PYTHON", "1")
    assert load_native("_native_text") is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_component_maxima(native, dtype):
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 35, (51, 73), dtype=np.int32)
    valid = rng.random(35) > 0.4
    distance = rng.uniform(0, 10, labels.shape).astype(np.float32)
    rgb = rng.uniform(-1, 255, (*labels.shape, 3)).astype(dtype)
    actual = native.glyph_maxima(labels[:, ::2], valid.view(np.uint8), distance[:, ::2], rgb[:, ::2], 35)
    labels, distance, rgb = labels[:, ::2], distance[:, ::2], rgb[:, ::2]
    selected = valid[labels]
    stroke = np.full(35, -np.inf, np.float32)
    colors = np.full((35, 3), -np.inf, dtype)
    np.maximum.at(stroke, labels[selected], distance[selected])
    for channel in range(3):
        np.maximum.at(colors[:, channel], labels[selected], rgb[selected][:, channel])
    for a, b in zip(actual, (stroke, colors)):
        np.testing.assert_array_equal(a, b)


def test_candidate_selection_at_thresholds(native):
    rng = np.random.default_rng(8)
    stats = rng.integers(0, 100, (150, 5), dtype=np.int32)
    ids = np.arange(1, 150, dtype=np.intp)
    nearby = rng.permutation(len(ids)).astype(np.intp)
    colors = rng.uniform(0, 25, (149, 3)).astype(np.float32)
    strokes = rng.uniform(1, 5, 149).astype(np.float32)
    for index in range(0, len(ids), 7):
        label, candidate = ids[index], ids[nearby]
        h = stats[label, 3]
        aligned = ((stats[candidate, 3] >= 0.6 * h) & (stats[candidate, 3] <= 1.5 * h)
                   & (np.abs(stats[candidate, 1] + stats[candidate, 3] - (stats[label, 1] + h)) <= 0.36 * h)
                   & (glyphring._maximum_channel(np.abs(colors[nearby] - colors[index])) < 22)
                   & (strokes[nearby] >= 0.6 * strokes[index])
                   & (strokes[nearby] <= 1.65 * strokes[index]))
        np.testing.assert_array_equal(native.aligned_members(
            stats, ids, nearby, colors, strokes, index, 0.6 * strokes[index], 1.65 * strokes[index]),
                                      candidate[aligned])


@pytest.mark.parametrize("shape", [(1, 41), (53, 1), (63, 89), (257, 281)])
def test_background_model_exact(native, monkeypatch, shape):
    rng = np.random.default_rng(9)
    rgb = rng.uniform(-2, 258, (*shape, 3)).astype(np.float32)
    samples = (rng.random(shape) > 0.25).astype(np.float32)
    monkeypatch.setattr(textgap, "_native", None)
    expected = textgap._background_model(rgb, samples)
    actual = native.background_model(rgb, samples)
    for first, second in zip(actual, expected):
        np.testing.assert_array_equal(first, second)


def test_shape_diversity_exact(native, monkeypatch):
    rng = np.random.default_rng(10)
    for count in (2, 4, 20):
        shapes = rng.random((count, 16, 12), dtype=np.float32)
        for repeated in (False, True):
            if repeated:
                shapes[1:] = shapes[0]
            monkeypatch.setattr(glyphring, "_native", native)
            actual = glyphring._has_diverse_glyph_shapes(shapes)
            monkeypatch.setattr(glyphring, "_native", None)
            assert actual == glyphring._has_diverse_glyph_shapes(shapes)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_shape_diversity_threshold_rounding(native, monkeypatch, dtype):
    boundary = dtype(0.08)
    for offset in (np.nextafter(boundary, dtype(0)), boundary,
                   np.nextafter(boundary, dtype(1))):
        shapes = np.zeros((4, 16, 12), dtype)
        shapes[1], shapes[2], shapes[3] = offset, 0.4, 0.8
        monkeypatch.setattr(glyphring, "_native", native)
        actual = glyphring._has_diverse_glyph_shapes(shapes)
        monkeypatch.setattr(glyphring, "_native", None)
        assert actual == glyphring._has_diverse_glyph_shapes(shapes)


def test_text_paths_exact(native, monkeypatch):
    image = _page()
    difference = glyphring._maximum_channel(np.abs(image - cv2.medianBlur(
        np.clip(np.rint(image), 0, 255).astype(np.uint8), 31)))
    actual_region = native.text_region(image, difference)
    actual_rows = glyphring._text_rows(image)
    actual_output, actual_report = textgap.apply(image)
    actual_mask = np.zeros(image.shape[:2], bool)
    actual_document, actual_document_report = textgap.restore_document_background(
        image, 1.2, document_mask=actual_mask)
    assert np.any(actual_region)
    assert actual_rows
    assert actual_document_report["accepted"]
    monkeypatch.setattr(textgap, "_native", None)
    monkeypatch.setattr(glyphring, "_native", None)
    np.testing.assert_array_equal(actual_region, textgap._text_region(image, difference))
    _assert_rows(actual_rows, glyphring._text_rows(image))
    expected_output, expected_report = textgap.apply(image)
    np.testing.assert_array_equal(actual_output, expected_output)
    assert actual_report == expected_report
    expected_mask = np.zeros(image.shape[:2], bool)
    expected_document, expected_document_report = textgap.restore_document_background(
        image, 1.2, document_mask=expected_mask)
    np.testing.assert_array_equal(actual_document, expected_document)
    np.testing.assert_array_equal(actual_mask, expected_mask)
    assert actual_document_report == expected_document_report
