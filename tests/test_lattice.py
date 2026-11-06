import json

import cv2
import numpy as np
import pytest

from gptimg_restore.cli import PRESET_STRENGTHS, STAGE_ORDER, main, resolve_strengths
from gptimg_restore.io_utils import load_image, save_image
from gptimg_restore.lattice import _sampled_tile_view, analyze, apply


def _irregular_cells(shape):
    """고정 주기가 없는 휘어진 세포 경계 합성."""
    field = np.random.default_rng(76).normal(size=shape).astype(np.float32)
    field = cv2.GaussianBlur(field, (0, 0), 2)
    field /= np.std(field)
    network = np.exp(-(field / 0.35) ** 2)
    return network - network.mean()


def test_sampled_lattice_tile_view_matches_sliding_windows_and_fits_32_bit():
    tile, step = 128, 96
    padded = np.arange(224 * 224 * 3, dtype=np.float32).reshape(224, 224, 3)
    expected = np.lib.stride_tricks.sliding_window_view(
        padded, (tile, tile), axis=(0, 1)
    )[::step, ::step]
    actual = _sampled_tile_view(padded, tile, step)
    assert np.array_equal(actual, expected)

    padded_size = 4096 + 2 * (tile // 2) + (-4096) % step
    large_padded = np.broadcast_to(np.zeros((1, 1, 3), np.float32),
                                   (padded_size, padded_size, 3))
    large_view = _sampled_tile_view(large_padded, tile, step)
    assert large_view.nbytes < 2**31


@pytest.mark.parametrize("polarity", [-1, 1])
@pytest.mark.parametrize("gradient", [False, True])
def test_signed_lattice_reduced_on_flat_and_gradient(polarity, gradient):
    y, x = np.mgrid[:192, :240].astype(np.float32)
    clean = np.zeros((192, 240, 3), np.float32) + [40, 90, 110]
    if gradient:
        clean += (0.04 * x + 0.03 * y)[..., None]
    network = _irregular_cells(x.shape)
    image = (clean + polarity * 1.6 * network[..., None]).astype(np.float32)
    original = image.copy()
    output, report = apply(image)
    selected = np.s_[20:-20, 20:-20]
    assert np.std((output - clean)[selected]) < np.std((image - clean)[selected]) * 0.4
    correction = output - image
    assert np.any(correction < -0.1)
    assert np.any(correction > 0.1)
    assert output.dtype == np.float32
    assert np.array_equal(original, image)
    assert report["eligible_fraction"] > 0.6
    assert np.max(np.abs(np.mean((output - image)[selected], axis=(0, 1)))) < 0.01
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("color", [[202, 151, 130], [95, 62, 45], [98, 52, 36], [240, 222, 214], [240, 238, 235], [35, 25, 20], [14, 10, 8]])
def test_skin_like_pores_unchanged(color):
    y, x = np.mgrid[:144, :192].astype(np.float32)
    pores = 1.2 * np.sin(x * 0.8) * np.cos(y * 0.9)
    image = (np.asarray(color) + pores[..., None]).astype(np.float32)
    output, report = apply(image)
    assert np.array_equal(output, image)
    assert report["skin_protected_fraction"] == 1


@pytest.mark.parametrize("texture", ["fabric", "fur"])
def test_genuine_fine_texture_unchanged(texture):
    y, x = np.mgrid[:192, :240].astype(np.float32)
    if texture == "fabric":
        pattern = 7 * np.sin(x * 2 * np.pi / 5) + 5 * np.cos(y * 2 * np.pi / 4)
    else:
        pattern = 9 * np.sin((x - y * 0.4) * 2 * np.pi / 6)
    image = np.repeat((100 + pattern)[..., None], 3, axis=2)
    output, _ = apply(image)
    assert np.array_equal(output, image)


@pytest.mark.parametrize("light_text", [False, True])
def test_text_strokes_antialiasing_and_counters_unchanged(light_text):
    y, x = np.mgrid[:192, :320].astype(np.float32)
    background = 35 if light_text else 170
    text = 230 if light_text else 20
    image = np.repeat((background + 0.8 * np.sin(x / 1.7) * np.cos(y / 2))[..., None], 3, axis=2)
    glyph = np.zeros((192, 320), np.uint8)
    cv2.putText(glyph, "B8 TEXT", (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 1.35, 255, 3, cv2.LINE_AA)
    alpha = glyph[..., None] / 255
    image = (image * (1 - alpha) + text * alpha).astype(np.float32)
    output, _ = apply(image)
    assert np.array_equal(output[glyph > 0], image[glyph > 0])
    edge_neighborhood = cv2.dilate((glyph > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert np.array_equal(output[edge_neighborhood], image[edge_neighborhood])


def test_clean_soft_gradient_and_edge_do_not_gain_halos():
    y, x = np.mgrid[:192, :240].astype(np.float32)
    image = np.repeat((75 + x * 0.1 + y * 0.03)[..., None], 3, axis=2)
    image[:, 120:124] = 0
    image[:, 124:] += 30
    output, _ = apply(image)
    assert np.max(np.abs(image - output)) < 1e-4


def test_neutral_light_plate_surface_is_eligible():
    y, x = np.mgrid[:144, :192].astype(np.float32)
    image = np.repeat((220 + _irregular_cells(x.shape))[..., None], 3, axis=2)
    output, report = apply(image)
    assert np.std(output[20:-20, 20:-20]) < np.std(image[20:-20, 20:-20]) * 0.4
    assert report["skin_protected_fraction"] == 0


@pytest.mark.parametrize("contrast", [3, 4, 6])
@pytest.mark.parametrize("polarity", [-1, 1])
@pytest.mark.parametrize("lattice_background", [False, True])
def test_faint_text_strokes_and_aa_unchanged(contrast, polarity, lattice_background):
    glyph = np.zeros((192, 320), np.uint8)
    cv2.putText(glyph, "TEXT", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 255, 2, cv2.LINE_AA)
    background = 100 + 0.6 * _irregular_cells(glyph.shape) if lattice_background else 100
    alpha = glyph.astype(np.float32) / 255
    gray = background * (1 - alpha) + (100 + polarity * contrast) * alpha
    image = np.repeat(gray[..., None], 3, axis=2)
    output, _ = apply(image)
    protected = cv2.dilate((glyph > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert np.array_equal(output[protected], image[protected])


@pytest.mark.parametrize("amplitude", [1, 2])
@pytest.mark.parametrize("chroma", [False, True])
def test_subtle_periodic_fabric_unchanged(amplitude, chroma):
    y, x = np.mgrid[:192, :240].astype(np.float32)
    weave = amplitude * (np.sin(x * 2 * np.pi / 5) + np.cos(y * 2 * np.pi / 4))
    color = np.array([1, -0.299 / 0.587, 0] if chroma else [1, 1, 1], np.float32)
    image = 100 + weave[..., None] * color
    output, _ = apply(image)
    assert np.array_equal(output, image)


def test_zero_constant_and_analysis():
    image = np.random.default_rng(11).uniform(0, 255, (40, 50, 3)).astype(np.float32)
    output, report = apply(image, 0)
    assert np.array_equal(image, output)
    assert output is not image
    assert report["mean_abs_change"] == [0.0] * 3
    assert analyze(image)["status"] == "insufficient_background"
    image[:] = [40, 90, 110]
    output, _ = apply(image)
    assert np.max(np.abs(image - output)) < 1e-4
    assert analyze(image)["score"] < 1e-3
    json.dumps(analyze(image), allow_nan=False)


def test_strong_only_preset_and_explicit_override():
    assert PRESET_STRENGTHS["safe"]["lattice"] == 0
    assert PRESET_STRENGTHS["balanced"]["lattice"] == 0
    assert PRESET_STRENGTHS["strong"]["lattice"] == 1
    assert resolve_strengths("safe", None, None, None, None, lattice=0.7)["lattice"] == 0.7
    assert STAGE_ORDER.index("mottle") < STAGE_ORDER.index("lattice") < STAGE_ORDER.index("surface")


def test_cli_lattice_float_pipeline_preserves_alpha_and_16bit(tmp_path):
    y, x = np.mgrid[:96, :128].astype(np.float32)
    rgb = np.repeat((60 + 0.7 * np.sin(x / 1.8) * np.cos(y / 2))[..., None], 3, axis=2)
    alpha = x / 127 * 255
    source = tmp_path / "lattice.png"
    save_image(source, rgb, alpha, bit_depth=16)
    loaded = load_image(source)
    assert main([str(source), "-o", str(tmp_path / "out"), "--dots", "0", "--periodic", "0",
                 "--grime", "0", "--mottle", "0", "--lattice", "1", "--bit-depth", "16", "--report"]) == 0
    output = load_image(tmp_path / "out" / "lattice_restored.png")
    expected, _ = apply(loaded.rgb)
    assert np.array_equal(output.alpha, loaded.alpha)
    assert np.max(np.abs(output.rgb - expected)) <= 1 / 257
    assert np.any(np.abs(output.rgb - np.rint(output.rgb)) > 0.01)
