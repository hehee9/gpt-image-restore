import json
from pathlib import Path
import sys
import struct
import types
import zlib

import numpy as np
import pytest
from PIL import Image, ImageOps

from gptimg_restore import cli
from gptimg_restore.cli import main, resolve_strengths
from gptimg_restore.io_utils import load_image


def _write_rgb(path: Path, pixels: np.ndarray) -> None:
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(path)


def _write_png_16_trns(path: Path, pixels: np.ndarray, transparent, orientation: int | None = None) -> None:
    pixels = np.asarray(pixels, dtype=np.uint16)
    if pixels.ndim == 2:
        channels = 1
        color_type = 0
        transparency = struct.pack(">H", transparent)
    else:
        channels = 3
        color_type = 2
        transparency = struct.pack(">HHH", *transparent)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    height, width = pixels.shape[:2]
    raw = b"".join(b"\x00" + row.astype(">u2", copy=False).tobytes() for row in pixels)
    chunks = [chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 16, color_type, 0, 0, 0))]
    if orientation is not None:
        exif = Image.Exif()
        exif[274] = orientation
        chunks.append(chunk(b"eXIf", exif.tobytes()[6:]))
    chunks.extend((chunk(b"tRNS", transparency), chunk(b"IDAT", zlib.compress(raw)), chunk(b"IEND", b"")))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"".join(chunks))


def _install_stage_stubs(monkeypatch, calls=None, delta=0.0):
    calls = calls if calls is not None else []
    for name in ("dots", "sparkle", "periodic", "grime", "mottle", "lattice", "surface", "contour", "defocus", "celfill"):
        module = types.ModuleType(f"gptimg_restore.{name}")

        def apply(rgb, strength=1.0, *, _name=name, **options):
            calls.append((_name, strength, options.get("mode")))
            report = {"stage": _name, "strength": strength}
            if _name == "contour":
                report.update(protected_line_art=False, flat_region_fraction=0.0,
                              max_abs_change=float(abs(delta * strength)))
            return (rgb + delta * strength).astype(np.float32), report

        module.apply = apply
        if name == "sparkle":
            def apply_final(rgb, strength=1.0, bit_depth=8):
                calls.append(("sparkle_final", strength, bit_depth))
                return (rgb + delta * strength).astype(np.float32), {"stage": "sparkle_final"}

            module.apply_final = apply_final
        if name in ("periodic", "dots", "sparkle", "grime", "mottle", "lattice", "contour", "defocus", "celfill"):
            def analyze(rgb, _name=name, **options):
                if _name == "grime":
                    return {"stage": _name, "score": None, "status": "insufficient_background"}
                return {"stage": _name, "mean": float(rgb.mean()), "mode": options.get("mode")}

            module.analyze = analyze
        monkeypatch.setitem(sys.modules, f"gptimg_restore.{name}", module)
    return calls


def test_preset_strengths_and_explicit_overrides():
    assert resolve_strengths("safe", None, None, None, None) == {"dots": 0.0, "sparkle": 0.0, "periodic": 1.0, "grime": 0.5, "mottle": 0.35, "lattice": 0.0, "surface": 0.0, "contour": 0.0, "defocus": 0.0, "cel_fill": 0.0, "dither": 0.0}
    assert resolve_strengths("balanced", None, None, None, None) == {"dots": 1.0, "sparkle": 0.0, "periodic": 1.0, "grime": 1.0, "mottle": 0.7, "lattice": 0.0, "surface": 0.0, "contour": 0.0, "defocus": 0.0, "cel_fill": 0.0, "dither": 0.0}
    assert resolve_strengths("strong", None, None, None, None) == {"dots": 1.0, "sparkle": 1.0, "periodic": 1.0, "grime": 1.1, "mottle": 1.0, "lattice": 1.0, "surface": 0.0, "contour": 1.0, "defocus": 1.0, "cel_fill": 1.0, "dither": 1.0}
    assert resolve_strengths("safe", 0.25, 0.5, 0.6, 0.75, 0.8, 0.9, dither=1.25, contour=0.4, cel_fill=0.6) == {"dots": 0.25, "sparkle": 0.9, "periodic": 0.5, "grime": 0.6, "mottle": 0.8, "lattice": 0.0, "surface": 0.75, "contour": 0.4, "defocus": 0.0, "cel_fill": 0.6, "dither": 1.25}


@pytest.mark.parametrize("preset", ["safe", "balanced"])
def test_dither_is_disabled_by_safe_and_balanced_presets(tmp_path, monkeypatch, preset):
    source = tmp_path / "source.png"
    original = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    _write_rgb(source, original)
    _install_stage_stubs(monkeypatch)
    outdir = tmp_path / preset

    assert main([str(source), "--outdir", str(outdir), "--preset", preset, "--report"]) == 0

    restored = np.asarray(Image.open(outdir / "source_restored.png"))
    np.testing.assert_array_equal(restored, original)
    report = json.loads((outdir / "source_report.json").read_text(encoding="utf-8"))
    assert report["stages"]["dither"]["applied"] is False


def test_dither_flag_overrides_preset_and_uses_original_input(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    original_pixels = np.full((3, 4, 3), 80, dtype=np.uint8)
    _write_rgb(source, original_pixels)
    _install_stage_stubs(monkeypatch, delta=0.75)
    original_inputs = []
    real_apply = cli.dither.apply

    def capture_apply(rgb, original, strength, bit_depth, seed=0, document_mask=None, cel_fill_mask=None):
        original_inputs.append((rgb.copy(), original.copy(), strength, bit_depth, seed, cel_fill_mask))
        return real_apply(rgb, original, strength, bit_depth, seed, document_mask=document_mask,
                          cel_fill_mask=cel_fill_mask)

    monkeypatch.setattr(cli.dither, "apply", capture_apply)
    outdir = tmp_path / "out"

    assert main([str(source), "--outdir", str(outdir), "--preset", "balanced",
                 "--dither", "0.8", "--dots", "1", "--sparkle", "0", "--periodic", "0",
                 "--grime", "0", "--mottle", "0", "--lattice", "0", "--surface", "0",
                 "--report"]) == 0

    assert len(original_inputs) == 1
    processed, original, strength, bit_depth, seed, cel_fill_mask = original_inputs[0]
    assert strength == 0.8
    assert bit_depth == 8
    assert seed == 0
    assert cel_fill_mask is None
    np.testing.assert_array_equal(original, np.broadcast_to(original_pixels, original.shape))
    assert np.all(processed != original)
    report = json.loads((outdir / "source_report.json").read_text(encoding="utf-8"))
    assert report["stages"]["dither"]["applied"] is True


def test_strong_dither_does_not_change_16_bit_output(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    _write_rgb(source, np.full((4, 4, 3), 80, dtype=np.uint8))
    _install_stage_stubs(monkeypatch, delta=0.6)
    shared_args = ["--preset", "strong", "--dots", "1", "--sparkle", "0", "--periodic", "0",
                   "--grime", "0", "--mottle", "0", "--lattice", "0", "--surface", "0",
                   "--bit-depth", "16"]
    strong_out = tmp_path / "strong"
    disabled_out = tmp_path / "disabled"

    assert main([str(source), "--outdir", str(strong_out), *shared_args, "--report"]) == 0
    assert main([str(source), "--outdir", str(disabled_out), *shared_args, "--dither", "0"]) == 0

    strong = load_image(strong_out / "source_restored.png")
    disabled = load_image(disabled_out / "source_restored.png")
    np.testing.assert_array_equal(strong.rgb, disabled.rgb)
    report = json.loads((strong_out / "source_report.json").read_text(encoding="utf-8"))
    assert report["stages"]["dither"]["applied"] is False


def test_stage_order_and_preset_override(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    _write_rgb(source, np.full((2, 2, 3), 80, dtype=np.uint8))
    calls = _install_stage_stubs(monkeypatch)

    result = main([
        str(source), "--outdir", str(tmp_path / "out"), "--preset", "safe", "--dots", "0.4", "--grime", "0.6", "--surface", "0.75",
        "--dots-mode", "strict", "--sparkle", "0.5", "--contour", "0.7", "--cel-fill", "0.6",
    ])

    assert result == 0
    assert calls == [
        ("dots", 0.4, "strict"),
        ("sparkle", 0.5, None),
        ("periodic", 1.0, None),
        ("grime", 0.6, None),
        ("mottle", 0.35, None),
        ("surface", 0.75, None),
        ("contour", 0.7, None),
        ("sparkle_final", 0.5, 8),
        ("celfill", 0.6, None),
    ]


def test_zero_cel_fill_skips_stage_and_passes_no_mask(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    _write_rgb(source, np.full((2, 2, 3), 80, dtype=np.uint8))
    calls = _install_stage_stubs(monkeypatch)
    monkeypatch.setitem(sys.modules, "gptimg_restore.celfill", None)
    masks = []
    real_apply = cli.dither.apply

    def capture_apply(rgb, original, strength, bit_depth, seed=0, *, document_mask=None, cel_fill_mask=None):
        masks.append(cel_fill_mask)
        return real_apply(rgb, original, strength, bit_depth, seed, document_mask=document_mask,
                          cel_fill_mask=cel_fill_mask)

    monkeypatch.setattr(cli.dither, "apply", capture_apply)
    outdir = tmp_path / "out"

    assert main([str(source), "--outdir", str(outdir), "--preset", "balanced", "--cel-fill", "0", "--report"]) == 0

    assert all(call[0] != "celfill" for call in calls)
    assert masks == [None]
    report = json.loads((outdir / "source_report.json").read_text(encoding="utf-8"))
    assert report["stages"]["cel_fill"] == {"enabled": False, "strength": 0.0}
    assert report["timings"]["stages_seconds"]["cel_fill"] == 0.0


def test_cel_fill_passes_original_context_and_changed_mask_to_dither(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    original = np.full((3, 4, 3), 80, dtype=np.uint8)
    _write_rgb(source, original)
    calls = _install_stage_stubs(monkeypatch)
    celfill_module = sys.modules["gptimg_restore.celfill"]
    contexts = []

    def apply_celfill(rgb, strength=1.0, *, context_rgb=None, corrected_mask=None):
        contexts.append((context_rgb.copy(), corrected_mask.copy(), strength))
        corrected_mask[1, 2] = True
        calls.append(("celfill", strength, None))
        return rgb, {"stage": "cel_fill", "changed_pixels": int(corrected_mask.sum())}

    monkeypatch.setattr(celfill_module, "apply", apply_celfill)
    dither_masks = []
    real_apply = cli.dither.apply

    def capture_dither(rgb, source_rgb, strength, bit_depth, seed=0, *, document_mask=None, cel_fill_mask=None):
        dither_masks.append(None if cel_fill_mask is None else cel_fill_mask.copy())
        return real_apply(rgb, source_rgb, strength, bit_depth, seed, document_mask=document_mask,
                          cel_fill_mask=cel_fill_mask)

    monkeypatch.setattr(cli.dither, "apply", capture_dither)
    outdir = tmp_path / "out"

    assert main([str(source), "--outdir", str(outdir), "--preset", "balanced", "--cel-fill", "0.75", "--report"]) == 0

    assert len(contexts) == 1
    context, initial_mask, strength = contexts[0]
    np.testing.assert_array_equal(context, np.broadcast_to(original, context.shape))
    assert initial_mask.dtype == np.bool_
    assert not initial_mask.any()
    assert strength == 0.75
    assert dither_masks[0].dtype == np.bool_
    assert dither_masks[0].shape == original.shape[:2]
    assert dither_masks[0][1, 2]
    assert dither_masks[0].sum() == 1
    report = json.loads((outdir / "source_report.json").read_text(encoding="utf-8"))
    assert report["stages"]["cel_fill"]["changed_pixels"] == 1


@pytest.mark.parametrize('preset,override,enabled', [
    ('safe', None, False), ('balanced', None, False), ('strong', None, True),
    ('strong', '0', False), ('balanced', '0.5', True),
])
def test_final_sparkle_follows_existing_preset_and_override_gate(tmp_path, monkeypatch,
                                                              preset, override, enabled):
    source = tmp_path / 'source.png'
    original = np.full((2, 2, 3), 80, np.uint8)
    _write_rgb(source, original)
    calls = _install_stage_stubs(monkeypatch)
    args = [str(source), '--outdir', str(tmp_path / 'out'), '--preset', preset, '--report']
    if override is not None:
        args += ['--sparkle', override]
    assert main(args) == 0
    assert any(call[0] == 'sparkle_final' for call in calls) is enabled
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / 'out/source_restored.png')), original)
    report = json.loads((tmp_path / 'out/source_report.json').read_text(encoding='utf-8'))
    assert ('final_pass' in report['stages']['sparkle']) is enabled


@pytest.mark.parametrize('bit_depth', [8, 16])
def test_final_sparkle_runs_after_surface_on_unrounded_float(tmp_path, monkeypatch, bit_depth):
    source = tmp_path / 'source.png'
    _write_rgb(source, np.full((2, 2, 3), 80, np.uint8))
    calls = _install_stage_stubs(monkeypatch, delta=0.13)
    seen = []

    def final(rgb, strength, bit_depth):
        calls.append(('sparkle_final', strength, bit_depth))
        seen.append(rgb.copy())
        return rgb.copy(), {'stage': 'sparkle_final'}

    monkeypatch.setattr(sys.modules['gptimg_restore.sparkle'], 'apply_final', final)
    assert main([str(source), '--outdir', str(tmp_path / 'out'), '--preset', 'strong',
                 '--surface', '0.3', '--bit-depth', str(bit_depth)]) == 0
    assert [call[0] for call in calls] == [
        'dots', 'sparkle', 'periodic', 'grime', 'mottle', 'lattice', 'surface', 'contour', 'defocus', 'sparkle_final', 'celfill']
    assert calls[-2][2] == bit_depth
    np.testing.assert_allclose(seen[0], 80 + 0.13 * 8.4, atol=2e-5)
    assert np.any(seen[0] != np.rint(seen[0]))


def test_zero_strengths_skip_import_and_preserve_pixels(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    original = np.array([[[0, 100, 255], [22, 33, 44]]], dtype=np.uint8)
    _write_rgb(source, original)
    for name in ("dots", "sparkle", "periodic", "grime", "mottle", "surface"):
        monkeypatch.delitem(sys.modules, f"gptimg_restore.{name}", raising=False)

    assert main([str(source), "--outdir", str(tmp_path / "out"), "--dots", "0", "--periodic", "0", "--grime", "0", "--mottle", "0", "--surface", "0"]) == 0
    restored = np.asarray(Image.open(tmp_path / "out" / "source_restored.png"))
    np.testing.assert_array_equal(restored, original)


def test_rgba_alpha_is_reattached_unchanged(tmp_path, monkeypatch):
    source = tmp_path / "alpha.png"
    pixels = np.array([[[10, 40, 90, 0], [30, 60, 120, 127], [50, 80, 150, 255]]], dtype=np.uint8)
    Image.fromarray(pixels, mode="RGBA").save(source)
    calls = _install_stage_stubs(monkeypatch, delta=1.0)

    assert main([str(source), "--outdir", str(tmp_path / "out"), "--preset", "balanced"]) == 0
    restored = np.asarray(Image.open(tmp_path / "out" / "alpha_restored.png"))
    np.testing.assert_array_equal(restored[:, :, 3], pixels[:, :, 3])
    assert calls[0] == ("dots", 1.0, "conservative")


def test_exif_orientation_transposes_8_bit_pixels(tmp_path):
    source = tmp_path / "oriented.png"
    original = np.array([[[10, 20, 30], [40, 50, 60], [70, 80, 90]], [[100, 110, 120], [130, 140, 150], [160, 170, 180]]], dtype=np.uint8)
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(original, mode="RGB").save(source, exif=exif)

    with Image.open(source) as image:
        expected = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32)
    loaded = load_image(source)

    assert loaded.size == (2, 3)
    np.testing.assert_array_equal(loaded.rgb, expected)


def test_grayscale_icc_is_not_embedded_in_rgb_output(tmp_path):
    source = tmp_path / "gray.png"
    Image.fromarray(np.array([[0, 100], [200, 255]], dtype=np.uint8), mode="L").save(source, icc_profile=b"gray-profile")

    assert main([str(source), "--outdir", str(tmp_path / "out"), "--dots", "0", "--periodic", "0", "--grime", "0", "--mottle", "0", "--surface", "0"]) == 0
    with Image.open(tmp_path / "out" / "gray_restored.png") as restored:
        assert restored.info.get("icc_profile") is None


def test_16_bit_png_round_trip(tmp_path):
    source = tmp_path / "sixteen.png"
    original = np.array([[0, 257, 12000, 65535], [41, 4096, 50000, 60000]], dtype=np.uint16)
    Image.fromarray(original).save(source)
    outdir = tmp_path / "out"

    assert main([str(source), "--outdir", str(outdir), "--dots", "0", "--periodic", "0", "--grime", "0", "--mottle", "0", "--surface", "0", "--bit-depth", "16"]) == 0
    restored = load_image(outdir / "sixteen_restored.png")
    restored16 = np.rint(restored.rgb[:, :, 0] * 257.0).astype(np.uint16)
    np.testing.assert_array_equal(restored16, original)


def test_folder_input_uses_sibling_restored_directory(tmp_path):
    source_dir = tmp_path / "images"
    source_dir.mkdir()
    _write_rgb(source_dir / "one.png", np.full((1, 1, 3), 123, dtype=np.uint8))

    assert main([str(source_dir), "--dots", "0", "--periodic", "0", "--grime", "0", "--mottle", "0", "--surface", "0"]) == 0
    assert (tmp_path / "restored" / "one_restored.png").is_file()


def test_analyze_writes_json_without_images(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.png"
    _write_rgb(source, np.full((2, 2, 3), 50, dtype=np.uint8))
    _install_stage_stubs(monkeypatch)
    outdir = tmp_path / "out"

    assert main([str(source), "--outdir", str(outdir), "--analyze", "--report", "--dots-mode", "strict"]) == 0
    report = json.loads((outdir / "source_report.json").read_text(encoding="utf-8"))
    assert set(report["stages"]) == {"periodic", "dots", "sparkle", "grime", "mottle", "lattice", "contour", "defocus", "cel_fill"}
    assert report["size"] == {"width": 2, "height": 2}
    assert report["stages"]["dots"]["mode"] == "strict"
    assert report["stages"]["grime"]["score"] is None
    assert report["stages"]["grime"]["status"] == "insufficient_background"
    assert not (outdir / "source_restored.png").exists()
    assert set(path.name for path in outdir.iterdir()) == {"source_report.json"}
    assert json.loads(capsys.readouterr().out)[0]["stages"]["dots"]["stage"] == "dots"


def test_analyze_requires_both_analysis_modules(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.png"
    _write_rgb(source, np.full((1, 1, 3), 50, dtype=np.uint8))
    monkeypatch.setitem(sys.modules, "gptimg_restore.grime", None)

    result = main([str(source), "--outdir", str(tmp_path / "out"), "--analyze"])

    assert result == 1
    assert "grime stage module" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_save_diff_writes_amplified_visualization(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    _write_rgb(source, np.full((1, 1, 3), 100, dtype=np.uint8))
    _install_stage_stubs(monkeypatch, delta=1.0)
    outdir = tmp_path / "out"

    assert main([str(source), "--outdir", str(outdir), "--dots", "1", "--periodic", "0", "--grime", "0", "--mottle", "0", "--surface", "0", "--save-diff"]) == 0
    diff = np.asarray(Image.open(outdir / "source_diff.png"))
    np.testing.assert_array_equal(diff[0, 0], np.array([144, 144, 144], dtype=np.uint8))


def test_output_collisions_abort_before_processing(tmp_path, capsys):
    png = tmp_path / "same.png"
    jpg = tmp_path / "same.jpg"
    _write_rgb(png, np.full((1, 1, 3), 10, dtype=np.uint8))
    Image.fromarray(np.full((1, 1, 3), 20, dtype=np.uint8), mode="RGB").save(jpg)
    outdir = tmp_path / "out"

    result = main([str(png), str(jpg), "--outdir", str(outdir)])

    assert result == 1
    assert "conflicts" in capsys.readouterr().err
    assert not outdir.exists()


def test_diff_collision_with_input_preserves_all_files(tmp_path, capsys):
    source = tmp_path / "source.png"
    diff_input = tmp_path / "source_diff.png"
    _write_rgb(source, np.full((1, 1, 3), 10, dtype=np.uint8))
    _write_rgb(diff_input, np.full((1, 1, 3), 20, dtype=np.uint8))
    original_diff = diff_input.read_bytes()

    result = main([
        str(source), str(diff_input), "--outdir", str(tmp_path), "--save-diff",
        "--dots", "0", "--periodic", "0", "--grime", "0", "--mottle", "0", "--surface", "0",
    ])

    assert result == 1
    assert "source_diff.png" in capsys.readouterr().err
    assert diff_input.read_bytes() == original_diff
    assert not (tmp_path / "source_restored.png").exists()


def test_16_bit_trns_alpha_and_orientation(tmp_path):
    gray_source = tmp_path / "gray_trns.png"
    gray = np.array([[100, 12345, 200], [300, 400, 500]], dtype=np.uint16)
    _write_png_16_trns(gray_source, gray, 12345, orientation=6)
    gray_loaded = load_image(gray_source)
    expected_gray = np.rot90(gray, 3)
    expected_alpha = np.where(expected_gray == 12345, 0, 65535).astype(np.uint16)

    assert gray_loaded.size == (2, 3)
    np.testing.assert_array_equal(np.rint(gray_loaded.rgb[:, :, 0] * 257).astype(np.uint16), expected_gray)
    np.testing.assert_array_equal(np.rint(gray_loaded.alpha * 257).astype(np.uint16), expected_alpha)

    rgb_source = tmp_path / "rgb_trns.png"
    transparent_rgb = (1000, 2000, 3000)
    rgb = np.array([[[1000, 2000, 3000], [4, 5, 6]]], dtype=np.uint16)
    _write_png_16_trns(rgb_source, rgb, transparent_rgb)
    rgb_loaded = load_image(rgb_source)
    np.testing.assert_array_equal(np.rint(rgb_loaded.alpha * 257).astype(np.uint16), np.array([[0, 65535]], dtype=np.uint16))
