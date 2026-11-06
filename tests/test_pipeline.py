import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

from gptimg_restore.cli import PRESET_STRENGTHS, main
from gptimg_restore.io_utils import load_image, save_image
from gptimg_restore.pipeline import PRESETS, STAGES, restore_array


@pytest.mark.parametrize("preset", ["safe", "balanced", "strong"])
@pytest.mark.parametrize("kind", ["alpha", "16bit"])
def test_api_matches_cli_png(tmp_path, preset, kind):
    y, x = np.mgrid[:64, :72]
    rgb = np.stack(((x * 3 + y) % 256, (y * 4) % 256, (x + y * 2) % 256), axis=-1).astype(np.float32)
    source = tmp_path / "source.png"
    if kind == "alpha":
        alpha = ((x + y) % 256).astype(np.float32)
        save_image(source, rgb, alpha)
        bit_depth = 8
    else:
        save_image(source, rgb + 0.125, None, bit_depth=16)
        bit_depth = 16
    loaded = load_image(source)
    output, _, _ = restore_array(loaded, preset=preset, bit_depth=bit_depth)
    expected = tmp_path / "api.png"
    save_image(expected, output, loaded.alpha, bit_depth=bit_depth, icc_profile=loaded.icc_profile)
    assert main([str(source), "--preset", preset, "--bit-depth", str(bit_depth), "-o", str(tmp_path / "cli")]) == 0
    assert expected.read_bytes() == (tmp_path / "cli" / "source_restored.png").read_bytes()


def test_progress_and_metadata(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32), (110, 120, 130)).save(source)
    calls = []
    restore_array(load_image(source), progress=lambda fraction, stage: calls.append((fraction, stage)))
    assert [stage for _, stage in calls] == list(STAGES)
    assert [fraction for fraction, _ in calls] == sorted(fraction for fraction, _ in calls)
    assert calls[-1][0] == 1.0
    assert PRESETS == PRESET_STRENGTHS


def test_serial_matches_threaded_strong(tmp_path):
    y, x = np.mgrid[:512, :512]
    rgb = np.full((512, 512, 3), 238, np.uint8)
    cv2.rectangle(rgb, (55, 240), (460, 511), (240, 205, 60), -1)
    cv2.rectangle(rgb, (55, 240), (460, 530), (20, 20, 20), 4, cv2.LINE_AA)
    cv2.ellipse(rgb, (260, 180), (105, 150), 0, 0, 360, (240, 175, 135), -1)
    cv2.ellipse(rgb, (260, 180), (105, 150), 0, 0, 360, (15, 15, 15), 3, cv2.LINE_AA)
    for eye_x in (205, 300):
        cv2.ellipse(rgb, (eye_x, 190), (18, 7), 0, 0, 360, (250, 250, 250), -1)
        cv2.ellipse(rgb, (eye_x, 190), (18, 7), 0, 0, 360, (10, 10, 10), 2, cv2.LINE_AA)
    wave = 2 * np.cos(2 * np.pi * (x - y) / 4 + 0.71)
    rgb = np.clip(rgb.astype(np.float32) + wave[..., None], 0, 255).astype(np.uint8)
    rgb[::31, ::29] = 245
    source = tmp_path / "pattern.png"
    Image.fromarray(rgb).save(source)
    script = Path(__file__).resolve().parents[1] / "restore_gpt_image.py"
    for name, serial in (("threaded", False), ("serial", True)):
        env = os.environ.copy()
        if serial:
            env["GPTIMG_RESTORE_THREADS"] = "1"
        else:
            env.pop("GPTIMG_RESTORE_THREADS", None)
        subprocess.run([sys.executable, str(script), str(source), "--preset", "strong", "--report", "-o", str(tmp_path / name)],
                       env=env, check=True, capture_output=True)
    assert (tmp_path / "threaded" / "pattern_restored.png").read_bytes() == (tmp_path / "serial" / "pattern_restored.png").read_bytes()
    report = json.loads((tmp_path / "serial" / "pattern_report.json").read_text(encoding="utf-8"))
    assert report["stages"]["periodic"]["status"] == "detected"
    assert report["stages"]["periodic"]["settings"]["spatial_gain"] is True
    assert report["stages"]["sparkle"]["candidate_count"] > 0
    assert report["stages"]["cel_fill"]["illustration_context"] is True
    assert report["stages"]["cel_fill"]["changed_pixels"] > 0
