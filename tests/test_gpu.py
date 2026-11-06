import builtins
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy import fft

from gptimg_restore import cli, grime
from gptimg_restore._gpu import GPUBackend, GPUUnavailableError


@pytest.fixture(scope="module")
def gpu():
    try:
        return GPUBackend()
    except GPUUnavailableError as error:
        pytest.skip(str(error))


def _cpu_band(values, radial, diagonal):
    height, width = values.shape[:2]
    padded = np.pad(values, ((48, 48 + (-height) % 48),
                            (48, 48 + (-width) % 48), (0, 0)), mode="reflect")
    correction = np.zeros_like(padded)
    mass = np.zeros(padded.shape[:2], np.float32)
    window = np.outer(np.hanning(96), np.hanning(96)).astype(np.float32)
    for top in range(0, padded.shape[0] - 95, 48):
        for left in range(0, padded.shape[1] - 95, 48):
            region = np.s_[top:top + 96, left:left + 96]
            spectrum = fft.rfft2(padded[region] * window[..., None], axes=(0, 1))
            power = np.mean(np.abs(spectrum) ** 2, axis=2)
            ratio = np.sum(power * radial * diagonal) / (np.sum(power * radial) + 1e-8)
            gain = np.clip((ratio - 0.25) / 0.35, 0, 1)
            attenuation = radial * (0.7 + (0.3 * gain).astype(np.float32) * diagonal)
            filtered = fft.irfft2(spectrum * attenuation[..., None], s=(96, 96), axes=(0, 1))
            correction[region] += filtered * window[..., None]
            mass[region] += window * window
    return correction[48:height + 48, 48:width + 48] / mass[48:height + 48, 48:width + 48, None]


def test_cpu_import_does_not_load_cupy():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; from gptimg_restore import cli, grime; "
         "assert cli._parser().parse_args(['image.png']).device == 'cpu'; "
         "assert not any(x == 'cupy' or x.startswith('cupy.') for x in sys.modules)"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_gpu_missing_cupy_is_clear_error(tmp_path, monkeypatch, capsys):
    from PIL import Image

    source = tmp_path / "input.png"
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(source)
    original_import = builtins.__import__

    def missing_cupy(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("test: CuPy unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_cupy)
    assert cli.main([str(source), "--device", "gpu", "-o", str(tmp_path / "out")]) == 1
    assert "CuPy/CUDA" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("shape", [(17, 23), (193, 167), (211, 1703)])
def test_gpu_fft_matches_cpu(gpu, shape):
    values = np.random.default_rng(42).normal(0, 4, (*shape, 3)).astype(np.float32)
    radial, diagonal, _, _ = grime._frequency_masks(96)
    expected = _cpu_band(values, radial, diagonal)
    actual = gpu.fft_band(values, radial, diagonal)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_gpu_structure_matches_cpu_and_thresholds(gpu):
    values = np.random.default_rng(21).normal(150, 2, (193, 167, 3)).astype(np.float32)
    values[35:72, 60:90] = 20
    values[100:150, 110:150] = 255
    _, protected = grime._edge_protection(values)
    expected = grime._structure_mask(values, protected)
    actual = gpu.structure_mask(values, protected)
    for cpu_values, gpu_values in zip(expected, actual):
        np.testing.assert_allclose(gpu_values, cpu_values, atol=2e-5, rtol=2e-5)
    for threshold in (0, 0.1, 0.9):
        assert np.array_equal(expected[0] > threshold, actual[0] > threshold)
    assert np.all(actual[0][protected] == 0)


def test_gpu_cli_saved_output_matches_cpu(gpu, tmp_path):
    from PIL import Image

    pixels = np.random.default_rng(3).integers(145, 156, (103, 117, 3), dtype=np.uint8)
    source = tmp_path / "input.png"
    Image.fromarray(pixels).save(source)
    options = [str(source), "--preset", "safe", "--periodic", "0", "--mottle", "0", "--report"]
    assert cli.main(options + ["-o", str(tmp_path / "cpu")]) == 0
    assert cli.main(options + ["--device", "gpu", "-o", str(tmp_path / "gpu")]) == 0
    cpu = np.asarray(Image.open(tmp_path / "cpu/input_restored.png"))
    device = np.asarray(Image.open(tmp_path / "gpu/input_restored.png"))
    assert np.abs(cpu.astype(np.int16) - device.astype(np.int16)).max() <= 1
    report = json.loads((tmp_path / "gpu/input_report.json").read_text("utf-8"))
    assert report["stages"]["grime"]["method"] == "local_oriented_band"


def test_gpu_defocus_matches_cpu(gpu):
    from gptimg_restore import defocus
    from gptimg_restore._gpu_defocus import _apply_gpu

    y, x = np.mgrid[:256, :320].astype(np.float32)
    gray = (80 + 5 * np.sin(x / 13) * np.cos(y / 17)
            - 8 * (0.5 + 0.5 * np.cos(x * 2 * np.pi / 11))
            * (0.5 + 0.5 * np.cos(y * 2 * np.pi / 11)))
    source = np.repeat(gray[..., None], 3, axis=2)
    source[40:90, 40:90] = (155, 110, 80)
    expected, report = defocus.apply(source, line_art_context=(False, 0))
    actual, weight, rejected = _apply_gpu(source, 1.0, gpu)
    _, cpu_weight, cpu_rejected = defocus._field_weight(source)
    np.testing.assert_allclose(actual, expected, atol=0.01, rtol=0)
    assert np.count_nonzero((weight > 0.1) != (cpu_weight > 0.1)) < 20
    assert rejected == cpu_rejected == report["rejected_islands"]
