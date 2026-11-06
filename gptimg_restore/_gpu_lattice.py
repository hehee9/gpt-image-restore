"""격자 단계의 CuPy 밀집 연산."""

import cv2
import numpy as np
from scipy import fft


def _initial_fields(backend, rgb):
    cp = backend.cp
    values = cp.asarray(rgb)
    smooth = backend._gaussian(values, 2.5)
    residual = values - smooth
    fine = cp.sum(residual * residual, axis=2) / 3
    gx, gy = backend._sobel(values, 1), backend._sobel(values, 0)
    edges = cp.any(gx * gx + gy * gy > 16, axis=2)
    return cp.asnumpy(smooth), cp.asnumpy(residual), cp.asnumpy(fine), cp.asnumpy(edges)


def _small_fields(backend, fine, smooth):
    cp = backend.cp
    fine = cp.sqrt(backend._gaussian(cp.asarray(fine), 2))
    smooth = cp.asarray(smooth)
    gx, gy = backend._sobel(smooth, 1) * 0.5, backend._sobel(smooth, 0) * 0.5
    xx = backend._gaussian(cp.mean(gx * gx, axis=2), 1.5)
    yy = backend._gaussian(cp.mean(gy * gy, axis=2), 1.5)
    xy = backend._gaussian(cp.mean(gx * gy, axis=2), 1.5)
    energy = cp.sqrt(xx + yy)
    coherence = cp.sqrt((xx - yy) ** 2 + 4 * xy * xy) / (xx + yy + 1e-6)
    texture = cp.clip((2.5 - fine) / 1.3, 0, 1)
    structure = cp.clip((2.5 - energy) / 1.7, 0, 1)
    structure *= 1 - cp.clip((coherence - 0.65) / 0.25, 0, 1) * cp.clip((energy - 0.3) / 0.7, 0, 1)
    result = cp.asnumpy(cp.stack((fine, texture, structure), axis=-1))
    return result[..., 0].copy(), result[..., 1].copy(), result[..., 2].copy()


def _directional_lines(backend, candidate):
    cp, ndi = backend.cp, backend.ndimage
    values = cp.asarray(candidate, dtype=cp.bool_)
    lines = cp.zeros_like(values)
    for angle in range(0, 180, 15):
        direction = np.deg2rad(angle)
        dx, dy = int(round(5 * np.cos(direction))), int(round(5 * np.sin(direction)))
        kernel = np.zeros((11, 11), np.uint8)
        cv2.line(kernel, (5 - dx, 5 - dy), (5 + dx, 5 + dy), 1)
        kernel = cp.asarray(kernel)
        eroded = ndi.binary_erosion(values, structure=kernel, border_value=1)
        lines |= ndi.binary_dilation(eroded, structure=kernel[::-1, ::-1], border_value=0)
    return cp.asnumpy(lines).astype(np.uint8)


def _periodic_weight(backend, residual):
    cp, ndi = backend.cp, backend.ndimage
    height, width = residual.shape[:2]
    tile, step = 128, 96
    padded = cp.pad(cp.asarray(residual), ((64, 64 + (-height) % step),
                                          (64, 64 + (-width) % step), (0, 0)), mode="reflect")
    windows = cp.lib.stride_tricks.sliding_window_view(padded, (tile, tile), axis=(0, 1))[::step, ::step]
    window = cp.asarray(np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32))
    fy, fx = np.meshgrid(fft.fftfreq(tile), fft.rfftfreq(tile), indexing="ij")
    band = cp.asarray(np.hypot(fx, fy) >= 0.065)
    concentration = cp.zeros(windows.shape[:2], cp.float32)
    for top in range(0, windows.shape[0], 4):
        for left in range(0, windows.shape[1], 16):
            batch = windows[top:top + 4, left:left + 16]
            spectrum = cp.fft.rfft2(batch * window, axes=(-2, -1))
            power = cp.sum(cp.abs(spectrum) ** 2, axis=2) * band
            power[..., 1:-1] *= 2
            total = cp.sum(power, axis=(-2, -1))
            peak = ndi.convolve(power, cp.ones((1, 1, 3, 3), cp.float32), mode="constant").max(axis=(-2, -1))
            concentration[top:top + batch.shape[0], left:left + batch.shape[1]] = cp.where(total > 1e-3, peak / cp.maximum(total, 1e-3), 0)
    protected = cp.asnumpy(cp.clip((concentration - 0.22) / 0.13, 0, 1))
    protected = cv2.dilate(protected, np.ones((3, 3), np.uint8))
    expanded = cv2.resize(protected, (protected.shape[1] * step, protected.shape[0] * step), interpolation=cv2.INTER_LINEAR)
    np.subtract(1, expanded, out=expanded)
    return expanded[48:48 + height, 48:48 + width]
