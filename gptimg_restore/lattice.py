"""@description 피부, 옅은 획과 주기적인 재질을 보호하며 평탄 배경의 세포형 격자 완화."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import fft


def _faint_strokes(residual, backend=None):
    """격자 셀보다 긴 방향성 획과 같은 줄의 연결된 글자 보호."""
    height, width = residual.shape[:2]
    size = (max(1, (width + 1) // 2), max(1, (height + 1) // 2))
    contrast = np.empty(residual.shape[:2], dtype=residual.dtype)
    channel_contrast = np.empty_like(contrast)
    np.abs(residual[..., 0], out=contrast)
    for channel in (1, 2):
        np.abs(residual[..., channel], out=channel_contrast)
        np.maximum(contrast, channel_contrast, out=contrast)
    contrast = cv2.resize(contrast, size, interpolation=cv2.INTER_AREA)
    del channel_contrast
    candidate = (contrast > 0.35).astype(np.uint8)
    del contrast
    if backend is None:
        lines = np.zeros_like(candidate)
        for angle in range(0, 180, 15):
            direction = np.deg2rad(angle)
            dx, dy = int(round(5 * np.cos(direction))), int(round(5 * np.sin(direction)))
            kernel = np.zeros((11, 11), np.uint8)
            cv2.line(kernel, (5 - dx, 5 - dy), (5 + dx, 5 + dy), 1)
            lines |= cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    else:
        from ._gpu_lattice import _directional_lines
        lines = _directional_lines(backend, candidate)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(candidate, 8)
    anchored = np.unique(labels[lines > 0])
    eligible = ((stats[:, cv2.CC_STAT_HEIGHT] >= 6)
                & (stats[:, cv2.CC_STAT_HEIGHT] < max(7, height // 3))
                & (stats[:, cv2.CC_STAT_WIDTH] < max(7, width // 3)))
    eligible[0] = False
    anchors = anchored[eligible[anchored]]
    keep = np.zeros(count, bool)
    for label in anchors:
        left, top, w, h, _ = stats[label]
        aligned = (eligible & (np.abs(centers[:, 1] - centers[label, 1]) < h * 0.35)
                   & (stats[:, cv2.CC_STAT_HEIGHT] >= h * 0.5)
                   & (stats[:, cv2.CC_STAT_HEIGHT] <= h * 2)
                   & (stats[:, cv2.CC_STAT_LEFT] < left + w + 2 * h)
                   & (stats[:, cv2.CC_STAT_LEFT] + stats[:, cv2.CC_STAT_WIDTH] > left - 2 * h))
        keep |= aligned
    protected = keep[labels].astype(np.uint8)
    protected = cv2.resize(protected, (width, height), interpolation=cv2.INTER_NEAREST)
    return protected > 0


def _periodic_texture_weight(residual, backend=None):
    """겹치는 창의 좁은 주파수 피크로 저대비 직조, 격자와 망점 보호."""
    if backend is not None:
        from ._gpu_lattice import _periodic_weight
        return _periodic_weight(backend, residual)
    height, width = residual.shape[:2]
    tile, step = 128, 96
    padding = tile // 2
    padded = cv2.copyMakeBorder(residual, padding, padding + (-height) % step,
                                padding, padding + (-width) % step, cv2.BORDER_REFLECT_101)
    windows = _sampled_tile_view(padded, tile, step)
    window = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32)
    fy, fx = np.meshgrid(fft.fftfreq(tile), fft.rfftfreq(tile), indexing="ij")
    radius = np.hypot(fx, fy)
    band = radius >= 0.065
    concentration = np.zeros(windows.shape[:2], np.float32)
    for row, tiles in enumerate(windows):
        spectrum = fft.rfft2(tiles * window, axes=(-2, -1))
        power = np.sum(np.abs(spectrum) ** 2, axis=1) * band
        power[:, :, 1:-1] *= 2
        for column, local_power in enumerate(power):
            total = float(np.sum(local_power))
            peak = cv2.boxFilter(local_power, -1, (3, 3), normalize=False,
                                 borderType=cv2.BORDER_CONSTANT).max()
            # 균일한 색에서 생기는 float32 누적오차는 주기 무늬에서 제외.
            if total > 1e-3:
                concentration[row, column] = peak / total
    protected = np.clip((concentration - 0.22) / 0.13, 0, 1)
    protected = cv2.dilate(protected, np.ones((3, 3), np.uint8))
    expanded = cv2.resize(protected, (protected.shape[1] * step, protected.shape[0] * step),
                          interpolation=cv2.INTER_LINEAR)
    offset = step // 2
    np.subtract(1, expanded, out=expanded)
    return expanded[offset:offset + height, offset:offset + width]


def _sampled_tile_view(array, tile, step):
    """간격을 둔 위치의 타일만 가리키는 읽기 전용 뷰를 만든다."""
    rows = (array.shape[0] - tile) // step + 1
    columns = (array.shape[1] - tile) // step + 1
    return np.lib.stride_tricks.as_strided(
        array,
        shape=(rows, columns, *array.shape[2:], tile, tile),
        strides=(array.strides[0] * step, array.strides[1] * step,
                 *array.strides[2:], array.strides[0], array.strides[1]),
        writeable=False,
    )


def _estimate(rgb, backend=None):
    """색, 미세 질감과 경계 거리로 제한한 부호 있는 국소 격자 잔차 추정."""
    if backend is None:
        smooth = cv2.GaussianBlur(rgb, (21, 21), 2.5)
        residual = rgb - smooth
    else:
        from ._gpu_lattice import _initial_fields
        smooth, residual, fine_energy, edges = _initial_fields(backend, rgb)
    height, width = rgb.shape[:2]
    mask_size = (max(1, (width + 1) // 2), max(1, (height + 1) // 2))
    smooth_small = cv2.resize(smooth, mask_size, interpolation=cv2.INTER_AREA)
    del smooth

    # 행 블록 제곱 임시 배열, 픽셀별 채널 평균 유지.
    if backend is None:
        fine_energy = np.empty((height, width), dtype=residual.dtype)
        for top in range(0, height, 128):
            bottom = min(top + 128, height)
            block = residual[top:bottom]
            energy_block = fine_energy[top:bottom]
            np.sum(block * block, axis=2, out=energy_block)
            np.divide(energy_block, 3, out=energy_block)
    fine_energy = cv2.resize(fine_energy, mask_size, interpolation=cv2.INTER_AREA)
    if backend is None:
        fine_energy = cv2.GaussianBlur(fine_energy, (0, 0), 2)
        np.sqrt(fine_energy, out=fine_energy)
        texture_weight = np.clip((2.5 - fine_energy) / 1.3, 0, 1)
        # 완만한 영역 판정만 절반 크기로 계산하고 실제 잔차와 획 경계는 원해상도 유지.
        gx = cv2.Sobel(smooth_small, cv2.CV_32F, 1, 0, scale=0.0625)
        gy = cv2.Sobel(smooth_small, cv2.CV_32F, 0, 1, scale=0.0625)
        xx = cv2.GaussianBlur(np.mean(gx * gx, axis=2), (0, 0), 1.5)
        yy = cv2.GaussianBlur(np.mean(gy * gy, axis=2), (0, 0), 1.5)
        xy = cv2.GaussianBlur(np.mean(gx * gy, axis=2), (0, 0), 1.5)
        energy = np.sqrt(xx + yy)
        coherence = np.sqrt((xx - yy) ** 2 + 4 * xy * xy) / (xx + yy + 1e-6)
        structure_weight = np.clip((2.5 - energy) / 1.7, 0, 1)
        structure_weight *= 1 - np.clip((coherence - 0.65) / 0.25, 0, 1) * np.clip(
            (energy - 0.3) / 0.7, 0, 1)
        del gx, gy, xx, yy, xy
    else:
        from ._gpu_lattice import _small_fields
        fine_energy, texture_weight, structure_weight = _small_fields(backend, fine_energy, smooth_small)

    red, green, blue = smooth_small[..., 0], smooth_small[..., 1], smooth_small[..., 2]
    # 절대 밝기 대신 색 비율로 어두운 피부까지 같은 기준 적용.
    skin_color = ((red >= green * 0.94) & (green >= blue * 0.90)
                  & (green >= red * 0.50) & (blue >= red * 0.32)
                  & (red - blue > red * 0.015))
    skin = skin_color & (fine_energy < 6)
    skin = cv2.resize(skin.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    skin = cv2.dilate(skin.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    del fine_energy, skin_color, red, green, blue, smooth_small

    if backend is None:
        edges = np.zeros((height, width), bool)
        for channel in range(3):
            raw_x = cv2.Sobel(rgb[..., channel], cv2.CV_32F, 1, 0, scale=0.125)
            raw_y = cv2.Sobel(rgb[..., channel], cv2.CV_32F, 0, 1, scale=0.125)
            np.square(raw_x, out=raw_x)
            np.square(raw_y, out=raw_y)
            np.add(raw_x, raw_y, out=raw_x)
            edges |= raw_x > 16
        del raw_x, raw_y
    # 21픽셀 커널의 지지 범위 전체를 보호 띠 안에 두어 문자색 혼입 차단.
    edges |= skin
    edges |= _faint_strokes(residual, backend)
    protected = edges
    distance = cv2.distanceTransform((~protected).astype(np.uint8), cv2.DIST_L2, 5)
    texture_weight *= structure_weight
    del structure_weight
    weight = cv2.resize(texture_weight, (width, height), interpolation=cv2.INTER_LINEAR)
    distance -= 10
    distance /= 3
    np.clip(distance, 0, 1, out=distance)
    weight *= distance
    weight[protected] = 0
    # 반사 경계에서 선형 그라데이션이 꺾이는 부분은 저장값 유지.
    border_y = np.minimum(np.arange(height), np.arange(height)[::-1])
    border_x = np.minimum(np.arange(width), np.arange(width)[::-1])
    weight *= np.clip((border_y - 10) / 3, 0, 1).astype(np.float32)[:, None]
    weight *= np.clip((border_x - 10) / 3, 0, 1).astype(np.float32)[None, :]
    if np.any(weight > 0):
        weight *= _periodic_texture_weight(residual, backend)
    return residual, weight, skin


def analyze(rgb: np.ndarray, *, backend=None) -> dict:
    """피부와 경계를 제외한 배경의 격자 대역 크기와 비율 분석."""
    residual, weight, skin = _estimate(rgb, backend)
    selected = weight > 0.5
    if not np.any(selected):
        return {"status": "insufficient_background", "score": None,
                "lattice_rms_dn": None, "eligible_fraction": 0.0,
                "skin_protected_fraction": float(np.mean(skin))}
    rms = float(np.sqrt(np.mean(residual[selected] ** 2)))
    return {"status": "ok", "score": float(min(100, rms * 25)),
            "lattice_rms_dn": rms, "eligible_fraction": float(np.mean(selected)),
            "skin_protected_fraction": float(np.mean(skin))}


def apply(rgb: np.ndarray, strength: float = 1.0, *, backend=None) -> tuple[np.ndarray, dict]:
    """평탄 배경의 약 2~14픽셀 밝고 어두운 격자를 함께 감쇠.

    피부색, 옅은 획과 안티앨리어싱, 주기 무늬와 고대비 재질을 제외하고 경계에서
    보정량을 점차 줄인다. 최종 저장 전까지 반올림 없는 float32 값을 유지한다.
    """
    if strength == 0:
        return rgb.copy(), {"eligible_fraction": 0.0, "mean_abs_change": [0.0] * 3}
    residual, weight, skin = _estimate(rgb, backend)
    correction = residual * (0.85 * strength * weight)[..., None]
    return (rgb - correction).astype(np.float32), {
        "method": "signed_background_lattice", "strength": float(strength),
        "sigma": 2.5, "eligible_fraction": float(np.mean(weight > 0.1)),
        "skin_protected_fraction": float(np.mean(skin)),
        "mean_abs_change": [float(np.mean(np.abs(correction[..., c]))) for c in range(3)],
    }
