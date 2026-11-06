"""@description 평탄 구역의 중간 크기 얼룩 완화와 문자/재질 보호."""

from __future__ import annotations

import cv2
import numpy as np


_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _linear_box_axis(rgb, allowed, radius, axis):
    """보호 경계를 넘지 않는 1차원 국소 직선 적합, 띠별 누적합 사용."""
    if radius == 1:
        output = rgb.copy()
        average = cv2.blur(rgb, (3, 1) if axis == 1 else (1, 3))
        if axis == 1:
            interior = allowed[:, 1:-1] & allowed[:, :-2] & allowed[:, 2:]
            np.copyto(output[:, 1:-1], average[:, 1:-1], where=interior[..., None])
        else:
            interior = allowed[1:-1] & allowed[:-2] & allowed[2:]
            np.copyto(output[1:-1], average[1:-1], where=interior[..., None])
        return output
    pixels = rgb if axis == 1 else rgb.transpose(1, 0, 2)
    mask = allowed if axis == 1 else allowed.T
    blurred = cv2.blur(rgb, (2 * radius + 1, 1) if axis == 1 else (1, 2 * radius + 1))
    output = blurred if axis == 1 else blurred.transpose(1, 0, 2)
    width = pixels.shape[1]
    x = np.arange(width, dtype=np.int32)[None, :]
    sums = moments = None
    for top in range(0, pixels.shape[0], 48):
        block = pixels[top:top + 48]
        selected = mask[top:top + 48]
        starts = np.maximum.accumulate(np.where(selected, 0, x + 1), axis=1)
        stops = np.minimum.accumulate(np.where(selected, width, x)[:, ::-1], axis=1)[:, ::-1]
        left = np.maximum(x - radius, starts)
        right = np.minimum(x + radius + 1, stops)
        partial = selected & ((left != x - radius) | (right != x + radius + 1))
        rows, columns = np.nonzero(partial)
        if not len(rows):
            np.copyto(output[top:top + 48], block, where=~selected[..., None])
            continue
        if sums is None:
            sums = np.empty((min(48, pixels.shape[0]), width + 1, 3), dtype=np.float64)
            moments = np.empty_like(sums)
        left, right = left[partial], right[partial]
        count = right - left
        mean_x = (left + right - 1) * 0.5
        block_sums = sums[:block.shape[0]]
        block_moments = moments[:block.shape[0]]
        block_sums[:, 0] = 0
        block_moments[:, 0] = 0
        np.cumsum(block, axis=1, dtype=np.float64, out=block_sums[:, 1:])
        np.cumsum(block * x[..., None], axis=1, dtype=np.float64, out=block_moments[:, 1:])
        mean = (block_sums[rows, right] - block_sums[rows, left]) / count[..., None]
        covariance = (block_moments[rows, right] - block_moments[rows, left]) / count[..., None]
        covariance -= mean_x[..., None] * mean
        variance = np.where(count > 1, (count * count - 1) / 12, 1)
        fitted = mean + covariance * ((columns - mean_x) / variance)[..., None]
        output[top:top + 48][partial] = fitted
        np.copyto(output[top:top + 48], block, where=~selected[..., None])
    return output if axis == 1 else output.transpose(1, 0, 2)


def _smooth_background(rgb, allowed, radii):
    """축별 경계를 지키는 반복 평활화로 국소 연결 경로만 평균."""
    output = rgb
    for radius in radii:
        output = _linear_box_axis(output, allowed, radius, 1)
        output = _linear_box_axis(output, allowed, radius, 0)
    return output


def _estimate(rgb, backend=None):
    """국소 경계, 평면 추세와 미세 질감을 보호하며 얼룩 추정."""
    if backend is not None:
        from ._gpu_mottle import estimate
        return estimate(rgb, backend)
    edges = np.zeros(rgb.shape[:2], bool)
    for channel in range(3):
        gx = cv2.Sobel(rgb[..., channel], cv2.CV_32F, 1, 0, scale=0.125)
        gy = cv2.Sobel(rgb[..., channel], cv2.CV_32F, 0, 1, scale=0.125)
        edges |= gx * gx + gy * gy > 16
    protected = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(1 - protected, 8)
    eligible = (stats[:, cv2.CC_STAT_AREA] >= 500) & (np.minimum(
        stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= 16)
    eligible[0] = False
    allowed = eligible[labels]
    del labels, gx, gy
    distance = cv2.distanceTransform(np.pad(allowed.astype(np.uint8), 1),
                                     cv2.DIST_L2, 5)[1:-1, 1:-1]
    fine_image = _smooth_background(rgb, allowed, (1, 1))
    band = _smooth_background(rgb, allowed, (13,))
    np.subtract(fine_image, band, out=band)
    # 획 바로 옆의 기존 후광을 배경 재질로 오인하지 않도록 표본 분리.
    texture_samples = allowed * np.clip((distance - 1) / 2, 0, 1)
    fine = np.zeros(rgb.shape[:2], np.float32)
    band_energy = np.zeros_like(fine)
    xx, yy, xy = np.zeros_like(fine), np.zeros_like(fine), np.zeros_like(fine)
    for channel in range(3):
        fine += (rgb[..., channel] - fine_image[..., channel]) ** 2 / 3
        band_energy += band[..., channel] ** 2 / 3
        sx = cv2.Sobel(band[..., channel], cv2.CV_32F, 1, 0, scale=0.125)
        sy = cv2.Sobel(band[..., channel], cv2.CV_32F, 0, 1, scale=0.125)
        xx += sx * sx / 3
        yy += sy * sy / 3
        xy += sx * sy / 3
    del fine_image, sx, sy
    support = cv2.GaussianBlur(texture_samples, (0, 0), 4)
    fine *= texture_samples
    fine = cv2.GaussianBlur(fine, (0, 0), 4) / np.maximum(support, 1e-6)
    texture_weight = np.clip((1.5 - np.sqrt(fine)) / 0.65, 0, 1)
    band_energy *= allowed
    band_energy = cv2.GaussianBlur(band_energy, (0, 0), 5)
    # 질감 평면을 구성 요소 마스크 버퍼로 재사용.
    fine[:] = allowed
    band_energy /= np.maximum(cv2.GaussianBlur(fine, (0, 0), 5), 1e-6)
    contrast_weight = np.clip((4 - np.sqrt(band_energy)) / 1.5, 0, 1)
    support = np.maximum(cv2.GaussianBlur(texture_samples, (0, 0), 10), 1e-6)
    xx *= texture_samples
    xx = cv2.GaussianBlur(xx, (0, 0), 10) / support
    yy *= texture_samples
    yy = cv2.GaussianBlur(yy, (0, 0), 10) / support
    xy *= texture_samples
    xy = cv2.GaussianBlur(xy, (0, 0), 10) / support
    del texture_samples, fine
    coherence = np.sqrt((xx - yy) ** 2 + 4 * xy * xy) / (xx + yy + 1e-6)
    structure_weight = 1 - np.clip((coherence - 0.7) / 0.2, 0, 1) * np.clip(
        (np.sqrt(xx + yy) - 0.3) / 0.4, 0, 1)
    weight = (texture_weight * contrast_weight * structure_weight
              * np.clip((distance - 0.5) / 1.5, 0, 1))
    weight[~allowed | np.all(rgb >= 250, axis=2)] = 0
    return band, weight


def analyze(rgb: np.ndarray, backend=None) -> dict:
    """보정 가능한 평탄 배경의 얼룩 크기와 비율 분석."""
    correction, weight = _estimate(rgb, backend=backend)
    selected = weight > 0.5
    if not np.any(selected):
        return {"status": "insufficient_background", "score": None,
                "mottle_rms_dn": None, "eligible_fraction": 0.0}
    rms = float(np.sqrt(np.mean(correction[selected] ** 2)))
    return {"status": "ok", "score": float(min(100, rms * 25)),
            "mottle_rms_dn": rms, "eligible_fraction": float(np.mean(selected))}


def apply(rgb: np.ndarray, strength: float = 1.0, backend=None) -> tuple[np.ndarray, dict]:
    """문자 경계와 미세 질감을 유지하면서 8~25픽셀 얼룩 완화.

    축별 국소 표본을 보호 경계에서 끊고 직선 적합으로 평면 추세를 유지한다.
    1픽셀 미세 성분은 유지하고, 국소 미세 질감이 강한 영역은 보정에서 제외한다.
    출력은 클리핑과 반올림 없이 float32로 반환한다.
    """
    if strength == 0:
        return rgb.copy(), {"eligible_fraction": 0.0, "mean_abs_change": [0.0] * 3}
    residual, weight = _estimate(rgb, backend=backend)
    luma = residual @ _LUMA
    correction = residual
    correction *= 0.95
    correction -= (luma * 0.05)[..., None]
    correction *= (weight * strength)[..., None]
    return (rgb - correction).astype(np.float32), {
        "method": "local_edge_linear_band", "strength": float(strength),
        "eligible_fraction": float(np.mean(weight > 0.1)),
        "mean_abs_change": [float(np.mean(np.abs(correction[..., c]))) for c in range(3)],
        "small_box_radii": [1, 1], "background_box_radii": [13],
    }
