"""색 에지를 보호하면서 국소 대각선 해칭과 넓은 미세 질감 완화."""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
from scipy import fft

from ._parallel import _median_blur
from . import textgap
from .glyphring import _maximum_channel


_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
# NumPy 버전별 기존 스칼라 승격 규칙 유지.
_GAIN_DTYPE = (np.float32(1) + 1e-8).dtype
_TILE = 96
_STEP = 48


def _frequency_masks(size):
    """2~10픽셀 대역과 양쪽 대각선 방향 가중치 구성."""
    fy, fx = np.meshgrid(fft.fftfreq(size), fft.rfftfreq(size), indexing="ij")
    radius = np.hypot(fx, fy)
    angle = (np.degrees(np.arctan2(fy, fx)) + 90) % 180
    lower = np.clip((radius - 0.08) / 0.07, 0, 1)
    upper = np.clip((0.60 - radius) / 0.12, 0, 1)
    radial = (lower * upper).astype(np.float32)
    angle_distance = np.minimum(np.abs(angle - 42), np.abs(angle - 138))
    diagonal = np.exp(-0.5 * (angle_distance / 17) ** 2).astype(np.float32)
    return radial, diagonal, radius, angle


def _edge_protection(rgb):
    """원본 색 경계와 2픽셀 보호 이웃 계산."""
    raw_x = cv2.Sobel(rgb, cv2.CV_32F, 1, 0, scale=0.125)
    raw_y = cv2.Sobel(rgb, cv2.CV_32F, 0, 1, scale=0.125)
    strong = _maximum_channel(raw_x * raw_x + raw_y * raw_y) > 100
    protected = cv2.dilate(strong.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return strong, protected


def _structure_mask(rgb, protected=None):
    """밝기와 색 경계, 선명한 선 구조, 흰색 클리핑 영역 보호."""
    smooth = cv2.GaussianBlur(rgb, (0, 0), 1.3)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, scale=0.125)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, scale=0.125)
    xx = cv2.GaussianBlur(np.mean(gx * gx, axis=2), (0, 0), 1.5)
    yy = cv2.GaussianBlur(np.mean(gy * gy, axis=2), (0, 0), 1.5)
    xy = cv2.GaussianBlur(np.mean(gx * gy, axis=2), (0, 0), 1.5)
    del smooth, gx, gy
    energy = np.sqrt(xx + yy)
    coherence = np.sqrt((xx - yy) ** 2 + 4 * xy * xy) / (xx + yy + 1e-6)
    structure = np.maximum(np.clip((energy - 1.0) / 5, 0, 1),
                           coherence * np.clip((energy - 0.6) / 2.4, 0, 1))
    del xx, yy, xy, coherence
    if protected is None:
        _, protected = _edge_protection(rgb)
    edge_soft = cv2.GaussianBlur(protected.astype(np.float32), (0, 0), 1.0)
    mask = (1 - structure) * (1 - edge_soft)
    mask[protected | np.all(rgb >= 250, axis=2)] = 0
    return mask.astype(np.float32), energy


def _text_gap_estimate(rgb, strong=None, protected=None):
    """문자색을 제외한 정규화 합성곱으로 좁은 배경 틈의 미세 질감 추정."""
    if strong is None:
        strong, protected = _edge_protection(rgb)
    dense_edges = cv2.blur(strong.astype(np.float32), (31, 31))
    density_weight = cv2.GaussianBlur((dense_edges > 0.04).astype(np.float32), (0, 0), 1.0)
    median = _median_blur(np.clip(np.rint(rgb), 0, 255).astype(np.uint8), 31)
    color_difference = _maximum_channel(np.abs(rgb - median))
    del median
    background_weight = np.clip((7 - color_difference) / 3, 0, 1)
    background_weight[protected | np.all(rgb >= 250, axis=2)] = 0
    del strong, protected, dense_edges, color_difference
    support = cv2.GaussianBlur(background_weight, (0, 0), 1.4)
    total = cv2.GaussianBlur(rgb * background_weight[..., None], (0, 0), 1.4)
    np.divide(total, support[..., None], out=total, where=support[..., None] > 0.05)
    total[support <= 0.05] = 0
    residual = rgb - total
    del total
    gap_weight = np.clip((8 - _maximum_channel(np.abs(residual))) / 3, 0, 1) * density_weight
    gap_weight[(support <= 0.05) | (background_weight == 0)] = 0
    return residual, gap_weight, density_weight


def _flat_interior_estimate(rgb, protected=None):
    """강한 경계로 둘러싸인 평탄 구역 안에서 넓은 얼룩 추정.

    내부 추정은 색 경계의 2픽셀 이웃을 표본과 보정 대상에서 제외한다. 구역별 평면
    추세를 유지하고, 같은 구역의 잔차만 정규화 합성곱으로 평균한다.
    두 평탄 구역 사이의 거의 직선인 경계는 확인된 울렁임과 후광도 완화한다.
    """
    if protected is None:
        _, protected = _edge_protection(rgb)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((~protected).astype(np.uint8), 8)
    del protected
    correction = np.zeros_like(rgb)
    weight = np.zeros(rgb.shape[:2], dtype=np.float32)
    eligible = {}
    height, width = rgb.shape[:2]
    for label in range(1, count):
        left, top, w, h, area = stats[label]
        if area < 200 or left == 0 or top == 0 or left + w == width or top + h == height:
            continue
        component = (labels[top:top + h, left:left + w] == label).astype(np.uint8)
        distance = cv2.distanceTransform(np.pad(component, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        if distance.max() < 6:
            continue
        pixels = rgb[top:top + h, left:left + w]
        selected = component > 0
        minimum = np.min(pixels[selected], axis=0)
        if np.array_equal(minimum, np.max(pixels[selected], axis=0)):
            eligible[label] = np.stack([minimum, np.zeros(3, np.float32), np.zeros(3, np.float32)])
            weight[top:top + h, left:left + w] = np.maximum(
                weight[top:top + h, left:left + w], np.clip((distance - 1) / 3, 0, 1))
            continue
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        coordinates = np.stack([np.ones_like(xx), xx - w / 2, yy - h / 2], axis=-1)
        trend, _, _, _ = np.linalg.lstsq(coordinates[selected], pixels[selected], rcond=None)
        residual = pixels - coordinates @ trend
        # 실제 재질이나 형태가 섞인 구역은 기존 미세 대역 경로 유지.
        if (np.max(np.sqrt(np.mean(residual[selected] ** 2, axis=0))) > 3
                or np.percentile(_maximum_channel(np.abs(residual[selected])), 95) > 6):
            continue
        plane = trend.copy()
        plane[0] -= (left + w / 2) * trend[1] + (top + h / 2) * trend[2]
        eligible[label] = plane
        support = cv2.GaussianBlur(component.astype(np.float32), (0, 0), 5,
                                   borderType=cv2.BORDER_CONSTANT)
        total = cv2.GaussianBlur(residual * component[..., None], (0, 0), 5,
                                 borderType=cv2.BORDER_CONSTANT)
        mean = np.zeros_like(total)
        np.divide(total, support[..., None], out=mean, where=support[..., None] > 0)
        correction[top:top + h, left:left + w][selected] = (residual - mean)[selected]
        weight[top:top + h, left:left + w] = np.maximum(
            weight[top:top + h, left:left + w], np.clip((distance - 1) / 3, 0, 1))
    if len(eligible) >= 2:
        boundary_residual, boundary_weight = _flat_boundary_estimate(rgb, labels, eligible)
        boundary = boundary_weight > 0
        correction[boundary] = boundary_residual[boundary]
        weight[boundary] = boundary_weight[boundary]
    weight[np.all(rgb >= 250, axis=2)] = 0
    return correction, weight


def _flat_boundary_estimate(rgb, labels, eligible):
    """인접한 두 평탄 구역 사이의 얇은 경계 후광과 작은 위치 요동 추정.

    두 구역을 가르는 보호 띠만 표본화한다. 거의 수평/수직인 직선으로
    설명되는 경계에서 가시적인 요동이나 후광이 확인되면, 두 색 평면과
    1픽셀 안티앨리어싱 전이로 복원한다. 교차점 주변은 점차 보정을 줄인다.
    """
    correction = np.zeros_like(rgb)
    weight = np.zeros(rgb.shape[:2], dtype=np.float32)
    pairs = defaultdict(list)
    for axis in (0, 1):
        oriented = labels if axis == 0 else labels.T
        for along, line in enumerate(oriented):
            changes = np.flatnonzero(np.diff(np.r_[False, line == 0, False]))
            for start, stop in zip(changes[::2], changes[1::2]):
                if start == 0 or stop == len(line) or stop - start > 10:
                    continue
                first, second = int(line[start - 1]), int(line[stop])
                if first == second or first not in eligible or second not in eligible:
                    continue
                pairs[axis, first, second].append((along, start, stop))
    for (axis, first, second), samples in pairs.items():
        samples = np.asarray(samples)
        if len(samples) < 12:
            continue
        # 같은 경계의 행/열 표본을 묶어 작은 배열 연산의 반복 비용 제한.
        lengths = samples[:, 2] - samples[:, 1] + 2
        offsets = np.arange(lengths.max(), dtype=np.float32)[None, :]
        normal = np.minimum(samples[:, 1, None] - 1 + offsets, samples[:, 2, None]).astype(np.float32)
        position = np.broadcast_to(samples[:, 0, None], normal.shape).astype(np.float32)
        xx, yy = (normal, position) if axis == 0 else (position, normal)
        coordinates = np.stack([np.ones_like(xx), xx, yy], axis=-1)
        first_color = coordinates @ eligible[first]
        difference = coordinates @ eligible[second] - first_color
        norm = np.sum(difference * difference, axis=2)
        pixels = rgb[yy.astype(int), xx.astype(int)]
        alpha = np.zeros_like(norm)
        np.divide(np.sum((pixels - first_color) * difference, axis=2), norm,
                  out=alpha, where=norm >= 400)
        crossings = ((alpha[:, :-1] <= 0.5) & (alpha[:, 1:] > 0.5)
                     & (offsets[:, 1:] < lengths[:, None]))
        valid = (norm.min(axis=1) >= 400) & (crossings.sum(axis=1) == 1)
        if np.count_nonzero(valid) < 12:
            continue
        rows = np.flatnonzero(valid)
        index = np.argmax(crossings[valid], axis=1)
        step = alpha[rows, index + 1] - alpha[rows, index]
        crossing = normal[rows, index] + (0.5 - alpha[rows, index]) / step
        overshoot = np.maximum(np.maximum(-alpha, alpha - 1), 0) * np.sqrt(norm)
        samples = np.column_stack([samples[valid], crossing, overshoot.max(axis=1)[valid], step])
        along = samples[:, 0]
        coordinates = np.stack([np.ones_like(along), along - along.mean()], axis=-1)
        line, _, _, _ = np.linalg.lstsq(coordinates, samples[:, 3], rcond=None)
        displacement = samples[:, 3] - coordinates @ line
        if (np.median(samples[:, 5]) > 0.85 or abs(line[1]) > 0.25
                or np.max(np.abs(displacement)) > 1.2
                or np.max(np.diff(along)) > 2):
            continue
        # 깨끗한 직선은 유지하고, 색 방향 6 DN 후광 또는 0.1픽셀 요동만 보정.
        if np.std(displacement) < 0.1 and np.percentile(samples[:, 4], 75) < 6:
            continue
        for index, (position, start, stop, _, _, _) in enumerate(samples):
            position, start, stop = int(position), int(start), int(stop)
            normal = np.arange(start, stop, dtype=np.float32)
            xx, yy = ((normal, np.full_like(normal, position)) if axis == 0
                      else (np.full_like(normal, position), normal))
            plane_coordinates = np.stack([np.ones_like(xx), xx, yy], axis=-1)
            first_color = plane_coordinates @ eligible[first]
            second_color = plane_coordinates @ eligible[second]
            edge = coordinates[index] @ line
            alpha = np.clip(normal - edge + 0.5, 0, 1)
            target = first_color * (1 - alpha[:, None]) + second_color * alpha[:, None]
            blend = np.clip(min(position - along.min(), along.max() - position) / 3, 0, 1)
            ix, iy = xx.astype(int), yy.astype(int)
            correction[iy, ix] = rgb[iy, ix] - target
            weight[iy, ix] = blend
    return correction, weight


def analyze(rgb: np.ndarray, *, backend=None, **options) -> dict:
    """약한 구조 영역의 대각선 대역 세기와 대표 방향/간격 측정.

    점수는 해당 질감의 가시 세기 지표로, 생성 모델을 판별하는 값이 아니다.
    방향은 화면의 수평축 기준 0~180도이며 간격은 선에 수직인 픽셀 거리다.
    """
    if min(rgb.shape[:2]) < _TILE:
        rgb = np.pad(rgb, ((0, max(0, _TILE - rgb.shape[0])),
                           (0, max(0, _TILE - rgb.shape[1])), (0, 0)), mode="reflect")
    radial, diagonal, radius, angle = _frequency_masks(_TILE)
    if backend is None:
        mask, _ = _structure_mask(rgb)
    else:
        _, protected = _edge_protection(rgb)
        mask, _ = backend.structure_mask(rgb, protected)
    background = cv2.erode((mask > 0.9).astype(np.uint8), np.ones((5, 5), np.uint8))
    residual = rgb - cv2.GaussianBlur(rgb, (0, 0), 1.8)
    window = np.outer(np.hanning(_TILE), np.hanning(_TILE)).astype(np.float32)
    angular = np.zeros(36, dtype=np.float64)
    radial_energy = np.zeros(16, dtype=np.float64)
    levels = []
    h, w = rgb.shape[:2]
    # 균등 간격 표본으로 큰 이미지의 분석 비용 제한.
    ys = np.unique(np.linspace(0, max(0, h - _TILE), min(12, max(1, h // _TILE))).astype(int))
    xs = np.unique(np.linspace(0, max(0, w - _TILE), min(12, max(1, w // _TILE))).astype(int))
    for top in ys:
        for left in xs:
            tile = residual[top:top + _TILE, left:left + _TILE]
            tile_mask = background[top:top + _TILE, left:left + _TILE]
            if tile.shape[:2] != (_TILE, _TILE) or np.mean(tile_mask) < 0.2:
                continue
            sample_window = window * tile_mask
            mean = np.sum(tile * sample_window[..., None], axis=(0, 1)) / np.sum(sample_window)
            spectrum = fft.rfft2((tile - mean) * sample_window[..., None], axes=(0, 1))
            power = np.mean(np.abs(spectrum) ** 2, axis=2)
            measure = power * radial * diagonal
            # rFFT의 양/음 주파수 쌍과 창의 에너지를 함께 보정.
            mirror = np.full_like(radial, 2)
            mirror[:, 0] = mirror[:, -1] = 1
            levels.append(float(np.sqrt(np.sum(measure * mirror) / (_TILE ** 2 * np.sum(sample_window ** 2)))))
            for k in range(36):
                sector = (angle >= k * 5) & (angle < (k + 1) * 5) & (radial > 0.5)
                angular[k] += np.sum(measure[sector]) / max(1, np.count_nonzero(sector))
            for k in range(16):
                annulus = (radius >= 0.1 + k * 0.025) & (radius < 0.125 + k * 0.025)
                radial_energy[k] += np.sum(measure[annulus]) / max(1, np.count_nonzero(annulus))
    if not levels:
        return {"score": None, "status": "insufficient_background", "diagonal_rms_dn": None,
                "dominant_angle_degrees": None, "dominant_period_pixels": None, "sampled_tiles": 0}
    if max(levels) < 1e-6:
        return {"score": 0.0, "status": "ok", "diagonal_rms_dn": 0.0, "dominant_angle_degrees": None,
                "dominant_period_pixels": None, "sampled_tiles": len(levels)}
    level = float(np.percentile(levels, 60))
    return {"score": float(min(100, level * 25)), "status": "ok", "diagonal_rms_dn": level,
            "dominant_angle_degrees": float(np.argmax(angular) * 5 + 2.5),
            "dominant_period_pixels": float(1 / (0.1125 + np.argmax(radial_energy) * 0.025)),
            "sampled_tiles": len(levels)}


def apply(
    rgb: np.ndarray,
    strength: float = 1.0,
    *,
    luma_strength: float = 0.8,
    chroma_strength: float = 0.9,
    backend=None,
    **options,
) -> tuple[np.ndarray, dict]:
    """겹친 국소 푸리에 창으로 표면 해칭 대역 완화.

    96픽셀 창, 48픽셀 간격으로 분석/합성하며 겹침 가중치로 경계 연결.
    기본 대역 감쇠에 국소 대각선 에너지 비율을 반영한 방향 감쇠를 추가한다.
    밀집된 문자 사이 배경은 문자색을 제외한 정규화 합성곱으로 별도 추정한다.
    반복된 문자 행의 평탄 배경은 넓은 색 추세로 정리하고 안티앨리어싱은 유지한다.
    강한 경계로 둘러싸인 평탄 구역은 같은 구역 안에서 넓은 얼룩도 줄인다.
    평탄 구역 경계와 밀집 문자 배경을 제외한 강한 경계 이웃은 유지한다.
    출력은 클리핑과 반올림 없이 float32로 반환한다.
    """
    if strength == 0:
        return rgb.copy(), {"masked_fraction": 0.0, "mean_abs_change": [0.0] * 3}
    strong, protected = _edge_protection(rgb)
    mask, _ = (_structure_mask(rgb, protected) if backend is None
               else backend.structure_mask(rgb, protected))
    radial, diagonal, _, _ = _frequency_masks(_TILE)
    residual = rgb - cv2.GaussianBlur(rgb, (0, 0), 1.8)
    if backend is not None:
        correction = backend.fft_band(residual, radial, diagonal)
        del residual
    else:
        pad = _STEP
        h, w = rgb.shape[:2]
        bottom = pad + (-h) % _STEP
        right = pad + (-w) % _STEP
        padded = np.pad(residual, ((pad, bottom), (pad, right), (0, 0)), mode="reflect")
        del residual
        correction = np.zeros_like(padded)
        weights = np.zeros(padded.shape[:2], dtype=np.float32)
        window = np.outer(np.hanning(_TILE), np.hanning(_TILE)).astype(np.float32)
        window_squared = window * window
        lefts = range(0, padded.shape[1] - _TILE + 1, _STEP)
        # 임시 배열 크기를 제한하는 8개 창 묶음.
        for top in range(0, padded.shape[0] - _TILE + 1, _STEP):
            for start in range(0, len(lefts), 8):
                slices = [np.s_[top:top + _TILE, left:left + _TILE] for left in lefts[start:start + 8]]
                tiles = np.stack([padded[sl] for sl in slices])
                spectrum = fft.rfft2(tiles * window[None, ..., None], axes=(1, 2))
                power = np.mean(np.abs(spectrum) ** 2, axis=3)
                diagonal_ratio = (np.sum(power * radial * diagonal, axis=(1, 2))
                                  / (np.sum(power * radial, axis=(1, 2)).astype(_GAIN_DTYPE, copy=False) + 1e-8))
                directional_gain = np.clip((diagonal_ratio - 0.25) / 0.35, 0, 1)
                scaled_gain = (0.3 * directional_gain).astype(diagonal.dtype, copy=False)
                attenuation = radial * (0.7 + scaled_gain[:, None, None] * diagonal)
                filtered = fft.irfft2(spectrum * attenuation[..., None], s=(_TILE, _TILE), axes=(1, 2))
                for index, sl in enumerate(slices):
                    correction[sl] += filtered[index] * window[..., None]
                    weights[sl] += window_squared
        correction = correction[pad:pad + h, pad:pad + w] / weights[pad:pad + h, pad:pad + w, None]
        del padded, tiles, spectrum, power, filtered, attenuation, weights
    correction_y = correction @ _LUMA
    correction = chroma_strength * correction + (luma_strength - chroma_strength) * correction_y[..., None]
    del correction_y
    correction *= mask[..., None]
    gap_residual, gap_weight, text_region = _text_gap_estimate(rgb, strong, protected)
    del strong
    gap_y = gap_residual @ _LUMA
    gap_residual *= chroma_strength
    gap_residual += (luma_strength - chroma_strength) * gap_y[..., None]
    correction *= 1 - text_region[..., None]
    correction += gap_weight[..., None] * gap_residual
    del gap_residual, gap_y
    active_weight = (1 - text_region) * mask + gap_weight
    text_gap_fraction = float(np.mean(gap_weight > 0.1))
    del text_region, mask, gap_weight
    interior_residual, interior_weight = _flat_interior_estimate(rgb, protected)
    del protected
    interior_y = interior_residual @ _LUMA
    interior_residual *= chroma_strength
    interior_residual += (luma_strength - chroma_strength) * interior_y[..., None]
    correction *= 1 - interior_weight[..., None]
    correction += interior_weight[..., None] * interior_residual
    del interior_residual, interior_y
    active_weight *= 1 - interior_weight
    active_weight += interior_weight
    masked_fraction = float(np.mean(active_weight > 0.1))
    flat_interior_fraction = float(np.mean(interior_weight > 0.1))
    del active_weight, interior_weight
    correction *= strength
    output = (rgb - correction).astype(np.float32, copy=False)
    del correction
    output, textgap_report = textgap.apply(output, strength=strength, backend=backend)
    correction = rgb - output
    return output, {"masked_fraction": masked_fraction,
                    "mean_abs_change": [float(v) for v in np.mean(np.abs(correction), axis=(0, 1))],
                    "method": "local_oriented_band", "tile_size": _TILE,
                    "text_gap_fraction": text_gap_fraction,
                    "flat_interior_fraction": flat_interior_fraction,
                    "interline_background": textgap_report,
                    "luma_strength": float(luma_strength), "chroma_strength": float(chroma_strength)}
