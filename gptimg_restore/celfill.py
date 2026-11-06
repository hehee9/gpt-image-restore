"""@description 선화 채색 면과 부드러운 경계의 영역별 모델 복원."""

from __future__ import annotations
from dataclasses import dataclass
import math
import os
import time
import cv2
import numpy as np
from scipy import ndimage
from scipy.special import ndtri, ndtr
from scipy.spatial import cKDTree
from scipy.signal import find_peaks
from .contour import _line_art_context
from ._native import load_native
from ._parallel import _executor

_native_ink = load_native("_native_celfill_ink")
_native_fit = load_native("_native_celfill_fit")

_LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def _rgb_max(values):
    """@description RGB 세 채널의 픽셀별 최댓값."""
    return np.maximum(np.maximum(values[..., 0], values[..., 1]), values[..., 2])


def _rgb_min(values):
    """@description RGB 세 채널의 픽셀별 최솟값."""
    return np.minimum(np.minimum(values[..., 0], values[..., 1]), values[..., 2])


def _rgb_range(values):
    """@description RGB 세 채널의 픽셀별 범위."""
    return _rgb_max(values) - _rgb_min(values)


def _polynomial_basis(x, y, count=10):
    """@description 채색 곡면 적합과 평가에 쓰는 3차 좌표 기저."""
    x, y = np.broadcast_arrays(x, y)
    terms = [np.ones_like(x), x, y]
    if count >= 6:
        terms.extend([x * x, x * y, y * y])
    if count == 10:
        terms.extend([x ** 3, x * x * y, x * y * y, y ** 3])
    return np.stack(terms, axis=-1)


def _nearest_distances(nearest):
    """@description 최근접 표본 좌표에서 경계 판정용 거리를 행 단위로 계산."""
    height, width = nearest.shape[1:]
    distances = np.empty((height, width), np.float32)
    xx = np.arange(width, dtype=np.float32)[None, :]
    for top in range(0, height, 128):
        bottom = min(height, top + 128)
        yy = np.arange(top, bottom, dtype=np.float32)[:, None]
        dx = nearest[1, top:bottom].astype(np.float32) - xx
        dy = nearest[0, top:bottom].astype(np.float32) - yy
        np.hypot(dx, dy, out=distances[top:bottom])
    return distances


def _region_is_consistent(data, model, interior, count):
    """@description 잔차 70백분위의 6 DN 이하 여부를 개수로 우선 판정."""
    if _native_fit is not None:
        return _native_fit._region_is_consistent(data, model, interior, count)
    below = 0
    for top in range(0, len(interior), 128):
        bottom = min(len(interior), top + 128)
        selected = interior[top:bottom]
        residual = _rgb_max(abs(data[top:bottom][selected] - model[top:bottom][selected]))
        below += np.count_nonzero(residual <= 6)
    position = (count - 1) * 0.7
    if below > math.ceil(position):
        return True
    if below <= math.floor(position):
        return False
    # 경계의 두 표본이 6 DN을 사이에 두는 경우만 원래 보간 백분위 계산.
    residual = np.empty(count, np.result_type(data.dtype, model.dtype))
    offset = 0
    for top in range(0, len(interior), 128):
        bottom = min(len(interior), top + 128)
        selected = interior[top:bottom]
        values = _rgb_max(abs(data[top:bottom][selected] - model[top:bottom][selected]))
        residual[offset:offset + len(values)] = values
        offset += len(values)
    return np.percentile(residual, 70) <= 6


def _boundary_coverage(alpha, support, offset_x, offset_y):
    """@description 원래 좌표의 보간 반올림을 유지하는 경계 피복률 계산."""
    direction = cv2.GaussianBlur(alpha, (7, 7), 0.8)
    gx = cv2.Sobel(direction, cv2.CV_32F, 1, 0, scale=0.125)
    gy = cv2.Sobel(direction, cv2.CV_32F, 0, 1, scale=0.125)
    magnitude = np.maximum(np.hypot(gx, gy), 0.001)
    tx, ty = -gy / magnitude, gx / magnitude
    ly = np.arange(alpha.shape[0], dtype=np.float32)[:, None] + np.float32(offset_y)
    lx = np.arange(alpha.shape[1], dtype=np.float32)[None, :] + np.float32(offset_x)
    coverage = np.zeros_like(alpha)
    mass = np.zeros_like(alpha)
    total_weight = 0.0
    supported_alpha = alpha * support
    for distance in range(-3, 4):
        weight = np.float32(np.exp(-distance * distance / (2 * 1.3**2)))
        mx, my = lx + tx * distance, ly + ty * distance
        if offset_x:
            mx -= np.float32(offset_x)
        if offset_y:
            my -= np.float32(offset_y)
        valid = cv2.remap(
            support, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        sampled = cv2.remap(
            supported_alpha, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        coverage += sampled * weight
        mass += valid * weight
        total_weight += float(weight)
    coverage /= np.maximum(mass, 0.001)
    return coverage, mass, total_weight


@dataclass
class _FillModels:
    values: np.ndarray
    selected: np.ndarray
    labels: np.ndarray
    coefficients: np.ndarray
    limits: np.ndarray
    centers: np.ndarray
    scales: np.ndarray
    gradient: np.ndarray
    regions: list[dict]


@dataclass
class _BoundaryModels:
    output: np.ndarray
    color_mask: np.ndarray
    pure_field: np.ndarray
    ink_distance: np.ndarray
    seed_distance: np.ndarray
    report: dict


def _detail_context(image):
    """@description 작은 흰 눈, 따뜻한 하이라이트와 밀집 금속 세부의 문맥."""
    height, width = image.shape[:2]
    red, green, blue = np.moveaxis(image, -1, 0)
    protected = np.zeros((height, width), np.uint8)
    white = (_rgb_min(image) > 218) & (_rgb_range(image) < 28)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        white.astype(np.uint8), 8
    )
    for label in range(1, count):
        left, top, w, h, area = stats[label]
        if (
            4 <= area <= image.shape[0] * image.shape[1] * 0.012
            and w < width * 0.25
            and h < height * 0.16
            and left > 0
            and top > 0
            and left + w < width
            and top + h < height
        ):
            protected[
                max(0, top - 9) : min(height, top + h + 9),
                max(0, left - 9) : min(width, left + w + 9),
            ] = 1
    warm_highlight = (red > 245) & (green > 222) & (blue > 200) & (red - blue > 12)
    warm_highlight = cv2.dilate(
        warm_highlight.astype(np.uint8), np.ones((9, 9), np.uint8)
    )
    protected |= warm_highlight
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, scale=0.125)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, scale=0.125)
    strong = _rgb_max(gx * gx + gy * gy) > 16
    density = cv2.GaussianBlur(strong.astype(np.float32), (0, 0), 9)
    neutral = (_rgb_range(image) < 35) & (red > 40) & (red < 220)
    dense_metal = ((density > 0.18) & neutral).astype(np.uint8)
    dense_metal = cv2.dilate(dense_metal, np.ones((13, 13), np.uint8))
    protected |= dense_metal
    return protected > 0


def _ink_masks(src):
    """@description 연결된 잉크 중심과 경계 단면 처리 지지 영역."""
    neutral_ink = (_rgb_max(src) < 90) & (_rgb_range(src) < 22)
    ni, il, ist, _ = cv2.connectedComponentsWithStats(neutral_ink.astype(np.uint8), 8)
    validink = np.zeros(ni, bool)
    for k in range(1, ni):
        x, y, ww, hh, area = ist[k]
        validink[k] = area >= 8 and max(ww, hh) >= 5
    core = validink[il]
    core |= neutral_ink & (
        cv2.dilate(core.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    )
    noiseink = neutral_ink & ~core
    luma0 = src @ _LUMA
    closing = cv2.morphologyEx(
        luma0, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )
    ink_support = (
        cv2.dilate(
            (core | (closing - luma0 > 30)).astype(np.uint8), np.ones((7, 7), np.uint8)
        )
        > 0
    )
    return core, noiseink, ink_support


def _faint_profile(plane, points, tangent):
    """@description 긴 능선의 대비, 반치폭과 단면 일관성 측정."""
    if _native_fit is not None:
        return _native_fit._faint_profile(plane, points, tangent)
    normal = np.array([-tangent[1], tangent[0]], np.float32)
    depths = np.arange(-8, 8.01, 0.5, dtype=np.float32)
    coordinates = points[:, None, :] + normal[None, None, :] * depths[None, :, None]
    profile = cv2.remap(
        plane,
        coordinates[..., 0],
        coordinates[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    profile = ndimage.gaussian_filter1d(profile, 1.0, axis=0, mode="nearest")
    left = np.mean(profile[:, :5], axis=1)
    right = np.mean(profile[:, -5:], axis=1)
    baseline = left[:, None] + (right - left)[:, None] * (depths[None, :] + 8) / 16
    residual = baseline - profile
    sign = 1.0 if np.median(residual[:, 16]) >= 0 else -1.0
    residual *= sign
    peaks = np.argmax(residual[:, 14:19], axis=1) + 14
    amplitudes = residual[np.arange(len(points)), peaks]
    widths = []
    for row, peak, amplitude in zip(residual, peaks, amplitudes):
        if amplitude <= 0:
            continue
        threshold = amplitude * 0.5
        left_index = int(peak)
        right_index = int(peak)
        while left_index > 0 and row[left_index - 1] > threshold:
            left_index -= 1
        while right_index < len(depths) - 1 and row[right_index + 1] > threshold:
            right_index += 1
        if left_index == 0 or right_index == len(depths) - 1:
            continue
        a = depths[left_index] - 0.5 * (row[left_index] - threshold) / max(
            row[left_index] - row[left_index - 1], 0.001
        )
        b = depths[right_index] + 0.5 * (row[right_index] - threshold) / max(
            row[right_index] - row[right_index + 1], 0.001
        )
        widths.append(float(b - a))
    return amplitudes, np.asarray(widths, np.float32), sign


def _parallel_ridge_family(
    candidate, normal_x, normal_y, contrast, box, center, tangent, extent, sign
):
    """@description 비슷한 간격으로 반복되는 평행 능선 묶음 판정."""
    h, w = candidate.shape
    x, y, bw, bh = box
    x0 = max(0, x - 36)
    x1 = min(w, x + bw + 36)
    y0 = max(0, y - 36)
    y1 = min(h, y + bh + 36)
    yy, xx = np.nonzero(candidate[y0:y1, x0:x1])
    xx += x0
    yy += y0
    if not len(xx):
        return False
    dx = xx - center[0]
    dy = yy - center[1]
    u = dx * tangent[0] + dy * tangent[1]
    v = -dx * tangent[1] + dy * tangent[0]
    alignment = abs(-normal_x[yy, xx] * tangent[1] + normal_y[yy, xx] * tangent[0])
    valid = (
        (abs(u) < extent * 0.55)
        & (abs(v) < 34)
        & (alignment > 0.93)
        & (contrast[yy, xx] * sign > 0)
    )
    histogram = np.bincount(
        np.rint(v[valid] + 36).astype(np.int32), minlength=73
    ).astype(np.float32)
    histogram = ndimage.gaussian_filter1d(histogram, 1)
    peaks, _ = find_peaks(
        histogram,
        distance=3,
        height=max(4, extent * 0.2),
        prominence=max(3, extent * 0.1),
    )
    if len(peaks) < 3:
        return False
    gaps = np.diff(peaks)
    regular = (gaps >= 3) & (gaps <= 30)
    for index in range(len(gaps) - 1):
        if (
            regular[index : index + 2].all()
            and max(gaps[index : index + 2]) / min(gaps[index : index + 2]) < 1.6
        ):
            return True
    return False


def _faint_strokes(src, core):
    """@description 반복 해칭을 제외한 길고 좁은 저대비 묘사 보호."""
    if _native_fit is not None:
        return _native_fit._faint_strokes(src, core)
    h, w = src.shape[:2]
    if core.all() or max(h, w) < 40:
        return np.zeros((h, w), bool), {
            "faint_stroke_pixels": 0,
            "faint_stroke_components": 0,
            "faint_family_candidates_rejected": 0,
        }
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    candidate = np.zeros((h, w), bool)
    best = np.zeros((h, w), np.float32)
    normal_x = np.zeros_like(best)
    normal_y = np.zeros_like(best)
    contrast = np.zeros_like(best)
    noise_map = np.ones_like(best)
    channel_map = np.zeros((h, w), np.uint8)
    # 작은 단면마다 OpenCV가 전체 채널을 복사하는 비용 제외.
    planes = [None, None, None]
    near_ink = cv2.dilate(core.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    kernel = int(np.clip(round(15 * max(h, w) / 1024), 15, 63)) | 1
    for channel in range(3):
        plane = np.ascontiguousarray(src[..., channel])
        residual = abs(plane - cv2.GaussianBlur(plane, (0, 0), 0.65))
        quantized = np.uint8(np.clip(np.rint(residual * 16), 0, 255))
        noise = np.maximum(
            cv2.medianBlur(quantized, kernel).astype(np.float32) / 16 / 0.6745 / 0.6,
            0.25,
        )
        del residual, quantized
        smooth = cv2.GaussianBlur(plane, (0, 0), 1.25)
        hxx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, scale=0.25)
        hyy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, scale=0.25)
        hxy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, scale=0.25)
        difference = np.sqrt((hxx - hyy) ** 2 + 4 * hxy * hxy)
        first = (hxx + hyy + difference) * 0.5
        second = (hxx + hyy - difference) * 0.5
        np.abs(first, out=first)
        np.abs(second, out=second)
        angle = 0.5 * np.arctan2(2 * hxy, hxx - hyy)
        use_second = second > first
        angle += np.float32(0)
        for top in range(0, h, 128):
            local_angle = angle[top:top + 128]
            selected = use_second[top:top + 128]
            local_angle[selected] = (
                local_angle[selected].astype(np.float64) + np.pi / 2
            ).astype(np.float32)
        del smooth, hxx, hyy, hxy, difference, local_angle, selected, use_second
        nx = np.cos(angle).astype(np.float32)
        ny = np.sin(angle).astype(np.float32)
        other = np.minimum(first, second)
        np.maximum(first, second, out=first)
        strength = first
        del first, second, angle
        ahead = cv2.remap(
            strength,
            xx + nx,
            yy + ny,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        behind = cv2.remap(
            strength,
            xx - nx,
            yy - ny,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        # 단면 샘플링 전에 능선 후보 확정, 사용이 끝난 전체 영상 배열 해제.
        eligible = (
            (strength > 0.035)
            & (strength > 1.8 * other)
            & (strength >= 0.85 * ahead)
            & (strength >= 0.85 * behind)
            & ~near_ink
        )
        del ahead, behind, other
        # 능선 후보 위치만 법선 단면 표본 계산.
        sample_y, sample_x = np.nonzero(eligible)
        if not len(sample_x):
            del plane, noise, nx, ny, strength, eligible
            continue
        sample_nx, sample_ny = nx[sample_y, sample_x], ny[sample_y, sample_x]
        sample_x_float = sample_x.astype(np.float32)
        sample_y_float = sample_y.astype(np.float32)
        left = np.zeros(len(sample_x), np.float32)
        right = np.zeros_like(left)
        map_x = np.zeros(((len(sample_x) + 1023) // 1024, 1024), np.float32)
        map_y = np.zeros_like(map_x)
        center = cv2.GaussianBlur(plane, (0, 0), 0.35)
        for distance in (3, 4, 5):
            map_x.flat[:len(sample_x)] = sample_x_float - sample_nx * distance
            map_y.flat[:len(sample_x)] = sample_y_float - sample_ny * distance
            left += (
                cv2.remap(
                    center,
                    map_x,
                    map_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                ).ravel()[:len(sample_x)]
                / 3
            )
            map_x.flat[:len(sample_x)] = sample_x_float + sample_nx * distance
            map_y.flat[:len(sample_x)] = sample_y_float + sample_ny * distance
            right += (
                cv2.remap(
                    center,
                    map_x,
                    map_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                ).ravel()[:len(sample_x)]
                / 3
            )
        center = center[sample_y, sample_x]
        a = left - center
        b = right - center
        amplitude = (a + b) * 0.5
        del center, left, right
        sample_noise = noise[sample_y, sample_x]
        eligible = (a * b > 0) & (
            np.minimum(abs(a), abs(b)) > np.maximum(0.6, sample_noise * 0.6)
        )
        eligible &= (np.minimum(abs(a), abs(b)) > np.maximum(abs(a), abs(b)) * 0.3) & (
            abs(amplitude) < 24
        )
        quality = strength[sample_y, sample_x] * abs(amplitude) / sample_noise
        update = eligible & (quality > best[sample_y, sample_x])
        candidate[sample_y[eligible], sample_x[eligible]] = True
        update_y, update_x = sample_y[update], sample_x[update]
        best[update_y, update_x] = quality[update]
        normal_x[update_y, update_x] = sample_nx[update]
        normal_y[update_y, update_x] = sample_ny[update]
        contrast[update_y, update_x] = amplitude[update]
        noise_map[update_y, update_x] = sample_noise[update]
        channel_map[update_y, update_x] = channel
        del (
            plane, noise, nx, ny, strength, a, b, amplitude, eligible, quality, update,
            sample_y, sample_x, sample_nx, sample_ny, sample_x_float, sample_y_float,
            sample_noise, update_y, update_x,
            map_x, map_y,
        )
    del yy, xx, near_ink
    protected = np.zeros((h, w), np.uint8)
    rejected_family = 0
    min_length = max(40, 40 * max(h, w) / 1024)
    for direction in np.arange(0, np.pi, np.pi / 6):
        tangent = np.array([np.cos(direction), np.sin(direction)], np.float32)
        alignment = abs(-normal_x * tangent[1] + normal_y * tangent[0])
        ridge_pixels = (
            candidate
            & (alignment > np.cos(np.deg2rad(25)))
            & (abs(contrast) >= np.maximum(1.5, 2.5 * noise_map))
        )
        connector = np.zeros((7, 7), np.uint8)
        start = tuple(np.rint(3 - tangent * 2).astype(int))
        stop = tuple(np.rint(3 + tangent * 2).astype(int))
        cv2.line(connector, start, stop, 1, 1)
        # Join short gaps along the stroke, without widening its observations.
        selected = cv2.dilate(ridge_pixels.astype(np.uint8), connector)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(selected, 8)
        for index in range(1, count):
            x, y, bw, bh, area = stats[index]
            if max(bw, bh) < min_length * 0.7 or area < min_length * 0.45:
                continue
            py, px = np.nonzero(
                (labels[y : y + bh, x : x + bw] == index)
                & ridge_pixels[y : y + bh, x : x + bw]
            )
            area = len(px)
            px = px + x
            py = py + y
            points = np.c_[px, py].astype(np.float32)
            center = points.mean(0)
            covariance = np.cov((points - center).T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            t = eigenvectors[:, -1].astype(np.float32)
            projection = (points - center) @ t
            extent = float(np.ptp(projection))
            if (
                extent < min_length
                or area / extent > 4.5
                or eigenvalues[-1] < 16 * max(eigenvalues[0], 0.01)
            ):
                continue
            bins = np.floor((projection - projection.min()) / 3).astype(np.int32)
            weights = best[py, px] ** 2
            counts = np.bincount(bins, weights=weights)
            occupied = counts > 0
            if occupied.mean() < 0.8:
                continue
            mx = np.bincount(bins, weights=px * weights) / np.maximum(counts, 0.0001)
            my = np.bincount(bins, weights=py * weights) / np.maximum(counts, 0.0001)
            centers = np.c_[mx[occupied], my[occupied]].astype(np.float32)
            centers = centers[:: max(1, len(centers) // 64)]
            chan = int(np.argmax(np.bincount(channel_map[py, px], minlength=3)))
            if planes[chan] is None:
                planes[chan] = np.ascontiguousarray(src[..., chan])
            amplitudes, widths, sign = _faint_profile(planes[chan], centers, t)
            local_noise = float(np.median(noise_map[py, px]))
            amplitude = float(np.median(amplitudes))
            if (
                len(widths) < len(centers) * 0.8
                or amplitude < 1.5 * local_noise
                or (amplitudes > 0).mean() < 0.9
            ):
                continue
            width = float(np.median(widths))
            if not 0.6 < width < 4.5 or np.ptp(np.percentile(widths, [10, 90])) > max(
                1.5, width * 0.8
            ):
                continue
            if np.median(abs(amplitudes - amplitude)) > amplitude * 0.45:
                continue
            if _parallel_ridge_family(
                candidate,
                normal_x,
                normal_y,
                contrast,
                stats[index, :4],
                center,
                t,
                extent,
                sign,
            ):
                rejected_family += 1
                continue
            ends = []
            along = (centers - center) @ t
            normal = np.array([-t[1], t[0]], np.float32)
            across = (centers - center) @ normal
            curve = np.polyfit(along, across, 2)
            for trace in (centers, centers[::-1]):
                end = trace[0].copy()
                end_u = float((end - center) @ t)
                direction_sign = np.sign((end - trace[-1]) @ t)
                extension = []
                missing = 0
                for distance in range(1, 33):
                    u = end_u + direction_sign * distance
                    probe = center + t * u + normal * np.polyval(curve, u)
                    step = t + normal * np.polyval(np.polyder(curve), u)
                    step /= max(float(np.linalg.norm(step)), 0.001)
                    offsets = np.array([0, -0.5, 0.5, -1, 1], np.float32)
                    accepted = []
                    for offset in offsets:
                        point = probe + normal * offset
                        if not (1 <= point[0] < w - 1 and 1 <= point[1] < h - 1):
                            continue
                        samples = (point + np.arange(-2, 3)[:, None] * step).astype(
                            np.float32
                        )
                        amps, spans, polarity = _faint_profile(planes[chan], samples, step)
                        if (
                            polarity == sign
                            and np.median(amps)
                            >= max(amplitude * 0.45, 1.5 * local_noise)
                            and len(spans)
                            and 0.5 < np.median(spans) < 5
                        ):
                            accepted.append((abs(float(offset)), point))
                    if not accepted:
                        missing += 1
                        if missing >= 3:
                            break
                        continue
                    missing = 0
                    extension.append(min(accepted, key=lambda item: item[0])[1])
                ends.append(np.asarray(extension, np.float32).reshape(-1, 2))
            centers = np.concatenate([ends[0][::-1], centers, ends[1]])
            cv2.polylines(
                protected,
                [np.rint(centers).astype(np.int32)],
                False,
                1,
                int(np.ceil(width)) + 2,
            )
    protected = protected > 0
    components = (
        cv2.connectedComponents(protected.astype(np.uint8), connectivity=8)[0] - 1
    )
    return protected, {
        "faint_stroke_pixels": int(protected.sum()),
        "faint_stroke_components": int(components),
        "faint_family_candidates_rejected": rejected_family,
    }


def _fit_regions(src, core, noiseink, protected=None):
    """@description 선으로 분리된 채색 영역의 강건한 저차 곡면 적합."""
    if _native_fit is not None:
        return _native_fit._fit_regions(src, core, noiseink, protected)
    h, w = src.shape[:2]
    smooth = cv2.GaussianBlur(src, (0, 0), 0.65)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, scale=0.125)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, scale=0.125)
    del smooth
    grad = _rgb_max(np.hypot(gx, gy))
    del gx, gy
    boundary = grad > 3.5
    near_ink = cv2.dilate(core.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    grid_y = np.arange(h, dtype=np.float32)[:, None]
    grid_x = np.arange(w, dtype=np.float32)[None, :]
    for sigma in (2.0, 4.0):
        smooth = cv2.GaussianBlur(src, (0, 0), sigma)
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, scale=0.125)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, scale=0.125)
        del smooth
        magnitude = np.hypot(gx, gy)
        channel = np.argmax(magnitude, axis=2)[..., None]
        peak = _rgb_max(magnitude)
        del magnitude
        nx = np.take_along_axis(gx, channel, axis=2)[..., 0] / np.maximum(peak, 0.001)
        ny = np.take_along_axis(gy, channel, axis=2)[..., 0] / np.maximum(peak, 0.001)
        del gx, gy, channel
        coarse = peak * sigma
        del peak
        ahead = cv2.remap(
            coarse,
            grid_x + nx,
            grid_y + ny,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        behind = cv2.remap(
            coarse,
            grid_x - nx,
            grid_y - ny,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        ridge = (coarse >= ahead) & (coarse >= behind) & ~near_ink
        del nx, ny, ahead, behind
        boundary |= ridge & (coarse > 3.5)
        grad = np.maximum(grad, np.where(ridge, coarse, 0))
        del coarse, ridge
    del grid_x, grid_y, near_ink
    barrier = cv2.dilate(boundary.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    barrier[cv2.dilate(noiseink.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = (
        False
    )
    barrier |= core
    if protected is not None:
        barrier |= protected
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (~barrier).astype(np.uint8), 8
    )
    fitted = np.zeros_like(src)
    accepted = np.zeros((h, w), bool)
    regions = []
    coeff = np.zeros((count, 10, 3), np.float32)
    limits = np.zeros((count, 2, 3), np.float32)
    centers = np.zeros((count, 2), np.float32)
    scales = np.ones((count, 2), np.float32)
    for k in range(1, count):
        x, y, ww, hh, area = stats[k]
        if area < 16:
            continue
        m = labels[y : y + hh, x : x + ww] == k
        d = cv2.distanceTransform(np.pad(m.astype(np.uint8), 1), cv2.DIST_L2, 5)[
            1:-1, 1:-1
        ]
        interior = m & (d >= min(2, float(d.max())))
        data = src[y : y + hh, x : x + ww]
        py, px = np.nonzero(interior)
        if len(px) < 6:
            continue
        step = max(1, len(px) // 12000)
        interior_count = len(px)
        sample_x, sample_y = px[::step].copy(), py[::step].copy()
        del px, py
        center_x, center_y = np.float32(ww / 2), np.float32(hh / 2)
        width, height = np.float32(ww), np.float32(hh)
        xx = (sample_x.astype(np.float32) - center_x) / width
        yy = (sample_y.astype(np.float32) - center_y) / height
        a = _polynomial_basis(xx, yy)
        v = data[sample_y, sample_x]
        c = np.linalg.lstsq(a[:, :3], v, rcond=0.003)[0]
        for _ in range(4):
            r = _rgb_max(abs(v - a[:, :3] @ c))
            weight = np.minimum(1, 3 / np.maximum(r, 1e-5))
            c = np.linalg.lstsq(
                a[:, :3] * weight[:, None], v * weight[:, None], rcond=0.003
            )[0]
        ncoef = 3
        degree = 1
        error = float(
            np.sqrt(np.average(np.mean((v - a[:, :3] @ c) ** 2, 1), weights=weight))
        )
        if min(ww, hh) >= 8 and area / (ww * hh) > 0.12:
            for trial in [6, 10]:
                cc = np.linalg.lstsq(
                    a[:, :trial] * weight[:, None], v * weight[:, None], rcond=0.003
                )[0]
                for _ in range(3):
                    rr = _rgb_max(abs(v - a[:, :trial] @ cc))
                    wt = np.minimum(1, 3 / np.maximum(rr, 1e-5))
                    cc = np.linalg.lstsq(
                        a[:, :trial] * wt[:, None], v * wt[:, None], rcond=0.003
                    )[0]
                ee = float(
                    np.sqrt(
                        np.average(np.mean((v - a[:, :trial] @ cc) ** 2, 1), weights=wt)
                    )
                )
                if error - ee > 0.25:
                    c = cc
                    ncoef = trial
                    degree = 2 if trial == 6 else 3
                    error = ee
        spans = np.array([0, 1, 1, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25])
        bound = np.sum(abs(c.astype(np.float64)) * spans[:ncoef, None], axis=0)
        # 원래 1.2 DN 평탄 판정의 보수적 상계, 부동소수점 오차 여유 포함.
        constant = bound.max() < 1 and abs(c[0]).max() < 512
        if not constant:
            model = np.empty((hh, ww, 3), np.float32)
            xx = (np.arange(ww, dtype=np.float32)[None, :] - center_x) / width
            for top in range(0, hh, 128):
                bottom = min(hh, top + 128)
                yy = (np.arange(top, bottom, dtype=np.float32)[:, None] - center_y) / height
                basis = _polynomial_basis(xx, yy, ncoef)
                model[top:bottom] = basis @ c
            del basis
            percentiles = np.percentile(model[m], [5, 95], axis=0)
            constant = (percentiles[1] - percentiles[0]).max() < 1.2
        if constant:
            median = np.median(v, axis=0)
            degree = 0
            c = median[None, :]
            ncoef = 1
        lowlimit = (np.percentile(v, 1, axis=0) - 2).astype(np.float32)
        highlimit = (np.percentile(v, 99, axis=0) + 2).astype(np.float32)
        if constant:
            model = np.broadcast_to(np.clip(c[0], lowlimit, highlimit), (hh, ww, 3))
        else:
            np.clip(model, lowlimit, highlimit, out=model)
        if not _region_is_consistent(data, model, interior, interior_count):
            del model
            continue
        fitted[y : y + hh, x : x + ww][m] = model[m]
        accepted[y : y + hh, x : x + ww][m] = True
        coeff[k, :ncoef] = c
        centers[k] = [x + ww / 2, y + hh / 2]
        scales[k] = [1 / ww, 1 / hh]
        limits[k] = [lowlimit, highlimit]
        regions.append(
            {
                "label": int(k),
                "area": int(area),
                "degree": degree,
                "box": [int(x), int(y), int(ww), int(hh)],
            }
        )
        del model
    return _FillModels(
        fitted, accepted, labels, coeff, limits, centers, scales, grad, regions
    )


def _render_color_boundaries(src, fit, core, geometry_rgb=None):
    """@description 원본 색 피복률과 접선 평활화를 통한 부픽셀 경계 복원."""
    h, w = src.shape[:2]
    fitted = fit.values
    accepted = fit.selected
    labels = fit.labels
    coeff = fit.coefficients
    limits = fit.limits
    centers = fit.centers
    scales = fit.scales
    count = len(coeff)
    # Extend models to the complete boundary ring, without retaining old AA pixels.
    nearest = ndimage.distance_transform_edt(
        ~accepted, return_distances=False, return_indices=True
    )
    dist_seed = _nearest_distances(nearest)
    filllabels = labels[nearest[0], nearest[1]]
    # 적합 값 버퍼를 출력으로 소비, 원본과 추가 RGB 전체 복사 제외.
    out = fitted
    out[~accepted] = src[~accepted]
    if core.any():
        dist_ink = cv2.distanceTransform(
            (~core).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
    else:
        # 잉크가 없는 경우 SciPy의 영상 밖 기준점 (-1, 0) 유지.
        dist_ink = np.hypot(
            np.arange(h, dtype=np.float32)[:, None] + 1,
            np.arange(w, dtype=np.float32)[None, :],
        )
    # Internal protection gaps with the same fill on every side are not
    # drawing boundaries. Leaving them unchanged creates false dark strokes.
    minimum = cv2.erode(filllabels.astype(np.float32), np.ones((7, 7), np.uint8))
    maximum = cv2.dilate(filllabels.astype(np.float32), np.ones((7, 7), np.uint8))
    holes = (
        (~accepted)
        & (minimum == maximum)
        & (dist_ink > 4)
        & (dist_seed < 6)
    )
    hole_y, hole_x = np.nonzero(holes)
    models = fitted[nearest[0, hole_y, hole_x], nearest[1, hole_y, hole_x]]
    valid_holes = _rgb_max(abs(models - src[hole_y, hole_x])) < np.where(
        _rgb_range(models) < 35, 40, 8
    )
    holes[hole_y, hole_x] = valid_holes
    out[hole_y[valid_holes], hole_x[valid_holes]] = models[valid_holes]
    del nearest, models, minimum, maximum, hole_y, hole_x, valid_holes
    # Labels select neighboring colors only. Coverage and edge position come
    # from the original samples, not from the discrete segmentation boundary.
    color_edge = np.zeros((h, w), bool)
    pair_a = np.zeros((h, w), np.int32)
    pair_b = np.zeros((h, w), np.int32)
    for dy, dx in [(1, 0), (0, 1)]:
        first = filllabels[: h - dy, : w - dx]
        second = filllabels[dy:, dx:]
        edge = first != second
        ca = np.minimum(first, second)
        cb = np.maximum(first, second)
        target = color_edge[: h - dy, : w - dx]
        target |= edge
        pair_a[: h - dy, : w - dx][edge] = ca[edge]
        pair_b[: h - dy, : w - dx][edge] = cb[edge]
    near_color = ndimage.distance_transform_edt(
        ~color_edge, return_distances=False, return_indices=True
    )
    del color_edge
    close_color = _nearest_distances(near_color) < 15
    pa = pair_a[near_color[0], near_color[1]]
    del pair_a
    pb = pair_b[near_color[0], near_color[1]]
    del pair_b, near_color
    quadratic_models = np.any(coeff[:, 3:6] != 0, axis=(1, 2))
    cubic_models = np.any(coeff[:, 6:10] != 0, axis=(1, 2))

    def _evaluate_coordinates(region_ids, x, y):
        """@description 주어진 정규 좌표의 채색 곡면 평가."""
        value = (
            coeff[region_ids, 0]
            + x[..., None] * coeff[region_ids, 1]
            + y[..., None] * coeff[region_ids, 2]
        )
        selected = quadratic_models[region_ids] | np.any(
            (value == 0) & np.signbit(value), axis=-1
        )
        ids_selected = region_ids[selected]
        sx, sy = x[selected, None], y[selected, None]
        value[selected] += (
            sx ** 2 * coeff[ids_selected, 3]
            + (sx * sy) * coeff[ids_selected, 4]
            + sy ** 2 * coeff[ids_selected, 5]
        )
        selected = cubic_models[region_ids] | np.any(
            (value == 0) & np.signbit(value), axis=-1
        )
        ids_selected = region_ids[selected]
        sx, sy = x[selected, None], y[selected, None]
        value[selected] += (
            sx ** 3 * coeff[ids_selected, 6]
            + (sx * sx * sy) * coeff[ids_selected, 7]
            + (sx * sy * sy) * coeff[ids_selected, 8]
            + sy ** 3 * coeff[ids_selected, 9]
        )
        return np.clip(value, limits[region_ids, 0], limits[region_ids, 1])

    def _evaluate_models(ids):
        """@description 연산 순서를 유지하는 행 묶음별 색 곡면 평가."""
        values = np.empty_like(src)
        xx = np.arange(w, dtype=np.float32)[None, :]

        def _evaluate_rows(top):
            """@description 서로 겹치지 않는 행 묶음의 채색 모델 평가."""
            bottom = min(h, top + 128)
            region_ids = ids[top:bottom]
            yy = np.arange(top, bottom, dtype=np.float32)[:, None]
            x = (xx - centers[region_ids, 0]) * scales[region_ids, 0]
            y = (yy - centers[region_ids, 1]) * scales[region_ids, 1]
            values[top:bottom] = _evaluate_coordinates(region_ids, x, y)

        with _executor(max_workers=min(4, os.cpu_count() or 1)) as executor:
            tuple(executor.map(_evaluate_rows, range(0, h, 128)))
        return values

    def _evaluate_points(ids, yy, xx):
        """@description 경계 주변에 필요한 표본의 채색 모델 평가."""
        values = np.empty((len(ids), 3), np.float32)

        def _evaluate_block(start):
            stop = min(len(ids), start + 131072)
            region_ids = ids[start:stop]
            x = (xx[start:stop].astype(np.float32) - centers[region_ids, 0]) * scales[region_ids, 0]
            y = (yy[start:stop].astype(np.float32) - centers[region_ids, 1]) * scales[region_ids, 1]
            values[start:stop] = _evaluate_coordinates(region_ids, x, y)

        with _executor(max_workers=min(4, os.cpu_count() or 1)) as executor:
            tuple(executor.map(_evaluate_block, range(0, len(ids), 131072)))
        return values

    # 0.8px 가우스 필터와 Sobel의 4px 문맥을 포함한 경계 띠만 평가.
    band = cv2.dilate(close_color.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    band_y, band_x = np.nonzero(band)
    ids_a, ids_b = pa[band_y, band_x], pb[band_y, band_x]
    del pa, pb, band
    fa = _evaluate_points(ids_a, band_y, band_x)
    fb = _evaluate_points(ids_b, band_y, band_x)
    vec = fb - fa
    del fb
    contrast = np.sum(vec * vec, axis=-1)
    geometry = src if geometry_rgb is None else geometry_rgb
    geometry_band = geometry[band_y, band_x]
    observed_alpha = np.sum((geometry_band - fa) * vec, axis=-1) / np.maximum(contrast, 1)
    alpha = np.zeros((h, w), np.float32)
    alpha[band_y, band_x] = observed_alpha
    orthogonal = _rgb_max(abs(geometry_band - (fa + observed_alpha[..., None] * vec)))
    color_aa = np.zeros((h, w), bool)
    lowcontrast = _rgb_max(abs(vec)) < 25
    endpoint_error = _rgb_max(
        abs(geometry_band - (fa + np.clip(observed_alpha, 0, 1)[..., None] * vec))
    )
    near = (
        close_color[band_y, band_x]
        & (dist_ink[band_y, band_x] > 3)
        & (contrast > 0.5)
        & (orthogonal < 10)
        & (endpoint_error < np.where(lowcontrast, 35, 15))
        & ((observed_alpha > -0.15) & (observed_alpha < 1.15) | lowcontrast)
        & ((filllabels[band_y, band_x] == ids_a) | (filllabels[band_y, band_x] == ids_b))
    )
    pair_ys, pair_xs = band_y[near], band_x[near]
    pair_ids = ids_a[near].astype(np.int64) * count + ids_b[near]
    fa, vec = fa[near], vec[near]
    del (
        contrast, orthogonal, lowcontrast, endpoint_error, close_color,
        band_y, band_x, geometry_band, observed_alpha, near, ids_a, ids_b,
    )
    order = np.argsort(pair_ids, kind="stable")
    pair_ids = pair_ids[order]
    pair_ys = pair_ys[order]
    pair_xs = pair_xs[order]
    fa, vec = fa[order], vec[order]
    del order
    purefield = _evaluate_models(filllabels)
    offsets = np.r_[0, np.flatnonzero(np.diff(pair_ids)) + 1, len(pair_ids)]

    def _render_pair(start, stop, executor):
        """@description 한 색 쌍의 경계를 문맥이 겹치는 작은 타일로 렌더링."""
        ys, xs = pair_ys[start:stop], pair_xs[start:stop]
        x0 = max(0, xs.min() - 18)
        x1 = min(w, xs.max() + 19)
        y0 = max(0, ys.min() - 18)
        y1 = min(h, ys.max() + 19)
        m = np.zeros((y1 - y0, x1 - x0), bool)
        local_y, local_x = ys - y0, xs - x0
        m[local_y, local_x] = True
        tiled = m.size > 256 * 256
        if tiled:
            columns = (m.shape[1] + 127) // 128
            tile_ids = (local_y // 128) * columns + local_x // 128
            tile_order = np.argsort(tile_ids, kind="stable")
            tile_ids = tile_ids[tile_order]
            breaks = np.r_[0, np.flatnonzero(np.diff(tile_ids)) + 1, len(tile_ids)]
            jobs = [tile_order[a:b] for a, b in zip(breaks[:-1], breaks[1:])]
        else:
            jobs = [np.arange(len(xs))]

        def _render_tile(selected):
            """@description 8픽셀 문맥과 원래 보간 좌표를 유지하는 타일 계산."""
            if tiled:
                tile_x = int(local_x[selected[0]] // 128) * 128
                tile_y = int(local_y[selected[0]] // 128) * 128
                left, top = max(0, tile_x - 8), max(0, tile_y - 8)
                right = min(m.shape[1], tile_x + 128 + 8)
                bottom = min(m.shape[0], tile_y + 128 + 8)
            else:
                left, top, right, bottom = 0, 0, m.shape[1], m.shape[0]
            sl = np.s_[y0 + top:y0 + bottom, x0 + left:x0 + right]
            al = np.clip(alpha[sl], 0, 1).astype(np.float32)
            support = m[top:bottom, left:right].astype(np.float32)
            coverage, mass, total_weight = _boundary_coverage(al, support, left, top)
            yy, xx = ys[selected], xs[selected]
            my, mx = local_y[selected] - top, local_x[selected] - left
            indices = start + selected
            original_coverage = np.clip(alpha[yy, xx], 0, 1)
            local_vec, local_fa = vec[indices], fa[indices]
            coverage_limit = 12 / np.maximum(_rgb_max(abs(local_vec)), 1)
            coverage = original_coverage + np.clip(
                coverage[my, mx] - original_coverage, -coverage_limit, coverage_limit
            )
            rendered = local_fa + coverage[..., None] * local_vec
            confidence = np.clip((mass[my, mx] / total_weight - 0.85) / 0.15, 0, 1)
            observed = geometry[yy, xx]
            rendered = observed + confidence[..., None] * (rendered - observed)
            edge_evidence = np.clip((fit.gradient[yy, xx] - 1.5) / 2, 0, 1) ** 2
            pure = purefield[yy, xx]
            interior = pure + edge_evidence[..., None] * (rendered - pure)
            rendered = np.where(accepted[yy, xx, None], interior, rendered)
            return yy, xx, rendered

        for yy, xx, rendered in executor.map(_render_tile, jobs):
            out[yy, xx] = rendered
            color_aa[yy, xx] = True

    with _executor(max_workers=min(4, os.cpu_count() or 1)) as executor:
        for start, stop in zip(offsets[:-1], offsets[1:]):
            if stop - start >= 8:
                _render_pair(start, stop, executor)
    report = {
        "color_aa_pixels": int(color_aa.sum()),
        "internal_gap_pixels": int(holes.sum()),
        "color_aa_tangent_sigma": 1.3,
    }
    return _BoundaryModels(out, color_aa, purefield, dist_ink, dist_seed, report)


def _protected_details(src):
    """@description 하이라이트, 눈과 반복 무늬를 가진 금속 잠금장치 보호."""
    h, w = src.shape[:2]
    # Exact highlights, eyes and zipper texture, not the former broad gray mask.
    olddetail = None
    warm = (
        (src[..., 0] > 245)
        & (src[..., 1] > 222)
        & (src[..., 2] > 200)
        & (src[..., 0] - src[..., 2] > 12)
    )
    detail = cv2.dilate(warm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    white = (_rgb_min(src) > 218) & (_rgb_range(src) < 28)
    n, ll, ss, _ = cv2.connectedComponentsWithStats(white.astype(np.uint8), 8)
    for k in range(1, n):
        x, y, ww, hh, area = ss[k]
        if (
            20 <= area <= 9000
            and ww > hh * 1.3
            and hh < h * 0.09
            and ww < w * 0.27
            and x > 0
            and y > 0
        ):
            detail[
                max(0, y - 3) : min(h, y + hh + 3), max(0, x - 3) : min(w, x + ww + 3)
            ] = True
    # Dense narrow neutral texture components identify metal fasteners.
    neutral = (_rgb_min(src) > 90) & (_rgb_max(src) < 220) & (_rgb_range(src) < 40)
    n, ll, ss, _ = cv2.connectedComponentsWithStats(neutral.astype(np.uint8), 8)
    for k in range(1, n):
        x, y, ww, hh, area = ss[k]
        if area < 1000 or not (
            30 < ww < w * 0.1 and ww * 1.5 < hh < min(h * 0.3, ww * 4)
        ):
            continue
        holes = sum(
            20 < a < 250 and x <= xx < x + ww and y <= yy < y + hh
            for xx, yy, wx, hy, a in ss[1:]
        )
        if holes >= 2:
            if olddetail is None:
                olddetail = _detail_context(src)
            y0 = max(0, y - 15)
            y1 = min(h, y + hh * 3)
            x0 = max(0, x - ww)
            x1 = min(w, x + ww * 2)
            detail[y0:y1, x0:x1] |= olddetail[y0:y1, x0:x1]
    return detail


def _render_ink_profiles(src, out, core, detail, color_aa, ink_support):
    """@description 잉크와 채색 모델 사이의 일관된 법선 피복 곡선 복원."""
    if _native_ink is not None:
        return _native_ink.render_ink_profiles(src, out, core, detail, color_aa, ink_support)
    h, w = src.shape[:2]
    # Model each observed contour as two smooth colors and one smooth normal
    # coverage curve. Original samples estimate geometry, never reconstructed AA.
    luma = src @ _LUMA
    edges = cv2.Canny(
        np.uint8(np.clip(np.rint(cv2.GaussianBlur(luma, (0, 0), 0.7)), 0, 255)), 8, 22
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    depths = np.arange(-12, 12.01, 0.5, dtype=np.float32)
    total = np.zeros_like(src)
    mass = np.zeros((h, w), np.float32)
    eligible_pixels = ~core & ~detail & ~color_aa & ink_support
    for contour in contours:
        raw = contour[:, 0].astype(np.float32)
        if len(raw) < 16:
            continue
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
        samples = np.arange(arc[-1], dtype=np.float32)
        if len(samples) < 16:
            continue
        points = np.column_stack(
            [
                ndimage.gaussian_filter1d(
                    np.interp(samples, arc, raw[:, c]), 1.4, mode="nearest"
                )
                for c in range(2)
            ]
        ).astype(np.float32)
        tangent = np.gradient(points, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-4)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]]).astype(np.float32)
        curvature = np.linalg.norm(np.gradient(tangent, axis=0), axis=1)
        geometry_good = curvature < 0.12
        geometry_good[:5] = False
        geometry_good[-5:] = False
        if not geometry_good.any():
            continue
        x0, y0 = np.maximum(np.floor(points.min(0)).astype(int) - 13, 0)
        x1, y1 = np.minimum(np.ceil(points.max(0)).astype(int) + 14, [w, h])
        tube = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.polylines(tube, [np.rint(points - [x0, y0]).astype(np.int32)], False, 1, 25)
        tube &= eligible_pixels[y0:y1, x0:x1]
        if not tube.any():
            continue
        coordinates = points[:, None, :] + normal[:, None, :] * depths[None, :, None]
        prof = cv2.remap(
            src,
            coordinates[..., 0],
            coordinates[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mod = cv2.remap(
            out,
            coordinates[..., 0],
            coordinates[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        ii = np.arange(len(points))
        center = len(depths) // 2
        direction = prof[:, center + 2] - prof[:, center - 2]
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1)
        projection = np.sum(prof * direction[:, None, :], axis=2)
        derivative = np.gradient(
            ndimage.gaussian_filter1d(projection, 1, axis=1), axis=1
        )
        peak = np.max(derivative[:, center - 2 : center + 3], axis=1)
        leftstop = derivative[:, : center - 2] < peak[:, None] * 0.15
        rightstop = derivative[:, center + 3 :] < peak[:, None] * 0.15
        imin = np.where(
            leftstop.any(1), center - 3 - np.argmax(leftstop[:, ::-1], axis=1), 0
        )
        imax = np.where(
            rightstop.any(1), center + 3 + np.argmax(rightstop, axis=1), len(depths) - 1
        )
        low = mod[ii, imin]
        high = mod[ii, imax]
        low = ndimage.median_filter(low, size=(9, 1), mode="nearest")
        high = ndimage.median_filter(high, size=(9, 1), mode="nearest")
        v = high - low
        contrast = np.sum(v * v, axis=1)
        if not np.any(geometry_good & (contrast > 20**2)):
            continue
        alpha = np.sum((prof - low[:, None, :]) * v[:, None, :], axis=2) / np.maximum(
            contrast[:, None], 1
        )
        dmin = np.minimum(depths[imin], depths[imax])
        dmax = np.maximum(depths[imin], depths[imax])
        valid = (
            (alpha > 0.07)
            & (alpha < 0.93)
            & (abs(depths[None, :]) < 8.25)
            & (depths[None, :] >= dmin[:, None])
            & (depths[None, :] <= dmax[:, None])
        )
        zz = np.zeros_like(alpha)
        zz[valid] = ndtri(alpha[valid])
        n = valid.sum(1)
        s0 = np.maximum(n, 1)
        sz = (zz * valid).sum(1)
        sd = (depths[None, :] * valid).sum(1)
        szz = (zz * zz * valid).sum(1)
        szd = (zz * depths[None, :] * valid).sum(1)
        width = (szd - sz * sd / s0) / np.maximum(szz - sz * sz / s0, 1e-5)
        offset = (sd - width * sz) / s0
        good = (
            (n >= 3)
            & (contrast > 20**2)
            & (abs(width) > 0.15)
            & (abs(width) < 8)
            & (abs(offset) < 4)
        )
        endpoint_change = np.maximum(
            _rgb_max(abs(np.gradient(low, axis=0))),
            _rgb_max(abs(np.gradient(high, axis=0))),
        )
        good &= geometry_good & (endpoint_change < 5)
        if not good.any():
            continue
        sign = np.where(width < 0, -1.0, 1.0)
        # Filter only along the contour, preserving the fitted cross-section.
        width = (
            ndimage.gaussian_filter1d(np.clip(abs(width), 0.3, 8), 4, mode="nearest")
            * sign
        )
        offset = ndimage.gaussian_filter1d(np.clip(offset, -4, 4), 4, mode="nearest")
        checked_depths = depths[13:36]
        rebuilt = (
            low[:, None, :]
            + ndtr(
                (checked_depths[None, :, None] - offset[:, None, None]) / width[:, None, None]
            )
            * v[:, None, :]
        )
        err = _rgb_max(abs(rebuilt - prof[:, 13:36]))
        checked = valid[:, 13:36]
        # 원래 49개 깊이 위치와 쌍별 합산 순서 유지.
        squared_error = np.zeros(valid.shape, err.dtype)
        squared_error[:, 13:36] = err * err * checked
        fitrms = np.sqrt(
            np.sum(squared_error, axis=1) / np.maximum(checked.sum(1), 1)
        )
        good &= fitrms < 6
        if not good.any():
            continue
        observed_width = width.copy()
        yy, xx = np.nonzero(tube)
        xx += x0
        yy += y0
        distance, index = cKDTree(points).query(np.c_[xx, yy])
        depall = np.sum((np.c_[xx, yy] - points[index]) * normal[index], axis=1)
        use = (
            good[index]
            & (distance < 12)
            & (depall > dmin[index] - 0.25)
            & (depall < dmax[index] + 0.25)
        )
        xx = xx[use]
        yy = yy[use]
        index = index[use]
        distance = distance[use]
        dep = np.sum((np.c_[xx, yy] - points[index]) * normal[index], axis=1)
        observed = (
            low[index]
            + ndtr((dep - offset[index]) / observed_width[index])[:, None] * v[index]
        )
        valid_pixel = _rgb_max(abs(src[yy, xx] - observed)) < 10
        xx = xx[valid_pixel]
        yy = yy[valid_pixel]
        index = index[valid_pixel]
        dep = dep[valid_pixel]
        distance = distance[valid_pixel]
        coverage = ndtr((dep - offset[index]) / width[index])
        value = low[index] + coverage[:, None] * v[index]
        # Restrict a side to its measured half-profile when the opposite edge of
        # a narrow line is also present in the same normal sampling window.
        confidence = np.maximum(0.05, 12 - distance)
        total[yy, xx] += value * confidence[:, None]
        mass[yy, xx] += confidence
    profile_aa = mass > 0
    out[profile_aa] = total[profile_aa] / mass[profile_aa, None]
    return out, profile_aa


def _restore(src, geometry_rgb=None):
    """@description 채색 곡면, 색 경계와 잉크 경계의 순차 복원."""
    started = time.perf_counter()
    src = np.asarray(src, np.float32)
    core, noiseink, ink_support = _ink_masks(src)
    faint, faint_report = _faint_strokes(src, core)
    fit = _fit_regions(src, core, noiseink, protected=faint)
    if not fit.selected.any():
        zero = np.zeros(src.shape[:2], bool)
        report = {
            "regions": [],
            "runtime_seconds": time.perf_counter() - started,
            "core_pixels": int(core.sum()),
            "core_changed": 0,
            "detail_pixels": int(faint.sum()),
            "aa_pixels": 0,
            "aa_delta_percentiles": [0.0] * 5,
            "changed_pixels": 0,
            "max_delta_dn": 0.0,
            **faint_report,
        }
        return (
            src.copy(),
            report,
            {
                "core": core,
                "detail": faint,
                "faint": faint,
                "aa": zero,
                "accepted": zero,
                "labels": fit.labels,
                "color_aa": zero,
            },
        )
    boundary = _render_color_boundaries(src, fit, core, geometry_rgb)
    accepted, labels, regions = fit.selected, fit.labels, fit.regions
    del fit
    detail = _protected_details(src) | faint
    out, profile_aa = _render_ink_profiles(
        src, boundary.output, core, detail, boundary.color_mask, ink_support
    )
    aa = profile_aa | boundary.color_mask
    out[core | detail] = src[core | detail]
    dist_seed = boundary.seed_distance
    dist_ink = boundary.ink_distance
    purefield = boundary.pure_field
    # Reconstruct low-contrast boundary gaps without absorbing thin highlights.
    completion = (
        (~accepted)
        & ~aa
        & ~core
        & ~detail
        & ~ink_support
        & (dist_seed < 6)
        & (_rgb_max(abs(purefield - src)) < 35)
        & (_rgb_range(purefield) < 35)
        & (_rgb_range(src) < 35)
    )
    if completion.any():
        completion_y, completion_x = np.nonzero(completion)
        lower = np.empty((len(completion_y), 3), np.float32)
        upper = np.empty_like(lower)
        for channel in range(3):
            local_low = cv2.erode(purefield[..., channel], np.ones((7, 7), np.uint8))
            local_high = cv2.dilate(purefield[..., channel], np.ones((7, 7), np.uint8))
            lower[:, channel] = local_low[completion_y, completion_x]
            upper[:, channel] = local_high[completion_y, completion_x]
            del local_low, local_high
        valid_completion = _rgb_max(upper - lower) < 30
        completion[completion_y, completion_x] = valid_completion
        completion_y = completion_y[valid_completion]
        completion_x = completion_x[valid_completion]
        lower, upper = lower[valid_completion], upper[valid_completion]
    if completion.any():
        for channel in range(3):
            filled = cv2.inpaint(
                out[..., channel], completion.astype(np.uint8), 3, cv2.INPAINT_NS
            )
            out[completion_y, completion_x, channel] = np.clip(
                filled[completion_y, completion_x], lower[:, channel], upper[:, channel]
            )
            del filled
        aa |= completion
    # Complete isolated unmodeled samples, which otherwise become dark dots
    # when their already reconstructed neighbors reveal the old residual.
    isolated = (
        (~accepted)
        & ~aa
        & ~core
        & ~detail
        & (dist_ink > 4)
        & (dist_seed < 6)
        & (_rgb_max(abs(purefield - src)) < 80)
        & (_rgb_max(abs(out - purefield)) > 10)
        & (_rgb_range(purefield) < 35)
        & (_rgb_range(src) < 35)
    )
    nc, ll, ss, _ = cv2.connectedComponentsWithStats(isolated.astype(np.uint8), 8)
    small = ss[:, 4] <= 6
    small[0] = False
    replace = small[ll]
    out[replace] = purefield[replace]
    speck_pixels = int(replace.sum())
    np.clip(out, 0, 255, out=out)
    out[core | detail] = src[core | detail]
    delta = _rgb_max(abs(out - src))
    report = {
        "regions": regions,
        "runtime_seconds": time.perf_counter() - started,
        "core_pixels": int(core.sum()),
        "core_changed": int(np.any(out[core] != src[core], axis=1).sum()),
        "detail_pixels": int(detail.sum()),
        "aa_pixels": int(aa.sum()),
        "aa_delta_percentiles": np.percentile(delta[aa], [50, 90, 95, 99, 100]).tolist()
        if aa.any()
        else [0.0] * 5,
        "changed_pixels": int((delta > 0).sum()),
        "max_delta_dn": float(delta.max()),
        "isolated_completed_pixels": speck_pixels,
        "boundary_completed_pixels": int(completion.sum()),
        **boundary.report,
        **faint_report,
    }
    return (
        out,
        report,
        {
            "core": core,
            "detail": detail,
            "faint": faint,
            "aa": aa,
            "accepted": accepted,
            "labels": labels,
            "color_aa": boundary.color_mask,
        },
    )


def _preserve_large_changes(source, proposed):
    """@description 고대비 변경 성분과 주변 세부의 원래 색 보존."""
    delta = _rgb_max(abs(proposed - source))
    trigger = delta > 24
    guarded = np.zeros(source.shape[:2], bool)
    report = {
        "safety_cap_dn": 24.0,
        "safety_trigger_pixels": int(trigger.sum()),
        "safety_reverted_pixels": 0,
        "safety_guarded_components": 0,
        "safety_bounded_components": 0,
        "safety_bounded_pixels": 0,
    }
    if not trigger.any():
        return proposed, guarded, report
    candidates = (delta > 12).astype(np.uint8)
    connected = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    connected |= candidates
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        connected, connectivity=8
    )
    selected = np.zeros(count, bool)
    selected[np.unique(labels[trigger])] = True
    selected[0] = False
    bounded = selected & (stats[:, cv2.CC_STAT_AREA] < 8)
    selected &= ~bounded
    if bounded.any():
        # Bound tiny mismatches without exposing source-colored islands.
        maxima = np.zeros(count, np.float32)
        for label in np.flatnonzero(bounded):
            x, y, width, height = stats[label, :4]
            local_labels = labels[y:y + height, x:x + width]
            local_delta = delta[y:y + height, x:x + width]
            maxima[label] = local_delta[local_labels == label].max()
        scales = np.ones(count, np.float32)
        scales[bounded] = 23.999 / maxima[bounded]
        proposed = source + (proposed - source) * scales[labels, None]
        report.update(
            safety_bounded_components=int(bounded.sum()),
            safety_bounded_pixels=int(bounded[labels].sum()),
        )
    guarded = (
        cv2.dilate(selected[labels].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    )
    guarded = (
        cv2.morphologyEx(
            guarded.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )
        > 0
    )
    guarded = ndimage.binary_fill_holes(guarded)
    distance = cv2.distanceTransform((~guarded).astype(np.uint8), cv2.DIST_L2, 5)
    confidence = np.clip(distance / 3, 0, 1)
    output = source + (proposed - source) * confidence[..., None]
    output[guarded] = source[guarded]
    report.update(
        safety_reverted_pixels=int(np.count_nonzero(guarded & (delta > 0))),
        safety_guarded_components=int(selected.sum()),
    )
    return output.astype(np.float32, copy=False), guarded, report


def apply(
    rgb: np.ndarray,
    strength: float = 1.0,
    *,
    context_rgb: np.ndarray | None = None,
    corrected_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """선화 문맥에서 채색 면과 경계 복원, 잉크 중심과 세부 유지.

    corrected_mask는 실제 변경 픽셀로 덮어쓴다. context_rgb는 선화 판정과
    원본 경계 피복률에 사용하며, 채색 모델에는 현재 단계의 RGB를 사용한다.
    """
    if strength < 0 or not math.isfinite(strength):
        raise ValueError("cel_fill stage strength must be finite and 0 or greater.")
    if corrected_mask is not None:
        corrected_mask.fill(False)
    source = np.asarray(rgb, np.float32)
    report = {
        "enabled": bool(strength),
        "strength": float(strength),
        "method": "cel_models_and_subpixel_coverage",
        "illustration_context": False,
        "eligible_fraction": 0.0,
        "changed_pixels": 0,
        "mean_abs_change": [0.0, 0.0, 0.0],
        "max_abs_change": 0.0,
    }
    if strength == 0:
        return source.copy(), report
    context = source if context_rgb is None else np.asarray(context_rgb, np.float32)
    enabled, flat = _line_art_context(context, context @ _LUMA)
    report.update(illustration_context=bool(enabled), flat_region_fraction=float(flat))
    if not enabled:
        return source.copy(), report
    restored, metrics, masks = _restore(source, geometry_rgb=context)
    np.subtract(restored, source, out=restored)
    restored *= np.float32(strength)
    restored += source
    np.clip(restored, 0, 255, out=restored)
    output = restored
    output, safety_mask, safety_report = _preserve_large_changes(source, output)
    output[masks["core"] | masks["detail"]] = source[masks["core"] | masks["detail"]]
    changed = np.any(output != source, axis=2)
    if corrected_mask is not None:
        corrected_mask[...] = changed & ~masks["core"] & ~masks["detail"]
    delta = np.abs(output - source)
    report.update(metrics)
    report.update(safety_report)
    report.update(
        changed_pixels=int(changed.sum()),
        eligible_fraction=float(
            np.mean((masks["accepted"] | masks["aa"]) & ~safety_mask)
        ),
        model_degree_counts={
            str(d): sum(r["degree"] == d for r in metrics["regions"]) for d in range(4)
        },
        ink_core_pixels=int(masks["core"].sum()),
        ink_core_changed_pixels=int((changed & masks["core"]).sum()),
        detail_protected_pixels=int(masks["detail"].sum()),
        detail_changed_pixels=int((changed & masks["detail"]).sum()),
        mean_abs_change=np.mean(delta, axis=(0, 1)).tolist(),
        max_abs_change=float(delta.max()),
        max_delta_dn=float(delta.max()),
        aa_delta_percentiles=np.percentile(
            _rgb_max(delta)[masks["aa"]], [50, 90, 95, 99, 100]
        ).tolist()
        if masks["aa"].any()
        else [0.0] * 5,
    )
    return output, report


def analyze(rgb: np.ndarray, *, context_rgb: np.ndarray | None = None) -> dict:
    """@description 선화 문맥과 채색 모델 복원 범위 분석."""
    return apply(rgb, context_rgb=context_rgb)[1]
