"""문자 행의 공통 테두리를 유지하는 획 인접 배경 복원."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

from ._native import load_native


_native = load_native("_native_text")


def _maximum_channel(rgb):
    """고정된 RGB 세 채널의 최댓값을 배열 연산으로 계산."""
    return np.maximum(np.maximum(rgb[..., 0], rgb[..., 1]), rgb[..., 2])


def _append_missing_rows(rows, candidates):
    """기존 문자 행과 겹치지 않는 복구 후보만 추가."""
    for row in candidates:
        if any(min(row[3], old[3]) - max(row[1], old[1]) >= 0.7 * min(row[3] - row[1], old[3] - old[1])
               and max(row[0], old[0]) < min(row[2], old[2]) for old in rows):
            continue
        rows.append(row)


def _has_diverse_glyph_shapes(shapes):
    """정규화 문자 모양의 반복 군집으로 행 다양성 판정.

    0~1 입력의 임계값 인접 쌍은 기존 정밀도로 재계산. 유효 모양 수가 기준
    미만인 순간 제외.
    """
    shapes = np.asarray(shapes)
    if _native is not None:
        return _native.has_diverse_glyph_shapes(shapes)
    count = len(shapes)
    size = shapes.shape[1] * shapes.shape[2]
    first, second = np.triu_indices(count, 1)
    differences = pdist(shapes.reshape(count, size), metric="cityblock") / size
    threshold = float(np.asarray(0.08, dtype=shapes.dtype))
    similar = differences < threshold
    error_bound = np.finfo(shapes.dtype).eps * (2 * size + 1)
    for index in np.flatnonzero(np.abs(differences - threshold) <= error_bound):
        similar[index] = np.mean(np.abs(shapes[first[index]] - shapes[second[index]])) < 0.08
    parents = list(range(count))
    sizes = [1] * count
    squared_sizes = count
    minimum = min(3.5, count)
    for left, right in zip(first[similar], second[similar]):
        while parents[left] != left:
            left = parents[left]
        while parents[right] != right:
            right = parents[right]
        if left == right:
            continue
        if sizes[left] < sizes[right]:
            left, right = right, left
        squared_sizes += 2 * sizes[left] * sizes[right]
        sizes[left] += sizes[right]
        parents[right] = left
        if count ** 2 / squared_sizes < minimum:
            return False
    return True


def _rows_at_scale(rgb, scale, missing_rows=None):
    """높이, 기준선, 획 두께와 잉크색이 반복되는 수평 문자 행 검출."""
    size = (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale)))
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    gray = _maximum_channel(small)
    median = cv2.medianBlur(np.clip(np.rint(gray), 0, 255).astype(np.uint8), 31)
    rows = []
    recovered_rows = []
    contrast = gray - median
    seed = (contrast > 24).astype(np.uint8)
    _, labels, stats, centers = cv2.connectedComponentsWithStats(seed, 8)
    x, y, width, height, area = stats.T
    valid = ((height >= 7) & (height <= 180) & (width >= 2)
             & (width <= 2.8 * height) & (area >= 12)
             & (area >= 0.12 * width * height) & (area <= 0.88 * width * height))
    valid[0] = False
    ids = np.flatnonzero(valid)
    if len(ids) < 3:
        return rows
    distance = cv2.distanceTransform(seed, cv2.DIST_L2, 5)
    ridge = (distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8))) & (distance > 0)
    if _native is not None:
        stroke_maxima, color_maxima = _native.glyph_maxima(
            labels, valid.view(np.uint8), distance, small, len(stats))
    else:
        stroke_maxima = np.full(len(stats), -np.inf, distance.dtype)
        color_maxima = np.full((len(stats), 3), -np.inf, small.dtype)
        selected = valid[labels]
        selected_labels = labels[selected]
        selected_pixels = small[selected]
        np.maximum.at(stroke_maxima, selected_labels, distance[selected])
        for channel in range(3):
            np.maximum.at(color_maxima[:, channel], selected_labels, selected_pixels[:, channel])
        del selected, selected_labels, selected_pixels
    stroke, colors = stroke_maxima[ids], color_maxima[ids]
    tree = cKDTree(centers[ids])
    used = set()
    rejected_groups = set()
    glyph_shapes = {}
    row_statistics = {}
    diverse_groups = set()
    membership = np.zeros(len(stats), bool)
    for i in np.argsort(-height[ids]):
        label = ids[i]
        if int(label) in used:
            continue
        h = height[label]
        nearby = np.asarray(tree.query_ball_point(centers[label], 16 * h), int)
        if _native is not None:
            members = _native.aligned_members(stats, ids, nearby, colors, stroke, i,
                                              0.6 * stroke[i], 1.65 * stroke[i])
        else:
            candidate = ids[nearby]
            aligned = ((height[candidate] >= 0.6 * h) & (height[candidate] <= 1.5 * h)
                       & (np.abs((y[candidate] + height[candidate]) - (y[label] + h)) <= 0.36 * h)
                       & (_maximum_channel(np.abs(colors[nearby] - colors[i])) < 22)
                       & (stroke[nearby] >= 0.6 * stroke[i]) & (stroke[nearby] <= 1.65 * stroke[i]))
            members = candidate[aligned]
        members = members[np.argsort(x[members])]
        minimum = 3 if h >= 45 else 4
        if len(members) < minimum:
            continue
        splits = np.flatnonzero(x[members[1:]] - (x[members[:-1]] + width[members[:-1]]) > 1.7 * h) + 1
        groups = np.split(members, splits)
        for group in groups:
            if len(group) < minimum or label not in group:
                continue
            group_key = tuple(group)
            if group_key in rejected_groups:
                continue
            left, right = np.min(x[group]), np.max(x[group] + width[group])
            if right - left < 3.5 * h:
                continue
            statistics_key = (group_key, h >= 45)
            if statistics_key in row_statistics:
                statistics = row_statistics[statistics_key]
                if statistics is None:
                    continue
                heights, recovered = statistics
            else:
                heights = height[group]
                tolerance = 0.27 if h >= 45 else 0.2
                baselines = y[group] + heights
                baseline_tolerance = 0.2 if h >= 45 else 0.12
                recovered = (np.std(heights) > tolerance * np.mean(heights)
                             or np.std(baselines) > baseline_tolerance * np.mean(heights))
                if recovered:
                    tall = heights >= 0.75 * np.max(heights)
                    if np.count_nonzero(tall) < minimum:
                        row_statistics[statistics_key] = None
                        continue
                    heights, baselines = heights[tall], baselines[tall]
                    if (np.std(heights) > tolerance * np.mean(heights)
                            or np.std(baselines) > baseline_tolerance * np.mean(heights)):
                        row_statistics[statistics_key] = None
                        continue
                stroke_widths = stroke[np.searchsorted(ids, group)]
                if np.std(stroke_widths) > 0.22 * np.mean(stroke_widths):
                    row_statistics[statistics_key] = None
                    continue
                row_statistics[statistics_key] = heights, recovered
            if group_key not in diverse_groups:
                shapes = []
                for member in group:
                    if member not in glyph_shapes:
                        component = labels[y[member]:y[member] + height[member],
                                           x[member]:x[member] + width[member]] == member
                        glyph_shapes[member] = cv2.resize(component.astype(np.float32), (12, 16),
                                                         interpolation=cv2.INTER_AREA)
                    shapes.append(glyph_shapes[member])
                if not _has_diverse_glyph_shapes(shapes):
                    rejected_groups.add(group_key)
                    continue
                diverse_groups.add(group_key)
            # 동일 색 잉크가 배경과 분리된 급격한 전이를 갖는지 확인.
            top, bottom = np.min(y[group]), np.max(y[group] + height[group])
            # 전체 마스크와 영상 경계 조건이 같은 1픽셀 침식 여백.
            box = np.s_[max(0, top - 1):bottom + 1, max(0, left - 1):right + 1]
            membership[group] = True
            pixels = membership[labels[box]]
            membership[group] = False
            ridge_widths = distance[box][pixels & ridge[box]]
            if np.std(ridge_widths) > 0.4 * np.mean(ridge_widths):
                rejected_groups.add(group_key)
                continue
            interior = cv2.erode(pixels.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            if np.count_nonzero(interior) < 0.08 * np.count_nonzero(pixels):
                rejected_groups.add(group_key)
                continue
            if np.median(contrast[box][interior]) < 80:
                rejected_groups.add(group_key)
                continue
            paint = np.median(small[box][interior], axis=0)
            if np.percentile(np.max(np.abs(small[box][interior] - paint), axis=1), 80) > 30:
                core = small[box][interior]
                core = core[_maximum_channel(core) >= np.median(_maximum_channel(core))]
                paint = np.median(core, axis=0)
                if np.percentile(np.max(np.abs(core - paint), axis=1), 80) > 30:
                    rejected_groups.add(group_key)
                    continue
                recovered = True
            row = (round(left / scale), round(top / scale), round(right / scale),
                   round(bottom / scale), float(np.median(heights) / scale), paint)
            if recovered:
                recovered_rows.append(row)
            else:
                rows.append(row)
                used.update(map(int, group))
    if missing_rows is not None:
        missing_rows.extend(recovered_rows)
    else:
        _append_missing_rows(rows, recovered_rows)
    return rows


def _text_rows(rgb):
    """작은 본문과 굵은 제목을 두 해상도에서 검출."""
    scale = min(1.0, 1600 / max(rgb.shape[:2]))
    missing_rows = []
    candidates = _rows_at_scale(rgb, scale, missing_rows) + _rows_at_scale(rgb, scale * 0.5, missing_rows)
    rows = []
    for left, top, right, bottom, height, paint in candidates:
        for index, (x0, y0, x1, y1, row_height, foreground) in enumerate(rows):
            overlap = min(bottom, y1) - max(top, y0)
            gap = max(x0 - right, left - x1, 0)
            if (overlap >= 0.7 * min(bottom - top, y1 - y0)
                    and abs(bottom - y1) <= 0.25 * max(height, row_height)
                    and max(height, row_height) <= 1.4 * min(height, row_height)
                    and gap <= 1.7 * max(height, row_height)
                    and np.max(np.abs(paint - foreground)) < 22):
                rows[index] = (min(left, x0), min(top, y0), max(right, x1), max(bottom, y1),
                               0.5 * (height + row_height), 0.5 * (paint + foreground))
                break
        else:
            rows.append((left, top, right, bottom, height, paint))
    _append_missing_rows(rows, missing_rows)
    return rows


def _local_plane(rgb, samples, sigma):
    """제외 마스크를 따르는 가우스 가중 국소 평면과 표본 지지 추정."""
    if max(rgb.shape[:2]) > 768 and sigma >= 10:
        size = ((rgb.shape[1] + 1) // 2, (rgb.shape[0] + 1) // 2)
        small_samples = cv2.resize(samples, size, interpolation=cv2.INTER_AREA)
        total = cv2.resize(rgb * samples[..., None], size, interpolation=cv2.INTER_AREA)
        small_rgb = total / np.maximum(small_samples[..., None], 1e-6)
        model, mass, determinant = _local_plane(small_rgb, small_samples, sigma / 2)
        full_size = (rgb.shape[1], rgb.shape[0])
        return (cv2.resize(model, full_size, interpolation=cv2.INTER_LINEAR),
                cv2.resize(mass, full_size, interpolation=cv2.INTER_LINEAR),
                cv2.resize(determinant * 16, full_size, interpolation=cv2.INTER_LINEAR))
    offsets = np.arange(-round(3 * sigma), round(3 * sigma) + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    first, second = kernel * offsets, kernel * offsets ** 2
    def _filtered(values, kx=kernel, ky=kernel):
        return cv2.sepFilter2D(values, cv2.CV_32F, kx, ky, borderType=cv2.BORDER_CONSTANT)
    mass = _filtered(samples)
    support = np.maximum(mass, 1e-6)
    mx, my = _filtered(samples, first) / support, _filtered(samples, kernel, first) / support
    vx = _filtered(samples, second) / support - mx * mx
    vy = _filtered(samples, kernel, second) / support - my * my
    xy = _filtered(samples, first, first) / support - mx * my
    total = rgb * samples[..., None]
    mean = _filtered(total) / support[..., None]
    cx = _filtered(total, first) / support[..., None] - mx[..., None] * mean
    cy = _filtered(total, kernel, first) / support[..., None] - my[..., None] * mean
    det = vx * vy - xy * xy
    denominator = np.maximum(det, 1e-4)[..., None]
    sx = (cx * vy[..., None] - cy * xy[..., None]) / denominator
    sy = (cy * vx[..., None] - cx * xy[..., None]) / denominator
    return mean - sx * mx[..., None] - sy * my[..., None], mass, det


def _contour_profile(residual, distance, ring):
    """행 전체의 거리/방향별 중앙값으로 공통 그림자와 발광 유지."""
    gx = cv2.Sobel(distance, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(distance, cv2.CV_32F, 0, 1)
    ring_distance = distance[ring]
    angle = (np.arctan2(gy[ring], gx[ring]) + 2 * np.pi) % (2 * np.pi)
    direction = np.floor(angle * (8 / (2 * np.pi))).astype(int) % 8
    depth = np.floor(ring_distance).astype(int)
    maximum_depth = int(np.max(depth))
    profile = np.zeros_like(residual)
    ring_depth, ring_direction, values = depth, direction, residual[ring]
    keys = ring_depth * 8 + ring_direction
    order = np.argsort(keys, kind="stable")
    keys, values = keys[order], values[order]
    colors = np.zeros((maximum_depth + 4, 8, 3), np.float32)
    for d in range(2, maximum_depth + 1):
        bounds = np.searchsorted(keys, np.arange(d * 8, (d + 1) * 8 + 1))
        if bounds[0] == bounds[-1]:
            continue
        common = np.median(values[bounds[0]:bounds[-1]], axis=0)
        for a in range(8):
            start, end = bounds[a:a + 2]
            if end - start >= 8:
                colors[d, a] = np.median(values[start:end], axis=0)
            else:
                colors[d, a] = common
    colors[:2] = colors[2]
    colors[maximum_depth + 1:] = colors[maximum_depth]
    padded = np.pad(colors, ((0, 0), (3, 3), (0, 0)), mode="wrap")
    smooth = cv2.GaussianBlur(padded, (0, 0), 0.8)[:, 3:-3]
    position = angle * (8 / (2 * np.pi)) - 0.5
    lower = np.floor(position).astype(int)
    fraction = (position - lower)[:, None]
    depth_fraction = (ring_distance - ring_depth)[:, None]
    first = smooth[ring_depth, lower % 8] * (1 - fraction) + smooth[ring_depth, (lower + 1) % 8] * fraction
    second = smooth[ring_depth + 1, lower % 8] * (1 - fraction) + smooth[ring_depth + 1, (lower + 1) % 8] * fraction
    profile[ring] = first * (1 - depth_fraction) + second * depth_fraction
    return profile


def _soft_edge_profile(rgb, background, foreground, distance):
    """흐린 획의 방향별 단조 감소 피복률과 설명 가능한 가장자리 추정."""
    shape = distance.shape
    nearby = distance <= 10
    rows = np.flatnonzero(np.any(nearby, axis=1))
    columns = np.flatnonzero(np.any(nearby, axis=0))
    box = np.s_[max(0, rows[0] - 1):rows[-1] + 2, max(0, columns[0] - 1):columns[-1] + 2]
    rgb, background, distance = rgb[box], background[box], distance[box]
    direction = foreground - background
    coverage = np.sum((rgb - background) * direction, axis=2)
    coverage /= np.maximum(np.sum(direction * direction, axis=2), 1)
    gx = cv2.Sobel(coverage, cv2.CV_32F, 1, 0, scale=0.125)
    gy = cv2.Sobel(coverage, cv2.CV_32F, 0, 1, scale=0.125)
    transition = (coverage > 0.15) & (coverage < 0.4) & (distance <= 3)
    if not np.any(transition):
        return None
    slope = float(np.median(np.sqrt(gx[transition] ** 2 + gy[transition] ** 2)))
    if slope >= 0.48:
        return None
    selected = (distance >= 1) & (distance <= 9)
    gx = cv2.Sobel(distance, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(distance, cv2.CV_32F, 0, 1)
    angle = (np.arctan2(gy[selected], gx[selected]) + 2 * np.pi) % (2 * np.pi)
    orientation = np.floor(angle * (16 / (2 * np.pi))).astype(int) % 16
    depth = np.rint(distance[selected] * 4).astype(int)
    keys = depth * 16 + orientation
    order = np.argsort(keys, kind="stable")
    keys, values = keys[order], coverage[selected][order]
    colors = np.zeros((37, 16), np.float32)
    common = np.zeros(37, np.float32)
    available = []
    for d in range(4, 37):
        start, end = np.searchsorted(keys, [d * 16, (d + 1) * 16])
        if end - start >= 4:
            common[d] = np.median(values[start:end])
            available.append(d)
    common = np.interp(np.arange(37), available, common[available])
    for a in range(16):
        available = []
        medians = []
        for d in range(4, 37):
            start, end = np.searchsorted(keys, [d * 16 + a, d * 16 + a + 1])
            if end - start >= 4:
                available.append(d)
                medians.append(np.median(values[start:end]))
        colors[:, a] = np.interp(np.arange(37), available, medians) if len(available) >= 2 else common
    colors = ndimage.gaussian_filter(colors, (0.5, 0.6), mode=("nearest", "wrap"))
    colors = np.minimum.accumulate(np.maximum(colors, 0), axis=0)
    radius = np.minimum(distance[selected] * 4, 36)
    lower = radius.astype(int)
    fraction = radius - np.floor(radius)
    position = angle * (16 / (2 * np.pi)) - 0.5
    first_angle = np.floor(position).astype(int)
    angle_fraction = position - np.floor(position)
    first = colors[lower, first_angle % 16] * (1 - angle_fraction) + colors[lower, (first_angle + 1) % 16] * angle_fraction
    second = colors[np.minimum(lower + 1, 36), first_angle % 16] * (1 - angle_fraction) + colors[np.minimum(lower + 1, 36), (first_angle + 1) % 16] * angle_fraction
    fitted = first * (1 - fraction) + second * fraction
    explained = np.zeros(distance.shape, bool)
    if slope < 0.4:
        # 넓게 퍼진 전이만 배경 추정 오차와 한 픽셀 위치 오차까지 포함.
        explained[selected] = (fitted > 0.0001) & (coverage[selected] > -0.01)
        explained = (cv2.dilate(explained.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & selected
    else:
        explained[selected] = (fitted > 0.002) & (coverage[selected] > 0)
    profile = np.zeros(distance.shape, np.float32)
    profile[selected] = fitted
    edge_mask = np.zeros(shape, bool)
    edge_mask[box] = explained
    return profile[explained, None] * direction[explained], edge_mask


def _background_curvature(rgb, samples, radius):
    """문자 없는 표본에서 선형 추세를 제거한 배경 곡률 추정."""
    height, width = samples.shape
    yy, xx = np.mgrid[0:height:4, 0:width:4].astype(np.float32)
    coordinates = np.stack([np.ones_like(xx), xx / width, yy / height], axis=2)
    selected = samples[::4, ::4] > 0
    if np.count_nonzero(selected) < 6:
        return np.full(samples.shape, 4, np.float32)
    plane = np.linalg.lstsq(coordinates[selected], rgb[::4, ::4][selected], rcond=None)[0]
    x = np.arange(width, dtype=np.float32) / width
    y = np.arange(height, dtype=np.float32) / height
    trend = plane[0] + x[None, :, None] * plane[1] + y[:, None, None] * plane[2]
    deviation = rgb - trend
    # 보정 상한보다 작은 배경 요동은 제거 대상이므로 곡률 보호에서 제외.
    if np.percentile(_maximum_channel(np.abs(deviation))[samples > 0], 90) <= 12:
        return np.zeros(samples.shape, np.float32)
    residual = deviation * samples[..., None]
    support = cv2.GaussianBlur(samples, (0, 0), radius)
    shape = cv2.GaussianBlur(residual, (0, 0), radius) / np.maximum(support[..., None], 1e-5)
    dxx = cv2.Sobel(shape, cv2.CV_32F, 2, 0, scale=0.25)
    dyy = cv2.Sobel(shape, cv2.CV_32F, 0, 2, scale=0.25)
    dxy = cv2.Sobel(shape, cv2.CV_32F, 1, 1, scale=0.25)
    return np.sqrt(_maximum_channel(dxx * dxx + dyy * dyy + 2 * dxy * dxy)) * radius ** 2


def apply(rgb, strength, excluded=None):
    """검증된 문자 행의 획/안티앨리어싱 밖 좁은 링만 보정."""
    rows = _text_rows(rgb)
    if not rows:
        return rgb.copy(), {"rows": 0, "eligible_fraction": 0.0}
    correction_sum = np.zeros_like(rgb)
    weight_sum = np.zeros(rgb.shape[:2], np.float32)
    immutable = np.zeros(rgb.shape[:2], bool)
    regions = []
    accepted = 0
    for left, top, right, bottom, glyph_height, foreground in rows:
        radius = float(np.clip(glyph_height * 0.27, 9, 16))
        if excluded is not None and np.all(excluded[max(0, top - round(radius)):bottom + round(radius),
                                                   max(0, left - round(radius)):right + round(radius)]):
            continue
        margin = round(4 * radius)
        x0 = max(0, left - max(margin, round(glyph_height) + 1))
        y0 = max(0, top - margin)
        x1, y1 = min(rgb.shape[1], right + margin), min(rgb.shape[0], bottom + margin)
        box = np.s_[y0:y1, x0:x1]
        pixels = rgb[box]
        median = cv2.medianBlur(np.clip(np.rint(pixels), 0, 255).astype(np.uint8), 31).astype(np.float32)
        row_box = np.zeros(pixels.shape[:2], np.uint8)
        extension = round((0.3 if glyph_height >= 45 else 0.2) * glyph_height)
        row_box[max(0, top - y0 - extension):bottom - y0 + extension,
                max(0, left - x0 - round(glyph_height)):right - x0 + extension] = 1
        background_value = np.median(_maximum_channel(median)[row_box > 0])
        threshold = min(0.5 * (float(np.max(foreground)) + background_value) - 8,
                        0.58 * float(np.max(foreground)))
        local_contrast = _maximum_channel(pixels - median)
        ink = _maximum_channel(pixels) > threshold
        ink &= row_box > 0
        _, labels = cv2.connectedComponents(ink.astype(np.uint8), 8)
        seeds = (local_contrast > 20) & (row_box > 0)
        keep = np.unique(labels[seeds & ink])
        keep = keep[keep != 0]
        ink = np.isin(labels, keep)
        if not np.any(ink):
            continue
        protected = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        immutable[box] |= protected
        distance = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5)
        regions.append((box, distance, radius, foreground))
    for box, distance, radius, foreground in regions:
        pixels = rgb[box]
        ring = (~immutable[box]) & (distance >= 2) & (distance < radius)
        if excluded is not None:
            ring &= excluded[box] == 0
        # 문자와 물체의 가는 경계는 표본에서 제외하되 완만한 얼룩은 허용.
        gx = cv2.Sobel(pixels, cv2.CV_32F, 1, 0, scale=0.125)
        gy = cv2.Sobel(pixels, cv2.CV_32F, 0, 1, scale=0.125)
        strong_edges = _maximum_channel(gx * gx + gy * gy) > 64
        edge_safe = cv2.dilate(strong_edges.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0
        near_ink = cv2.dilate(immutable[box].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        material_edges = (strong_edges & ~near_ink).astype(np.uint8)
        material_edges = cv2.morphologyEx(material_edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        boundary = cv2.dilate(material_edges, np.ones((3, 3), np.uint8))
        count, labels = cv2.connectedComponents(1 - boundary, 8)
        votes = np.bincount(labels[ring], minlength=count)
        votes[0] = 0
        if not np.any(votes):
            continue
        material = labels == np.argmax(votes)
        ring &= material
        samples = ((distance >= radius * (7 / 9)) & (distance <= radius * (28 / 9))
                   & edge_safe & material & ~immutable[box]).astype(np.float32)
        model, mass, det = _local_plane(pixels, samples, radius * (4 / 3))
        error = _maximum_channel(np.abs(pixels - model))
        validation = cv2.GaussianBlur(samples * (error < 20), (0, 0), radius)
        validation /= np.maximum(cv2.GaussianBlur(samples, (0, 0), radius), 1e-5)
        confidence = np.clip((validation - 0.1) / 0.15, 0, 1)
        curvature = _background_curvature(pixels, samples, radius)
        confidence *= np.clip((3 - curvature) / 2, 0, 1)
        confidence[(mass < 0.025) | (det < 1)] = 0
        ring &= confidence > 0
        if not np.any(ring):
            continue
        residual = pixels - model
        ring &= error < 25
        if not np.any(ring):
            continue
        profile = _contour_profile(residual, distance, ring)
        soft = _soft_edge_profile(pixels, model, foreground, distance)
        if soft is not None:
            soft_profile, soft_edge = soft
            profile[soft_edge] = soft_profile
        irregular = residual - profile
        if soft is not None:
            # 매끄러운 획 전이로 설명되는 부분은 모델 오차 범위만 보정.
            irregular[soft_edge] = np.clip(irregular[soft_edge], -1, 1)
        # 보정 폭의 끝에서 기울기까지 0으로 이어지는 완만한 가중치.
        inner = np.clip((distance - 1.0) / 1.5, 0, 1)
        outer = np.clip((radius - distance) / (0.4 * radius), 0, 1)
        inner = inner * inner * (3 - 2 * inner)
        outer = outer * outer * (3 - 2 * outer)
        weight = ring * inner * outer * confidence * min(strength, 1.0) * 0.85
        correction_sum[box] += np.clip(irregular, -12, 12) * weight[..., None]
        weight_sum[box] += weight
        accepted += 1
    correction_sum *= np.minimum(1, 0.85 / np.maximum(weight_sum, 1e-6))[..., None]
    output = rgb - correction_sum
    output[immutable] = rgb[immutable]
    return output, {"rows": accepted, "eligible_fraction": float(np.mean((weight_sum > 0) & ~immutable))}
