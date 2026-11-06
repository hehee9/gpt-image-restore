"""@description 매끈한 물체 윤곽의 법선 단면을 보존하는 윤곽 방향 잔물결 완화."""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ._native import load_native
from .glyphring import _text_rows
from .textgap import _text_region


_native = load_native("_native_contour")


def _arc_blur(values, sigma):
    """윤곽 호 방향만 따르는 가우스 평균."""
    return cv2.GaussianBlur(values, (1, 0), sigmaX=0, sigmaY=sigma,
                            borderType=cv2.BORDER_REPLICATE)


def _compact_graphic_mask(rgb):
    """평탄 그래픽의 개별 문자와 작은 기호를 명암 양쪽에서 보호."""
    luma = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    difference = luma - cv2.GaussianBlur(luma, (0, 0), 20)
    mask = np.zeros(luma.shape, np.uint8)
    for polarity in (-1, 1):
        signed_difference = difference * polarity
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            (signed_difference > 20).astype(np.uint8), 8)
        for label, (x, y, width, height, area) in enumerate(stats[1:], 1):
            # 큰 저대비 음영 덩어리는 굵은 제목 획으로 보호하지 않는다.
            if not (2 <= height <= 220 and 1 <= width <= 3 * height and area >= 3):
                continue
            if height >= 45:
                region = np.s_[y:y + height, x:x + width]
                if not np.any((labels[region] == label) & (signed_difference[region] >= 80)):
                    continue
            mask[max(0, y - 3):y + height + 3, max(0, x - 3):x + width + 3] = 1
    return mask


def _text_mask(rgb, flat_fraction):
    """밀집 문단과 양쪽 명암의 문자 행, 짧은 제목을 포함한 보호 영역."""
    mask = np.zeros(rgb.shape[:2], np.uint8)
    height, width = mask.shape
    scale = min(1.0, 1600 / max(height, width))
    size = (round(width * scale), round(height * scale))
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    median = cv2.medianBlur(np.clip(np.rint(small), 0, 255).astype(np.uint8), 31)
    dense = _text_region(small, np.max(np.abs(small - median), axis=2))
    dense = cv2.dilate(dense, np.ones((7, 7), np.uint8))
    if scale < 1:
        dense = cv2.resize(dense, (width, height), interpolation=cv2.INTER_NEAREST)
    mask[dense > 0] = 1
    if np.mean(mask) > 0.5:
        mask[:] = 1
        return mask
    # 이미 보호한 밀집 문단을 배경으로 채워 개별 문자 행의 중복 검출 비용 제외.
    rows_input = rgb.copy()
    background = cv2.resize(median, (width, height), interpolation=cv2.INTER_LINEAR) if scale < 1 else median
    rows_input[mask > 0] = background[mask > 0]
    rows = _text_rows(rows_input) + _text_rows(255 - rows_input)
    if flat_fraction >= 0.6:
        mask |= _compact_graphic_mask(rows_input)
    for left, top, right, bottom, row_height, _ in rows:
        extension = round(2.5 * row_height) if row_height >= 45 else 0
        padding = max(3, math.ceil(0.15 * row_height)) if row_height >= 45 else 3
        mask[max(0, top - padding):min(height, bottom + padding),
             max(0, left - extension - padding):min(width, right + extension + padding)] = 1
    return mask


def _arc_geometry(contour):
    """1픽셀 호 길이 재표본과 저곡률 법선 좌표 구성."""
    points = contour[:, 0].astype(np.float32)
    steps = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    arc = np.r_[0, np.cumsum(steps)]
    samples = np.arange(arc[-1], dtype=np.float32)
    smooth = np.column_stack([ndimage.gaussian_filter1d(
        np.interp(samples, arc, points[:, channel]), 6, mode="nearest") for channel in range(2)])
    tangent = np.gradient(smooth, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    curvature = np.linalg.norm(np.gradient(tangent, axis=0), axis=1)
    return smooth.astype(np.float32), normal.astype(np.float32), curvature


def _line_art_context(rgb, luma):
    """넓은 평탄 채색과 양쪽이 밝은 검정 획이 공존하는 선화 문맥 판정."""
    scale = 1024 / max(rgb.shape[:2])
    if scale != 1:
        size = (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, size, interpolation=interpolation)
        luma = cv2.resize(luma, size, interpolation=interpolation)
    fine = np.max(np.abs(rgb - cv2.GaussianBlur(rgb, (0, 0), 1)), axis=2)
    gradient = np.maximum(np.abs(cv2.Sobel(luma, cv2.CV_32F, 1, 0, scale=0.125)),
                          np.abs(cv2.Sobel(luma, cv2.CV_32F, 0, 1, scale=0.125)))
    flat = (fine < 1) & (gradient < 2)
    flat_fraction = float(np.mean(flat))
    if flat_fraction < 0.6:
        return False, flat_fraction
    closing = cv2.morphologyEx(luma, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    dark_ridges = (np.max(rgb, axis=2) < 110) & (closing - luma > 25)
    ridge_pixels = dark_ridges.astype(np.uint8)
    rules = (cv2.morphologyEx(ridge_pixels, cv2.MORPH_OPEN, np.ones((1, 49), np.uint8))
             | cv2.morphologyEx(ridge_pixels, cv2.MORPH_OPEN, np.ones((49, 1), np.uint8)))
    dark_ridges[cv2.dilate(rules, np.ones((3, 3), np.uint8)) > 0] = False
    compact = _compact_graphic_mask(rgb)
    for left, top, right, bottom, _, _ in _text_rows(255 - rgb) + _text_rows(rgb):
        row = np.s_[max(0, top - 3):bottom + 3]
        dark_ridges[row, max(0, left - 3):right + 3] = False
        # 검출된 행 옆에 남은 짧은 문자 조각도 획 분포 근거에서 제외.
        dark_ridges[row] &= compact[row] == 0
    yy, xx = np.nonzero(dark_ridges)
    if not len(xx):
        return False, flat_fraction
    # 짧은 문자와 작은 아바타 조각만 모인 화면을 긴 잉크선 망과 구별.
    extended_fraction = float(np.mean(compact[dark_ridges] == 0))
    if extended_fraction < 0.1:
        return False, flat_fraction
    # 멀리 떨어진 흰 여백 대신 긴 잉크선 양옆의 채색 면에서 평탄성 확인.
    extended = (dark_ridges & (compact == 0)).astype(np.uint8)
    outer = cv2.dilate(extended, np.ones((17, 17), np.uint8)) > 0
    inner = cv2.dilate(dark_ridges.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    adjacent = outer & ~inner
    adjacent_flat = (fine < 2) & (gradient < 3)
    if not np.any(adjacent) or np.mean(adjacent_flat[adjacent]) < 0.5:
        return False, flat_fraction
    # 여백 대신 실제 잉크선이 있는 범위에서 분포를 판단하고 문자 행은 제외.
    subject = dark_ridges[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    h, w = subject.shape
    if min(h, w) < 4:
        return False, flat_fraction
    interior = subject[h // 4:3 * h // 4, w // 4:3 * w // 4]
    covered_tiles = sum(np.mean(tile) > 0.005
                        for row in np.array_split(subject, 4, axis=0)
                        for tile in np.array_split(row, 4, axis=1))
    return bool(np.mean(subject) > 0.005 and np.mean(interior) > 0.005
                and covered_tiles >= 10), flat_fraction


def _thin_strokes(pixels):
    """끝점과 분기 검출을 위한 1픽셀 폭 획 추출."""
    center = pixels.astype(np.uint8)
    codes = np.arange(256, dtype=np.uint8)
    neighbors = tuple((codes >> bit) & 1 for bit in range(8))
    count = sum(neighbors)
    transitions = sum((a == 0) & (b > 0)
                      for a, b in zip(neighbors, neighbors[1:] + neighbors[:1]))
    north, _, east, _, south, _, west, _ = neighbors
    common = (count >= 2) & (count <= 6) & (transitions == 1)
    lookups = (
        common & (north * east * south == 0) & (east * south * west == 0),
        common & (north * east * west == 0) & (north * south * west == 0),
    )
    kernel = np.array([[128, 1, 2], [64, 0, 4], [32, 16, 8]], np.float32)
    while True:
        changed = False
        for lookup in lookups:
            # 이웃 8개의 삭제 조건을 동일한 256항 표로 계산해 큰 임시 배열 제외.
            codes = cv2.filter2D(center, -1, kernel, borderType=cv2.BORDER_CONSTANT)
            remove = cv2.bitwise_and(center, cv2.LUT(codes, lookup.astype(np.uint8)))
            if cv2.countNonZero(remove):
                cv2.subtract(center, remove, dst=center)
                changed = True
        if not changed:
            return center


def _open_stroke_components(ridges, labels, component_count):
    """끝점과 실제 분기점을 가진 열린/분기 획의 연결 요소 판정."""
    skeleton = _thin_strokes(ridges)
    padded = np.pad(skeleton, 1)
    neighbors = (padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:],
                 padded[2:, 2:], padded[2:, 1:-1], padded[2:, :-2],
                 padded[1:-1, :-2], padded[:-2, :-2])
    transitions = np.zeros_like(skeleton)
    for a, b in zip(neighbors, neighbors[1:] + neighbors[:1]):
        transitions += (a == 0) & (b > 0)
    ends = (skeleton > 0) & (sum(neighbors) == 1)
    junctions = (skeleton > 0) & (transitions >= 3)
    end_count = np.bincount(labels[ends], minlength=component_count)
    junction_count = np.bincount(labels[junctions], minlength=component_count)
    opened = (end_count > 0) | (junction_count > 0)
    opened[0] = False
    return opened, end_count, junction_count


def _panel_has_ink_network(rgb):
    """단일 테두리/물체 곡선과 구별되는 분산된 다중 잉크선 망 확인."""
    luma = rgb @ np.array([.299, .587, .114], np.float32)
    closing = cv2.morphologyEx(luma, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    ridges = (np.max(rgb, axis=2) < 110) & (closing - luma > 25)
    for left, top, right, bottom, _, _ in _text_rows(rgb) + _text_rows(255 - rgb):
        ridges[max(0, top - 3):bottom + 3, max(0, left - 3):right + 3] = False
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ridges.astype(np.uint8), 8)
    valid = ((stats[:, cv2.CC_STAT_AREA] >= 6)
             & (np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= 6))
    valid[0] = False
    opened, _, _ = _open_stroke_components(valid[labels], labels, len(stats))
    valid &= opened
    if np.count_nonzero(valid) < 6:
        return False
    areas = stats[valid, cv2.CC_STAT_AREA]
    if np.max(areas) >= 0.5 * np.sum(areas):
        return False
    ridges = valid[labels]
    covered_tiles = sum(np.mean(tile) > 0.005
                        for row in np.array_split(ridges, 3, axis=0)
                        for tile in np.array_split(row, 3, axis=1))
    return covered_tiles >= 6


def _framed_illustration_mask(rgb):
    """반복된 컷 안의 가는 검정 획과 인접 경계만 국소적으로 보호."""
    height, width = rgb.shape[:2]
    scale = 1024 / max(height, width)
    small = cv2.resize(rgb, (round(width * scale), round(height * scale)),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    dark = (np.max(small, axis=2) < 45).astype(np.uint8)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    rectangular = []
    for contour in contours:
        left, top, w, h = cv2.boundingRect(contour)
        if w < 0.65 * small.shape[1] or h < 0.1 * small.shape[0]:
            continue
        boxes.append((left, top, w, h))
        if cv2.contourArea(contour) > 0.97 * w * h:
            rectangular.append((left, w))
    mask = np.zeros(small.shape[:2], np.uint8)
    panels = set()
    for left, w in rectangular:
        aligned = [(x, y, bw, bh) for x, y, bw, bh in boxes
                   if abs(x - left) < 0.02 * small.shape[1] and abs(bw - w) < 0.02 * small.shape[1]]
        verified = sum(abs(x - left) < 0.02 * small.shape[1] and abs(bw - w) < 0.02 * small.shape[1]
                       for x, bw in rectangular)
        if verified < 2:
            continue
        panels.update(aligned)
    for x, y, bw, bh in panels:
        # 컷 테두리 자체를 빼고 내부의 여러 그림 획을 확인한 컷만 보호.
        if not _panel_has_ink_network(small[y + 8:y + bh - 8, x + 8:x + bw - 8]):
            continue
        mask[max(0, y - 2):y + bh + 2, max(0, x - 2):x + bw + 2] = 1
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    if not np.any(mask):
        return mask
    # 컷 배치는 축소해서 찾되 가는 잉크 획은 원본 해상도에서 보존.
    luma = rgb @ np.array([.299, .587, .114], np.float32)
    kernel_size = 2 * math.ceil(7 / scale) + 1
    closing = cv2.morphologyEx(luma, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    dark = np.max(rgb, axis=2) < 110
    contrast = closing - luma
    left, top, region_width, region_height = cv2.boundingRect(mask)
    region = np.s_[top:top + region_height, left:left + region_width]
    dark, contrast = dark[region], contrast[region]
    weak = dark & (contrast > 10) & (mask[region] > 0)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(weak.astype(np.uint8), 8)
    opened, _, _ = _open_stroke_components(weak, labels, len(stats))
    # 앞 단계에서 대비가 낮아진 연결 획도 유지하되 고립된 미세 잡음은 제외.
    strokes = (stats[:, cv2.CC_STAT_AREA] >= 6) & (
        np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= 6)
    strokes[0] = False
    ink = ((dark & (contrast > 25)) | strokes[labels]) & opened[labels]
    mask[region] &= ink.astype(np.uint8)
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))


def _restore(rgb, strength):
    """윤곽별 신뢰도와 법선 단면으로 제한된 부동소수점 보정."""
    height, width = rgb.shape[:2]
    luma = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    line_art, flat_fraction = _line_art_context(rgb, luma)
    if line_art:
        return rgb.copy(), {
            "enabled": True, "strength": float(strength), "protected_line_art": True,
            "flat_region_fraction": flat_fraction, "candidate_contours": 0,
            "accepted_contours": 0, "masked_fraction": 0.0, "mean_confidence": 0.0,
            "mean_abs_change": [0.0, 0.0, 0.0], "max_abs_change": 0.0,
        }, np.zeros((height, width), np.float32)
    blurred = cv2.GaussianBlur(luma, (0, 0), 1.7)
    edges = cv2.Canny(np.clip(np.rint(blurred), 0, 255).astype(np.uint8), 12, 30)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    density = cv2.GaussianBlur((edges > 0).astype(np.float32), (0, 0), 7)
    protected = _text_mask(rgb, flat_fraction)
    protected_text_fraction = float(np.mean(protected))
    illustration = _framed_illustration_mask(rgb)
    protected |= illustration
    confidence = np.zeros((height, width), np.float32)
    correction = np.zeros_like(rgb, dtype=np.float32)
    depths = np.arange(-15, 16, dtype=np.float32)
    offsets = np.arange(-56, 57, dtype=np.float32)
    # 가우스 가중 이차 추정으로 넓은 명암 곡률을 유지하며 14~32픽셀 잔물결 분리.
    long_kernel = np.exp(-0.5 * (offsets / 14) ** 2) * (1.5 - 0.5 * (offsets / 14) ** 2)
    long_kernel /= np.sum(long_kernel)
    accepted = 0
    candidates = 0
    for contour in contours:
        if len(contour) < 100:
            continue
        candidates += 1
        points, normal, curvature = _arc_geometry(contour)
        # 대체로 볼록한 윤곽의 국소 오목/볼록 전환은 손잡이와 패임으로 보호.
        if len(points) >= 250 and np.mean(curvature < 0.02) > 0.9:
            normal_change = np.gradient(normal, axis=0)
            signed_curvature = normal[:, 0] * normal_change[:, 1] - normal[:, 1] * normal_change[:, 0]
            dominant_turn = np.sign(np.median(signed_curvature))
            if np.mean(signed_curvature * dominant_turn >= -0.001) > 0.85:
                turning = ndimage.uniform_filter1d(curvature, 25, mode="nearest") * 25
                reverse_bend = ndimage.maximum_filter1d(
                    ((curvature > 0.025) & (signed_curvature * dominant_turn < 0)).astype(np.uint8), 65)
                bends = points[(curvature > 0.04) & (turning < 2.4)
                               & (reverse_bend > 0)][::3]
                for x, y in np.rint(bends).astype(int):
                    cv2.circle(protected, (int(x), int(y)), 20, 1, -1)
        arc_weight = np.clip((0.10 - curvature) / 0.05, 0, 1)
        edge_density = cv2.remap(density, points[:, 0:1], points[:, 1:2], cv2.INTER_LINEAR)[:, 0]
        arc_weight *= np.clip((0.15 - edge_density) / 0.06, 0, 1)
        arc_weight[:24] = 0
        arc_weight[-24:] = 0
        if not np.any(arc_weight > 0):
            continue
        x0, y0 = np.maximum(np.floor(points.min(axis=0)).astype(int) - 16, 0)
        x1, y1 = np.minimum(np.ceil(points.max(axis=0)).astype(int) + 17, [width, height])
        protected_region = protected[y0:y1, x0:x1]
        if (protected[(y0 + y1) // 2, (x0 + x1) // 2]
                and cv2.countNonZero(protected_region) == protected_region.size):
            continue
        coordinates = points[:, None, :] + normal[:, None, :] * depths[None, :, None]
        profile = cv2.remap(rgb, coordinates[..., 0], coordinates[..., 1], cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
        short_model = _arc_blur(profile, 7)
        long_model = cv2.sepFilter2D(profile, cv2.CV_32F, np.ones(1, np.float32), long_kernel,
                                     borderType=cv2.BORDER_REPLICATE)
        # 단면 변화가 큰 돌출물과 장식은 긴 호 길이 추정에서 제외.
        disagreement = np.sqrt(np.mean((short_model - long_model) ** 2, axis=(1, 2)))
        consistency = np.clip((5 - disagreement) / 3, 0, 1)
        model = short_model + consistency[:, None, None] * (long_model - short_model)
        residual = profile - model
        fine = residual - _arc_blur(residual, 0.8)
        fine_rms = np.sqrt(_arc_blur(np.mean(fine ** 2, axis=2), 5))
        total_rms = np.sqrt(_arc_blur(np.mean(residual ** 2, axis=2), 5))
        weight = (arc_weight[:, None] * np.clip((4 - fine_rms) / 2, 0, 1)
                  * np.clip((10 - total_rms) / 4, 0, 1))
        if not np.any(weight > 0.05):
            continue
        tube = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.polylines(tube, [np.rint(points - [x0, y0]).astype(np.int32)], False, 1, 31)
        yy, xx = np.nonzero(tube)
        if tube.size < 5 * len(xx):
            # 좁은 띠는 거리 변환 후 호 방향 재정렬, 넓은 빈 상자는 희소 최근접 탐색.
            seeds = np.ones_like(tube)
            pixels = np.rint(points - [x0, y0]).astype(int)
            seeds[pixels[:, 1], pixels[:, 0]] = 0
            _, labels = cv2.distanceTransformWithLabels(
                seeds, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            lookup = np.zeros(int(labels.max()) + 1, np.int32)
            lookup[labels[pixels[:, 1], pixels[:, 0]]] = np.arange(len(points))
            index = lookup[labels[yy, xx]]
            if _native is not None:
                _native.refine_index(index, xx, yy, x0, y0, points, normal)
            else:
                positions = np.column_stack((xx + x0, yy + y0))
                displacement = positions - points[index]
                shift = np.rint(displacement[:, 0] * normal[index, 1]
                                - displacement[:, 1] * normal[index, 0]).astype(int)
                index = np.clip(index + shift, 0, len(points) - 1)
                nearest_squared = np.sum((positions - points[index]) ** 2, axis=1)
                for offset in (-1, 1):
                    candidate = np.clip(index + offset, 0, len(points) - 1)
                    squared = np.sum((positions - points[candidate]) ** 2, axis=1)
                    closer = squared < nearest_squared
                    index[closer] = candidate[closer]
                    nearest_squared[closer] = squared[closer]
        else:
            _, index = cKDTree(points).query(np.column_stack((xx + x0, yy + y0)))
        if _native is not None:
            accepted += _native.apply_profile(rgb, protected, confidence, correction,
                                              points, normal, weight, model, index, xx, yy, x0, y0)
            continue
        xx, yy = xx + x0, yy + y0
        distance = np.linalg.norm(np.column_stack((xx, yy)) - points[index], axis=1)
        depth = np.sum((np.column_stack((xx, yy)) - points[index]) * normal[index], axis=1)
        valid = (distance < 14) & (protected[yy, xx] == 0)
        xx, yy, index, depth, distance = (value[valid] for value in (xx, yy, index, depth, distance))
        sample_depth = np.clip(depth + 15, 0, 30).astype(np.float32)
        lower = sample_depth.astype(np.int32)
        upper = np.minimum(lower + 1, 30)
        fraction = sample_depth - lower
        pixel_weight = (weight[index, lower] * (1 - fraction)
                        + weight[index, upper] * fraction)
        pixel_weight *= np.clip((14 - distance) / 4, 0, 1)
        use = pixel_weight > confidence[yy, xx]
        xx, yy, pixel_weight = xx[use], yy[use], pixel_weight[use]
        if not len(xx):
            continue
        accepted += 1
        mix = fraction[use, None]
        target = (model[index[use], lower[use]] * (1 - mix)
                  + model[index[use], upper[use]] * mix)
        confidence[yy, xx] = pixel_weight
        correction[yy, xx] = np.clip(rgb[yy, xx] - target, -10, 10) * pixel_weight[:, None]
    confidence[protected > 0] = 0
    correction[protected > 0] = 0
    output = (rgb - strength * correction).astype(np.float32)
    change = output - rgb
    report = {"enabled": True, "strength": float(strength), "candidate_contours": candidates,
              "protected_line_art": False, "flat_region_fraction": flat_fraction,
              "protected_illustration_fraction": float(np.mean(illustration)),
              "accepted_contours": accepted, "masked_fraction": float(np.mean(confidence > 0.1)),
              "mean_confidence": float(np.mean(confidence)),
              "protected_text_fraction": protected_text_fraction,
              "mean_abs_change": np.mean(np.abs(change), axis=(0, 1)).tolist(),
              "max_abs_change": float(np.max(np.abs(change)))}
    return output, report, confidence


def apply(rgb: np.ndarray, strength: float = 1.0) -> tuple[np.ndarray, dict]:
    """법선 단면 유지와 질감/문자 보호를 병행한 윤곽 잔물결 보정."""
    if strength < 0 or not math.isfinite(strength):
        raise ValueError("contour stage strength must be finite and 0 or greater.")
    if strength == 0:
        return rgb.copy(), {"enabled": False, "strength": 0.0, "masked_fraction": 0.0}
    output, report, _ = _restore(np.asarray(rgb, dtype=np.float32), strength)
    return output, report


def analyze(rgb: np.ndarray) -> dict:
    """윤곽 후보, 보호 범위와 예상 보정량 분석."""
    _, report = apply(rgb)
    return report
