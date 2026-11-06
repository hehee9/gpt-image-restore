"""@description 표면의 작은 밝은 점과 어두운 점 검출 및 잔차 감쇠."""

from __future__ import annotations

from functools import partial
import os

import cv2
import numpy as np
from scipy import ndimage
from scipy.sparse import csr_array
from scipy.spatial import cKDTree

from ._parallel import _executor, _median_blur
from ._native import load_native


_native = load_native("_native_sparkle")


_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
# NumPy 버전별 기존 float32 스칼라 곱셈 자료형 보존.
_SCALAR_PRODUCT_DTYPE = (np.float32(1) * 0.1).dtype
_SIGMAS = (0.7, 1.0, 1.4, 2.0)
_BATCH_SIZE = 4096
_WORKERS = min(4, os.cpu_count() or 1)
_PATCH_AXIS = np.arange(13)
_PATCH_Y, _PATCH_X = np.mgrid[:13, :13]
_RING = (_PATCH_X - 6) ** 2 + (_PATCH_Y - 6) ** 2 >= 16
_SURFACE_Y, _SURFACE_X = np.mgrid[-15:16, -15:16]
_SURFACE_RING = np.maximum(np.abs(_SURFACE_Y), np.abs(_SURFACE_X)) > 6
_SURFACE_Y, _SURFACE_X = _SURFACE_Y[_SURFACE_RING], _SURFACE_X[_SURFACE_RING]
_SURROUND_ANGLES = np.arange(8) * (2 * np.pi / 8)
_SURROUND_RADII = np.array([8, 16, 24])[:, None]
_SURROUND_Y = np.rint(np.sin(_SURROUND_ANGLES) * _SURROUND_RADII).astype(int)
_SURROUND_X = np.rint(np.cos(_SURROUND_ANGLES) * _SURROUND_RADII).astype(int)
_BIT_WEIGHTS = (1 << _PATCH_AXIS).astype(np.uint16)
_BIT_VALUES = np.arange(1 << 13, dtype=np.uint16)
_BIT_COUNTS = np.unpackbits(_BIT_VALUES.view(np.uint8)).reshape(-1, 16).sum(axis=1)
_BIT_PIXELS = (_BIT_VALUES[:, None] & _BIT_WEIGHTS) != 0
_BIT_SUM_X = np.sum(_BIT_PIXELS * _PATCH_AXIS, axis=1)
_BIT_SUM_X2 = np.sum(_BIT_PIXELS * _PATCH_AXIS ** 2, axis=1)
_BIT_SPANS = np.zeros(1 << 13, np.uint8)
for _value in range(1, 1 << 13):
    _BIT_SPANS[_value] = _value.bit_length() - (_value & -_value).bit_length() + 1


def _blur_patches(patches: np.ndarray, sigma: float) -> np.ndarray:
    """각 패치의 반사 경계를 유지하는 축별 일괄 가우시안 필터."""
    count = len(patches)
    horizontal = cv2.GaussianBlur(patches.reshape(-1, 13), (0, 1), sigma)
    columns = horizontal.reshape(count, 13, 13).transpose(0, 2, 1).reshape(-1, 13)
    vertical = cv2.GaussianBlur(columns, (0, 1), sigma)
    return vertical.reshape(count, 13, 13).transpose(0, 2, 1)


def _dilate_cores(cores: np.ndarray) -> np.ndarray:
    """서로 이웃한 패치가 섞이지 않는 3×3 코어 팽창."""
    count = len(cores)
    horizontal = cv2.dilate(cores.reshape(-1, 13), np.ones((1, 3), np.uint8))
    columns = horizontal.reshape(count, 13, 13).transpose(0, 2, 1).reshape(-1, 13)
    vertical = cv2.dilate(columns, np.ones((1, 3), np.uint8))
    return vertical.reshape(count, 13, 13).transpose(0, 2, 1)


def _core_geometry(patches: np.ndarray, contrasts: np.ndarray, minimum: float = 6) -> tuple:
    """연결 코어를 일괄 확장하고 행 비트맵으로 크기와 공분산 계산."""
    count = len(patches)
    foreground = patches >= np.maximum(minimum, contrasts[:, None, None] * 0.5)
    centers = np.zeros((count, 7, 7), bool)
    centers[:, 3, 3] = True
    inner = ndimage.binary_propagation(centers, structure=np.ones((1, 3, 3), bool),
                                       mask=foreground[:, 3:10, 3:10])
    cores = np.zeros_like(foreground)
    cores[:, 3:10, 3:10] = inner
    spilling = (np.any(inner[:, 0], axis=1) | np.any(inner[:, -1], axis=1)
                | np.any(inner[:, :, 0], axis=1) | np.any(inner[:, :, -1], axis=1))
    oversized = np.sum(inner, axis=(1, 2)) > 24
    cores[oversized] = False
    spilling &= ~oversized
    if np.any(spilling):
        centers = np.zeros((int(spilling.sum()), 13, 13), bool)
        centers[:, 6, 6] = True
        cores[spilling] = ndimage.binary_propagation(
            centers, structure=np.ones((1, 3, 3), bool), mask=foreground[spilling])
    state = np.sum(cores * _BIT_WEIGHTS, axis=2, dtype=np.uint16)
    area = _BIT_COUNTS[state].sum(axis=1)
    bw = _BIT_SPANS[np.bitwise_or.reduce(state, axis=1)]
    occupied_rows = np.sum((state != 0) * _BIT_WEIGHTS, axis=1, dtype=np.uint16)
    bh = _BIT_SPANS[occupied_rows]
    counts = _BIT_COUNTS[state]
    sums_x = _BIT_SUM_X[state]
    sx = np.sum(sums_x, axis=1, dtype=np.float64)
    sy = np.sum(counts * _PATCH_AXIS, axis=1, dtype=np.float64)
    mx, my = sx / np.maximum(area, 1), sy / np.maximum(area, 1)
    denominator = np.maximum(area.astype(np.int64) - 1, 1)
    xx = (np.sum(_BIT_SUM_X2[state], axis=1) - sx * mx) / denominator
    yy = (np.sum(counts * _PATCH_AXIS ** 2, axis=1) - sy * my) / denominator
    xy = (np.sum(sums_x * _PATCH_AXIS, axis=1) - sx * my) / denominator
    xx = np.where(area > 2, xx, 0.25)
    yy = np.where(area > 2, yy, 0.25)
    xy = np.where(area > 2, xy, 0)
    difference = np.sqrt((xx - yy) ** 2 + 4 * xy * xy)
    axis_ratio_squared = (xx + yy + difference + 0.5) / (xx + yy - difference + 0.5)
    return cores, area, bw, bh, mx, my, axis_ratio_squared


def _regular_points(points: np.ndarray, contrasts: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """비슷한 간격의 직선 배열과 격자에 속하는 하이라이트 보호."""
    protected = np.zeros(len(points), bool)
    if len(points) < 4:
        return protected
    tree = cKDTree(points)
    _, neighbors_all = tree.query(points, k=min(13, len(points)), distance_upper_bound=48)
    anchors, firsts, seconds, steps = [], [], [], []
    for start in range(0, len(points), _BATCH_SIZE):
        stop = min(start + _BATCH_SIZE, len(points))
        index = np.arange(start, stop)
        raw_neighbors = neighbors_all[start:stop, 1:]
        valid = raw_neighbors < len(points)
        neighbors = np.minimum(raw_neighbors, len(points) - 1)
        valid &= ((contrasts[neighbors] >= 0.65 * contrasts[index, None])
                  & (contrasts[neighbors] <= 1.55 * contrasts[index, None])
                  & (areas[neighbors] >= 0.55 * areas[index, None])
                  & (areas[neighbors] <= 1.8 * areas[index, None]))
        vectors = points[neighbors] - points[index, None]
        distances = np.linalg.norm(vectors, axis=2)
        unit = np.divide(vectors, distances[..., None], out=np.zeros_like(vectors),
                         where=valid[..., None])
        opposite = unit @ unit.transpose(0, 2, 1) < -0.985
        same_spacing = np.abs(distances[:, :, None] - distances[:, None, :]) < np.maximum(
            1.25, 0.12 * np.minimum(distances[:, :, None], distances[:, None, :]))
        pair_mask = opposite & same_spacing & valid[:, :, None] & valid[:, None, :]
        pair_mask &= np.triu(np.ones(pair_mask.shape[1:], bool), 1)
        row, first, second = np.nonzero(pair_mask)
        anchors.append(index[row])
        firsts.append(neighbors[row, first])
        seconds.append(neighbors[row, second])
        steps.append((vectors[row, first] - vectors[row, second]) * 0.5)
    anchor = np.concatenate(anchors)
    if not len(anchor):
        return protected
    first, second = np.concatenate(firsts), np.concatenate(seconds)
    step = np.concatenate(steps)
    targets = np.stack((points[anchor] + 2 * step, points[anchor] - 2 * step), axis=1)
    distance, confirmation = tree.query(targets)
    tolerance = np.maximum(1.25, 0.12 * np.linalg.norm(step, axis=1).astype(np.float64))
    matched = ((distance <= tolerance[:, None])
               & (contrasts[confirmation] >= 0.65 * contrasts[anchor, None])
               & (contrasts[confirmation] <= 1.55 * contrasts[anchor, None])
               & (areas[confirmation] >= 0.55 * areas[anchor, None])
               & (areas[confirmation] <= 1.8 * areas[anchor, None]))
    confirmed = np.any(matched, axis=1)
    protected[anchor[confirmed]] = True
    protected[first[confirmed]] = True
    protected[second[confirmed]] = True
    protected[confirmation[matched]] = True
    return protected


def _eligible_cores(luma, residual, scale, ys, xs, contrasts, start):
    """독립적인 후보 묶음의 코어 크기, 질감 대비와 곡률 판정."""
    y, x = ys[start:start + _BATCH_SIZE], xs[start:start + _BATCH_SIZE]
    contrast = contrasts[start:start + _BATCH_SIZE]
    patch_y = y[:, None, None] + _PATCH_AXIS[None, :, None] - 6
    patch_x = x[:, None, None] + _PATCH_AXIS[None, None, :] - 6
    patches = residual[patch_y, patch_x]
    cores, area, bw, bh, mx, my, axis_ratio_squared = _core_geometry(patches, contrast)
    selected = (area > 0) & (area <= 24) & (np.maximum(bw, bh) <= 7) & (axis_ratio_squared <= 6.25)
    if not np.any(selected):
        return None
    y, x, contrast = y[selected], x[selected], contrast[selected]
    cores, area = cores[selected], area[selected]
    bw, bh, mx, my = bw[selected], bh[selected], mx[selected], my[selected]
    patches = patches[selected]
    samples = patches[:, _RING]
    samples.sort(axis=1)
    middle = samples.shape[1] // 2
    median = (samples[:, middle - 1] + samples[:, middle]) * 0.5
    samples = np.abs(samples - median[:, None])
    samples.sort(axis=1)
    texture = (samples[:, middle - 1] + samples[:, middle]) * (0.5 * 1.4826)
    wide = (area > 12) | (np.maximum(bw, bh) > 4)
    selected = ((contrast >= np.maximum(12, 2.1 * texture)) & (texture >= 0.8)
                & (~wide | (contrast >= np.maximum(28, 3.0 * texture))))
    if not np.any(selected):
        return None
    y, x, contrast = y[selected], x[selected], contrast[selected]
    cores, area, mx, my = cores[selected], area[selected], mx[selected], my[selected]
    patches = luma[y[:, None, None] + _PATCH_AXIS[None, :, None] - 6,
                   x[:, None, None] + _PATCH_AXIS[None, None, :] - 6]
    selected = np.zeros(len(x), bool)
    for index, sigma in enumerate(_SIGMAS):
        at_scale = scale[y, x] == index
        if not np.any(at_scale):
            continue
        smooth = _blur_patches(patches[at_scale], sigma)
        dxx = smooth[:, 6, 7] + smooth[:, 6, 5] - 2 * smooth[:, 6, 6]
        dyy = smooth[:, 7, 6] + smooth[:, 5, 6] - 2 * smooth[:, 6, 6]
        dxy = (smooth[:, 7, 7] - smooth[:, 7, 5] - smooth[:, 5, 7] + smooth[:, 5, 5]) * 0.25
        matrices = np.stack((dxx, dxy, dxy, dyy), axis=1).reshape(-1, 2, 2)
        curvature = np.linalg.eigvalsh(matrices)
        negative = curvature[:, 1] < -0.25
        ratio = np.divide(curvature[:, 0], curvature[:, 1],
                           out=np.full(len(curvature), np.inf, curvature.dtype), where=negative)
        selected[at_scale] = negative & (ratio <= 8)
    if not np.any(selected):
        return None
    return (y[selected], x[selected], contrast[selected], cores[selected],
            area[selected], mx[selected], my[selected])


def _round_eye_component(patch, labels, stats, label):
    """둥근 암부의 형상과 밝은 둘레를 확인하고 반사광 여유 영역 반환."""
    left, top, bw, bh, area = stats[label]
    if (min(bw, bh) < 12 or max(bw, bh) > 128 or area < 100
            or max(bw, bh) > 1.7 * min(bw, bh)
            or left == 0 or top == 0 or left + bw == patch.shape[1]
            or top + bh == patch.shape[0] or area / (bw * bh) < 0.55):
        return None
    mask = labels == label
    my, mx = np.nonzero(mask)
    covariance = np.cov(np.stack((mx, my)))
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[-1] > 2.25 * eigenvalues[0]:
        return None
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = cv2.arcLength(contours[0], True)
    if 4 * np.pi * area < 0.70 * perimeter * perimeter:
        return None
    interior = float(np.median(patch[mask]))
    variation = float(np.std(patch[mask]))
    if variation > 10:
        return None
    # 성분 경계와 커널 반경 밖의 영점 영역을 제외한 정확한 팽창.
    rx, ry = int(bw // 2), int(bh // 2)
    x0, x1 = max(0, left - rx), min(patch.shape[1], left + bw + rx)
    y0, y1 = max(0, top - ry), min(patch.shape[0], top + bh + ry)
    local_mask = mask[y0:y1, x0:x1]
    outside = cv2.dilate(local_mask.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (rx * 2 + 1, ry * 2 + 1)))
    ring = (outside > 0) & ~local_mask
    if np.median(patch[y0:y1, x0:x1][ring]) < interior + max(14, 0.25 * interior):
        return None
    yy, xx = np.ogrid[:patch.shape[0], :patch.shape[1]]
    radial = ((xx - mx.mean()) / (bw * 0.5)) ** 2 + ((yy - my.mean()) / (bh * 0.5)) ** 2
    far_ring = (radial > 4) & (radial < 9)
    if np.count_nonzero(far_ring) < 100:
        return None
    far_contrast = float(np.median(patch[far_ring])) - interior
    if variation <= 4.5:
        if far_contrast < max(28, 0.6 * interior):
            return None
    else:
        outer_ring = (radial > 16) & (radial < 36)
        if (np.count_nonzero(outer_ring) < 100 or far_contrast < 20
                or np.median(patch[outer_ring]) < interior + 60):
            return None
    angles = np.arange(12) * (2 * np.pi / 12)
    sample_x = np.rint(mx.mean() + np.cos(angles) * bw * 0.75).astype(int)
    sample_y = np.rint(my.mean() + np.sin(angles) * bh * 0.75).astype(int)
    sample_x = np.clip(sample_x, 0, patch.shape[1] - 1)
    sample_y = np.clip(sample_y, 0, patch.shape[0] - 1)
    if np.count_nonzero(patch[sample_y, sample_x] > interior + 10) < 8:
        return None
    padding = round(min(bw, bh) * 0.75)
    return cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1))).astype(bool)


def _eye_candidate(background: np.ndarray, radius: int, openings: list,
                   point: tuple) -> tuple | None:
    """독립적인 한 후보의 눈 형상 확인 및 보호 영역 반환."""
    height, width = background.shape
    y, x = point
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    patch = background[y0:y1, x0:x1]
    cy, cx = int(y - y0), int(x - x0)
    center = patch[max(0, cy - 5):cy + 6, max(0, cx - 5):cx + 6]
    # 온전한 11×11 중앙 패치의 20백분위는 보간 없는 24번 순서 통계량.
    dark = (float(np.partition(center, 24, axis=None)[24]) if center.size == 121
            else float(np.percentile(center, 20)))
    # 실수 임계값 네 개의 판정 결과를 순서 코드로 보존.
    levels = np.asarray([dark + offset for offset in (8, 12, 16, 24)], dtype=patch.dtype)
    ordinal = np.zeros(patch.shape, np.uint8)
    for level in levels:
        ordinal += patch > level
    for opening in openings:
        # 중첩 이진 개방 결과를 단일 순서 코드 닫기 연산으로 복원.
        closed = cv2.morphologyEx(ordinal, cv2.MORPH_CLOSE, opening)
        for offset in range(4):
            region = (closed <= offset).astype(np.uint8)
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            possible = False
            if _native is not None:
                possible = _native.possible_eye(contours, *patch.shape, cy, cx)
            else:
                for contour in contours:
                    left, top, bw, bh = cv2.boundingRect(contour)
                    if (min(bw, bh) >= 12 and max(bw, bh) <= 128
                            and max(bw, bh) <= 1.7 * min(bw, bh)
                            and left > 0 and top > 0
                            and left + bw < patch.shape[1] and top + bh < patch.shape[0]
                            and left < cx + 13 and left + bw > cx - 12
                            and top < cy + 13 and top + bh > cy - 12):
                        possible = True
                        break
            if not possible:
                continue
            # 4방향 외부 배경을 표시한 뒤 남은 닫힌 구멍 채움.
            flooded = np.pad(region, 1)
            cv2.floodFill(flooded, None, (0, 0), 2, flags=4)
            region = (flooded[1:-1, 1:-1] != 2).astype(np.uint8)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
            nearby = np.unique(labels[max(0, cy - 12):cy + 13, max(0, cx - 12):cx + 13])
            for label in nearby[nearby > 0]:
                keep = _round_eye_component(patch, labels, stats, label)
                if keep is not None:
                    return y0, y1, x0, x1, keep
    return None


def _eye_regions_at_scale(background: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, int]:
    """밝은 주변에 둘러싸인 작고 둥근 암부를 홍채/동공 보호 구역으로 선택."""
    height, width = background.shape
    protected = np.zeros((height, width), bool)
    if not len(xs):
        return protected, 0
    local = cv2.blur(background, (5, 5))
    near = cv2.blur(background, (31, 31))
    far = cv2.blur(background, (81, 81))
    bowl = np.maximum(near[ys, xs], far[ys, xs]) - local[ys, xs]
    interest = (local[ys, xs] < 100) & (bowl >= 12)
    del local, near, far
    if np.any(interest):
        mean = cv2.blur(background, (11, 11))
        variance = np.maximum(cv2.blur(background * background, (11, 11)) - mean * mean, 0)
        smooth_interior = cv2.erode(variance, np.ones((15, 15), np.uint8))
        interest &= smooth_interior[ys, xs] <= 9
        del mean, variance, smooth_interior
    count = 0
    radius = min(80, max(40, int(min(height, width) * 0.08)))
    openings = [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in (13, 21)]
    points = list(zip(ys[interest], xs[interest]))
    classify = partial(_eye_candidate, background, radius, openings)
    with _executor(max_workers=_WORKERS) as workers:
        for start in range(0, len(points), 32):
            active = [point for point in points[start:start + 32] if not protected[point]]
            proposals = workers.map(classify, active) if len(active) >= 8 else map(classify, active)
            # 형상 판정만 병렬화하고 중복 보호와 영역 수는 원래 후보 순서로 반영.
            for point, proposal in zip(active, proposals):
                if proposal is not None and not protected[point]:
                    y0, y1, x0, x1, keep = proposal
                    protected[y0:y1, x0:x1] |= keep
                    count += 1
    return protected, count


def _eye_regions(background: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, int]:
    """영상 피라미드의 각 크기에서 같은 눈 형상을 확인하고 원래 좌표로 통합."""
    height, width = background.shape
    protected, count = _eye_regions_at_scale(background, ys, xs)
    if not len(xs):
        return protected, count
    mean = cv2.blur(background, (11, 11))
    smooth_interior = None
    angles = np.arange(12) * (2 * np.pi / 12)
    divisor = 2
    while min(height, width) // divisor >= 64:
        remaining = ~protected[ys, xs]
        if smooth_interior is not None:
            remaining &= smooth_interior
        if not np.any(remaining):
            break
        index = np.flatnonzero(remaining)
        candidate_y, candidate_x = ys[index], xs[index]
        bright = np.zeros(len(index), np.uint8)
        possible = np.ones(len(index), bool)
        # 앞의 다섯 방향이 모두 어두우면 나머지가 밝아도 8/12 조건을 통과할 수 없다.
        for begin, end in ((0, 5), (5, 12)):
            active = np.flatnonzero(possible)
            if not len(active):
                break
            sample_x = np.rint(candidate_x[active, None] + np.cos(angles[begin:end]) * (32 * divisor)).astype(int)
            sample_y = np.rint(candidate_y[active, None] + np.sin(angles[begin:end]) * (32 * divisor)).astype(int)
            sample_x = np.clip(sample_x, 0, width - 1)
            sample_y = np.clip(sample_y, 0, height - 1)
            bright[active] += np.count_nonzero(
                background[sample_y, sample_x] > mean[candidate_y[active], candidate_x[active], None] + 10, axis=1).astype(np.uint8)
            possible &= bright + (12 - end) >= 8
        index = index[possible]
        if len(index) and smooth_interior is None:
            variance = np.maximum(cv2.blur(background * background, (11, 11)) - mean * mean, 0)
            smooth_interior = cv2.erode(variance, np.ones((15, 15), np.uint8))[ys, xs] <= 9
        if smooth_interior is not None:
            index = index[smooth_interior[index]]
        candidate_y, candidate_x = ys[index], xs[index]
        if not len(candidate_x):
            divisor *= 2
            continue
        shape = (width // divisor, height // divisor)
        reduced = cv2.resize(background, shape, interpolation=cv2.INTER_AREA)
        scaled_x = np.minimum(candidate_x // divisor, shape[0] - 1)
        scaled_y = np.minimum(candidate_y // divisor, shape[1] - 1)
        _, distinct = np.unique(np.stack((scaled_y, scaled_x), axis=1), axis=0, return_index=True)
        scaled_y, scaled_x = scaled_y[distinct], scaled_x[distinct]
        keep, found = _eye_regions_at_scale(reduced, scaled_y, scaled_x)
        if found:
            protected |= cv2.resize(keep.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            count += found
        divisor *= 2
    return protected, count


def _surface_floors(background: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """후보 코어를 제외한 31px 둘레의 밝은 표면 지지를 제한된 묶음으로 계산."""
    height, width = background.shape
    py = ys[:, None] + _SURFACE_Y
    px = xs[:, None] + _SURFACE_X
    valid = (py >= 0) & (py < height) & (px >= 0) & (px < width)
    samples = background[np.clip(py, 0, height - 1), np.clip(px, 0, width - 1)]
    samples[~valid] = np.inf
    samples.sort(axis=1)
    rank = (np.sum(valid, axis=1) - 1).astype(np.float32) * np.float32(0.95)
    lower = np.floor(rank).astype(int)
    upper = np.ceil(rank).astype(int)
    rows = np.arange(len(ys))
    first, second = samples[rows, lower], samples[rows, upper]
    fraction = rank - lower.astype(np.float32)
    difference = second - first
    return np.where(fraction >= 0.5, second - difference * (1 - fraction),
                    first + difference * fraction)


def _surrounded_by_strokes(background: np.ndarray, y: int, x: int) -> bool:
    """넓은 암부 표면과 달리 둘레 잉크가 가는 띠나 유한한 형상인지 확인."""
    height, width = background.shape
    top, bottom = max(0, y - 40), min(height, y + 41)
    left, right = max(0, x - 40), min(width, x + 41)
    patch = background[top:bottom, left:right]
    dark = patch < background[y, x] + 4
    dark = dark.astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    label = labels[y - top, x - left]
    if not label:
        return False
    sx, sy, sw, sh, _ = stats[label]
    borders = (int(sx == 0) + int(sy == 0) + int(sx + sw == right - left)
               + int(sy + sh == bottom - top))
    if borders <= 2:
        return True
    thickness = cv2.distanceTransform(dark, cv2.DIST_L2, 5)
    if np.percentile(thickness[labels == label], 90) <= 10:
        return True
    py = np.clip(y + _SURROUND_Y, 0, height - 1)
    px = np.clip(x + _SURROUND_X, 0, width - 1)
    bright_directions = background[py, px] >= background[y, x] + 4
    return bool(np.any(np.sum(bright_directions, axis=1) >= 6))


def _attached_to_bright_surface(luma: np.ndarray, y: int, x: int, floor: float) -> bool:
    """잔차에서는 분리됐지만 원본의 큰 밝은 면에 연결된 가장자리 보호."""
    height, width = luma.shape
    top, bottom = max(0, y - 6), min(height, y + 7)
    left, right = max(0, x - 6), min(width, x + 7)
    bright = (luma[top:bottom, left:right] >= floor - 24).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    label = labels[y - top, x - left]
    return bool(label and stats[label, cv2.CC_STAT_AREA] > 24)


def _text_points(rgb: np.ndarray, surface: np.ndarray, luma: np.ndarray,
                 residual: np.ndarray,
                 ys: np.ndarray, xs: np.ndarray, *, inverse: bool = False) -> np.ndarray:
    """어두운 평탄 배경의 같은 색 글자 행과 짧은 분리 획 보호."""
    protected = np.zeros(len(ys), bool)
    quiet = cv2.blur((np.abs(residual) <= 8).astype(np.float32), (41, 41))
    interest = quiet[ys, xs] >= 0.45
    if not np.any(interest):
        return protected
    foreground = ((residual >= 12) & (quiet >= 0.4)).astype(np.uint8)
    backgrounds = {}
    protected |= _text_rows(rgb, surface, residual, foreground, ys, xs, interest, backgrounds,
                            inverse=inverse)
    joined = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((5, 1), np.uint8))
    protected |= _text_rows(rgb, surface, residual, joined, ys, xs, interest & ~protected, backgrounds,
                            inverse=inverse)
    return protected


def _smooth_text_background(patch: np.ndarray, minimum_fraction: float,
                            maximum_rms: float) -> bool:
    """문자 사이 배경색의 충분한 지지와 완만한 색 추세 확인."""
    base = np.median(patch.reshape(-1, 3), axis=0)
    samples = np.max(np.abs(patch - base), axis=2) <= 12
    if np.mean(samples) < minimum_fraction:
        return False
    # 상수 배경도 이미 허용 오차 이내면 평면 적합 불필요. 경계값은 기존 계산 유지.
    values = patch[samples]
    constant_error = values.astype(np.float64) - base
    if np.max(np.mean(constant_error * constant_error, axis=0)) < (maximum_rms - 0.001) ** 2:
        return True
    height, width = patch.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    coordinates = np.stack((np.ones_like(xx), xx / width, yy / height), axis=-1)
    plane = np.linalg.lstsq(coordinates[samples], values, rcond=None)[0]
    error = values - coordinates[samples] @ plane
    return bool(np.max(np.sqrt(np.mean(error * error, axis=0))) <= maximum_rms)


def _glyph_relations(boxes: np.ndarray, colors: np.ndarray, centers: np.ndarray,
                     anchors: np.ndarray) -> csr_array:
    """@description 중심선과 색상 범위 기반 글자 관계 계산."""
    heights = boxes[:, 3]
    cy = boxes[:, 1] + heights * 0.5
    # 글자 높이 상한 80px의 중심선 허용차 28px, 반경 180px 이웃의 쌍별 거리 360px.
    points = np.column_stack((centers[anchors, 0].astype(np.float64) / 360,
                              cy[anchors].astype(np.float64) / 28,
                              colors[anchors].astype(np.float64) / 24))
    # float32 경계 반올림을 포함하는 검색 여유, 원래 판정식으로 최종 선택.
    neighbors = cKDTree(points).query_ball_point(points, 1.00001, p=np.inf)
    rows, columns = [], []
    for row, raw in zip(anchors, neighbors):
        column = anchors[np.asarray(raw, int)]
        offset = centers[column].astype(np.float64) - centers[row].astype(np.float64)
        matched = ((np.hypot(offset[:, 0], offset[:, 1]) <= 360)
                   & (np.abs(cy[row] - cy[column]) <= 0.35 * np.maximum(heights[row], heights[column]))
                   & (heights[row] >= 0.8 * heights[column])
                   & (heights[row] <= 1.25 * heights[column])
                   & (np.max(np.abs(colors[row] - colors[column]), axis=1) <= 24))
        column = column[matched]
        rows.append(np.full(len(column), row))
        columns.append(column)
    rows, columns = np.concatenate(rows), np.concatenate(columns)
    return csr_array((np.ones(len(rows), bool), (rows, columns)), shape=(len(boxes), len(boxes)))


def _text_neighborhoods(tree, indices, ys, xs, rgb, surface, residual, colors, inverse=False):
    """@description 후보 묶음별 주변 글자 검색과 RGB 색상 판정."""
    neighborhoods = []
    for start in range(0, len(indices), 128):
        selected = indices[start:start + 128]
        y, x = ys[selected], xs[selected]
        raw = tree.query_ball_point(np.column_stack((x, y)), 180, return_sorted=False)
        counts = np.fromiter(map(len, raw), dtype=int, count=len(raw))
        owner = np.repeat(np.arange(len(selected)), counts)
        nearby = np.concatenate(raw).astype(int)
        base = surface[y, x]
        source = rgb[y, x]
        if inverse:
            base, source = 255 - base, 255 - source
        difference = source - base
        axis = colors[nearby] - base[owner]
        dot = (axis[:, None, :] @ difference[owner, :, None])[:, 0, 0]
        alpha = dot / np.maximum(np.sum(axis * axis, axis=1), 1)
        error = np.max(np.abs(difference[owner] - alpha[:, None] * axis), axis=1)
        limit = 0.1 * residual[y, x].astype(_SCALAR_PRODUCT_DTYPE, copy=False)
        # 기존 float32 배열과 스칼라 비교의 판정 정밀도로 복원.
        limit = np.maximum(10, limit).astype(error.dtype, copy=False)
        same_color = ((alpha >= 0.03) & (alpha <= 1.3)
                      & (error <= limit[owner]))
        boundaries = np.r_[0, counts.cumsum()]
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            neighborhoods.append(nearby[begin:end][same_color[begin:end]])
    return neighborhoods


def _text_rows(rgb: np.ndarray, surface: np.ndarray, residual: np.ndarray,
               foreground: np.ndarray, ys: np.ndarray, xs: np.ndarray,
               interest: np.ndarray, backgrounds: dict, *, inverse: bool = False) -> np.ndarray:
    """같은 색의 글자 행과 행 뒤의 매끈한 배경을 함께 확인."""
    protected = np.zeros(len(ys), bool)
    if not np.any(interest):
        return protected
    _, labels, stats, centers = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    left, top, width, height, area = stats.T
    glyph = ((height >= 5) & (height <= 80) & (width >= 2) & (width <= 2.5 * height)
             & (area >= 6) & (area <= 2000)
             & ((area <= 0.9 * width * height) | (width <= 0.3 * height))
             & ((width >= 0.3 * height) | (area >= 0.5 * width * height)))
    glyph[0] = False
    ids = np.flatnonzero(glyph)
    if not len(ids):
        return protected
    if _native is not None:
        colors = _native.component_colors(rgb, labels, stats, ids, inverse)
    else:
        colors = np.empty((len(ids), 3), np.float32)
        for index, label in enumerate(ids):
            x0, y0, bw, bh, _ = stats[label]
            region = np.s_[y0:y0 + bh, x0:x0 + bw]
            pixels = rgb[region][labels[region] == label]
            if inverse:
                pixels = 255 - pixels
            colors[index] = np.max(pixels, axis=0)
    boxes = stats[ids].astype(np.float32)
    centers = centers[ids].astype(np.float32)
    tree = cKDTree(centers)
    indices = np.flatnonzero(interest)
    neighborhoods = _text_neighborhoods(tree, indices, ys, xs, rgb, surface, residual, colors, inverse)
    relations = None
    reuse_relations = None
    smooth_rows = {}
    for index, nearby in zip(indices, neighborhoods):
        y, x = int(ys[index]), int(xs[index])
        if not len(nearby):
            continue
        bx, by, bw, bh, _ = boxes[nearby].T
        dx = np.maximum(np.maximum(bx - x, x - (bx + bw)), 0)
        dy = np.maximum(np.maximum(by - y, y - (by + bh)), 0)
        local_stroke = (np.hypot(dx, dy) <= 6) & (boxes[nearby, 4] > 24)
        for component in np.flatnonzero(local_stroke):
            margin = max(8, round(float(bh[component]) * 0.5))
            x0 = max(0, int(min(x, bx[component])) - margin)
            x1 = min(rgb.shape[1], int(max(x + 1, bx[component] + bw[component])) + margin)
            y0 = max(0, int(min(y, by[component])) - margin)
            y1 = min(rgb.shape[0], int(max(y + 1, by[component] + bh[component])) + margin)
            key = (x0, y0, x1, y1, 0.6, 4)
            if key not in backgrounds:
                patch = rgb[y0:y1, x0:x1]
                if inverse:
                    patch = 255 - patch
                backgrounds[key] = _smooth_text_background(patch, 0.6, 4)
            if backgrounds[key]:
                protected[index] = True
                break
        if protected[index] or len(nearby) < 3:
            continue
        if reuse_relations is None:
            needed = np.unique(np.concatenate(neighborhoods))
            # 적은 후보가 넓게 흩어진 경우 행별 관계 구축보다 개별 비교가 저렴.
            reuse_relations = len(needed) <= 4 * len(neighborhoods)
            if reuse_relations:
                points = centers[needed]
                pair_count = np.sum(cKDTree(points).query_ball_point(points, 360, return_length=True))
                reuse_relations = pair_count < sum(len(neighbors) ** 2 for neighbors in neighborhoods)
            if reuse_relations:
                relations = _glyph_relations(boxes, colors, centers, needed)
                relation_counts = np.diff(relations.indptr)
                row_index = np.repeat(np.arange(len(boxes)), relation_counts)
                related = boxes[relations.indices]
                limits = np.full((len(boxes), 4), np.inf, np.float32)
                limits[:, 2:] = -np.inf
                np.minimum.at(limits[:, 0], row_index, related[:, 0])
                np.minimum.at(limits[:, 1], row_index, related[:, 1])
                np.maximum.at(limits[:, 2], row_index, related[:, 0] + related[:, 2])
                np.maximum.at(limits[:, 3], row_index, related[:, 1] + related[:, 3])
        if reuse_relations:
            # 전체 관계에서도 세 글자를 채우지 못하는 기준 획은 행 후보에서 제외.
            bounds = limits[nearby]
            anchors = np.flatnonzero((relation_counts[nearby] >= 3)
                                     & (bounds[:, 2] - bounds[:, 0] >= 2 * bh)
                                     & (x >= bounds[:, 0] - bh) & (x <= bounds[:, 2] + bh)
                                     & (y >= bounds[:, 1] - 2) & (y <= bounds[:, 3] + 2))
            if not len(anchors):
                continue
            aligned = relations[nearby[anchors]][:, nearby].toarray()
            anchor_height = bh[anchors]
        else:
            paint = colors[nearby]
            if _native is not None:
                aligned = _native.aligned_rows(by, bh, paint)
            else:
                cy = by + bh * 0.5
                aligned = ((np.abs(cy[:, None] - cy) <= 0.35 * np.maximum(bh[:, None], bh))
                           & (bh[:, None] >= 0.8 * bh) & (bh[:, None] <= 1.25 * bh)
                           & (np.max(np.abs(paint[:, None] - paint), axis=2) <= 24))
            anchor_height = bh
        if _native is not None:
            row_left, row_right, row_top, row_bottom, rows = _native.row_bounds(
                aligned.view(np.uint8), bx, by, bw, bh, np.hypot(dx, dy), anchor_height, x, y)
        else:
            row_left = np.min(np.where(aligned, bx, np.inf), axis=1)
            row_right = np.max(np.where(aligned, bx + bw, -np.inf), axis=1)
            row_top = np.min(np.where(aligned, by, np.inf), axis=1)
            row_bottom = np.max(np.where(aligned, by + bh, -np.inf), axis=1)
            near = np.min(np.where(aligned, np.hypot(dx, dy), np.inf), axis=1)
            rows = ((np.sum(aligned, axis=1) >= 3)
                    & (row_right - row_left >= 2 * anchor_height)
                    & (x >= row_left - anchor_height) & (x <= row_right + anchor_height)
                    & (y >= row_top - 2) & (y <= row_bottom + 2) & (near <= anchor_height))
        for row in np.flatnonzero(rows):
            members = tuple(nearby[aligned[row]])
            if members not in smooth_rows:
                margin = round(float(anchor_height[row]) * 0.5)
                x0 = max(0, int(row_left[row]) - margin)
                x1 = min(rgb.shape[1], int(row_right[row]) + margin)
                y0 = max(0, int(row_top[row]) - margin)
                y1 = min(rgb.shape[0], int(row_bottom[row]) + margin)
                key = (x0, y0, x1, y1, 0.5, 6)
                if key not in backgrounds:
                    patch = rgb[y0:y1, x0:x1]
                    if inverse:
                        patch = 255 - patch
                    backgrounds[key] = _smooth_text_background(patch, 0.5, 6)
                smooth_rows[members] = backgrounds[key]
            if smooth_rows[members]:
                protected[index] = True
                break
    return protected


def _median_surface(rgb: np.ndarray) -> np.ndarray:
    """채널별 9픽셀 중앙값으로 기존 점 판정의 표면색 계산."""
    quantized = np.rint(rgb)
    np.clip(quantized, 0, 255, out=quantized)
    quantized = quantized.astype(np.uint8)
    if (np.array_equal(quantized[..., 0], quantized[..., 1])
            and np.array_equal(quantized[..., 1], quantized[..., 2])):
        return np.repeat(_median_blur(quantized[..., 0], 9)[..., None], 3, axis=2).astype(np.float32)
    return _median_blur(quantized, 9).astype(np.float32)


def _detect(rgb: np.ndarray, *, surface: np.ndarray | None = None,
            excluded: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """다중 크기 양의 봉우리에서 작은 중성 잔차와 비주기적 점 선택."""
    height, width = rgb.shape[:2]
    luma = rgb @ _LUMA
    surface = _median_surface(rgb) if surface is None else surface
    background = surface @ _LUMA
    residual = luma - background
    response = np.full((height, width), -np.inf, np.float32)
    scale = np.zeros((height, width), np.uint8)
    for index, sigma in enumerate(_SIGMAS):
        fine = cv2.GaussianBlur(luma, (0, 0), sigma)
        coarse = cv2.GaussianBlur(luma, (0, 0), sigma * 1.6)
        current = fine - coarse
        larger = current > response
        scale[larger] = index
        np.maximum(response, current, out=response)
    seeds = (response >= cv2.dilate(response, np.ones((3, 3), np.uint8))) & (response >= 1.5)
    seeds &= (residual >= 12) & (background < 150)
    seeds[:9] = seeds[-9:] = False
    seeds[:, :9] = seeds[:, -9:] = False
    ys, xs = np.nonzero(seeds)
    order = np.argsort(response[ys, xs])[::-1]
    ys, xs = ys[order], xs[order]
    contrasts = residual[ys, xs]
    delta = rgb[ys, xs] - surface[ys, xs]
    eligible = (np.min(delta, axis=1) > 0) & (np.ptp(delta, axis=1) <= np.maximum(6, 0.35 * contrasts))
    ys, xs, contrasts = ys[eligible], xs[eligible], contrasts[eligible]
    # 후보 추출이 끝난 전역 응답과 마지막 필터 작업 버퍼 수명 종료.
    del fine, coarse, current, larger, response, seeds, order, delta, eligible
    classify = partial(_eligible_cores, luma, residual, scale, ys, xs, contrasts)
    with _executor(max_workers=_WORKERS) as workers:
        geometry = [batch for batch in workers.map(classify, range(0, len(xs), _BATCH_SIZE))
                    if batch is not None]
    del classify, scale
    occupied = np.zeros((height, width), bool)
    if geometry:
        ys, xs, contrasts, cores, areas, mx, my = (np.concatenate(parts) for parts in zip(*geometry))
        selected = []
        for start in range(0, len(xs), _BATCH_SIZE):
            blocked = _dilate_cores(cores[start:start + _BATCH_SIZE].astype(np.uint8)).astype(bool)
            blocked[:, 4:9, 4:9] = True
            if _native is not None:
                selected.extend(_native.select_bright(ys, xs, blocked.view(np.uint8),
                                                       occupied.view(np.uint8), start))
            else:
                for offset in range(len(blocked)):
                    index = start + offset
                    y, x = ys[index], xs[index]
                    if not occupied[y, x]:
                        occupied[y - 6:y + 7, x - 6:x + 7] |= blocked[offset]
                        selected.append(index)
        ys, xs, contrasts, cores, areas = (a[selected] for a in (ys, xs, contrasts, cores, areas))
        points = np.stack((xs + mx[selected] - 6, ys + my[selected] - 6), axis=1).astype(np.float32)
        regular = _regular_points(points, contrasts.astype(np.float64), areas)
        ys, xs, contrasts, cores = (a[~regular] for a in (ys, xs, contrasts, cores))
    else:
        ys, xs, contrasts = np.empty(0, int), np.empty(0, int), np.empty(0, np.float32)
        cores = np.empty((0, 13, 13), bool)
        regular = np.empty(0, bool)
    del geometry, occupied
    eye_mask, eye_regions = _eye_regions(background, ys, xs)
    eye_points = eye_mask[ys, xs]
    ys, xs, contrasts, cores = (a[~eye_points] for a in (ys, xs, contrasts, cores))
    if excluded is not None:
        selected = ~excluded[ys, xs]
        ys, xs, contrasts, cores = (a[selected] for a in (ys, xs, contrasts, cores))
    dx = cv2.Sobel(background, cv2.CV_32F, 1, 0, ksize=1, scale=0.5)
    dy = cv2.Sobel(background, cv2.CV_32F, 0, 1, ksize=1, scale=0.5)
    flat_fraction = cv2.blur((dx * dx + dy * dy <= 0.25).astype(np.float32), (257, 257))
    del dx, dy
    text_points = _text_points(rgb, surface, luma, residual, ys, xs)
    ys, xs, contrasts, cores = (a[~text_points] for a in (ys, xs, contrasts, cores))
    support_max = cv2.dilate(background, np.ones((31, 31), np.uint8))
    weight = np.zeros((height, width), np.float32)
    for start in range(0, len(xs), _BATCH_SIZE):
        y, x = ys[start:start + _BATCH_SIZE], xs[start:start + _BATCH_SIZE]
        contrast, core = contrasts[start:start + _BATCH_SIZE], cores[start:start + _BATCH_SIZE]
        shoulder = _dilate_cores(core.astype(np.float32))
        feather = np.maximum(_blur_patches(shoulder, 0.65), core)
        patch_y = y[:, None, None] + _PATCH_AXIS[None, :, None] - 6
        patch_x = x[:, None, None] + _PATCH_AXIS[None, None, :] - 6
        footprint = np.clip((residual[patch_y, patch_x] - 3) / np.maximum(6, 0.25 * contrast[:, None, None]), 0, 1)
        feather *= footprint
        # 넓은 밝기 지지로 잉크에 오염된 중앙값 배제, 암부 후보의 표면 아래 감쇠 제한.
        flat = ((flat_fraction[y, x] >= 0.2)
                & ((support_max[y, x] >= 150) | (luma[y, x] <= support_max[y, x] + 24)))
        floors = np.full(len(y), -np.inf, np.float32)
        floors[flat] = _surface_floors(background, y[flat], x[flat])
        bright = flat & (floors >= 150)
        matched = flat & (np.abs(luma[y, x] - floors) <= 24)
        strokes = np.zeros(len(y), bool)
        for index in np.flatnonzero(bright | matched):
            strokes[index] = _surrounded_by_strokes(background, int(y[index]), int(x[index]))
            if bright[index] and not strokes[index]:
                strokes[index] = _attached_to_bright_surface(luma, int(y[index]), int(x[index]), float(floors[index]))
        feather[bright & strokes] = 0
        capped = matched & ~bright & strokes
        py, px = patch_y[capped], patch_x[capped]
        positive = np.maximum(rgb[py, px] - surface[py, px], 0) @ _LUMA
        limit = np.maximum(luma[py, px] - floors[capped, None, None], 0)
        fraction = np.divide(limit, positive, out=np.zeros_like(limit), where=positive > 0)
        feather[capped] *= np.minimum(fraction, 1)
        np.maximum.at(weight.ravel(), (patch_y * width + patch_x).ravel(), feather.ravel())
    weight[eye_mask] = 0
    report = {
        "stage": "sparkle", "candidate_count": len(contrasts),
        "regular_highlights_protected": int(regular.sum()),
        "eye_regions_protected": eye_regions,
        "eye_catchlights_protected": int(eye_points.sum()),
        "masked_pixels": int(np.count_nonzero(weight)),
        "mask_fraction": float(np.mean(weight > 0)),
        "median_contrast_dn": float(np.median(contrasts)) if len(contrasts) else None,
        "maximum_contrast_dn": float(np.max(contrasts)) if len(contrasts) else None,
        "parameters": {"sigmas_px": list(_SIGMAS), "residual_min_dn": 12,
                       "compact_area_max_px": 12, "wide_area_max_px": 24,
                       "attenuation": 0.95},
    }
    return weight, surface, report


def _dark_stroke_chains(points: np.ndarray, colors: np.ndarray, contrasts: np.ndarray,
                        areas: np.ndarray) -> np.ndarray:
    """불균일한 간격으로 이어지는 비슷한 색의 세 점 이상을 박음질/획으로 보호."""
    protected = np.zeros(len(points), bool)
    if len(points) < 3:
        return protected
    tree = cKDTree(points)
    distance, neighbor = tree.query(points, k=min(9, len(points)), distance_upper_bound=40)
    for start in range(0, len(points), _BATCH_SIZE):
        index = np.arange(start, min(start + _BATCH_SIZE, len(points)))
        distances = distance[index, 1:]
        raw = neighbor[index, 1:]
        neighbors = np.minimum(raw, len(points) - 1)
        valid = ((raw < len(points)) & (distances >= 4) & (contrasts[index, None] >= 6)
                 & (areas[neighbors] >= 0.25 * areas[index, None])
                 & (areas[neighbors] <= 4 * areas[index, None])
                 & (np.max(np.abs(colors[neighbors] - colors[index, None]), axis=2)
                    <= np.maximum(12, 0.75 * contrasts[index, None])))
        vectors = points[neighbors] - points[index, None]
        unit = np.divide(vectors, distances[..., None], out=np.zeros_like(vectors),
                         where=valid[..., None])
        opposite = unit @ unit.transpose(0, 2, 1) < -0.93
        paired = valid[:, :, None] & valid[:, None, :]
        numerator = np.minimum(distances[:, :, None], distances[:, None, :])
        denominator = np.maximum(distances[:, :, None], distances[:, None, :])
        ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=paired)
        pairs = opposite & (ratio >= 0.35) & paired
        rows, first, second = np.nonzero(pairs)
        protected[index[rows]] = True
        protected[neighbors[rows, first]] = True
        protected[neighbors[rows, second]] = True
    return protected


def _dark_points(rgb: np.ndarray, surface: np.ndarray, bright_weight: np.ndarray) -> tuple:
    """일관된 주변 표면보다 어두운 소형 코어 선택 및 실제 잉크/피부 보호."""
    height, width = rgb.shape[:2]
    luma, background = rgb @ _LUMA, surface @ _LUMA
    residual = background - luma
    scatter = cv2.blur(np.abs(residual), (13, 13))
    seeds = ((residual >= np.maximum(1.25, 4 * scatter))
             & (residual >= cv2.dilate(residual, np.ones((3, 3), np.uint8))))
    structure = cv2.morphologyEx(background, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) >= 16
    structure = cv2.dilate(structure.astype(np.uint8), np.ones((21, 21), np.uint8))
    seeds &= structure == 0
    seeds[:9] = seeds[-9:] = False
    seeds[:, :9] = seeds[:, -9:] = False
    ys, xs = np.nonzero(seeds)
    order = np.argsort(residual[ys, xs])[::-1]
    ys, xs = ys[order], xs[order]
    del scatter, seeds, structure, order
    selected_parts = []
    for start in range(0, len(xs), _BATCH_SIZE):
        y, x = ys[start:start + _BATCH_SIZE], xs[start:start + _BATCH_SIZE]
        py = y[:, None, None] + _PATCH_AXIS[None, :, None] - 6
        px = x[:, None, None] + _PATCH_AXIS[None, None, :] - 6
        patches = luma[py, px]
        samples = patches[:, _RING]
        reference = np.median(samples, axis=1)
        texture = 1.4826 * np.median(np.abs(samples - reference[:, None]), axis=1)
        contrast = reference - luma[y, x]
        cores, area, bw, bh, mx, my, ratio = _core_geometry(
            reference[:, None, None] - patches, contrast, minimum=0.9)
        selected = ((area > 0) & (area <= 24) & (np.maximum(bw, bh) <= 7) & (ratio <= 6.25)
                    & (contrast >= np.maximum(1.25, 3.5 * texture))
                    & ((contrast >= 12) | ((area <= 9) & (np.maximum(bw, bh) <= 3))))
        if not np.any(selected):
            continue
        y, x, reference, texture, contrast, cores, area, mx, my, patches, samples = (
            a[selected] for a in (y, x, reference, texture, contrast, cores, area, mx, my,
                                  patches, samples))
        spread = np.percentile(samples, 90, axis=1) - np.percentile(samples, 10, axis=1)
        selected = spread <= np.maximum(6, 0.9 * contrast)
        # 주변 네 방향의 중앙값이 맞아야 경계/음영을 가로질러 채우지 않는다.
        sides = np.stack([np.median(patches[:, mask], axis=1) for mask in
                          (_PATCH_X <= 2, _PATCH_X >= 10, _PATCH_Y <= 2, _PATCH_Y >= 10)], axis=1)
        selected &= np.ptp(sides, axis=1) <= np.maximum(3, 0.4 * contrast)
        # 밝은 표면의 깊은 검정은 동공, 점, 인쇄 잉크로 보수적으로 보존.
        selected &= ~((reference >= 100) & (contrast >= 48)
                      & (luma[y, x] < reference * 0.5))
        if not np.any(selected):
            continue
        y, x, reference, texture, contrast, cores, area, mx, my, spread = (
            a[selected] for a in (y, x, reference, texture, contrast, cores, area, mx, my, spread))
        py = y[:, None, None] + _PATCH_AXIS[None, :, None] - 6
        px = x[:, None, None] + _PATCH_AXIS[None, None, :] - 6
        # 중앙값 표면에도 남는 강한 경계 주변의 회색 획/그림자는 채우지 않는다.
        base = background[py, px][:, 2:11, 2:11]
        edge = (np.any(np.abs(np.diff(base, axis=1)) >= 8, axis=(1, 2))
                | np.any(np.abs(np.diff(base, axis=2)) >= 8, axis=(1, 2)))
        selected = ~edge
        if not np.any(selected):
            continue
        y, x, reference, texture, contrast, cores, area, mx, my, spread, py, px = (
            a[selected] for a in (y, x, reference, texture, contrast, cores, area, mx, my,
                                  spread, py, px))
        reference_rgb = np.median(rgb[py, px][:, _RING], axis=1)
        delta = reference_rgb - rgb[y, x]
        selected = ((np.min(delta, axis=1) >= -0.75)
                     & (np.max(delta, axis=1) >= 1.75)
                     & (np.ptp(delta, axis=1) <= np.maximum(4, 0.6 * contrast)))
        r, g, b = reference_rgb.T
        skin = ((reference >= 25) & (r - g >= 5) & (g - b >= 2)
                & (r <= 3 * np.maximum(g, 1)) & (b >= 0.1 * np.maximum(r, 1)))
        selected &= ~skin
        if np.any(selected):
            # 매끈한 바탕의 강한 점은 희미한 표면 잡음 대신 의도적인 표식으로 보존.
            flat_mark = (texture <= 0.75) & (spread <= 2) & (contrast >= 12)
            selected_parts.append(tuple(a[selected] for a in
                                        (y, x, contrast, cores, area, mx, my, reference_rgb, flat_mark)))
    weight = np.zeros((height, width), np.float32)
    target = surface
    report = {"dark_candidate_count": 0, "dark_masked_pixels": 0,
              "dark_regular_points_protected": 0, "dark_text_points_protected": 0,
              "dark_eye_points_protected": 0, "dark_stroke_points_protected": 0,
              "dark_flat_marks_protected": 0}
    if not selected_parts:
        return weight, target, report
    target = surface.copy()
    ys, xs, contrasts, cores, areas, mx, my, references, flat_marks = (
        np.concatenate(parts) for parts in zip(*selected_parts))
    occupied = np.zeros((height, width), bool)
    if _native is not None:
        distinct = _native.select_distinct(ys, xs, occupied.view(np.uint8))
    else:
        distinct = []
        for index, (y, x) in enumerate(zip(ys, xs)):
            if not occupied[y, x]:
                occupied[y - 2:y + 3, x - 2:x + 3] = True
                distinct.append(index)
    ys, xs, contrasts, cores, areas, mx, my, references, flat_marks = (
        a[distinct] for a in (ys, xs, contrasts, cores, areas, mx, my, references, flat_marks))
    points = np.stack((xs + mx - 6, ys + my - 6), axis=1).astype(np.float32)
    _, distinct = np.unique(points, axis=0, return_index=True)
    distinct.sort()
    points = points[distinct]
    ys, xs, contrasts, cores, areas, references, flat_marks = (
        a[distinct] for a in (ys, xs, contrasts, cores, areas, references, flat_marks))
    regular = _regular_points(points, contrasts.astype(np.float64), areas)
    text_reference = cv2.morphologyEx(luma, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    text_residual = np.maximum(residual, text_reference - luma)
    del text_reference
    # 검출에서 탈락한 긴 획도 연결 증거로 사용해 박음질 끝의 짧은 획 보호.
    _, labels, stats, centers = cv2.connectedComponentsWithStats(
        (text_residual >= 6).astype(np.uint8), connectivity=8)
    context = ((stats[:, 4] >= 3) & (stats[:, 4] <= 2000)
               & (np.maximum(stats[:, 2], stats[:, 3]) <= 80))
    context[0] = False
    ids = np.flatnonzero(context)
    if _native is not None:
        colors, context_contrast = _native.component_colors(rgb, labels, stats, ids,
                                                            residual=text_residual)
    else:
        colors = np.empty((len(ids), 3), np.float32)
        context_contrast = np.empty(len(ids), np.float32)
        for index, label in enumerate(ids):
            left, top, bw, bh, _ = stats[label]
            region = np.s_[top:top + bh, left:left + bw]
            mask = labels[region] == label
            colors[index] = np.min(rgb[region][mask], axis=0)
            context_contrast[index] = np.max(text_residual[region][mask])
    strokes = _dark_stroke_chains(
        np.concatenate((points, centers[ids].astype(np.float32))),
        np.concatenate((rgb[ys, xs], colors)), np.concatenate((contrasts, context_contrast)),
        np.concatenate((areas, stats[ids, cv2.CC_STAT_AREA])))[:len(xs)]
    del labels, stats, centers, colors, context_contrast
    # 글자 판정에 필요한 픽셀과 배경 영역만 반전하여 전역 RGB 복제 제외.
    text = _text_points(rgb, surface, luma, text_residual, ys, xs, inverse=True)
    del text_residual, residual, luma
    eye_mask, _ = _eye_regions(background, ys, xs)
    del background
    eyes = eye_mask[ys, xs]
    keep = ~(regular | strokes | text | eyes | flat_marks)
    report.update({"dark_regular_points_protected": int(regular.sum()),
                   "dark_text_points_protected": int(text.sum()),
                   "dark_eye_points_protected": int(eyes.sum()),
                   "dark_stroke_points_protected": int(strokes.sum()),
                   "dark_flat_marks_protected": int(flat_marks.sum())})
    ys, xs, contrasts, cores, references = (
        a[keep] for a in (ys, xs, contrasts, cores, references))
    for start in range(0, len(xs), _BATCH_SIZE):
        y, x = ys[start:start + _BATCH_SIZE], xs[start:start + _BATCH_SIZE]
        contrast = contrasts[start:start + _BATCH_SIZE]
        core = cores[start:start + _BATCH_SIZE]
        feather = np.maximum(_blur_patches(_dilate_cores(core.astype(np.float32)), 0.65), core)
        py = y[:, None, None] + _PATCH_AXIS[None, :, None] - 6
        px = x[:, None, None] + _PATCH_AXIS[None, None, :] - 6
        reference = references[start:start + _BATCH_SIZE]
        deficit = (reference[:, None, None] - rgb[py, px]) @ _LUMA
        feather *= np.clip((deficit - 0.25) / np.maximum(1, 0.25 * contrast[:, None, None]), 0, 1)
        # 높은 가중치의 후보가 주변 RGB 기준도 소유하도록 같은 순서로 기록.
        if _native is not None:
            _native.scatter_dark(y, x, feather, reference, weight, target)
        else:
            for index in range(len(y)):
                yy, xx = py[index], px[index]
                stronger = feather[index] > weight[yy, xx]
                weight[yy, xx] = np.maximum(weight[yy, xx], feather[index])
                target[yy, xx] = np.where(stronger[..., None], reference[index], target[yy, xx])
    weight[(bright_weight > 0) | eye_mask] = 0
    report.update({"dark_candidate_count": len(xs), "dark_masked_pixels": int(np.count_nonzero(weight))})
    return weight, target, report


def analyze(rgb: np.ndarray) -> dict:
    """밝은 점과 어두운 점의 후보 수와 대비 분석."""
    weight, surface, report = _detect(rgb)
    report.update(_dark_points(rgb, surface, weight)[2])
    return report


def apply(rgb: np.ndarray, strength: float = 1.0) -> tuple[np.ndarray, dict]:
    """국소 표면 쪽으로 양/음의 점 잔차와 부드러운 후광 감쇠."""
    if strength == 0:
        return rgb.copy(), {"stage": "sparkle", "strength": 0.0,
                            "detection_measured": False, "candidate_count": None,
                            "changed_pixels": 0, "correction_rms_dn": 0.0}
    weight, surface, report = _detect(rgb)
    correction = rgb - surface
    np.maximum(correction, 0, out=correction)
    correction *= (weight * min(strength * 0.95, 1))[..., None]
    output = (rgb - correction).astype(np.float32, copy=False)
    dark_weight, dark_surface, dark_report = _dark_points(rgb, surface, weight)
    for start in range(0, rgb.shape[0], 64):
        region = np.s_[start:start + 64]
        dark_correction = np.maximum(dark_surface[region] - rgb[region], 0)
        dark_correction *= (dark_weight[region] * min(strength * 0.95, 1))[..., None]
        output[region] += dark_correction
        correction[region] -= dark_correction
    report.update(dark_report)
    report.update({"strength": float(strength), "detection_measured": True,
                   "changed_pixels": int(np.count_nonzero(np.any(output != rgb, axis=2))),
                   "correction_rms_dn": float(np.sqrt(np.mean(correction ** 2)))})
    return output, report


def _saved_view(rgb: np.ndarray, bit_depth: int) -> np.ndarray:
    """실제 버퍼를 유지하고 8/16비트 저장 시 보일 값으로 판정용 영상 생성."""
    scale = 1 if bit_depth == 8 else 257
    decision = np.clip(rgb, 0, 255)
    decision *= scale
    np.rint(decision, out=decision)
    decision /= scale
    return decision


def apply_final(rgb: np.ndarray, strength: float = 1.0,
                bit_depth: int = 8) -> tuple[np.ndarray, dict]:
    """저장 직전의 양방향 잔차를 기존 강도로 감쇠하고 float 버퍼 유지.

    저장 비트 깊이의 임시 영상으로 먼저 밝은 점 판정, 밝은 점 보정 후 다시
    같은 저장값 기준으로 어두운 점 판정. 실제 반올림은 이미지 저장 시 한 번뿐이다.
    """
    if strength == 0:
        return rgb.copy(), {"stage": "sparkle_final", "strength": 0.0,
                            "detection_measured": False, "changed_pixels": 0}
    decision = _saved_view(rgb, bit_depth)
    surface = _median_surface(decision)
    # 추가 보정에서 피부의 미세 묘사와 실제 획/경계 주변을 보수적으로 유지.
    background = surface @ _LUMA
    red, green, blue = surface[..., 0], surface[..., 1], surface[..., 2]
    skin = ((background >= 25) & (red - green >= 5) & (green - blue >= 2)
            & (red <= 3 * np.maximum(green, 1)) & (blue >= 0.1 * np.maximum(red, 1)))
    structure = cv2.morphologyEx(background, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) >= 16
    structure = cv2.dilate(structure.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    protected = skin | structure
    # 규칙 배열과 눈 문맥은 모두 유지하고, 보정 발자국 전체가 보호된 후보만 조기 제외.
    excluded = cv2.erode(protected.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    weight, surface, report = _detect(decision, surface=surface, excluded=excluded)
    report["final_bright_protected_pixels"] = int(np.count_nonzero((weight > 0) & protected))
    weight[protected] = 0
    report["masked_pixels"] = int(np.count_nonzero(weight))
    report["mask_fraction"] = float(np.mean(weight > 0))
    gain = min(strength * 0.95, 1)
    output = np.empty(rgb.shape, np.float32)
    for start in range(0, rgb.shape[0], 64):
        region = np.s_[start:start + 64]
        correction = np.maximum(rgb[region] - surface[region], 0) * (weight[region] * gain)[..., None]
        output[region] = rgb[region] - correction
    decision = _saved_view(output, bit_depth)
    # 채널 뷰 참조 해제로 밝은 보정 RGB 표면 버퍼 수명 종료.
    del correction, surface, red, green, blue, background
    del skin, structure, protected, excluded
    dark_weight, dark_surface, dark_report = _dark_points(decision, _median_surface(decision), weight)
    for start in range(0, rgb.shape[0], 64):
        region = np.s_[start:start + 64]
        correction = np.maximum(dark_surface[region] - output[region], 0) * (dark_weight[region] * gain)[..., None]
        output[region] += correction
    report.update(dark_report)
    report.update({"stage": "sparkle_final", "strength": float(strength),
                   "bit_depth": int(bit_depth), "detection_measured": True,
                   "changed_pixels": int(np.count_nonzero(np.any(output != rgb, axis=2))),
                   "correction_rms_dn": float(np.sqrt(np.mean((rgb - output) ** 2)))})
    return output, report
