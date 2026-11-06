# cython: boundscheck=False, wraparound=False, initializedcheck=False
"""@description 후보 순서를 유지하는 밝은 점과 어두운 점 검사 커널."""
import numpy as np
cimport cython
from libc.math cimport fabsf, INFINITY
from libc.stdint cimport int32_t, int64_t

ctypedef fused pixel_t:
    float
    double


def component_colors(const pixel_t[:, :, :] rgb, const int32_t[:, :] labels,
                     const int32_t[:, :] stats, const int64_t[:] ids,
                     bint inverse=False, residual=None):
    """성분 경계 내 색상 극값과 암부 대비 계산."""
    cdef Py_ssize_t i, y, x, c, label, left, top, width, height
    cdef float[:, :] colors = np.empty((ids.shape[0], 3), np.float32)
    cdef float[:] contrast = np.empty(ids.shape[0], np.float32)
    cdef const pixel_t[:, :] residual_view
    cdef pixel_t values[3]
    cdef pixel_t value, maximum
    cdef bint dark = residual is not None
    if dark:
        residual_view = residual
    for i in range(ids.shape[0]):
        label = ids[i]
        left, top = stats[label, 0], stats[label, 1]
        width, height = stats[label, 2], stats[label, 3]
        for c in range(3):
            values[c] = INFINITY if dark else -INFINITY
        maximum = -INFINITY
        for y in range(top, top + height):
            for x in range(left, left + width):
                if labels[y, x] != label:
                    continue
                for c in range(3):
                    value = rgb[y, x, c]
                    if inverse:
                        value = <pixel_t>(255 - value)
                    if (dark and value < values[c]) or (not dark and value > values[c]):
                        values[c] = value
                if dark and residual_view[y, x] > maximum:
                    maximum = residual_view[y, x]
        for c in range(3):
            colors[i, c] = values[c]
        contrast[i] = maximum
    if dark:
        return np.asarray(colors), np.asarray(contrast)
    return np.asarray(colors)


def possible_eye(contours, Py_ssize_t height, Py_ssize_t width,
                 Py_ssize_t cy, Py_ssize_t cx):
    """외곽선 경계와 후보 중심으로 눈 형상 사전 판정."""
    cdef const int32_t[:, :, :] contour
    cdef Py_ssize_t i, left, right, top, bottom, bw, bh, short, long
    for raw in contours:
        contour = raw
        left = right = contour[0, 0, 0]
        top = bottom = contour[0, 0, 1]
        for i in range(1, contour.shape[0]):
            left = min(left, contour[i, 0, 0])
            right = max(right, contour[i, 0, 0])
            top = min(top, contour[i, 0, 1])
            bottom = max(bottom, contour[i, 0, 1])
        bw, bh = right - left + 1, bottom - top + 1
        short, long = min(bw, bh), max(bw, bh)
        if (short >= 12 and long <= 128 and long <= 1.7 * short
                and left > 0 and top > 0 and left + bw < width and top + bh < height
                and left < cx + 13 and left + bw > cx - 12
                and top < cy + 13 and top + bh > cy - 12):
            return True
    return False


def aligned_rows(const float[:] by, const float[:] bh, const float[:, :] paint):
    """float32 반올림을 유지하는 문자 높이와 색상 관계 판정."""
    cdef Py_ssize_t i, j, c, n = bh.shape[0]
    cdef unsigned char[:, :] aligned = np.empty((n, n), np.uint8)
    cdef float cyi, cyj, maximum, difference
    cdef bint match
    for i in range(n):
        cyi = <float>(by[i] + <float>(bh[i] * <float>0.5))
        for j in range(n):
            cyj = <float>(by[j] + <float>(bh[j] * <float>0.5))
            maximum = max(bh[i], bh[j])
            match = (fabsf(<float>(cyi - cyj)) <= <float>(<float>0.35 * maximum)
                     and bh[i] >= <float>(<float>0.8 * bh[j])
                     and bh[i] <= <float>(<float>1.25 * bh[j]))
            if match:
                for c in range(3):
                    difference = fabsf(<float>(paint[i, c] - paint[j, c]))
                    if difference > 24:
                        match = False
                        break
            aligned[i, j] = match
    return np.asarray(aligned).view(np.bool_)


def row_bounds(const unsigned char[:, :] aligned, const float[:] bx,
               const float[:] by, const float[:] bw, const float[:] bh,
               const float[:] distances, const float[:] anchor_height, int x, int y):
    """문자 행별 경계와 후보 포함 여부를 원래 순서로 계산."""
    cdef Py_ssize_t i, j, count, n = aligned.shape[0], m = aligned.shape[1]
    cdef float[:, :] bounds = np.empty((4, n), np.float32)
    cdef unsigned char[:] rows = np.empty(n, np.uint8)
    cdef float left, right, top, bottom, near, ah, edge
    for i in range(n):
        left = top = near = INFINITY
        right = bottom = -INFINITY
        count = 0
        for j in range(m):
            if aligned[i, j]:
                count += 1
                left = min(left, bx[j])
                top = min(top, by[j])
                edge = <float>(bx[j] + bw[j])
                right = max(right, edge)
                edge = <float>(by[j] + bh[j])
                bottom = max(bottom, edge)
                near = min(near, distances[j])
        bounds[0, i], bounds[1, i] = left, right
        bounds[2, i], bounds[3, i] = top, bottom
        ah = anchor_height[i]
        rows[i] = (count >= 3 and <float>(right - left) >= <float>(2 * ah)
                   and x >= <float>(left - ah) and x <= <float>(right + ah)
                   and y >= <float>(top - 2) and y <= <float>(bottom + 2) and near <= ah)
    return (*np.asarray(bounds), np.asarray(rows).view(np.bool_))


def select_distinct(const int64_t[:] ys, const int64_t[:] xs,
                    unsigned char[:, :] occupied):
    """앞선 암부 후보와 겹치는 중심을 순서대로 제외."""
    cdef Py_ssize_t i, y, x, yy, xx
    selected = []
    for i in range(ys.shape[0]):
        y, x = ys[i], xs[i]
        if not occupied[y, x]:
            for yy in range(y - 2, y + 3):
                for xx in range(x - 2, x + 3):
                    occupied[yy, xx] = 1
            selected.append(i)
    return selected


def select_bright(const int64_t[:] ys, const int64_t[:] xs,
                  const unsigned char[:, :, :] blocked, unsigned char[:, :] occupied,
                  Py_ssize_t start):
    """앞선 밝은 후보의 코어와 겹치는 중심을 순서대로 제외."""
    cdef Py_ssize_t i, y, x, yy, xx
    selected = []
    for i in range(blocked.shape[0]):
        y, x = ys[start + i], xs[start + i]
        if not occupied[y, x]:
            for yy in range(13):
                for xx in range(13):
                    occupied[y + yy - 6, x + xx - 6] |= blocked[i, yy, xx]
            selected.append(start + i)
    return selected


def scatter_dark(const int64_t[:] ys, const int64_t[:] xs,
                 const float[:, :, :] feather, const pixel_t[:, :] reference,
                 float[:, :] weight, float[:, :, :] target):
    """동일 가중치의 소유권을 유지하며 암부 표면과 가중치 기록."""
    cdef Py_ssize_t i, yy, xx, y, x, c
    cdef float value
    for i in range(ys.shape[0]):
        for yy in range(13):
            y = ys[i] + yy - 6
            for xx in range(13):
                x = xs[i] + xx - 6
                value = feather[i, yy, xx]
                if value > weight[y, x]:
                    weight[y, x] = value
                    for c in range(3):
                        target[y, x, c] = reference[i, c]
