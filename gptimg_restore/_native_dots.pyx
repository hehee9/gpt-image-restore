# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""작은 점 후보의 순차 검증 및 누적."""

import numpy as np
from libc.math cimport INFINITY


ctypedef fused _mad_float:
    float
    double


# NumPy 버전별 스칼라 뺄셈의 자료형 유지.
cdef bint _scalar_float32 = (np.float32(1) - 1.0).dtype == np.float32


def _component_candidates(const int[:, :] labels, const int[:, :] stats,
                          const float[:, :] residual):
    """컴포넌트 순서 및 첫 최대값을 보존하는 후보 추출."""
    cdef Py_ssize_t component_id, x, y, x0, y0, bw, bh, area, peak_y, peak_x
    cdef float best, value
    cdef list candidates = []
    for component_id in range(1, stats.shape[0]):
        x0, y0 = stats[component_id, 0], stats[component_id, 1]
        bw, bh, area = stats[component_id, 2], stats[component_id, 3], stats[component_id, 4]
        if area < 1 or area > 80 or bw > 15 or bh > 15:
            continue
        best = -INFINITY
        peak_y, peak_x = y0, x0
        for y in range(y0, y0 + bh):
            for x in range(x0, x0 + bw):
                if labels[y, x] == component_id:
                    value = residual[y, x]
                    if value > best:
                        best, peak_y, peak_x = value, y, x
        candidates.append((peak_y, peak_x, float(best)))
    return candidates


def _validate_luma(batch, const float[:, :] luma, const float[:] backgrounds,
                   const _mad_float[:] mads, const float[:, :] edge_distance,
                   thresholds, background_texture):
    """NumPy 자료형 승격과 8방향 연결을 보존하는 휘도 후보 검증."""
    cdef double residual_min = thresholds["residual_min_dn"]
    cdef double contrast_min = thresholds["contrast_min_dn"]
    cdef double mad_max = 0.0
    cdef bint limit_mad = thresholds["ring_mad_max_dn"] is not None
    cdef double edge_min = thresholds["edge_distance_min_px"]
    cdef int area_min = thresholds["core_area_min_px"]
    cdef int area_max = thresholds["core_area_max_px"]
    cdef int bbox_max = thresholds["core_bbox_max_px"]
    cdef double texture_max = thresholds["texture_max_dn"]
    cdef Py_ssize_t i, peak_y, peak_x, y, x, yy, xx, j, k
    cdef int head, tail, pos, ymin, ymax, xmin, xmax
    cdef int queue[361]
    cdef unsigned char core[361]
    cdef float background, half_height, delta
    cdef double contrast, mad, peak_residual
    cdef list accepted = []
    if limit_mad:
        mad_max = thresholds["ring_mad_max_dn"]
    for i in range(len(batch)):
        peak_y, peak_x, peak_residual = batch[i]
        background = backgrounds[i]
        mad = mads[i]
        if _scalar_float32:
            contrast = <float>(luma[peak_y, peak_x] - background)
        else:
            contrast = <double>luma[peak_y, peak_x] - <double>background
        if peak_residual < residual_min:
            continue
        if contrast < max(contrast_min, 6.0 * mad):
            continue
        if limit_mad and mad > mad_max:
            continue
        if edge_distance[peak_y, peak_x] < edge_min:
            continue
        half_height = max(4.0, 0.5 * <double>contrast)
        for y in range(19):
            for x in range(19):
                delta = luma[peak_y - 9 + y, peak_x - 9 + x] - background
                core[y * 19 + x] = delta >= half_height
        if core[180] == 0:
            continue
        queue[0], core[180] = 180, 2
        head, tail = 0, 1
        ymin, ymax, xmin, xmax = 9, 9, 9, 9
        while head < tail:
            pos = queue[head]
            head += 1
            y, x = pos // 19, pos % 19
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y
            if x < xmin:
                xmin = x
            if x > xmax:
                xmax = x
            for yy in range(max(0, y - 1), min(19, y + 2)):
                for xx in range(max(0, x - 1), min(19, x + 2)):
                    j = yy * 19 + xx
                    if core[j] == 1:
                        core[j] = 2
                        queue[tail] = j
                        tail += 1
        if tail < area_min or tail > area_max:
            continue
        if xmax - xmin + 1 > bbox_max or ymax - ymin + 1 > bbox_max:
            continue
        local_y = np.empty(tail, dtype=np.intp)
        local_x = np.empty(tail, dtype=np.intp)
        k = 0
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                if core[y * 19 + x] == 2:
                    local_y[k], local_x[k] = y, x
                    k += 1
        if background_texture(luma.base, peak_y, peak_x, local_y, local_x,
                              float(background)) > texture_max:
            continue
        accepted.append((peak_y, peak_x, peak_y - 9 + local_y,
                         peak_x - 9 + local_x, float(contrast)))
    return accepted


def _accumulate_rgb(batch, const float[:, :, :] rgb, const float[:, :] clean_rgbs,
                    unsigned char[:, :] core_mask, target_sum, target_count,
                    contrasts, channel_min, bint collect_targets):
    """후보 및 픽셀 순서를 보존하는 RGB 필터와 대상 누적."""
    cdef Py_ssize_t i, j, c, peak_y, peak_x, y, x
    cdef float increment
    cdef double minimum = 0.0
    cdef bint limit_channel = channel_min is not None
    cdef bint reject
    cdef float[:, :, :] sums
    cdef unsigned short[:, :] counts
    cdef const Py_ssize_t[:] core_ys, core_xs
    if limit_channel:
        minimum = channel_min
    for i in range(len(batch)):
        peak_y, peak_x, ys, xs, contrast = batch[i]
        if limit_channel:
            reject = False
            for c in range(3):
                increment = rgb[peak_y, peak_x, c] - clean_rgbs[i, c]
                if increment < minimum:
                    reject = True
            if reject:
                continue
        core_ys, core_xs = ys, xs
        if collect_targets:
            if target_sum is None:
                target_sum = np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype=np.float32)
                target_count = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint16)
            sums, counts = target_sum, target_count
        for j in range(core_ys.shape[0]):
            y, x = core_ys[j], core_xs[j]
            core_mask[y, x] = 1
            if collect_targets:
                for c in range(3):
                    sums[y, x, c] += clean_rgbs[i, c]
                counts[y, x] += 1
        contrasts.append(contrast)
    return target_sum, target_count
