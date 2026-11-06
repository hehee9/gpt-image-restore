# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""문자 후보, 연결 구역과 국소 배경의 선택적 C 반복."""

import cv2
import numpy as np

ctypedef fused color_t:
    float
    double


def glyph_maxima(const int[:, :] labels, const unsigned char[:] valid,
                 const float[:, :] distance, const color_t[:, :, :] rgb,
                 Py_ssize_t count):
    """유효 연결 구역의 획 거리와 채널 최댓값 집계."""
    stroke_array = np.full(count, -np.inf, np.float32)
    color_array = np.full((count, 3), -np.inf, np.float32 if color_t is float else np.float64)
    cdef float[:] strokes = stroke_array
    cdef color_t[:, :] colors = color_array
    cdef Py_ssize_t y, x, channel
    cdef int label
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            label = labels[y, x]
            if not valid[label]:
                continue
            if distance[y, x] >= strokes[label]:
                strokes[label] = distance[y, x]
            for channel in range(3):
                if rgb[y, x, channel] >= colors[label, channel]:
                    colors[label, channel] = rgb[y, x, channel]
    return stroke_array, color_array


def aligned_members(const int[:, :] stats, const Py_ssize_t[:] ids,
                    const Py_ssize_t[:] nearby, const color_t[:, :] colors,
                    const float[:] strokes, Py_ssize_t index,
                    float lower_stroke, float upper_stroke):
    """같은 기준선, 잉크색과 획 두께의 인접 문자 선택."""
    result = np.empty(nearby.shape[0], dtype=np.intp)
    cdef Py_ssize_t[:] output = result
    cdef Py_ssize_t k, other, used = 0
    cdef int label = ids[index], candidate, h = stats[label, 3]
    cdef double baseline = stats[label, 1] + h
    cdef color_t difference, maximum
    for k in range(nearby.shape[0]):
        other = nearby[k]
        candidate = ids[other]
        if (stats[candidate, 3] < 0.6 * h or stats[candidate, 3] > 1.5 * h
                or abs(stats[candidate, 1] + stats[candidate, 3] - baseline) > 0.36 * h
                or strokes[other] < lower_stroke or strokes[other] > upper_stroke):
            continue
        maximum = abs(colors[other, 0] - colors[index, 0])
        difference = abs(colors[other, 1] - colors[index, 1])
        if difference > maximum:
            maximum = difference
        difference = abs(colors[other, 2] - colors[index, 2])
        if difference > maximum:
            maximum = difference
        if maximum < 22:
            output[used] = candidate
            used += 1
    return result[:used]


def has_diverse_glyph_shapes(const color_t[:, :, :] shapes):
    """모양 거리와 유사 군집 크기를 같은 쌍 순서로 계산."""
    return _diverse_shapes(shapes, np.asarray(shapes))


cdef _diverse_shapes(const color_t[:, :, :] shapes, array):
    cdef Py_ssize_t count = shapes.shape[0]
    cdef Py_ssize_t size = shapes.shape[1] * shapes.shape[2]
    parent_array = np.arange(count, dtype=np.intp)
    size_array = np.ones(count, dtype=np.intp)
    cdef Py_ssize_t[:] parents = parent_array
    cdef Py_ssize_t[:] sizes = size_array
    cdef Py_ssize_t a, b, y, x, left, right, squared = count
    cdef double minimum = min(3.5, count)
    cdef double threshold = float(np.asarray(0.08, dtype=array.dtype))
    cdef double error_bound = np.finfo(array.dtype).eps * (2 * size + 1)
    cdef double difference, total
    cdef bint similar
    for a in range(count):
        for b in range(a + 1, count):
            total = 0
            for y in range(shapes.shape[1]):
                for x in range(shapes.shape[2]):
                    total += abs(<double>shapes[a, y, x] - <double>shapes[b, y, x])
            difference = total / size
            similar = difference < threshold
            if abs(difference - threshold) <= error_bound:
                similar = np.mean(np.abs(array[a] - array[b])) < 0.08
            if not similar:
                continue
            left, right = a, b
            while parents[left] != left:
                left = parents[left]
            while parents[right] != right:
                right = parents[right]
            if left == right:
                continue
            if sizes[left] < sizes[right]:
                left, right = right, left
            squared += 2 * sizes[left] * sizes[right]
            sizes[left] += sizes[right]
            parents[right] = left
            if <double>(count * count) / squared < minimum:
                return False
    return True


cdef bint _repeated_bands(const int[:, :] labels, const int[:, :] stats):
    cdef Py_ssize_t i, y, x, valid_count = 0, selected, start = -1
    cdef Py_ssize_t previous_start = -1, previous_stop = -1, pairs = 0
    glyph_array = np.zeros(stats.shape[0], np.uint8)
    cdef unsigned char[:] glyph = glyph_array
    for i in range(1, stats.shape[0]):
        if (4 <= stats[i, 4] <= 1000 and 4 <= stats[i, 3] <= 36 and stats[i, 2] <= 80):
            glyph[i] = 1
            valid_count += 1
    if valid_count < 15:
        return False
    for y in range(labels.shape[0] + 1):
        selected = 0
        if y < labels.shape[0]:
            for x in range(labels.shape[1]):
                selected += glyph[labels[y, x]]
        if <double>selected / labels.shape[1] > 0.045:
            if start < 0:
                start = y
        elif start >= 0:
            if y - start >= 4:
                if (previous_start >= 0 and 1 <= start - previous_stop <= 20
                        and 5 <= previous_stop - previous_start <= 36 and 5 <= y - start <= 36):
                    pairs += 1
                    if pairs >= 2:
                        return True
                previous_start, previous_stop = start, y
            start = -1
    return False


def text_region(rgb, difference):
    """타일별 연결 구역과 반복 문자 띠를 C 반복으로 선별."""
    from .glyphring import _maximum_channel
    ink = (difference > 20).astype(np.uint8)
    region = np.zeros(ink.shape, np.uint8)
    cdef Py_ssize_t top, left
    for top in range(0, rgb.shape[0], 64):
        for left in range(0, rgb.shape[1], 96):
            patch = ink[top:top + 128, left:left + 192]
            if min(patch.shape) < 64:
                continue
            _, labels, stats, _ = cv2.connectedComponentsWithStats(patch, 8)
            if not _repeated_bands(labels, stats):
                continue
            sample = difference[top:top + 128, left:left + 192] < 8
            if np.mean(sample) < 0.4:
                continue
            pixels = rgb[top:top + 128, left:left + 192]
            yy, xx = np.mgrid[:patch.shape[0], :patch.shape[1]].astype(np.float32)
            coordinates = np.stack([np.ones_like(xx), xx / 192, yy / 128], axis=-1)
            plane = np.linalg.lstsq(coordinates[sample], pixels[sample], rcond=None)[0]
            residual = pixels[sample] - coordinates[sample] @ plane
            if (np.max(np.sqrt(np.mean(residual ** 2, axis=0))) > 6
                    or np.percentile(_maximum_channel(np.abs(residual)), 90) > 12):
                continue
            region[top:top + 128, left:left + 192] = 1
    return region


def background_model(rgb, samples):
    """동일한 수평 합성곱을 공유하는 국소 배경 적합."""
    offsets = np.arange(-120, 121, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / 40) ** 2)
    kernel /= kernel.sum()
    first = kernel * offsets
    second = first * offsets
    identity = np.ones(1, np.float32)
    horizontal = [cv2.sepFilter2D(samples, cv2.CV_32F, k, identity, borderType=cv2.BORDER_CONSTANT)
                  for k in (kernel, first, second)]
    moments = [cv2.sepFilter2D(horizontal[x], cv2.CV_32F, identity, k, borderType=cv2.BORDER_CONSTANT)
               for x, k in ((0, kernel), (1, kernel), (0, first),
                            (2, kernel), (0, second), (1, first))]
    del horizontal
    total = rgb * samples[..., None]
    horizontal = [cv2.sepFilter2D(total, cv2.CV_32F, k, identity, borderType=cv2.BORDER_CONSTANT)
                  for k in (kernel, first)]
    colors = [cv2.sepFilter2D(horizontal[x], cv2.CV_32F, identity, k, borderType=cv2.BORDER_CONSTANT)
              for x, k in ((0, kernel), (1, kernel), (0, first))]
    del horizontal, total
    return _fit_background(moments, colors)


cdef _fit_background(moments, colors):
    cdef float[:, :] mass = moments[0]
    cdef float[:, :] mx_values = moments[1]
    cdef float[:, :] my_values = moments[2]
    cdef float[:, :] vx_values = moments[3]
    cdef float[:, :] vy_values = moments[4]
    cdef float[:, :] xy_values = moments[5]
    cdef float[:, :, :] mean = colors[0]
    cdef float[:, :, :] cx_values = colors[1]
    cdef float[:, :, :] cy_values = colors[2]
    determinant = np.empty_like(moments[0])
    cdef float[:, :] det_values = determinant
    cdef Py_ssize_t y, x, channel
    cdef float support, mx, my, vx, vy, xy, det, denominator, value, cx, cy, sx, sy
    for y in range(mass.shape[0]):
        for x in range(mass.shape[1]):
            support = max(mass[y, x], <float>1e-6)
            mx = mx_values[y, x] / support
            my = my_values[y, x] / support
            vx = <float>(vx_values[y, x] / support) - <float>(mx * mx)
            vy = <float>(vy_values[y, x] / support) - <float>(my * my)
            xy = <float>(xy_values[y, x] / support) - <float>(mx * my)
            det = <float>(vx * vy) - <float>(xy * xy)
            det_values[y, x] = det
            denominator = max(det, <float>1e-4)
            for channel in range(3):
                value = mean[y, x, channel] / support
                cx = <float>(cx_values[y, x, channel] / support) - <float>(mx * value)
                cy = <float>(cy_values[y, x, channel] / support) - <float>(my * value)
                sx = <float>(<float>(cx * vy) - <float>(cy * xy)) / denominator
                sy = <float>(<float>(cy * vx) - <float>(cx * xy)) / denominator
                value -= <float>(sx * mx)
                value -= <float>(sy * my)
                mean[y, x, channel] = value
    return colors[0], moments[0], determinant
