# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""@description 윤곽 띠의 최근접 표본 탐색과 법선 단면 보정."""

from libc.math cimport nearbyint, sqrt


ctypedef fused index_t:
    int
    Py_ssize_t


cdef inline Py_ssize_t _clip_index(Py_ssize_t value, Py_ssize_t end) noexcept:
    return 0 if value < 0 else end if value > end else value


def refine_index(int[:] index, const Py_ssize_t[:] xx, const Py_ssize_t[:] yy,
                 Py_ssize_t x0, Py_ssize_t y0,
                 const float[:, :] points, const float[:, :] normal):
    """@description 접선 이동 뒤 -1, +1 순서로 최근접 표본 재선택."""
    cdef Py_ssize_t pixel, selected, candidate, shift, offset
    cdef Py_ssize_t end = points.shape[0] - 1
    cdef double x, y, dx, dy, nearest, squared
    for pixel in range(index.shape[0]):
        x = <double>(xx[pixel] + x0)
        y = <double>(yy[pixel] + y0)
        selected = index[pixel]
        dx = x - points[selected, 0]
        dy = y - points[selected, 1]
        shift = <Py_ssize_t>nearbyint(dx * normal[selected, 1] - dy * normal[selected, 0])
        selected = _clip_index(selected + shift, end)
        dx = x - points[selected, 0]
        dy = y - points[selected, 1]
        nearest = dx * dx + dy * dy
        for offset in (-1, 1):
            candidate = _clip_index(selected + offset, end)
            dx = x - points[candidate, 0]
            dy = y - points[candidate, 1]
            squared = dx * dx + dy * dy
            if squared < nearest:
                selected = candidate
                nearest = squared
        index[pixel] = <int>selected


def apply_profile(const float[:, :, :] rgb, const unsigned char[:, :] protected,
                  float[:, :] confidence, float[:, :, :] correction,
                  const float[:, :] points, const float[:, :] normal,
                  const double[:, :] weight, const float[:, :, :] model,
                  const index_t[:] index, const Py_ssize_t[:] xx, const Py_ssize_t[:] yy,
                  Py_ssize_t x0, Py_ssize_t y0):
    """@description float64 보간 순서와 float32 저장을 보존한 픽셀 보정."""
    cdef Py_ssize_t pixel, x, y, selected, lower, upper, channel
    cdef float sample_depth
    cdef double dx, dy, distance, depth, fraction, taper, pixel_weight, target, delta
    cdef bint accepted = False
    for pixel in range(xx.shape[0]):
        x = xx[pixel] + x0
        y = yy[pixel] + y0
        selected = index[pixel]
        dx = <double>x - points[selected, 0]
        dy = <double>y - points[selected, 1]
        distance = sqrt(dx * dx + dy * dy)
        if distance >= 14 or protected[y, x] != 0:
            continue
        depth = dx * normal[selected, 0] + dy * normal[selected, 1]
        depth = depth + 15
        sample_depth = <float>(0 if depth < 0 else 30 if depth > 30 else depth)
        lower = <Py_ssize_t>sample_depth
        upper = lower + 1 if lower < 30 else 30
        fraction = <double>sample_depth - lower
        pixel_weight = weight[selected, lower] * (1 - fraction) + weight[selected, upper] * fraction
        taper = (14 - distance) / 4
        pixel_weight *= 0 if taper < 0 else 1 if taper > 1 else taper
        if pixel_weight <= confidence[y, x]:
            continue
        accepted = True
        confidence[y, x] = <float>pixel_weight
        for channel in range(3):
            target = model[selected, lower, channel] * (1 - fraction) + model[selected, upper, channel] * fraction
            delta = rgb[y, x, channel] - target
            delta = -10 if delta < -10 else 10 if delta > 10 else delta
            correction[y, x, channel] = <float>(delta * pixel_weight)
    return accepted
