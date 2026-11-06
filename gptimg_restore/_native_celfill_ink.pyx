# cython: language_level=3, boundscheck=False
"""@description 잉크 윤곽선 순서와 배열 연산을 보존하는 선택적 네이티브 렌더러."""

import cv2
import numpy as np
from scipy import ndimage
from scipy.special import ndtri, ndtr
from scipy.spatial import cKDTree

_LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def _rgb_max(values):
    """@description RGB 세 채널의 픽셀별 최댓값."""
    return np.maximum(np.maximum(values[..., 0], values[..., 1]), values[..., 2])


def _coordinates(const float[:, :] points, const float[:, :] normal,
                 const float[:] depths):
    """@description float32 곱셈과 덧셈의 개별 반올림을 보존하는 법선 좌표."""
    cdef Py_ssize_t i, j, channel
    cdef float product
    result = np.empty((points.shape[0], depths.shape[0], 2), np.float32)
    cdef float[:, :, :] target = result
    for i in range(points.shape[0]):
        for j in range(depths.shape[0]):
            for channel in range(2):
                product = normal[i, channel] * depths[j]
                target[i, j, channel] = points[i, channel] + product
    return result


def _projection(const float[:, :, :] profile, const float[:, :] direction):
    """@description 세 채널의 float32 곱셈과 순차 합산."""
    cdef Py_ssize_t i, j
    cdef float a, b, c, partial
    result = np.empty((profile.shape[0], profile.shape[1]), np.float32)
    cdef float[:, :] target = result
    for i in range(profile.shape[0]):
        for j in range(profile.shape[1]):
            a = profile[i, j, 0] * direction[i, 0]
            b = profile[i, j, 1] * direction[i, 1]
            c = profile[i, j, 2] * direction[i, 2]
            partial = a + b
            target[i, j] = partial + c
    return result


def _alpha(const float[:, :, :] profile, const float[:, :] low,
           const float[:, :] vector, const float[:] contrast):
    """@description 중간 float32 배열의 반올림을 유지하는 피복률."""
    cdef Py_ssize_t i, j
    cdef float a, b, c, partial, denominator
    result = np.empty((profile.shape[0], profile.shape[1]), np.float32)
    cdef float[:, :] target = result
    for i in range(profile.shape[0]):
        denominator = contrast[i] if contrast[i] > 1 else 1
        for j in range(profile.shape[1]):
            a = profile[i, j, 0] - low[i, 0]
            a = a * vector[i, 0]
            b = profile[i, j, 1] - low[i, 1]
            b = b * vector[i, 1]
            c = profile[i, j, 2] - low[i, 2]
            c = c * vector[i, 2]
            partial = a + b
            partial = partial + c
            target[i, j] = partial / denominator
    return result


def _accumulate(float[:, :, :] total, float[:, :] mass,
                const Py_ssize_t[:] yy, const Py_ssize_t[:] xx,
                const double[:, :] value, const double[:] confidence):
    """@description 윤곽선별 float64 합산 후 float32 저장 순서 유지."""
    cdef Py_ssize_t i, channel, y, x
    cdef double weighted
    for i in range(yy.shape[0]):
        y, x = yy[i], xx[i]
        for channel in range(3):
            weighted = value[i, channel] * confidence[i]
            total[y, x, channel] = <float>(<double>total[y, x, channel] + weighted)
        mass[y, x] = <float>(<double>mass[y, x] + confidence[i])


def render_ink_profiles(src, out, core, detail, color_aa, ink_support):
    """@description 잉크와 채색 모델 사이의 일관된 법선 피복 곡선 복원."""
    h, w = src.shape[:2]
    # 원래 표본으로 두 색과 법선 피복 곡선의 기하 추정.
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
        coordinates = _coordinates(points, normal, depths)
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
        projection = _projection(prof, direction)
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
        alpha = _alpha(prof, low, v, contrast)
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
        # 단면 적합을 보존하며 윤곽선 방향으로만 필터링.
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
        # 좁은 선의 반대쪽 경계가 함께 표본화될 때 측정한 반쪽 단면만 사용.
        confidence = np.maximum(0.05, 12 - distance)
        _accumulate(total, mass, yy, xx, value, confidence)
    profile_aa = mass > 0
    out[profile_aa] = total[profile_aa] / mass[profile_aa, None]
    return out, profile_aa

