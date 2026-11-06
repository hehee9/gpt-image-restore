# cython: language_level=3, boundscheck=False, cdivision=True
"""@description 채색 면 적합과 저대비 능선 판정의 선택적 가속."""
import math
import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from libc.math cimport fabsf


def _rgb_max(values):
    """@description RGB 세 채널의 픽셀별 최댓값."""
    return np.maximum(np.maximum(values[..., 0], values[..., 1]), values[..., 2])


def _profile_widths(const float[:, :] residual, const Py_ssize_t[:] peaks,
                    const float[:] amplitudes, const float[:] depths):
    """@description float32 연산 순서를 유지하는 반치폭 탐색."""
    cdef Py_ssize_t i, left_index, right_index, used = 0
    cdef float threshold, difference, denominator, delta, a, b
    widths = np.empty(residual.shape[0], np.float32)
    cdef float[:] output = widths
    for i in range(residual.shape[0]):
        if amplitudes[i] <= 0:
            continue
        threshold = amplitudes[i] * <float>0.5
        left_index = peaks[i]
        right_index = peaks[i]
        while left_index > 0 and residual[i, left_index - 1] > threshold:
            left_index -= 1
        while right_index < depths.shape[0] - 1 and residual[i, right_index + 1] > threshold:
            right_index += 1
        if left_index == 0 or right_index == depths.shape[0] - 1:
            continue
        difference = residual[i, left_index] - threshold
        delta = <float>0.5 * difference
        denominator = residual[i, left_index] - residual[i, left_index - 1]
        if denominator < <float>0.001:
            denominator = <float>0.001
        delta = delta / denominator
        a = depths[left_index] - delta
        difference = residual[i, right_index] - threshold
        delta = <float>0.5 * difference
        denominator = residual[i, right_index] - residual[i, right_index + 1]
        if denominator < <float>0.001:
            denominator = <float>0.001
        delta = delta / denominator
        b = depths[right_index] + delta
        output[used] = b - a
        used += 1
    return widths[:used]


def _ridge_pixels(candidate, const float[:, :] normal_x, const float[:, :] normal_y,
                  const float[:, :] contrast, const float[:, :] noise,
                  const float[:] tangent, double cutoff):
    """@description 중간 배열 없이 방향별 능선 후보 선별."""
    cdef const unsigned char[:, :] candidates = candidate.view(np.uint8)
    cdef Py_ssize_t y, x
    cdef float first, second, alignment, threshold
    result = np.zeros(candidate.shape, bool)
    cdef unsigned char[:, :] output = result.view(np.uint8)
    for y in range(normal_x.shape[0]):
        for x in range(normal_x.shape[1]):
            if not candidates[y, x]:
                continue
            first = -normal_x[y, x] * tangent[1]
            second = normal_y[y, x] * tangent[0]
            alignment = first + second
            if fabsf(alignment) <= cutoff:
                continue
            threshold = <float>2.5 * noise[y, x]
            if threshold < <float>1.5:
                threshold = <float>1.5
            output[y, x] = fabsf(contrast[y, x]) >= threshold
    return result


def _region_is_consistent(const float[:, :, :] data, const float[:, :, :] model,
                          interior, Py_ssize_t count):
    """@description 잔차 개수 우선 판정과 경계 표본의 원래 백분위 계산."""
    cdef const unsigned char[:, :] selected = interior.view(np.uint8)
    cdef Py_ssize_t y, x, channel, below = 0, offset = 0
    cdef float value, maximum
    cdef double position = (count - 1) * 0.7
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if not selected[y, x]:
                continue
            maximum = fabsf(data[y, x, 0] - model[y, x, 0])
            for channel in range(1, 3):
                value = fabsf(data[y, x, channel] - model[y, x, channel])
                if value > maximum:
                    maximum = value
            below += maximum <= 6
    if below > math.ceil(position):
        return True
    if below <= math.floor(position):
        return False
    residual = np.empty(count, np.float32)
    cdef float[:] values = residual
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if not selected[y, x]:
                continue
            maximum = fabsf(data[y, x, 0] - model[y, x, 0])
            for channel in range(1, 3):
                value = fabsf(data[y, x, channel] - model[y, x, channel])
                if value > maximum:
                    maximum = value
            values[offset] = maximum
            offset += 1
    return np.percentile(residual, 70) <= 6


def _faint_profile(plane, points, tangent):
    """@description 긴 능선의 대비, 반치폭과 단면 일관성 측정."""
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
    return amplitudes, _profile_widths(residual, peaks, amplitudes, depths), sign


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
    cdef Py_ssize_t channel, top, index, distance
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
        ridge_pixels = _ridge_pixels(
            candidate, normal_x, normal_y, contrast, noise_map, tangent,
            np.cos(np.deg2rad(25)),
        )
        connector = np.zeros((7, 7), np.uint8)
        start = tuple(np.rint(3 - tangent * 2).astype(int))
        stop = tuple(np.rint(3 + tangent * 2).astype(int))
        cv2.line(connector, start, stop, 1, 1)
        # 원래 관측 폭을 유지하며 선 방향의 짧은 틈 연결.
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
    from .celfill import _FillModels, _polynomial_basis
    cdef Py_ssize_t k, top, bottom, step, interior_count, ncoef, degree, trial
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



