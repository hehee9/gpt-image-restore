"""밀집 문자 사이의 단색/완만한 그라데이션 배경 복원."""

from __future__ import annotations

import cv2
import numpy as np

from ._parallel import _median_blur
from . import glyphring
from ._native import load_native


_native = load_native("_native_text")


def _text_region(rgb, difference):
    """반복된 문자 행과 평탄 배경을 함께 확인한 영역 선택."""
    if _native is not None:
        return _native.text_region(rgb, difference)
    ink = (difference > 20).astype(np.uint8)
    region = np.zeros(ink.shape, np.uint8)
    for top in range(0, rgb.shape[0], 64):
        for left in range(0, rgb.shape[1], 96):
            patch = ink[top:top + 128, left:left + 192]
            if min(patch.shape) < 64:
                continue
            _, labels, stats, _ = cv2.connectedComponentsWithStats(patch, 8)
            glyph = ((stats[:, cv2.CC_STAT_AREA] >= 4)
                     & (stats[:, cv2.CC_STAT_AREA] <= 1000)
                     & (stats[:, cv2.CC_STAT_HEIGHT] >= 4)
                     & (stats[:, cv2.CC_STAT_HEIGHT] <= 36)
                     & (stats[:, cv2.CC_STAT_WIDTH] <= 80))
            glyph[0] = False
            if np.count_nonzero(glyph) < 15:
                continue
            projection = np.mean(glyph[labels], axis=1) > 0.045
            bounds = np.flatnonzero(np.diff(np.r_[False, projection, False]))
            bands = [(a, b) for a, b in zip(bounds[::2], bounds[1::2]) if b - a >= 4]
            pairs = sum(1 <= c - b <= 20 and 5 <= b - a <= 36 and 5 <= d - c <= 36
                        for (a, b), (c, d) in zip(bands, bands[1:]))
            if pairs < 2:
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
                    or np.percentile(glyphring._maximum_channel(np.abs(residual)), 90) > 12):
                continue
            region[top:top + 128, left:left + 192] = 1
    return region


def _background_model(rgb, samples):
    """문자색을 제외한 40픽셀 국소 직선 적합으로 넓은 색 추세 추정."""
    if _native is not None:
        return _native.background_model(rgb, samples)
    offsets = np.arange(-120, 121, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / 40) ** 2)
    kernel /= kernel.sum()
    first = kernel * offsets
    second = first * offsets
    mass = cv2.sepFilter2D(samples, cv2.CV_32F, kernel, kernel, borderType=cv2.BORDER_CONSTANT)
    support = np.maximum(mass, 1e-6)
    mx = cv2.sepFilter2D(samples, cv2.CV_32F, first, kernel, borderType=cv2.BORDER_CONSTANT) / support
    my = cv2.sepFilter2D(samples, cv2.CV_32F, kernel, first, borderType=cv2.BORDER_CONSTANT) / support
    vx = (cv2.sepFilter2D(samples, cv2.CV_32F, second, kernel, borderType=cv2.BORDER_CONSTANT)
          / support - mx * mx)
    vy = (cv2.sepFilter2D(samples, cv2.CV_32F, kernel, second, borderType=cv2.BORDER_CONSTANT)
          / support - my * my)
    xy = (cv2.sepFilter2D(samples, cv2.CV_32F, first, first, borderType=cv2.BORDER_CONSTANT)
          / support - mx * my)
    determinant = vx * vy - xy * xy
    total = rgb * samples[..., None]
    mean = cv2.sepFilter2D(total, cv2.CV_32F, kernel, kernel,
                         borderType=cv2.BORDER_CONSTANT) / support[..., None]
    cx = (cv2.sepFilter2D(total, cv2.CV_32F, first, kernel,
                        borderType=cv2.BORDER_CONSTANT) / support[..., None]
          - mx[..., None] * mean)
    cy = (cv2.sepFilter2D(total, cv2.CV_32F, kernel, first,
                        borderType=cv2.BORDER_CONSTANT) / support[..., None]
          - my[..., None] * mean)
    del total
    denominator = np.maximum(determinant, 1e-4)[..., None]
    sx = (cx * vy[..., None] - cy * xy[..., None]) / denominator
    sy = (cy * vx[..., None] - cx * xy[..., None]) / denominator
    np.multiply(sx, mx[..., None], out=sx)
    mean -= sx
    np.multiply(sy, my[..., None], out=sy)
    mean -= sy
    return mean, mass, determinant


def _segmented_background(rgb, samples, median, region, *, backend=None):
    """배경색 경계 보호와 같은 연결 구역 안의 색 추세 적합."""
    edges = np.zeros(rgb.shape[:2], bool)
    for channel in range(3):
        gx = cv2.Sobel(median[..., channel], cv2.CV_32F, 1, 0, scale=0.125)
        gy = cv2.Sobel(median[..., channel], cv2.CV_32F, 0, 1, scale=0.125)
        edges |= gx * gx + gy * gy > 16
    _, edge_labels, edge_stats, _ = cv2.connectedComponentsWithStats(edges.astype(np.uint8), 8)
    extended = ((np.maximum(edge_stats[:, cv2.CC_STAT_WIDTH], edge_stats[:, cv2.CC_STAT_HEIGHT]) >= 32)
                & (edge_stats[:, cv2.CC_STAT_AREA] >= 16))
    extended[0] = False
    edges = extended[edge_labels]
    # 중앙값 배경의 경계 이동을 원래 색 경계로 보완하고, 그 사이의 얇은 띠도 보호.
    nearby = cv2.dilate(edges.astype(np.uint8), np.ones((33, 33), np.uint8)) > 0
    for channel in range(3):
        gx = cv2.Sobel(rgb[..., channel], cv2.CV_32F, 1, 0, scale=0.125)
        gy = cv2.Sobel(rgb[..., channel], cv2.CV_32F, 0, 1, scale=0.125)
        edges |= nearby & (gx * gx + gy * gy > 16)
    boundary = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(1 - boundary, 8)
    if backend is not None:
        from ._gpu_textgap import _segmented_background_gpu
        return _segmented_background_gpu(backend, rgb, samples, labels, stats, region)
    background = rgb.copy()
    mass = np.zeros(rgb.shape[:2], np.float32)
    determinant = np.zeros_like(mass)
    for label in range(1, len(stats)):
        left, top, width, height, area = stats[label]
        if area < 500 or min(width, height) < 16:
            continue
        box = np.s_[top:top + height, left:left + width]
        component = labels[box] == label
        if not np.any(region[box][component]):
            continue
        model, support, det = _background_model(rgb[box], samples[box] * component)
        background[box][component] = model[component]
        mass[box][component] = support[component]
        determinant[box][component] = det[component]
    return background, mass, determinant


def _coherent_details(rgb, glyphs):
    """문자색 투영과 별도로 가는 선과 닫힌 아이콘의 약한 경계 보호."""
    edges = np.zeros(rgb.shape[:2], bool)
    for channel in range(3):
        gx = cv2.Sobel(rgb[..., channel], cv2.CV_32F, 1, 0, scale=0.125)
        gy = cv2.Sobel(rgb[..., channel], cv2.CV_32F, 0, 1, scale=0.125)
        edges |= gx * gx + gy * gy > 16
    excluded = cv2.dilate(glyphs.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    candidate = (edges & ~excluded).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    protected = np.zeros_like(candidate)
    for label in range(1, len(stats)):
        left, top, width, height, area = stats[label]
        if area < 12 or max(width, height) < 10:
            continue
        if max(width, height) > 128 and min(width, height) > 32:
            continue
        box = np.s_[top:top + height, left:left + width]
        component = (labels[box] == label).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(protected[box], contours, -1, 1, cv2.FILLED)
    return cv2.dilate(protected, np.ones((3, 3), np.uint8)) > 0


def _restore_text_region(rgb, strength, handled=None, *, backend=None):
    """문자 획과 1픽셀 안티앨리어싱을 유지하며 줄 사이 배경 정리.

    반복된 문자 행과 평탄 배경이 함께 있는 영역만 보정한다. 문자색/배경색
    투영의 10% 이상인 픽셀과 그 1픽셀 이웃을 그대로 유지한다.
    배경색 경계와 독립적인 약한 선 구조도 보호한다. 나머지 배경은 같은
    색 구역의 넓은 색 추세로 복원하며, 겹친 타일의 중앙만 출력한다.
    """
    median = _median_blur(np.clip(np.rint(rgb), 0, 255).astype(np.uint8), 31)
    difference = glyphring._maximum_channel(np.abs(rgb - median))
    region = _text_region(rgb, difference)
    if handled is not None:
        handled[:] = region
    if not np.any(region):
        return rgb.copy(), {"eligible_fraction": 0.0}
    core = cv2.erode((difference > 40).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    core &= region > 0
    if not np.any(core):
        return rgb.copy(), {"eligible_fraction": 0.0}
    foreground = np.median(rgb[core], axis=0)
    samples = cv2.dilate((difference > 10).astype(np.uint8), np.ones((7, 7), np.uint8)) == 0
    del difference, core
    output = rgb.copy()
    selected = 0
    absolute_change = np.zeros(3, np.float64)
    rows = np.flatnonzero(np.any(region, axis=1))
    columns = np.flatnonzero(np.any(region, axis=0))
    # 모델 반경 120픽셀보다 넓은 여백으로 타일 경계의 샘플과 보호 띠 공유.
    for top in range(max(0, rows[0] - 32), min(rgb.shape[0], rows[-1] + 33), 512):
        bottom = min(top + 512, rgb.shape[0])
        for left in range(max(0, columns[0] - 32), min(rgb.shape[1], columns[-1] + 33), 512):
            right = min(left + 512, rgb.shape[1])
            y0, x0 = max(0, top - 128), max(0, left - 128)
            y1, x1 = min(rgb.shape[0], bottom + 128), min(rgb.shape[1], right + 128)
            box = np.s_[y0:y1, x0:x1]
            local_region = region[box].astype(np.float32)
            if not np.any(local_region):
                continue
            pixels = rgb[box]
            background, mass, determinant = _segmented_background(
                pixels, samples[box].astype(np.float32), median[box], local_region, backend=backend)
            residual = pixels - background
            direction = foreground - background
            norm = np.sum(direction ** 2, axis=2)
            alpha = np.sum(residual * direction, axis=2) / np.maximum(norm, 1)
            # 10% 전이의 양쪽 보간 표본까지 유지해 안티앨리어싱 폭 변화 방지.
            ink = (alpha >= 0.1) | (glyphring._maximum_channel(np.abs(pixels - median[box])) > 20)
            protected = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            protected |= _coherent_details(pixels, protected)
            orthogonal = residual - alpha[..., None] * direction
            weight = cv2.GaussianBlur(local_region, (0, 0), 8)
            weight[protected | (mass < 0.02) | (determinant < 1) | (norm < 1600)
                   | (glyphring._maximum_channel(np.abs(orthogonal)) > 15)
                   | (glyphring._maximum_channel(np.abs(residual)) > 25)] = 0
            center = np.s_[top - y0:bottom - y0, left - x0:right - x0]
            correction = residual[center] * (weight[center] * min(strength, 1.0))[..., None]
            output[top:bottom, left:right] -= correction
            selected += np.count_nonzero(weight[center] > 0.1)
            absolute_change += np.sum(np.abs(correction), axis=(0, 1), dtype=np.float64)
    pixels = rgb.shape[0] * rgb.shape[1]
    return output, {
        "eligible_fraction": float(selected / pixels),
        "background_sigma": 40,
        "mean_abs_change": (absolute_change / pixels).tolist(),
    }


def _document_plane(rgb, samples):
    """색 표본의 4픽셀 간격 전역 평면 적합."""
    height, width = samples.shape
    yy, xx = np.mgrid[:height:4, :width:4].astype(np.float32)
    coordinates = np.stack([np.ones_like(xx), xx / width, yy / height], axis=2)
    selected = samples[::4, ::4]
    coefficients = np.linalg.lstsq(coordinates[selected], rgb[::4, ::4][selected], rcond=None)[0]
    x = np.arange(width, dtype=np.float32)[None, :, None] / width
    y = np.arange(height, dtype=np.float32)[:, None, None] / height
    return coefficients[0] + x * coefficients[1] + y * coefficients[2]


def _faint_panel_edge(values, excluded, left, top, width, height, sign):
    """기존 획과 분리된 면 경계의 부호 일관성과 가로 대비 확인."""
    edges = ((values, excluded, left, left + width, top, min(top + 2, top + height - 1), top - 3),
             (values, excluded, left, left + width, top + height - 1, max(top, top + height - 3), top + height + 2),
             (values.T, excluded.T, top, top + height, left, min(left + 2, left + width - 1), left - 3),
             (values.T, excluded.T, top, top + height, left + width - 1, max(left, left + width - 3), left + width + 2))
    for pixels, blocked, start, stop, boundary, inner, outer in edges:
        if not 0 <= outer < pixels.shape[0]:
            continue
        span = np.s_[start:stop]
        valid = ~(blocked[boundary, span] | blocked[inner, span] | blocked[outer, span])
        if np.count_nonzero(valid) < max(8, 0.2 * (stop - start)):
            continue
        inside = np.maximum(sign * pixels[boundary, span], sign * pixels[inner, span])
        contrast = inside - sign * pixels[outer, span]
        if np.mean(contrast[valid] > 0.75) >= 0.7:
            return True
    return False


def _faint_document_details(residual, excluded):
    """약한 직선, 닫힌 면과 같은 줄에 반복된 획의 형태 보호."""
    protected = np.zeros(excluded.shape, np.uint8)
    image_height, image_width = excluded.shape
    separated = cv2.dilate(excluded.astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
    for channel in range(3):
        values = residual[..., channel]
        horizontal = values[::4, 1::4] - values[::4, :-1:4]
        vertical = values[1::4, ::4] - values[:-1:4, ::4]
        horizontal = horizontal[~(excluded[::4, 1::4] | excluded[::4, :-1:4])]
        vertical = vertical[~(excluded[1::4, ::4] | excluded[:-1:4, ::4])]
        deviations = [np.median(np.abs(delta - np.median(delta)))
                      for delta in (horizontal, vertical) if delta.size]
        # 독립 정규 잡음의 인접 차이 중앙절대편차를 원래 표준편차로 환산.
        noise = min(deviations) / 0.953872 if deviations else 0
        for sign in (-1, 1):
            signed = values * sign
            candidate = ((signed > max(1.25, 4 * noise)) & ~excluded).astype(np.uint8)
            contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            steps = signed - cv2.erode(signed, np.ones((3, 3), np.uint8))
            fragments = []
            for contour in contours:
                left, top, width, height = cv2.boundingRect(contour)
                if max(width, height) < 4:
                    continue
                boundary_sides = (int(left == 0) + int(top == 0)
                                  + int(left + width == image_width)
                                  + int(top + height == image_height))
                if boundary_sides == 4:
                    continue
                center, sides, angle = cv2.minAreaRect(contour)
                short, long = sorted(sides)
                rectangular = cv2.contourArea(contour) >= 0.9 * max(short * long, 1)
                pixels = candidate[top:top + height, left:left + width]
                signal = sign * residual[top:top + height, left:left + width, channel][pixels > 0]
                if np.percentile(signal, 75) < 2:
                    continue
                straight_side = max(np.mean(pixels[:3]) if top > 0 else 0,
                                    np.mean(pixels[-3:]) if top + height < image_height else 0,
                                    np.mean(pixels[:, :3]) if left > 0 else 0,
                                    np.mean(pixels[:, -3:]) if left + width < image_width else 0) >= 0.9
                raster_area = cv2.contourArea(contour) + cv2.arcLength(contour, True) / 2 + 1
                independent = np.mean(separated[top:top + height, left:left + width][pixels > 0]) >= 0.85
                component = np.zeros((height + 2, width + 2), np.uint8)
                cv2.drawContours(component, [contour - np.array([[[left - 1, top - 1]]])],
                                 -1, 1, cv2.FILLED)
                boundary = (component - cv2.erode(component, np.ones((3, 3), np.uint8)))[1:-1, 1:-1] > 0
                if top == 0:
                    boundary[0] = False
                if top + height == image_height:
                    boundary[-1] = False
                if left == 0:
                    boundary[:, 0] = False
                if left + width == image_width:
                    boundary[:, -1] = False
                boundary &= separated[top:top + height, left:left + width]
                sharp = (np.count_nonzero(boundary) >= 4
                         and np.mean(steps[top:top + height, left:left + width][boundary]
                                     >= max(0.65, np.percentile(signal, 75) * 0.5, 3 * noise)) >= 0.7)
                line = (short <= 4 and long >= 24 and independent and sharp
                        and raster_area >= 0.7 * (short + 1) * (long + 1))
                # 충분히 넓고 닫힌 사각 면은 흐린 워터마크 가장자리도 유지.
                soft_rectangle = boundary_sides == 0 and short >= 24 and rectangular
                panel = (short >= 6 and long >= 24 and (rectangular or straight_side)
                         and (sharp or soft_rectangle)
                         and _faint_panel_edge(residual[..., channel], excluded,
                                               left, top, width, height, sign))
                closed = (boundary_sides == 0 and min(width, height) >= 6
                          and 16 <= cv2.contourArea(contour) <= image_height * image_width * 0.25
                          and independent and sharp)
                if line or panel or closed:
                    # 면의 흐린 가장자리까지 포함하고 가는 선에는 좁은 여백만 추가.
                    margin = 12 if panel else 3
                    box = (center, (sides[0] + 2 * margin, sides[1] + 2 * margin), angle)
                    cv2.fillConvexPoly(protected, np.rint(cv2.boxPoints(box)).astype(np.int32), 1)
                if (2 <= width <= 80 and 1 <= height <= 36
                        and raster_area >= 4
                        and independent and sharp):
                    fragments.append((left, top, width, height))
            if len(fragments) < 2:
                continue
            parts = np.asarray(fragments, np.int32)
            bottoms = parts[:, 1] + parts[:, 3]
            order = np.argsort(bottoms)
            ordered_bottoms = bottoms[order]
            used = np.zeros(len(parts), bool)
            for index in order:
                if used[index]:
                    continue
                left, top, width, height = parts[index]
                tolerance = max(1.5, height * 0.18)
                start, stop = np.searchsorted(ordered_bottoms, [bottoms[index] - tolerance,
                                                               bottoms[index] + tolerance])
                nearby = order[start:stop]
                nearby = nearby[(parts[nearby, 3] >= height * 0.65)
                                & (parts[nearby, 3] <= height * 1.5)]
                nearby = nearby[np.argsort(parts[nearby, 0])]
                gaps = parts[nearby[1:], 0] - (parts[nearby[:-1], 0] + parts[nearby[:-1], 2])
                for group in np.split(nearby, np.flatnonzero(gaps > max(6, 2 * height)) + 1):
                    words = (height >= 4 and len(group) >= 2
                             and np.all(parts[group, 2] >= 2 * parts[group, 3]))
                    minimum = 2 if words else (3 if height <= 3 else 4)
                    if len(group) < minimum:
                        continue
                    x0, y0 = np.min(parts[group, :2], axis=0)
                    x1, y1 = np.max(parts[group, :2] + parts[group, 2:], axis=0)
                    if x1 - x0 < max(24, (6 if words else 4) * height):
                        continue
                    cv2.rectangle(protected, (int(x0 - 3), int(y0 - 3)),
                                  (int(x1 + 2), int(y1 + 2)), 1, cv2.FILLED)
                    used[group] = True
    return protected > 0


def _restore_document_background(rgb, document_mask=None):
    """전역 단색 근접 문서 배경 인증과 획 바깥의 완만한 색 평면 복원.

    분산된 배경색 표본과 반복 문자 영역을 함께 확인한다. 획과 색 경계는
    1픽셀 이웃까지 유지하고, 전역 그라데이션은 평면 기울기로 보존한다.
    """
    report = {"accepted": False, "eligible_fraction": 0.0}
    if min(rgb.shape[:2]) < 64:
        return rgb, report
    small = rgb[::4, ::4]
    quantized = np.clip(small // 8, 0, 31).astype(np.int32)
    keys = (quantized[..., 0] * 1024 + quantized[..., 1] * 32 + quantized[..., 2]).ravel()
    dominant = np.bincount(keys, minlength=32768).argmax()
    color = np.median(small.reshape(-1, 3)[keys == dominant], axis=0)
    near = glyphring._maximum_channel(np.abs(rgb - color)) < 18
    report["dominant_fraction"] = float(np.mean(near))
    if report["dominant_fraction"] < 0.5:
        return rgb, report
    height, width = near.shape
    fractions = [np.mean(near[row * height // 6:(row + 1) * height // 6,
                              column * width // 4:(column + 1) * width // 4])
                 for row in range(6) for column in range(4)]
    report["minimum_tile_support"] = float(min(fractions))
    if report["minimum_tile_support"] < 0.28:
        return rgb, report
    margin = max(3, round(3 * max(height, width) / 1600))
    samples = cv2.erode(near.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)) > 0
    if np.mean(samples) < 0.08:
        return rgb, report
    model = _document_plane(rgb, samples)
    error = glyphring._maximum_channel(np.abs(rgb - model))
    refined = samples & (error < 12)
    if np.mean(refined) < 0.08:
        return rgb, report
    model = _document_plane(rgb, refined)
    error = glyphring._maximum_channel(np.abs(rgb - model))
    report["residual_p90_dn"], report["residual_p99_dn"] = map(
        float, np.percentile(error[samples], [90, 99]))
    if report["residual_p90_dn"] > 6 or report["residual_p99_dn"] > 12:
        return rgb, report
    scale = min(1.0, 1600 / max(rgb.shape[:2]))
    text_pixels = (cv2.resize(rgb, (round(width * scale), round(height * scale)),
                              interpolation=cv2.INTER_AREA) if scale < 1 else rgb)
    median = _median_blur(np.clip(np.rint(text_pixels), 0, 255).astype(np.uint8), 31)
    region = _text_region(text_pixels, glyphring._maximum_channel(np.abs(text_pixels - median)))
    report["text_region_fraction"] = float(np.mean(region))
    if report["text_region_fraction"] < 0.08:
        return rgb, report
    protected = cv2.dilate((error > 12).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    protected |= _coherent_details(rgb, protected)
    faint = _faint_document_details(rgb - model, protected)
    protected |= faint
    if np.any(faint):
        surface_samples = refined & ~protected
        if np.mean(surface_samples) < 0.08:
            return rgb, report
        model = _document_plane(rgb, surface_samples)
        protected |= cv2.dilate((glyphring._maximum_channel(np.abs(rgb - model)) > 12).astype(np.uint8),
                                np.ones((3, 3), np.uint8)) > 0
    output = rgb.copy()
    output[~protected] = model[~protected]
    if document_mask is not None:
        document_mask[~protected] = True
    report.update({"accepted": True, "eligible_fraction": float(np.mean(~protected)),
                   "protected_pixels": int(np.count_nonzero(protected))})
    return output, report


def restore_document_background(rgb: np.ndarray, strength: float = 1.0, *,
                                document_mask=None) -> tuple[np.ndarray, dict]:
    """강도 1.1 이상에서 단색 근접 문서의 최종 배경 복원."""
    if strength < 1.1:
        return rgb, {"accepted": False, "eligible_fraction": 0.0}
    return _restore_document_background(rgb, document_mask=document_mask)


def apply(rgb: np.ndarray, strength: float = 1.0, *, backend=None) -> tuple[np.ndarray, dict]:
    """밀집 문자 후보 주변만 분석하며 배경색 경계와 약한 선 구조 보호."""
    if strength == 0:
        return rgb.copy(), {"eligible_fraction": 0.0}
    height, width = rgb.shape[:2]
    reduced = cv2.resize(rgb, ((width + 3) // 4, (height + 3) // 4), interpolation=cv2.INTER_AREA)
    edges = np.zeros(reduced.shape[:2], bool)
    for channel in range(3):
        gx = cv2.Sobel(reduced[..., channel], cv2.CV_32F, 1, 0, scale=0.125)
        gy = cv2.Sobel(reduced[..., channel], cv2.CV_32F, 0, 1, scale=0.125)
        edges |= gx * gx + gy * gy > 1
    rows = np.flatnonzero(np.any(edges, axis=1))
    columns = np.flatnonzero(np.any(edges, axis=0))
    if not len(rows):
        return rgb.copy(), {"eligible_fraction": 0.0}
    # 검출 격자를 원본 좌표에 맞추고 모델 반경보다 넓은 배경 여백 유지.
    top = max(0, (int(rows[0] * height / reduced.shape[0]) - 160) // 64 * 64)
    left = max(0, (int(columns[0] * width / reduced.shape[1]) - 160) // 96 * 96)
    bottom = min(height, int((rows[-1] + 1) * height / reduced.shape[0]) + 160)
    right = min(width, int((columns[-1] + 1) * width / reduced.shape[1]) + 160)
    del reduced, edges, gx, gy
    handled = np.zeros((bottom - top, right - left), np.uint8)
    corrected, report = _restore_text_region(rgb[top:bottom, left:right], strength, handled, backend=backend)
    corrected, ring_report = glyphring.apply(corrected, strength, excluded=handled)
    output = rgb.copy()
    output[top:bottom, left:right] = corrected
    area_fraction = (bottom - top) * (right - left) / (height * width)
    report["eligible_fraction"] *= area_fraction
    ring_report["eligible_fraction"] *= area_fraction
    report["eligible_fraction"] += ring_report["eligible_fraction"]
    report["glyph_ring"] = ring_report
    if "mean_abs_change" in report:
        report["mean_abs_change"] = [change * area_fraction for change in report["mean_abs_change"]]
    return output, report
