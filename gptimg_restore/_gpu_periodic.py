"""CPU의 피크 적합을 유지하는 주기 성분 GPU 밀집 연산."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import periodic as cpu


def _gaussian(values, sigma, backend):
    """SciPy와 같은 double 계수 및 축별 float32 반올림으로 평활화."""
    cp = backend.cp
    radius = int(4 * sigma + 0.5)
    positions = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 / sigma ** 2 * positions ** 2)
    weights /= weights.sum()
    device_weights = cp.asarray(weights)
    for axis in (0, 1):
        values = backend.ndimage.correlate1d(
            values.astype(cp.float64), device_weights, axis=axis, output=cp.float32)
    return values


def _spectrum_maxima(power, backend, discovery_excess_db, min_amplitude):
    """장치에서 스펙트럼 필터와 피크 마스크 계산."""
    cp, ndi = backend.cp, backend.ndimage
    h, w = power.shape[:2]
    fy = cp.fft.fftfreq(h)[:, None]
    fx = cp.fft.fftfreq(w)[None, :]
    angle = cp.degrees(cp.arctan2(abs(fy), abs(fx)))
    radius = cp.hypot(fx, fy)
    sectors = (angle >= 15) & (angle <= 75) & (radius > 0.1)
    axes = ((abs(fy) < 1.5 / h) & (abs(fx) > 0.18)) | ((abs(fx) < 1.5 / w) & (abs(fy) > 0.18))
    half = (fx > 0) | ((abs(fx) < 1e-12) & (fy > 0))
    relevant = (sectors | axes) & half
    score = cp.full((h, w), -100.0, cp.float32)
    for c in range(3):
        log = 10 * cp.log10(cp.maximum(power[..., c], 1e-20))
        score = cp.maximum(score, log - ndi.uniform_filter(log, size=15, mode="wrap"))
    peak_power = power.max(axis=2)
    maxima = relevant & (score >= discovery_excess_db)
    maxima &= peak_power == ndi.maximum_filter(peak_power, size=3, mode="wrap")
    maxima &= 2 * cp.sqrt(peak_power) >= min_amplitude * 0.25
    return maxima


def _prepare(image, backend, edge_distance, discovery_excess_db, max_components, min_amplitude):
    """한 이미지의 필터, 상대색 변환, FFT를 장치에서 연속 계산."""
    cp, ndi = backend.cp, backend.ndimage
    values = cp.asarray(image)
    smooth = _gaussian(values, 0.8, backend)
    edges = cp.zeros(image.shape[:2], cp.bool_)
    for c in range(3):
        gx = ndi.sobel(smooth[..., c], axis=1) / 8
        gy = ndi.sobel(smooth[..., c], axis=0) / 8
        edges |= cp.hypot(gx, gy) > 8
    clipped = (values.min(axis=2) >= 254.5) | (values.max(axis=2) <= 0.5)
    # 기본 원형 팽창과 동일한 거리 판정. 에지가 없는 EDT의 암묵적 경계는 CPU 유지.
    host_edges, host_clipped = cp.asnumpy(edges), cp.asnumpy(clipped)
    if edge_distance == 3.0 and host_edges.any():
        mask = ~ndi.binary_dilation(edges, structure=cp.asarray(cpu._EDGE_FOOTPRINT)) & ~clipped
        host_mask = cp.asnumpy(mask)
    else:
        host_mask = (ndimage.distance_transform_edt(~host_edges) > edge_distance) & ~host_clipped
        mask = cp.asarray(host_mask)
    if np.count_nonzero(host_mask) < 128:
        return host_mask, host_clipped, None, None, None
    y = values[..., 0] * cp.float32(0.2126) + values[..., 1] * cp.float32(0.7152)
    y += values[..., 2] * cp.float32(0.0722)
    opponent = cp.stack((y, values[..., 0] - values[..., 1], values[..., 2] - values[..., 1]), axis=-1)
    residual = opponent - _gaussian(opponent, cpu._SIGMA, backend)
    center = cp.median(residual[mask], axis=0)
    h, w = image.shape[:2]
    # NumPy의 창 계수와 float32 반올림을 그대로 사용.
    host_window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32) * host_mask
    window = cp.asarray(host_window)
    # 홀수 폭의 나이퀴스트 인접 켤레 피크가 float32 FFT 오차로 서로 가리는 현상 방지.
    transform = cp.fft.fft2(((residual - center) * window[..., None]).astype(cp.float64), axes=(0, 1))
    power = abs(transform).astype(cp.float32)
    # CPU와 같은 float32 누적 합을 사용해 FFT 정규화 배율 유지.
    power /= host_window.sum()
    power **= 2
    maxima = _spectrum_maxima(power, backend, discovery_excess_db, min_amplitude)
    return host_mask, host_clipped, cp.asnumpy(residual), cp.asnumpy(power), cp.asnumpy(maxima)


def _candidates(power, maxima, min_excess_db, discovery_excess_db, max_components):
    """GPU 피크 마스크의 후보만 CPU에서 보간, 정렬, 중복 제거."""
    h, w = power.shape[:2]
    sx, sy = 16 * (w // 16) / w, 16 * (h // 16) / h
    candidates = []
    for nx, ny in cpu._GRID:
        expected = cpu._canonical(nx * sx / 8, ny * sy / 8)
        refined, excess, location, channel = cpu._local_peak(power, *expected)
        if float(excess.max()) < min_excess_db:
            continue
        ex, ey = int(round(expected[0] * w)) % w, int(round(expected[1] * h)) % h
        matched = location == (ey, ex)
        fx, fy = expected if matched else refined
        if not matched and float(excess.max()) < discovery_excess_db:
            continue
        candidates.append({"frequency": (fx, fy), "refined_frequency": refined,
                           "family": [nx, ny] if matched else None,
                           "excess": excess, "channel": channel,
                           "origin_phase": float(np.pi * ((nx / 8) * (sx - 1) + (ny / 8) * (sy - 1))) if matched else 0.0})
    if np.any(maxima):
        locations = np.argwhere(maxima)
        peak_power = power.max(axis=2)
        order = np.argsort(peak_power[maxima])[::-1][:max_components * 4]
        for y, x in locations[order]:
            refined, excess, _, channel = cpu._local_peak(power, x / w, y / h, search_radius=0)
            ys, xs = (y + np.arange(-3, 4)) % h, (x + np.arange(-3, 4)) % w
            patch = power[np.ix_(ys, xs, [channel])][..., 0]
            half_power = patch >= power[y, x, channel] * 0.5
            isolated = not (half_power[0].any() or half_power[-1].any() or half_power[:, 0].any() or half_power[:, -1].any())
            if isolated and float(excess.max()) >= discovery_excess_db:
                candidates.append({"frequency": refined, "refined_frequency": refined,
                                   "family": None, "excess": excess, "channel": channel, "origin_phase": 0.0})
    candidates.sort(key=lambda item: (item["family"] is None, -float(item["excess"].max())))
    unique = []
    for candidate in candidates:
        if all(cpu._frequency_distance(candidate["frequency"], other["frequency"], w, h) >= 2.5 for other in unique):
            unique.append(candidate)
        if len(unique) >= max_components:
            break
    return unique


def _estimate(rgb, backend, *, min_excess_db=12.0, discovery_excess_db=18.0,
              min_amplitude=0.02, discovery_min_amplitude=0.05,
              edge_distance=3.0, spatial_gain=True, max_components=16):
    """GPU 밀집 연산 뒤 기존 CPU 주파수 보정과 강건 회귀 실행."""
    image = np.asarray(rgb, dtype=np.float32)
    h, w = image.shape[:2]
    report = {"stage": "periodic", "status": "no_significant_component", "path": "none",
              "native_size_model_matched": False, "components": [],
              "settings": {"min_excess_db": float(min_excess_db), "discovery_excess_db": float(discovery_excess_db),
                           "min_amplitude": float(min_amplitude), "edge_distance": float(edge_distance),
                           "discovery_min_amplitude": float(discovery_min_amplitude),
                           "spatial_gain": bool(spatial_gain), "max_components": int(max_components)}}
    if min(h, w) < 32:
        return report, None
    mask, clipped, residual, power, maxima = _prepare(
        image, backend, edge_distance, discovery_excess_db, max_components, min_amplitude)
    report["background_fraction"] = float(mask.mean())
    report["clipped_fraction"] = float(clipped.mean())
    indices = np.flatnonzero(mask)
    if len(indices) < 128:
        report["status"] = "insufficient_background"
        return report, None
    candidates = _candidates(power, maxima, min_excess_db, discovery_excess_db, max_components)
    del power, maxima
    report["candidate_count"] = len(candidates)
    if not candidates:
        return report, None
    if len(indices) > cpu._SAMPLES:
        indices = np.random.default_rng(0).choice(indices, cpu._SAMPLES, replace=False)
    y, x = np.divmod(indices, w)
    target = residual.reshape(-1, 3)[indices].astype(np.float64)
    del residual, indices
    for candidate in candidates:
        cpu._refine_background(candidate, x, y, target, w, h)
        candidate["frequency"] = cpu._snap_frequency(candidate["frequency"], w, h)
    separated = []
    for candidate in candidates:
        if all(cpu._frequency_distance(candidate["frequency"], other["frequency"], w, h) >= 2.5 for other in separated):
            separated.append(candidate)
    candidates = separated
    matrix, coefficients, errors, weights, noise = cpu._fit(candidates, x, y, target)
    active = []
    for candidate in candidates:
        start, stop = candidate["columns"]
        threshold = min_excess_db if candidate["family"] is not None else discovery_excess_db
        amplitude_floor = min_amplitude if candidate["family"] is not None else max(min_amplitude, discovery_min_amplitude)
        channels, accepted = {}, False
        for c, name in enumerate(cpu._CHANNEL_NAMES):
            amplitude = float(np.linalg.norm(coefficients[start:stop, c]))
            error = float(np.linalg.norm(errors[start:stop, c]))
            significant = candidate["excess"][c] >= threshold and amplitude >= max(amplitude_floor, 4 * error)
            channels[name] = {"amplitude": amplitude, "spectral_excess_db": float(candidate["excess"][c]),
                              "fit_standard_error": error, "accepted": bool(significant),
                              "cosine_coefficient": float(coefficients[start, c]),
                              "sine_coefficient": float(coefficients[start+1, c]) if stop-start == 2 else 0.0}
            if not significant:
                coefficients[start:stop, c] = 0
            accepted |= significant
        if accepted:
            active.append(candidate)
            report["components"].append({"frequency": [float(v) for v in candidate["frequency"]],
                                         "refined_frequency": [float(v) for v in candidate["refined_frequency"]],
                                         "eighth_grid_vector": candidate["family"],
                                         "resize_origin_phase": candidate["origin_phase"], "channels": channels})
    if not active:
        return report, None
    native = sum(item["family"] is not None for item in active)
    report["status"] = "detected"
    report["path"] = "native" if native == len(active) else "discovered" if native == 0 else "mixed"
    report["native_size_model_matched"] = bool(native)
    model = {"candidates": active, "coefficients": coefficients, "matrix": matrix,
             "target": target, "weights": weights, "noise": noise, "x": x, "y": y,
             "mask": mask, "clipped": clipped, "shape": (h, w), "spatial_gain": spatial_gain}
    return report, model


def _gain_grid(model, candidate, channel):
    """CPU 표본 회귀를 256픽셀의 작은 진폭 격자로 반환."""
    h, w = model["shape"]
    tile = 256
    gh, gw = (h + tile - 1) // tile, (w + tile - 1) // tile
    start, stop = candidate["columns"]
    coefficients = model["coefficients"][:, channel]
    carrier = model["matrix"][:, start:stop] @ coefficients[start:stop]
    other = model["matrix"] @ coefficients - carrier
    target = model["target"][:, channel] - other
    weights = model["weights"][:, channel]
    cells = (model["y"] // tile) * gw + model["x"] // tile
    mass = np.bincount(cells, weights=weights, minlength=gh * gw)
    carrier_mean = np.bincount(cells, weights=weights * carrier, minlength=gh * gw)
    target_mean = np.bincount(cells, weights=weights * target, minlength=gh * gw)
    numerator = np.bincount(cells, weights=weights * carrier * target, minlength=gh * gw)
    denominator = np.bincount(cells, weights=weights * carrier ** 2, minlength=gh * gw)
    occupied = mass > 0
    numerator[occupied] -= carrier_mean[occupied] * target_mean[occupied] / mass[occupied]
    denominator[occupied] -= carrier_mean[occupied] ** 2 / mass[occupied]
    denominator = np.maximum(denominator, 0)
    regularizer = 16 * model["noise"][channel] ** 2
    gain = (numerator + regularizer) / (denominator + regularizer)
    gain = np.clip(gain.reshape(gh, gw), 0, 2)
    for iy in range(gh):
        for ix in range(gw):
            region = model["clipped"][iy*tile:min((iy+1)*tile,h), ix*tile:min((ix+1)*tile,w)]
            if region.mean() > 0.95:
                gain[iy, ix] = 0
    return ndimage.gaussian_filter(gain, 0.6, mode="nearest").astype(np.float32)


def _expand_gain(gain, height, width, tile, backend):
    """CPU와 같은 네 모서리 누적 순서로 장치 진폭 지도 생성."""
    cp = backend.cp
    gh, gw = gain.shape
    x = (cp.arange(width, dtype=cp.float64) + 0.5) / tile - 0.5
    left = cp.floor(x).astype(cp.int64)
    fraction_x = x - left
    left_rows = gain[:, cp.clip(left, 0, gw - 1)] * (1 - fraction_x)
    right_rows = gain[:, cp.clip(left + 1, 0, gw - 1)] * fraction_x
    y = (cp.arange(height, dtype=cp.float64) + 0.5) / tile - 0.5
    top = cp.floor(y).astype(cp.int64)
    fraction_y = (y - top)[:, None]
    upper, lower = cp.clip(top, 0, gh - 1), cp.clip(top + 1, 0, gh - 1)
    values = left_rows[upper] * (1 - fraction_y)
    values += right_rows[upper] * (1 - fraction_y)
    values += left_rows[lower] * fraction_y
    values += right_rows[lower] * fraction_y
    return values.astype(cp.float32)


def _carrier_basis(fx, fy, origin_phase, height, width, backend):
    """float64 위상과 축별 삼각함수로 장치 보정 기저 생성."""
    cp = backend.cp
    phase_x = 2 * np.pi * fx * cp.arange(width, dtype=cp.float64) + origin_phase
    phase_y = 2 * np.pi * fy * cp.arange(height, dtype=cp.float64)
    cx, sx, cy, sy = cp.cos(phase_x), cp.sin(phase_x), cp.cos(phase_y), cp.sin(phase_y)
    cosine = (cy[:, None] * cx - sy[:, None] * sx).astype(cp.float32)
    sine = None if cpu._self_conjugate(fx, fy) else (sy[:, None] * cx + cy[:, None] * sx).astype(cp.float32)
    return cosine, sine


def _reconstruct(rgb, model, backend, strength):
    """진폭 격자를 일괄 전송한 뒤 전체 보정장과 RGB 출력 합성."""
    cp = backend.cp
    h, w = model["shape"]
    grids, positions = [], {}
    for i, candidate in enumerate(model["candidates"]):
        start, stop = candidate["columns"]
        for c in range(3):
            if model["spatial_gain"] and np.any(model["coefficients"][start:stop, c]):
                positions[i, c] = len(grids)
                grids.append(_gain_grid(model, candidate, c))
    device_grids = cp.asarray(np.stack(grids)) if grids else None
    coefficients = cp.asarray(model["coefficients"])
    correction = cp.zeros((h, w, 3), cp.float32)
    for i, candidate in enumerate(model["candidates"]):
        fx, fy = candidate["frequency"]
        cosine, sine = _carrier_basis(fx, fy, candidate["origin_phase"], h, w, backend)
        start, stop = candidate["columns"]
        for c in range(3):
            if not np.any(model["coefficients"][start:stop, c]):
                continue
            carrier = coefficients[start, c] * cosine
            if sine is not None:
                carrier = carrier + coefficients[start+1, c] * sine
            if model["spatial_gain"]:
                carrier *= _expand_gain(device_grids[positions[i, c]], h, w, 256, backend)
            correction[..., c] += carrier
    clipped = cp.asarray(model["clipped"])
    protection = 1 - _gaussian(clipped.astype(cp.float32), 2, backend)
    protection[clipped] = 0
    correction *= protection[..., None]
    y, u, v = cp.moveaxis(correction, -1, 0)
    green = y - 0.2126 * u - 0.0722 * v
    removed = cp.stack((green + u, green, green + v), axis=-1) * cp.float32(strength)
    # 보고서 축약값도 출력과 함께 한 번에 반환.
    output = cp.asarray(rgb, dtype=cp.float32) - removed
    stats = cp.concatenate((cp.sqrt(cp.mean(removed ** 2))[None], cp.mean(abs(removed), axis=(0, 1))))
    host_output, host_stats = cp.asnumpy(output), cp.asnumpy(stats)
    return host_output, float(host_stats[0]), [float(v) for v in host_stats[1:]]


def _apply(rgb, backend, strength=1.0, **options):
    """주기 성분 GPU 경로와 기존 보고서 계약 연결."""
    if strength == 0:
        return cpu.apply(rgb, strength=0, **options)
    report, model = _estimate(rgb, backend, **options)
    report["strength"] = float(strength)
    if model is None:
        report.update(removed_rms=0.0, mean_abs_change=[0.0] * 3)
        return np.asarray(rgb, dtype=np.float32).copy(), report
    output, rms, changes = _reconstruct(rgb, model, backend, strength)
    report["removed_rms"], report["mean_abs_change"] = rms, changes
    return output.astype(np.float32, copy=False), report
