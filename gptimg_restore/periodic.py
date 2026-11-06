"""크기 보정 격자와 독립적인 스펙트럼 피크의 보수적 제거."""

from __future__ import annotations

import os

import numpy as np
from scipy import fft, ndimage, optimize
from ._parallel import _SERIAL, _executor


_CHANNEL_NAMES = ("luma", "r_minus_g", "b_minus_g")
_GRID = (
    (1, -1), (1, 1), (2, -2), (2, 2), (3, -3), (3, 3),
    (3, -1), (3, 1), (1, -3), (1, 3),
    (4, 0), (0, 4), (4, -4), (4, 4),
)
_SIGMA = 1.2
_SAMPLES = 65536
_EDGE_OFFSETS = np.arange(-3, 4)
_EDGE_FOOTPRINT = _EDGE_OFFSETS[:, None] ** 2 + _EDGE_OFFSETS[None, :] ** 2 <= 9


def _canonical(fx, fy):
    """실수 신호의 켤레 주파수를 한 대표 좌표로 정규화."""
    fx, fy = (fx + 0.5) % 1 - 0.5, (fy + 0.5) % 1 - 0.5
    if abs(fx + 0.5) < 1e-10:
        fx, fy = 0.5, abs(fy)
    elif fx < -1e-10 or (abs(fx) < 1e-10 and fy < 0):
        fx, fy = -fx, -fy
    if abs(fy + 0.5) < 1e-10:
        fy = 0.5
    return float(fx), float(fy)


def _self_conjugate(fx, fy):
    """사인 기저가 사라지는 DC/나이퀴스트 조합 판정."""
    return all(min(abs(f), abs(abs(f) - 0.5)) < 1e-9 for f in (fx, fy))


def _frequency_distance(a, b, width, height):
    """DFT 경계와 실수 신호의 켤레를 함께 고려한 빈 단위 거리."""
    scale = np.array([width, height])
    a, b = np.asarray(a), np.asarray(b)
    difference = (a - b + 0.5) % 1 - 0.5
    conjugate = (a + b + 0.5) % 1 - 0.5
    return min(np.linalg.norm(difference * scale), np.linalg.norm(conjugate * scale))


def _snap_frequency(frequency, width, height):
    """분해능보다 훨씬 작은 나이퀴스트 보간 오차로 사인 기저가 생기는 현상 방지."""
    nearest = tuple(0.0 if abs(value) < 0.25 else np.copysign(0.5, value) for value in frequency)
    distance = np.linalg.norm((np.asarray(frequency) - nearest) * [width, height])
    return _canonical(*nearest) if distance < 0.05 else frequency


def _opponent(rgb):
    y = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return np.stack((y, rgb[..., 0] - rgb[..., 1], rgb[..., 2] - rgb[..., 1]), axis=-1)


def _rgb(opponent):
    y, u, v = np.moveaxis(opponent, -1, 0)
    green = y - 0.2126 * u - 0.0722 * v
    return np.stack((green + u, green, green + v), axis=-1)


def _background_mask(rgb, edge_distance):
    """주기 성분 자체를 에지로 오인하지 않도록 약하게 평활한 RGB 사용."""
    smooth = ndimage.gaussian_filter(rgb, (0.8, 0.8, 0))
    edges = np.zeros(rgb.shape[:2], bool)
    for c in range(3):
        gx = ndimage.sobel(smooth[..., c], axis=1) / 8
        gy = ndimage.sobel(smooth[..., c], axis=0) / 8
        edges |= np.hypot(gx, gy) > 8
    clipped = (rgb.min(axis=2) >= 254.5) | (rgb.max(axis=2) <= 0.5)
    if edge_distance == 3.0 and edges.any():
        background = ~ndimage.binary_dilation(edges, structure=_EDGE_FOOTPRINT)
    else:
        background = ndimage.distance_transform_edt(~edges) > edge_distance
    mask = background & ~clipped
    return mask, clipped


def _local_peak(power, fx, fy, search_radius=1):
    """주변 중앙값 대비 피크 강도와 포물선 보간 좌표 측정."""
    h, w = power.shape[:2]
    cy, cx = int(round(fy * h)) % h, int(round(fx * w)) % w
    offsets = np.arange(-search_radius, search_radius + 1)
    ys, xs = (cy + offsets) % h, (cx + offsets) % w
    patch = power[np.ix_(ys, xs, np.arange(3))]
    radius = 7
    by = (cy + np.arange(-radius, radius + 1)) % h
    bx = (cx + np.arange(-radius, radius + 1)) % w
    background = power[np.ix_(by, bx, np.arange(3))]
    ring = np.ones((2 * radius + 1, 2 * radius + 1), bool)
    ring[radius-2:radius+3, radius-2:radius+3] = False
    floor = np.maximum(np.median(background[ring], axis=0), 1e-20)
    ratios = patch / floor
    py, px, channel = np.unravel_index(np.argmax(ratios), ratios.shape)
    iy, ix = int(ys[py]), int(xs[px])
    excess = 10 * np.log10(np.maximum(power[iy, ix], 1e-20) / floor)
    coordinates = []
    for vector, pos, length in (
        (power[iy, (ix + np.array([-1, 0, 1])) % w, channel], ix, w),
        (power[(iy + np.array([-1, 0, 1])) % h, ix, channel], iy, h),
    ):
        log = np.log(np.maximum(vector, 1e-20))
        denominator = log[0] - 2 * log[1] + log[2]
        delta = 0.0 if abs(denominator) < 1e-12 else 0.5 * (log[0] - log[2]) / denominator
        coordinates.append((pos + np.clip(delta, -0.5, 0.5)) / length)
    refined = _canonical(*coordinates)
    return refined, excess, (iy, ix), int(channel)


def _candidates(power, min_excess_db, discovery_excess_db, max_components, min_amplitude):
    h, w = power.shape[:2]
    sx, sy = 16 * (w // 16) / w, 16 * (h // 16) / h
    candidates = []
    for nx, ny in _GRID:
        expected = _canonical(nx * sx / 8, ny * sy / 8)
        refined, excess, location, channel = _local_peak(power, *expected)
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
    fy = fft.fftfreq(h)[:, None]
    fx = fft.fftfreq(w)[None, :]
    abs_fx = abs(fx)
    relevant = np.empty((h, w), dtype=bool)
    for start in range(0, h, 128):
        stop = min(start + 128, h)
        row_fy = fy[start:stop]
        abs_fy = abs(row_fy)
        angle = np.degrees(np.arctan2(abs_fy, abs_fx))
        radius = np.hypot(fx, row_fy)
        sectors = ((angle >= 15) & (angle <= 75) & (radius > 0.1))
        axes = ((abs_fy < 1.5 / h) & (abs_fx > 0.18)) | ((abs_fx < 1.5 / w) & (abs_fy > 0.18))
        half = (fx > 0) | ((abs_fx < 1e-12) & (row_fy > 0))
        relevant[start:stop] = (sectors | axes) & half
    del abs_fx, abs_fy, angle, radius, sectors, axes, half, row_fy
    score = np.full((h, w), -100.0, np.float32)
    for c in range(3):
        log = 10 * np.log10(np.maximum(power[..., c], 1e-20))
        score = np.maximum(score, log - ndimage.uniform_filter(log, size=15, mode="wrap"))
    peak_power = power.max(axis=2)
    maxima = relevant & (score >= discovery_excess_db)
    maxima &= peak_power == ndimage.maximum_filter(peak_power, size=3, mode="wrap")
    maxima &= 2 * np.sqrt(peak_power) >= min_amplitude * 0.25
    if np.any(maxima):
        locations = np.argwhere(maxima)
        order = np.argsort(peak_power[maxima])[::-1][:max_components * 4]
        del score, peak_power, relevant, maxima
        for y, x in locations[order]:
            refined, excess, _, channel = _local_peak(power, x / w, y / h, search_radius=0)
            ys = (y + np.arange(-3, 4)) % h
            xs = (x + np.arange(-3, 4)) % w
            patch = power[np.ix_(ys, xs, [channel])][..., 0]
            half_power = patch >= power[y, x, channel] * 0.5
            isolated = not (half_power[0].any() or half_power[-1].any() or half_power[:, 0].any() or half_power[:, -1].any())
            if isolated and float(excess.max()) >= discovery_excess_db:
                candidates.append({"frequency": refined, "refined_frequency": refined,
                                   "family": None, "excess": excess, "channel": channel, "origin_phase": 0.0})
    else:
        del score, peak_power, relevant, maxima
    candidates.sort(key=lambda item: (item["family"] is None, -float(item["excess"].max())))
    unique = []
    for candidate in candidates:
        duplicate = False
        for other in unique:
            if _frequency_distance(candidate["frequency"], other["frequency"], w, h) < 2.5:
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
        if len(unique) >= max_components:
            break
    return unique


def _highpass_gain(fx, fy):
    positions = np.arange(-5, 6)
    kernel = np.exp(-0.5 * (positions / _SIGMA) ** 2)
    kernel /= kernel.sum()
    return float(1 - np.dot(kernel, np.cos(2 * np.pi * fx * positions)) * np.dot(kernel, np.cos(2 * np.pi * fy * positions)))


def _refine_background(candidate, x, y, target, w, h):
    """비정수 주파수의 작은 FFT 보간 오차를 배경 표본으로 보정."""
    native = candidate["family"] is not None
    # 같은 FFT 빈이라도 중심이 0.1빈 이상 벗어나면 크기 격자 좌표 고정 해제
    if native and _frequency_distance(candidate["frequency"], candidate["refined_frequency"], w, h) < 0.1:
        return
    fx, fy = candidate["frequency"]
    channel = candidate["channel"]
    t = target[:, channel] - np.mean(target[:, channel])

    def objective(offset):
        phase = 2 * np.pi * ((fx + offset[0] / w) * x + (fy + offset[1] / h) * y)
        cosine, sine = np.cos(phase), np.sin(phase)
        return -((np.dot(t, cosine) ** 2 + np.dot(t, sine) ** 2) / len(t) ** 2)

    result = optimize.minimize(objective, [0.0, 0.0], method="Nelder-Mead",
                               bounds=[(-0.6, 0.6), (-0.6, 0.6)],
                               options={"maxiter": 35, "xatol": 0.002, "fatol": 1e-10,
                                        "initial_simplex": np.array([[0, 0], [0.1, 0], [0, 0.1]])})
    refined = _canonical(fx + result.x[0] / w, fy + result.x[1] / h)
    if not native or _frequency_distance((fx, fy), refined, w, h) >= 0.1:
        candidate["frequency"] = refined


def _fit(candidates, x, y, target):
    columns = []
    for candidate in candidates:
        fx, fy = candidate["frequency"]
        phase = 2 * np.pi * (fx * x + fy * y) + candidate["origin_phase"]
        gain = _highpass_gain(fx, fy)
        start = len(columns)
        columns.append(np.cos(phase) * gain)
        if not _self_conjugate(fx, fy):
            columns.append(np.sin(phase) * gain)
        candidate["columns"] = (start, len(columns))
    matrix = np.column_stack(columns + [np.ones(len(x))])
    coefficients = np.zeros((matrix.shape[1], 3), np.float64)
    errors = np.zeros_like(coefficients)
    weights = np.ones_like(target, dtype=np.float64)
    noise = np.ones(3)
    for c in range(3):
        for iteration in range(4):
            weighted = matrix * weights[:, c, None]
            gram = matrix.T @ weighted
            coefficients[:, c] = np.linalg.solve(gram, weighted.T @ target[:, c])
            residual = target[:, c] - matrix @ coefficients[:, c]
            noise[c] = max(1.4826 * np.median(abs(residual - np.median(residual))), 1e-5)
            if iteration < 3:
                weights[:, c] = np.minimum(1.0, 2.5 * noise[c] / np.maximum(abs(residual), 1e-12))
        errors[:, c] = noise[c] * np.sqrt(np.maximum(np.diag(np.linalg.inv(gram)), 0))
    return matrix[:, :-1], coefficients[:-1], errors[:-1], weights, noise


def _estimate(rgb, *, min_excess_db=12.0, discovery_excess_db=18.0,
              min_amplitude=0.02, discovery_min_amplitude=0.05,
              edge_distance=3.0, spatial_gain=True,
              max_components=16):
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
    mask, clipped = _background_mask(image, edge_distance)
    report["background_fraction"] = float(mask.mean())
    report["clipped_fraction"] = float(clipped.mean())
    indices = np.flatnonzero(mask)
    if len(indices) < 128:
        report["status"] = "insufficient_background"
        return report, None
    opponent = _opponent(image)
    residual = opponent - ndimage.gaussian_filter(opponent, (_SIGMA, _SIGMA, 0))
    del opponent
    center = np.median(residual[mask], axis=0)
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32) * mask
    transform_input = residual - center
    transform_input *= window[..., None]
    transform = fft.fft2(transform_input, axes=(0, 1), workers=min(2, os.cpu_count() or 1))
    del transform_input
    power = abs(transform)
    power /= window.sum()
    power **= 2
    del transform
    del window
    candidates = _candidates(power, min_excess_db, discovery_excess_db, max_components, min_amplitude)
    del power
    report["candidate_count"] = len(candidates)
    if not candidates:
        return report, None
    if len(indices) > _SAMPLES:
        indices = np.random.default_rng(0).choice(indices, _SAMPLES, replace=False)
    y, x = np.divmod(indices, w)
    target = residual.reshape(-1, 3)[indices].astype(np.float64)
    del residual, indices
    for candidate in candidates:
        _refine_background(candidate, x, y, target, w, h)
        candidate["frequency"] = _snap_frequency(candidate["frequency"], w, h)
    separated = []
    for candidate in candidates:
        if all(_frequency_distance(candidate["frequency"], other["frequency"], w, h) >= 2.5 for other in separated):
            separated.append(candidate)
    candidates = separated
    matrix, coefficients, errors, weights, noise = _fit(candidates, x, y, target)
    active = []
    for candidate in candidates:
        start, stop = candidate["columns"]
        threshold = min_excess_db if candidate["family"] is not None else discovery_excess_db
        amplitude_floor = min_amplitude if candidate["family"] is not None else max(min_amplitude, discovery_min_amplitude)
        channels = {}
        accepted = False
        for c, name in enumerate(_CHANNEL_NAMES):
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


def _slow_gain(model, candidate, channel):
    """256픽셀 격자의 배경 추정치를 평활한 선택적 진폭 지도로 변환."""
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
    # 클리핑으로 표본 위상이 불균형해도 국소 DC가 진폭으로 새지 않도록 중심화
    occupied = mass > 0
    numerator[occupied] -= carrier_mean[occupied] * target_mean[occupied] / mass[occupied]
    denominator[occupied] -= carrier_mean[occupied] ** 2 / mass[occupied]
    denominator = np.maximum(denominator, 0)
    regularizer = 16 * model["noise"][channel] ** 2
    gain = (numerator + regularizer) / (denominator + regularizer)
    gain = np.clip(gain.reshape(gh, gw), 0, 2)
    # 완전히 클리핑된 셀은 원래 패턴을 추정할 수 없으므로 보정 억제
    for iy in range(gh):
        for ix in range(gw):
            region = model["clipped"][iy*tile:min((iy+1)*tile,h), ix*tile:min((ix+1)*tile,w)]
            if region.mean() > 0.95:
                gain[iy, ix] = 0
    gain = ndimage.gaussian_filter(gain, 0.6, mode="nearest")
    return _expand_gain(gain.astype(np.float32), h, w, tile)


def _expand_gain(gain, height, width, tile):
    """반 픽셀 격자의 선형 진폭 보간과 행 단위 메모리 제한."""
    gh, gw = gain.shape
    x = (np.arange(width, dtype=np.float64) + 0.5) / tile - 0.5
    left = np.floor(x).astype(np.intp)
    fraction_x = x - left
    left_rows = gain[:, np.clip(left, 0, gw - 1)] * (1 - fraction_x)
    right_rows = gain[:, np.clip(left + 1, 0, gw - 1)] * fraction_x
    output = np.empty((height, width), np.float32)
    for start in range(0, height, 128):
        stop = min(start + 128, height)
        y = (np.arange(start, stop, dtype=np.float64) + 0.5) / tile - 0.5
        top = np.floor(y).astype(np.intp)
        fraction_y = (y - top)[:, None]
        upper, lower = np.clip(top, 0, gh - 1), np.clip(top + 1, 0, gh - 1)
        # ndimage.zoom과 같은 네 모서리 누적 순서
        values = left_rows[upper] * (1 - fraction_y)
        values += right_rows[upper] * (1 - fraction_y)
        values += left_rows[lower] * fraction_y
        values += right_rows[lower] * fraction_y
        output[start:stop] = values
    return output


def _carrier_basis(fx, fy, origin_phase, height, width):
    """축별 삼각함수와 덧셈 정리를 이용한 주기 기저 생성."""
    phase_x = 2 * np.pi * fx * np.arange(width) + origin_phase
    phase_y = 2 * np.pi * fy * np.arange(height)
    cx, sx = np.cos(phase_x), np.sin(phase_x)
    cy, sy = np.cos(phase_y), np.sin(phase_y)
    cosine = np.empty((height, width), np.float32)
    sine = None if _self_conjugate(fx, fy) else np.empty_like(cosine)
    for start in range(0, height, 128):
        stop = min(start + 128, height)
        cosine[start:stop] = cy[start:stop, None] * cx - sy[start:stop, None] * sx
        if sine is not None:
            sine[start:stop] = sy[start:stop, None] * cx + cy[start:stop, None] * sx
    return cosine, sine


def analyze(rgb: np.ndarray, *, backend=None, **options) -> dict:
    """유의한 좁은 피크만 진단하며 이미지 수정 생략.

    기본 문턱은 알려진 격자에서 12dB, 새 피크에서 18dB이다.
    배경 적합 진폭 0.02DN(새 피크 0.05DN)와 표준오차의 4배도 함께 넘어야 한다.
    반복 무늬의 주파수와 겹칠 수 있어 문자/강한 에지는 적합에서 제외한다.
    기본 공간 진폭 보정은 256픽셀 격자를 평활하며, spatial_gain=False로 고정 가능하다.
    """
    if backend is not None:
        from ._gpu_periodic import _estimate as estimate_gpu

        report, _ = estimate_gpu(rgb, backend, **options)
        return report
    report, _ = _estimate(rgb, **options)
    return report


def apply(rgb: np.ndarray, strength: float = 1.0, *, backend=None, **options) -> tuple[np.ndarray, dict]:
    """유의한 실수 사인/코사인 성분 제거. 출력 클리핑과 반올림 생략."""
    if backend is not None:
        from ._gpu_periodic import _apply

        return _apply(rgb, backend, strength=strength, **options)
    if strength == 0:
        return rgb.copy(), {"stage": "periodic", "status": "disabled", "components": [],
                            "strength": 0.0, "removed_rms": 0.0, "mean_abs_change": [0.0] * 3}
    report, model = _estimate(rgb, **options)
    report["strength"] = float(strength)
    if model is None:
        report.update(removed_rms=0.0, mean_abs_change=[0.0] * 3)
        return np.asarray(rgb, dtype=np.float32).copy(), report
    h, w = model["shape"]
    correction = np.zeros((h, w, 3), np.float32)
    executor = _executor(max_workers=min(3, os.cpu_count() or 1)) if model["spatial_gain"] and not _SERIAL else None
    for candidate in model["candidates"]:
        fx, fy = candidate["frequency"]
        cosine, sine = _carrier_basis(fx, fy, candidate["origin_phase"], h, w)
        start, stop = candidate["columns"]
        gains = {
            c: executor.submit(_slow_gain, model, candidate, c)
            for c in range(3) if np.any(model["coefficients"][start:stop, c])
        } if executor else {}
        for c in range(3):
            values = model["coefficients"][start:stop, c]
            if not np.any(values):
                continue
            carrier = values[0] * cosine
            if sine is not None:
                carrier = carrier + values[1] * sine
            if model["spatial_gain"]:
                carrier *= gains[c].result() if executor else _slow_gain(model, candidate, c)
            correction[..., c] += carrier
    if executor:
        executor.shutdown()
    protection = 1 - ndimage.gaussian_filter(model["clipped"].astype(np.float32), 2)
    protection[model["clipped"]] = 0
    correction *= protection[..., None]
    removed = _rgb(correction) * np.float32(strength)
    output = np.asarray(rgb, dtype=np.float32) - removed
    report["removed_rms"] = float(np.sqrt(np.mean(removed ** 2)))
    report["mean_abs_change"] = [float(v) for v in np.mean(abs(removed), axis=(0, 1))]
    return output.astype(np.float32, copy=False), report
