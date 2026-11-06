"""문자 배경의 구역별 국소 평면을 GPU에서 계산."""

from __future__ import annotations

import numpy as np


def _segmented_background_gpu(backend, rgb, samples, labels, stats, region):
    """연결 구역을 장치에 유지하며 OpenCV 상수 경계와 동일한 필터 적용."""
    cp = backend.cp
    ndimage = backend.ndimage
    pixels = cp.asarray(rgb)
    weights = cp.asarray(samples)
    device_labels = cp.asarray(labels)
    background = pixels.copy()
    mass_output = cp.zeros(labels.shape, cp.float32)
    determinant_output = cp.zeros_like(mass_output)
    offsets = np.arange(-120, 121, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / 40) ** 2)
    kernel /= kernel.sum()
    kernels = [cp.asarray(kernel), cp.asarray(kernel * offsets),
               cp.asarray(kernel * offsets * offsets)]

    def filtered(values, horizontal, vertical):
        return ndimage.correlate1d(
            ndimage.correlate1d(values, kernels[horizontal], axis=1, mode="constant"),
            kernels[vertical], axis=0, mode="constant")

    for label in range(1, len(stats)):
        left, top, width, height, area = stats[label]
        if area < 500 or min(width, height) < 16:
            continue
        box = np.s_[top:top + height, left:left + width]
        component_host = labels[box] == label
        if not np.any(region[box][component_host]):
            continue
        component = device_labels[box] == label
        sample = weights[box] * component
        mass = filtered(sample, 0, 0)
        support = cp.maximum(mass, 1e-6)
        mx = filtered(sample, 1, 0) / support
        my = filtered(sample, 0, 1) / support
        vx = filtered(sample, 2, 0) / support - mx * mx
        vy = filtered(sample, 0, 2) / support - my * my
        xy = filtered(sample, 1, 1) / support - mx * my
        determinant = vx * vy - xy * xy
        total = pixels[box] * sample[..., None]
        mean = filtered(total, 0, 0) / support[..., None]
        cx = filtered(total, 1, 0) / support[..., None] - mx[..., None] * mean
        cy = filtered(total, 0, 1) / support[..., None] - my[..., None] * mean
        denominator = cp.maximum(determinant, 1e-4)[..., None]
        sx = (cx * vy[..., None] - cy * xy[..., None]) / denominator
        sy = (cy * vx[..., None] - cx * xy[..., None]) / denominator
        mean -= sx * mx[..., None]
        mean -= sy * my[..., None]
        background[box][component] = mean[component]
        mass_output[box][component] = mass[component]
        determinant_output[box][component] = determinant[component]
    result = cp.asnumpy(cp.concatenate((background, mass_output[..., None],
                                        determinant_output[..., None]), axis=2))
    return result[..., :3].copy(), result[..., 3].copy(), result[..., 4].copy()
