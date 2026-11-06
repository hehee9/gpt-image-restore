"""얼룩 추정의 밀집 연산을 GPU에서 수행."""

import cv2
import numpy as np


def _linear_box_axis(rgb, allowed, radius, axis, backend):
    cp = backend.cp
    average = backend.ndimage.uniform_filter1d(rgb, 2 * radius + 1, axis=axis, mode="mirror")
    if radius == 1:
        output = rgb.copy()
        if axis == 1:
            interior = allowed[:, 1:-1] & allowed[:, :-2] & allowed[:, 2:]
            output[:, 1:-1] = cp.where(interior[..., None], average[:, 1:-1], output[:, 1:-1])
        else:
            interior = allowed[1:-1] & allowed[:-2] & allowed[2:]
            output[1:-1] = cp.where(interior[..., None], average[1:-1], output[1:-1])
        return output
    pixels = rgb if axis == 1 else rgb.transpose(1, 0, 2)
    mask = allowed if axis == 1 else allowed.T
    output = average if axis == 1 else average.transpose(1, 0, 2)
    width = pixels.shape[1]
    x = cp.arange(width, dtype=cp.int32)[None, :]
    for top in range(0, pixels.shape[0], 48):
        block = pixels[top:top + 48]
        selected = mask[top:top + 48]
        starts = backend.ndimage.maximum_filter1d(
            cp.where(selected, 0, x + 1), radius + 1, axis=1,
            origin=radius // 2, mode="constant", cval=0)
        stops = backend.ndimage.minimum_filter1d(
            cp.where(selected, width, x), radius + 1, axis=1,
            origin=-(radius + 1) // 2, mode="constant", cval=width)
        left = cp.maximum(x - radius, starts)
        right = cp.minimum(x + radius + 1, stops)
        partial = selected & ((left != x - radius) | (right != x + radius + 1))
        rows, columns = cp.nonzero(partial)
        left, right = left[partial], right[partial]
        count = right - left
        mean_x = (left + right - 1) * 0.5
        shape = (block.shape[0], width + 1, 3)
        sums = cp.zeros(shape, dtype=cp.float64)
        moments = cp.zeros(shape, dtype=cp.float64)
        sums[:, 1:] = cp.cumsum(block, axis=1, dtype=cp.float64)
        moments[:, 1:] = cp.cumsum(block * x[..., None], axis=1, dtype=cp.float64)
        mean = (sums[rows, right] - sums[rows, left]) / count[..., None]
        covariance = (moments[rows, right] - moments[rows, left]) / count[..., None]
        covariance -= mean_x[..., None] * mean
        variance = cp.where(count > 1, (count * count - 1) / 12, 1)
        fitted = mean + covariance * ((columns - mean_x) / variance)[..., None]
        output[top:top + 48][partial] = fitted
        output[top:top + 48] = cp.where(selected[..., None], output[top:top + 48], block)
    return output if axis == 1 else output.transpose(1, 0, 2)


def estimate(rgb, backend):
    """CPU 연결 요소와 거리 판정 후 GPU에서 보정장과 가중치 계산."""
    edges = np.zeros(rgb.shape[:2], bool)
    for channel in range(3):
        gx = cv2.Sobel(rgb[..., channel], cv2.CV_32F, 1, 0, scale=0.125)
        gy = cv2.Sobel(rgb[..., channel], cv2.CV_32F, 0, 1, scale=0.125)
        edges |= gx * gx + gy * gy > 16
    protected = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(1 - protected, 8)
    eligible = (stats[:, cv2.CC_STAT_AREA] >= 500) & (np.minimum(
        stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= 16)
    eligible[0] = False
    allowed = eligible[labels]
    distance = cv2.distanceTransform(np.pad(allowed.astype(np.uint8), 1),
                                     cv2.DIST_L2, 5)[1:-1, 1:-1]
    cp = backend.cp
    values, allowed, distance = cp.asarray(rgb), cp.asarray(allowed), cp.asarray(distance)
    fine_image = values
    for radius in (1, 1):
        for axis in (1, 0):
            fine_image = _linear_box_axis(fine_image, allowed, radius, axis, backend)
    band = values
    for axis in (1, 0):
        band = _linear_box_axis(band, allowed, 13, axis, backend)
    band = fine_image - band
    texture_samples = allowed * cp.clip((distance - 1) / 2, 0, 1)
    fine = cp.zeros(rgb.shape[:2], cp.float32)
    band_energy = cp.zeros_like(fine)
    xx, yy, xy = cp.zeros_like(fine), cp.zeros_like(fine), cp.zeros_like(fine)
    for channel in range(3):
        fine += (values[..., channel] - fine_image[..., channel]) ** 2 / 3
        band_energy += band[..., channel] ** 2 / 3
        sx = backend._sobel(band[..., channel], 1)
        sy = backend._sobel(band[..., channel], 0)
        xx += sx * sx / 3
        yy += sy * sy / 3
        xy += sx * sy / 3
    support = backend._gaussian(texture_samples, 4)
    fine = backend._gaussian(fine * texture_samples, 4) / cp.maximum(support, 1e-6)
    texture_weight = cp.clip((1.5 - cp.sqrt(fine)) / 0.65, 0, 1)
    band_energy = backend._gaussian(band_energy * allowed, 5)
    band_energy /= cp.maximum(backend._gaussian(allowed.astype(cp.float32), 5), 1e-6)
    contrast_weight = cp.clip((4 - cp.sqrt(band_energy)) / 1.5, 0, 1)
    support = cp.maximum(backend._gaussian(texture_samples, 10), 1e-6)
    xx = backend._gaussian(xx * texture_samples, 10) / support
    yy = backend._gaussian(yy * texture_samples, 10) / support
    xy = backend._gaussian(xy * texture_samples, 10) / support
    coherence = cp.sqrt((xx - yy) ** 2 + 4 * xy * xy) / (xx + yy + 1e-6)
    structure_weight = 1 - cp.clip((coherence - 0.7) / 0.2, 0, 1) * cp.clip(
        (cp.sqrt(xx + yy) - 0.3) / 0.4, 0, 1)
    weight = (texture_weight * contrast_weight * structure_weight
              * cp.clip((distance - 0.5) / 1.5, 0, 1))
    weight[~allowed | cp.all(values >= 250, axis=2)] = 0
    return cp.asnumpy(band), cp.asnumpy(weight)
