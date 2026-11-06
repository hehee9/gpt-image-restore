"""비초점 영역의 밀집 필터 및 보정장 GPU 연산."""

import math

import cv2
import numpy as np

from ._parallel import _median_blur
from .defocus import _LUMA, _REFERENCE_AREA


def _apply_gpu(rgb, strength, backend):
    """필터를 GPU에서 계산하고 거리 변환과 연결 요소 판정을 CPU에서 유지."""
    cp, ndi = backend.cp, backend.ndimage
    values = cp.asarray(rgb)
    y = values[..., 0] * _LUMA[0] + values[..., 1] * _LUMA[1] + values[..., 2] * _LUMA[2]
    y -= y[0, 0].copy()
    g06, g08, g13, g26, g35 = (backend._gaussian(y, sigma)
                                  for sigma in (0.6, 0.8, 1.3, 2.6, 3.5))
    def _rms(values):
        return cp.sqrt(cp.maximum(ndi.uniform_filter(values * values, 17, mode="mirror"), 0))
    fine = _rms(y - g06)
    band_rms = _rms(g08 - g35)
    weight = (cp.clip((0.65 - fine / cp.maximum(band_rms, 0.1)) / 0.25, 0, 1)
              * cp.clip((1.8 - fine) / 0.6, 0, 1))
    gx, gy = backend._sobel(g35, 1), backend._sobel(g35, 0)
    weight *= cp.clip((6 - cp.hypot(gx, gy)) / 2, 0, 1)
    weight = backend._gaussian(weight, 2.5)
    field_fine, field_band = _rms(y - g13), _rms(g13 - g26)
    support = ((field_fine / cp.maximum(field_band, 0.1) < 0.95)
               & (field_fine < 3)).astype(cp.float32)
    area_scale = y.size / _REFERENCE_AREA
    field_size = max(25, int(round(49 * math.sqrt(area_scale))) | 1)
    occupancy = ndi.uniform_filter(support, field_size, mode="mirror")
    weight *= cp.clip((occupancy - 0.60) / 0.20, 0, 1)

    surface = _median_blur(np.uint8(np.rint(np.clip(rgb, 0, 255))), 9).astype(np.float32)
    surface = cp.asarray(surface)
    red, green, blue = surface.transpose(2, 0, 1)
    surface_luma = red * _LUMA[0] + green * _LUMA[1] + blue * _LUMA[2]
    skin = ((surface_luma >= 25) & (red - green >= 5) & (green - blue >= 2)
            & (red <= 3 * cp.maximum(green, 1)) & (blue >= 0.1 * cp.maximum(red, 1)))
    skin = ndi.maximum_filter(skin, 3, mode="nearest")
    edge = ndi.maximum_filter(y, 3, mode="nearest") - ndi.minimum_filter(y, 3, mode="nearest") > 24
    protected = cp.asnumpy(skin | edge | (fine >= 3))
    distance = cv2.distanceTransform((~protected).astype(np.uint8), cv2.DIST_L2, 5)
    weight *= cp.clip(cp.asarray(distance) / 4, 0, 1)
    weight[cp.asarray(protected)] = 0
    weight_cpu = cp.asnumpy(weight)
    minimum_area = max(128, int(round(512 * area_scale)))
    count, labels = cv2.connectedComponents((weight_cpu > 0.001).astype(np.uint8), connectivity=8)
    core_distance = max(10, 20 * math.sqrt(area_scale))
    core = (weight_cpu > 0.1) & (distance >= core_distance)
    core_area = np.bincount(labels[core], minlength=count)
    keep = core_area >= minimum_area
    keep[0] = False
    rejected = int(np.count_nonzero(~keep[1:]))
    weight_cpu[~keep[labels]] = 0
    weight = cp.asarray(weight_cpu)
    band = g08 - g26
    negative = cp.maximum(0.5 * (cp.sqrt(band * band + 0.25) - band) - 0.25, 0)
    output = values + (negative * weight * strength)[..., None]
    return cp.asnumpy(output), weight_cpu, rejected
