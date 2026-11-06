"""@description 넓은 비초점 영역의 어두운 점형 골 완화."""

from __future__ import annotations

import math

import cv2
import numpy as np

from ._parallel import _median_blur
from .contour import _line_art_context


_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
_REFERENCE_AREA = 941 * 1672


def _field_weight(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """@description 미세 질감, 넓은 비초점 지지와 작은 고립 영역을 이용한 선택."""
    y = rgb @ _LUMA
    # 균일한 면의 가우시안 차분 반올림 오차를 영점 이동으로 제거.
    y -= y[0, 0]
    g06 = cv2.GaussianBlur(y, (0, 0), 0.6)
    g08 = cv2.GaussianBlur(y, (0, 0), 0.8)
    g13 = cv2.GaussianBlur(y, (0, 0), 1.3)
    g26 = cv2.GaussianBlur(y, (0, 0), 2.6)
    g35 = cv2.GaussianBlur(y, (0, 0), 3.5)
    fine = np.sqrt(np.maximum(cv2.blur((y - g06) ** 2, (17, 17)), 0))
    band_rms = np.sqrt(np.maximum(cv2.blur((g08 - g35) ** 2, (17, 17)), 0))
    weight = (np.clip((0.65 - fine / np.maximum(band_rms, 0.1)) / 0.25, 0, 1)
              * np.clip((1.8 - fine) / 0.6, 0, 1))
    gx = cv2.Sobel(g35, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    gy = cv2.Sobel(g35, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    weight *= np.clip((6 - np.hypot(gx, gy)) / 2, 0, 1)
    weight = cv2.GaussianBlur(weight, (0, 0), 2.5)

    field_fine = np.sqrt(np.maximum(cv2.blur((y - g13) ** 2, (17, 17)), 0))
    field_band = np.sqrt(np.maximum(cv2.blur((g13 - g26) ** 2, (17, 17)), 0))
    support = ((field_fine / np.maximum(field_band, 0.1) < 0.95)
               & (field_fine < 3)).astype(np.float32)
    area_scale = y.size / _REFERENCE_AREA
    field_size = max(25, int(round(49 * math.sqrt(area_scale))) | 1)
    occupancy = cv2.blur(support, (field_size, field_size))
    weight *= np.clip((occupancy - 0.60) / 0.20, 0, 1)

    surface = _median_blur(np.uint8(np.rint(np.clip(rgb, 0, 255))), 9).astype(np.float32)
    red, green, blue = surface.transpose(2, 0, 1)
    skin = ((surface @ _LUMA >= 25) & (red - green >= 5) & (green - blue >= 2)
            & (red <= 3 * np.maximum(green, 1)) & (blue >= 0.1 * np.maximum(red, 1)))
    skin = cv2.dilate(skin.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    edge = cv2.morphologyEx(y, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 24
    protected = skin | edge | (fine >= 3)
    distance = cv2.distanceTransform((~protected).astype(np.uint8), cv2.DIST_L2, 5)
    weight *= np.clip(distance / 4, 0, 1)
    weight[protected] = 0

    minimum_area = max(128, int(round(512 * area_scale)))
    count, labels = cv2.connectedComponents((weight > 0.001).astype(np.uint8), connectivity=8)
    # 선명한 피사체를 감싸는 가느다란 띠는 넓은 비초점 면의 근거에서 제외.
    core_distance = max(10, 20 * math.sqrt(area_scale))
    core = (weight > 0.1) & (distance >= core_distance)
    core_area = np.bincount(labels[core], minlength=count)
    keep = core_area >= minimum_area
    keep[0] = False
    rejected = int(np.count_nonzero(~keep[1:]))
    weight[~keep[labels]] = 0
    return g08 - g26, weight, rejected


def apply(rgb: np.ndarray, strength: float = 1.0,
          line_art_context: tuple[bool, float] | None = None, backend=None) -> tuple[np.ndarray, dict]:
    """@description 피부와 작은 고립 영역을 보호한 어두운 대역 감쇠.

    넓은 비초점 영역에서만 음의 가우시안 차분 대역을 완화한다. 공통 RGB
    보정량을 더하며 반올림과 출력 범위 제한은 기존 저장 단계에 맡긴다.
    같은 입력에서 계산한 선화 판정은 재사용할 수 있다.
    """
    if strength < 0 or not math.isfinite(strength):
        raise ValueError("defocus stage strength must be finite and 0 or greater.")
    if strength == 0:
        return rgb.copy(), {"enabled": False, "strength": 0.0, "eligible_fraction": 0.0}
    source = np.asarray(rgb, dtype=np.float32)
    if line_art_context is None:
        line_art_context = _line_art_context(
            source, source @ np.array([0.299, 0.587, 0.114], np.float32))
    line_art, flat_fraction = line_art_context
    if line_art:
        return source.copy(), {
            "enabled": True, "strength": float(strength), "protected_line_art": True,
            "flat_region_fraction": flat_fraction, "eligible_fraction": 0.0,
            "rejected_islands": 0, "mean_abs_change": [0.0] * 3, "max_abs_change": 0.0,
        }
    if backend is not None:
        from ._gpu_defocus import _apply_gpu
        output, weight, rejected = _apply_gpu(source, strength, backend)
        change = output - source
        return output, {
            "enabled": True, "strength": float(strength),
            "protected_line_art": False, "flat_region_fraction": flat_fraction,
            "method": "defocused_negative_band", "eligible_fraction": float(np.mean(weight > 0.1)),
            "rejected_islands": rejected,
            "mean_abs_change": np.mean(np.abs(change), axis=(0, 1)).tolist(),
            "max_abs_change": float(np.max(np.abs(change))),
        }
    band, weight, rejected = _field_weight(source)
    negative = np.maximum(0.5 * (np.sqrt(band * band + 0.25) - band) - 0.25, 0)
    correction = negative * weight * strength
    output = (source + correction[..., None]).astype(np.float32)
    change = output - source
    return output, {
        "enabled": True, "strength": float(strength),
        "protected_line_art": False, "flat_region_fraction": flat_fraction,
        "method": "defocused_negative_band", "eligible_fraction": float(np.mean(weight > 0.1)),
        "rejected_islands": rejected,
        "mean_abs_change": np.mean(np.abs(change), axis=(0, 1)).tolist(),
        "max_abs_change": float(np.max(np.abs(change))),
    }


def analyze(rgb: np.ndarray, backend=None) -> dict:
    """@description 비초점 영역의 선택 비율과 예상 어두운 골 보정량 분석."""
    _, report = apply(rgb, backend=backend)
    return report
