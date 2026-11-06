"""구조를 보호하면서 표면의 중간 크기 미세 질감을 완화한다."""

import cv2
import numpy as np


def apply(
    rgb: np.ndarray,
    strength: float = 1.0,
    *,
    luma_strength: float = 0.45,
    chroma_strength: float = 0.8,
    **options,
) -> tuple[np.ndarray, dict]:
    """3~8픽셀 대역을 약한 구조 영역에서 채도 성분 위주로 줄인다.

    sigma 1과 3의 가우시안 차이로 대역을 분리한다. 밝기 배율 0.45와
    색차 배율 0.8은 표면 개선과 디테일 보존 사이의 기본 절충값이다.
    기울기, 구조 텐서, 주변 강한 에지와 밝은 클리핑 영역으로 마스크를
    만들고 부드럽게 연결한다. 세 채널이 모두 250 이상인 픽셀은 유지한다.
    strength는 각 배율을 선형으로 조절한다.
    """
    if strength == 0:
        return rgb.copy(), {"masked_fraction": 0.0, "mean_abs_change": [0.0] * 3}

    y = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    rx = cv2.Sobel(rgb[..., 0] - y, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    ry = cv2.Sobel(rgb[..., 0] - y, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    bx = cv2.Sobel(rgb[..., 2] - y, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    by = cv2.Sobel(rgb[..., 2] - y, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    gradient_xx = gx * gx + 0.5 * (rx * rx + bx * bx)
    gradient_yy = gy * gy + 0.5 * (ry * ry + by * by)
    gradient_xy = gx * gy + 0.5 * (rx * ry + bx * by)
    xx = cv2.GaussianBlur(gradient_xx, (0, 0), 2.0)
    yy = cv2.GaussianBlur(gradient_yy, (0, 0), 2.0)
    xy = cv2.GaussianBlur(gradient_xy, (0, 0), 2.0)
    energy = xx + yy
    coherence = np.sqrt((xx - yy) ** 2 + 4 * xy * xy) / (energy + 1e-6)
    gradient_protection = np.clip((np.sqrt(energy) - 1.5) / 5.0, 0, 1)
    coherent_protection = np.clip((coherence - 0.45) / 0.4, 0, 1)
    coherent_protection *= np.clip((np.sqrt(energy) - 0.7) / 2.5, 0, 1)
    strong_edges = (gradient_xx + gradient_yy > 49.0).astype(np.float32)
    edge_density = cv2.GaussianBlur(strong_edges, (0, 0), 2.0)
    edge_protection = np.clip(edge_density * 4.0, 0, 1)
    protection = np.maximum(np.maximum(gradient_protection, coherent_protection), edge_protection)
    protection = cv2.GaussianBlur(protection, (0, 0), 1.5)
    white = np.clip((rgb.min(axis=2) - 245.0) / 8.0, 0, 1)
    white = cv2.GaussianBlur(white, (0, 0), 2.0)
    mask = (1.0 - protection) * (1.0 - white)
    mask[rgb.min(axis=2) >= 250.0] = 0.0

    low = cv2.GaussianBlur(rgb, (0, 0), 1.0)
    high = cv2.GaussianBlur(rgb, (0, 0), 3.0)
    band = low - high
    band_y = band[..., 0] * 0.299 + band[..., 1] * 0.587 + band[..., 2] * 0.114
    correction = (chroma_strength * band + (luma_strength - chroma_strength) * band_y[..., None])
    correction *= (strength * mask)[..., None]
    output = rgb - correction
    return output, {
        "masked_fraction": float(np.mean(mask > 0.1)),
        "mean_abs_change": [float(v) for v in np.mean(np.abs(correction), axis=(0, 1))],
    }
