"""@description 보정된 픽셀의 8비트 저장 밴딩 완화."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


_LUMA = np.array([0.299, 0.587, 0.114], np.float32)
_BLUE_NOISE_SIZE = 64
_BLUE_NOISE_CHUNK_ROWS = 64


def _generate_blue_noise_thresholds() -> np.ndarray:
    """64×64 보이드 앤 클러스터 임계값 행렬 생성."""
    size = _BLUE_NOISE_SIZE
    count = size * size
    yy, xx = np.mgrid[:size, :size]
    xx = np.minimum(xx, size - xx)
    yy = np.minimum(yy, size - yy)
    kernel = np.exp(-(xx * xx + yy * yy) / (2 * 1.5 ** 2)).astype(np.float32)

    rng = np.random.default_rng(731)
    binary = np.zeros(count, dtype=bool)
    binary[rng.choice(count, count // 2, replace=False)] = True
    binary = binary.reshape(size, size)
    energy = np.fft.ifft2(np.fft.fft2(binary) * np.fft.fft2(kernel)).real.astype(np.float32)

    def _update_energy(target_energy: np.ndarray, index: int, sign: int) -> None:
        """지정한 픽셀의 에너지 갱신."""
        y, x = divmod(int(index), size)
        target_energy[:] += np.roll(np.roll(kernel, y, axis=0), x, axis=1) * sign

    for _ in range(count * 2):
        cluster = int(np.argmax(np.where(binary, energy, -np.inf)))
        binary.ravel()[cluster] = False
        _update_energy(energy, cluster, -1)
        void = int(np.argmin(np.where(~binary, energy, np.inf)))
        binary.ravel()[void] = True
        _update_energy(energy, void, 1)
        if cluster == void:
            break

    rank = np.zeros((size, size), dtype=np.int32)
    mask = binary.copy()
    ranked_energy = energy.copy()
    for value in range(count // 2 - 1, -1, -1):
        index = int(np.argmax(np.where(mask, ranked_energy, -np.inf)))
        rank.ravel()[index] = value
        mask.ravel()[index] = False
        _update_energy(ranked_energy, index, -1)

    mask = binary.copy()
    ranked_energy = energy.copy()
    for value in range(count // 2, count):
        index = int(np.argmin(np.where(~mask, ranked_energy, np.inf)))
        rank.ravel()[index] = value
        mask.ravel()[index] = True
        _update_energy(ranked_energy, index, 1)

    return (rank + 0.5) / count


@lru_cache(maxsize=1)
def _blue_noise_thresholds() -> np.ndarray:
    """패키지에 저장된 64×64 블루 노이즈 임계값 행렬 로드."""
    return np.load(Path(__file__).with_name("blue_noise_thresholds.npy"), allow_pickle=False)


def apply(rgb: np.ndarray, original: np.ndarray, strength: float, bit_depth: int,
          seed: int = 0, *, document_mask=None, cel_fill_mask=None) -> tuple[np.ndarray, dict]:
    """보정량과 평탄도에 비례한 삼각 분포 노이즈를 8비트 저장 전 적용.

    문서 복원 마스크 안에서는 좁은 글자 사이도 같은 강도로 디더링한다.
    셀 채움 마스크 안에서는 블루 노이즈 임계값으로 8비트 양자화한다.
    """
    report = {
        "applied": False,
        "mean_weight": 0.0,
        "mean_flatness": 0.0,
        "weighted_fraction": 0.0,
        "seed": int(seed),
    }
    if strength <= 0 or bit_depth != 8:
        return rgb, report

    change = np.max(np.abs(rgb - original), axis=2)
    luma = rgb @ _LUMA
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    gradient = cv2.magnitude(gx, gy)
    gradient = cv2.dilate(gradient, np.ones((5, 5), np.uint8))
    flatness = np.clip(1 - gradient / 1.0, 0, 1)
    weight = np.clip(change / 0.5, 0, 1) * strength * flatness
    mean_flatness = float(np.mean(flatness))
    del change, luma, gx, gy, gradient, flatness
    if document_mask is not None:
        weight[document_mask] = strength
    report.update({
        "applied": True,
        "mean_weight": float(np.mean(weight)),
        "mean_flatness": mean_flatness,
        "weighted_fraction": float(np.mean(weight > 0)),
    })
    rng = np.random.default_rng(seed)
    noise = rng.random(rgb.shape, dtype=np.float32) + rng.random(rgb.shape, dtype=np.float32) - 1
    output = rgb + noise * weight[..., None]
    del noise, weight
    if cel_fill_mask is not None and np.any(cel_fill_mask):
        height, width = rgb.shape[:2]
        thresholds = _blue_noise_thresholds()
        x_indices = np.arange(width) % _BLUE_NOISE_SIZE
        for y_start in range(0, height, _BLUE_NOISE_CHUNK_ROWS):
            y_stop = min(y_start + _BLUE_NOISE_CHUNK_ROWS, height)
            chunk_mask = cel_fill_mask[y_start:y_stop]
            threshold_rows = thresholds[
                np.arange(y_start, y_stop)[:, None] % _BLUE_NOISE_SIZE,
                x_indices,
            ]
            selected_thresholds = threshold_rows[chunk_mask, None]
            rgb_chunk = rgb[y_start:y_stop]
            output_chunk = output[y_start:y_stop]
            quantized = np.floor(rgb_chunk[chunk_mask] + 1.0 - selected_thresholds)
            output_chunk[chunk_mask] = np.clip(quantized, 0.0, 255.0)
            del chunk_mask, threshold_rows, selected_thresholds, rgb_chunk, output_chunk, quantized
    return output.astype(np.float32, copy=False), report
