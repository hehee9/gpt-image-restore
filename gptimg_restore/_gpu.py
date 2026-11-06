"""명시적으로 선택한 NVIDIA GPU의 밀집 연산 블록."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path

import cv2
import numpy as np


class GPUUnavailableError(RuntimeError):
    """선택한 GPU 실행 환경의 초기화 실패."""


class GPUBackend:
    """블록 내부 배열을 장치에 유지하는 선택형 CuPy 실행기."""

    def __init__(self):
        self._dll_handles = []
        self._configure_cuda_wheels()
        try:
            import cupy as cp
            from cupy.cuda.compiler import CompileException
            from cupyx.scipy import ndimage
        except (ImportError, OSError) as error:
            raise GPUUnavailableError(
                "Could not load CuPy/CUDA required for GPU execution. "
                "Install the gpu extra and NVIDIA drivers. "
                f"Cause: {error}"
            ) from error
        self.cp = cp
        self.ndimage = ndimage
        try:
            cp.cuda.Device(0).use()
            cp.empty(1)
            cp.cuda.Stream.null.synchronize()
            # 실제 사용하는 JIT와 FFT 라이브러리의 초기화 확인.
            self._gaussian(cp.zeros((3, 3), cp.float32), 1.0)
            cp.fft.rfft2(cp.zeros((4, 4), cp.float32))
            cp.cuda.Stream.null.synchronize()
        except (cp.cuda.runtime.CUDARuntimeError, CompileException, OSError, RuntimeError) as error:
            raise GPUUnavailableError(
                f"Could not initialize the GPU runtime. Check the CUDA 12 components and NVIDIA drivers. Cause: {error}"
            ) from error

    def _configure_cuda_wheels(self):
        """설치된 CUDA 12 휠의 헤더와 라이브러리를 프로세스 안에서 연결."""
        for distribution, folder in (
            ("nvidia-cuda-runtime-cu12", "cuda_runtime"),
            ("nvidia-cuda-nvrtc-cu12", "cuda_nvrtc"),
            ("nvidia-cufft-cu12", "cufft"),
            ("nvidia-nvjitlink-cu12", "nvjitlink"),
        ):
            try:
                package = importlib.metadata.distribution(distribution)
            except importlib.metadata.PackageNotFoundError:
                continue
            root = Path(package.locate_file(f"nvidia/{folder}"))
            if folder == "cuda_runtime":
                # 전역 CUDA_PATH의 다른 주버전 헤더와 혼합 방지.
                os.environ["CUDA_PATH"] = str(root)
            if os.name == "nt":
                binary = root / "bin"
                self._dll_handles.append(os.add_dll_directory(str(binary)))
                os.environ["PATH"] = str(binary) + os.pathsep + os.environ.get("PATH", "")

    def _gaussian(self, values, sigma):
        """OpenCV Gaussian 계수와 REFLECT_101 경계 적용."""
        size = int(round(sigma * 8 + 1)) | 1
        weights = self.cp.asarray(cv2.getGaussianKernel(size, sigma, cv2.CV_32F).ravel())
        return self.ndimage.correlate1d(
            self.ndimage.correlate1d(values, weights, axis=1, mode="mirror"),
            weights, axis=0, mode="mirror")

    def _sobel(self, values, axis):
        """OpenCV와 같은 계수, 경계, 축 배율의 Sobel 계산."""
        cp = self.cp
        derivative = cp.asarray([-1, 0, 1], dtype=cp.float32)
        smooth = cp.asarray([1, 2, 1], dtype=cp.float32)
        wx, wy = (derivative, smooth) if axis == 1 else (smooth, derivative)
        return self.ndimage.correlate1d(
            self.ndimage.correlate1d(values, wx, axis=1, mode="mirror"),
            wy, axis=0, mode="mirror") * cp.float32(0.125)

    def structure_mask(self, rgb, protected):
        """CPU의 경계 보호 판정을 유지하며 구조 마스크와 에너지 반환."""
        cp = self.cp
        values = cp.asarray(rgb)
        protected_device = cp.asarray(protected)
        smooth = self._gaussian(values, 1.3)
        gx, gy = self._sobel(smooth, 1), self._sobel(smooth, 0)
        xx = self._gaussian(cp.mean(gx * gx, axis=2), 1.5)
        yy = self._gaussian(cp.mean(gy * gy, axis=2), 1.5)
        xy = self._gaussian(cp.mean(gx * gy, axis=2), 1.5)
        energy = cp.sqrt(xx + yy)
        coherence = cp.sqrt((xx - yy) ** 2 + 4 * xy * xy) / (xx + yy + 1e-6)
        structure = cp.maximum(cp.clip((energy - 1.0) / 5, 0, 1),
                               coherence * cp.clip((energy - 0.6) / 2.4, 0, 1))
        edge_soft = self._gaussian(protected_device.astype(cp.float32), 1.0)
        mask = (1 - structure) * (1 - edge_soft)
        mask[protected_device | cp.all(values >= 250, axis=2)] = 0
        result = cp.asnumpy(cp.stack((mask, energy), axis=-1))
        return result[..., 0].copy(), result[..., 1].copy()

    def fft_band(self, residual, radial, diagonal):
        """최대 128개 타일 FFT와 네 겹침 묶음으로 보정장 합성."""
        cp = self.cp
        height, width = residual.shape[:2]
        padded = cp.pad(cp.asarray(residual),
                        ((48, 48 + (-height) % 48), (48, 48 + (-width) % 48), (0, 0)),
                        mode="reflect")
        tiles = cp.lib.stride_tricks.sliding_window_view(padded, (96, 96), axis=(0, 1))[::48, ::48]
        window = cp.asarray(np.outer(np.hanning(96), np.hanning(96)).astype(np.float32))
        radial, diagonal = cp.asarray(radial), cp.asarray(diagonal)
        correction = cp.zeros_like(padded)
        mass = cp.zeros(padded.shape[:2], cp.float32)
        for top in range(0, tiles.shape[0], 4):
            for left in range(0, tiles.shape[1], 32):
                batch = tiles[top:top + 4, left:left + 32]
                spectrum = cp.fft.rfft2(batch * window, axes=(-2, -1))
                power = cp.mean(cp.abs(spectrum) ** 2, axis=2)
                ratio = (cp.sum(power * radial * diagonal, axis=(-2, -1))
                         / (cp.sum(power * radial, axis=(-2, -1)) + 1e-8))
                gain = cp.clip((ratio - 0.25) / 0.35, 0, 1)
                attenuation = radial * (0.7 + (0.3 * gain)[..., None, None] * diagonal)
                filtered = cp.fft.irfft2(spectrum * attenuation[:, :, None],
                                        s=(96, 96), axes=(-2, -1)) * window
                for parity_y in (0, 1):
                    for parity_x in (0, 1):
                        group = filtered[parity_y::2, parity_x::2]
                        rows, cols = group.shape[:2]
                        if rows == 0 or cols == 0:
                            continue
                        panel = group.transpose(0, 3, 1, 4, 2).reshape(rows * 96, cols * 96, 3)
                        y, x = (top + parity_y) * 48, (left + parity_x) * 48
                        correction[y:y + rows * 96, x:x + cols * 96] += panel
                        mass[y:y + rows * 96, x:x + cols * 96] += cp.tile(window * window, (rows, cols))
        return cp.asnumpy(correction[48:height + 48, 48:width + 48]
                          / mass[48:height + 48, 48:width + 48, None])
