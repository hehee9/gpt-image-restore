"""@description 중앙값 필터의 전체 너비 행 띠 병렬 계산."""

import os
import sys

import cv2
import numpy as np


_SERIAL = sys.platform == "emscripten" or os.environ.get("GPTIMG_RESTORE_THREADS") == "1"
_WORKERS = 1 if _SERIAL else min(4, os.cpu_count() or 1)
if _SERIAL:
    cv2.setNumThreads(1)


class _SerialExecutor:
    """스레드 없이 같은 순서로 계산."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def map(self, fn, values):
        return map(fn, values)


def _executor(max_workers):
    """실행 환경에 맞는 계산기 생성."""
    if _SERIAL:
        return _SerialExecutor()
    from concurrent.futures import ThreadPoolExecutor
    return ThreadPoolExecutor(max_workers=max_workers)


_POOL = _executor(_WORKERS) if _WORKERS > 1 else None


def _median_blur(source: np.ndarray, ksize: int) -> np.ndarray:
    """중앙값 필터의 반경을 포함한 행 띠 계산."""
    if _WORKERS <= 1 or source.size < 262144 or source.shape[0] < 128:
        return cv2.medianBlur(source, ksize)

    height = source.shape[0]
    radius = ksize // 2
    output = np.empty_like(source)
    bounds = np.linspace(0, height, _WORKERS + 1, dtype=int)

    def _compute(index: int) -> None:
        """실제 이웃 행을 포함한 띠 계산과 유효 행 저장."""
        top, bottom = bounds[index:index + 2]
        start, stop = max(0, top - radius), min(height, bottom + radius)
        filtered = cv2.medianBlur(source[start:stop], ksize)
        output[top:bottom] = filtered[top - start:bottom - start]

    list(_POOL.map(_compute, range(_WORKERS)))
    return output
