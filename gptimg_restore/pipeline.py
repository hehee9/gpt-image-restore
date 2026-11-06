"""@description 이미지 복원 단계와 프리셋 공개 API."""

from __future__ import annotations

import importlib
import math
import time

import numpy as np

PRESETS = {
    "safe": {"dots": 0.0, "sparkle": 0.0, "periodic": 1.0, "grime": 0.5, "mottle": 0.35, "lattice": 0.0, "surface": 0.0, "contour": 0.0, "defocus": 0.0, "cel_fill": 0.0, "dither": 0.0},
    "balanced": {"dots": 1.0, "sparkle": 0.0, "periodic": 1.0, "grime": 1.0, "mottle": 0.7, "lattice": 0.0, "surface": 0.0, "contour": 0.0, "defocus": 0.0, "cel_fill": 0.0, "dither": 0.0},
    "strong": {"dots": 1.0, "sparkle": 1.0, "periodic": 1.0, "grime": 1.1, "mottle": 1.0, "lattice": 1.0, "surface": 0.0, "contour": 1.0, "defocus": 1.0, "cel_fill": 1.0, "dither": 1.0},
}
STAGE_ORDER = ("dots", "sparkle", "periodic", "grime", "mottle", "lattice", "surface", "contour", "defocus")
STAGES = STAGE_ORDER + ("document_background", "sparkle_final", "cel_fill", "dither")


class StageUnavailableError(RuntimeError):
    """활성화된 단계 모듈 로드 실패."""


def _load_stage(name: str):
    """단계 모듈 로드 및 실패 원인 전달."""
    try:
        return importlib.import_module(f".{name}", package=__package__)
    except ImportError as error:
        raise StageUnavailableError(f"Could not load enabled {name} stage module: {error}") from error


def restore_array(image, preset="balanced", overrides=None, device="cpu", progress=None, **options):
    """로드된 이미지를 복원하고 RGB 배열, 단계 보고서, 단계별 시간을 반환.

    image는 io_utils.load_image의 LoadedImage다. rgb는 (높이, 너비, 3)의
    float32 RGB 배열이며 값 범위는 0~255다. 16비트 입력도 257로 나눈
    동일 범위의 float32로 표현한다. alpha는 별도 float32 배열 또는 None이며
    복원 과정에서 변경하지 않는다. 저장 시 원본 alpha와 ICC 프로필을 전달한다.
    overrides에는 단계 이름과 강도, dots_mode를 지정할 수 있다.
    options의 bit_depth는 출력 비트 깊이(기본 8), backend는 재사용할 GPU다.
    progress는 각 단계 경계에서 (완료 비율, 단계 이름)을 받는다.
    """
    from . import dither, textgap

    bit_depth = options.pop("bit_depth", 8)
    backend = options.pop("backend", None)
    if options:
        raise TypeError(f"Unsupported options: {', '.join(options)}")
    if device not in ("cpu", "gpu"):
        raise ValueError(f"Unsupported device: {device}")
    if bit_depth not in (8, 16):
        raise ValueError(f"Unsupported bit depth: {bit_depth}")
    if device == "gpu" and backend is None:
        from ._gpu import GPUBackend
        backend = GPUBackend()
    strengths = dict(PRESETS[preset])
    overrides = dict(overrides or {})
    dots_mode = overrides.pop("dots_mode", "conservative")
    if dots_mode not in ("conservative", "strict"):
        raise ValueError(f"Unsupported dot detection mode: {dots_mode}")
    unknown = overrides.keys() - strengths.keys()
    if unknown:
        raise ValueError(f"Unsupported stages: {', '.join(sorted(unknown))}")
    strengths.update({name: value for name, value in overrides.items() if value is not None})
    dither_strength = strengths["dither"]
    if dither_strength < 0 or not math.isfinite(dither_strength):
        raise ValueError(f"dither stage strength must be finite and 0 or greater: {dither_strength}")
    cel_fill_strength = strengths["cel_fill"]
    if cel_fill_strength < 0 or not math.isfinite(cel_fill_strength):
        raise ValueError(f"cel_fill stage strength must be finite and 0 or greater: {cel_fill_strength}")
    if image.bit_depth == 8:
        rgb = image.rgb
        original = image.rgb.astype(np.uint8)
    else:
        original = image.rgb.copy()
        rgb = original.copy()
    document_mask = np.zeros(rgb.shape[:2], bool) if strengths["grime"] >= 1.1 else None
    stage_reports = {}
    stage_timings = {}
    for name in STAGE_ORDER:
        strength = strengths[name]
        if strength < 0 or not math.isfinite(strength):
            raise ValueError(f"{name} stage strength must be finite and 0 or greater: {strength}")
        if strength == 0:
            stage_reports[name] = {"enabled": False, "strength": strength}
            stage_timings[name] = 0.0
            if progress is not None:
                progress((STAGE_ORDER.index(name) + 1) / 13, name)
            continue
        module = _load_stage(name)
        stage_started = time.perf_counter()
        if name == "dots":
            rgb, stage_reports[name] = module.apply(rgb, strength=strength, mode=dots_mode)
        elif name in ("periodic", "grime", "mottle", "lattice") and backend is not None:
            rgb, stage_reports[name] = module.apply(rgb, strength=strength, backend=backend)
        elif name == "defocus" and strengths["contour"] > 0:
            # 윤곽 단계 무변경 시 같은 입력의 선화 판정 재사용.
            contour_report = stage_reports["contour"]
            context = ((contour_report["protected_line_art"], contour_report["flat_region_fraction"])
                       if contour_report["max_abs_change"] == 0.0 else None)
            rgb, stage_reports[name] = module.apply(rgb, strength=strength, line_art_context=context, backend=backend)
        elif name == "defocus" and backend is not None:
            rgb, stage_reports[name] = module.apply(rgb, strength=strength, backend=backend)
        else:
            rgb, stage_reports[name] = module.apply(rgb, strength=strength)
        stage_timings[name] = time.perf_counter() - stage_started
        if progress is not None:
            progress((STAGE_ORDER.index(name) + 1) / 13, name)

    if document_mask is not None:
        stage_started = time.perf_counter()
        rgb, stage_reports["document_background"] = textgap.restore_document_background(
            rgb, strength=strengths["grime"], document_mask=document_mask)
        stage_timings["document_background"] = time.perf_counter() - stage_started

    if progress is not None:
        progress(10 / 13, "document_background")

    if strengths["sparkle"] > 0:
        stage_started = time.perf_counter()
        rgb, final_sparkle = _load_stage("sparkle").apply_final(
            rgb, strength=strengths["sparkle"], bit_depth=bit_depth)
        stage_reports["sparkle"]["final_pass"] = final_sparkle
        stage_timings["sparkle"] += time.perf_counter() - stage_started

    if progress is not None:
        progress(11 / 13, "sparkle_final")

    cel_fill_mask = None
    if cel_fill_strength == 0:
        stage_reports["cel_fill"] = {"enabled": False, "strength": cel_fill_strength}
        stage_timings["cel_fill"] = 0.0
    else:
        if image.bit_depth == 8:
            original = original.astype(np.float32)
        cel_fill_mask = np.zeros(rgb.shape[:2], dtype=bool)
        stage_started = time.perf_counter()
        rgb, stage_reports["cel_fill"] = _load_stage("celfill").apply(
            rgb, strength=cel_fill_strength, context_rgb=original,
            corrected_mask=cel_fill_mask)
        stage_timings["cel_fill"] = time.perf_counter() - stage_started

    if progress is not None:
        progress(12 / 13, "cel_fill")

    if dither_strength > 0 and bit_depth == 8 and original.dtype == np.uint8:
        original = original.astype(np.float32)
    stage_started = time.perf_counter()
    rgb, stage_reports["dither"] = dither.apply(
        rgb, original, strength=dither_strength, bit_depth=bit_depth, seed=0,
        document_mask=document_mask, cel_fill_mask=cel_fill_mask)
    stage_timings["dither"] = time.perf_counter() - stage_started

    if progress is not None:
        progress(1.0, "dither")
    return rgb, stage_reports, stage_timings
