"""Detect and remove compact bright-dot artifacts."""

from __future__ import annotations

import cv2
import numpy as np

from ._native import load_native


_native = load_native("_native_dots")


_MODES = {
    "conservative": {
        "residual_min_dn": 12.0,
        "contrast_min_dn": 12.0,
        "ring_mad_max_dn": 2.0,
        "texture_max_dn": 2.0,
        "core_area_min_px": 1,
        "core_area_max_px": 5,
        "core_bbox_max_px": 3,
        "edge_distance_min_px": 5.0,
        "channel_increment_min_dn": 6.0,
    },
    "strict": {
        "residual_min_dn": 8.0,
        "contrast_min_dn": 8.0,
        "ring_mad_max_dn": None,
        "texture_max_dn": 5.0,
        "core_area_min_px": 1,
        "core_area_max_px": 40,
        "core_bbox_max_px": 9,
        "edge_distance_min_px": 4.0,
        "channel_increment_min_dn": None,
    },
}

_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_RING_MASK = np.fromfunction(
    lambda y, x: (np.hypot(x - 9, y - 9) >= 6) & (np.hypot(x - 9, y - 9) <= 9),
    (19, 19),
    dtype=int,
)
_RING_OFFSET_Y, _RING_OFFSET_X = np.nonzero(_RING_MASK)
_RING_BATCH_SIZE = 512
_MAD_DTYPE = (np.float32(1) * 1.4826).dtype


def _ring_pixels(image: np.ndarray, peak_ys: np.ndarray, peak_xs: np.ndarray) -> np.ndarray:
    """고정 링 픽셀을 후보 순서대로 수집."""
    ring_ys = peak_ys[:, None] - 9 + _RING_OFFSET_Y
    ring_xs = peak_xs[:, None] - 9 + _RING_OFFSET_X
    return image[ring_ys, ring_xs]


def _resolve_mode(options: dict) -> tuple[str, dict]:
    """Select one documented parameter set and reject unknown options."""
    options = dict(options)
    mode = options.pop("mode", "conservative")
    if options:
        name = next(iter(options))
        raise TypeError(f"unexpected option: {name}")
    if mode not in _MODES:
        raise ValueError("mode must be 'conservative' or 'strict'")
    return mode, _MODES[mode]


def _parameters(mode: str, thresholds: dict) -> dict:
    """Return a JSON-ready description of the effective detector settings."""
    return {
        **thresholds,
        "primary_residual_min_dn": 4.0,
        "primary_sigma_multiplier": 6.0,
        "primary_component_area_px": [1, 80],
        "primary_component_bbox_max_px": 15,
        "excluded_border_px": 10,
        "median_kernel_px": {"local": 3, "background": 11, "noise": 31},
        "ring_radius_px": [6, 9],
        "half_height_fraction": 0.5,
        "half_height_min_dn": 4.0,
        "mode": mode,
    }


def _background_texture(
    luma: np.ndarray,
    peak_y: int,
    peak_x: int,
    core_y: np.ndarray,
    core_x: np.ndarray,
    ring_background: float,
) -> float:
    """Measure original-context texture after removing only this candidate core."""
    height, width = luma.shape
    patch_y0 = max(peak_y - 20, 0)
    patch_y1 = min(peak_y + 21, height)
    patch_x0 = max(peak_x - 20, 0)
    patch_x1 = min(peak_x + 21, width)
    patch = luma[patch_y0:patch_y1, patch_x0:patch_x1]
    top = max(20 - peak_y, 0)
    bottom = max(peak_y + 21 - height, 0)
    left = max(20 - peak_x, 0)
    right = max(peak_x + 21 - width, 0)
    if top or bottom or left or right:
        patch = cv2.copyMakeBorder(
            patch,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_REFLECT_101 | cv2.BORDER_ISOLATED,
        )
    patch = patch.copy()
    patch[11 + core_y, 11 + core_x] = ring_background
    smooth_context = cv2.GaussianBlur(patch, (0, 0), 2.0)
    return float(np.std(smooth_context[10:31, 10:31].astype(np.float64)))


def _empty_report(mode: str, parameters: dict) -> dict:
    """Build a compact report when detection was not run."""
    return {
        "stage": "dots",
        "mode": mode,
        "candidate_count": None,
        "masked_pixels": None,
        "mask_fraction": None,
        "candidates_per_megapixel": None,
        "median_contrast_dn": None,
        "maximum_contrast_dn": None,
        "parameters": parameters,
    }


def _detect(
    rgb: np.ndarray,
    mode: str,
    thresholds: dict,
    *,
    collect_targets: bool,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Build qualifying half-height cores, averaged RGB targets, and statistics."""
    height, width = rgb.shape[:2]
    y = rgb @ _LUMA_WEIGHTS
    u = np.rint(y).astype(np.uint8)
    median3 = cv2.medianBlur(u, 3)
    median11 = cv2.medianBlur(u, 11)
    absolute_residual = y - median3
    np.abs(absolute_residual, out=absolute_residual)
    np.rint(absolute_residual, out=absolute_residual)
    del median3, u
    absolute_residual = absolute_residual.astype(np.uint8)
    sigma = cv2.medianBlur(absolute_residual, 31).astype(np.float32) * 1.4826
    del absolute_residual
    residual = y - median11
    sigma *= 6.0
    np.maximum(sigma, 4.0, out=sigma)
    primary = residual >= sigma
    del sigma
    primary[:10, :] = False
    primary[-10:, :] = False
    primary[:, :10] = False
    primary[:, -10:] = False

    edges = cv2.Canny(median11, 20, 40)
    del median11
    edge_distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    del edges

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        primary.view(np.uint8), connectivity=8
    )
    del primary

    core_mask = np.zeros((height, width), dtype=bool)
    target_sum = None
    target_count = None
    contrasts = []

    if _native is not None:
        candidates = _native._component_candidates(labels, stats, residual)
    else:
        candidates = []
        for component_id in range(1, component_count):
            x0, y0, box_width, box_height, area = stats[component_id]
            if area < 1 or area > 80 or box_width > 15 or box_height > 15:
                continue

            component = labels[y0:y0 + box_height, x0:x0 + box_width] == component_id
            component_residual = np.where(
                component,
                residual[y0:y0 + box_height, x0:x0 + box_width],
                -np.inf,
            )
            peak_offset = int(np.argmax(component_residual))
            peak_y = y0 + peak_offset // box_width
            peak_x = x0 + peak_offset % box_width

            candidates.append((peak_y, peak_x, float(residual[peak_y, peak_x])))

    del labels, stats, residual

    rgb_candidates = []
    for start in range(0, len(candidates), _RING_BATCH_SIZE):
        batch = candidates[start:start + _RING_BATCH_SIZE]
        peak_ys = np.fromiter((candidate[0] for candidate in batch), dtype=np.intp)
        peak_xs = np.fromiter((candidate[1] for candidate in batch), dtype=np.intp)
        ring_luma = _ring_pixels(y, peak_ys, peak_xs)
        ring_backgrounds = np.median(ring_luma, axis=1)
        ring_mads = np.median(
            np.abs(ring_luma - ring_backgrounds[:, None]), axis=1
        ).astype(_MAD_DTYPE, copy=False)
        ring_mads = 1.4826 * ring_mads

        if _native is not None:
            rgb_candidates.extend(_native._validate_luma(
                batch, y, ring_backgrounds, ring_mads, edge_distance,
                thresholds, _background_texture,
            ))
            del ring_luma, ring_backgrounds, ring_mads, peak_ys, peak_xs
            continue

        for candidate_index, (peak_y, peak_x, peak_residual) in enumerate(batch):
            ring_background = float(ring_backgrounds[candidate_index])
            ring_mad = float(ring_mads[candidate_index])
            patch_y = slice(peak_y - 9, peak_y + 10)
            patch_x = slice(peak_x - 9, peak_x + 10)
            patch_luma = y[patch_y, patch_x]
            peak_contrast = float(y[peak_y, peak_x] - ring_background)

            if peak_residual < thresholds["residual_min_dn"]:
                continue
            if peak_contrast < max(thresholds["contrast_min_dn"], 6.0 * ring_mad):
                continue
            if (
                thresholds["ring_mad_max_dn"] is not None
                and ring_mad > thresholds["ring_mad_max_dn"]
            ):
                continue
            if edge_distance[peak_y, peak_x] < thresholds["edge_distance_min_px"]:
                continue

            half_height = (patch_luma - ring_background) >= max(4.0, 0.5 * peak_contrast)
            core_pixels = half_height.astype(np.uint8)
            if core_pixels[9, 9] == 0:
                continue
            core_area, _, _, core_rect = cv2.floodFill(
                core_pixels, None, (9, 9), 2, flags=8
            )
            core_x, core_y, core_width, core_height = core_rect
            if not thresholds["core_area_min_px"] <= core_area <= thresholds["core_area_max_px"]:
                continue
            if (
                core_width > thresholds["core_bbox_max_px"]
                or core_height > thresholds["core_bbox_max_px"]
            ):
                continue

            local_core = (
                core_pixels[core_y:core_y + core_height, core_x:core_x + core_width] == 2
            )
            core_local_y, core_local_x = np.nonzero(local_core)
            if _background_texture(
                y, peak_y, peak_x, core_local_y + core_y, core_local_x + core_x,
                ring_background,
            ) > thresholds["texture_max_dn"]:
                continue

            core_ys = peak_y - 9 + core_y + core_local_y
            core_xs = peak_x - 9 + core_x + core_local_x
            rgb_candidates.append((peak_y, peak_x, core_ys, core_xs, peak_contrast))

        del ring_luma, ring_backgrounds, ring_mads, peak_ys, peak_xs

    del candidates, y, edge_distance

    for start in range(0, len(rgb_candidates), _RING_BATCH_SIZE):
        batch = rgb_candidates[start:start + _RING_BATCH_SIZE]
        peak_ys = np.fromiter((candidate[0] for candidate in batch), dtype=np.intp)
        peak_xs = np.fromiter((candidate[1] for candidate in batch), dtype=np.intp)
        ring_rgb = _ring_pixels(rgb, peak_ys, peak_xs)
        clean_rgbs = np.median(ring_rgb, axis=1).astype(np.float32)

        if _native is not None:
            target_sum, target_count = _native._accumulate_rgb(
                batch, rgb, clean_rgbs, core_mask.view(np.uint8), target_sum,
                target_count, contrasts, thresholds["channel_increment_min_dn"],
                collect_targets,
            )
            del ring_rgb, clean_rgbs, peak_ys, peak_xs
            continue

        for candidate_index, (peak_y, peak_x, core_ys, core_xs, peak_contrast) in enumerate(batch):
            clean_rgb = clean_rgbs[candidate_index]
            if thresholds["channel_increment_min_dn"] is not None:
                channel_increment = rgb[peak_y, peak_x] - clean_rgb
                if np.any(channel_increment < thresholds["channel_increment_min_dn"]):
                    continue

            core_mask[core_ys, core_xs] = True
            if collect_targets:
                if target_sum is None:
                    target_sum = np.zeros((height, width, 3), dtype=np.float32)
                    target_count = np.zeros((height, width), dtype=np.uint16)
                target_sum[core_ys, core_xs] += clean_rgb
                target_count[core_ys, core_xs] += 1
            contrasts.append(peak_contrast)

        del ring_rgb, clean_rgbs, peak_ys, peak_xs

    masked_pixels = int(np.count_nonzero(core_mask))
    usable_area = max(height - 20, 0) * max(width - 20, 0)
    report = {
        "stage": "dots",
        "mode": mode,
        "candidate_count": int(len(contrasts)),
        "masked_pixels": masked_pixels,
        "mask_fraction": float(masked_pixels / (height * width)) if height * width else 0.0,
        "candidates_per_megapixel": (
            float(len(contrasts) * 1_000_000 / usable_area) if usable_area else 0.0
        ),
        "median_contrast_dn": float(np.median(contrasts)) if contrasts else None,
        "maximum_contrast_dn": float(np.max(contrasts)) if contrasts else None,
        "parameters": _parameters(mode, thresholds),
    }
    targets = None
    if collect_targets and target_sum is not None:
        targets = target_sum
        if masked_pixels:
            targets[core_mask] = target_sum[core_mask] / target_count[core_mask, None]
        del target_count
    return core_mask, targets, report


def analyze(rgb: np.ndarray, **options) -> dict:
    """Analyze compact bright-dot candidates without modifying the image."""
    mode, thresholds = _resolve_mode(options)
    _, _, report = _detect(rgb, mode, thresholds, collect_targets=False)
    return report


def apply(rgb: np.ndarray, strength: float = 1.0, **options) -> tuple[np.ndarray, dict]:
    """Replace qualifying dot cores with their local RGB ring medians."""
    mode, thresholds = _resolve_mode(options)
    if strength == 0:
        output = rgb.copy()
        report = _empty_report(mode, _parameters(mode, thresholds))
        report.update({
            "detection_measured": False,
            "strength": float(strength),
            "changed_pixels": 0,
            "correction_rms_dn": 0.0,
        })
        return output, report

    mask, targets, report = _detect(rgb, mode, thresholds, collect_targets=True)
    output = rgb.copy()
    if np.any(mask):
        original = rgb[mask]
        correction = np.float32(strength) * (targets[mask] - original)
        del targets
        output[mask] = original + correction
        changed_pixels = int(np.count_nonzero(np.any(output != rgb, axis=2)))
        correction_rms = float(np.sqrt(np.mean(correction * correction)))
    else:
        changed_pixels = 0
        correction_rms = 0.0
    report.update({
        "detection_measured": True,
        "strength": float(strength),
        "changed_pixels": changed_pixels,
        "correction_rms_dn": correction_rms,
    })
    return output.astype(np.float32, copy=False), report
