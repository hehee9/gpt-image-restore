"""Build the GitHub Pages before/after showcase.

Usage:
    python tools/build_showcase.py
    python tools/build_showcase.py --config path/to/config.json --docs path/to/docs

For every entry in ``items`` of the config (default: tools/showcase_config.json),
the script restores the input image in-process with the repository's restore
pipeline (presets ``balanced`` and ``strong``) and rebuilds:

- docs/media/<id>/       full-resolution lossless WebP original, balanced and strong images
- docs/previews/<id>.png README preview (normal and contrast-boosted comparisons)
- docs/showcase.json     manifest the page renders from
- docs/readme-preview.md Markdown snippet linking each preview to its page anchor

Item fields:
    id       page anchor; lowercase letters, digits and hyphens (e.g. "handbag")
    title    Korean section title
    input    image path, absolute or relative to the config file. Inputs may
             live outside the repository; only the derived images in docs/ are
             published.
    view     optional [x, y, width, height] rectangle in source pixels that the
             page shows initially
    crops    optional list of [x, y, width, height] rectangles in source pixels,
             used only for the README preview. When omitted, the view rectangle
             is used, or the first automatic crop if view is also omitted.

Files listed in the previous showcase.json that the new build no longer
writes are deleted, so removing an item from the config and rebuilding also
removes its assets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gptimg_restore import cli  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "tools" / "showcase_config.json"
DEFAULT_DOCS = REPO_ROOT / "docs"
MANIFEST_NAME = "showcase.json"
SNIPPET_NAME = "readme-preview.md"
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
AUTO_CROP_SIDE = 192
AUTO_CROP_COUNT = 2
MIN_CROP_SIDE = 16
PREVIEW_TARGET = 384
PREVIEW_GAP = 4

# docs/site.js의 대비 강조 설정과 동기화합니다.
BOOST_MAX_GAIN = 4
BOOST_MIN_GAIN = 0.3
BOOST_SMOOTH_SIGMA = 3
BOOST_TAIL_PERCENT = 0.2
BOOST_BANDS = {"cel-illustration": [(160, 178, 4)]}




# =================================== Helpers/Utilities ===================================


def _load_config(path: Path) -> dict:
    """Load the config and validate every item at the input boundary."""
    config = json.loads(path.read_text(encoding="utf-8"))
    seen = set()
    for item in config["items"]:
        item_id = item["id"]
        if not ID_PATTERN.match(item_id):
            raise ValueError(f"id may contain only lowercase letters, digits, and hyphens: {item_id!r}")
        if item_id in seen:
            raise ValueError(f"duplicate id: {item_id}")
        seen.add(item_id)
        if "preset" in item:
            raise ValueError(f"{item_id}: the preset option is no longer supported")
        if not item["title"].strip():
            raise ValueError(f"{item_id}: title is empty")
        source = Path(item["input"])
        item["input"] = source if source.is_absolute() else (path.parent / source).resolve()
        if not item["input"].is_file():
            raise ValueError(f"{item_id}: input image not found: {item['input']}")
    return config


def _restore(source: Path, preset: str, workdir: Path) -> Path:
    """Run the restore pipeline in-process and return the restored PNG path."""
    if cli.main([str(source), "--preset", preset, "-o", str(workdir)]) != 0:
        raise RuntimeError(f"restore failed: {source}")
    return workdir / f"{source.stem}_restored.png"


def _load_pixels(path: Path, mode: str) -> np.ndarray:
    """Load an image as a uint8 array in the given mode."""
    with Image.open(path) as image:
        return np.asarray(image.convert(mode))


def _auto_crops(before: np.ndarray, after: np.ndarray) -> list[list[int]]:
    """Pick the non-overlapping windows with the largest mean absolute change."""
    height, width = before.shape[:2]
    side = min(AUTO_CROP_SIDE, width, height)
    stride = max(side // 2, 1)
    change = np.abs(before[..., :3].astype(np.int16) - after[..., :3].astype(np.int16)).sum(axis=2)
    integral = np.pad(change.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    windows = []
    for y in range(0, height - side + 1, stride):
        for x in range(0, width - side + 1, stride):
            total = (integral[y + side, x + side] - integral[y, x + side]
                     - integral[y + side, x] + integral[y, x])
            windows.append((-total, y, x))
    windows.sort()
    picked: list[list[int]] = []
    for _, y, x in windows:
        if all(abs(x - px) >= side or abs(y - py) >= side for px, py, _, _ in picked):
            picked.append([x, y, side, side])
            if len(picked) == AUTO_CROP_COUNT:
                break
    return picked


def _validate_crop(item_id: str, crop: list[int], width: int, height: int) -> None:
    """Reject crop rectangles outside the image or too small to inspect."""
    x, y, w, h = crop
    if w < MIN_CROP_SIDE or h < MIN_CROP_SIDE or x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(f"{item_id}: invalid crop rectangle {crop} (image {width}x{height})")


def _save_webp(image: Image.Image, path: Path) -> None:
    """Save a lossless WebP that keeps exact pixel values."""
    image.save(path, "WEBP", lossless=True, quality=100, method=4, exact=True)


def _boost_values(pixels: np.ndarray) -> np.ndarray:
    """캔버스의 알파 처리 후 RGB 최댓값을 구합니다."""
    rgb = pixels[..., :3]
    if pixels.shape[2] == 4:
        alpha = pixels[..., 3:4].astype(np.float64)
        premultiplied = np.floor(rgb.astype(np.float64) * alpha / 255 + 0.5)
        rgb = np.zeros_like(premultiplied)
        np.divide(premultiplied * 255, alpha, out=rgb, where=alpha != 0)
        rgb = np.floor(rgb + 0.5).clip(0, 255).astype(np.uint8)
    return rgb.max(axis=2)


def _boost_curve(values: np.ndarray, item_id: str) -> np.ndarray:
    """원본 전체의 밝기 분포로 페이지와 같은 강조 곡선을 구합니다."""
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    tail = values.size * BOOST_TAIL_PERCENT / 100
    low = int(np.searchsorted(np.cumsum(histogram), tail, side="right"))
    high = 255 - int(np.searchsorted(np.cumsum(histogram[::-1]), tail, side="right"))
    radius = int(np.ceil(BOOST_SMOOTH_SIGMA * 3))
    density = np.zeros(256)
    for offset in range(-radius, radius + 1):
        source = np.clip(np.arange(256) + offset, 0, 255)
        density += histogram[source] * np.exp(-offset * offset / (2 * BOOST_SMOOTH_SIGMA ** 2))
    floors = np.full(256, BOOST_MIN_GAIN)
    for start, end, gain in BOOST_BANDS.get(item_id, []):
        floors[start:end + 1] = np.maximum(floors[start:end + 1], gain)

    def fill_slopes(scale: float) -> np.ndarray:
        slopes = np.zeros(256)
        slopes[low:high + 1] = np.minimum(
            BOOST_MAX_GAIN, np.maximum(floors[low:high + 1], scale * density[low:high + 1])
        )
        return slopes

    lower, upper = 0.0, 1.0
    while fill_slopes(upper).sum() < 255 and upper < 1e12:
        upper *= 2
    for _ in range(60):
        middle = (lower + upper) / 2
        if fill_slopes(middle).sum() < 255:
            lower = middle
        else:
            upper = middle
    slopes = fill_slopes(upper)
    shift = max(0, (255 - slopes.sum()) / 2)
    cumulative = np.cumsum(slopes) - slopes
    return np.clip(np.floor(shift + cumulative + slopes / 2 + 0.5), 0, 255).astype(np.uint8)


def _preview(before: np.ndarray, after: np.ndarray, boosted_before: np.ndarray,
             boosted_after: np.ndarray) -> Image.Image:
    """기본 비교와 대비 강조 비교를 최근접 확대하여 위아래로 배치합니다."""
    height, width = before.shape[:2]
    scale = max(1, round(PREVIEW_TARGET / max(width, height)))
    tile_width, tile_height = width * scale, height * scale
    canvas = Image.new("RGB", (tile_width * 2 + PREVIEW_GAP, tile_height * 2 + PREVIEW_GAP), (255, 255, 255))
    for index, array in enumerate((before, after, boosted_before, boosted_after)):
        patch = Image.fromarray(array).convert("RGB")
        canvas.paste(
            patch.resize((tile_width, tile_height), Image.NEAREST),
            ((index % 2) * (tile_width + PREVIEW_GAP), (index // 2) * (tile_height + PREVIEW_GAP)),
        )
    return canvas


def _remove_stale(docs: Path, previous: list[str], current: set[str]) -> None:
    """Delete files from the previous build that this build no longer writes."""
    for relative in previous:
        if relative in current:
            continue
        path = docs / relative
        path.unlink(missing_ok=True)
        parent = path.parent
        while parent != docs and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent




# =================================== Main Logic ===================================


def build_item(item: dict, docs: Path, workdir: Path) -> tuple[dict, list[str]]:
    """Restore one item, write its images, and return its manifest entry and file list."""
    item_id = item["id"]
    with Image.open(item["input"]) as probe:
        mode = "RGBA" if "A" in probe.getbands() or "transparency" in probe.info else "RGB"
    before = _load_pixels(item["input"], mode)
    balanced_path = _restore(item["input"], "balanced", workdir)
    balanced = _load_pixels(balanced_path, mode)
    strong_path = _restore(item["input"], "strong", workdir)
    strong = _load_pixels(strong_path, mode)
    height, width = before.shape[:2]
    crops = item.get("crops")
    view = item.get("view")
    if view:
        _validate_crop(item_id, view, width, height)
    if crops:
        for crop in crops:
            _validate_crop(item_id, crop, width, height)
        preview_crop = crops[0]
    elif view:
        preview_crop = view
    else:
        crops = _auto_crops(before, strong)
        for crop in crops:
            _validate_crop(item_id, crop, width, height)
        preview_crop = crops[0]

    (docs / "media" / item_id).mkdir(parents=True, exist_ok=True)
    (docs / "previews").mkdir(exist_ok=True)
    files = []

    images = {}
    for label, array in (("original", before), ("balanced", balanced), ("strong", strong)):
        relative = f"media/{item_id}/{label}.webp"
        _save_webp(Image.fromarray(array), docs / relative)
        images[label] = relative
        files.append(relative)

    x, y, w, h = preview_crop
    preview_relative = f"previews/{item_id}.png"
    before_values = _boost_values(before)
    curve = _boost_curve(before_values, item_id)
    boosted_before = curve[before_values[y:y + h, x:x + w]]
    boosted_after = curve[_boost_values(strong[y:y + h, x:x + w])]
    _preview(before[y:y + h, x:x + w], strong[y:y + h, x:x + w], boosted_before, boosted_after).save(
        docs / preview_relative, optimize=True
    )
    files.append(preview_relative)

    entry = {
        "id": item_id,
        "title": item["title"],
        "width": width,
        "height": height,
        "images": images,
        "preview": preview_relative,
    }
    if view:
        entry["view"] = view
    return entry, files


def write_snippet(entries: list[dict], site_url: str, docs: Path) -> None:
    """Write the README Markdown snippet linking each preview to its page anchor."""
    lines = ["<!-- tools/build_showcase.py가 생성합니다. README.md에 붙여 넣어 사용합니다. -->", ""]
    for entry in entries:
        alt = f"{entry['title']}: 원본과 보정 결과 비교"
        lines.append(f"[![{alt}](docs/{entry['preview']})]({site_url}#{entry['id']})")
    (docs / SNIPPET_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Rebuild every showcase asset from the config."""
    parser = argparse.ArgumentParser(description="Build the GitHub Pages before/after showcase.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="config file path")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS, help="output docs directory")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = _load_config(config_path)
    docs = args.docs.resolve()
    manifest_path = docs / MANIFEST_NAME
    previous = []
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]

    entries, files = [], []
    with tempfile.TemporaryDirectory(prefix="showcase-") as workdir:
        for item in config["items"]:
            print(f"Processing: {item['id']}", flush=True)
            entry, item_files = build_item(item, docs, Path(workdir))
            entries.append(entry)
            files.extend(item_files)

    manifest = {"items": entries, "files": files}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_snippet(entries, config["site_url"], docs)
    _remove_stale(docs, previous, set(files))
    print(f"Done: {len(entries)} items -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
