"""Image loading and writing helpers for the restoration CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import zlib

import numpy as np
from PIL import Image, ImageOps


@dataclass
class LoadedImage:
    rgb: np.ndarray
    alpha: np.ndarray | None
    size: tuple[int, int]
    mode: str
    icc_profile: bytes | None
    bit_depth: int


def _decode_png_16(path: Path) -> tuple[np.ndarray, np.ndarray | None] | None:
    """Decode 16-bit PNG samples without Pillow's RGB precision loss."""
    with path.open("rb") as handle:
        header = handle.read(33)
        if len(header) < 33 or not header.startswith(b"\x89PNG\r\n\x1a\n") or header[12:16] != b"IHDR":
            return None
        width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack_from(">IIBBBBB", header, 16)
        if bit_depth != 16:
            return None
        data = header + handle.read()

    if compression != 0 or filtering != 0:
        return None

    compressed = bytearray()
    transparent = None
    offset = 8
    while offset + 12 <= len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4:offset + 8]
        payload_start = offset + 8
        payload_end = payload_start + length
        if payload_end + 4 > len(data):
            return None
        payload = data[payload_start:payload_end]
        if kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"tRNS":
            transparent = payload
        elif kind == b"IEND":
            break
        offset = payload_end + 4

    if bit_depth != 16 or color_type not in (0, 2, 4, 6):
        return None

    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    pixels = np.zeros((height, width, channels), dtype=np.uint16)
    passes = (
        ((0, 0, 1, 1),)
        if interlace == 0
        else (
            (0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
            (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2),
        )
    )
    stream = zlib.decompress(bytes(compressed))
    cursor = 0

    for x0, y0, dx, dy in passes:
        pass_width = max(0, (width - x0 + dx - 1) // dx)
        pass_height = max(0, (height - y0 + dy - 1) // dy)
        if pass_width == 0 or pass_height == 0:
            continue
        row_bytes = pass_width * channels * 2
        bytes_per_pixel = channels * 2
        previous = np.zeros(row_bytes, dtype=np.uint8)
        pass_rows = np.empty((pass_height, pass_width, channels), dtype=np.uint16)

        for row_index in range(pass_height):
            filter_type = stream[cursor]
            cursor += 1
            raw = np.frombuffer(stream, dtype=np.uint8, count=row_bytes, offset=cursor)
            cursor += row_bytes
            row = raw.copy()

            if filter_type == 1:
                for lane in range(bytes_per_pixel):
                    values = raw[lane::bytes_per_pixel].astype(np.uint32)
                    row[lane::bytes_per_pixel] = (np.cumsum(values, dtype=np.uint32) & 255).astype(np.uint8)
            elif filter_type == 2:
                row = ((raw.astype(np.uint16) + previous.astype(np.uint16)) & 255).astype(np.uint8)
            elif filter_type in (3, 4):
                for start in range(0, row_bytes, bytes_per_pixel):
                    end = start + bytes_per_pixel
                    left = row[start - bytes_per_pixel:start].astype(np.int16) if start else np.zeros(bytes_per_pixel, dtype=np.int16)
                    up = previous[start:end].astype(np.int16)
                    if filter_type == 3:
                        predictor = (left + up) // 2
                    else:
                        upper_left = previous[start - bytes_per_pixel:start].astype(np.int16) if start else np.zeros(bytes_per_pixel, dtype=np.int16)
                        base = left + up - upper_left
                        distance_left = np.abs(base - left)
                        distance_up = np.abs(base - up)
                        distance_upper_left = np.abs(base - upper_left)
                        predictor = np.where(
                            (distance_left <= distance_up) & (distance_left <= distance_upper_left),
                            left,
                            np.where(distance_up <= distance_upper_left, up, upper_left),
                        )
                    row[start:end] = ((raw[start:end].astype(np.int16) + predictor) & 255).astype(np.uint8)
            elif filter_type != 0:
                return None

            samples = np.frombuffer(row.tobytes(), dtype=">u2").astype(np.uint16)
            pass_rows[row_index] = samples.reshape(pass_width, channels)
            previous = row

        pixels[y0:y0 + dy * pass_height:dy, x0:x0 + dx * pass_width:dx] = pass_rows

    if channels in (1, 2):
        gray = pixels[:, :, 0]
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        if channels == 2:
            alpha = pixels[:, :, 1]
        elif transparent is not None:
            transparent_gray = struct.unpack(">H", transparent)[0]
            alpha = np.where(gray == transparent_gray, 0, 65535).astype(np.uint16)
        else:
            alpha = None
    else:
        rgb = pixels[:, :, :3]
        if channels == 4:
            alpha = pixels[:, :, 3]
        elif transparent is not None:
            transparent_rgb = np.frombuffer(transparent, dtype=">u2").astype(np.uint16)
            matches = np.all(rgb == transparent_rgb, axis=2)
            alpha = np.where(matches, 0, 65535).astype(np.uint16)
        else:
            alpha = None
    return rgb, alpha


def _transpose_exif(array: np.ndarray, orientation: int) -> np.ndarray:
    """Apply an EXIF orientation to a decoded array."""
    if orientation == 2:
        return np.flip(array, axis=1)
    if orientation == 3:
        return np.rot90(array, 2, axes=(0, 1))
    if orientation == 4:
        return np.flip(array, axis=0)
    if orientation == 5:
        return np.swapaxes(array, 0, 1)
    if orientation == 6:
        return np.rot90(array, 3, axes=(0, 1))
    if orientation == 7:
        return np.flip(np.swapaxes(array, 0, 1), axis=(0, 1))
    if orientation == 8:
        return np.rot90(array, 1, axes=(0, 1))
    return array


def load_image(path: str | Path) -> LoadedImage:
    """Load RGB pixels, an optional alpha plane, and source color metadata."""
    path = Path(path)
    with Image.open(path) as source_image:
        source_image.load()
        original_mode = source_image.mode
        orientation = source_image.getexif().get(274, 1)
        image = ImageOps.exif_transpose(source_image)
        size = image.size
        mode = original_mode
        grayscale_modes = {"1", "L", "LA", "I", "I;16", "I;16B", "I;16L"}
        icc_profile = None if mode in grayscale_modes else image.info.get("icc_profile")
        has_alpha = "A" in image.getbands() or "transparency" in image.info
        decoded_16 = _decode_png_16(path) if path.suffix.lower() == ".png" else None

        if decoded_16 is not None:
            rgb16, alpha16 = decoded_16
            rgb16 = _transpose_exif(rgb16, orientation)
            alpha16 = _transpose_exif(alpha16, orientation) if alpha16 is not None else None
            rgb = rgb16.astype(np.float32) / 257.0
            alpha = alpha16.astype(np.float32) / 257.0 if alpha16 is not None else None
            bit_depth = 16
        elif image.mode.startswith("I;16") or image.mode == "I":
            gray = np.asarray(image, dtype=np.int64)
            gray = np.clip(gray, 0, 65535).astype(np.uint16)
            rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32) / 257.0
            alpha = None
            bit_depth = 16
        else:
            if has_alpha:
                converted = np.asarray(image.convert("RGBA"), dtype=np.uint8)
                rgb = converted[:, :, :3].astype(np.float32)
                alpha = converted[:, :, 3].astype(np.float32)
            else:
                converted = np.asarray(image.convert("RGB"), dtype=np.uint8)
                rgb = converted.astype(np.float32)
                alpha = None
            bit_depth = 8

    return LoadedImage(
        rgb=np.ascontiguousarray(rgb, dtype=np.float32),
        alpha=np.ascontiguousarray(alpha, dtype=np.float32) if alpha is not None else None,
        size=size,
        mode=mode,
        icc_profile=icc_profile,
        bit_depth=bit_depth,
    )


def _png_chunk(handle, kind: bytes, payload: bytes) -> None:
    """Write one PNG chunk and its CRC."""
    handle.write(struct.pack(">I", len(payload)))
    handle.write(kind)
    handle.write(payload)
    handle.write(struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


def _save_png_16(path: Path, rgb: np.ndarray, alpha: np.ndarray | None, icc_profile: bytes | None) -> None:
    """Write RGB or RGBA samples as a 16-bit PNG."""
    channels = 4 if alpha is not None else 3
    clipped_rgb = np.rint(np.clip(rgb, 0.0, 255.0) * 257.0).astype(np.uint16)
    if alpha is not None:
        clipped_alpha = np.rint(np.clip(alpha, 0.0, 255.0) * 257.0).astype(np.uint16)
        pixels = np.concatenate((clipped_rgb, clipped_alpha[:, :, None]), axis=2)
    else:
        pixels = clipped_rgb

    height, width = pixels.shape[:2]
    raw = b"".join(b"\x00" + row.astype(">u2", copy=False).tobytes() for row in pixels)
    compressed = zlib.compress(raw)
    with path.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        _png_chunk(handle, b"IHDR", struct.pack(">IIBBBBB", width, height, 16, 6 if channels == 4 else 2, 0, 0, 0))
        if icc_profile:
            profile = b"ICC Profile\x00\x00" + zlib.compress(icc_profile)
            _png_chunk(handle, b"iCCP", profile)
        _png_chunk(handle, b"IDAT", compressed)
        _png_chunk(handle, b"IEND", b"")


def save_image(
    path: str | Path,
    rgb: np.ndarray,
    alpha: np.ndarray | None = None,
    *,
    image_format: str = "png",
    quality: int = 90,
    bit_depth: int = 8,
    icc_profile: bytes | None = None,
) -> None:
    """Save pixels in the requested format and bit depth."""
    path = Path(path)
    image_format = image_format.lower()
    if image_format == "jpeg":
        image_format = "jpg"
    if bit_depth == 16:
        if image_format != "png":
            raise ValueError("16-bit output is only supported for PNG.")
        _save_png_16(path, rgb, alpha, icc_profile)
        return

    clipped = np.rint(np.clip(rgb, 0.0, 255.0)).astype(np.uint8)
    if alpha is not None and image_format == "jpg" and not np.all(alpha == 255.0):
        raise ValueError("JPEG cannot store transparency. Choose PNG or WebP.")
    if alpha is not None and image_format != "jpg":
        clipped_alpha = np.rint(np.clip(alpha, 0.0, 255.0)).astype(np.uint8)
        pixels = np.concatenate((clipped, clipped_alpha[:, :, None]), axis=2)
    else:
        pixels = clipped

    image = Image.fromarray(pixels)
    options = {"icc_profile": icc_profile} if icc_profile else {}
    if image_format == "jpg":
        image.save(path, format="JPEG", quality=quality, **options)
    elif image_format == "webp":
        image.save(path, format="WEBP", quality=quality, **options)
    elif image_format == "png":
        image.save(path, format="PNG", **options)
    else:
        raise ValueError(f"Unsupported output format: {image_format}")
