"""Safe image loading/validation shared by data preparation and inference."""
from __future__ import annotations

import io
import os
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF"}
DEFAULT_MAX_BYTES = 10 * 1024 * 1024      # 10 MB upload limit in the app
DEFAULT_MAX_PIXELS = 40_000_000           # decompression-bomb guard
DEFAULT_MIN_SIDE = 32


class InvalidImageError(ValueError):
    """Raised with a user-readable message when an image cannot be used."""


def load_image(source, *, max_bytes: int | None = DEFAULT_MAX_BYTES, max_pixels: int = DEFAULT_MAX_PIXELS,
               min_side: int = DEFAULT_MIN_SIDE, allowed_formats=ALLOWED_FORMATS) -> Image.Image:
    """Open ``source`` (path, bytes or file-like), validate it fully, and return an RGB image.

    Validation: size limit, decodable by Pillow, allowed format, pixel-count limit, minimum
    side length. Animated GIFs use their first frame. EXIF orientation is applied.
    """
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if not path.is_file():
            raise InvalidImageError(f"Image file not found: {path.name}")
        if max_bytes is not None and path.stat().st_size > max_bytes:
            raise InvalidImageError(f"Image is larger than {max_bytes // (1024 * 1024)} MB.")
        data = path.read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif hasattr(source, "read"):
        data = source.read()
    else:
        raise InvalidImageError(f"Unsupported image input type: {type(source).__name__}")
    if not data:
        raise InvalidImageError("Image file is empty.")
    if max_bytes is not None and len(data) > max_bytes:
        raise InvalidImageError(f"Image is larger than {max_bytes // (1024 * 1024)} MB.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            w, h = probe.size
            probe.verify()                  # detects truncated / corrupt files
        if fmt not in allowed_formats:
            raise InvalidImageError(f"Unsupported image format {fmt}. Use JPG or PNG.")
        if w * h > max_pixels:
            raise InvalidImageError(f"Image is too large ({w}x{h} pixels).")
        if min(w, h) < min_side:
            raise InvalidImageError(f"Image is too small ({w}x{h}); minimum side is {min_side}px.")
        img = Image.open(io.BytesIO(data))  # verify() invalidates the object: reopen
        img.seek(0)
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.load()
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as e:
        raise InvalidImageError(f"The file could not be read as an image ({type(e).__name__}).") from e
    return img
