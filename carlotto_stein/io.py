"""Tiny dependency-light image I/O helpers.

Supports ``.npy`` natively and greyscale PGM (binary P5 / ASCII P2) so the
package needs nothing beyond NumPy.  If Pillow happens to be installed, any
format it understands is accepted too.
"""

from __future__ import annotations

import numpy as np


def _load_pgm(path: str) -> np.ndarray:
    with open(path, "rb") as fh:
        data = fh.read()

    def tokens(buf: bytes):
        i, n = 0, len(buf)
        while i < n:
            while i < n and buf[i : i + 1].isspace():
                i += 1
            if i < n and buf[i : i + 1] == b"#":            # comment to end of line
                while i < n and buf[i : i + 1] not in (b"\n", b"\r"):
                    i += 1
                continue
            start = i
            while i < n and not buf[i : i + 1].isspace():
                i += 1
            yield buf[start:i]

    it = tokens(data)
    magic = next(it)
    if magic not in (b"P5", b"P2"):
        raise ValueError(f"not a PGM file: {path}")
    width = int(next(it))
    height = int(next(it))
    maxval = int(next(it))

    if magic == b"P2":
        vals = [int(next(it)) for _ in range(width * height)]
        arr = np.asarray(vals, dtype=np.float32).reshape(height, width)
    else:
        # One whitespace byte separates the header from the binary raster.
        header_end = data.index(str(maxval).encode(), 0) + len(str(maxval))
        raster = data[header_end + 1 :]
        dtype = np.uint8 if maxval < 256 else ">u2"
        arr = np.frombuffer(raster, dtype=dtype, count=width * height)
        arr = arr.astype(np.float32).reshape(height, width)
    return arr


def load_image(path: str) -> np.ndarray:
    """Load ``path`` as a 2-D float32 greyscale array."""
    lower = path.lower()
    if lower.endswith(".npy"):
        arr = np.load(path)
    elif lower.endswith((".pgm",)):
        arr = _load_pgm(path)
    else:
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional dep
            raise ValueError(
                f"unsupported image format for {path!r}; install Pillow or use "
                ".npy/.pgm"
            ) from exc
        arr = np.asarray(Image.open(path).convert("F"))

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:                       # collapse colour to luminance
        arr = arr[..., :3].mean(axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {arr.shape}")
    return arr


def save_pgm(path: str, arr: np.ndarray) -> None:
    """Write a 2-D array to an 8-bit binary PGM (min-max normalized)."""
    a = np.asarray(arr, dtype=np.float64)
    finite = a[np.isfinite(a)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 1.0
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    img = (norm * 255.0 + 0.5).astype(np.uint8)
    h, w = img.shape
    with open(path, "wb") as fh:
        fh.write(f"P5\n{w} {h}\n255\n".encode())
        fh.write(img.tobytes())
