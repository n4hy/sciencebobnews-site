"""Synthetic fractal terrain with embedded artificial objects.

Used by the demo, the benchmark, and the tests.  Terrain is generated as
fractional Brownian motion via spectral synthesis (a ``1/f**beta`` power
spectrum), which gives a controllable Hurst exponent -- the ground truth the
detector is supposed to recover.  "Artificial" objects are geometrically smooth
patches whose local fractal dimension is far lower than the surrounding terrain.
"""

from __future__ import annotations

import numpy as np


def fbm_terrain(shape=(512, 512), hurst: float = 0.7, seed: int = 0) -> np.ndarray:
    """Generate fBm terrain with the given Hurst exponent ``H``.

    The 2-D fBm power spectrum scales as ``k ** -(2H + 2)``; synthesizing in the
    Fourier domain with that envelope and random phase yields a field whose
    structure-function slope is ``H``.
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape
    ky = np.fft.fftfreq(rows)[:, None]
    kx = np.fft.fftfreq(cols)[None, :]
    k = np.sqrt(ky * ky + kx * kx)
    k[0, 0] = 1.0                                   # avoid division by zero at DC

    beta = 2.0 * hurst + 2.0
    amplitude = k ** (-beta / 2.0)
    amplitude[0, 0] = 0.0                           # zero mean

    phase = rng.uniform(0.0, 2.0 * np.pi, size=shape)
    spectrum = amplitude * np.exp(1j * phase)
    field = np.fft.ifft2(spectrum).real

    field -= field.min()
    if field.max() > 0:
        field /= field.max()
    return field.astype(np.float32)


def add_artificial_object(terrain: np.ndarray, center, size, kind: str = "rect",
                          contrast: float = 0.35) -> np.ndarray:
    """Stamp a smooth (low fractal-dimension) patch onto ``terrain``.

    Returns a copy; ``kind`` is ``"rect"`` or ``"disk"``.  The patch replaces
    the fractal terrain with a smooth ramp, mimicking a flat/faceted man-made
    surface that breaks the local scaling law.
    """
    out = terrain.copy()
    rows, cols = out.shape
    cy, cx = center
    if kind == "rect":
        hy, hx = size if isinstance(size, tuple) else (size, size)
        y0, y1 = max(0, cy - hy), min(rows, cy + hy)
        x0, x1 = max(0, cx - hx), min(cols, cx + hx)
        base = float(out[y0:y1, x0:x1].mean())
        yy = np.linspace(-1, 1, y1 - y0)[:, None]
        ramp = base + contrast * 0.5 * yy            # gentle, smooth gradient
        out[y0:y1, x0:x1] = ramp
    elif kind == "disk":
        radius = size if not isinstance(size, tuple) else size[0]
        yy, xx = np.ogrid[:rows, :cols]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
        base = float(out[mask].mean()) if mask.any() else 0.0
        out[mask] = base + contrast * 0.5
    else:
        raise ValueError(f"unknown object kind: {kind!r}")
    return out.astype(np.float32)


def scene_with_objects(shape=(512, 512), hurst: float = 0.7, seed: int = 0):
    """Return ``(image, truth_mask)`` -- fBm terrain with a few planted objects.

    ``truth_mask`` marks the planted object pixels, for scoring detections.
    """
    terrain = fbm_terrain(shape, hurst=hurst, seed=seed)
    rows, cols = shape
    img = terrain
    truth = np.zeros(shape, dtype=bool)

    objects = [
        (int(rows * 0.30), int(cols * 0.30), "rect", (18, 26)),
        (int(rows * 0.68), int(cols * 0.62), "disk", 22),
        (int(rows * 0.50), int(cols * 0.82), "rect", (12, 12)),
    ]
    for cy, cx, kind, size in objects:
        img = add_artificial_object(img, (cy, cx), size, kind=kind)
        if kind == "rect":
            hy, hx = size
            truth[max(0, cy - hy):cy + hy, max(0, cx - hx):cx + hx] = True
        else:
            yy, xx = np.ogrid[:rows, :cols]
            truth[(yy - cy) ** 2 + (xx - cx) ** 2 <= size * size] = True

    return img.astype(np.float32), truth
