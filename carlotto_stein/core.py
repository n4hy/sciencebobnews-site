"""Vectorized Carlotto-Stein fractal artificial-object detector.

Reference
---------
M. J. Carlotto and M. C. Stein, "A Method for Searching for Artificial Objects
on Planetary Surfaces," *Journal of the British Interplanetary Society (JBIS)*,
Vol. 43, pp. 209-216, 1990.

Method (as implemented here)
----------------------------
Natural planetary terrain, imaged as an intensity field ``I(x, y)``, is well
modelled as fractional Brownian motion (fBm).  For fBm the expected absolute
intensity increment between two points separated by a lag ``r`` obeys a power
law::

    E[ |I(p) - I(p + r)| ]  ~  r ** H

where ``H`` (the Hurst exponent, 0 < H < 1) fixes the surface fractal dimension
``D = 3 - H``.  Carlotto & Stein estimate ``H`` *locally* -- inside a sliding
window around every pixel -- by measuring the increment statistic at several
lags and fitting a straight line in log-log space.  Two quantities fall out per
pixel:

* the **local fractal dimension** ``D = 3 - H``; and
* the **fit residual** -- how badly the local increments depart from a single
  power law.

Man-made objects violate the fractal model: they are locally smoother
(anomalously *low* fractal dimension) and/or they break the single-power-law
scaling (high residual).  Combining a fractal-dimension anomaly with the
residual yields a detection map whose bright regions are candidate artificial
objects.

Acceleration
------------
Every step is expressed as elementwise array math plus separable box filters:

* Increments are shifted absolute differences (edge-clamped, no wrap-around).
* The local expectation ``E[.]`` over the window is a box mean evaluated with a
  **summed-area table** (integral image), so its cost is O(pixels) and does not
  grow with the window size.
* The per-pixel log-log line fit is a closed-form least-squares reduction over
  the lag axis (the design matrix is shared by every pixel, so the normal
  equations collapse to a single ``einsum``).

Because the code only ever touches the injected array module ``xp``, passing
CuPy instead of NumPy runs the whole pipeline on a GPU with no code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# Default increment lags (pixels). Log-spaced-ish and kept modest so the
# largest shift stays small relative to typical windows.
DEFAULT_LAGS: tuple[int, ...] = (1, 2, 3, 4, 6, 8)


def _shift_clamped(img, drow: int, dcol: int, xp):
    """Return ``out[i, j] = img[i + drow, j + dcol]`` with edge clamping.

    Edge clamping (rather than wrap-around) prevents spurious huge increments
    from opposite image borders leaking into the structure function.
    """
    nrows, ncols = img.shape
    r = xp.clip(xp.arange(nrows) + drow, 0, nrows - 1)
    c = xp.clip(xp.arange(ncols) + dcol, 0, ncols - 1)
    return img[r][:, c]


def isotropic_increment(img, lag: int, xp):
    """Mean absolute intensity increment at a given ``lag``.

    Averaged over the two axis-aligned directions ``(+lag, 0)`` and
    ``(0, +lag)`` -- both at exact Euclidean distance ``lag`` -- to reduce
    directional bias while keeping the lag distance well defined.
    """
    dx = xp.abs(img - _shift_clamped(img, lag, 0, xp))
    dy = xp.abs(img - _shift_clamped(img, 0, lag, xp))
    return 0.5 * (dx + dy)


def box_mean(field, radius: int, xp):
    """Mean of ``field`` over a ``(2*radius+1)`` square window per pixel.

    Implemented with a summed-area table so runtime is independent of
    ``radius``.  ``field`` is edge-replicated by ``radius`` first, so every
    output window is full and the divisor is the constant window area (border
    pixels see a replicated -- not truncated -- neighbourhood).
    """
    if radius < 1:
        return field
    nrows, ncols = field.shape
    win = 2 * radius + 1

    # Edge-pad, then accumulate in float64 for numerical stability of the
    # running sums (cumsum of many terms loses precision badly in float32).
    padded = xp.pad(field, ((radius, radius), (radius, radius)), mode="edge")
    integ = xp.zeros((nrows + 2 * radius + 1, ncols + 2 * radius + 1), dtype=xp.float64)
    integ[1:, 1:] = xp.cumsum(xp.cumsum(padded, axis=0, dtype=xp.float64), axis=1)

    total = (
        integ[win : win + nrows, win : win + ncols]
        - integ[0:nrows, win : win + ncols]
        - integ[win : win + nrows, 0:ncols]
        + integ[0:nrows, 0:ncols]
    )
    return (total / float(win * win)).astype(field.dtype, copy=False)


def structure_function(img, lags: Sequence[int], window_radius: int, xp):
    """Local structure function ``S_r(x, y)`` for each lag in ``lags``.

    Returns an array of shape ``(len(lags), nrows, ncols)`` where entry
    ``[k]`` is the windowed mean absolute increment at ``lags[k]``.
    """
    layers = [box_mean(isotropic_increment(img, int(r), xp), window_radius, xp) for r in lags]
    return xp.stack(layers, axis=0)


@dataclass
class FractalMaps:
    """Per-pixel outputs of the Carlotto-Stein fit.

    Attributes
    ----------
    fractal_dimension:
        ``D = 3 - H`` in ``[2, 3]`` for ideal fBm terrain.
    hurst:
        The estimated Hurst exponent ``H`` (structure-function slope).
    fit_residual:
        RMS residual of the per-pixel log-log line fit (fractal-model
        goodness-of-fit; large => non-fractal / structured).
    anomaly:
        Combined detection score (see :func:`combine_anomaly`).
    lags:
        The lag values used, for reference.
    backend:
        Name of the array backend that produced the maps.
    """

    fractal_dimension: object
    hurst: object
    fit_residual: object
    anomaly: object
    lags: tuple[int, ...]
    backend: str


def fit_fractal(struct, lags: Sequence[int], xp, eps: float = 1e-6):
    """Closed-form per-pixel log-log least-squares fit of the structure function.

    Parameters
    ----------
    struct:
        ``(K, nrows, ncols)`` stack from :func:`structure_function`.
    lags:
        The ``K`` lag values (must match ``struct``).

    Returns
    -------
    (hurst, fit_residual):
        Two ``(nrows, ncols)`` arrays -- the slope ``H`` and the RMS log-log
        residual.
    """
    x = xp.log(xp.asarray(lags, dtype=xp.float64))            # (K,)
    xbar = x.mean()
    xc = x - xbar                                             # (K,)
    sxx = float((xc * xc).sum())

    y = xp.log(xp.clip(struct, eps, None)).astype(xp.float64)  # (K, R, C)
    ybar = y.mean(axis=0)                                     # (R, C)

    # slope = sum_k xc_k * y_k / Sxx  (since sum_k xc_k = 0, no ybar term needed)
    slope = xp.tensordot(xc, y, axes=([0], [0])) / sxx        # (R, C)
    intercept = ybar - slope * xbar                          # (R, C)

    pred = slope[None, :, :] * x[:, None, None] + intercept[None, :, :]
    resid = xp.sqrt(xp.mean((y - pred) ** 2, axis=0))         # (R, C)
    return slope, resid


def _robust_center_scale(a, xp):
    """Median and a robust (MAD-based) scale, ignoring non-finite values."""
    finite = a[xp.isfinite(a)]
    if finite.size == 0:
        return 0.0, 1.0
    med = float(xp.median(finite))
    mad = float(xp.median(xp.abs(finite - med)))
    scale = 1.4826 * mad
    if not scale > 0:
        scale = float(xp.std(finite)) or 1.0
    return med, scale


def combine_anomaly(hurst, fit_residual, xp, dimension_weight: float = 1.0,
                    residual_weight: float = 1.0):
    """Fuse a low-fractal-dimension cue and a high-residual cue into one score.

    Both cues are robustly standardized (median / MAD) so the score is scale
    free.  The dimension cue fires on *smoother-than-terrain* pixels (low ``D``,
    i.e. high ``H``); the residual cue fires where the single-power-law fractal
    model fails.  The result is non-negative, with larger => more artificial.
    """
    D = 3.0 - hurst
    d_med, d_scale = _robust_center_scale(D, xp)
    r_med, r_scale = _robust_center_scale(fit_residual, xp)

    # Low-dimension departure (only smoother-than-typical counts as evidence).
    dim_cue = xp.clip((d_med - D) / d_scale, 0.0, None)
    res_cue = xp.clip((fit_residual - r_med) / r_scale, 0.0, None)

    score = dimension_weight * dim_cue + residual_weight * res_cue
    return score


def carlotto_stein(img, lags: Sequence[int] = DEFAULT_LAGS, window_radius: int = 8,
                   xp=None, backend_name: str = "numpy", **anomaly_kwargs) -> FractalMaps:
    """Run the full detector on a 2-D intensity image.

    Parameters
    ----------
    img:
        2-D array on the same backend as ``xp`` (float recommended).
    lags:
        Increment lags in pixels.
    window_radius:
        Half-size of the local estimation window; the window is
        ``(2*window_radius + 1)`` square.
    xp:
        The array module (``numpy`` or ``cupy``).  Defaults to ``numpy``.
    """
    if xp is None:
        import numpy as xp  # type: ignore

    img = xp.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {tuple(img.shape)}")
    img = img.astype(xp.float32, copy=False)

    struct = structure_function(img, lags, window_radius, xp)
    hurst, resid = fit_fractal(struct, lags, xp)
    anomaly = combine_anomaly(hurst, resid, xp, **anomaly_kwargs)

    return FractalMaps(
        fractal_dimension=3.0 - hurst,
        hurst=hurst,
        fit_residual=resid,
        anomaly=anomaly,
        lags=tuple(int(r) for r in lags),
        backend=backend_name,
    )
