"""High-level, backend-aware entry point for the Carlotto-Stein detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import core
from .backends import Backend, get_backend


@dataclass
class DetectionResult:
    """Detector output, always returned as host (NumPy) arrays.

    Attributes
    ----------
    fractal_dimension, hurst, fit_residual, anomaly:
        ``(rows, cols)`` NumPy maps (see :class:`carlotto_stein.core.FractalMaps`).
    candidate_mask:
        Boolean map of pixels whose anomaly score exceeds ``threshold``.
    threshold:
        The scalar threshold used to build ``candidate_mask``.
    backend:
        ``"numpy"`` or ``"cupy"``.
    compute_seconds:
        Device-synchronized wall time of the core computation only.
    lags, window_radius:
        Parameters used.
    """

    fractal_dimension: object
    hurst: object
    fit_residual: object
    anomaly: object
    candidate_mask: object
    threshold: float
    backend: str
    compute_seconds: float
    lags: tuple[int, ...]
    window_radius: int


class CarlottoSteinDetector:
    """Configurable detector that runs on CPU or GPU.

    Parameters
    ----------
    lags:
        Increment lags (pixels) for the structure function.
    window_radius:
        Half-size of the local fractal-estimation window.
    backend:
        ``"auto"`` (GPU if available else CPU), ``"cpu"``/``"numpy"``, or
        ``"gpu"``/``"cupy"``.
    threshold_sigma:
        Candidate mask cutoff, expressed in robust standard deviations of the
        anomaly score above its median.
    """

    def __init__(self, lags: Sequence[int] = core.DEFAULT_LAGS, window_radius: int = 8,
                 backend: str = "auto", threshold_sigma: float = 4.0,
                 dimension_weight: float = 2.0, residual_weight: float = 1.0):
        self.lags = tuple(int(r) for r in lags)
        self.window_radius = int(window_radius)
        self.threshold_sigma = float(threshold_sigma)
        self.dimension_weight = float(dimension_weight)
        self.residual_weight = float(residual_weight)
        self._backend: Backend = get_backend(backend)

    @property
    def backend(self) -> Backend:
        return self._backend

    def detect(self, image) -> DetectionResult:
        """Detect candidate artificial objects in a 2-D intensity ``image``."""
        be = self._backend
        xp = be.xp

        dev_img = be.asarray(image)

        maps, seconds = be.time_call(
            core.carlotto_stein,
            dev_img,
            lags=self.lags,
            window_radius=self.window_radius,
            xp=xp,
            backend_name=be.name,
            dimension_weight=self.dimension_weight,
            residual_weight=self.residual_weight,
        )

        # Threshold on the device, then bring everything back to the host.
        anomaly = maps.anomaly
        med, scale = core._robust_center_scale(anomaly, xp)
        threshold = med + self.threshold_sigma * scale
        mask = anomaly > threshold

        return DetectionResult(
            fractal_dimension=be.to_numpy(maps.fractal_dimension),
            hurst=be.to_numpy(maps.hurst),
            fit_residual=be.to_numpy(maps.fit_residual),
            anomaly=be.to_numpy(anomaly),
            candidate_mask=be.to_numpy(mask),
            threshold=float(threshold),
            backend=be.name,
            compute_seconds=float(seconds),
            lags=self.lags,
            window_radius=self.window_radius,
        )
