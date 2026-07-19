"""Hardware-accelerated Carlotto-Stein artificial-object detector.

A high-speed implementation of the fractal-model anomaly detector from
M. J. Carlotto & M. C. Stein, "A Method for Searching for Artificial Objects on
Planetary Surfaces," JBIS Vol. 43, pp. 209-216 (1990).

The same vectorized kernels run on CPU (NumPy) or GPU (CuPy); acceleration comes
from an O(pixels) summed-area-table structure-function evaluator plus a
closed-form per-pixel log-log fit, both trivially data-parallel.

Quick start
-----------
>>> from carlotto_stein import CarlottoSteinDetector, scene_with_objects
>>> image, truth = scene_with_objects()
>>> det = CarlottoSteinDetector(backend="auto")   # GPU if available
>>> result = det.detect(image)
>>> result.candidate_mask.shape == image.shape
True
"""

from .backends import Backend, get_backend, gpu_available
from .core import (
    DEFAULT_LAGS,
    FractalMaps,
    carlotto_stein,
    combine_anomaly,
    fit_fractal,
    structure_function,
)
from .detector import CarlottoSteinDetector, DetectionResult
from .synthetic import add_artificial_object, fbm_terrain, scene_with_objects

__version__ = "0.1.0"

__all__ = [
    "Backend",
    "get_backend",
    "gpu_available",
    "DEFAULT_LAGS",
    "FractalMaps",
    "carlotto_stein",
    "combine_anomaly",
    "fit_fractal",
    "structure_function",
    "CarlottoSteinDetector",
    "DetectionResult",
    "fbm_terrain",
    "add_artificial_object",
    "scene_with_objects",
    "__version__",
]
