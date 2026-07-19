"""Array-backend selection for CPU / GPU execution.

The Carlotto-Stein detector is written once against a NumPy-compatible array
module (``xp``).  Because CuPy exposes the same API as NumPy, the *identical*
kernels in :mod:`carlotto_stein.core` run unchanged on an NVIDIA/ROCm GPU when
CuPy is installed and a device is available.  This module is the single place
that decides which module to hand back and how to move data on/off the device.

Preference values accepted by :func:`get_backend`:

``"auto"``   use the GPU (CuPy) if importable *and* a device is present, else CPU.
``"gpu"``    force CuPy; raise if it is not usable.
``"cpu"``    force NumPy.
``"cupy"``   alias of ``"gpu"``.
``"numpy"``  alias of ``"cpu"``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


def _try_import_cupy():
    """Return an *usable* cupy module or ``None``.

    "Usable" means importable **and** with at least one visible device, so that
    an accidental CPU-only CuPy install (or a machine with no GPU) transparently
    falls back to NumPy instead of raising deep inside a kernel launch.
    """
    try:
        import cupy  # type: ignore
    except Exception:
        return None
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return None
    except Exception:
        return None
    return cupy


@dataclass
class Backend:
    """A thin handle over the chosen array module.

    Attributes
    ----------
    xp:
        The array module (``numpy`` or ``cupy``) -- pass this to the kernels.
    name:
        ``"numpy"`` or ``"cupy"``.
    on_gpu:
        ``True`` when kernels execute on an accelerator.
    """

    xp: Any
    name: str
    on_gpu: bool

    def asarray(self, a, dtype=None):
        """Move a host array onto the backend device."""
        return self.xp.asarray(a, dtype=dtype)

    def to_numpy(self, a):
        """Bring a backend array back to a host ``numpy.ndarray``."""
        if self.on_gpu:
            return self.xp.asnumpy(a)
        import numpy as _np

        return _np.asarray(a)

    def synchronize(self):
        """Block until queued device work has finished (no-op on CPU).

        Required for honest timing: GPU kernel launches are asynchronous, so a
        benchmark that does not synchronize measures launch latency, not compute.
        """
        if self.on_gpu:
            self.xp.cuda.runtime.deviceSynchronize()

    def time_call(self, fn, *args, **kwargs):
        """Run ``fn`` once, returning ``(result, wall_seconds)`` with a
        device sync so GPU timings are real."""
        self.synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        self.synchronize()
        return out, time.perf_counter() - t0


def get_backend(prefer: str = "auto") -> Backend:
    """Resolve ``prefer`` to a concrete :class:`Backend`."""
    prefer = (prefer or "auto").lower()

    if prefer in ("cpu", "numpy"):
        import numpy as np

        return Backend(xp=np, name="numpy", on_gpu=False)

    if prefer in ("gpu", "cupy"):
        cupy = _try_import_cupy()
        if cupy is None:
            raise RuntimeError(
                "GPU backend requested but CuPy is unavailable or no CUDA/ROCm "
                "device was found. Install a matching cupy wheel (e.g. "
                "`pip install cupy-cuda12x`) or use prefer='cpu'."
            )
        return Backend(xp=cupy, name="cupy", on_gpu=True)

    if prefer == "auto":
        cupy = _try_import_cupy()
        if cupy is not None:
            return Backend(xp=cupy, name="cupy", on_gpu=True)
        import numpy as np

        return Backend(xp=np, name="numpy", on_gpu=False)

    raise ValueError(f"unknown backend preference: {prefer!r}")


def gpu_available() -> bool:
    """``True`` if an accelerated backend can be constructed."""
    return _try_import_cupy() is not None
