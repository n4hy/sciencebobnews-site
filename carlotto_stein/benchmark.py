"""Benchmark and, when a GPU is present, report CPU->GPU speedup.

Run::

    python -m carlotto_stein.benchmark --size 1024 --repeat 5

Times the *core* computation only (device-synchronized), for CPU and -- if
CuPy + a device are available -- GPU, and prints the speedup.  Also verifies the
two backends agree to floating-point tolerance so the acceleration is known to
be correct, not just fast.
"""

from __future__ import annotations

import argparse

import numpy as np

from .backends import get_backend, gpu_available
from .core import DEFAULT_LAGS, carlotto_stein
from .synthetic import fbm_terrain


def _time_backend(prefer: str, image: np.ndarray, lags, window, repeat: int):
    be = get_backend(prefer)
    dev = be.asarray(image)
    # Warm-up (JIT / allocator / autotuning) is excluded from the measurement.
    maps, _ = be.time_call(carlotto_stein, dev, lags=lags, window_radius=window,
                           xp=be.xp, backend_name=be.name)
    best = float("inf")
    for _ in range(repeat):
        _, sec = be.time_call(carlotto_stein, dev, lags=lags, window_radius=window,
                              xp=be.xp, backend_name=be.name)
        best = min(best, sec)
    return be, maps, best


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Carlotto-Stein CPU/GPU benchmark")
    p.add_argument("--size", type=int, default=1024, help="square image side (pixels)")
    p.add_argument("--window", type=int, default=8, help="window radius")
    p.add_argument("--repeat", type=int, default=5, help="timed repetitions (best is kept)")
    p.add_argument("--hurst", type=float, default=0.7)
    args = p.parse_args(argv)

    image = fbm_terrain((args.size, args.size), hurst=args.hurst, seed=1)
    lags = DEFAULT_LAGS

    print(f"image           : {args.size} x {args.size}  ({image.size:,} px)")
    print(f"lags            : {lags}   window radius: {args.window}")

    _, cpu_maps, cpu_t = _time_backend("cpu", image, lags, args.window, args.repeat)
    cpu_px = image.size / cpu_t
    print(f"CPU (numpy)     : {cpu_t * 1e3:8.2f} ms   ({cpu_px / 1e6:7.1f} Mpx/s)")

    if gpu_available():
        be, gpu_maps, gpu_t = _time_backend("gpu", image, lags, args.window, args.repeat)
        gpu_px = image.size / gpu_t
        print(f"GPU (cupy)      : {gpu_t * 1e3:8.2f} ms   ({gpu_px / 1e6:7.1f} Mpx/s)")
        print(f"speedup         : {cpu_t / gpu_t:6.1f}x")

        # Correctness cross-check.
        a = cpu_maps.anomaly
        b = be.to_numpy(gpu_maps.anomaly)
        max_abs = float(np.max(np.abs(a - b)))
        print(f"CPU/GPU max|Δ|  : {max_abs:.3e} (anomaly map)")
    else:
        print("GPU (cupy)      : unavailable -- install cupy on a CUDA/ROCm host "
              "for the accelerated path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
