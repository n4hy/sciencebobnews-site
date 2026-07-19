"""Command-line interface: run the detector on an image.

Examples
--------
Run on a file, write the anomaly and mask maps as PGM::

    python -m carlotto_stein scene.pgm --out-prefix result --backend auto

Run on a generated synthetic scene (no input needed)::

    python -m carlotto_stein --demo --out-prefix demo
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .detector import CarlottoSteinDetector
from .io import load_image, save_pgm
from .synthetic import scene_with_objects


def _parse_lags(text: str) -> tuple[int, ...]:
    return tuple(int(t) for t in text.replace(" ", "").split(",") if t)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="carlotto_stein",
        description="Hardware-accelerated Carlotto-Stein artificial-object detector "
                    "(JBIS 1990).",
    )
    p.add_argument("image", nargs="?", help="input image (.npy, .pgm, or any PIL format)")
    p.add_argument("--demo", action="store_true",
                   help="ignore INPUT and run on a synthetic fractal scene")
    p.add_argument("--backend", default="auto", choices=["auto", "cpu", "gpu", "numpy", "cupy"],
                   help="array backend (default: auto = GPU if available)")
    p.add_argument("--lags", type=_parse_lags, default=None,
                   help="comma-separated increment lags, e.g. 1,2,3,4,6,8")
    p.add_argument("--window", type=int, default=8, help="window half-size (radius)")
    p.add_argument("--sigma", type=float, default=4.0,
                   help="candidate threshold in robust sigmas above the median")
    p.add_argument("--out-prefix", default=None,
                   help="write <prefix>_anomaly.pgm, <prefix>_dimension.pgm, "
                        "<prefix>_mask.pgm")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.demo or not args.image:
        image, _truth = scene_with_objects()
        source = "synthetic demo scene"
    else:
        image = load_image(args.image)
        source = args.image

    kwargs = dict(backend=args.backend, window_radius=args.window, threshold_sigma=args.sigma)
    if args.lags:
        kwargs["lags"] = args.lags
    det = CarlottoSteinDetector(**kwargs)

    result = det.detect(image)

    n_hits = int(result.candidate_mask.sum())
    print(f"source          : {source}")
    print(f"image shape     : {image.shape[0]} x {image.shape[1]}")
    print(f"backend         : {result.backend}")
    print(f"lags            : {result.lags}")
    print(f"window radius   : {result.window_radius}")
    print(f"compute time    : {result.compute_seconds * 1e3:.2f} ms")
    print(f"fractal dim.    : median={np.nanmedian(result.fractal_dimension):.3f}")
    print(f"anomaly thresh. : {result.threshold:.3f}")
    print(f"candidate px    : {n_hits} ({100.0 * n_hits / image.size:.3f}% of image)")

    if args.out_prefix:
        save_pgm(f"{args.out_prefix}_anomaly.pgm", result.anomaly)
        save_pgm(f"{args.out_prefix}_dimension.pgm", result.fractal_dimension)
        save_pgm(f"{args.out_prefix}_mask.pgm", result.candidate_mask.astype(np.float32))
        print(f"wrote           : {args.out_prefix}_anomaly.pgm, "
              f"{args.out_prefix}_dimension.pgm, {args.out_prefix}_mask.pgm")

    return 0


if __name__ == "__main__":
    sys.exit(main())
