"""End-to-end demo: generate a fractal scene, detect the planted objects.

Writes PGM previews next to this script and prints a small detection report.
No GPU required -- runs on the CPU backend and uses the GPU automatically if
CuPy is installed.

    python examples/synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

# Allow `python examples/synthetic_demo.py` from the repo root without install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from carlotto_stein import CarlottoSteinDetector
from carlotto_stein.io import save_pgm
from carlotto_stein.synthetic import scene_with_objects


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    image, truth = scene_with_objects((512, 512), hurst=0.7, seed=42)

    det = CarlottoSteinDetector(backend="auto", window_radius=12, threshold_sigma=4.0)
    res = det.detect(image)

    hits = res.candidate_mask & truth
    detected = res.candidate_mask.sum()
    precision = hits.sum() / max(1, detected)
    recall = hits.sum() / max(1, truth.sum())

    print(f"backend         : {res.backend}")
    print(f"compute time    : {res.compute_seconds * 1e3:.2f} ms")
    print(f"terrain D (med) : {np.median(res.fractal_dimension):.3f}")
    print(f"planted objects : {int(truth.sum())} px")
    print(f"flagged         : {int(detected)} px")
    print(f"precision/recall: {precision:.2f} / {recall:.2f}")

    save_pgm(os.path.join(out_dir, "demo_input.pgm"), image)
    save_pgm(os.path.join(out_dir, "demo_dimension.pgm"), res.fractal_dimension)
    save_pgm(os.path.join(out_dir, "demo_anomaly.pgm"), res.anomaly)
    save_pgm(os.path.join(out_dir, "demo_mask.pgm"), res.candidate_mask.astype(np.float32))
    print("wrote demo_input/dimension/anomaly/mask .pgm to", out_dir)


if __name__ == "__main__":
    main()
