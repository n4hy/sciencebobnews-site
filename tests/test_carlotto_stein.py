"""Correctness tests for the Carlotto-Stein detector.

These run on the CPU (NumPy) backend, which is always available.  Where a GPU is
present, the backend-agreement test also exercises the accelerated path and
asserts CPU/GPU parity.
"""

from __future__ import annotations

import numpy as np
import pytest

from carlotto_stein import (
    CarlottoSteinDetector,
    carlotto_stein,
    fbm_terrain,
    gpu_available,
    scene_with_objects,
)
from carlotto_stein.backends import get_backend
from carlotto_stein.core import box_mean, isotropic_increment, structure_function


def test_box_mean_matches_naive():
    rng = np.random.default_rng(0)
    field = rng.standard_normal((37, 41)).astype(np.float32)
    radius = 3
    got = box_mean(field, radius, np)

    padded = np.pad(field, radius, mode="edge")
    win = 2 * radius + 1
    naive = np.empty_like(field)
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            naive[i, j] = padded[i : i + win, j : j + win].mean()

    assert np.allclose(got, naive, atol=1e-4)


def test_box_mean_constant_field_is_identity():
    field = np.full((20, 25), 7.0, dtype=np.float32)
    assert np.allclose(box_mean(field, 5, np), 7.0, atol=1e-5)


def test_increment_zero_on_flat_image():
    flat = np.ones((16, 16), dtype=np.float32)
    inc = isotropic_increment(flat, 3, np)
    assert np.allclose(inc, 0.0)


def test_structure_function_shape():
    img = fbm_terrain((64, 64), hurst=0.6, seed=3)
    lags = (1, 2, 4, 8)
    s = structure_function(img, lags, window_radius=6, xp=np)
    assert s.shape == (len(lags), 64, 64)
    assert np.all(s >= 0.0)


@pytest.mark.parametrize("hurst", [0.4, 0.6, 0.8])
def test_recovers_hurst_of_fbm(hurst):
    """On pure fBm the recovered fractal dimension should track D = 3 - H."""
    img = fbm_terrain((512, 512), hurst=hurst, seed=7)
    maps = carlotto_stein(img, window_radius=16, xp=np)
    # Use the interior median to avoid edge-replication bias.
    interior = maps.fractal_dimension[32:-32, 32:-32]
    est_D = float(np.median(interior))
    assert abs(est_D - (3.0 - hurst)) < 0.35, f"D={est_D}, expected≈{3.0 - hurst}"


def test_detects_planted_objects():
    """Planted smooth objects must be preferentially flagged over terrain."""
    window = 16
    image, truth = scene_with_objects((512, 512), hurst=0.6, seed=11)
    det = CarlottoSteinDetector(backend="cpu", window_radius=window, threshold_sigma=4.0)
    res = det.detect(image)

    mean_anom_obj = float(res.anomaly[truth].mean())
    mean_anom_bg = float(res.anomaly[~truth].mean())
    assert mean_anom_obj > 3.0 * mean_anom_bg + 1e-6

    # A window-radius box mean spreads each object's signature by ~`window`
    # pixels, so score detections against the truth mask dilated by the window
    # radius (a standard localization tolerance).
    truth_dil = box_mean(truth.astype(np.float32), window, np) > 0.0
    hits = res.candidate_mask & truth_dil
    assert res.candidate_mask.sum() > 0
    precision = hits.sum() / max(1, res.candidate_mask.sum())
    recall = (res.candidate_mask & truth).sum() / max(1, truth.sum())
    assert precision > 0.8, f"precision={precision}"
    assert recall > 0.3, f"recall={recall}"


def test_result_arrays_are_host_numpy():
    image, _ = scene_with_objects((128, 128), seed=1)
    res = CarlottoSteinDetector(backend="cpu").detect(image)
    for arr in (res.fractal_dimension, res.hurst, res.fit_residual,
                res.anomaly, res.candidate_mask):
        assert isinstance(arr, np.ndarray)
    assert res.candidate_mask.dtype == bool
    assert res.compute_seconds >= 0.0


def test_rejects_non_2d_input():
    det = CarlottoSteinDetector(backend="cpu")
    with pytest.raises(ValueError):
        det.detect(np.zeros((4, 4, 3), dtype=np.float32))


def test_cpu_backend_always_constructs():
    be = get_backend("cpu")
    assert be.name == "numpy"
    assert be.on_gpu is False


@pytest.mark.skipif(not gpu_available(), reason="no GPU / CuPy backend available")
def test_cpu_gpu_agreement():
    image, _ = scene_with_objects((256, 256), seed=5)
    cpu = CarlottoSteinDetector(backend="cpu").detect(image)
    gpu = CarlottoSteinDetector(backend="gpu").detect(image)
    assert np.allclose(cpu.anomaly, gpu.anomaly, atol=1e-3, rtol=1e-3)
    assert np.allclose(cpu.fractal_dimension, gpu.fractal_dimension, atol=1e-3, rtol=1e-3)
