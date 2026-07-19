# carlotto_stein — hardware-accelerated fractal artificial-object detector

A high-speed implementation of the **Carlotto–Stein technique** for finding
artificial (man-made) objects on planetary surfaces:

> M. J. Carlotto and M. C. Stein, *"A Method for Searching for Artificial
> Objects on Planetary Surfaces,"* **Journal of the British Interplanetary
> Society (JBIS)**, Vol. 43, pp. 209–216, 1990.

The same vectorized kernels run on **CPU (NumPy)** or **GPU (CuPy)** — no code
changes, just a backend switch. On a CUDA/ROCm host the whole pipeline executes
on the accelerator.

---

## The method

Natural terrain, imaged as an intensity field `I(x, y)`, is well modeled as
**fractional Brownian motion (fBm)**. For fBm the expected absolute intensity
increment between two points separated by a lag `r` follows a power law:

```
E[ |I(p) − I(p + r)| ]  ∝  r^H
```

where `H` is the **Hurst exponent** (`0 < H < 1`), which sets the surface
**fractal dimension** `D = 3 − H`.

Carlotto & Stein estimate `H` **locally** — inside a sliding window around every
pixel — by measuring the increment statistic at several lags `r` and fitting a
straight line in log–log space. The slope is `H`; two maps fall out per pixel:

| Map | Meaning |
|---|---|
| **Fractal dimension** `D = 3 − H` | how rough the local surface is |
| **Fit residual** | how badly the local region departs from a single power law (fractal goodness-of-fit) |

**Artificial objects break the fractal model**: they are locally smoother
(anomalously *low* fractal dimension) and/or they violate the single-power-law
scaling (high residual). Fusing a low-dimension cue with a high-residual cue
produces an **anomaly map** whose bright regions are candidate artificial
objects, plus a thresholded **candidate mask**.

---

## Why it's fast (the "acceleration hardware" part)

Every step is elementwise array math plus separable box filters, so it maps
directly onto data-parallel hardware:

1. **Increments** are edge-clamped shifted absolute differences (no wrap-around
   artifacts across image borders).
2. **The local expectation `E[·]`** over the window is a box mean evaluated with
   a **summed-area table (integral image)**. Cost is `O(pixels)` and *does not
   grow with the window size* — a large algorithmic win over a naive
   window-by-window average.
3. **The per-pixel log–log line fit** is a closed-form least-squares reduction
   over the lag axis. Because every pixel shares the same design matrix (the
   lags), the normal equations collapse to a single `einsum`/`tensordot`.

Because the code only ever touches the injected array module `xp`, passing CuPy
instead of NumPy runs the identical kernels on a GPU. The benchmark
(`python -m carlotto_stein.benchmark`) times both backends with proper device
synchronization and **cross-checks that the GPU result matches the CPU result**,
so the speed-up is verified correct, not just fast.

---

## Install

```bash
pip install numpy                 # CPU path (only hard dependency)
pip install cupy-cuda12x          # optional: GPU path (match your CUDA/ROCm)
pip install pillow                # optional: load PNG/JPEG/TIFF inputs
```

Or, from this repo:

```bash
pip install -e .            # CPU
pip install -e ".[gpu]"     # + CuPy (CUDA 12.x wheel)
```

---

## Usage

### Library

```python
from carlotto_stein import CarlottoSteinDetector, scene_with_objects

image, truth = scene_with_objects()          # synthetic fractal test scene
det = CarlottoSteinDetector(backend="auto")   # GPU if available, else CPU
res = det.detect(image)

res.fractal_dimension   # (H, W) float map, D = 3 − H
res.fit_residual        # (H, W) fractal goodness-of-fit
res.anomaly             # (H, W) combined detection score
res.candidate_mask      # (H, W) bool, anomaly > threshold
res.backend             # "numpy" or "cupy"
res.compute_seconds     # device-synchronized compute time
```

`detect()` always returns host **NumPy** arrays, regardless of backend.

### Command line

```bash
# Synthetic demo scene (no input needed)
python -m carlotto_stein --demo --out-prefix demo

# Your own image (.npy, .pgm, or any Pillow format)
python -m carlotto_stein scene.pgm --backend auto --out-prefix result \
       --lags 1,2,3,4,6,8 --window 12 --sigma 4
```

Writes `<prefix>_anomaly.pgm`, `<prefix>_dimension.pgm`, `<prefix>_mask.pgm`.

### Benchmark

```bash
python -m carlotto_stein.benchmark --size 2048 --repeat 5
```

Prints CPU (and, where available, GPU) throughput, the measured speed-up, and
the CPU↔GPU max abs difference.

---

## Parameters worth knowing

| Parameter | Default | Effect |
|---|---|---|
| `lags` | `(1, 2, 3, 4, 6, 8)` | pixel separations sampled by the structure function |
| `window_radius` | `8` | half-size of the local fractal-estimation window; larger = smoother/steadier `H`, coarser localization |
| `threshold_sigma` | `4.0` | candidate cutoff, in robust (MAD) sigmas above the anomaly median |
| `dimension_weight` | `2.0` | weight of the low-fractal-dimension cue (primary discriminant) |
| `residual_weight` | `1.0` | weight of the non-fractal-residual cue (confirmation) |

The dimension cue is weighted above the residual cue by default because, on real
terrain, the single-window residual is the noisier of the two — consistent with
Carlotto & Stein's emphasis on fractal *dimension* as the primary discriminant.

---

## Layout

```
carlotto_stein/
  core.py        vectorized kernels (structure function, box mean, log-log fit, fusion)
  backends.py    CPU/GPU backend selection + device transfer + honest timing
  detector.py    CarlottoSteinDetector high-level API
  synthetic.py   fBm terrain + planted objects for demo/tests
  io.py          dependency-light .npy / PGM load & save (Pillow optional)
  benchmark.py   CPU vs GPU timing + correctness cross-check
  __main__.py    CLI
tests/           correctness tests (fBm slope recovery, box-mean parity, detection, CPU/GPU parity)
examples/        end-to-end synthetic demo
```

---

## Notes & limitations

- Reliability of any single detection is a function of terrain roughness: as
  terrain `H` approaches an object's own smoothness, contrast shrinks — this is
  inherent to the physics, not a bug.
- A single fBm realization contains genuinely non-fractal patches; the residual
  cue will flag them. Threshold and window should be tuned to the imagery.
- Ratings/detections are automated, not editorial.

*Part of the Science Bob News project — free to use with credit (see the code
repo's LICENSE).*
