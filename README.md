# ComfyUI Custom Node — Phase Congruency Corner

A ComfyUI custom node that detects corners using Phase Congruency, an
illumination-invariant and contrast-invariant feature detector based on the
coherence of Fourier phase components across multiple scales and orientations
(Kovesi, 1999).

## What is Phase Congruency Corner Detection?

Classical corner detectors (Harris, Shi-Tomasi, FAST) rely on intensity
gradients, so their output changes with lighting conditions and local contrast.
Phase Congruency Corner Detection takes a fundamentally different approach:

> A corner is detected at locations where Fourier components are maximally
> in phase across **all** orientations simultaneously — regardless of amplitude.

- **Edges** have high phase congruency in one dominant orientation.
- **Corners** have high phase congruency in **every** orientation at once.

This makes the detector:
- **Illumination invariant** — absolute brightness has no effect
- **Contrast invariant** — works equally on high-contrast and low-contrast regions
- **Perceptually meaningful** — closely matches human visual corner perception
- **Theoretically grounded** — based on Kovesi (1999) using log-Gabor filters

## Node

| Property | Value |
|---|---|
| Node name | `Phase Congruency Corner` |
| Category | `image/filters` |
| Input | `IMAGE` (RGB or grayscale, any resolution) |
| Output | `IMAGE` (grayscale corner map, white = strong corners) |

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `image` | IMAGE | — | — | Input image tensor |
| `nscale` | INT | 4 | 2–8 | Number of log-Gabor filter scales |
| `norient` | INT | 6 | 2–8 | Number of filter orientations |
| `min_wavelength` | INT | 3 | 2–20 | Minimum wavelength of finest-scale filter (pixels) |
| `mult` | FLOAT | 2.1 | 1.5–4.0 | Multiplicative factor between successive filter scales |
| `sigma_on_f` | FLOAT | 0.55 | 0.1–1.0 | Bandwidth of each log-Gabor filter (ratio σ/f₀) |
| `k` | FLOAT | 2.0 | 0.5–10.0 | Noise threshold in std devs above mean (Rayleigh) |
| `cutoff` | FLOAT | 0.5 | 0.1–0.9 | Butterworth low-pass filter cutoff frequency |

## Algorithm

### Step 1 — Per-orientation Phase Congruency maps

For each of the `norient` orientations θ_i:

1. Build a log-Gabor bandpass filter at each of the `nscale` scales,
   multiplied by an angular spread function (raised cosine).
2. Apply each filter in the frequency domain (FFT) to obtain complex responses.
3. Sum complex responses across all scales → energy vector (E_real, E_imag).
4. Estimate noise threshold T via Rayleigh statistics on finest-scale amplitude:
   `T = mean_noise + k × noise_std`
5. Compute per-orientation phase congruency:
   `PC_i = max(|energy| − T, 0) / (Σ amplitude + ε)`

### Step 2 — Structure matrix

Accumulate a symmetric 2×2 structure matrix at each pixel:

```
M = Σ_i  PC_i · [[cos²θ_i,    cosθ_i·sinθ_i],
                  [cosθ_i·sinθ_i,    sin²θ_i ]]
```

This is analogous to the Harris structure matrix, but uses phase congruency
values instead of intensity gradients.

### Step 3 — Corner response

The corner strength is the **minimum eigenvalue** of M:

```
λ_min = (Mxx + Myy) / 2 − sqrt(((Mxx − Myy) / 2)² + Mxy²)
```

A large λ_min means high phase congruency in every spatial direction,
which is the geometric signature of a corner point.

## Comparison: Edge vs. Corner

| | Phase Congruency Edge | Phase Congruency Corner |
|---|---|---|
| Strong response when | PC is high in **one** direction | PC is high in **all** directions |
| Aggregation method | Mean of per-orientation PC maps | Min. eigenvalue of structure matrix |
| Typical output | Thin edge contours | Isolated corner blobs |

## Comparison with Other Corner Detectors

| Method | Illumination Invariant | Contrast Invariant | Multi-scale |
|--------|----------------------|-------------------|-------------|
| Harris | No | No | No |
| Shi-Tomasi | No | No | No |
| FAST | No | No | No |
| Phase Congruency Corner | **Yes** | **Yes** | **Yes** |

## Parameter Tuning Guide

| Goal | Adjustment |
|------|-----------|
| Detect finer / more localised corners | Decrease `min_wavelength` |
| Detect coarser / broader junctions | Increase `min_wavelength` |
| More sensitive (at risk of noise) | Decrease `k` (e.g. 1.0) |
| Cleaner output, fewer false corners | Increase `k` (e.g. 4.0–6.0) |
| Better isotropy across all angles | Increase `norient` (e.g. 8) |
| Richer multi-scale information | Increase `nscale` (e.g. 6) |
| Reduce high-frequency ringing | Decrease `cutoff` (e.g. 0.4) |

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/bemoregt/ComfyUI_PCPoint.git
```

Restart ComfyUI. The node will appear under **image/filters → Phase Congruency Corner**.

## Requirements

All dependencies are already present in a standard ComfyUI installation:

```
numpy
torch
```

## Reference

Kovesi, P. (1999). Image Features from Phase Congruency.
*Videre: Journal of Computer Vision Research*, 1(3).

## License

MIT License — free for personal and commercial use.
