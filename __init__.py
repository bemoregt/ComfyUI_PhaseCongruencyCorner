import torch
import numpy as np
import math


class PhaseCongruencyCorner:
    """
    ComfyUI Custom Node: Phase Congruency Corner Detection

    Detects corners using Phase Congruency — an illumination/contrast invariant
    feature detector based on the coherence of Fourier phase components across
    multiple scales and orientations (Kovesi, 1999).

    Corner detection principle:
      Edges have high PC in one dominant orientation.
      Corners have high PC across ALL orientations simultaneously.

    Algorithm:
      1. Compute per-orientation PC maps using log-Gabor filters.
      2. Build a 2×2 structure matrix from per-orientation PC values:
             M = Σ_i  PC_i · [[cos²θ_i,  cosθ_i·sinθ_i],
                               [cosθ_i·sinθ_i,  sin²θ_i]]
      3. Corner response R = λ_min(M)  (minimum eigenvalue of M).
         High λ_min means strong PC in every direction → corner.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "nscale": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Number of log-Gabor filter scales"
                }),
                "norient": ("INT", {
                    "default": 6,
                    "min": 2,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Number of filter orientations"
                }),
                "min_wavelength": ("INT", {
                    "default": 3,
                    "min": 2,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Minimum wavelength of log-Gabor filter (pixels)"
                }),
                "mult": ("FLOAT", {
                    "default": 2.1,
                    "min": 1.5,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "Scaling factor between successive filter scales"
                }),
                "sigma_on_f": ("FLOAT", {
                    "default": 0.55,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Bandwidth of log-Gabor filter (σ/f₀)"
                }),
                "k": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "Noise threshold in std devs above mean (Rayleigh)"
                }),
                "cutoff": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Butterworth low-pass filter cutoff frequency"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("phase_congruency_corner",)
    FUNCTION = "compute_corner"
    CATEGORY = "image/filters"

    def compute_corner(self, image, nscale, norient, min_wavelength,
                       mult, sigma_on_f, k, cutoff):
        """
        image: [B, H, W, C] torch tensor, float32, values in [0, 1]
        """
        batch_results = []

        for b in range(image.shape[0]):
            img_np = image[b].cpu().numpy()  # [H, W, C]

            # Convert to grayscale float64 in [0, 1]
            if img_np.shape[2] == 1:
                gray = img_np[:, :, 0].astype(np.float64)
            else:
                gray = (0.299 * img_np[:, :, 0] +
                        0.587 * img_np[:, :, 1] +
                        0.114 * img_np[:, :, 2]).astype(np.float64)

            corner = self._phase_congruency_corner(
                gray, nscale, norient, min_wavelength,
                mult, sigma_on_f, k, cutoff
            )

            # Normalize to [0, 1]
            c_min, c_max = corner.min(), corner.max()
            if c_max - c_min > 1e-10:
                corner = (corner - c_min) / (c_max - c_min)
            else:
                corner = np.zeros_like(corner)

            # [H, W] → [H, W, 3] RGB (grayscale as RGB)
            corner_rgb = np.stack([corner, corner, corner], axis=-1).astype(np.float32)
            batch_results.append(torch.from_numpy(corner_rgb))

        output = torch.stack(batch_results, dim=0)  # [B, H, W, 3]
        return (output,)

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def _phase_congruency_corner(self, img, nscale, norient, min_wavelength,
                                  mult, sigma_on_f, k, cutoff):
        """
        Compute the Phase Congruency Corner response for a 2-D grayscale image.

        Returns a 2-D array (same size as img) with the corner strength map.
        High values correspond to corner-like structures.

        Method:
          For each of the `norient` orientations θ_i we compute a PC map
          PC_i using log-Gabor filters exactly as in Kovesi's edge detector.

          We then build a symmetric 2×2 structure matrix at each pixel:

              M = Σ_i  PC_i · [[cos²θ_i,  cosθ_i·sinθ_i],
                                [cosθ_i·sinθ_i,  sin²θ_i]]

          The minimum eigenvalue of M is the corner response R.
          λ_min is large only when PC_i is high for ALL orientations
          (not just one), which is the geometric signature of a corner.
        """
        rows, cols = img.shape
        epsilon = 1e-4

        # ── Frequency coordinate grids (DC at center) ──────────────────
        cx = (np.arange(cols) - cols // 2) / cols
        cy = (np.arange(rows) - rows // 2) / rows
        x, y = np.meshgrid(cx, cy)

        radius = np.sqrt(x ** 2 + y ** 2)
        radius[rows // 2, cols // 2] = 1.0   # avoid log(0) at DC

        theta_grid = np.arctan2(-y, x)        # orientation of each frequency

        # Butterworth low-pass to suppress high-frequency ringing
        lp = self._lowpass_filter(rows, cols, 0.45, 15)

        # FFT of image (DC shifted to center)
        img_fft = np.fft.fftshift(np.fft.fft2(img))

        # ── Accumulate structure-matrix components ──────────────────────
        # M = [[Mxx, Mxy], [Mxy, Myy]]  (symmetric)
        Mxx = np.zeros((rows, cols))
        Myy = np.zeros((rows, cols))
        Mxy = np.zeros((rows, cols))

        for orient_idx in range(norient):
            angle = orient_idx * math.pi / norient   # filter orientation θ_i

            # ── Angular spread (raised cosine) ──────────────────────────
            ds = (np.sin(theta_grid) * math.cos(angle) -
                  np.cos(theta_grid) * math.sin(angle))
            dc = (np.cos(theta_grid) * math.cos(angle) +
                  np.sin(theta_grid) * math.sin(angle))
            dtheta = np.abs(np.arctan2(ds, dc))
            dtheta = np.minimum(dtheta * norient / 2.0, math.pi)
            spread = (np.cos(dtheta) + 1.0) / 2.0

            sum_an      = np.zeros((rows, cols))
            energy_real = np.zeros((rows, cols))
            energy_imag = np.zeros((rows, cols))
            an_finest   = None

            for scale in range(nscale):
                wavelength = min_wavelength * (mult ** scale)
                fo = 1.0 / wavelength

                # Log-Gabor radial component
                log_gabor = np.exp(
                    -(np.log(radius / fo)) ** 2 /
                    (2.0 * (math.log(sigma_on_f) ** 2))
                )
                log_gabor[rows // 2, cols // 2] = 0.0   # zero DC

                filt = log_gabor * spread * lp

                # Complex filter response (inverse FFT)
                resp = np.fft.ifft2(np.fft.ifftshift(img_fft * filt))
                re, im = resp.real, resp.imag
                an = np.sqrt(re ** 2 + im ** 2)

                sum_an      += an
                energy_real += re
                energy_imag += im

                if scale == 0:
                    an_finest = an

            # ── Rayleigh noise threshold ────────────────────────────────
            tau       = np.sqrt(np.mean(an_finest ** 2) / 2.0 + epsilon)
            mean_n    = tau * math.sqrt(math.pi / 2.0)
            std_n     = tau * math.sqrt((4.0 - math.pi) / 2.0)
            T         = mean_n + k * std_n

            # Per-orientation phase congruency
            energy   = np.sqrt(energy_real ** 2 + energy_imag ** 2)
            pc_i     = np.maximum(energy - T, 0.0) / (sum_an + epsilon)

            # ── Accumulate structure matrix ─────────────────────────────
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            Mxx += pc_i * (cos_a ** 2)
            Myy += pc_i * (sin_a ** 2)
            Mxy += pc_i * (cos_a * sin_a)

        # ── Minimum eigenvalue of M at each pixel ───────────────────────
        # For a 2×2 symmetric matrix [[a, b], [b, c]]:
        #   λ_min = (a+c)/2 - sqrt(((a-c)/2)² + b²)
        trace_half = (Mxx + Myy) / 2.0
        det_term   = np.sqrt(((Mxx - Myy) / 2.0) ** 2 + Mxy ** 2)
        lambda_min = trace_half - det_term   # minimum eigenvalue

        # Clip negative values (numerical noise)
        corner = np.maximum(lambda_min, 0.0)
        return corner

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _lowpass_filter(self, rows, cols, cutoff, n):
        """
        Butterworth low-pass filter in the frequency domain.
        cutoff : cutoff as fraction of Nyquist (0..0.5)
        n      : filter order
        """
        if cols % 2 == 0:
            cx = np.arange(-cols // 2, cols // 2) / cols
        else:
            cx = np.arange(-(cols - 1) // 2, (cols - 1) // 2 + 1) / cols

        if rows % 2 == 0:
            cy = np.arange(-rows // 2, rows // 2) / rows
        else:
            cy = np.arange(-(rows - 1) // 2, (rows - 1) // 2 + 1) / rows

        x, y = np.meshgrid(cx, cy)
        radius = np.sqrt(x ** 2 + y ** 2)
        lp = 1.0 / (1.0 + (radius / cutoff) ** (2 * n))
        return lp


# ── ComfyUI registration ────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "PhaseCongruencyCorner": PhaseCongruencyCorner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhaseCongruencyCorner": "Phase Congruency Corner"
}
