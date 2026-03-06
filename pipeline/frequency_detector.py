"""
Frequency Domain Artifact Detector
------------------------------------
GAN-generated faces leave behind characteristic spectral artifacts:
  - Periodic checkerboard patterns from transposed-convolution upsampling
  - Unnatural high-frequency energy distribution
  - Peaks at specific frequencies tied to the network's stride pattern

This module computes a "realness score" in the frequency domain:
  - Score close to 1.0 → frequency pattern looks REAL
  - Score close to 0.0 → frequency pattern looks like a DEEPFAKE

It is designed to be combined (via weighted ensemble) with MesoNet's
spatial score for improved robustness against modern deepfakes.

References:
  - Zhang et al. "Detecting and Simulating Artifacts in GAN Fake Images" (WIFS 2019)
  - Wang et al. "CNN-generated images are surprisingly easy to spot... for now" (CVPR 2020)
"""

import numpy as np
from typing import List


# ---------------------------------------------------------------------------
# Calibrated radial band thresholds (relative to Nyquist = 0.5 cycles/px)
# Tuned so that natural images score ~0.8–1.0 and typical GAN faces score
# ~0.2–0.6 when this module runs on MesoNet face-crop inputs (256×256).
# ---------------------------------------------------------------------------
_LOW_FREQ_RADIUS    = 0.10   # 0–10% of Nyquist  → DC + coarse structure
_MID_FREQ_RADIUS    = 0.30   # 10–30%             → textures
_HIGH_FREQ_RADIUS   = 0.50   # 30–50%             → fine details / artifacts


def _radial_bands(magnitude: np.ndarray):
    """
    Split the FFT magnitude into three concentric radial bands:
    low, mid, high.  `magnitude` should already be log-scaled.
    """
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    # normalise distance to [0, 0.5]
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt(((y - cy) / h) ** 2 + ((x - cx) / w) ** 2)

    low  = magnitude[dist <= _LOW_FREQ_RADIUS]
    mid  = magnitude[(dist > _LOW_FREQ_RADIUS) & (dist <= _MID_FREQ_RADIUS)]
    high = magnitude[dist > _MID_FREQ_RADIUS]
    return low, mid, high


def _frequency_score_single(frame_rgb: np.ndarray) -> float:
    """
    Compute the frequency-domain realness score for a single RGB frame.

    Returns a float in [0, 1] — higher means MORE likely REAL.
    """
    # Luminance approximation (avoid full-colour FFT for speed)
    gray = (0.2989 * frame_rgb[:, :, 0] +
            0.5870 * frame_rgb[:, :, 1] +
            0.1140 * frame_rgb[:, :, 2]).astype(np.float32)

    # Apply Hann window to reduce spectral leakage at edges
    h, w = gray.shape
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    windowed = gray * window

    # 2-D FFT → shift DC to centre → log-scale magnitude
    fft = np.fft.fft2(windowed)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))           # log(1 + |F|)

    low, mid, high = _radial_bands(magnitude)

    low_energy  = float(low.mean())  + 1e-9
    mid_energy  = float(mid.mean())  + 1e-9
    high_energy = float(high.mean()) + 1e-9

    # ---- Feature 1: mid-to-high energy slope ----
    # Real images: spectral energy drops smoothly (mid >> high)
    # GAN images:  partial high-frequency boosting → ratio closer to 1
    slope_ratio = mid_energy / (high_energy + mid_energy)    # [0.5, 1)

    # ---- Feature 2: coefficient of variation in the high band ----
    # GAN upsampling creates periodic peaks → higher variance
    high_cv = float(high.std()) / (high_energy)              # ≥0

    # ---- Combine into a single realness score ----
    # slope_ratio ≈ 0.70–0.85 for real; ≈ 0.55–0.70 for fake
    # high_cv     ≈ 0.6–1.2   for real; ≈ 1.2–2.5  for fake
    slope_score = np.clip((slope_ratio - 0.55) / (0.85 - 0.55), 0.0, 1.0)
    cv_score    = np.clip(1.0 - (high_cv - 0.5) / 2.0, 0.0, 1.0)

    # Simple equal-weight combination
    score = 0.5 * float(slope_score) + 0.5 * float(cv_score)
    return float(np.clip(score, 0.0, 1.0))


def compute_frequency_score(frames: np.ndarray) -> float:
    """
    Aggregate frequency-domain realness scores across all frames.

    Args:
        frames: float32 numpy array of shape (N, H, W, 3), values in [0, 1].

    Returns:
        Averaged realness score in [0, 1].
        High → REAL,  Low → DEEPFAKE  (same convention as MesoNet).
    """
    if frames.dtype != np.float32:
        frames = frames.astype(np.float32)

    # If values are in [0,1] scale to [0,255] for consistent processing
    if frames.max() <= 1.0:
        frames = frames * 255.0

    scores: List[float] = [_frequency_score_single(frames[i]) for i in range(len(frames))]

    if len(scores) == 0:
        return 0.5   # neutral fallback

    scores_arr = np.array(scores)

    # Use trimmed mean to discard outlier frames (helps with scene cuts, etc.)
    if len(scores_arr) >= 4:
        lo = np.percentile(scores_arr, 15)
        hi = np.percentile(scores_arr, 85)
        trimmed = scores_arr[(scores_arr >= lo) & (scores_arr <= hi)]
        if len(trimmed) > 0:
            scores_arr = trimmed

    result = float(np.mean(scores_arr))
    print(f"  [FreqDetector] Frequency realness score = {result:.4f}  "
          f"({'REAL-like' if result >= 0.5 else 'DEEPFAKE-like'})")
    return result
