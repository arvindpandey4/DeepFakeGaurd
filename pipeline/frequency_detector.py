import numpy as np
from typing import List

_LOW_FREQ_RADIUS    = 0.10
_MID_FREQ_RADIUS    = 0.30
_HIGH_FREQ_RADIUS   = 0.50


def _radial_bands(magnitude: np.ndarray):
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt(((y - cy) / h) ** 2 + ((x - cx) / w) ** 2)

    low  = magnitude[dist <= _LOW_FREQ_RADIUS]
    mid  = magnitude[(dist > _LOW_FREQ_RADIUS) & (dist <= _MID_FREQ_RADIUS)]
    high = magnitude[dist > _MID_FREQ_RADIUS]
    return low, mid, high


def _frequency_score_single(frame_rgb: np.ndarray) -> float:
    gray = (0.2989 * frame_rgb[:, :, 0] +
            0.5870 * frame_rgb[:, :, 1] +
            0.1140 * frame_rgb[:, :, 2]).astype(np.float32)

    h, w = gray.shape
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    windowed = gray * window

    fft = np.fft.fft2(windowed)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))

    low, mid, high = _radial_bands(magnitude)

    low_energy  = float(low.mean())  + 1e-9
    mid_energy  = float(mid.mean())  + 1e-9
    high_energy = float(high.mean()) + 1e-9

    slope_ratio = mid_energy / (high_energy + mid_energy)
    high_cv = float(high.std()) / (high_energy)

    slope_score = np.clip((slope_ratio - 0.55) / (0.85 - 0.55), 0.0, 1.0)
    cv_score    = np.clip(1.0 - (high_cv - 0.5) / 2.0, 0.0, 1.0)

    score = 0.5 * float(slope_score) + 0.5 * float(cv_score)
    return float(np.clip(score, 0.0, 1.0))


def compute_frequency_score(frames: np.ndarray) -> float:
    if frames.dtype != np.float32:
        frames = frames.astype(np.float32)

    if frames.max() <= 1.0:
        frames = frames * 255.0

    scores: List[float] = [_frequency_score_single(frames[i]) for i in range(len(frames))]

    if len(scores) == 0:
        return 0.5

    scores_arr = np.array(scores)

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