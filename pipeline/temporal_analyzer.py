import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


def compute_variance(scores: np.ndarray) -> float:
    return float(np.var(scores))


def compute_trend_slope(scores: np.ndarray, timestamps: np.ndarray) -> float:
    if len(scores) < 2:
        return 0.0
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, scores)
    return float(slope)


def count_threshold_crossings(scores: np.ndarray, threshold: float = 0.5) -> int:
    crossings = 0
    for i in range(1, len(scores)):
        prev_above = scores[i-1] >= threshold
        curr_above = scores[i] >= threshold
        if prev_above != curr_above:
            crossings += 1
    return crossings


def compute_consecutive_changes(scores: np.ndarray, threshold: float = 0.1) -> int:
    if len(scores) < 2:
        return 0
    
    diffs = np.abs(np.diff(scores))
    significant_changes = np.sum(diffs > threshold)
    return int(significant_changes)


def compute_stability_score(
    variance: float,
    crossings: int,
    num_frames: int,
    max_variance: float = 0.1
) -> float:
    variance_score = 1.0 - min(variance / max_variance, 1.0)
    crossing_rate = crossings / max(num_frames - 1, 1)
    crossing_score = 1.0 - min(crossing_rate * 2, 1.0)
    stability = 0.6 * variance_score + 0.4 * crossing_score
    return float(np.clip(stability, 0.0, 1.0))


def detect_flickering(
    scores: np.ndarray,
    variance_threshold: float = 0.05,
    crossing_threshold: int = 3
) -> bool:
    variance = compute_variance(scores)
    crossings = count_threshold_crossings(scores)
    return variance > variance_threshold or crossings >= crossing_threshold


def compute_moving_average(scores: np.ndarray, window_size: int = 3) -> np.ndarray:
    if len(scores) < window_size:
        return scores
    
    padded = np.pad(scores, (window_size//2, window_size//2), mode='edge')
    smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
    return smoothed[:len(scores)]


def analyze_temporal_consistency(
    frame_scores: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    decision_threshold: float = 0.5
) -> Dict:
    num_frames = len(frame_scores)
    
    if timestamps is None:
        timestamps = np.arange(num_frames, dtype=np.float32)
    
    variance = compute_variance(frame_scores)
    trend_slope = compute_trend_slope(frame_scores, timestamps)
    crossings = count_threshold_crossings(frame_scores, decision_threshold)
    significant_changes = compute_consecutive_changes(frame_scores, threshold=0.1)
    
    stability = compute_stability_score(variance, crossings, num_frames)
    is_flickering = detect_flickering(frame_scores, variance_threshold=0.05, crossing_threshold=3)
    smoothed_scores = compute_moving_average(frame_scores, window_size=3)
    
    mean_score = float(np.mean(frame_scores))
    std_score = float(np.std(frame_scores))
    min_score = float(np.min(frame_scores))
    max_score = float(np.max(frame_scores))
    
    if stability > 0.8:
        interpretation = "Highly stable predictions - consistent with real video"
    elif stability > 0.6:
        interpretation = "Moderately stable predictions - some variation present"
    elif stability > 0.4:
        interpretation = "Unstable predictions - possible deepfake indicators"
    else:
        interpretation = "Highly unstable predictions - strong deepfake indicators"
    
    return {
        'variance': variance,
        'std_deviation': std_score,
        'trend_slope': trend_slope,
        'threshold_crossings': crossings,
        'significant_changes': significant_changes,
        'stability_score': stability,
        'is_flickering': is_flickering,
        'mean_score': mean_score,
        'min_score': min_score,
        'max_score': max_score,
        'score_range': max_score - min_score,
        'frame_scores': frame_scores.tolist(),
        'smoothed_scores': smoothed_scores.tolist(),
        'timestamps': timestamps.tolist(),
        'interpretation': interpretation,
        'num_frames': num_frames
    }


def compare_temporal_patterns(
    real_scores: np.ndarray,
    fake_scores: np.ndarray
) -> Dict:
    real_variance = compute_variance(real_scores)
    fake_variance = compute_variance(fake_scores)
    
    real_crossings = count_threshold_crossings(real_scores)
    fake_crossings = count_threshold_crossings(fake_scores)
    
    return {
        'real_variance': real_variance,
        'fake_variance': fake_variance,
        'variance_ratio': fake_variance / (real_variance + 1e-9),
        'real_crossings': real_crossings,
        'fake_crossings': fake_crossings,
        'crossing_ratio': fake_crossings / (real_crossings + 1)
    }


if __name__ == "__main__":
    print("Temporal Analyzer Module - Flickering Detection")
    print("=" * 60)
    
    real_scores = np.random.normal(0.2, 0.05, 50)
    real_scores = np.clip(real_scores, 0, 1)
    
    fake_scores = np.random.normal(0.7, 0.15, 50)
    fake_scores = np.clip(fake_scores, 0, 1)
    
    print("\nAnalyzing stable (real) video:")
    real_analysis = analyze_temporal_consistency(real_scores)
    print(f"  Variance: {real_analysis['variance']:.4f}")
    print(f"  Stability: {real_analysis['stability_score']:.4f}")
    print(f"  Flickering: {real_analysis['is_flickering']}")
    print(f"  {real_analysis['interpretation']}")
    
    print("\nAnalyzing unstable (fake) video:")
    fake_analysis = analyze_temporal_consistency(fake_scores)
    print(f"  Variance: {fake_analysis['variance']:.4f}")
    print(f"  Stability: {fake_analysis['stability_score']:.4f}")
    print(f"  Flickering: {fake_analysis['is_flickering']}")
    print(f"  {fake_analysis['interpretation']}")