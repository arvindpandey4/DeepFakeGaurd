"""
Temporal Consistency Analysis
Detects flickering and inconsistencies in per-frame predictions over time
"""

import numpy as np  # type: ignore
from typing import Dict, List, Tuple, Optional
from scipy import stats  # type: ignore


def compute_variance(scores: np.ndarray) -> float:
    """
    Compute variance of prediction scores
    High variance indicates flickering/instability
    """
    return float(np.var(scores))


def compute_trend_slope(scores: np.ndarray, timestamps: np.ndarray) -> float:
    """
    Compute linear trend slope using least squares
    Positive slope = increasing fake probability over time
    Negative slope = decreasing fake probability over time
    """
    if len(scores) < 2:
        return 0.0
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, scores)
    return float(slope)


def count_threshold_crossings(scores: np.ndarray, threshold: float = 0.5) -> int:
    """
    Count how many times the score crosses the decision threshold
    High count indicates unstable/flickering predictions
    """
    crossings = 0
    for i in range(1, len(scores)):
        prev_above = scores[i-1] >= threshold
        curr_above = scores[i] >= threshold
        if prev_above != curr_above:
            crossings += 1
    return crossings


def compute_consecutive_changes(scores: np.ndarray, threshold: float = 0.1) -> int:
    """
    Count significant frame-to-frame changes
    A change is significant if |score[i] - score[i-1]| > threshold
    """
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
    """
    Compute overall temporal stability score (0-1)
    
    1.0 = perfectly stable (real video)
    0.0 = highly unstable (likely deepfake)
    
    Args:
        variance: Score variance
        crossings: Number of threshold crossings
        num_frames: Total number of frames
        max_variance: Maximum expected variance for real videos
        
    Returns:
        Stability score in [0, 1]
    """
    # Variance component (0-1, lower variance = higher stability)
    variance_score = 1.0 - min(variance / max_variance, 1.0)
    
    # Crossing component (0-1, fewer crossings = higher stability)
    crossing_rate = crossings / max(num_frames - 1, 1)
    crossing_score = 1.0 - min(crossing_rate * 2, 1.0)  # Scale by 2 for sensitivity
    
    # Weighted combination
    stability = 0.6 * variance_score + 0.4 * crossing_score
    
    return float(np.clip(stability, 0.0, 1.0))


def detect_flickering(
    scores: np.ndarray,
    variance_threshold: float = 0.05,
    crossing_threshold: int = 3
) -> bool:
    """
    Detect if the video exhibits flickering behavior
    
    Args:
        scores: Per-frame prediction scores
        variance_threshold: Variance above which flickering is detected
        crossing_threshold: Number of crossings above which flickering is detected
        
    Returns:
        True if flickering detected, False otherwise
    """
    variance = compute_variance(scores)
    crossings = count_threshold_crossings(scores)
    
    return variance > variance_threshold or crossings >= crossing_threshold


def compute_moving_average(scores: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Compute moving average of scores to smooth out noise
    
    Args:
        scores: Per-frame scores
        window_size: Size of moving average window
        
    Returns:
        Smoothed scores
    """
    if len(scores) < window_size:
        return scores
    
    # Pad edges to maintain length
    padded = np.pad(scores, (window_size//2, window_size//2), mode='edge')
    smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
    
    return smoothed[:len(scores)]


def analyze_temporal_consistency(
    frame_scores: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    decision_threshold: float = 0.5
) -> Dict:
    """
    Comprehensive temporal consistency analysis
    
    Args:
        frame_scores: Per-frame deepfake probability scores (N,)
        timestamps: Frame timestamps in seconds (N,) - auto-generated if None
        decision_threshold: Classification threshold (default 0.5)
        
    Returns:
        Dictionary with temporal metrics and analysis
    """
    num_frames = len(frame_scores)
    
    # Generate timestamps if not provided
    if timestamps is None:
        timestamps = np.arange(num_frames, dtype=np.float32)
    
    # Core metrics
    variance = compute_variance(frame_scores)
    trend_slope = compute_trend_slope(frame_scores, timestamps)
    crossings = count_threshold_crossings(frame_scores, decision_threshold)
    significant_changes = compute_consecutive_changes(frame_scores, threshold=0.1)
    
    # Stability score
    stability = compute_stability_score(variance, crossings, num_frames)
    
    # Flickering detection
    is_flickering = detect_flickering(frame_scores, variance_threshold=0.05, crossing_threshold=3)
    
    # Smoothed scores for visualization
    smoothed_scores = compute_moving_average(frame_scores, window_size=3)
    
    # Statistical summary
    mean_score = float(np.mean(frame_scores))
    std_score = float(np.std(frame_scores))
    min_score = float(np.min(frame_scores))
    max_score = float(np.max(frame_scores))
    
    # Interpretation
    if stability > 0.8:
        interpretation = "Highly stable predictions - consistent with real video"
    elif stability > 0.6:
        interpretation = "Moderately stable predictions - some variation present"
    elif stability > 0.4:
        interpretation = "Unstable predictions - possible deepfake indicators"
    else:
        interpretation = "Highly unstable predictions - strong deepfake indicators"
    
    return {
        # Core metrics
        'variance': variance,
        'std_deviation': std_score,
        'trend_slope': trend_slope,
        'threshold_crossings': crossings,
        'significant_changes': significant_changes,
        
        # Scores
        'stability_score': stability,
        'is_flickering': is_flickering,
        
        # Statistics
        'mean_score': mean_score,
        'min_score': min_score,
        'max_score': max_score,
        'score_range': max_score - min_score,
        
        # Data for visualization
        'frame_scores': frame_scores.tolist(),
        'smoothed_scores': smoothed_scores.tolist(),
        'timestamps': timestamps.tolist(),
        
        # Interpretation
        'interpretation': interpretation,
        'num_frames': num_frames
    }


def compare_temporal_patterns(
    real_scores: np.ndarray,
    fake_scores: np.ndarray
) -> Dict:
    """
    Compare temporal patterns between real and fake video predictions
    Useful for validation and threshold tuning
    
    Args:
        real_scores: Scores from known real videos
        fake_scores: Scores from known fake videos
        
    Returns:
        Comparison metrics
    """
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
    
    # Example: Stable real video
    real_scores = np.random.normal(0.2, 0.05, 50)  # Low variance around 0.2
    real_scores = np.clip(real_scores, 0, 1)
    
    # Example: Flickering fake video
    fake_scores = np.random.normal(0.7, 0.15, 50)  # High variance around 0.7
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
