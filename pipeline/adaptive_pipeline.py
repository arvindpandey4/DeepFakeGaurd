"""
Multi-Stage Adaptive Inference Pipeline for Deepfake Detection
"""

import numpy as np # type: ignore
import time
import os
import sys
import requests  # type: ignore
from typing import Dict, List, Tuple, Any, Optional, Union

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.mesonet import Meso4, MesoInception4 # type: ignore
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.mesonet import Meso4, MesoInception4 # type: ignore

try:
    from .config import (
        WEIGHTS_PATH,
        PIPELINE_CONFIG,
        CLASSIFICATION_CONFIG,
        get_stage_config,
        print_config
    )
except ImportError:
    from pipeline.config import (
        WEIGHTS_PATH,
        PIPELINE_CONFIG,
        CLASSIFICATION_CONFIG,
        get_stage_config,
        print_config
    )

try:
    from .frame_extractor import FrameExtractor, preprocess_frames # type: ignore
except ImportError:
    from pipeline.frame_extractor import FrameExtractor, preprocess_frames # type: ignore

try:
    from .frequency_detector import compute_frequency_score  # type: ignore
except ImportError:
    from pipeline.frequency_detector import compute_frequency_score  # type: ignore


# ---------------------------------------------------------------------------
# Ensemble weights: how much to trust each signal
# ---------------------------------------------------------------------------
_SPATIAL_WEIGHT    = 0.70   # MesoNet CNN spatial score
_FREQUENCY_WEIGHT  = 0.30   # FFT frequency-domain score


# ---------------------------------------------------------------------------
# Probability normalisation utility  (Task 7 from paper alignment spec)
# ---------------------------------------------------------------------------
def normalize_to_fake_prob(score: float, mode: str = "real_prob") -> float:
    """
    Convert any detector output to a **deepfake probability** before
    ensemble fusion.  This is the single choke-point for probability
    semantics — all detectors must pass through here.

    Args:
        score: Raw detector output, already in [0, 1].
        mode:  "real_prob"  → detector output is P(real);  return 1 - score
               "fake_prob"  → detector output is P(fake);  return score as-is

    Returns:
        p_fake ∈ [0.001, 0.999]  (clamped for numerical stability)
    """
    if mode == "real_prob":
        p_fake = 1.0 - score
    elif mode == "fake_prob":
        p_fake = score
    else:
        raise ValueError(f"normalize_to_fake_prob: unknown mode '{mode}'")
    # Clamp to (0, 1) open interval — prevents log(0) / division issues
    return float(np.clip(p_fake, 0.001, 0.999))

# URL for the more accurate MesoInception4 weights (official MesoNet repo)
_INCEPTION_WEIGHTS_URL = (
    "https://github.com/DariusAf/MesoNet/raw/master/weights/MesoInception4_DF.h5"
)


class AdaptivePipeline:
    """
    Multi-Stage Adaptive Inference Pipeline — Enhanced Edition

    Improvements over the baseline:
      1. MesoInception4 architecture (auto-downloaded on first run) — more
         accurate than Meso4, uses Inception modules for multi-scale features.
      2. Frequency-domain ensemble — FFT-based GAN artifact detector runs in
         parallel with MesoNet and is blended into the final score (30%).
      3. Test-Time Augmentation (TTA) — each batch is also run on horizontally
         flipped frames; scores are averaged, reducing prediction variance.
      4. Trimmed-mean aggregation — top/bottom 15% frame scores are discarded
         before averaging, preventing outlier frames from dominating.
      5. Unsharp-mask sharpening — applied before model inference to expose
         GAN blending seams that normal pixel averages smooth away.
    """

    def __init__(self, weights_path: Optional[str] = None, model_type: str = "MesoInception4"):
        self.model_type: str = model_type

        # Prefer MesoInception4 weights; fall back to Meso4 if absent
        self.weights_path: str = weights_path or WEIGHTS_PATH
        self._maybe_upgrade_to_inception()

        self.model: Optional[Union[Meso4, MesoInception4]] = None

        self.stats: Dict[str, Any] = {
            'total_videos': 0,
            'stage1_exits': 0,
            'stage2_exits': 0,
            'stage3_exits': 0,
            'total_time': 0.0,
            'stage_times': {1: 0.0, 2: 0.0, 3: 0.0}
        }

        self._load_model()

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def _maybe_upgrade_to_inception(self) -> None:
        """
        If MesoInception4_DF.h5 is not present, attempt to download it.
        Falls back to Meso4 silently if the download fails.
        """
        weights_dir = os.path.dirname(self.weights_path)
        inception_path = os.path.join(weights_dir, "MesoInception4_DF.h5")

        if os.path.exists(inception_path):
            print(f"[Upgrade] Using MesoInception4_DF.h5 (better accuracy)")
            self.weights_path = inception_path
            self.model_type   = "MesoInception4"
            return

        print("[Upgrade] MesoInception4_DF.h5 not found — trying to download...")
        try:
            r = requests.get(_INCEPTION_WEIGHTS_URL, stream=True, timeout=60)
            r.raise_for_status()
            with open(inception_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_kb = os.path.getsize(inception_path) / 1024
            print(f"[Upgrade] ✓ Downloaded MesoInception4_DF.h5 ({size_kb:.0f} KB)")
            self.weights_path = inception_path
            self.model_type   = "MesoInception4"
        except Exception as e:
            print(f"[Upgrade] Download failed ({e}) — using Meso4_DF.h5")
            self.model_type = "Meso4"
    
    def _load_model(self):
        """Load the MesoNet model (Meso4 or MesoInception4)"""
        print(f"Loading {self.model_type} model...")

        if self.model_type == "MesoInception4":
            self.model = MesoInception4(input_shape=(256, 256, 3))
        else:
            self.model = Meso4(input_shape=(256, 256, 3))

        model = self.model
        if model is None:
            raise RuntimeError(f"Failed to create {self.model_type} model")
        model.build()

        if os.path.exists(self.weights_path):
            print(f"Loading weights from: {self.weights_path}")
            model.load_weights(self.weights_path)
            print("✓ Model loaded successfully!")
        else:
            print(f"⚠ Warning: Weights not found at {self.weights_path}")
            print("  Model will use random initialization (for demo purposes)")

    # ------------------------------------------------------------------
    # Core inference helpers
    # ------------------------------------------------------------------

    def _run_model_on_frames(self, frames: np.ndarray) -> np.ndarray:
        """Run the CNN on pre-processed float32 frames, return raw predictions."""
        model = self.model
        if model is None:
            raise RuntimeError("Model used before initialization")
        batch_size = int(PIPELINE_CONFIG.get('batch_size', 16))
        return model.predict(frames, batch_size=batch_size, verbose=0)

    def _aggregate_predictions(self, preds: np.ndarray) -> float:
        """
        Trimmed-mean aggregation: discard the top and bottom 15% frame scores
        (handles scene-cut frames, partial faces, etc.) then average the rest.
        """
        flat = preds.flatten()
        if len(flat) >= 4:
            lo = np.percentile(flat, 15)
            hi = np.percentile(flat, 85)
            trimmed = flat[(flat >= lo) & (flat <= hi)]
            if len(trimmed) > 0:
                flat = trimmed
        return float(np.mean(flat))

    def _predict_frames(self, frames: np.ndarray) -> float:
        """
        Full prediction pipeline for a batch of face-cropped frames:
          1. Sharpen + resize to 256×256 + normalise
          2. Run MesoNet (spatial score)  — outputs real_prob by convention
          3. Run MesoNet on horizontally flipped frames (TTA)
          4. Average original + flipped spatial scores
          5. Convert spatial real_prob → fake_prob via normalize_to_fake_prob()
          6. Compute frequency-domain score (FFT)  — also real_prob
          7. Convert frequency real_prob → fake_prob via normalize_to_fake_prob()
          8. Ensemble: p(s) = w_spatial·p_fake_spatial + w_freq·p_fake_freq

        Returns:
            p(s) ∈ [0.001, 0.999]  — DEEPFAKE probability
            (paper def: P(y=1|f), y=1 ↔ deepfake, y=0 ↔ real)
            High p(s) → DEEPFAKE,  Low p(s) → REAL
        """
        # Step 1 — resize / sharpen / normalise
        target_res = (256, 256)
        processed = preprocess_frames(frames, target_shape=target_res,
                                      normalize=True, sharpen=True)

        # Step 2 — MesoNet on original orientation (output: real_prob)
        preds_orig = self._run_model_on_frames(processed)
        spatial_orig_real = self._aggregate_predictions(preds_orig)

        # Step 3 — TTA: horizontal flip (output: real_prob)
        flipped = np.array([np.fliplr(f) for f in processed])
        preds_flip = self._run_model_on_frames(flipped)
        spatial_flip_real = self._aggregate_predictions(preds_flip)

        # Step 4 — average original + flipped (still real_prob)
        spatial_score_real = (spatial_orig_real + spatial_flip_real) / 2.0

        # Step 5 — convert CNN real_prob → deepfake_prob
        p_fake_spatial = normalize_to_fake_prob(spatial_score_real, mode="real_prob")
        print(f"  [CNN] p_fake_spatial = {p_fake_spatial:.4f}  "
              f"(real_orig={spatial_orig_real:.4f}, real_flip={spatial_flip_real:.4f})")

        # Step 6 — frequency domain score (uses un-normalised frames; output: real_prob)
        raw_frames = preprocess_frames(frames, target_shape=target_res,
                                       normalize=False, sharpen=False)
        freq_score_real = compute_frequency_score(raw_frames)

        # Step 7 — convert frequency real_prob → deepfake_prob
        p_fake_freq = normalize_to_fake_prob(freq_score_real, mode="real_prob")
        print(f"  [FreqEnsemble] p_fake_freq = {p_fake_freq:.4f}")

        # Step 8 — ensemble: p(s) = w_spatial * p_fake_spatial + w_freq * p_fake_freq
        ensemble = _SPATIAL_WEIGHT * p_fake_spatial + _FREQUENCY_WEIGHT * p_fake_freq
        # Clamp to [0.001, 0.999] for numerical stability (Task 8)
        p_s = float(np.clip(ensemble, 0.001, 0.999))
        print(f"  [Ensemble] Final p(s) = {p_s:.4f}  "
              f"(deepfake prob; spatial×{_SPATIAL_WEIGHT} + freq×{_FREQUENCY_WEIGHT})")
        return p_s
    
    def _process_stage(self, 
                        video_path: str, 
                        stage_number: int,
                        stage_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single stage following Section IV.B formulation
        
        Args:
            video_path: Path to video
            stage_number: Stage number s
            stage_config: Stage configuration parameters
            
        Returns:
            Dictionary with stage results including confidence-based metadata
        """
        stage_start = time.time()
        
        if PIPELINE_CONFIG['verbose']:
            print(f"\n[STAGE {stage_number}] Escalation Level: {stage_config['name']}")
            print(f"  Configuration: Res={stage_config['resolution']}, Target_FPS={stage_config['frames_per_second']}")
        
        # Frame extraction (Section IV.C: n_s)
        with FrameExtractor(video_path) as extractor:
            frames = extractor.extract_frames_adaptive(stage_config)
            
        # Inference and Aggregation (Section IV.B: Equation 4)
        p_s = self._predict_frames(frames)
        
        # Confidence Assessment (Section IV.D: Equation 19)
        # p(s) is now deepfake probability: P(y=1|f), y=1 ↔ deepfake
        # confidence = max(p_s, 1-p_s) — symmetric around 0.5 decision boundary
        confidence_magnitude = max(p_s, 1.0 - p_s)

        stage_time = time.time() - stage_start

        # Early termination condition (Section IV.B: Equation 5)
        # Exit when confidence >= tau_s (thresholds are monotone-decreasing: τ1 > τ2 > τ3)
        tau_s = stage_config['confidence_threshold']
        should_exit = (confidence_magnitude >= tau_s) or (stage_number == 3)

        # Predicted Class (Equation 6 — paper definition)
        # p(s) >= 0.5 → DEEPFAKE  |  p(s) < 0.5 → REAL
        label = "DEEPFAKE" if p_s >= 0.5 else "REAL"

        if PIPELINE_CONFIG['verbose']:
            print(f"  p(s) = {p_s:.4f}, Confidence = {confidence_magnitude:.4f}")
            print(f"  Threshold tau_{stage_number} = {tau_s:.2f}")
            print(f"  Decision: {label} ({'EXIT' if should_exit else 'ESCALATE'})")
            print(f"  Compute Time: {stage_time:.2f}s")

        return {
            'stage': stage_number,
            'p_s': p_s,
            'confidence': confidence_magnitude,
            'label': label,
            'time': stage_time,
            'frames_processed': len(frames),
            'should_exit': should_exit
        }
    
    def predict(self, video_path: str) -> Dict:
        """
        Predict whether a video is deepfake using adaptive pipeline
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with prediction results
        """
        total_start = time.time()
        
        print(f"\n{'#'*70}")
        print(f"ADAPTIVE DEEPFAKE DETECTION PIPELINE")
        print(f"{'#'*70}")
        print(f"Video: {os.path.basename(video_path)}")
        
        # Process each stage
        # Initialize with placeholder to satisfy linter
        final_result: Dict[str, Any] = {'label': 'UNKNOWN', 'p_s': 0.0, 'confidence': 0.0}
        exit_stage: int = 0
        
        for stage_num in [1, 2, 3]:
            stage_config = get_stage_config(stage_num)
            result = self._process_stage(video_path, stage_num, stage_config)
            
            # Update stage-specific time tracking
            self.stats['stage_times'][stage_num] += result['time']
            
            # Check for escalation logic (Section IV.B: Escalation Logic)
            if result['should_exit']:
                final_result = result
                exit_stage = stage_num
                
                # Update exit statistics
                self.stats[f'stage{stage_num}_exits'] += 1
                break
        
        total_time = time.time() - total_start
        
        # Update global statistics
        self.stats['total_videos'] += 1
        self.stats['total_time'] += total_time
        
        # Extract values into variables to satisfy linter (avoids complex f-string expressions)
        # Using cast or careful access to avoid "attribute base undefined"
        f_label: str = str(final_result['label'])
        f_p_s: float = float(final_result['p_s'])
        f_conf: float = float(final_result['confidence'])

        # Print final result
        print(f"\n{'='*60}")
        print(f"FINAL RESULT")
        print(f"{'='*60}")
        print(f"  Prediction: {f_label}")
        print(f"  p(s) = {f_p_s:.4f}  (deepfake prob: high=DEEPFAKE, low=REAL)")
        print(f"  Confidence: {f_conf:.4f}")
        print(f"  Exit Stage: {exit_stage}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"{'#'*70}\n")
        
        return {
            'video': video_path,
            'label': f_label,
            'probability': f_p_s,
            'confidence': f_conf,
            'exit_stage': exit_stage,
            'total_time': total_time,
            'stage_results': final_result
        }
    
    def predict_batch(self, video_paths: List[str]) -> List[Dict]:
        """
        Predict on multiple videos
        
        Args:
            video_paths: List of video paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, video_path in enumerate(video_paths, 1):
            print(f"\nProcessing video {i}/{len(video_paths)}")
            result = self.predict(video_path)
            results.append(result)
        
        return results
    
    def print_statistics(self):
        """Print pipeline statistics"""
        if self.stats['total_videos'] == 0:
            print("No videos processed yet.")
            return
        
        print(f"\n{'='*70}")
        print(f"PIPELINE STATISTICS")
        print(f"{'='*70}")
        print(f"Total Videos Processed: {self.stats['total_videos']}")
        print(f"\nExit Distribution:")
        print(f"  Stage 1 (Fast):     {self.stats['stage1_exits']:3d} ({self.stats['stage1_exits']/self.stats['total_videos']*100:5.1f}%)")
        print(f"  Stage 2 (Balanced): {self.stats['stage2_exits']:3d} ({self.stats['stage2_exits']/self.stats['total_videos']*100:5.1f}%)")
        print(f"  Stage 3 (Accurate): {self.stats['stage3_exits']:3d} ({self.stats['stage3_exits']/self.stats['total_videos']*100:5.1f}%)")
        
        print(f"\nAverage Time per Video: {self.stats['total_time']/self.stats['total_videos']:.2f}s")
        print(f"Total Processing Time: {self.stats['total_time']:.2f}s")
        
        print(f"\nTime per Stage:")
        for stage in [1, 2, 3]:
            avg_time = self.stats['stage_times'][stage] / self.stats['total_videos']
            print(f"  Stage {stage}: {avg_time:.2f}s average")
        
        print(f"{'='*70}\n")
    
    def reset_statistics(self):
        """Reset statistics"""
        self.stats = {
            'total_videos': 0,
            'stage1_exits': 0,
            'stage2_exits': 0,
            'stage3_exits': 0,
            'total_time': 0,
            'stage_times': {1: 0, 2: 0, 3: 0}
        }


if __name__ == "__main__":
    # Test pipeline
    print("Adaptive Pipeline Module - Test Mode")
    print("=" * 70)
    
    # Print configuration
    print_config()
    
    # Create pipeline
    pipeline = AdaptivePipeline()
    
    print("\n✓ Pipeline initialized successfully!")
    print("\nTo use the pipeline:")
    print("  pipeline.predict('path/to/video.mp4')")
    print("  pipeline.print_statistics()")
