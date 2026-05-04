import numpy as np
import time
import os
import sys
import requests
from typing import Dict, List, Tuple, Any, Optional, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.mesonet import Meso4, MesoInception4
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.mesonet import Meso4, MesoInception4

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
    from .frame_extractor import FrameExtractor, preprocess_frames
except ImportError:
    from pipeline.frame_extractor import FrameExtractor, preprocess_frames

try:
    from .frequency_detector import compute_frequency_score
except ImportError:
    from pipeline.frequency_detector import compute_frequency_score

try:
    from .explainability import generate_explanation_for_frames
except ImportError:
    from pipeline.explainability import generate_explanation_for_frames

try:
    from .temporal_analyzer import analyze_temporal_consistency
except ImportError:
    from pipeline.temporal_analyzer import analyze_temporal_consistency


_SPATIAL_WEIGHT    = 0.70
_FREQUENCY_WEIGHT  = 0.30


def normalize_to_fake_prob(score: float, mode: str = "real_prob") -> float:
    if mode == "real_prob":
        p_fake = 1.0 - score
    elif mode == "fake_prob":
        p_fake = score
    else:
        raise ValueError(f"normalize_to_fake_prob: unknown mode '{mode}'")
    return float(np.clip(p_fake, 0.001, 0.999))

_INCEPTION_WEIGHTS_URL = (
    "https://github.com/DariusAf/MesoNet/raw/master/weights/MesoInception4_DF.h5"
)


class AdaptivePipeline:
    def __init__(self, weights_path: Optional[str] = None, model_type: str = "MesoInception4"):
        self.model_type: str = model_type
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

    def _maybe_upgrade_to_inception(self) -> None:
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

    def _run_model_on_frames(self, frames: np.ndarray) -> np.ndarray:
        model = self.model
        if model is None:
            raise RuntimeError("Model used before initialization")
        batch_size = int(PIPELINE_CONFIG.get('batch_size', 16))
        return model.predict(frames, batch_size=batch_size, verbose=0)

    def _aggregate_predictions(self, preds: np.ndarray) -> float:
        flat = preds.flatten()
        if len(flat) >= 4:
            lo = np.percentile(flat, 15)
            hi = np.percentile(flat, 85)
            trimmed = flat[(flat >= lo) & (flat <= hi)]
            if len(trimmed) > 0:
                flat = trimmed
        return float(np.mean(flat))

    def _predict_frames(self, frames: np.ndarray) -> float:
        target_res = (256, 256)
        processed = preprocess_frames(frames, target_shape=target_res,
                                      normalize=True, sharpen=True)

        preds_orig = self._run_model_on_frames(processed)
        spatial_orig_real = self._aggregate_predictions(preds_orig)

        flipped = np.array([np.fliplr(f) for f in processed])
        preds_flip = self._run_model_on_frames(flipped)
        spatial_flip_real = self._aggregate_predictions(preds_flip)

        spatial_score_real = (spatial_orig_real + spatial_flip_real) / 2.0

        p_fake_spatial = normalize_to_fake_prob(spatial_score_real, mode="real_prob")
        print(f"  [CNN] p_fake_spatial = {p_fake_spatial:.4f}  "
              f"(real_orig={spatial_orig_real:.4f}, real_flip={spatial_flip_real:.4f})")

        raw_frames = preprocess_frames(frames, target_shape=target_res,
                                       normalize=False, sharpen=False)
        freq_score_real = compute_frequency_score(raw_frames)

        p_fake_freq = normalize_to_fake_prob(freq_score_real, mode="real_prob")
        print(f"  [FreqEnsemble] p_fake_freq = {p_fake_freq:.4f}")

        ensemble = _SPATIAL_WEIGHT * p_fake_spatial + _FREQUENCY_WEIGHT * p_fake_freq
        p_s = float(np.clip(ensemble, 0.001, 0.999))
        print(f"  [Ensemble] Final p(s) = {p_s:.4f}  "
              f"(deepfake prob; spatial×{_SPATIAL_WEIGHT} + freq×{_FREQUENCY_WEIGHT})")
        return p_s

    def _predict_frames_with_details(self, frames: np.ndarray, sample_rate: int = 3) -> Tuple[float, np.ndarray, np.ndarray]:
        if sample_rate > 1 and len(frames) > 10:
            sampled_indices = np.arange(0, len(frames), sample_rate)
            frames_to_process = frames[sampled_indices]
        else:
            frames_to_process = frames
            sampled_indices = np.arange(len(frames))
        
        target_res = (256, 256)
        processed = preprocess_frames(frames_to_process, target_shape=target_res,
                                      normalize=True, sharpen=True)

        preds_orig = self._run_model_on_frames(processed)
        flipped = np.array([np.fliplr(f) for f in processed])
        preds_flip = self._run_model_on_frames(flipped)
        
        per_frame_real = (preds_orig.flatten() + preds_flip.flatten()) / 2.0
        per_frame_fake = np.array([normalize_to_fake_prob(score, mode="real_prob") 
                                   for score in per_frame_real])
        
        if sample_rate > 1 and len(frames) > 10:
            full_per_frame_fake = np.interp(
                np.arange(len(frames)), 
                sampled_indices, 
                per_frame_fake
            )
        else:
            full_per_frame_fake = per_frame_fake
        
        raw_frames = preprocess_frames(frames_to_process, target_shape=target_res,
                                       normalize=False, sharpen=False)
        freq_score_real = compute_frequency_score(raw_frames)
        p_fake_freq = normalize_to_fake_prob(freq_score_real, mode="real_prob")
        
        all_processed = preprocess_frames(frames, target_shape=target_res,
                                          normalize=True, sharpen=True)
        all_preds_orig = self._run_model_on_frames(all_processed)
        all_flipped = np.array([np.fliplr(f) for f in all_processed])
        all_preds_flip = self._run_model_on_frames(all_flipped)
        
        spatial_score_real = self._aggregate_predictions(all_preds_orig)
        spatial_flip_real = self._aggregate_predictions(all_preds_flip)
        spatial_avg_real = (spatial_score_real + spatial_flip_real) / 2.0
        p_fake_spatial = normalize_to_fake_prob(spatial_avg_real, mode="real_prob")
        
        ensemble = _SPATIAL_WEIGHT * p_fake_spatial + _FREQUENCY_WEIGHT * p_fake_freq
        p_s = float(np.clip(ensemble, 0.001, 0.999))
        
        print(f"  [CNN] p_fake_spatial = {p_fake_spatial:.4f}")
        print(f"  [FreqEnsemble] p_fake_freq = {p_fake_freq:.4f}")
        print(f"  [Ensemble] Final p(s) = {p_s:.4f}")
        print(f"  [PerFrame] Scores range: [{full_per_frame_fake.min():.3f}, {full_per_frame_fake.max():.3f}]")
        print(f"  [Optimization] Processed {len(frames_to_process)}/{len(frames)} frames")
        
        return p_s, full_per_frame_fake, processed
    
    def _process_stage(self, 
                        video_path: str, 
                        stage_number: int,
                        stage_config: Dict[str, Any]) -> Dict[str, Any]:
        stage_start = time.time()
        
        if PIPELINE_CONFIG['verbose']:
            print(f"\n[STAGE {stage_number}] Escalation Level: {stage_config['name']}")
            print(f"  Configuration: Res={stage_config['resolution']}, Target_FPS={stage_config['frames_per_second']}")
        
        with FrameExtractor(video_path) as extractor:
            frames = extractor.extract_frames_adaptive(stage_config)
            
        p_s = self._predict_frames(frames)
        confidence_magnitude = max(p_s, 1.0 - p_s)
        stage_time = time.time() - stage_start
        tau_s = stage_config['confidence_threshold']
        should_exit = (confidence_magnitude >= tau_s) or (stage_number == 3)
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
        total_start = time.time()
        
        print(f"\n{'#'*70}")
        print(f"ADAPTIVE DEEPFAKE DETECTION PIPELINE")
        print(f"{'#'*70}")
        print(f"Video: {os.path.basename(video_path)}")
        
        final_result: Dict[str, Any] = {'label': 'UNKNOWN', 'p_s': 0.0, 'confidence': 0.0}
        exit_stage: int = 0
        
        for stage_num in [1, 2, 3]:
            stage_config = get_stage_config(stage_num)
            result = self._process_stage(video_path, stage_num, stage_config)
            
            self.stats['stage_times'][stage_num] += result['time']
            
            if result['should_exit']:
                final_result = result
                exit_stage = stage_num
                self.stats[f'stage{stage_num}_exits'] += 1
                break
        
        total_time = time.time() - total_start
        self.stats['total_videos'] += 1
        self.stats['total_time'] += total_time
        
        f_label: str = str(final_result['label'])
        f_p_s: float = float(final_result['p_s'])
        f_conf: float = float(final_result['confidence'])

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
        results = []
        
        for i, video_path in enumerate(video_paths, 1):
            print(f"\nProcessing video {i}/{len(video_paths)}")
            result = self.predict(video_path)
            results.append(result)
        
        return results
    
    def print_statistics(self):
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
        self.stats = {
            'total_videos': 0,
            'stage1_exits': 0,
            'stage2_exits': 0,
            'stage3_exits': 0,
            'total_time': 0,
            'stage_times': {1: 0, 2: 0, 3: 0}
        }

    def predict_with_explanation(self, video_path: str, top_k: int = 5) -> Dict:
        total_start = time.time()
        
        print(f"\n{'#'*70}")
        print(f"ADAPTIVE PIPELINE WITH EXPLAINABILITY")
        print(f"{'#'*70}")
        print(f"Video: {os.path.basename(video_path)}")
        
        final_result: Dict[str, Any] = {'label': 'UNKNOWN', 'p_s': 0.0, 'confidence': 0.0}
        exit_stage: int = 0
        all_frames: List[np.ndarray] = []
        all_processed: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        
        for stage_num in [1, 2, 3]:
            stage_config = get_stage_config(stage_num)
            stage_start = time.time()
            
            if PIPELINE_CONFIG['verbose']:
                print(f"\n[STAGE {stage_num}] {stage_config['name']}")
            
            with FrameExtractor(video_path) as extractor:
                frames = extractor.extract_frames_adaptive(stage_config)
            
            p_s, per_frame_scores, processed_frames = self._predict_frames_with_details(frames)
            
            all_frames.append(frames)
            all_processed.append(processed_frames)
            all_scores.append(per_frame_scores)
            
            confidence_magnitude = max(p_s, 1.0 - p_s)
            tau_s = stage_config['confidence_threshold']
            should_exit = (confidence_magnitude >= tau_s) or (stage_num == 3)
            label = "DEEPFAKE" if p_s >= 0.5 else "REAL"
            
            stage_time = time.time() - stage_start
            self.stats['stage_times'][stage_num] += stage_time
            
            if should_exit:
                final_result = {
                    'stage': stage_num,
                    'p_s': p_s,
                    'confidence': confidence_magnitude,
                    'label': label,
                    'time': stage_time,
                    'frames_processed': len(frames),
                    'should_exit': should_exit
                }
                exit_stage = stage_num
                self.stats[f'stage{stage_num}_exits'] += 1
                break
        
        total_time = time.time() - total_start
        self.stats['total_videos'] += 1
        self.stats['total_time'] += total_time
        
        exit_frames = all_frames[exit_stage - 1]
        exit_processed = all_processed[exit_stage - 1]
        exit_scores = all_scores[exit_stage - 1]
        
        print(f"\n[Explainability] Generating Grad-CAM heatmaps for top {top_k} frames...")
        explanations = generate_explanation_for_frames(
            self.model,
            exit_processed,
            exit_scores,
            exit_frames,
            top_k=min(top_k, len(exit_frames))
        )
        
        f_label: str = str(final_result['label'])
        f_p_s: float = float(final_result['p_s'])
        f_conf: float = float(final_result['confidence'])
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULT WITH EXPLANATIONS")
        print(f"{'='*60}")
        print(f"  Prediction: {f_label}")
        print(f"  Probability: {f_p_s:.4f}")
        print(f"  Confidence: {f_conf:.4f}")
        print(f"  Exit Stage: {exit_stage}")
        print(f"  Explanations: {len(explanations)} heatmaps generated")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"{'#'*70}\n")
        
        return {
            'video': video_path,
            'label': f_label,
            'probability': f_p_s,
            'confidence': f_conf,
            'exit_stage': exit_stage,
            'total_time': total_time,
            'stage_results': final_result,
            'explanations': explanations
        }

    def predict_with_temporal(self, video_path: str) -> Dict:
        total_start = time.time()
        
        print(f"\n{'#'*70}")
        print(f"ADAPTIVE PIPELINE WITH TEMPORAL ANALYSIS")
        print(f"{'#'*70}")
        print(f"Video: {os.path.basename(video_path)}")
        
        final_result: Dict[str, Any] = {'label': 'UNKNOWN', 'p_s': 0.0, 'confidence': 0.0}
        exit_stage: int = 0
        all_frame_scores: List[float] = []
        all_timestamps: List[float] = []
        
        for stage_num in [1, 2, 3]:
            stage_config = get_stage_config(stage_num)
            stage_start = time.time()
            
            if PIPELINE_CONFIG['verbose']:
                print(f"\n[STAGE {stage_num}] {stage_config['name']}")
            
            with FrameExtractor(video_path) as extractor:
                frames = extractor.extract_frames_adaptive(stage_config)
                video_info = extractor.get_video_info()
            
            p_s, per_frame_scores, _ = self._predict_frames_with_details(frames)
            
            fps = stage_config['frames_per_second']
            duration = video_info.get('duration', len(frames) / fps)
            timestamps = np.linspace(0, duration, len(per_frame_scores))
            
            all_frame_scores.extend(per_frame_scores.tolist())
            all_timestamps.extend(timestamps.tolist())
            
            confidence_magnitude = max(p_s, 1.0 - p_s)
            tau_s = stage_config['confidence_threshold']
            should_exit = (confidence_magnitude >= tau_s) or (stage_num == 3)
            label = "DEEPFAKE" if p_s >= 0.5 else "REAL"
            
            stage_time = time.time() - stage_start
            self.stats['stage_times'][stage_num] += stage_time
            
            if should_exit:
                final_result = {
                    'stage': stage_num,
                    'p_s': p_s,
                    'confidence': confidence_magnitude,
                    'label': label,
                    'time': stage_time,
                    'frames_processed': len(frames),
                    'should_exit': should_exit
                }
                exit_stage = stage_num
                self.stats[f'stage{stage_num}_exits'] += 1
                break
        
        total_time = time.time() - total_start
        self.stats['total_videos'] += 1
        self.stats['total_time'] += total_time
        
        print(f"\n[Temporal] Analyzing frame-to-frame consistency...")
        temporal_analysis = analyze_temporal_consistency(
            np.array(all_frame_scores),
            np.array(all_timestamps)
        )
        
        f_label: str = str(final_result['label'])
        f_p_s: float = float(final_result['p_s'])
        f_conf: float = float(final_result['confidence'])
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULT WITH TEMPORAL ANALYSIS")
        print(f"{'='*60}")
        print(f"  Prediction: {f_label}")
        print(f"  Probability: {f_p_s:.4f}")
        print(f"  Confidence: {f_conf:.4f}")
        print(f"  Exit Stage: {exit_stage}")
        print(f"\n  Temporal Metrics:")
        print(f"    Stability Score: {temporal_analysis['stability_score']:.4f}")
        print(f"    Variance: {temporal_analysis['variance']:.4f}")
        print(f"    Flickering: {temporal_analysis['is_flickering']}")
        print(f"    Threshold Crossings: {temporal_analysis['threshold_crossings']}")
        print(f"    {temporal_analysis['interpretation']}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"{'#'*70}\n")
        
        return {
            'video': video_path,
            'label': f_label,
            'probability': f_p_s,
            'confidence': f_conf,
            'exit_stage': exit_stage,
            'total_time': total_time,
            'stage_results': final_result,
            'temporal_analysis': temporal_analysis
        }


if __name__ == "__main__":
    print("Adaptive Pipeline Module - Test Mode")
    print("=" * 70)
    
    print_config()
    pipeline = AdaptivePipeline()
    
    print("\n✓ Pipeline initialized successfully!")
    print("\nTo use the pipeline:")
    print("  pipeline.predict('path/to/video.mp4')")
    print("  pipeline.print_statistics()")