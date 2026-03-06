"""
Multi-Stage Adaptive Inference Pipeline for Deepfake Detection
"""

import numpy as np # type: ignore
import time
from typing import Dict, List, Tuple
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.mesonet import Meso4, MesoInception4 # type: ignore
except ImportError:
    # Fallback for different execution contexts
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.mesonet import Meso4, MesoInception4 # type: ignore

try:
    from .config import ( # type: ignore
        WEIGHTS_PATH, 
        PIPELINE_CONFIG, 
        CLASSIFICATION_CONFIG, 
        get_stage_config,
        print_config
    )
except ImportError:
    from pipeline.config import ( # type: ignore
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
from typing import Dict, List, Tuple, Any, Optional, Union
# Remove the tensorflow import if not strictly needed for logic, 
# or keep it if you need specific types, but meso classes are wrappers.


class AdaptivePipeline:
    """
    Multi-Stage Adaptive Inference Pipeline
    
    Progressively analyzes videos using three stages:
    1. Fast Inference: Low resolution, few frames
    2. Balanced Inference: Medium resolution, more frames
    3. Accurate Inference: High resolution, many frames
    
    Videos exit early if confidence is high enough.
    """
    
    def __init__(self, weights_path: Optional[str] = None, model_type: str = "Meso4"):
        """
        Initialize the adaptive pipeline
        
        Args:
            weights_path: Path to pretrained weights
            model_type: "Meso4" or "MesoInception4"
        """
        self.model_type: str = model_type
        self.weights_path: str = weights_path or WEIGHTS_PATH
        self.model: Optional[Union[Meso4, MesoInception4]] = None
        
        # Statistics tracking with explicit typing
        self.stats: Dict[str, Any] = {
            'total_videos': 0,
            'stage1_exits': 0,
            'stage2_exits': 0,
            'stage3_exits': 0,
            'total_time': 0.0,
            'stage_times': {1: 0.0, 2: 0.0, 3: 0.0}
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load the MesoNet model"""
        print(f"Loading {self.model_type} model...")
        
        if self.model_type == "Meso4":
            self.model = Meso4(input_shape=(256, 256, 3))
        elif self.model_type == "MesoInception4":
            self.model = MesoInception4(input_shape=(256, 256, 3))
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Ensure model is initialized before building
        model = self.model
        if model is None:
            raise RuntimeError(f"Failed to create {self.model_type} model")
        model.build()
        
        # Load weights if available
        if os.path.exists(self.weights_path):
            print(f"Loading weights from: {self.weights_path}")
            model.load_weights(self.weights_path)
            print("✓ Model loaded successfully!")
        else:
            print(f"⚠ Warning: Weights not found at {self.weights_path}")
            print("  Model will use random initialization (for demo purposes)")
    
    def _predict_frames(self, frames: np.ndarray) -> float:
        """
        Calculates image-level probabilities and aggregates them.
        Equation 3 & 4: p(s) = (1/|Ss|) * sum(pi)

        MesoNet Convention (Meso4_DF.h5):
            p_i close to 1.0 => frame is REAL
            p_i close to 0.0 => frame is DEEPFAKE

        Args:
            frames: numpy array of processed frames

        Returns:
            average_probability p(s)  [high = real, low = deepfake]
        """
        # Preprocess frames — MesoNet expects (256, 256, 3)
        target_resolution = (256, 256)
        processed_frames = preprocess_frames(frames, target_shape=target_resolution, normalize=True)

        # Batch inference
        model = self.model
        if model is None:
            raise RuntimeError("Model used before initialization")

        config_batch_size = int(PIPELINE_CONFIG.get('batch_size', 1))

        predictions = model.predict(
            processed_frames,
            batch_size=config_batch_size,
            verbose=0
        )

        # Aggregate predictions using arithmetic average (Equation 18)
        avg_probability = float(np.mean(predictions))

        return avg_probability
    
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
        # MesoNet convention: p_s >= 0.5 means REAL, p_s < 0.5 means DEEPFAKE
        # confidence_magnitude measures how far from the 0.5 decision boundary
        confidence_magnitude = max(p_s, 1 - p_s)

        stage_time = time.time() - stage_start

        # Early termination condition (Section IV.B: Equation 5)
        # Condition: max(p_s, 1 - p_s) >= tau_s
        tau_s = stage_config['confidence_threshold']
        should_exit = (confidence_magnitude >= tau_s) or (stage_number == 3)

        # Predicted Class (Equation 6)
        # HIGH p_s (>= 0.5) => REAL  |  LOW p_s (< 0.5) => DEEPFAKE
        label = "REAL" if p_s >= 0.5 else "DEEPFAKE"

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
        print(f"  p(s) = {f_p_s:.4f}  (high=REAL, low=DEEPFAKE)")
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
