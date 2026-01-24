"""
Multi-Stage Adaptive Inference Pipeline for Deepfake Detection
"""

import numpy as np
import time
from typing import Dict, List, Tuple
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mesonet import Meso4, MesoInception4
from pipeline.frame_extractor import FrameExtractor, preprocess_frames
from pipeline.config import *


class AdaptivePipeline:
    """
    Multi-Stage Adaptive Inference Pipeline
    
    Progressively analyzes videos using three stages:
    1. Fast Inference: Low resolution, few frames
    2. Balanced Inference: Medium resolution, more frames
    3. Accurate Inference: High resolution, many frames
    
    Videos exit early if confidence is high enough.
    """
    
    def __init__(self, weights_path: str = None, model_type: str = "Meso4"):
        """
        Initialize the adaptive pipeline
        
        Args:
            weights_path: Path to pretrained weights
            model_type: "Meso4" or "MesoInception4"
        """
        self.model_type = model_type
        self.weights_path = weights_path or WEIGHTS_PATH
        self.model = None
        
        # Statistics tracking
        self.stats = {
            'total_videos': 0,
            'stage1_exits': 0,
            'stage2_exits': 0,
            'stage3_exits': 0,
            'total_time': 0,
            'stage_times': {1: 0, 2: 0, 3: 0}
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
        
        self.model.build()
        
        # Load weights if available
        if os.path.exists(self.weights_path):
            print(f"Loading weights from: {self.weights_path}")
            self.model.load_weights(self.weights_path)
            print("✓ Model loaded successfully!")
        else:
            print(f"⚠ Warning: Weights not found at {self.weights_path}")
            print("  Model will use random initialization (for demo purposes)")
    
    def _predict_frames(self, frames: np.ndarray, stage_config: dict) -> Tuple[float, float]:
        """
        Predict on a batch of frames
        
        Args:
            frames: numpy array of frames
            stage_config: Configuration for current stage
            
        Returns:
            (average_probability, confidence)
        """
        # Preprocess frames
        processed_frames = preprocess_frames(frames, normalize=True)
        
        # Resize frames to model's expected input if needed
        if processed_frames.shape[1:3] != (256, 256):
            import cv2
            resized_frames = []
            for frame in processed_frames:
                resized = cv2.resize(frame, (256, 256))
                resized_frames.append(resized)
            processed_frames = np.array(resized_frames)
        
        # Make predictions
        predictions = self.model.predict(processed_frames, verbose=0)
        
        # Aggregate predictions
        avg_probability = np.mean(predictions)
        
        # Calculate confidence (distance from 0.5)
        confidence = abs(avg_probability - 0.5) * 2
        
        return float(avg_probability), float(confidence)
    
    def _process_stage(self, 
                       video_path: str, 
                       stage_number: int,
                       stage_config: dict) -> Dict:
        """
        Process a single stage
        
        Args:
            video_path: Path to video
            stage_number: Stage number (1, 2, or 3)
            stage_config: Stage configuration
            
        Returns:
            Dictionary with stage results
        """
        stage_start = time.time()
        
        if PIPELINE_CONFIG['verbose']:
            print(f"\n{'='*60}")
            print(f"Stage {stage_number}: {stage_config['name']}")
            print(f"{'='*60}")
            print(f"  Frames/sec: {stage_config['frames_per_second']}")
            print(f"  Resolution: {stage_config['resolution']}")
            print(f"  Threshold: {stage_config['confidence_threshold']}")
        
        # Extract frames
        with FrameExtractor(video_path) as extractor:
            frames = extractor.extract_frames_adaptive(stage_config)
            
            if PIPELINE_CONFIG['verbose']:
                print(f"  Extracted: {len(frames)} frames")
        
        # Make prediction
        probability, confidence = self._predict_frames(frames, stage_config)
        
        stage_time = time.time() - stage_start
        
        # Determine label
        label = "DEEPFAKE" if probability > CLASSIFICATION_CONFIG['deepfake_threshold'] else "REAL"
        
        if PIPELINE_CONFIG['verbose']:
            print(f"  Probability: {probability:.4f}")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  Prediction: {label}")
            print(f"  Time: {stage_time:.2f}s")
        
        # Check if we should exit this stage
        should_exit = confidence >= stage_config['confidence_threshold']
        
        if PIPELINE_CONFIG['verbose']:
            if should_exit:
                print(f"  ✓ High confidence - Exiting at Stage {stage_number}")
            else:
                print(f"  → Low confidence - Moving to next stage")
        
        return {
            'stage': stage_number,
            'probability': probability,
            'confidence': confidence,
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
        final_result = None
        exit_stage = None
        
        for stage_num in [1, 2, 3]:
            stage_config = get_stage_config(stage_num)
            result = self._process_stage(video_path, stage_num, stage_config)
            
            # Update statistics
            self.stats['stage_times'][stage_num] += result['time']
            
            # Check if we should exit
            if result['should_exit'] or stage_num == 3:
                final_result = result
                exit_stage = stage_num
                
                # Update exit statistics
                if stage_num == 1:
                    self.stats['stage1_exits'] += 1
                elif stage_num == 2:
                    self.stats['stage2_exits'] += 1
                else:
                    self.stats['stage3_exits'] += 1
                
                break
        
        total_time = time.time() - total_start
        
        # Update global statistics
        self.stats['total_videos'] += 1
        self.stats['total_time'] += total_time
        
        # Print final result
        print(f"\n{'='*60}")
        print(f"FINAL RESULT")
        print(f"{'='*60}")
        print(f"  Prediction: {final_result['label']}")
        print(f"  Probability: {final_result['probability']:.4f}")
        print(f"  Confidence: {final_result['confidence']:.4f}")
        print(f"  Exit Stage: {exit_stage}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"{'#'*70}\n")
        
        return {
            'video': video_path,
            'label': final_result['label'],
            'probability': final_result['probability'],
            'confidence': final_result['confidence'],
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
