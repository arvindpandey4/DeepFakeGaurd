"""
Configuration for Multi-Stage Adaptive Inference Pipeline
"""

import os

from typing import Dict, List, Any, Tuple, Optional

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_TYPE: str = "Meso4"  # Options: "Meso4", "MesoInception4"
WEIGHTS_PATH: str = os.path.join("models", "weights", "Meso4_DF.h5")

# ============================================================================
# STAGE 1: FAST INFERENCE (n1 << N, r1 < 1)
# ============================================================================
# Goal: Quickly filter obvious deepfakes/real videos
# Strategy: Minimal frames, reduced resolution

STAGE1_CONFIG: Dict[str, Any] = {
    "name": "Fast Inference",
    "frames_per_second": 0.5,         # Extract 1 frame every 2 seconds (n1)
    "resolution": (128, 128),         # Scale factor r1 = 0.5 (native is 256)
    "confidence_threshold": 0.85,     # [TUNED] Lowered from 0.90 → 0.85 to reduce false early exits
    "description": "Quick filtering of obvious cases using global sampling"
}

# ============================================================================
# STAGE 2: BALANCED INFERENCE (n1 < n2 < N, r1 < r2 < 1)
# ============================================================================
# Goal: Handle moderately difficult videos
# Strategy: More frames, medium resolution

STAGE2_CONFIG: Dict[str, Any] = {
    "name": "Balanced Inference",
    "frames_per_second": 2,           # Extract 2 frames per second (n2)
    "resolution": (192, 192),         # Scale factor r2 = 0.75
    "confidence_threshold": 0.65,     # [TUNED] Lowered from 0.80 → 0.65 to force hard cases into Stage 3
    "description": "Moderate analysis for uncertain cases"
}

# ============================================================================
# STAGE 3: ACCURATE INFERENCE (n3 ≈ N, r3 ≈ 1)
# ============================================================================
# Goal: Thorough analysis of hardest videos
# Strategy: High density frames, full processing resolution

STAGE3_CONFIG: Dict[str, Any] = {
    "name": "Accurate Inference",
    "frames_per_second": 5,           # Extract 5 frames per second (n3)
    "resolution": (256, 256),         # Scale factor r3 = 1.0 (Full Native)
    "confidence_threshold": 0.0,      # Always make final decision at terminal stage
    "description": "Thorough analysis for difficult cases"
}

# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================

PIPELINE_CONFIG: Dict[str, Any] = {
    "stages": [STAGE1_CONFIG, STAGE2_CONFIG, STAGE3_CONFIG],
    "aggregation_method": "arithmetic_average", # Strict adherence to Equation 18
    "batch_size": 16,                 # Optimized for CPU-limited environments
    "verbose": True,                  # Print stage information
}

# ============================================================================
# VIDEO PROCESSING
# ============================================================================

VIDEO_CONFIG: Dict[str, Any] = {
    "max_duration": 60,               # Maximum video duration to process (seconds)
    "face_detection": True,           # [ENABLED] Face crop preprocessing via OpenCV Haar/DNN cascade
    "normalize": True,                # Normalize pixel values to [0, 1]
}

# ============================================================================
# CLASSIFICATION THRESHOLDS
# ============================================================================

CLASSIFICATION_CONFIG: Dict[str, Any] = {
    "deepfake_threshold": 0.5,        # Probability threshold for deepfake classification
    "labels": {
        0: "REAL",
        1: "DEEPFAKE"
    }
}

# ============================================================================
# PERFORMANCE TRACKING
# ============================================================================

METRICS_CONFIG: Dict[str, Any] = {
    "track_time": True,               # Track processing time per stage
    "track_confidence": True,         # Track confidence scores
    "save_results": True,             # Save results to file
    "results_dir": "results",         # Directory to save results
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_stage_config(stage_number):
    """Get configuration for a specific stage (1, 2, or 3)"""
    if stage_number == 1:
        return STAGE1_CONFIG
    elif stage_number == 2:
        return STAGE2_CONFIG
    elif stage_number == 3:
        return STAGE3_CONFIG
    else:
        raise ValueError(f"Invalid stage number: {stage_number}. Must be 1, 2, or 3.")

def print_config():
    """Print current configuration"""
    print("=" * 70)
    print("MULTI-STAGE ADAPTIVE INFERENCE PIPELINE - CONFIGURATION")
    print("=" * 70)
    
    print(f"\nModel: {MODEL_TYPE}")
    print(f"Weights: {WEIGHTS_PATH}")
    
    stages_list: List[Dict[str, Any]] = PIPELINE_CONFIG["stages"]
    for i, stage in enumerate(stages_list, 1):
        print(f"\n--- Stage {i}: {stage['name']} ---")
        print(f"  Frames/sec: {stage['frames_per_second']}")
        print(f"  Resolution: {stage['resolution']}")
        print(f"  Confidence Threshold: {stage['confidence_threshold']}")
        print(f"  Description: {stage['description']}")
    
    print(f"\nAggregation Method: {PIPELINE_CONFIG['aggregation_method']}")
    print(f"Batch Size: {PIPELINE_CONFIG['batch_size']}")
    print(f"Deepfake Threshold: {CLASSIFICATION_CONFIG['deepfake_threshold']}")
    
    print("=" * 70)


if __name__ == "__main__":
    print_config()
