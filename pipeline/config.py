"""
Configuration for Multi-Stage Adaptive Inference Pipeline
"""

import os

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_TYPE = "Meso4"  # Options: "Meso4", "MesoInception4"
WEIGHTS_PATH = os.path.join("models", "weights", "Meso4_DF.h5")

# ============================================================================
# STAGE 1: FAST INFERENCE
# ============================================================================
# Goal: Quickly filter obvious deepfakes/real videos
# Strategy: Minimal frames, low resolution

STAGE1_CONFIG = {
    "name": "Fast Inference",
    "frames_per_second": 1,          # Extract 1 frame per second
    "resolution": (64, 64),           # Low resolution for speed
    "confidence_threshold": 0.85,     # High confidence to exit early
    "description": "Quick filtering of obvious cases"
}

# ============================================================================
# STAGE 2: BALANCED INFERENCE
# ============================================================================
# Goal: Handle moderately difficult videos
# Strategy: More frames, medium resolution

STAGE2_CONFIG = {
    "name": "Balanced Inference",
    "frames_per_second": 5,           # Extract 5 frames per second
    "resolution": (128, 128),         # Medium resolution
    "confidence_threshold": 0.75,     # Moderate confidence threshold
    "description": "Moderate analysis for uncertain cases"
}

# ============================================================================
# STAGE 3: ACCURATE INFERENCE
# ============================================================================
# Goal: Thorough analysis of hardest videos
# Strategy: Many frames, high resolution

STAGE3_CONFIG = {
    "name": "Accurate Inference",
    "frames_per_second": 10,          # Extract 10 frames per second
    "resolution": (256, 256),         # High resolution (model's native)
    "confidence_threshold": 0.0,      # Always make final decision
    "description": "Thorough analysis for difficult cases"
}

# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================

PIPELINE_CONFIG = {
    "stages": [STAGE1_CONFIG, STAGE2_CONFIG, STAGE3_CONFIG],
    "aggregation_method": "average",  # How to combine frame predictions: "average", "max", "voting"
    "batch_size": 32,                 # Batch size for model inference
    "verbose": True,                  # Print stage information
}

# ============================================================================
# VIDEO PROCESSING
# ============================================================================

VIDEO_CONFIG = {
    "max_duration": 60,               # Maximum video duration to process (seconds)
    "face_detection": False,          # Enable face detection preprocessing (requires additional setup)
    "normalize": True,                # Normalize pixel values to [0, 1]
}

# ============================================================================
# CLASSIFICATION THRESHOLDS
# ============================================================================

CLASSIFICATION_CONFIG = {
    "deepfake_threshold": 0.5,        # Probability threshold for deepfake classification
    "labels": {
        0: "REAL",
        1: "DEEPFAKE"
    }
}

# ============================================================================
# PERFORMANCE TRACKING
# ============================================================================

METRICS_CONFIG = {
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
    
    for i, stage in enumerate(PIPELINE_CONFIG["stages"], 1):
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
