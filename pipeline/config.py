import os
from typing import Dict, List, Any, Tuple, Optional

MODEL_TYPE: str = "Meso4"
WEIGHTS_PATH: str = os.path.join("models", "weights", "Meso4_DF.h5")

STAGE1_CONFIG: Dict[str, Any] = {
    "name": "Fast Inference",
    "frames_per_second": 0.5,
    "resolution": (128, 128),
    "confidence_threshold": 0.85,
    "description": "Quick filtering of obvious cases using global sampling"
}

STAGE2_CONFIG: Dict[str, Any] = {
    "name": "Balanced Inference",
    "frames_per_second": 2,
    "resolution": (192, 192),
    "confidence_threshold": 0.75,
    "description": "Moderate analysis for uncertain cases"
}

STAGE3_CONFIG: Dict[str, Any] = {
    "name": "Accurate Inference",
    "frames_per_second": 5,
    "resolution": (256, 256),
    "confidence_threshold": 0.65,
    "description": "Thorough analysis for difficult cases"
}

PIPELINE_CONFIG: Dict[str, Any] = {
    "stages": [STAGE1_CONFIG, STAGE2_CONFIG, STAGE3_CONFIG],
    "aggregation_method": "arithmetic_average",
    "batch_size": 16,
    "verbose": True,
}

VIDEO_CONFIG: Dict[str, Any] = {
    "max_duration": 60,
    "face_detection": True,
    "normalize": True,
}

CLASSIFICATION_CONFIG: Dict[str, Any] = {
    "deepfake_threshold": 0.5,
    "labels": {
        0: "REAL",
        1: "DEEPFAKE"
    }
}

METRICS_CONFIG: Dict[str, Any] = {
    "track_time": True,
    "track_confidence": True,
    "save_results": True,
    "results_dir": "results",
}

def get_stage_config(stage_number):
    if stage_number == 1:
        return STAGE1_CONFIG
    elif stage_number == 2:
        return STAGE2_CONFIG
    elif stage_number == 3:
        return STAGE3_CONFIG
    else:
        raise ValueError(f"Invalid stage number: {stage_number}. Must be 1, 2, or 3.")

def print_config():
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