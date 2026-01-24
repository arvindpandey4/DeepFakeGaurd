"""
Video Frame Extraction for Multi-Stage Pipeline
"""

import cv2
import numpy as np
from typing import List, Tuple
import os


class FrameExtractor:
    """
    Extract frames from video at different rates and resolutions
    for multi-stage adaptive inference
    """
    
    def __init__(self, video_path: str):
        """
        Initialize frame extractor
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.cap = None
        self.fps = None
        self.total_frames = None
        self.duration = None
        
        self._initialize_video()
    
    def _initialize_video(self):
        """Open video and extract metadata"""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
    
    def extract_frames(self, 
                      frames_per_second: float = 1.0,
                      resolution: Tuple[int, int] = (256, 256),
                      max_frames: int = None) -> np.ndarray:
        """
        Extract frames from video
        
        Args:
            frames_per_second: Number of frames to extract per second
            resolution: Target resolution (width, height)
            max_frames: Maximum number of frames to extract
            
        Returns:
            numpy array of shape (num_frames, height, width, 3)
        """
        if self.cap is None or not self.cap.isOpened():
            self._initialize_video()
        
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Calculate frame interval
        frame_interval = int(self.fps / frames_per_second) if frames_per_second > 0 else 1
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                # Resize frame
                resized_frame = cv2.resize(frame, resolution)
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                
                frames.append(rgb_frame)
                extracted_count += 1
                
                # Check if we've reached max frames
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video: {self.video_path}")
        
        return np.array(frames)
    
    def extract_frames_adaptive(self, stage_config: dict) -> np.ndarray:
        """
        Extract frames based on stage configuration
        
        Args:
            stage_config: Dictionary with 'frames_per_second' and 'resolution'
            
        Returns:
            numpy array of extracted frames
        """
        return self.extract_frames(
            frames_per_second=stage_config['frames_per_second'],
            resolution=stage_config['resolution']
        )
    
    def get_video_info(self) -> dict:
        """Get video metadata"""
        return {
            'path': self.video_path,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'duration': self.duration,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
    
    def close(self):
        """Release video capture"""
        if self.cap is not None:
            self.cap.release()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def preprocess_frames(frames: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Preprocess frames for model input
    
    Args:
        frames: numpy array of frames
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        Preprocessed frames
    """
    if normalize:
        frames = frames.astype(np.float32) / 255.0
    
    return frames


def visualize_frame_extraction(video_path: str, output_path: str = None):
    """
    Visualize frame extraction at different stages
    
    Args:
        video_path: Path to video
        output_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    extractor = FrameExtractor(video_path)
    
    # Extract frames at different rates
    stage1_frames = extractor.extract_frames(frames_per_second=1, resolution=(64, 64), max_frames=5)
    stage2_frames = extractor.extract_frames(frames_per_second=5, resolution=(128, 128), max_frames=5)
    stage3_frames = extractor.extract_frames(frames_per_second=10, resolution=(256, 256), max_frames=5)
    
    # Create visualization
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle('Frame Extraction at Different Stages', fontsize=16)
    
    stages = [
        (stage1_frames, "Stage 1: Fast (1 fps, 64x64)"),
        (stage2_frames, "Stage 2: Balanced (5 fps, 128x128)"),
        (stage3_frames, "Stage 3: Accurate (10 fps, 256x256)")
    ]
    
    for row, (frames, title) in enumerate(stages):
        for col in range(min(5, len(frames))):
            axes[row, col].imshow(frames[col])
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    extractor.close()


if __name__ == "__main__":
    # Test frame extraction
    print("Frame Extractor Module - Test Mode")
    print("=" * 60)
    
    # This would require an actual video file to test
    # Example usage:
    # extractor = FrameExtractor("sample_video.mp4")
    # info = extractor.get_video_info()
    # print(info)
    # frames = extractor.extract_frames(frames_per_second=1, resolution=(64, 64))
    # print(f"Extracted {len(frames)} frames")
    
    print("Module loaded successfully!")
    print("Use FrameExtractor class to extract frames from videos.")
