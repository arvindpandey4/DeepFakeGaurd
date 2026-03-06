"""
Video Frame Extraction for Multi-Stage Pipeline
"""

import cv2 # type: ignore
import numpy as np # type: ignore
from typing import List, Tuple, Optional, Any
import os

try:
    import matplotlib.pyplot as plt # type: ignore
except ImportError:
    plt = None


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
        self.video_path: str = video_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 0.0
        self.total_frames: int = 0
        self.duration: float = 0.0
        
        self._initialize_video()
    
    def _initialize_video(self):
        """Open video and extract metadata"""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        
        # Explicit check for None to satisfy linter type narrowing
        cap = cv2.VideoCapture(self.video_path)
        
        if cap is None:
            raise ValueError(f"Failed to create VideoCapture object: {self.video_path}")
            
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        self.cap = cap
        self.fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = float(self.total_frames / self.fps) if self.fps > 0 else 0.0
    
    def extract_frames(self, 
                      frames_per_second: float = 1.0,
                      resolution: Tuple[int, int] = (256, 256),
                      max_frames: Optional[int] = None) -> np.ndarray:
        """
        Extract frames from video using optimized seeking
        
        Args:
            frames_per_second: Number of frames to extract per second
            resolution: Target resolution (width, height)
            max_frames: Maximum number of frames to extract
            
        Returns:
            numpy array of shape (num_frames, height, width, 3)
        """
        cap_init = self.cap
        if cap_init is None:
            self._initialize_video()
            cap_init = self.cap
        
        if cap_init is not None and not cap_init.isOpened():
            self._initialize_video()
            cap_init = self.cap
            
        # At this point self.cap is guaranteed initialized by _initialize_video or it raised
        cap = self.cap
        if cap is None:
             raise ValueError(f"Failed to initialize video capture: {self.video_path}")

        if self.total_frames <= 0:
            raise ValueError(f"Video has no frames: {self.video_path}")

        # Calculate total frames to extract
        num_to_extract = int(self.duration * frames_per_second)
        if num_to_extract < 1:
            num_to_extract = 1
            
        if max_frames:
            num_to_extract = min(num_to_extract, max_frames)
            
        # Calculate indices for uniform sampling (Section IV.C: Fast Stage)
        if num_to_extract > 1:
            indices = np.linspace(0, self.total_frames - 1, num_to_extract, dtype=int)
        else:
            indices = [self.total_frames // 2] 
            
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            resized_frame = cv2.resize(frame, resolution)
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            
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
        cap = self.cap
        if cap is None:
            return {}
        return {
            'path': self.video_path,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'duration': self.duration,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
    
    def close(self):
        """Release video capture"""
        cap = self.cap
        if cap is not None:
            cap.release()
            self.cap = None
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def preprocess_frames(frames: np.ndarray, 
                      target_shape: Optional[Tuple[int, int]] = None, 
                      normalize: bool = True) -> np.ndarray:
    """
    Preprocess frames for model input
    
    Args:
        frames: numpy array of frames (N, H, W, 3)
        target_shape: Optional (height, width) to resize to
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        Preprocessed frames
    """
    # Resize if needed
    if target_shape is not None:
        resized_frames = []
        for i in range(len(frames)):
            # cv2.resize expects (width, height)
            # target_shape is usually (H, W) for Keras, we swap to (W, H) for CV2
            resized_frames.append(cv2.resize(frames[i], (target_shape[1], target_shape[0])))
        frames = np.array(resized_frames)

    if normalize:
        frames = frames.astype(np.float32) / 255.0
    
    return frames


def visualize_frame_extraction(video_path: str, output_path: Optional[str] = None):
    """
    Visualize frame extraction at different stages
    
    Args:
        video_path: Path to video
        output_path: Path to save visualization
    """
    if plt is None:
        print("⚠ Matplotlib not found. Visualization disabled.")
        return
    
    with FrameExtractor(video_path) as extractor:
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
        
        # Use iterator to avoid indexing syntax which confuses the linter
        axes_iter = iter(axes.flatten())
        
        for stage_data in stages:
            frames_subset: Any = stage_data[0]
            title: str = str(stage_data[1])
            
            # Take at most 5 frames
            frames_to_show = frames_subset[:5] if len(frames_subset) > 5 else frames_subset
            
            for col_idx in range(5):
                try:
                    ax = next(axes_iter)
                    if col_idx < len(frames_to_show):
                        # Use index access on a casted Any to avoid 'Sized' error
                        # and assign to a local variable first
                        frame_array: Any = frames_to_show
                        current_frame = frame_array[col_idx]
                        ax.imshow(current_frame)
                        ax.axis('off')
                        if col_idx == 0:
                            ax.set_ylabel(title, fontsize=10)
                    else:
                        ax.axis('off')
                except StopIteration:
                    break
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        else:
            plt.show()


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
