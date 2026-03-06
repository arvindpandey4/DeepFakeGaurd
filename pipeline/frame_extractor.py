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


# ---------------------------------------------------------------------------
# Face Detector (Option C) — OpenCV Haar Cascade, zero extra dependencies
# ---------------------------------------------------------------------------

class FaceDetector:
    """
    Lightweight face detector using OpenCV's built-in Haar cascade.
    Crops the face region from a frame, padding by `pad_ratio` on each side
    so the forehead / chin are fully visible — critical for MesoNet accuracy.

    Falls back gracefully to the full frame when no face is detected.
    """

    def __init__(self, pad_ratio: float = 0.20, min_face_ratio: float = 0.05):
        """
        Args:
            pad_ratio:       Fraction of the face bbox to add as padding (default 20%).
            min_face_ratio:  Minimum face area as fraction of frame area to accept detection.
        """
        self.pad_ratio = pad_ratio
        self.min_face_ratio = min_face_ratio
        self._cascade: Optional[Any] = None
        self._load_cascade()

    def _load_cascade(self) -> None:
        """Load the frontal-face Haar cascade bundled with OpenCV."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():  # type: ignore
            print("[FaceDetector] WARNING: Haar cascade failed to load — face crop disabled.")
            self._cascade = None

    def crop_face(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Detect the largest face in `frame_bgr` and return the padded crop.

        Args:
            frame_bgr: A single BGR frame as a numpy array (H, W, 3).

        Returns:
            Cropped face region (BGR) or the original frame if no face found.
        """
        if self._cascade is None:
            return frame_bgr

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        frame_area = float(h * w)

        faces = self._cascade.detectMultiScale(  # type: ignore
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if faces is None or (hasattr(faces, '__len__') and len(faces) == 0):
            return frame_bgr  # fallback: whole frame

        # Pick the largest detected face
        faces_arr = np.array(faces)
        areas = faces_arr[:, 2] * faces_arr[:, 3]
        best = faces_arr[int(np.argmax(areas))]
        fx, fy, fw, fh = int(best[0]), int(best[1]), int(best[2]), int(best[3])

        # Skip tiny detections (likely noise)
        if (fw * fh) / frame_area < self.min_face_ratio:
            return frame_bgr

        # Add padding
        pad_x = int(fw * self.pad_ratio)
        pad_y = int(fh * self.pad_ratio)
        x1 = max(0, fx - pad_x)
        y1 = max(0, fy - pad_y)
        x2 = min(w, fx + fw + pad_x)
        y2 = min(h, fy + fh + pad_y)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return frame_bgr  # guard against empty crop

        return crop


# Singleton detector — initialised once, reused across all calls
_face_detector = FaceDetector()


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
                      max_frames: Optional[int] = None,
                      use_face_crop: bool = True) -> np.ndarray:
        """
        Extract frames from video using optimized seeking
        
        Args:
            frames_per_second: Number of frames to extract per second
            resolution: Target resolution (width, height)
            max_frames: Maximum number of frames to extract
            use_face_crop: If True, detect and crop face region before resizing
            
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
        faces_found = 0
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, frame = cap.read()
            
            if not ret:
                continue

            # [Option C] Face crop: detect and crop the face region first
            if use_face_crop:
                cropped = _face_detector.crop_face(frame)
                if cropped is not frame:   # a face was found and cropped
                    faces_found += 1
                frame = cropped

            resized_frame = cv2.resize(frame, resolution)
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)

        if use_face_crop:
            print(f"  [FaceDetector] Faces detected in {faces_found}/{len(indices)} sampled frames")
            
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
        # Read face_crop setting from config (VIDEO_CONFIG propagates it via pipeline)
        from pipeline.config import VIDEO_CONFIG  # type: ignore
        use_face_crop: bool = bool(VIDEO_CONFIG.get("face_detection", True))
        return self.extract_frames(
            frames_per_second=stage_config['frames_per_second'],
            resolution=stage_config['resolution'],
            use_face_crop=use_face_crop
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
