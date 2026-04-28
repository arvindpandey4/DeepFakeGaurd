"""
Grad-CAM Explainability for MesoNet
Generates heatmaps showing which pixels contributed to the deepfake decision
"""

import numpy as np  # type: ignore
import cv2  # type: ignore
import tensorflow as tf  # type: ignore
from typing import List, Tuple, Optional, Any
import base64
from io import BytesIO
from PIL import Image  # type: ignore


def get_last_conv_layer_name(model) -> str:
    """
    Automatically find the last convolutional layer in the model
    
    Args:
        model: Keras model (Meso4 or MesoInception4)
        
    Returns:
        Name of the last Conv2D layer
    """
    for layer in reversed(model.model.layers):  # type: ignore
        if 'conv' in layer.name.lower():
            return layer.name
    # Fallback for Meso4
    return "conv2d_3"


def generate_gradcam_heatmap(
    model,
    frames: np.ndarray,
    pred_scores: np.ndarray,
    last_conv_layer_name: Optional[str] = None,
    target_size: Tuple[int, int] = (128, 128)  # Reduced from 256x256 for speed
) -> np.ndarray:
    """
    Generate Grad-CAM heatmaps for a batch of frames
    
    Args:
        model: MesoNet model (Meso4 or MesoInception4)
        frames: Preprocessed frames (N, H, W, 3) in [0, 1] range
        pred_scores: Prediction scores for each frame (N,)
        last_conv_layer_name: Name of last conv layer (auto-detected if None)
        
    Returns:
        Heatmaps as numpy array (N, H, W) in [0, 255] range
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)
    
    # Create a model that outputs both predictions and conv layer activations
    grad_model = tf.keras.models.Model(
        inputs=model.model.inputs,  # type: ignore
        outputs=[
            model.model.get_layer(last_conv_layer_name).output,  # type: ignore
            model.model.output  # type: ignore
        ]
    )
    
    heatmaps = []
    
    for i in range(len(frames)):
        frame = frames[i:i+1]  # Keep batch dimension
        
        # Compute gradient of output w.r.t. last conv layer
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(frame)
            # Use the actual prediction as the target
            loss = predictions[0]
        
        # Gradient of the output w.r.t. the conv layer output
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients (importance weights)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv outputs by the gradients
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        # Weighted combination of feature maps
        for j in range(len(pooled_grads)):
            conv_outputs[:, :, j] *= pooled_grads[j]
        
        # Average across all feature maps
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # Normalize to [0, 1]
        heatmap = np.maximum(heatmap, 0)  # ReLU
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Resize to target size (faster processing)
        heatmap = cv2.resize(heatmap, target_size)
        
        # Scale to [0, 255]
        heatmap = (heatmap * 255).astype(np.uint8)
        
        heatmaps.append(heatmap)
    
    return np.array(heatmaps)


def overlay_heatmap_on_frame(
    frame: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.6,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay a heatmap on the original frame
    
    Args:
        frame: Original frame (H, W, 3) in [0, 1] or [0, 255] range
        heatmap: Heatmap (H, W) in [0, 255] range
        alpha: Transparency of heatmap overlay (0-1)
        colormap: OpenCV colormap to use
        
    Returns:
        Overlayed image (H, W, 3) in [0, 255] range
    """
    # Ensure frame is uint8
    if frame.max() <= 1.0:
        frame = (frame * 255).astype(np.uint8)
    else:
        frame = frame.astype(np.uint8)
    
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    
    # Resize heatmap to match frame if needed
    if heatmap_colored.shape[:2] != frame.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, (frame.shape[1], frame.shape[0]))
    
    # Overlay
    overlayed = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed


def frame_to_base64(frame: np.ndarray, format: str = 'JPEG', quality: int = 85) -> str:
    """
    Convert a frame to base64-encoded image string
    
    Args:
        frame: Image array (H, W, 3)
        format: Image format (PNG, JPEG) - JPEG is faster
        quality: JPEG quality (85 = good balance of size/quality)
        
    Returns:
        Base64 string with data URI prefix
    """
    # Convert to PIL Image
    if frame.max() <= 1.0:
        frame = (frame * 255).astype(np.uint8)
    
    # Convert BGR to RGB if needed (OpenCV uses BGR)
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    img = Image.fromarray(frame)
    
    # Save to bytes buffer
    buffer = BytesIO()
    if format.upper() == 'JPEG':
        img.save(buffer, format=format, quality=quality, optimize=True)
    else:
        img.save(buffer, format=format)
    buffer.seek(0)
    
    # Encode to base64
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    # Add data URI prefix
    mime_type = f'image/{format.lower()}'
    return f'data:{mime_type};base64,{img_base64}'


def generate_explanation_for_frames(
    model,
    frames: np.ndarray,
    pred_scores: np.ndarray,
    original_frames: np.ndarray,
    top_k: int = 5
) -> List[dict]:
    """
    Generate Grad-CAM explanations for the top-k most suspicious frames
    
    Args:
        model: MesoNet model
        frames: Preprocessed frames (N, H, W, 3) in [0, 1]
        pred_scores: Prediction scores (N,) - deepfake probability
        original_frames: Original frames before preprocessing (N, H, W, 3)
        top_k: Number of top suspicious frames to explain
        
    Returns:
        List of dicts with frame_idx, score, heatmap_base64, overlay_base64
    """
    # Find top-k most suspicious frames (highest deepfake probability)
    top_indices = np.argsort(pred_scores)[-top_k:][::-1]
    
    # Generate heatmaps for top frames
    top_frames = frames[top_indices]
    top_scores = pred_scores[top_indices]
    
    heatmaps = generate_gradcam_heatmap(model, top_frames, top_scores)
    
    explanations = []
    
    for i, idx in enumerate(top_indices):
        # Get original frame for overlay
        orig_frame = original_frames[idx]
        heatmap = heatmaps[i]
        
        # Create overlay
        overlay = overlay_heatmap_on_frame(orig_frame, heatmap, alpha=0.5)
        
        # Convert to base64 (use JPEG for smaller size/faster transfer)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_base64 = frame_to_base64(heatmap_colored, format='JPEG', quality=90)
        overlay_base64 = frame_to_base64(overlay, format='JPEG', quality=90)
        
        explanations.append({
            'frame_idx': int(idx),
            'score': float(pred_scores[idx]),
            'heatmap': heatmap_base64,
            'overlay': overlay_base64,
            'description': f"Frame {idx}: {pred_scores[idx]:.2%} deepfake probability"
        })
    
    return explanations


if __name__ == "__main__":
    print("Explainability Module - Grad-CAM Implementation")
    print("=" * 60)
    print("This module provides visual explanations for MesoNet predictions")
    print("using Gradient-weighted Class Activation Mapping (Grad-CAM)")
