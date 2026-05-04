import numpy as np
import cv2
import tensorflow as tf
from typing import List, Tuple, Optional, Any
import base64
from io import BytesIO
from PIL import Image


def get_last_conv_layer_name(model) -> str:
    for layer in reversed(model.model.layers):
        if 'conv' in layer.name.lower():
            return layer.name
    return "conv2d_3"


def generate_gradcam_heatmap(
    model,
    frames: np.ndarray,
    pred_scores: np.ndarray,
    last_conv_layer_name: Optional[str] = None,
    target_size: Tuple[int, int] = (128, 128)
) -> np.ndarray:
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)
    
    grad_model = tf.keras.models.Model(
        inputs=model.model.inputs,
        outputs=[
            model.model.get_layer(last_conv_layer_name).output,
            model.model.output
        ]
    )
    
    heatmaps = []
    
    for i in range(len(frames)):
        frame = frames[i:i+1]
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(frame)
            loss = predictions[0]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        for j in range(len(pooled_grads)):
            conv_outputs[:, :, j] *= pooled_grads[j]
        
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        heatmap = cv2.resize(heatmap, target_size)
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmaps.append(heatmap)
    
    return np.array(heatmaps)


def overlay_heatmap_on_frame(
    frame: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.6,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    if frame.max() <= 1.0:
        frame = (frame * 255).astype(np.uint8)
    else:
        frame = frame.astype(np.uint8)
    
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    
    if heatmap_colored.shape[:2] != frame.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, (frame.shape[1], frame.shape[0]))
    
    overlayed = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed


def frame_to_base64(frame: np.ndarray, format: str = 'JPEG', quality: int = 85) -> str:
    if frame.max() <= 1.0:
        frame = (frame * 255).astype(np.uint8)
    
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    img = Image.fromarray(frame)
    buffer = BytesIO()
    if format.upper() == 'JPEG':
        img.save(buffer, format=format, quality=quality, optimize=True)
    else:
        img.save(buffer, format=format)
    buffer.seek(0)
    
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    mime_type = f'image/{format.lower()}'
    return f'data:{mime_type};base64,{img_base64}'


def generate_explanation_for_frames(
    model,
    frames: np.ndarray,
    pred_scores: np.ndarray,
    original_frames: np.ndarray,
    top_k: int = 5
) -> List[dict]:
    top_indices = np.argsort(pred_scores)[-top_k:][::-1]
    top_frames = frames[top_indices]
    top_scores = pred_scores[top_indices]
    
    heatmaps = generate_gradcam_heatmap(model, top_frames, top_scores)
    explanations = []
    
    for i, idx in enumerate(top_indices):
        orig_frame = original_frames[idx]
        heatmap = heatmaps[i]
        
        overlay = overlay_heatmap_on_frame(orig_frame, heatmap, alpha=0.5)
        
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