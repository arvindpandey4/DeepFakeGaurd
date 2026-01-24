# Multi-Stage Adaptive Inference Pipeline for Deepfake Video Detection

## 🎯 Project Overview

This project implements an **efficient, adaptive deepfake detection system** that progressively analyzes videos using multiple stages, applying expensive analysis only when necessary.

### The Problem
Traditional deepfake detection systems process every video with the same computational intensity, wasting resources on obvious cases.

### Our Solution
A **Multi-Stage Adaptive Inference Pipeline** that:
- ✅ Quickly filters obvious deepfakes/real videos (Stage 1)
- ✅ Applies moderate analysis to uncertain cases (Stage 2)
- ✅ Uses thorough analysis only for the hardest videos (Stage 3)

## 🔑 Key Innovation

**We don't use three different models.**  
We use **ONE pretrained MesoNet model** with varying:
- Number of frames processed
- Input resolution
- Computation intensity

## 🏗️ Architecture

```
Input Video
    ↓
Stage 1: Fast Inference (Low res, few frames)
    ↓ (only if uncertain)
Stage 2: Balanced Inference (Medium res, more frames)
    ↓ (only if still uncertain)
Stage 3: Accurate Inference (High res, many frames)
    ↓
Final Decision (Deepfake / Real)
```

## 🚀 Features

- **MesoNet (Meso4)** - Lightweight CNN for deepfake detection
- **CPU-friendly** - Runs efficiently on standard hardware
- **Pretrained weights** - Ready to use out of the box
- **Adaptive processing** - Smart resource allocation
- **Performance metrics** - Track time savings and accuracy

## 📊 Expected Performance

For 100 videos:
- ~60% exit at Stage 1 (Fast)
- ~30% exit at Stage 2 (Balanced)
- ~10% reach Stage 3 (Accurate)

**Result**: Significant time savings without sacrificing accuracy

## 🛠️ Technology Stack

- **Python 3.8+**
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Video processing
- **NumPy** - Numerical operations
- **Matplotlib** - Visualization

## 📁 Project Structure

```
Major Project/
├── models/
│   ├── mesonet.py              # MesoNet architecture
│   └── weights/
│       └── Meso4_DF.h5         # Pretrained weights
├── pipeline/
│   ├── adaptive_pipeline.py    # Multi-stage pipeline
│   ├── frame_extractor.py      # Video frame extraction
│   └── config.py               # Configuration settings
├── utils/
│   ├── metrics.py              # Performance tracking
│   └── visualization.py        # Results visualization
├── demo.py                     # Main demo script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🎬 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Pretrained Weights
The MesoNet weights will be automatically downloaded on first run, or manually download from:
- [DariusAf/MesoNet Repository](https://github.com/DariusAf/MesoNet)

### 3. Run Demo
```bash
python demo.py --video path/to/video.mp4
```

Or test with sample videos:
```bash
python demo.py --demo
```

## 📖 Usage Examples

### Single Video Analysis
```python
from pipeline.adaptive_pipeline import AdaptivePipeline

pipeline = AdaptivePipeline()
result = pipeline.predict("video.mp4")

print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Exit Stage: {result['stage']}")
print(f"Processing Time: {result['time']:.2f}s")
```

### Batch Processing
```python
videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
results = pipeline.predict_batch(videos)

# View statistics
pipeline.print_statistics()
```

## 🎓 What This Project Demonstrates

### ✅ We DO claim:
- Efficient, practical deepfake detection
- Adaptive inference scheduling
- Significant computational savings
- System-level optimization

### ❌ We DON'T claim:
- Real-time live detection
- New deepfake detection model
- Model training/fine-tuning
- State-of-the-art accuracy

## 📈 Configuration

Edit `pipeline/config.py` to customize:

```python
# Stage 1: Fast Inference
STAGE1_FRAMES_PER_SEC = 1
STAGE1_RESOLUTION = (64, 64)
STAGE1_CONFIDENCE_THRESHOLD = 0.85

# Stage 2: Balanced Inference
STAGE2_FRAMES_PER_SEC = 5
STAGE2_RESOLUTION = (128, 128)
STAGE2_CONFIDENCE_THRESHOLD = 0.75

# Stage 3: Accurate Inference
STAGE3_FRAMES_PER_SEC = 10
STAGE3_RESOLUTION = (256, 256)
```

## 🔬 Model Details

**MesoNet (Meso4)**
- Designed specifically for deepfake detection
- Trained on FaceForensics++ dataset
- Lightweight architecture (< 1MB)
- Fast inference on CPU
- Focus on mesoscopic properties of images

## 📊 Performance Metrics

The system tracks:
- **Processing time per stage**
- **Exit distribution** (% videos per stage)
- **Average confidence scores**
- **Total time savings** vs. full processing
- **Accuracy** (when ground truth available)

## 🤝 Contributing

This is a demo project for educational purposes. Feel free to:
- Experiment with different thresholds
- Try other lightweight models
- Add more stages
- Implement face detection preprocessing

## 📚 References

- **MesoNet Paper**: "MesoNet: a Compact Facial Video Forgery Detection Network"
- **Repository**: [DariusAf/MesoNet](https://github.com/DariusAf/MesoNet)
- **Dataset**: FaceForensics++

## 📝 License

This project is for educational and research purposes.

## 👨‍💻 Author

Arvind - Major Project Demo

---

**Note**: This is a demonstration of adaptive inference scheduling for deepfake detection, not a production-ready system.
