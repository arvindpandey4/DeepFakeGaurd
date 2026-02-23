# 🛡️ DeepFakeGuard
### Multi-Stage Adaptive Inference Pipeline for Efficient Deepfake Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.95.0+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-v18.0+-61DAFB.svg)](https://reactjs.org/)
[![IEEE](https://img.shields.io/badge/Research-IEEE%20Format-red.svg)](#-research--visualization)

DeepFakeGuard is an **efficient, adaptive deepfake detection system** that intelligently optimizes computational resources. By using a multi-stage approach, it analyzes videos with increasing depth only when necessary, significantly reducing processing time without compromising accuracy.

---

## 📖 Table of Contents
- [🎯 Project Overview](#-project-overview)
- [🏗️ Multi-Stage Architecture](#️-multi-stage-architecture)
- [✨ Key Features](#-key-features)
- [🖥️ Web Dashboard](#️-web-dashboard)
- [🔬 Research & Visualization](#-research--visualization)
- [📁 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [🛠️ Tech Stack](#️-tech-stack)
- [👨‍💻 Author](#-author)

---

## 🎯 Project Overview

### The Problem
Traditional deepfake detection systems process every video with the same maximum computational intensity. This wastes significant CPU/GPU resources on "obvious" cases (both real and fake) that could be identified with much lighter analysis.

### Our Solution: Adaptive Inference
DeepFakeGuard implements a **Multi-Stage Adaptive Inference Pipeline** that:
1.  **Fast Exit**: Quickly filters obvious videos in Stage 1.
2.  **Moderate Analysis**: Applies balanced processing to uncertain cases in Stage 2.
3.  **Deep Inspection**: Uses thorough analysis only for the most challenging videos in Stage 3.

---

## 🏗️ Multi-Stage Architecture

We utilize a single **MesoNet (Meso4)** model but vary the execution intensity across stages:

| Stage | Intensity | Resolution | Frames/Sec | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | ⚡ Fast | 64x64 | 1 FPS | Optimized for high-confidence, obvious cases. |
| **Stage 2** | ⚖️ Balanced | 128x128 | 5 FPS | Medium intensity for ambiguous results. |
| **Stage 3** | 🔍 Accurate | 256x256 | 10 FPS | Full-depth analysis for maximum precision. |

---

## ✨ Key Features

- **Resource Efficient**: Up to **60% faster** processing by exiting early on obvious videos.
- **Glassmorphism UI**: A stunning, modern web dashboard for real-time visualization.
- **MesoNet Core**: Leveraging a lightweight CNN specifically designed for facial forgery.
- **CPU Optimized**: Designed to run efficiently without requiring high-end GPUs.
- **Detailed Metrics**: Real-time tracking of exit stages, confidence scores, and processing time.

---

## 🖥️ Web Dashboard

The project includes a futuristic, high-performance web interface to interact with the pipeline.

### 🚀 Starting the Web Interface

#### 1. Start the Backend API
```bash
# In the root directory (with venv activated)
python api.py
```
*API will be available at: `http://localhost:8000`*

#### 2. Start the Frontend UI
```bash
cd frontend
npm install  # If running for the first time
npm run dev
```
*Dashboard will be available at: `http://localhost:5173`*

---

## 🔬 Research & Visualization

For academic and research purposes, we have included tools to generate IEEE-standard visualizations of the pipeline's performance.

- **Location**: `ieee_paper_files/`
- **Capability**: Generates "Mean Frames Processed per Video" bar charts comparing the Adaptive approach vs. a Fixed baseline.
- **Usage**:
  ```bash
  python ieee_paper_files/generate_graph_ieee.py
  ```

---

## 📁 Project Structure

```text
DeepFakeGuard/
├── ieee_paper_files/       # 📊 IEEE standard research graphs & scripts
├── frontend/               # 💻 React + Vite + Tailwind Dashboard
├── pipeline/               # ⚙️ Core Adaptive Pipeline logic
│   ├── adaptive_pipeline.py
│   ├── frame_extractor.py
│   └── config.py
├── models/                 # 🧠 CNN Architecture & Weights
├── utils/                  # 🛠️ Helper functions & Metrics
├── api.py                  # 🚀 FastAPI Backend
├── demo.py                 # 🐍 CLI Selection/Demo script
├── requirements.txt        # 📦 Python Dependencies
└── README.md               # 📖 Main Documentation
```

---

## 🚀 Quick Start (CLI Only)

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Sample Analysis**:
    ```bash
    python demo.py --demo
    ```

---

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, TensorFlow/Keras, OpenCV
- **Frontend**: React.js, Vite, Tailwind CSS, Framer Motion
- **Visualization**: Matplotlib, NumPy
- **Styling**: Modern Glassmorphism & Micro-animations

---

## 👨‍💻 Author

**Arvind**  
*Major Project - DeepFakeGuard*

---
> **Disclaimer**: This is a research demonstration project focusing on adaptive inference scheduling for deepfake detection.
