# Deepfake Detective - Web Interface 🕵️‍♂️💻

The project now includes a stunning, high-performance web dashboard to visualize the Multi-Stage Adaptive Inference Pipeline.

## 🚀 Quick Start

### 1. Start the Backend (API)
This serves the AI model.
```bash
# In your main project folder (with venv activated)
python api.py
```
*Server will start at: http://localhost:8000*

### 2. Start the Frontend (UI)
This launches the glossy dashboard.
```bash
# Open a NEW terminal
cd frontend
npm run dev
```
*Dashboard will run at: http://localhost:5173*

## ✨ Features
- **Real-time Visualization**: Watch the "scanning" effect as the model processes frames.
- **Stage Breakdown**: See exactly which stage (1, 2, or 3) the video exits at.
- **Detailed Metrics**: View probabilities, confidence scores, and processing time.
- **Glassmorphism UI**: High-end, futuristic design with fluid animations.

## 🛠️ Architecture
- **Backend**: FastAPI (Python) running the `AdaptivePipeline`.
- **Frontend**: React + Vite + Tailwind CSS + Framer Motion.
- **Integration**: The frontend sends the video file to `/analyze`, and the backend returns the JSON result of the pipeline execution.
