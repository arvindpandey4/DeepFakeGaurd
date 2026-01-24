import os
import shutil
import uuid
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Add project root to path for imports
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from pipeline.adaptive_pipeline import AdaptivePipeline

app = FastAPI(title="Deepfake Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any origin for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pipeline (Global to avoid reloading weights every request)
WEIGHTS_PATH = os.path.join(project_root, "models", "weights", "Meso4_DF.h5")
print(f"Loading Pipeline with weights from: {WEIGHTS_PATH}")
pipeline = AdaptivePipeline(weights_path=WEIGHTS_PATH)

# Directory for uploads
UPLOAD_DIR = os.path.join(project_root, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount uploads so frontend can access processed videos if needed (optional)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.get("/")
def read_root():
    return {"status": "Deepfake Detection API is running"}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """
    Upload a video file and analyze it using the Multi-Stage Adaptive Pipeline.
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Generate unique filename
    file_id = str(uuid.uuid4())
    extension = os.path.splitext(file.filename)[1]
    filename = f"{file_id}{extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Save uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")

    try:
        # Run prediction
        # We assume pipeline.predict() is a synchronous, CPU-bound operation.
        # In a production app, we might offload this to a background task or thread pool,
        # but for this demo, running it directly is fine as FastAPI runs in a threadpool for def functions.
        result = pipeline.predict(file_path)
        
        # Add relative URL for the video
        result["video_url"] = f"/uploads/{filename}"
        result["filename"] = file.filename
        
        # Return complete results
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# --- Demo / Hugging Face Integration ---

CACHE_DIR = os.path.join(project_root, "hf_cache")

@app.get("/demo-videos")
def list_demo_videos():
    """List all videos currently available in the Hugging Face cache"""
    videos = []
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    if os.path.exists(CACHE_DIR):
        for root, dirs, files in os.walk(CACHE_DIR):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    full_path = os.path.join(root, file)
                    # Create a readable name (e.g., "deepfake/1.mp4")
                    rel_path = os.path.relpath(full_path, CACHE_DIR)
                    
                    videos.append({
                        "filename": file,
                        "path_id": rel_path,  # Use relative path as ID
                        "full_path": full_path,
                        "type": "DEEPFAKE" if "deepfake" in full_path.lower() else "REAL" # Naive guess based on folder
                    })
    
    return videos

from pydantic import BaseModel

class DemoRequest(BaseModel):
    path_id: str

@app.post("/analyze-demo")
def analyze_demo_video(request: DemoRequest):
    """Analyze a video already existing in the cache"""
    file_path = os.path.join(CACHE_DIR, request.path_id)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video not found in cache")
        
    try:
        # Run pipeline
        result = pipeline.predict(file_path)
        
        # We need to serve this file statically so frontend can play it
        # Create a symlink or copy to uploads dir so it's accessible via /uploads
        # Or better yet, serve hf_cache statically. Let's safe copy to uploads for simplicity.
        filename = os.path.basename(file_path)
        upload_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(upload_path):
            shutil.copy2(file_path, upload_path)
            
        result["video_url"] = f"/uploads/{filename}"
        result["filename"] = filename
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
