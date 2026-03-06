import os
import shutil
import uuid
import time
import threading
import itertools
import urllib.parse
import requests  # type: ignore
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from pydantic import BaseModel  # type: ignore
from typing import List, Dict, Optional, Any
import sys
from huggingface_hub import hf_hub_download, list_repo_files, hf_hub_url  # type: ignore

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from pipeline.adaptive_pipeline import AdaptivePipeline # type: ignore

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

# --- Dynamic Hugging Face Integration ---

UNIDATAPRO_REPO = "UniDataPro/deepfake-videos-dataset"
ALT_DEEPFAKE_REPO = "DGSpitzer/Cyberpunk-Anime-Diffusion"  # fallback if UniDataPro has issues
CACHE_DIR = os.path.join(project_root, "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ── In-memory cache so the frontend never waits 30s ──────────────────────────
_video_cache: List[Dict[str, Any]] = []
_cache_lock = threading.Lock()
_cache_ready = False


def _fetch_videos_background():
    """Runs once in a background thread at startup (and on manual sync)."""
    global _video_cache, _cache_ready
    print("[HF] Background fetch started...")
    t0 = time.time()
    results: List[Dict[str, Any]] = []

    # 1. REAL videos from Kinetics400 (5 samples)
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("liuhuanjim013/kinetics400", split="train", streaming=True)
        for item in itertools.islice(ds, 5):
            clip_name = item.get("clip_name", "kinetics_video")
            if not clip_name.endswith('.mp4'):
                clip_name += '.mp4'
            url = item.get("video link", "")
            hf_path = None
            if url and "resolve/main/" in url:
                hf_path = urllib.parse.unquote(url.split("resolve/main/")[-1])
            results.append({
                "id": f"kinetics400_{clip_name.replace('.', '_')}",
                "repo_id": "liuhuanjim013/kinetics400" if hf_path else None,
                "name": clip_name,
                "hf_path": hf_path,
                "url": url,
                "type": "REAL",
                "description": f"Real: {item.get('action_class', 'Kinetics400')}"
            })
        print(f"[HF] Got {sum(1 for v in results if v['type']=='REAL')} REAL videos")
    except Exception as e:
        print(f"[HF] Error fetching Kinetics400: {e}")

    # 2. DEEPFAKE videos — Primary: UniDataPro (face-swap MP4s)
    deepfake_count = 0
    try:
        files = list(list_repo_files(UNIDATAPRO_REPO, repo_type="dataset"))
        video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
        fake_files = [
            f for f in files
            if isinstance(f, str)
            and f.startswith("deepfake/")
            and f.lower().endswith(video_extensions)
        ][:5]
        for f in fake_files:
            fname = os.path.basename(f)
            results.append({
                "id": f"unidatapro_{fname.replace('.', '_')}",
                "repo_id": UNIDATAPRO_REPO,
                "name": fname,
                "hf_path": f,
                "url": hf_hub_url(repo_id=UNIDATAPRO_REPO, filename=f, repo_type="dataset"),
                "type": "DEEPFAKE",
                "description": f"Deepfake: {fname} (UniDataPro face-swap)"
            })
            deepfake_count += 1
        print(f"[HF] Got {deepfake_count} DEEPFAKE videos from UniDataPro")
    except Exception as e:
        print(f"[HF] Error fetching UniDataPro: {e}")

    # 2b. DEEPFAKE videos — Secondary fallback: TraoreIbrahim/deepfake_face_videos_dataset
    if deepfake_count < 3:
        try:
            FALLBACK_REPO = "TraoreIbrahim/deepfake_face_videos_dataset"
            fallback_files = list(list_repo_files(FALLBACK_REPO, repo_type="dataset"))
            video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
            fallback_fakes = [
                f for f in fallback_files
                if isinstance(f, str) and f.lower().endswith(video_extensions)
            ][: (5 - deepfake_count)]
            for f in fallback_fakes:
                fname = os.path.basename(f)
                results.append({
                    "id": f"fallback_{fname.replace('.', '_')}",
                    "repo_id": FALLBACK_REPO,
                    "name": fname,
                    "hf_path": f,
                    "url": hf_hub_url(repo_id=FALLBACK_REPO, filename=f, repo_type="dataset"),
                    "type": "DEEPFAKE",
                    "description": f"Deepfake: {fname} (fallback dataset)"
                })
            print(f"[HF] Fallback: added {len(fallback_fakes)} extra DEEPFAKE videos")
        except Exception as e:
            print(f"[HF] Fallback dataset also failed: {e}")

    with _cache_lock:
        _video_cache = results
        _cache_ready = True
    print(f"[HF] Cache ready — {len(results)} videos in {time.time()-t0:.1f}s")


# Kick off background fetch immediately at import time
_bg_thread = threading.Thread(target=_fetch_videos_background, daemon=True)
_bg_thread.start()


@app.get("/available-remote-videos")
def list_available_remote_videos():
    """Returns the cached video list instantly. 'is_ready' tells the UI if the cache has loaded."""
    with _cache_lock:
        videos = list(_video_cache)
        ready = _cache_ready

    results = []
    for video in videos:
        local_name = f"{video['type'].lower()}_{video['name']}"
        video_path = os.path.join(CACHE_DIR, local_name)
        video_copy: Any = video.copy()
        video_copy["local_name"] = local_name
        video_copy["is_downloaded"] = os.path.exists(video_path) and os.path.getsize(video_path) > 0
        results.append(video_copy)

    return {"is_ready": ready, "videos": results}


@app.post("/sync-remote-videos")
def sync_remote_videos():
    """Manually trigger a background re-fetch of the HF video list."""
    global _cache_ready
    with _cache_lock:
        _cache_ready = False
    t = threading.Thread(target=_fetch_videos_background, daemon=True)
    t.start()
    return {"status": "sync started"}

class DownloadRequest(BaseModel):
    video_id: str

@app.post("/download-remote-video")
async def download_remote_video(request: DownloadRequest):
    """Download a video from HF Hub to the local cache"""
    with _cache_lock:
        all_videos = list(_video_cache)
    video = next((v for v in all_videos if v["id"] == request.video_id), None)
    
    if not video:
        raise HTTPException(status_code=404, detail="Video configuration not found")
    
    local_name = f"{video['type'].lower()}_{video['name']}"
    target_path = os.path.join(CACHE_DIR, local_name)
    
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        return {"status": "success", "message": "Video already downloaded", "path": target_path}

    try:
        video_type = video.get("type", "REAL")
        video_repo_id = video.get("repo_id")
        video_hf_path = video.get("hf_path")
        video_url = video.get("url")
        
        # For REAL videos: always download via URL
        if video_repo_id is None or video_hf_path is None:
            if not video_url:
                raise HTTPException(status_code=400, detail="No download URL available for this video")
            print(f"Downloading REAL video from: {video_url}")
            response = requests.get(video_url, stream=True, timeout=60, allow_redirects=True)
            response.raise_for_status()
            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=16384):
                    if chunk:
                        f.write(chunk)
        else:
            # For DEEPFAKE videos: download via HF Hub
            print(f"Downloading {video_type} via HF Hub: {video_hf_path} -> {target_path}")
            path = hf_hub_download(
                repo_id=video_repo_id,
                filename=video_hf_path,
                repo_type="dataset",
                local_dir=CACHE_DIR,
                local_dir_use_symlinks=False
            )
            # Rename to our local_name
            if path != target_path and os.path.exists(path):
                if os.path.exists(target_path): os.remove(target_path)
                os.rename(path, target_path)

        return {"status": "success", "message": "Download complete", "video": video}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.get("/demo-videos")
def list_demo_videos():
    """List all videos currently available in the local cache"""
    videos = []
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    if os.path.exists(CACHE_DIR):
        for root, dirs, files in os.walk(CACHE_DIR):
            # Skip hidden/system directories like .huggingface
            dirs[:] = [d for d in dirs if not d.startswith('.')]  # type: ignore
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, CACHE_DIR)
                    fname_lower = file.lower()
                    # Determine type by prefix: deepfake_ = DEEPFAKE, real_ = REAL
                    if fname_lower.startswith("deepfake_"):
                        video_type = "DEEPFAKE"
                    elif fname_lower.startswith("real_"):
                        video_type = "REAL"
                    else:
                        # Fallback: keyword match
                        video_type = "DEEPFAKE" if any(x in fname_lower for x in ["fake", "forged", "df"]) else "REAL"
                    
                    videos.append({
                        "filename": file,
                        "path_id": rel_path,
                        "full_path": full_path,
                        "type": video_type
                    })
    
    return videos



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


@app.post("/clear-cache")
def clear_cache():
    """Wipe all downloaded videos and uploaded results to start fresh."""
    try:
        # Clear hf_cache
        if os.path.exists(CACHE_DIR):
            for filename in os.listdir(CACHE_DIR):
                file_path = os.path.join(CACHE_DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        
        # Clear uploads
        if os.path.exists(UPLOAD_DIR):
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        
        return {"status": "success", "message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@app.get("/stats")
def get_pipeline_stats():
    """Get overall pipeline performance statistics."""
    return pipeline.stats

if __name__ == "__main__":
    import uvicorn # type: ignore
    uvicorn.run(app, host="0.0.0.0", port=8000)
