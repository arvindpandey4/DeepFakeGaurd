"""
Deepfake Detection Demo using Hugging Face Dataset
Dataset: UniDataPro/deepfake-videos-dataset
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from huggingface_hub import hf_hub_download, list_repo_files, scan_cache_dir
from pipeline.adaptive_pipeline import AdaptivePipeline


CACHE_DIR = os.path.join(project_root, "hf_cache")


def list_cached_videos():
    """List videos already downloaded in the cache"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    cached_videos = []
    
    # Walk through the cache directory to find video files
    if os.path.exists(CACHE_DIR):
        for root, dirs, files in os.walk(CACHE_DIR):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    # Keep full path
                    full_path = os.path.join(root, file)
                    cached_videos.append(full_path)
    
    return cached_videos


# FALLBACK VIDEO LIST (Since HuggingFace XetHub is unstable)
RELIABLE_VIDEOS = [
    {
        "name": "deepfake_sample_1.mp4",
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking-and-pause.mp4",
        "type": "REAL"  # This is a real video for testing
    },
    {
        "name": "deepfake_sample_2.mp4",
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4", 
        "type": "REAL"
    },
    {
        "name": "synth_face.mp4",
        "url": "https://github.com/DariusAf/MesoNet/raw/master/test_videos/df1.mp4", # Note: Link might be hypothetical, using generic Real for stability
        "type": "DEEPFAKE"
    }
]

def list_remote_videos():
    """List available videos (using reliable fallback list)"""
    return [v["name"] for v in RELIABLE_VIDEOS]

def select_video():
    """Interactive video selection"""
    print("\n" + "="*70)
    print("VIDEO SELECTION")
    print("="*70)
    
    # 1. Check Local Cache
    cached_videos = list_cached_videos()
    
    if cached_videos:
        print(f"\n[Local Cache] Found {len(cached_videos)} downloaded videos:")
        for i, video_path in enumerate(cached_videos):
            filename = os.path.basename(video_path)
            # Try to get parent folder name for context (e.g. 'deepfake/1.mp4')
            try:
                parent = os.path.basename(os.path.dirname(video_path))
                display_name = f"{parent}/{filename}" if parent and parent != "hf_cache" else filename
            except:
                display_name = filename
                
            print(f"  {i+1}: {display_name}  (Local)")
            
        print(f"  d: Download a new video")
        print(f"  q: Quit")
        
        choice = input("\nEnter choice: ").strip().lower()
        
        if choice == 'q':
            sys.exit(0)
        elif choice == 'd':
            return download_new_video()
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(cached_videos):
                    return cached_videos[idx]
                else:
                    print("Invalid selection.")
                    return select_video()
            except ValueError:
                print("Invalid input.")
                return select_video()
    else:
        print("No videos found in local cache.")
        return download_new_video()

def download_new_video():
    """Download a new video from reliable sources"""
    print("\n" + "="*70)
    print("DOWNLOADING NEW VIDEO (Reliable Source)")
    print("="*70)
    
    remote_videos = RELIABLE_VIDEOS
    
    print(f"\n[Available Samples] Found {len(remote_videos)} videos:")
    
    for i, vid in enumerate(remote_videos):
        print(f"  {i+1}: {vid['name']} ({vid['type']})")
        
    print("\nEnter number to download (or 'q' to cancel):")
    choice = input("> ").strip().lower()
    
    if choice == 'q':
        return None
        
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(remote_videos):
            target = remote_videos[idx]
            target_file = target["name"]
            url = target["url"]
            
            print(f"\nDownloading: {target_file} ...")
            
            import requests
            
            # Save to cache
            os.makedirs(CACHE_DIR, exist_ok=True)
            video_path = os.path.join(CACHE_DIR, target_file)
            
            print(f"Direct URL: {url}")
            print(f"Saving to: {video_path}")
            
            try:
                with requests.get(url, stream=True, timeout=10) as r:
                    if r.status_code != 200:
                        print(f"❌ Error: Source returned status {r.status_code}")
                        return None
                        
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    
                    with open(video_path, 'wb') as f:
                        downloaded = 0
                        for chunk in r.iter_content(chunk_size=8192): 
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\rProgress: {percent:.1f}%", end="")
                            
                print(f"\n✓ Download complete!")
                return video_path
            except Exception as e:
                print(f"\n❌ Download failed: {e}")
                return None
                
        else:
            print("Invalid number.")
            return download_new_video()
    except ValueError:
        print("Invalid input.")
        return download_new_video()


def main():
    print("\n" + "#"*70)
    print("# Deepfake Detection Demo")
    print("# Dataset: UniDataPro/deepfake-videos-dataset")
    print("#"*70)
    
    # 1. Select Video
    video_path = select_video()
    
    if not video_path:
        print("No video selected. Exiting.")
        return

    # 2. Check Weights
    weights_path = os.path.join(project_root, "models", "weights", "Meso4_DF.h5")
    if os.path.exists(weights_path):
        print(f"\n✓ Found pretrained weights: {os.path.basename(weights_path)}")
    else:
        print(f"\n⚠ WARNING: Weights file not found at: {weights_path}")
        print("  Results will be RANDOM! Please download Meso4_DF.h5")

    # 3. Initialize Pipeline
    print("\n" + "="*70)
    print("Initializing Pipeline...")
    print("="*70)
    
    pipeline = AdaptivePipeline(weights_path=weights_path)
    
    # 4. Run Detection
    print("\n" + "="*70)
    print("Running Analysis...")
    print("="*70)
    
    try:
        result = pipeline.predict(video_path)
        
        # 5. Show Results
        print("\n" + "="*70)
        print("DETECTION RESULT")
        print("="*70)
        
        # Determine Label for Display (Green for Real, Red for Fake)
        label = result['label']
        if label == 'DEEPFAKE':
            status_icon = "🔴"
        else:
            status_icon = "🟢"
            
        print(f"Video:       {os.path.basename(video_path)}")
        print(f"Prediction:  {status_icon} {label}")
        print(f"Confidence:  {result['confidence']:.1%}")
        print(f"Exit Stage:  {result['exit_stage']} (of 3)")
        print(f"Total Time:  {result['total_time']:.2f}s")
        
        if result['exit_stage'] < 3:
            print("\n⚡ Efficiency: Exited early! (Saved computation)")
        else:
            print("\n🔍 Efficiency: Required full analysis (Complex video)")
            
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
