"""
Quick diagnostic: Test HF connectivity for both Real (Kinetics400) and Deepfake (UniDataPro) sources.
Run with: .\venv\Scripts\python.exe test_hf.py
"""
import time
import os
import urllib.parse
import itertools

UNIDATAPRO_REPO = "UniDataPro/deepfake-videos-dataset"


print("=" * 60)
print("HUGGING FACE CONNECTIVITY DIAGNOSTIC")
print("=" * 60)

# ─── TEST 1: HF Login token ───────────────────────────────────────────────────
print("\n[1] Checking HF token...")
try:
    from huggingface_hub import whoami  # type: ignore
    user = whoami()
    print(f"  ✓ Logged in as: {user['name']}")
except Exception as e:
    print(f"  ✗ NOT authenticated: {e}")

# ─── TEST 2: Kinetics400 streaming ───────────────────────────────────────────
print("\n[2] Fetching 5 REAL videos from Kinetics400 (streaming)...")
t0 = time.time()
real_videos = []
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

        video = {
            "name": clip_name,
            "type": "REAL",
            "action": item.get("action_class", "?"),
            "hf_path": hf_path,
            "url": url[:80] + "..." if url and len(url) > 80 else url,
        }
        real_videos.append(video)
        print(f"  ✓ {clip_name}  [{video['action']}]   hf_path={'YES' if hf_path else 'NO (URL only)'}")

    print(f"  → Got {len(real_videos)} REAL entries in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"  ✗ FAILED in {time.time()-t0:.1f}s: {e}")

# ─── TEST 3: UniDataPro listing ───────────────────────────────────────────────
print("\n[3] Listing 5 DEEPFAKE videos from UniDataPro...")
t0 = time.time()
try:
    from huggingface_hub import list_repo_files, hf_hub_url  # type: ignore
    files = list(list_repo_files(UNIDATAPRO_REPO, repo_type="dataset"))
    video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
    fake_files = [
        f for f in files
        if isinstance(f, str)
        and f.startswith("deepfake/")
        and f.lower().endswith(video_extensions)
    ][:5]

    if not fake_files:
        print("  ✗ No deepfake files found! Check repo structure or access.")
        # Show first 10 files to help debug
        print(f"  First 10 files in repo: {files[:10]}")
    else:
        for f in fake_files:
            print(f"  ✓ {os.path.basename(f)}")
        print(f"  → Got {len(fake_files)} DEEPFAKE entries in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"  ✗ FAILED in {time.time()-t0:.1f}s: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
