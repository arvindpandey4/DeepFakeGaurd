import time
import requests
from huggingface_hub import hf_hub_download

def check_connection():
    print("1. Testing ping to huggingface.co...")
    try:
        start = time.time()
        response = requests.get("https://huggingface.co", timeout=5)
        ping = (time.time() - start) * 1000
        print(f"   ✅ Success! Ping: {ping:.0f}ms (Status: {response.status_code})")
    except Exception as e:
        print(f"   ❌ Failed to reach Hugging Face: {e}")
        return

    print("\n2. Testing download speed (Small metadata file)...")
    try:
        start = time.time()
        # Try downloading a tiny file from the dataset repo
        path = hf_hub_download(
            repo_id="UniDataPro/deepfake-videos-dataset",
            filename=".gitattributes",
            repo_type="dataset",
            force_download=True  # Force to test actual speed
        )
        duration = time.time() - start
        print(f"   ✅ Download successful!")
        print(f"   Time taken: {duration:.2f}s")
        print(f"   File saved to: {path}")
    except Exception as e:
        print(f"   ❌ Download failed: {e}")

if __name__ == "__main__":
    print("--- Hugging Face Config Check ---")
    check_connection()
