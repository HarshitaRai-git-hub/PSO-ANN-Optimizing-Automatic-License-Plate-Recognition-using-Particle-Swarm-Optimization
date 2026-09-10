"""
Download EasyOCR models manually with retry logic.
Models: CRAFT text detection + English recognition
"""
import urllib.request
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

model_dir = os.path.join(os.path.expanduser("~"), ".EasyOCR", "model")
os.makedirs(model_dir, exist_ok=True)


temp_zip = os.path.join(model_dir, "temp.zip")
try:
    if os.path.exists(temp_zip):
        os.remove(temp_zip)
        print(f"Cleaned up {temp_zip}")
except:
    print(f"Could not remove {temp_zip}, will skip.")

models = {
    "craft_mlt_25k.pth": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/craft_mlt_25k.zip",
    "english_g2.pth": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip",
}

for name, url in models.items():
    dest = os.path.join(model_dir, name)
    if os.path.exists(dest):
        print(f"[OK] {name} already exists, skipping.")
        continue
    
    print(f"\nDownloading {name}...")
    print(f"  URL: {url}")
    
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            
            def progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    pct = min(100, downloaded * 100 / total_size)
                    mb_done = downloaded / (1024*1024)
                    mb_total = total_size / (1024*1024)
                    print(f"\r  Progress: {pct:.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)", end="", flush=True)
            
            
            zip_path = dest + ".zip"
            urllib.request.urlretrieve(url, zip_path, reporthook=progress)
            print()
            
            
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as z:
                
                z.extractall(model_dir)
            os.remove(zip_path)
            
            print(f"  [OK] {name} downloaded and extracted!")
            break
        except Exception as e:
            print(f"\n  [RETRY {attempt+1}/3] Error: {e}")
            time.sleep(2)
    else:
        print(f"  [FAIL] Could not download {name} after 3 attempts.")

print("\nDone! Checking model directory:")
for f in os.listdir(model_dir):
    size_mb = os.path.getsize(os.path.join(model_dir, f)) / (1024*1024)
    print(f"  {f} ({size_mb:.1f} MB)")
