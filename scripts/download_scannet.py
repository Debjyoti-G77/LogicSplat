"""
Download ScanNet scenes for training.
Downloads only the files needed for geometry extraction:
- _vh_clean_2.ply (mesh)
- _vh_clean_2.labels.ply (semantic labels)
- .aggregation.json (object groupings)
- _vh_clean_2.0.010000.segs.json (segmentation)

Usage:
    python scripts/download_scannet.py
"""
import subprocess
import sys
import os

OUTPUT_DIR = "data/scannet"
SCRIPT = "download-scannet.py"

# Start with 20 diverse indoor scenes
SCENES = [
    "scene0000_00", "scene0001_00", "scene0002_00", "scene0003_00",
    "scene0004_00", "scene0005_00", "scene0006_00", "scene0007_00",
    "scene0008_00", "scene0009_00", "scene0010_00", "scene0011_00",
    "scene0012_00", "scene0013_00", "scene0014_00", "scene0015_00",
    "scene0016_00", "scene0017_00", "scene0018_00", "scene0019_00",
]

FILE_TYPES = [
    "_vh_clean_2.ply",
    "_vh_clean_2.labels.ply",
    ".aggregation.json",
    "_vh_clean_2.0.010000.segs.json",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for scene in SCENES:
    print(f"\n{'='*50}")
    print(f"Downloading {scene}...")
    for ftype in FILE_TYPES:
        # check if already downloaded
        expected = os.path.join(OUTPUT_DIR, "scans", scene, f"{scene}{ftype}")
        if os.path.exists(expected):
            print(f"  SKIP {ftype} (already exists)")
            continue
        result = subprocess.run(
            [sys.executable, SCRIPT, "-o", OUTPUT_DIR, "--id", scene, "--type", ftype],
            input="yes\n",
            text=True,
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"  ERROR downloading {ftype}")
    print(f"  Done {scene}")

print("\nAll downloads complete.")
