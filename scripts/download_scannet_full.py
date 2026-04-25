"""
Download all 1513 ScanNet scenes.
Only downloads the 4 files needed for geometry extraction (~50MB per scene, ~75GB total).
Skips already-downloaded scenes.

Usage:
    python scripts/download_scannet_full.py
"""
import subprocess
import sys
import os

OUTPUT_DIR = "D:/scannet"
SCRIPT = "download-scannet.py"

FILE_TYPES = [
    "_vh_clean_2.ply",
    "_vh_clean_2.labels.ply",
    ".aggregation.json",
    "_vh_clean_2.0.010000.segs.json",
]

# All 1513 ScanNet v2 scenes
# Format: scene%04d_%02d
scenes = []
for space_id in range(707):  # 0-706 space IDs
    for scan_id in range(3):  # usually 00, 01, 02
        scenes.append(f"scene{space_id:04d}_{scan_id:02d}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Total scenes to download: {len(scenes)}")
print(f"Estimated size: ~{len(scenes) * 50 // 1024}GB")
print("Skipping already-downloaded scenes.\n")

downloaded = 0
skipped = 0
failed = 0

for i, scene in enumerate(scenes):
    scene_dir = os.path.join(OUTPUT_DIR, "scans", scene)

    # check if all 4 files already exist
    all_exist = all(
        os.path.exists(os.path.join(scene_dir, f"{scene}{ftype}"))
        for ftype in FILE_TYPES
    )
    if all_exist:
        skipped += 1
        continue

    print(f"[{i+1}/{len(scenes)}] Downloading {scene}...")

    success = True
    for ftype in FILE_TYPES:
        expected = os.path.join(scene_dir, f"{scene}{ftype}")
        if os.path.exists(expected):
            continue
        result = subprocess.run(
            [sys.executable, SCRIPT, "-o", OUTPUT_DIR, "--id", scene, "--type", ftype],
            input="yes\n",
            text=True,
            capture_output=True,
        )
        # check file was actually created, not just that subprocess ran
        if not os.path.exists(expected):
            success = False
            break

    if success:
        downloaded += 1
        if downloaded % 50 == 0:
            print(f"\n  Progress: {downloaded} downloaded, {skipped} skipped, {failed} failed\n")
    else:
        failed += 1

print(f"\nDone. Downloaded: {downloaded}, Skipped: {skipped}, Failed/Missing: {failed}")
print(f"Total scenes available: {downloaded + skipped}")
