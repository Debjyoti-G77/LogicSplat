"""Find which ScanNet scenes are being skipped during cache building."""
import sys
sys.path.insert(0, ".")
import os
from src.dataset.loader_scannet import load_scannet_scene, _scene_files

SCANNET_DIR = "D:/scannet/scans"

scene_ids = sorted([
    name for name in os.listdir(SCANNET_DIR)
    if os.path.isdir(os.path.join(SCANNET_DIR, name))
])

print(f"Total scenes: {len(scene_ids)}")
skipped = []

for scene_id in scene_ids:
    scene_dir = os.path.join(SCANNET_DIR, scene_id)
    try:
        _scene_files(scene_dir, scene_id)
    except FileNotFoundError as e:
        skipped.append((scene_id, str(e)))

print(f"Skipped (missing files): {len(skipped)}")
for scene_id, reason in skipped:
    print(f"  {scene_id}: {reason}")
