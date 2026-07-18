import sys
sys.path.insert(0, ".")
from src.dataset.loader_3rscan_splat import load_3dssg_annotations

objects_by_scan, rels_by_scan = load_3dssg_annotations()
print(f"Objects: {len(objects_by_scan)} scenes")
print(f"Relationships: {len(rels_by_scan)} scenes")

# Check first scene
first_scan = list(objects_by_scan.keys())[0]
print(f"\nFirst scene: {first_scan}")
print(f"  Objects: {len(objects_by_scan[first_scan])}")
print(f"  Relations: {len(rels_by_scan[first_scan])}")

# Check relationship format
rels = rels_by_scan[first_scan]
print(f"\n  First 5 relations:")
for r in rels[:5]:
    print(f"    {r}")

# Count how many scenes have splats
import os
splats_dir = "D:/3rscan_splats"
if os.path.isdir(splats_dir):
    splat_scenes = set(os.listdir(splats_dir))
    overlap = splat_scenes & set(objects_by_scan.keys()) & set(rels_by_scan.keys())
    print(f"\nScenes with splats + objects + rels: {len(overlap)}")
else:
    print(f"\nSplats dir not found: {splats_dir}")
