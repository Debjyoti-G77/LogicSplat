"""
STEP 3 — Run full YOLO labeling on all available scenes.
Saves yolo_labels.json in each scene directory.
"""
import sys
sys.path.insert(0, ".")
import os
import warnings
warnings.filterwarnings("ignore")

from src.gaussian.loader import load_gaussian_ply, filter_gaussians
from src.gaussian.clustering import gaussian_to_objects
from src.labeling.yolo_labeler import label_objects_with_yolo

DATA_DIR = "D:/logicsplat_data/processed"

scenes = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
    and os.path.exists(os.path.join(DATA_DIR, d, "splat.ply"))
])

print(f"Found {len(scenes)} scene(s): {scenes}")

for scene_id in scenes:
    scene_dir = os.path.join(DATA_DIR, scene_id)
    splat_path = os.path.join(scene_dir, "splat.ply")
    transforms_path = os.path.join(scene_dir, "ns_data", "transforms.json")
    images_dir = os.path.join(scene_dir, "ns_data", "images")

    print(f"\n{'='*60}")
    print(f"Scene: {scene_id}")
    print(f"{'='*60}")

    if not os.path.exists(transforms_path):
        print(f"  SKIP — no transforms.json at {transforms_path}")
        continue
    if not os.path.isdir(images_dir):
        print(f"  SKIP — no images dir at {images_dir}")
        continue

    # Force re-run by removing cached labels
    labels_cache = os.path.join(scene_dir, "yolo_labels.json")
    if os.path.exists(labels_cache):
        os.remove(labels_cache)
        print(f"  Removed cached labels to force fresh run")

    print("  Loading and clustering Gaussians...")
    cloud = load_gaussian_ply(splat_path)
    filtered = filter_gaussians(cloud, opacity_threshold=0.1)
    objects, _ = gaussian_to_objects(filtered)
    print(f"  {len(objects)} objects found")

    objects = label_objects_with_yolo(
        objects,
        transforms_path,
        images_dir,
        n_frames=30,
        confidence=0.25,
        scene_dir=scene_dir,
    )

    print(f"\n  Final labels for {scene_id}:")
    for obj in objects:
        print(f"    Obj {obj.uid}: {obj.label}  centroid={obj.centroid.round(3)}")

print("\nDone.")
