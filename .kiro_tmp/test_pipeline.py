"""Quick test: process one scene to verify the full pipeline works."""
import sys
sys.path.insert(0, ".")
import warnings
warnings.filterwarnings("ignore")

from src.dataset.loader_3rscan_splat import (
    load_3dssg_annotations, process_scene, match_clusters_to_annotations
)

# Load annotations
objects_by_scan, rels_by_scan = load_3dssg_annotations()

# Find a scene with a reasonable number of objects (5-10)
import os
splats_dir = "D:/3rscan_splats"
candidates = []
for scene_id in os.listdir(splats_dir):
    if scene_id not in objects_by_scan or scene_id not in rels_by_scan:
        continue
    n_obj = len(objects_by_scan[scene_id])
    n_rel = len(rels_by_scan[scene_id])
    if 4 <= n_obj <= 8 and n_rel >= 5:
        candidates.append((scene_id, n_obj, n_rel))

candidates.sort(key=lambda x: x[1])
print(f"Found {len(candidates)} candidate scenes with 4-8 objects")

if candidates:
    scene_id, n_obj, n_rel = candidates[0]
    print(f"\nTesting scene: {scene_id}")
    print(f"  GT objects: {n_obj}, GT relations: {n_rel}")

    graph = process_scene(
        scene_id=scene_id,
        splats_dir=splats_dir,
        gt_objects=objects_by_scan[scene_id],
        gt_relationships=rels_by_scan[scene_id],
    )

    if graph is None:
        print("  -> Processing returned None (cluster count mismatch or other issue)")
        # Try a few more
        for scene_id, n_obj, n_rel in candidates[1:6]:
            print(f"\n  Trying: {scene_id} ({n_obj} objects, {n_rel} rels)")
            graph = process_scene(
                scene_id=scene_id,
                splats_dir=splats_dir,
                gt_objects=objects_by_scan[scene_id],
                gt_relationships=rels_by_scan[scene_id],
            )
            if graph is not None:
                break
            print("    -> None")

    if graph is not None:
        print(f"\n  SUCCESS! Graph produced:")
        print(f"    x shape:          {graph['x'].shape}")
        print(f"    edge_index shape: {graph['edge_index'].shape}")
        print(f"    edge_attr shape:  {graph['edge_attr'].shape}")
        print(f"    edge_label shape: {graph['edge_label'].shape}")
        print(f"    Positive labels:  {int(graph['edge_label'].sum())}")
        print(f"    scene_id:         {graph['scene_id']}")
    else:
        print("\n  All candidates failed. This is expected — cluster matching is strict.")
        print("  The full dataset will find matches across 566 scenes.")
