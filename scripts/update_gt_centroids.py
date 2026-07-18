"""
Update GT centroids for scenes 06-13 using the actual splat data after SOR filtering.

This script:
1. Loads each scene's splat.ply with the same pipeline used at inference time
   (opacity filter → SOR prune → HDBSCAN clustering → Z-flip)
2. Reports the resulting cluster centroids so we can verify that the Hungarian
   matching in evaluate_scenes.py will assign them correctly.

NOTE: This script is INFORMATIONAL only — it does NOT modify any GT JSON files.
      The actual cluster→GT matching happens at evaluation time via
      build_cluster_to_gt_name_map() in evaluate_scenes.py.

Usage:
    python scripts/update_gt_centroids.py
    python scripts/update_gt_centroids.py --scenes scene_07 scene_12
"""

import sys
import json
import warnings
import argparse
import numpy as np

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from scipy.optimize import linear_sum_assignment
from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
from src.gaussian.clustering import gaussian_to_objects


def report_scene(scene: str) -> None:
    gt_path  = f"D:/logicsplat_data/processed/{scene}/ground_truth_relations.json"
    ply_path = f"D:/logicsplat_data/processed/{scene}/splat.ply"

    import os
    if not os.path.exists(ply_path):
        print(f"\n[{scene}] SKIP — splat.ply not found")
        return
    if not os.path.exists(gt_path):
        print(f"\n[{scene}] SKIP — ground_truth_relations.json not found")
        return

    with open(gt_path) as f:
        gt = json.load(f)

    gt_objects = gt["objects"]
    n_gt = len(gt_objects)

    # ── Load + filter (mirrors run_inference pipeline) ────────────────────────
    cloud    = load_gaussian_ply(ply_path)
    filtered = filter_gaussians(cloud, opacity_threshold=0.1)
    pruned   = prune_isolated_gaussians(filtered, nb_neighbors=20, std_ratio=2.0)

    objects, params = gaussian_to_objects(
        pruned,
        target_min=max(2, n_gt - 1),
        target_max=n_gt + 1,
    )

    # ── Z-flip (same as run_inference) ────────────────────────────────────────
    for o in objects:
        o.centroid   = o.centroid.copy();   o.centroid[2]   *= -1
        o.bbox_min   = o.bbox_min.copy();   o.bbox_min[2]   *= -1
        o.bbox_max   = o.bbox_max.copy();   o.bbox_max[2]   *= -1
        o.bbox_min[2], o.bbox_max[2] = (
            min(o.bbox_min[2], o.bbox_max[2]),
            max(o.bbox_min[2], o.bbox_max[2]),
        )

    print(f"\n{'─'*60}")
    print(f"  {scene}: {len(objects)} clusters for {n_gt} GT objects")
    print(f"  Clustering params: {params}")
    print(f"\n  Cluster centroids (after Z-flip):")
    for o in objects:
        print(f"    Obj{o.uid}: pts={o.point_count:5d}  "
              f"centroid=({o.centroid[0]:+.3f}, {o.centroid[1]:+.3f}, {o.centroid[2]:+.3f})")

    # ── GT centroids (with Z-flip applied) ───────────────────────────────────
    gt_centroids = []
    for gt_obj in gt_objects:
        if "centroid" in gt_obj:
            c = np.array(gt_obj["centroid"], dtype=float)
            c[2] *= -1
            gt_centroids.append(c)
        else:
            gt_centroids.append(None)

    has_coords = any(c is not None for c in gt_centroids)

    print(f"\n  GT centroids (after Z-flip):")
    for i, gt_obj in enumerate(gt_objects):
        c = gt_centroids[i]
        if c is not None:
            print(f"    GT[{i}] '{gt_obj['name']}': "
                  f"({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f})")
        else:
            print(f"    GT[{i}] '{gt_obj['name']}': (no centroid in JSON)")

    if not has_coords or not objects:
        print(f"\n  [SKIP] Cannot compute Hungarian matching — missing GT coords or no clusters")
        return

    # ── Hungarian matching ────────────────────────────────────────────────────
    known = [c for c in gt_centroids if c is not None]
    mean_c = np.mean(known, axis=0)
    gt_arr = np.array([c if c is not None else mean_c for c in gt_centroids])

    cluster_centroids = np.array([o.centroid for o in objects])
    n_clusters = len(objects)
    n_gt_arr   = len(gt_objects)

    cost_matrix = np.zeros((n_clusters, n_gt_arr))
    for i, cc in enumerate(cluster_centroids):
        for j, gc in enumerate(gt_arr):
            cost_matrix[i, j] = np.linalg.norm(cc - gc)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    print(f"\n  Hungarian assignment (cluster → GT object):")
    total_cost = 0.0
    for ci, gi in zip(row_ind, col_ind):
        dist = cost_matrix[ci, gi]
        total_cost += dist
        print(f"    Cluster {objects[ci].uid} (pts={objects[ci].point_count}) "
              f"→ '{gt_objects[gi]['name']}'  dist={dist:.4f}")
    print(f"  Total assignment cost: {total_cost:.4f}")

    # Warn about unmatched GT objects (more GT objects than clusters)
    matched_gt = set(col_ind)
    for gi, gt_obj in enumerate(gt_objects):
        if gi not in matched_gt:
            print(f"  ⚠  GT object '{gt_obj['name']}' (id={gi}) has NO matched cluster!")


def main():
    parser = argparse.ArgumentParser(
        description="Report cluster centroids and Hungarian matching for scenes (informational only)"
    )
    parser.add_argument(
        "--scenes", nargs="+",
        default=["scene_06", "scene_07", "scene_08", "scene_09",
                 "scene_10", "scene_11", "scene_12", "scene_13"],
        help="Scene IDs to process (default: scene_06 through scene_13)",
    )
    args = parser.parse_args()

    print("LogicSplat — GT Centroid Reporter (informational, no files modified)")
    print(f"Scenes: {', '.join(args.scenes)}")

    for scene in args.scenes:
        report_scene(scene)

    print(f"\n{'='*60}")
    print("Done. No GT JSON files were modified.")


if __name__ == "__main__":
    main()
