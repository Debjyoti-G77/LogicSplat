"""
Validate preprocessing pipeline on scenes 6-13 using GT object centroids.

For each scene:
  1. Run full pipeline (filter → prune → RANSAC table removal → HDBSCAN)
  2. Hungarian-match clusters to GT object centroids (raw PLY-space distance)
  3. Report: match error, bbox sizes, any missed objects

This is validation only — no GT labels used for training.
"""
import sys, os, json, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.gaussian.loader import (
    load_gaussian_ply, filter_gaussians,
    prune_isolated_gaussians, remove_table_background,
)
from src.gaussian.clustering import gaussian_to_objects

DATA_DIR = "D:/logicsplat_data/processed"
SCENES   = [f"scene_{i:02d}" for i in range(6, 14)]

def hungarian_match(cluster_centroids, gt_centroids):
    """Returns (row_ind → cluster, col_ind → GT), cost matrix."""
    cost = np.linalg.norm(
        cluster_centroids[:, None, :] - gt_centroids[None, :, :], axis=2
    )
    r, c = linear_sum_assignment(cost)
    return r, c, cost

def run_pipeline(ply_path, n_hint):
    cloud = load_gaussian_ply(ply_path)
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    n_raw = cloud.num_gaussians
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)
    cloud = remove_table_background(cloud)
    n_clean = cloud.num_gaussians

    objects, params = gaussian_to_objects(
        cloud,
        target_min=n_hint,
        target_max=n_hint + 3,
        n_exact=n_hint,
    )
    if len(objects) < n_hint - 1:
        for cw in [0.5, 0.7, 1.0]:
            o2, p2 = gaussian_to_objects(
                cloud, target_min=n_hint, target_max=n_hint+3,
                n_exact=n_hint, color_weight=cw,
            )
            if abs(len(o2) - n_hint) < abs(len(objects) - n_hint):
                objects, params = o2, p2

    return objects, n_raw, n_clean

print("=" * 90)
print("  Preprocessing Validation — GT centroid matching (scenes 6-13)")
print("=" * 90)
print()

all_errors = []
all_missed = 0
all_gt = 0

for scene_name in SCENES:
    scene_dir = os.path.join(DATA_DIR, scene_name)
    ply_path  = os.path.join(scene_dir, "splat.ply")
    gt_path   = os.path.join(scene_dir, "ground_truth_relations.json")

    if not os.path.exists(ply_path) or not os.path.exists(gt_path):
        print(f"{scene_name}: MISSING files")
        continue

    with open(gt_path) as f:
        gt = json.load(f)

    gt_objects   = gt["objects"]
    gt_centroids = np.array([o["centroid"] for o in gt_objects])
    gt_names     = [o["name"] for o in gt_objects]
    n_gt         = len(gt_objects)

    objects, n_raw, n_clean = run_pipeline(ply_path, n_hint=n_gt)
    n_found = len(objects)

    print(f"{'-'*70}")
    print(f"  {scene_name}: GT={n_gt}  Found={n_found}  "
          f"raw={n_raw:,} -> clean={n_clean:,} ({100*n_clean/n_raw:.0f}%)")

    if n_found == 0:
        print(f"  ERROR: no objects found")
        all_missed += n_gt
        all_gt += n_gt
        continue

    clust_centroids = np.array([o.centroid for o in objects])
    clust_diags     = [round(float(np.linalg.norm(o.bbox_max - o.bbox_min)), 3)
                       for o in objects]

    # Compute GT scene extent to report normalised distances too
    gt_bbox_min = gt_centroids.min(axis=0)
    gt_bbox_max = gt_centroids.max(axis=0)
    gt_extent   = max(float(np.linalg.norm(gt_bbox_max - gt_bbox_min)), 1e-6)

    r_idx, c_idx, cost = hungarian_match(clust_centroids, gt_centroids)

    print(f"  {'Cluster->GT':<35}  raw_dist   norm_dist  clust_diag")
    print(f"  {'-'*65}")

    matched_gt = set()
    for ri, ci in zip(r_idx, c_idx):
        d_raw  = cost[ri, ci]
        d_norm = d_raw / gt_extent
        name   = gt_names[ci]
        diag   = clust_diags[ri]
        status = "OK" if d_norm < 0.3 else ("WARN" if d_norm < 0.6 else "MISS")
        print(f"  cluster_{ri} -> {name:<22}  {d_raw:7.3f}    {d_norm:7.3f}    {diag:.3f}  [{status}]")
        all_errors.append(d_norm)
        matched_gt.add(ci)

    missed = [gt_names[i] for i in range(n_gt) if i not in matched_gt]
    if missed:
        print(f"  UNMATCHED GT: {missed}")
    all_missed += len(missed)
    all_gt     += n_gt

    print(f"  Avg norm match-error = {np.mean([cost[ri,ci]/gt_extent for ri,ci in zip(r_idx,c_idx)]):.3f}")

print()
print("=" * 90)
print(f"  Overall avg normalised centroid error : {np.mean(all_errors):.3f}  (target < 0.20)")
print(f"  Missed objects                        : {all_missed}/{all_gt}")
print(f"  Fraction matched within 0.3 norm-dist : "
      f"{sum(1 for e in all_errors if e<0.3)/len(all_errors)*100:.0f}%")
print(f"  Fraction matched within 0.2 norm-dist : "
      f"{sum(1 for e in all_errors if e<0.2)/len(all_errors)*100:.0f}%")
print("=" * 90)
