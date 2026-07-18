"""
Find the right adjacent_to distance threshold for scene_06.
Prints XY distance for every pair, marking GT adjacent pairs.
Uses fixed mcs=861.
"""
import sys
sys.path.insert(0, ".")
import json
import numpy as np
from src.gaussian.loader import load_gaussian_ply, filter_gaussians
from src.gaussian.clustering import gaussian_to_objects

ply     = "D:/logicsplat_data/processed/scene_06/splat.ply"
gt_path = "D:/logicsplat_data/processed/scene_06/ground_truth_relations.json"

with open(gt_path) as f:
    gt = json.load(f)

cloud   = load_gaussian_ply(ply)
cf      = filter_gaussians(cloud, opacity_threshold=0.1)
objects, _ = gaussian_to_objects(cf, min_cluster_size=861)

# Z-flip
for o in objects:
    o.centroid = o.centroid.copy(); o.centroid[2] *= -1
    o.bbox_min = o.bbox_min.copy(); o.bbox_min[2] *= -1
    o.bbox_max = o.bbox_max.copy(); o.bbox_max[2] *= -1
    o.bbox_min[2], o.bbox_max[2] = min(o.bbox_min[2], o.bbox_max[2]), max(o.bbox_min[2], o.bbox_max[2])

# GT centroid matching
gt_objects = gt["objects"]
gt_cents = np.array([o["centroid"] for o in gt_objects], dtype=float)
gt_cents[:, 2] *= -1
cluster_cents = np.array([o.centroid for o in objects])
assigned = set()
uid_to_name = {}
for gi, gt_obj in enumerate(gt_objects):
    dists = np.linalg.norm(cluster_cents - gt_cents[gi], axis=1)
    for used in assigned:
        dists[used] = np.inf
    best = int(np.argmin(dists))
    uid_to_name[best] = gt_obj["name"]
    assigned.add(best)

gt_adjacent = {(r["subject"], r["object"])
               for r in gt["relations"] if r["relation"] == "adjacent_to"}

print(f"{'Subject':<14} {'Object':<14} {'dist_xy':>8} {'dist_3d':>8} {'avg_diag':>9} {'ratio':>7}  GT")
print("-" * 70)

rows = []
for i, a in enumerate(objects):
    for j, b in enumerate(objects):
        if i == j:
            continue
        sname = uid_to_name.get(i, f"obj{i}")
        dname = uid_to_name.get(j, f"obj{j}")
        delta = a.centroid - b.centroid
        dist_xy = float(np.linalg.norm(delta[:2]))
        dist_3d = float(np.linalg.norm(delta))
        diag_a  = float(np.linalg.norm(a.bbox_max - a.bbox_min))
        diag_b  = float(np.linalg.norm(b.bbox_max - b.bbox_min))
        avg_diag = (diag_a + diag_b) / 2
        ratio = dist_xy / avg_diag
        is_gt = (sname, dname) in gt_adjacent
        rows.append((dist_xy, sname, dname, dist_3d, avg_diag, ratio, is_gt))

rows.sort()
for dist_xy, sname, dname, dist_3d, avg_diag, ratio, is_gt in rows:
    marker = " *" if is_gt else ""
    print(f"{sname:<14} {dname:<14} {dist_xy:>8.3f} {dist_3d:>8.3f} {avg_diag:>9.3f} {ratio:>7.3f}{marker}")
