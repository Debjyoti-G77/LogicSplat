"""
Diagnose on_top_of FPs for scene_06 after Z-flip.
Uses fixed mcs=861 (known good from last run).
"""
import sys
sys.path.insert(0, ".")
import json
import numpy as np
from src.gaussian.loader import load_gaussian_ply, filter_gaussians
from src.gaussian.clustering import gaussian_to_objects
from src.relations.geometry import derive_relations, _bbox_xy_overlap
from src.relations.schema import RELATION_NAMES

ply     = "D:/logicsplat_data/processed/scene_06/splat.ply"
gt_path = "D:/logicsplat_data/processed/scene_06/ground_truth_relations.json"

with open(gt_path) as f:
    gt = json.load(f)

cloud   = load_gaussian_ply(ply)
cf      = filter_gaussians(cloud, opacity_threshold=0.1)
objects, _ = gaussian_to_objects(cf, min_cluster_size=861)

# Z-flip (same as run_inference)
for o in objects:
    o.centroid = o.centroid.copy(); o.centroid[2] *= -1
    o.bbox_min = o.bbox_min.copy(); o.bbox_min[2] *= -1
    o.bbox_max = o.bbox_max.copy(); o.bbox_max[2] *= -1
    o.bbox_min[2], o.bbox_max[2] = min(o.bbox_min[2], o.bbox_max[2]), max(o.bbox_min[2], o.bbox_max[2])

# GT centroid matching (Z-flipped)
gt_objects = gt["objects"]
gt_cents = np.array([o["centroid"] for o in gt_objects], dtype=float)
gt_cents[:, 2] *= -1
cluster_cents = np.array([o.centroid for o in objects])
assigned = set()
uid_to_name = {}
for gi, gt_obj in enumerate(gt_objects):
    dists = np.linalg.norm(cluster_cents - gt_cents[gi], axis=1)
    for used in assigned:
        dists[[o.uid for o in objects].index(used)] = np.inf
    best = int(np.argmin(dists))
    uid_to_name[objects[best].uid] = gt_obj["name"]
    assigned.add(objects[best].uid)

print("Cluster → GT name:")
for uid, name in sorted(uid_to_name.items()):
    o = objects[uid]
    print(f"  {uid} ({name}): centroid_z={o.centroid[2]:.3f}  "
          f"bbox_z=[{o.bbox_min[2]:.3f}, {o.bbox_max[2]:.3f}]  "
          f"height={o.bbox_max[2]-o.bbox_min[2]:.3f}")

print()
print("All pairs — on_top_of / under details:")
for i, a in enumerate(objects):
    for j, b in enumerate(objects):
        if i == j:
            continue
        vgap     = float(a.bbox_min[2] - b.bbox_max[2])
        avg_h    = (a.bbox_max[2] - a.bbox_min[2] + b.bbox_max[2] - b.bbox_min[2]) / 2
        xy_ov    = _bbox_xy_overlap(a.bbox_min, a.bbox_max, b.bbox_min, b.bbox_max)
        a_above  = a.centroid[2] > b.centroid[2]
        thresh   = avg_h * 0.15
        fires    = a_above and (-avg_h * 0.5 < vgap <= thresh) and xy_ov
        name_a   = uid_to_name.get(i, f"obj{i}")
        name_b   = uid_to_name.get(j, f"obj{j}")
        # check if GT has on_top_of for this pair
        gt_has   = any(r["subject"] == name_a and r["relation"] == "on_top_of"
                       and r["object"] == name_b for r in gt["relations"])
        if fires or gt_has:
            print(f"  {name_a} → {name_b}: vgap={vgap:.3f}  avg_h={avg_h:.3f}  "
                  f"thresh={thresh:.3f}  xy_overlap={xy_ov}  a_above={a_above}  "
                  f"FIRES={fires}  GT={gt_has}")
