"""
Diagnose behind/in_front_of FNs for scene_06.
Shows delta_x, delta_y, delta_z for all GT directional pairs.
Uses fixed mcs=861.
"""
import sys
sys.path.insert(0, ".")
import json
import numpy as np
from src.gaussian.loader import load_gaussian_ply, filter_gaussians
from src.gaussian.clustering import gaussian_to_objects
from src.relations.geometry import _bbox_xy_overlap

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
name_to_uid = {}
for gi, gt_obj in enumerate(gt_objects):
    dists = np.linalg.norm(cluster_cents - gt_cents[gi], axis=1)
    for used in assigned:
        dists[used] = np.inf
    best = int(np.argmin(dists))
    uid_to_name[best] = gt_obj["name"]
    name_to_uid[gt_obj["name"]] = best
    assigned.add(best)

# For each GT behind/in_front_of relation, show the actual deltas
target_rels = {"behind", "in_front_of", "to_the_left_of", "to_the_right_of"}
gt_rel_map = {"to_the_left_of": "left_of", "to_the_right_of": "right_of",
              "behind": "behind", "in_front_of": "in_front_of"}

print(f"{'Subject':<14} {'Rel':<14} {'Object':<14} {'dx':>7} {'dy':>7} {'dz':>7} "
      f"{'|dx|':>7} {'|dy|':>7} {'dom_axis':<10} {'would_fire'}")
print("-" * 100)

for r in gt["relations"]:
    rel = r.get("relation", "")
    if rel not in target_rels:
        continue
    subj = r["subject"]
    obj  = r["object"]
    si = name_to_uid.get(subj)
    oi = name_to_uid.get(obj)
    if si is None or oi is None:
        continue
    a = objects[si]
    b = objects[oi]
    delta = a.centroid - b.centroid
    dx, dy, dz = delta
    adx, ady, adz = abs(dx), abs(dy), abs(dz)
    avg_size = (np.linalg.norm(a.bbox_max - a.bbox_min) + np.linalg.norm(b.bbox_max - b.bbox_min)) / 2
    dist_3d = np.linalg.norm(delta)
    separated = dist_3d > avg_size * 0.3

    if adx > ady and adx > adz:
        dom = "X"
        fires_left  = dx > 0
        fires_right = dx < 0
    elif ady > adx and ady > adz:
        dom = "Y"
        fires_front = dy < 0
        fires_behind = dy > 0
    else:
        dom = "Z/tie"

    schema_rel = gt_rel_map.get(rel, rel)
    # would current rule fire the correct relation?
    if adx > ady and adx > adz:
        would_fire = (schema_rel == "left_of" and dx < 0) or (schema_rel == "right_of" and dx > 0)
    elif ady > adx and ady > adz:
        would_fire = (schema_rel == "behind" and dy > 0) or (schema_rel == "in_front_of" and dy < 0)
    else:
        would_fire = False

    print(f"{subj:<14} {schema_rel:<14} {obj:<14} {dx:>7.3f} {dy:>7.3f} {dz:>7.3f} "
          f"{adx:>7.3f} {ady:>7.3f} {dom:<10} {str(would_fire)}")
