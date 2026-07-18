"""Diagnose on_top_of for scene_06 — print bbox/centroid values and rule evaluation."""
import sys
sys.path.insert(0, ".")
import json
import numpy as np
from src.gaussian.loader import load_gaussian_ply, filter_gaussians
from src.gaussian.clustering import (
    gaussian_to_objects, auto_min_cluster_size,
    auto_sat_threshold,
)
from sklearn.preprocessing import StandardScaler
from src.relations.geometry import derive_relations, _bbox_xy_overlap
from src.relations.schema import RELATION_NAMES

ply      = "D:/logicsplat_data/processed/scene_06/splat.ply"
gt_path  = "D:/logicsplat_data/processed/scene_06/ground_truth_relations.json"

with open(gt_path) as f:
    gt = json.load(f)
n_hint = len(gt["objects"])
print(f"GT objects ({n_hint}): {[o['name'] for o in gt['objects']]}")
print()

cloud = load_gaussian_ply(ply)
cf    = filter_gaussians(cloud, opacity_threshold=0.1)

sat_thresh = auto_sat_threshold(cf.rgb)
sat        = cf.rgb.astype(float).max(axis=1) - cf.rgb.astype(float).min(axis=1)
sat_mask   = sat > sat_thresh
if sat_mask.sum() < 0.05 * cf.num_gaussians:
    sat_mask = sat > sat_thresh * 0.5
if sat_mask.sum() < 0.05 * cf.num_gaussians:
    sat_mask = np.ones(cf.num_gaussians, dtype=bool)

xyz        = cf.xyz[sat_mask]
rgb2       = cf.rgb[sat_mask]
color_norm = rgb2.astype(np.float32) / 255.0 * 0.3
X          = np.concatenate([xyz, color_norm], axis=1)
X_scaled   = StandardScaler().fit_transform(X)

# Use fixed mcs from last successful run (scene_06 found 5 objects at mcs=1291)
mcs = 1291
print(f"Using fixed mcs={mcs}")

objects, params = gaussian_to_objects(cf, min_cluster_size=mcs)
print(f"Clusters found: {len(objects)}")
print()

for o in objects:
    size = o.bbox_max - o.bbox_min
    print(f"  Obj {o.uid}: centroid={np.round(o.centroid, 3)}")
    print(f"           bbox_min={np.round(o.bbox_min, 3)}")
    print(f"           bbox_max={np.round(o.bbox_max, 3)}")
    print(f"           size    ={np.round(size, 3)}  pts={o.point_count}")
    print()

print("--- on_top_of diagnosis for all pairs ---")
for i, a in enumerate(objects):
    for j, b in enumerate(objects):
        if i == j:
            continue
        rels      = derive_relations(a.bbox_min, a.bbox_max, b.bbox_min, b.bbox_max)
        rel_names = [RELATION_NAMES[int(r)] for r in rels]
        vgap      = float(a.bbox_min[2] - b.bbox_max[2])
        avg_h     = (a.bbox_max[2] - a.bbox_min[2] + b.bbox_max[2] - b.bbox_min[2]) / 2
        xy_ov     = _bbox_xy_overlap(a.bbox_min, a.bbox_max, b.bbox_min, b.bbox_max)
        a_above   = a.centroid[2] > b.centroid[2]
        print(
            f"  ({i}->{j}): vgap={vgap:.3f}  avg_h={avg_h:.3f}  "
            f"xy_overlap={xy_ov}  a_above={a_above}  => {rel_names}"
        )
