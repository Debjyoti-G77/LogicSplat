"""
Extract GeoKAN-Gamma's actual predicted (post-repair) relations for
scene_06, with object names, for the Figure 5 / qualitative snapshot
rebuild. Reuses the exact verified pipeline from eval_geokan_tabletop.py
(same model, same thresholds, same repair module) but isolates one scene
and translates predictions back to named triples instead of integer indices.
"""
import sys
sys.path.insert(0, ".")

import os
import json
import numpy as np
import torch

from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians, remove_table_background
from src.gaussian.clustering import gaussian_to_objects, extract_gaussian_node_features
from scripts.build_3rscan_graphs import extract_3rscan_edge_features
from geokan_gamma_relation import GeoKANGammaRelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, Relation
from src.graph.definitions import Object3D
from src.repair.symbolic_repair import SceneGraphRepair

sys.path.insert(0, "scripts")
from eval_geokan_tabletop import (
    DATA_DIR, MODEL_PATH, THRESHOLDS_PATH, normalize_edge_features,
    create_virtual_table, load_model, load_gt, cluster_scene, build_graph,
    hungarian_match, run_inference, apply_symbolic_repair, GT_RELATION_MAP,
)

_REL_NAME_TO_IDX = {v: k for k, v in RELATION_NAMES.items()}
SCENE = "scene_06"

model, thresholds = load_model()
print(f"Model loaded. Evaluating {SCENE}...")

scene_dir = os.path.join(DATA_DIR, SCENE)
ply_path = os.path.join(scene_dir, "splat.ply")
gt = load_gt(scene_dir)
n_objects_hint = len(gt["objects"])

objects, params = cluster_scene(ply_path, n_objects_hint)
is_z_down = False  # scene_06 is Z-UP per SCENE_Z_FLIP in eval_geokan_tabletop.py
table_obj = create_virtual_table(objects, is_z_down=is_z_down)
has_table = table_obj is not None
if has_table:
    table_obj.uid = len(objects)
    objects = list(objects) + [table_obj]

gt_items_only = [o for o in gt["objects"] if o["name"] != "table"]
gt_table_idx = next((i for i, o in enumerate(gt["objects"]) if o["name"] == "table"), None)
n_match = len(objects) - (1 if has_table else 0)
mapping = hungarian_match(objects[:n_match], gt_items_only)
if has_table and gt_table_idx is not None:
    mapping[len(objects) - 1] = {"gt_idx": gt_table_idx, "gt_name": "table", "distance": 0.0}

print("Hungarian match (cluster -> GT object):")
for idx, info in sorted(mapping.items()):
    print(f"  {idx} -> {info['gt_name']} (d={info['distance']:.3f})")

x, edge_index, edge_attr = build_graph(objects)
table_idx = (len(objects) - 1) if has_table else None
predictions, pred_scores = run_inference(model, x, edge_index, edge_attr, thresholds, table_idx=table_idx)

idx_to_name = {idx: info["gt_name"] for idx, info in mapping.items()}

# Ground truth set (named), normalised through GT_RELATION_MAP so aliases
# like "to_the_left_of" match the canonical "left_of" used in predictions
# -- this was the bug in the first pass (48/84 instead of the manuscript's
# verified 66/84): raw GT strings were compared directly without alias
# normalisation, so every left_of/right_of prediction was wrongly flagged.
gt_named = set()
for rel in gt["relations"]:
    rel_idx = GT_RELATION_MAP.get(rel["relation"])
    if rel_idx is None:
        continue
    canonical_name = RELATION_NAMES[rel_idx]
    gt_named.add((rel["subject"], canonical_name, rel["object"]))

# Pre-repair named predictions
pre_named = []
for (s, r, d) in predictions:
    rel_name = RELATION_NAMES.get(r)
    if rel_name is None:
        continue
    pre_named.append((idx_to_name.get(s, f"node{s}"), rel_name, idx_to_name.get(d, f"node{d}")))

# Apply repair (same as eval_geokan_tabletop.py's per-method repair step)
repaired_set, rstats = apply_symbolic_repair(predictions, pred_scores)
post_named = []
for (s, r, d) in repaired_set:
    rel_name = RELATION_NAMES.get(r)
    if rel_name is None:
        continue
    post_named.append((idx_to_name.get(s, f"node{s}"), rel_name, idx_to_name.get(d, f"node{d}")))

print(f"\nPre-repair: {len(pre_named)} relations")
print(f"Post-repair: {len(post_named)} relations")
print(f"Repair stats: removed={rstats.relations_removed}, added={rstats.relations_added}, iters={rstats.iterations}")

correct = [t for t in post_named if t in gt_named]
incorrect = [t for t in post_named if t not in gt_named]
print(f"\nPost-repair correct: {len(correct)} / {len(post_named)}")

out = {
    "post_repair_relations": [
        {"subject": s, "relation": r, "object": d, "correct": (s, r, d) in gt_named}
        for (s, r, d) in sorted(post_named)
    ],
}
with open("scripts/_scene06_predictions.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved scripts/_scene06_predictions.json")

print("\nAll post-repair relations:")
for s, r, d in sorted(post_named):
    mark = "OK " if (s, r, d) in gt_named else "ERR"
    print(f"  [{mark}] {s} --{r}--> {d}")
