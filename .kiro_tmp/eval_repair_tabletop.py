"""
Evaluate symbolic repair on cross-domain tabletop scenes (scene_06..scene_13).
Runs GeoKAN inference → thresholds → repair → compare before/after.
"""
import sys
sys.path.insert(0, ".")

import os
import json
import numpy as np
import torch
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
from src.gaussian.clustering import gaussian_to_objects, extract_gaussian_node_features
from scripts.build_3rscan_graphs import extract_3rscan_edge_features
from geokan_relation import GeoKANRelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, Relation
from src.repair.symbolic_repair import SceneGraphRepair

DATA_DIR = "D:/logicsplat_data/processed"
MODEL_PATH = "models/geokan_relation_v4.pt"
THRESHOLDS_PATH = "models/geokan_relation_v4_thresholds.json"
SCENES = [f"scene_{i:02d}" for i in range(6, 14)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GT_RELATION_MAP = {
    "on_top_of": int(Relation.ON_TOP_OF),
    "under": int(Relation.UNDER),
    "attached_to": int(Relation.ATTACHED_TO),
    "adjacent_to": int(Relation.ADJACENT_TO),
    "to_the_left_of": int(Relation.LEFT_OF),
    "left_of": int(Relation.LEFT_OF),
    "to_the_right_of": int(Relation.RIGHT_OF),
    "right_of": int(Relation.RIGHT_OF),
    "in_front_of": int(Relation.IN_FRONT_OF),
    "behind": int(Relation.BEHIND),
    "higher_than": int(Relation.HIGHER_THAN),
    "lower_than": int(Relation.LOWER_THAN),
}

# Load model
state = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
hidden_dim = state["node_encoder.0.weight"].shape[0]
edge_feat_dim = state["conv1.lin_edge.weight"].shape[1]

model = GeoKANRelationGNN(
    node_feat_dim=10, edge_feat_dim=edge_feat_dim,
    hidden_dim=hidden_dim, num_relations=NUM_RELATIONS,
).to(DEVICE)
model.load_state_dict(state)
model.eval()

with open(THRESHOLDS_PATH) as f:
    thresholds = {int(k): v for k, v in json.load(f).items()}

repairer = SceneGraphRepair(max_iterations=10, verbose=False)

print("=" * 70)
print("SYMBOLIC REPAIR ON CROSS-DOMAIN TABLETOP SCENES")
print("=" * 70)

total_before = {"tp": 0, "fp": 0, "fn": 0}
total_after = {"tp": 0, "fp": 0, "fn": 0}
total_stats = {"contradictions": 0, "removed": 0, "added": 0, "iterations": 0}
n_scenes = 0

for scene_name in SCENES:
    scene_dir = os.path.join(DATA_DIR, scene_name)
    ply_path = os.path.join(scene_dir, "splat.ply")
    gt_path = os.path.join(scene_dir, "ground_truth_relations.json")

    if not os.path.exists(ply_path) or not os.path.exists(gt_path):
        continue

    # Load GT
    with open(gt_path) as f:
        gt_data = json.load(f)

    n_objects_hint = len(gt_data["objects"])

    # Cluster
    cloud = load_gaussian_ply(ply_path)
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)

    objects, params = gaussian_to_objects(
        cloud, target_min=max(2, n_objects_hint - 1), target_max=n_objects_hint + 1
    )

    if len(objects) < 2:
        continue

    # Hungarian match
    gt_objects = gt_data["objects"]
    pred_centroids = np.array([o.centroid for o in objects])
    gt_centroids = []
    for go in gt_objects:
        if "centroid" in go:
            gt_centroids.append(np.array(go["centroid"]))

    if not gt_centroids:
        continue

    gt_centroids_arr = np.array(gt_centroids)
    cost = np.zeros((len(objects), len(gt_centroids)))
    for i in range(len(objects)):
        for j in range(len(gt_centroids)):
            cost[i, j] = np.linalg.norm(pred_centroids[i] - gt_centroids_arr[j])

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {}  # pred_idx → gt_name
    for r, c in zip(row_ind, col_ind):
        mapping[r] = gt_objects[c]["name"]

    name_to_pred_idx = {v: k for k, v in mapping.items()}

    # Build graph
    all_mins = np.stack([o.bbox_min for o in objects])
    all_maxs = np.stack([o.bbox_max for o in objects])
    scene_extent = np.maximum(all_maxs.max(axis=0) - all_mins.min(axis=0), 1e-6)
    scene_min = all_mins.min(axis=0)

    obj_sizes = [np.maximum(o.size, 1e-6) for o in objects]
    obj_diags = [float(np.linalg.norm(s)) for s in obj_sizes]
    scene_mean_diag = float(np.mean(obj_diags))
    obj_volumes = [float(np.prod(s)) for s in obj_sizes]
    scene_median_volume = float(np.median(obj_volumes))

    centroid_zs = np.array([o.centroid[2] for o in objects])
    sorted_z_idx = np.argsort(centroid_zs)
    z_ranks = np.zeros(len(objects))
    if len(objects) > 1:
        for rank, obj_idx in enumerate(sorted_z_idx):
            z_ranks[obj_idx] = rank / (len(objects) - 1)

    x = np.stack([
        extract_gaussian_node_features(o, scene_extent, scene_min,
                                       scene_mean_diag=scene_mean_diag,
                                       scene_median_volume=scene_median_volume,
                                       z_rank=float(z_ranks[i]))
        for i, o in enumerate(objects)
    ])

    src, dst, edge_feats = [], [], []
    for i in range(len(objects)):
        for j in range(len(objects)):
            if i != j:
                src.append(i)
                dst.append(j)
                edge_feats.append(extract_3rscan_edge_features(objects[i], objects[j], scene_extent))

    x_t = torch.tensor(x, dtype=torch.float32).to(DEVICE)
    ei_t = torch.tensor([src, dst], dtype=torch.long).to(DEVICE)
    ea_t = torch.tensor(np.stack(edge_feats), dtype=torch.float32).to(DEVICE)

    # Inference
    with torch.no_grad():
        logits = model(x_t, ei_t, ea_t)
        probs = torch.sigmoid(logits).cpu()

    # Apply thresholds → predictions before repair
    pred_triples_before = set()
    pred_with_conf = []

    edge_idx = 0
    for i in range(len(objects)):
        for j in range(len(objects)):
            if i == j:
                continue
            for rel_idx in range(NUM_RELATIONS):
                score = float(probs[edge_idx, rel_idx])
                thresh = thresholds.get(rel_idx, 0.5)
                if score >= thresh:
                    src_name = mapping.get(i, f"obj_{i}")
                    dst_name = mapping.get(j, f"obj_{j}")
                    pred_triples_before.add((src_name, RELATION_NAMES[rel_idx], dst_name))
                    pred_with_conf.append((src_name, RELATION_NAMES[rel_idx], dst_name, score))
            edge_idx += 1

    # Build GT set
    gt_set = set()
    for rel in gt_data["relations"]:
        rel_name = rel["relation"]
        if rel_name == "to_the_left_of":
            rel_name = "left_of"
        elif rel_name == "to_the_right_of":
            rel_name = "right_of"
        subj = rel["subject"]
        obj = rel["object"]
        if subj in name_to_pred_idx and obj in name_to_pred_idx:
            gt_set.add((subj, rel_name, obj))

    if len(gt_set) == 0:
        continue

    # Run repair
    repaired, stats = repairer.repair(pred_with_conf)
    pred_triples_after = set((s, r, o) for s, r, o, c in repaired)

    # Compute metrics
    tp_b = len(pred_triples_before & gt_set)
    fp_b = len(pred_triples_before - gt_set)
    fn_b = len(gt_set - pred_triples_before)

    tp_a = len(pred_triples_after & gt_set)
    fp_a = len(pred_triples_after - gt_set)
    fn_a = len(gt_set - pred_triples_after)

    total_before["tp"] += tp_b
    total_before["fp"] += fp_b
    total_before["fn"] += fn_b
    total_after["tp"] += tp_a
    total_after["fp"] += fp_a
    total_after["fn"] += fn_a
    total_stats["contradictions"] += stats.contradictions_found
    total_stats["removed"] += stats.relations_removed
    total_stats["added"] += stats.relations_added
    total_stats["iterations"] += stats.iterations
    n_scenes += 1

    # Per-scene
    p_b = tp_b / max(tp_b + fp_b, 1)
    r_b = tp_b / max(tp_b + fn_b, 1)
    f1_b = 2 * p_b * r_b / max(p_b + r_b, 1e-9)
    p_a = tp_a / max(tp_a + fp_a, 1)
    r_a = tp_a / max(tp_a + fn_a, 1)
    f1_a = 2 * p_a * r_a / max(p_a + r_a, 1e-9)

    print(f"  {scene_name}: F1 {f1_b:.3f} → {f1_a:.3f} ({f1_a-f1_b:+.3f}) | "
          f"contradictions={stats.contradictions_found} removed={stats.relations_removed} "
          f"added={stats.relations_added}")

# Aggregate
def f1(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-9), p, r

f1_b, p_b, r_b = f1(total_before["tp"], total_before["fp"], total_before["fn"])
f1_a, p_a, r_a = f1(total_after["tp"], total_after["fp"], total_after["fn"])

print(f"\n{'=' * 70}")
print("AGGREGATE RESULTS (8 tabletop scenes)")
print(f"{'=' * 70}")
print(f"\n{'Metric':<20s} {'Before Repair':>15s} {'After Repair':>15s} {'Delta':>10s}")
print("-" * 60)
print(f"{'Micro F1':<20s} {f1_b:>15.4f} {f1_a:>15.4f} {f1_a-f1_b:>+10.4f}")
print(f"{'Precision':<20s} {p_b:>15.4f} {p_a:>15.4f} {p_a-p_b:>+10.4f}")
print(f"{'Recall':<20s} {r_b:>15.4f} {r_a:>15.4f} {r_a-r_b:>+10.4f}")
print(f"{'TP':<20s} {total_before['tp']:>15d} {total_after['tp']:>15d}")
print(f"{'FP':<20s} {total_before['fp']:>15d} {total_after['fp']:>15d}")
print(f"{'FN':<20s} {total_before['fn']:>15d} {total_after['fn']:>15d}")

print(f"\n  Repair stats:")
print(f"    Contradictions: {total_stats['contradictions']} ({total_stats['contradictions']/max(n_scenes,1):.1f}/scene)")
print(f"    Removed:        {total_stats['removed']} ({total_stats['removed']/max(n_scenes,1):.1f}/scene)")
print(f"    Added:          {total_stats['added']} ({total_stats['added']/max(n_scenes,1):.1f}/scene)")
print(f"    Avg iterations: {total_stats['iterations']/max(n_scenes,1):.1f}")
print(f"\nDone. {n_scenes} scenes evaluated.")
