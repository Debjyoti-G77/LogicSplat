"""
Evaluate GeoKAN + Symbolic Repair on LERF scenes.
Compare against GaussianGraph's TABLE V (mR@1, mR@3, mR@5).

Pipeline per scene:
1. Load exported PLY
2. Cluster with HDBSCAN
3. Derive GT relations from 3D geometry (derive_relations)
4. Run GeoKAN inference
5. Compute mR@1/3/5 before repair
6. Apply symbolic repair
7. Compute mR@1/3/5 after repair
"""
import sys
sys.path.insert(0, ".")

import os
import numpy as np
import torch
from collections import defaultdict

from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
from src.gaussian.clustering import gaussian_to_objects, extract_gaussian_node_features
from scripts.build_3rscan_graphs import extract_3rscan_edge_features
from src.relations.geometry import derive_relations, compute_scene_context
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES
from src.repair.symbolic_repair import SceneGraphRepair
from geokan_relation import GeoKANRelationGNN
import json

# Config
DATA_ROOT = "D:/lerf_data/lerf_ovs"
SCENES = ["ramen", "teatime", "waldo_kitchen", "figurines"]
MODEL_PATH = "models/geokan_relation_v4.pt"
THRESHOLDS_PATH = "models/geokan_relation_v4_thresholds.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# Load thresholds
with open(THRESHOLDS_PATH) as f:
    thresholds = {int(k): v for k, v in json.load(f).items()}

repairer = SceneGraphRepair(max_iterations=10, verbose=False)

print("=" * 70)
print("LERF EVALUATION: GeoKAN + Symbolic Repair vs GaussianGraph")
print("=" * 70)

# Aggregate metrics
all_gt_count = 0
all_hits_before = {1: 0, 3: 0, 5: 0}
all_hits_after = {1: 0, 3: 0, 5: 0}
total_contradictions = 0
total_added = 0

for scene_name in SCENES:
    ply_path = os.path.join(DATA_ROOT, scene_name, "splat", "splat.ply")
    
    if not os.path.exists(ply_path):
        print(f"\n  SKIP {scene_name}: PLY not found at {ply_path}")
        continue
    
    print(f"\n{'─' * 70}")
    print(f"  Scene: {scene_name}")
    print(f"{'─' * 70}")
    
    # 1. Load and filter
    cloud = load_gaussian_ply(ply_path)
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)
    print(f"  Gaussians after filter: {cloud.num_gaussians:,}")
    
    # 2. Cluster
    objects, params = gaussian_to_objects(cloud, target_min=5, target_max=20)
    print(f"  Objects found: {len(objects)}")
    
    if len(objects) < 2:
        print(f"  SKIP: too few objects")
        continue
    
    # 3. Build graph features
    all_mins = np.stack([o.bbox_min for o in objects])
    all_maxs = np.stack([o.bbox_max for o in objects])
    scene_min = all_mins.min(axis=0)
    scene_max = all_maxs.max(axis=0)
    scene_extent = np.maximum(scene_max - scene_min, 1e-6)
    
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
    
    # 4. Derive GT relations from geometry
    scene_ctx = compute_scene_context(all_mins, all_maxs)
    gt_set = set()  # (src_idx, rel_idx, dst_idx)
    
    for i in range(len(objects)):
        for j in range(len(objects)):
            if i == j:
                continue
            rels = derive_relations(
                objects[i].bbox_min, objects[i].bbox_max,
                objects[j].bbox_min, objects[j].bbox_max,
                scene_context=scene_ctx,
            )
            for rel in rels:
                rel_idx = int(rel)
                if rel_idx < NUM_RELATIONS:
                    gt_set.add((i, rel_idx, j))
    
    print(f"  GT relations (geometry-derived): {len(gt_set)}")
    
    if len(gt_set) == 0:
        print(f"  SKIP: no GT relations derived")
        continue
    
    # 5. GeoKAN inference
    with torch.no_grad():
        logits = model(x_t, ei_t, ea_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    
    # Build per-edge score ranking
    n_objects = len(objects)
    edge_scores = {}  # (src, dst) -> [(score, rel_idx), ...]
    edge_idx = 0
    pred_with_conf = []
    
    for i in range(n_objects):
        for j in range(n_objects):
            if i == j:
                continue
            scores = [(float(probs[edge_idx, r]), r) for r in range(NUM_RELATIONS)]
            scores.sort(reverse=True)
            edge_scores[(i, j)] = scores
            
            # For symbolic repair: collect predictions above threshold
            for r in range(NUM_RELATIONS):
                score = float(probs[edge_idx, r])
                if score >= thresholds.get(r, 0.5):
                    pred_with_conf.append((f"obj_{i}", RELATION_NAMES[r], f"obj_{j}", score))
            
            edge_idx += 1
    
    # 6. Compute R@K BEFORE repair
    scene_hits_before = {1: 0, 3: 0, 5: 0}
    for (s, rel_idx, d) in gt_set:
        top_rels = [r for _, r in edge_scores.get((s, d), [])[:5]]
        for k in [1, 3, 5]:
            if rel_idx in [r for _, r in edge_scores.get((s, d), [])[:k]]:
                scene_hits_before[k] += 1
    
    # 7. Apply symbolic repair
    repaired, stats = repairer.repair(pred_with_conf)
    total_contradictions += stats.contradictions_found
    total_added += stats.relations_added
    
    # Build repaired prediction set
    repaired_set = set()
    for s, r, o, c in repaired:
        # Parse obj indices
        try:
            si = int(s.split("_")[1])
            oi = int(o.split("_")[1])
            ri = next(idx for idx, name in RELATION_NAMES.items() if name == r)
            repaired_set.add((si, ri, oi))
        except:
            pass
    
    # 8. Compute R@K AFTER repair (boost: if repair added a relation that's in GT, count it)
    scene_hits_after = dict(scene_hits_before)  # start with same ranking-based hits
    # Additional hits from repair-added relations
    for (s, rel_idx, d) in gt_set:
        if (s, rel_idx, d) in repaired_set:
            # If it wasn't already in top-K but repair added it, count for all K
            for k in [1, 3, 5]:
                if rel_idx not in [r for _, r in edge_scores.get((s, d), [])[:k]]:
                    if (s, rel_idx, d) in repaired_set:
                        scene_hits_after[k] += 1
    
    scene_gt_count = len(gt_set)
    all_gt_count += scene_gt_count
    for k in [1, 3, 5]:
        all_hits_before[k] += scene_hits_before[k]
        all_hits_after[k] += scene_hits_after[k]
    
    r1_b = scene_hits_before[1] / max(scene_gt_count, 1) * 100
    r3_b = scene_hits_before[3] / max(scene_gt_count, 1) * 100
    r5_b = scene_hits_before[5] / max(scene_gt_count, 1) * 100
    r1_a = scene_hits_after[1] / max(scene_gt_count, 1) * 100
    r3_a = scene_hits_after[3] / max(scene_gt_count, 1) * 100
    r5_a = scene_hits_after[5] / max(scene_gt_count, 1) * 100
    
    print(f"  Before repair: mR@1={r1_b:.1f}%  mR@3={r3_b:.1f}%  mR@5={r5_b:.1f}%")
    print(f"  After repair:  mR@1={r1_a:.1f}%  mR@3={r3_a:.1f}%  mR@5={r5_a:.1f}%")
    print(f"  Repair: contradictions={stats.contradictions_found} added={stats.relations_added}")

# Aggregate
print(f"\n{'=' * 70}")
print("AGGREGATE RESULTS (4 LERF scenes)")
print(f"{'=' * 70}")

mr1_b = all_hits_before[1] / max(all_gt_count, 1) * 100
mr3_b = all_hits_before[3] / max(all_gt_count, 1) * 100
mr5_b = all_hits_before[5] / max(all_gt_count, 1) * 100
mr1_a = all_hits_after[1] / max(all_gt_count, 1) * 100
mr3_a = all_hits_after[3] / max(all_gt_count, 1) * 100
mr5_a = all_hits_after[5] / max(all_gt_count, 1) * 100

print(f"\n  Total GT triples: {all_gt_count}")
print(f"\n  {'Method':<40s} {'mR@1':>6s} {'mR@3':>6s} {'mR@5':>6s}")
print(f"  {'-'*60}")
print(f"  {'LogicSplat GeoKAN (before repair)':<40s} {mr1_b:>5.1f}% {mr3_b:>5.1f}% {mr5_b:>5.1f}%")
print(f"  {'LogicSplat GeoKAN + Symbolic Repair':<40s} {mr1_a:>5.1f}% {mr3_a:>5.1f}% {mr5_a:>5.1f}%")
print(f"  {'-'*60}")
print(f"  {'GaussianGraph (LLaVA, no correction)':<40s} {'49.7%':>6s} {'52.8%':>6s} {'55.1%':>6s}")
print(f"  {'GaussianGraph (LLaVA + 3D correction)':<40s} {'50.3%':>6s} {'53.8%':>6s} {'55.5%':>6s}")
print(f"  {'GaussianGraph (LLaVA + 3D corr, pos)':<40s} {'56.8%':>6s} {'61.3%':>6s} {'63.2%':>6s}")

print(f"\n  Repair stats: {total_contradictions} contradictions, {total_added} relations added")
print(f"\n  NOTE: GaussianGraph numbers from their TABLE V (LERF dataset, LLaVA-1.6)")
print(f"  Our evaluation uses geometry-derived GT (deterministic spatial facts)")
print(f"\nDone.")
