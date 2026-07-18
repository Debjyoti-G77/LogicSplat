"""
Evaluate the matched-budget MLP-head baseline (trained in train_mlp_baseline.py)
on the 4 LERF scenes, reusing the exact same GT-derivation, clustering, and
repair-impact-on-R@K logic already verified for GeoKAN-Gamma in
results/rerun_lerf_eval.py. Only the model and thresholds are swapped.

Usage:
    python results/eval_mlp_baseline_lerf.py
"""
import sys
sys.path.insert(0, ".")

import os
import json
import numpy as np
import torch
import torch.nn as nn

from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
from src.gaussian.clustering import gaussian_to_objects, extract_gaussian_node_features
from scripts.build_3rscan_graphs import extract_3rscan_edge_features
from src.relations.geometry import derive_relations, compute_scene_context
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES
from src.repair.symbolic_repair import SceneGraphRepair
from train_geokan_variants import GeoKANVariantGNN

DATA_ROOT = "D:/lerf_data/lerf_ovs"
SCENES = ["ramen", "teatime", "waldo_kitchen", "figurines"]
SCENE_N_EXACT = {"ramen": 13, "teatime": 10, "waldo_kitchen": 10, "figurines": 10}
MODEL_PATH = "models/mlp_baseline_relation.pt"
RESULTS_JSON = "results/mlp_baseline_results.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLPCompareLayer(nn.Module):
    """Same definition as in train_mlp_baseline.py -- needed to load the checkpoint."""

    def __init__(self, in_dim: int, out_dim: int, n_bases: int = 12,
                 metric_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        expand_dim = in_dim * n_bases
        self.expand = nn.Sequential(
            nn.Linear(in_dim, expand_dim), nn.GELU(), nn.Dropout(dropout),
        )
        mix_in_dim = expand_dim + in_dim
        self.linear_mix = nn.Sequential(
            nn.Linear(mix_in_dim, out_dim), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, u):
        u_normed = self.bn(u)
        expanded = self.expand(u_normed)
        return self.linear_mix(torch.cat([expanded, u_normed], dim=-1))

    def metric_regularization(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)


state = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
hidden_dim = state["node_encoder.0.weight"].shape[0]
edge_feat_dim = state["conv1.lin_edge.weight"].shape[1]

model = GeoKANVariantGNN(
    layer_cls=MLPCompareLayer,
    node_feat_dim=10, edge_feat_dim=edge_feat_dim,
    hidden_dim=hidden_dim, num_relations=NUM_RELATIONS,
).to(DEVICE)
model.load_state_dict(state, strict=True)
model.eval()
print(f"Model loaded (MLP-baseline): {sum(p.numel() for p in model.parameters()):,} params")

with open(RESULTS_JSON) as f:
    mlp_saved = json.load(f)
thresholds = {int(k): v for k, v in mlp_saved["thresholds"].items()}

repairer = SceneGraphRepair(max_iterations=10, verbose=False)

print("=" * 70)
print("LERF EVALUATION: MLP-baseline + Symbolic Repair")
print("=" * 70)

all_gt_count = 0
all_hits_before = {1: 0, 3: 0, 5: 0}
all_hits_after = {1: 0, 3: 0, 5: 0}
total_contradictions = 0
total_added = 0
per_scene_results = {}
tp_before_sum = fp_before_sum = fn_before_sum = 0
tp_after_sum = fp_after_sum = fn_after_sum = 0


def compute_f1(predictions, gt_set):
    tp = len(predictions & gt_set)
    fp = len(predictions - gt_set)
    fn = len(gt_set - predictions)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


for scene_name in SCENES:
    ply_path = os.path.join(DATA_ROOT, scene_name, "splat", "splat.ply")
    if not os.path.exists(ply_path):
        print(f"\n  SKIP {scene_name}: PLY not found")
        continue

    print(f"\n{'-' * 70}\n  Scene: {scene_name}\n{'-' * 70}")

    cloud = load_gaussian_ply(ply_path)
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)

    n_exact = SCENE_N_EXACT[scene_name]
    objects, params = gaussian_to_objects(
        cloud, target_min=n_exact, target_max=n_exact + 3, n_exact=n_exact,
    )
    print(f"  Objects found: {len(objects)}")
    if len(objects) < 2:
        print(f"  SKIP: too few objects")
        continue

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

    scene_ctx = compute_scene_context(all_mins, all_maxs)
    gt_set = set()
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

    with torch.no_grad():
        logits = model(x_t, ei_t, ea_t)
        probs = torch.sigmoid(logits).cpu().numpy()

    n_objects = len(objects)
    edge_scores = {}
    edge_idx = 0
    pred_with_conf = []
    pred_set_before = set()
    for i in range(n_objects):
        for j in range(n_objects):
            if i == j:
                continue
            scores = [(float(probs[edge_idx, r]), r) for r in range(NUM_RELATIONS)]
            scores.sort(reverse=True)
            edge_scores[(i, j)] = scores
            for r in range(NUM_RELATIONS):
                score = float(probs[edge_idx, r])
                if score >= thresholds.get(r, 0.5):
                    pred_with_conf.append((f"obj_{i}", RELATION_NAMES[r], f"obj_{j}", score))
                    pred_set_before.add((i, r, j))
            edge_idx += 1

    scene_hits_before = {1: 0, 3: 0, 5: 0}
    for (s, rel_idx, d) in gt_set:
        for k in [1, 3, 5]:
            if rel_idx in [r for _, r in edge_scores.get((s, d), [])[:k]]:
                scene_hits_before[k] += 1

    repaired, stats = repairer.repair(pred_with_conf)
    total_contradictions += stats.contradictions_found
    total_added += stats.relations_added

    repaired_set = set()
    for s, r, o, c in repaired:
        try:
            si = int(s.split("_")[1])
            oi = int(o.split("_")[1])
            ri = next(idx for idx, name in RELATION_NAMES.items() if name == r)
            repaired_set.add((si, ri, oi))
        except Exception:
            pass

    scene_hits_after = dict(scene_hits_before)
    for (s, rel_idx, d) in gt_set:
        if (s, rel_idx, d) in repaired_set:
            for k in [1, 3, 5]:
                if rel_idx not in [r for _, r in edge_scores.get((s, d), [])[:k]]:
                    scene_hits_after[k] += 1

    scene_gt_count = len(gt_set)
    all_gt_count += scene_gt_count
    for k in [1, 3, 5]:
        all_hits_before[k] += scene_hits_before[k]
        all_hits_after[k] += scene_hits_after[k]

    r_b = {k: scene_hits_before[k] / max(scene_gt_count, 1) for k in [1, 3, 5]}
    r_a = {k: scene_hits_after[k] / max(scene_gt_count, 1) for k in [1, 3, 5]}

    f1_before = compute_f1(pred_set_before, gt_set)
    f1_after = compute_f1(repaired_set, gt_set)
    tp_before_sum += f1_before["tp"]; fp_before_sum += f1_before["fp"]; fn_before_sum += f1_before["fn"]
    tp_after_sum += f1_after["tp"]; fp_after_sum += f1_after["fp"]; fn_after_sum += f1_after["fn"]

    print(f"  Before repair: mR@1={r_b[1]*100:.1f}%  mR@3={r_b[3]*100:.1f}%  mR@5={r_b[5]*100:.1f}%  F1={f1_before['f1']:.4f}")
    print(f"  After repair:  mR@1={r_a[1]*100:.1f}%  mR@3={r_a[3]*100:.1f}%  mR@5={r_a[5]*100:.1f}%  F1={f1_after['f1']:.4f}")
    print(f"  Repair: contradictions={stats.contradictions_found} added={stats.relations_added}")

    per_scene_results[scene_name] = {
        "objects": len(objects), "gt_triples": scene_gt_count,
        "mr1_before": r_b[1], "mr3_before": r_b[3], "mr5_before": r_b[5],
        "mr1_after": r_a[1], "mr3_after": r_a[3], "mr5_after": r_a[5],
        "f1_before": f1_before["f1"], "f1_after": f1_after["f1"],
    }

mr_b = {k: all_hits_before[k] / max(all_gt_count, 1) for k in [1, 3, 5]}
mr_a = {k: all_hits_after[k] / max(all_gt_count, 1) for k in [1, 3, 5]}

micro_p_before = tp_before_sum / max(tp_before_sum + fp_before_sum, 1)
micro_r_before = tp_before_sum / max(tp_before_sum + fn_before_sum, 1)
micro_f1_before = 2 * micro_p_before * micro_r_before / max(micro_p_before + micro_r_before, 1e-9)

micro_p_after = tp_after_sum / max(tp_after_sum + fp_after_sum, 1)
micro_r_after = tp_after_sum / max(tp_after_sum + fn_after_sum, 1)
micro_f1_after = 2 * micro_p_after * micro_r_after / max(micro_p_after + micro_r_after, 1e-9)

print(f"\n{'=' * 70}\nAGGREGATE RESULTS (4 LERF scenes) -- MLP-baseline\n{'=' * 70}")
print(f"\n  Total GT triples: {all_gt_count}")
print(f"  Before repair: mR@1={mr_b[1]*100:.1f}%  mR@3={mr_b[3]*100:.1f}%  mR@5={mr_b[5]*100:.1f}%  "
      f"Micro F1={micro_f1_before:.4f} (P={micro_p_before:.4f} R={micro_r_before:.4f})")
print(f"  After repair:  mR@1={mr_a[1]*100:.1f}%  mR@3={mr_a[3]*100:.1f}%  mR@5={mr_a[5]*100:.1f}%  "
      f"Micro F1={micro_f1_after:.4f} (P={micro_p_after:.4f} R={micro_r_after:.4f})")
print(f"  Repair stats: {total_contradictions} contradictions, {total_added} relations added")

with open("results/mlp_baseline_lerf_results.json", "w") as f:
    json.dump({
        "per_scene": per_scene_results,
        "aggregate": {
            "total_gt_triples": all_gt_count,
            "mr1_before": mr_b[1], "mr3_before": mr_b[3], "mr5_before": mr_b[5],
            "mr1_after": mr_a[1], "mr3_after": mr_a[3], "mr5_after": mr_a[5],
            "micro_f1_before": micro_f1_before, "micro_f1_after": micro_f1_after,
            "total_contradictions": total_contradictions, "total_added": total_added,
        },
    }, f, indent=2)
print("\nSaved -> results/mlp_baseline_lerf_results.json")
