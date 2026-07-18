"""
Evaluate symbolic repair impact on GeoKAN v4 predictions (validation set).
Compares before/after repair: F1, contradictions removed, relations added.
"""
import sys
sys.path.insert(0, ".")

import os
import json
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from collections import defaultdict

from geokan_relation import GeoKANRelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES
from src.repair.symbolic_repair import SceneGraphRepair, compute_metrics

# Load graphs
cache_dir = "D:/logicsplat_data/3rscan_graph_cache"
files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
print(f"Loading {len(files)} graphs...")

graphs = []
for fname in files:
    g = torch.load(os.path.join(cache_dir, fname), weights_only=False)
    graphs.append(g)

# Same split as training (85% train, 15% val)
rng = np.random.default_rng(42)
indices = np.arange(len(graphs))
rng.shuffle(indices)
train_size = int(len(graphs) * 0.85)
val_idx = indices[train_size:]
val_graphs = [graphs[i] for i in val_idx]
print(f"Val set: {len(val_graphs)} scenes")

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GeoKANRelationGNN(
    node_feat_dim=10, edge_feat_dim=22, hidden_dim=128,
    num_relations=NUM_RELATIONS, dropout=0.2
).to(device)
model.load_state_dict(torch.load("models/geokan_relation_v4.pt", map_location=device, weights_only=True))
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# Load thresholds
with open("models/geokan_relation_v4_thresholds.json") as f:
    thresholds = {int(k): v for k, v in json.load(f).items()}

# Run inference on val set, apply thresholds, then run repair
repairer = SceneGraphRepair(max_iterations=10, verbose=False)

total_before_repair = {"tp": 0, "fp": 0, "fn": 0}
total_after_repair = {"tp": 0, "fp": 0, "fn": 0}
total_contradictions = 0
total_removed = 0
total_added = 0
total_iterations = 0
n_scenes = 0

print(f"\nRunning inference + repair on {len(val_graphs)} val scenes...")
print("=" * 70)

for g_idx, g in enumerate(val_graphs):
    x = g["x"].to(device)
    edge_index = g["edge_index"].to(device)
    edge_attr = g["edge_attr"].to(device)
    edge_label = g["edge_label"]  # (E, 10)
    
    n_nodes = x.shape[0]
    n_edges = edge_index.shape[1]
    
    with torch.no_grad():
        logits = model(x, edge_index, edge_attr)
        probs = torch.sigmoid(logits).cpu()
    
    # Apply thresholds
    preds = torch.zeros_like(probs)
    for rel_idx in range(NUM_RELATIONS):
        thresh = thresholds.get(rel_idx, 0.5)
        preds[:, rel_idx] = (probs[:, rel_idx] >= thresh).float()
    
    # Build GT set and prediction set as named triples
    src_list = edge_index[0].cpu().tolist()
    dst_list = edge_index[1].cpu().tolist()
    
    gt_set = set()
    pred_set_before = set()
    pred_triples_with_conf = []
    
    for e_idx in range(n_edges):
        src = src_list[e_idx]
        dst = dst_list[e_idx]
        
        # GT
        for rel_idx in range(NUM_RELATIONS):
            if edge_label[e_idx, rel_idx] >= 0.5:
                gt_set.add((f"obj_{src}", RELATION_NAMES[rel_idx], f"obj_{dst}"))
        
        # Predictions (before repair)
        for rel_idx in range(NUM_RELATIONS):
            if preds[e_idx, rel_idx] == 1:
                rel_name = RELATION_NAMES[rel_idx]
                conf = float(probs[e_idx, rel_idx])
                pred_set_before.add((f"obj_{src}", rel_name, f"obj_{dst}"))
                pred_triples_with_conf.append((f"obj_{src}", rel_name, f"obj_{dst}", conf))
    
    if len(gt_set) == 0:
        continue
    
    # Compute before-repair metrics
    tp_before = len(pred_set_before & gt_set)
    fp_before = len(pred_set_before - gt_set)
    fn_before = len(gt_set - pred_set_before)
    
    # Run symbolic repair
    repaired_triples, stats = repairer.repair(pred_triples_with_conf)
    
    # Build after-repair prediction set
    pred_set_after = set((s, r, o) for s, r, o, c in repaired_triples)
    
    tp_after = len(pred_set_after & gt_set)
    fp_after = len(pred_set_after - gt_set)
    fn_after = len(gt_set - pred_set_after)
    
    total_before_repair["tp"] += tp_before
    total_before_repair["fp"] += fp_before
    total_before_repair["fn"] += fn_before
    total_after_repair["tp"] += tp_after
    total_after_repair["fp"] += fp_after
    total_after_repair["fn"] += fn_after
    total_contradictions += stats.contradictions_found
    total_removed += stats.relations_removed
    total_added += stats.relations_added
    total_iterations += stats.iterations
    n_scenes += 1

# Compute aggregate metrics
def compute_f1(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return p, r, f1

p_before, r_before, f1_before = compute_f1(
    total_before_repair["tp"], total_before_repair["fp"], total_before_repair["fn"]
)
p_after, r_after, f1_after = compute_f1(
    total_after_repair["tp"], total_after_repair["fp"], total_after_repair["fn"]
)

print(f"\n{'=' * 70}")
print("SYMBOLIC REPAIR EVALUATION (GeoKAN v4, 3DSSG Val Set)")
print(f"{'=' * 70}")
print(f"\nScenes evaluated: {n_scenes}")
print(f"\n{'Metric':<25s} {'Before Repair':>15s} {'After Repair':>15s} {'Delta':>10s}")
print("-" * 65)
print(f"{'Precision':<25s} {p_before:>15.4f} {p_after:>15.4f} {p_after-p_before:>+10.4f}")
print(f"{'Recall':<25s} {r_before:>15.4f} {r_after:>15.4f} {r_after-r_before:>+10.4f}")
print(f"{'F1':<25s} {f1_before:>15.4f} {f1_after:>15.4f} {f1_after-f1_before:>+10.4f}")
print(f"{'TP':<25s} {total_before_repair['tp']:>15d} {total_after_repair['tp']:>15d} {total_after_repair['tp']-total_before_repair['tp']:>+10d}")
print(f"{'FP':<25s} {total_before_repair['fp']:>15d} {total_after_repair['fp']:>15d} {total_after_repair['fp']-total_before_repair['fp']:>+10d}")
print(f"{'FN':<25s} {total_before_repair['fn']:>15d} {total_after_repair['fn']:>15d} {total_after_repair['fn']-total_before_repair['fn']:>+10d}")

print(f"\n{'=' * 70}")
print("REPAIR STATISTICS")
print(f"{'=' * 70}")
print(f"  Total contradictions found:  {total_contradictions}")
print(f"  Total relations removed:     {total_removed}")
print(f"  Total relations added:       {total_added}")
print(f"  Mean contradictions/scene:   {total_contradictions/max(n_scenes,1):.1f}")
print(f"  Mean relations removed/scene:{total_removed/max(n_scenes,1):.1f}")
print(f"  Mean relations added/scene:  {total_added/max(n_scenes,1):.1f}")
print(f"  Mean iterations/scene:       {total_iterations/max(n_scenes,1):.1f}")

print(f"\n{'=' * 70}")
print("COMPARISON WITH GAUSSIANGRAPH")
print(f"{'=' * 70}")
print(f"""
  Aspect                    | GaussianGraph (3D Correction) | LogicSplat (Symbolic Repair)
  --------------------------+-------------------------------+-----------------------------
  Method                    | Learned geometric correction  | Deterministic constraint prop.
  Parameters                | Trainable (part of model)     | Zero (pure logic)
  Guarantees                | No formal guarantees          | Guaranteed consistency
  Requires training         | Yes                           | No
  F1 improvement            | Not reported on same metric   | {f1_after-f1_before:+.4f}
  Contradictions resolved   | Implicit (learned)            | {total_contradictions} explicit
  Relations added (inverses)| No                            | {total_added}
  Convergence               | Single pass                   | Fixed-point ({total_iterations/max(n_scenes,1):.1f} avg iter)
""")

print("Done.")
