"""
Diagnose why in_front_of/behind have F1~0.16 while left_of/right_of have F1~0.33.

Hypothesis: ScanNet Y axis (depth) has no consistent world-frame meaning
across scenes — each scene scanned from different direction.

Run: python scripts/analyze_directional_bias.py
"""
import sys, os
sys.path.insert(0, ".")

import torch
import numpy as np
from collections import Counter, defaultdict
from src.dataset.loader_scannet import SceneGraphDatasetScanNet
from src.relations.schema import RELATION_NAMES, Relation

SCANNET_DIR = "D:/scannet/scans"
CACHE_DIR   = "data/scannet_cache"

print("Loading dataset from cache...")
ds = SceneGraphDatasetScanNet(scannet_dir=SCANNET_DIR, cache_dir=CACHE_DIR, verbose=False)
print(f"Loaded {len(ds)} scenes\n")

# ── 1. Class distribution ─────────────────────────────────────────────────────
counts = Counter()
for g in ds.graphs:
    counts.update(g["edge_label"].tolist())

total = sum(counts.values())
print("=== Class Distribution ===")
for idx in sorted(counts):
    pct = 100 * counts[idx] / total
    print(f"  {RELATION_NAMES[idx]:20s}  {counts[idx]:7,}  ({pct:5.1f}%)")

# ── 2. Edge feature stats per relation ───────────────────────────────────────
print("\n=== Edge Feature [0]=delta_z  [1]=xy_dist  [2]=dist_3d  per relation ===")
feat_by_rel = defaultdict(list)
for g in ds.graphs:
    labels = g["edge_label"].tolist()
    feats  = g["edge_attr"].numpy()
    for i, lbl in enumerate(labels):
        feat_by_rel[lbl].append(feats[i])

for idx in sorted(feat_by_rel):
    arr = np.array(feat_by_rel[idx])
    print(f"  {RELATION_NAMES[idx]:20s}  "
          f"delta_z: {arr[:,0].mean():+.3f}±{arr[:,0].std():.3f}  "
          f"xy_dist: {arr[:,1].mean():.3f}±{arr[:,1].std():.3f}  "
          f"dist_3d: {arr[:,2].mean():.3f}±{arr[:,2].std():.3f}")

# ── 3. Separability: can delta_z alone distinguish left_of vs in_front_of? ───
print("\n=== Separability check: left_of vs in_front_of ===")
lof = np.array(feat_by_rel[Relation.LEFT_OF])
ifo = np.array(feat_by_rel[Relation.IN_FRONT_OF])
print(f"  left_of    xy_dist mean={lof[:,1].mean():.3f}  delta_z mean={lof[:,0].mean():+.3f}")
print(f"  in_front_of xy_dist mean={ifo[:,1].mean():.3f}  delta_z mean={ifo[:,0].mean():+.3f}")
print(f"  → These look similar to the model because both are 'far in XY, small delta_z'")
print(f"  → The model needs the XY *direction* (angle), not just distance")

# ── 4. Check: are in_front_of/behind labels consistent across scenes? ─────────
print("\n=== Consistency check: do in_front_of edges cluster in Y direction? ===")
# For each in_front_of edge, check if node_A centroid_y < node_B centroid_y
# node features: [0]=cx, [1]=cy, [2]=cz (normalized)
y_diffs_ifo, y_diffs_beh = [], []
for g in ds.graphs:
    labels = g["edge_label"].tolist()
    x      = g["x"].numpy()
    ei     = g["edge_index"].numpy()
    for k, lbl in enumerate(labels):
        src, dst = ei[0, k], ei[1, k]
        dy = float(x[src, 1] - x[dst, 1])  # centroid_y difference
        if lbl == Relation.IN_FRONT_OF:
            y_diffs_ifo.append(dy)
        elif lbl == Relation.BEHIND:
            y_diffs_beh.append(dy)

y_ifo = np.array(y_diffs_ifo)
y_beh = np.array(y_diffs_beh)
print(f"  in_front_of: delta_cy = {y_ifo.mean():+.4f} ± {y_ifo.std():.4f}  "
      f"(should be consistently negative if Y axis is consistent)")
print(f"  behind:      delta_cy = {y_beh.mean():+.4f} ± {y_beh.std():.4f}  "
      f"(should be consistently positive)")
print(f"  → If std >> |mean|, the Y axis is inconsistent across scenes (confirmed bug)")
print(f"  → Ratio |mean|/std: ifo={abs(y_ifo.mean())/y_ifo.std():.3f}  "
      f"beh={abs(y_beh.mean())/y_beh.std():.3f}  (>0.5 = learnable, <0.2 = noise)")
