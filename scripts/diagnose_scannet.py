"""Diagnose the ScanNet dataset — check class balance, feature ranges, split sizes."""
import sys
sys.path.insert(0, ".")
import torch
import numpy as np
from collections import Counter
from torch.utils.data import random_split
from src.dataset.loader_scannet import SceneGraphDatasetScanNet
from src.relations.schema import RELATION_NAMES, NUM_RELATIONS

ds = SceneGraphDatasetScanNet(verbose=False)
print(f"Total scenes: {len(ds)}")
print(f"NUM_RELATIONS: {NUM_RELATIONS}")

# count all edges and labels
all_labels = []
all_x = []
all_ea = []
for g in ds.graphs:
    all_labels.extend(g["edge_label"].tolist())
    all_x.append(g["x"])
    all_ea.append(g["edge_attr"])

print(f"\nTotal edges: {len(all_labels)}")
counts = Counter(all_labels)
total = len(all_labels)
print("\nRelation distribution:")
for idx in range(NUM_RELATIONS):
    c = counts.get(idx, 0)
    pct = 100 * c / total
    print(f"  [{idx:2d}] {RELATION_NAMES[idx]:20s}  {c:6d}  ({pct:.1f}%)")

# check split sizes
test_size  = max(1, int(len(ds) * 0.15))
val_size   = max(1, int(len(ds) * 0.15))
train_size = len(ds) - val_size - test_size
print(f"\nSplit: train={train_size} val={val_size} test={test_size}")
print(f"With only {len(ds)} scenes, test set = {test_size} scene(s)")

# check feature ranges
x_all = torch.cat(all_x, dim=0).numpy()
ea_all = torch.cat(all_ea, dim=0).numpy()
print(f"\nNode features (x): shape={x_all.shape}")
print(f"  min={x_all.min():.3f}  max={x_all.max():.3f}  mean={x_all.mean():.3f}")
print(f"  any NaN: {np.isnan(x_all).any()}  any Inf: {np.isinf(x_all).any()}")

print(f"\nEdge features: shape={ea_all.shape}")
print(f"  min={ea_all.min():.3f}  max={ea_all.max():.3f}  mean={ea_all.mean():.3f}")
print(f"  any NaN: {np.isnan(ea_all).any()}  any Inf: {np.isinf(ea_all).any()}")

# check objects per scene
n_objs = [g["x"].shape[0] for g in ds.graphs]
print(f"\nObjects per scene: min={min(n_objs)} max={max(n_objs)} mean={np.mean(n_objs):.1f}")
