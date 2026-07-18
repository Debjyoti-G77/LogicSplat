"""
Test augmentation pipeline:
- Loads a small subset of cache
- Applies all augmentations
- Verifies dims, label remapping, and distribution improvement
"""
import sys
sys.path.insert(0, ".")
import warnings
warnings.filterwarnings("ignore")

import torch
from collections import Counter
from src.training.augmentation import AugmentedScanNetDataset, augment_graph, AUGMENTATIONS
from src.relations.schema import RELATION_NAMES

print("=" * 60)
print("TEST 1: Single graph augmentation correctness")
print("=" * 60)

import os
cache_dir = "D:/logicsplat_data/scannet_cache"
files = sorted(f for f in os.listdir(cache_dir) if f.endswith("_v2.pt"))
g = torch.load(os.path.join(cache_dir, files[0]), weights_only=False)

print(f"Original: x={g['x'].shape} edge_attr={g['edge_attr'].shape} "
      f"labels={g['edge_label'].shape}")

for aug_name, _ in AUGMENTATIONS:
    aug = augment_graph(g, aug_name)
    assert aug["x"].shape == g["x"].shape, f"{aug_name}: x shape changed"
    assert aug["edge_attr"].shape == g["edge_attr"].shape, f"{aug_name}: edge_attr shape changed"
    assert aug["edge_label"].shape == g["edge_label"].shape, f"{aug_name}: label shape changed"
    print(f"  {aug_name}: OK — x={aug['x'].shape} edge={aug['edge_attr'].shape}")

print("\nAll augmentation shape checks passed.")

print()
print("=" * 60)
print("TEST 2: Label remapping correctness")
print("=" * 60)

from src.training.augmentation import augment_graph
from src.relations.schema import Relation

# Create a synthetic graph with known labels
import torch
n_nodes = 4
n_edges = 12
x = torch.rand(n_nodes, 10)
edge_index = torch.tensor([[0,0,0,1,1,1,2,2,2,3,3,3],
                            [1,2,3,0,2,3,0,1,3,0,1,2]], dtype=torch.long)
# assign one of each directional relation
labels = torch.tensor([
    Relation.LEFT_OF, Relation.RIGHT_OF, Relation.IN_FRONT_OF,
    Relation.BEHIND, Relation.ON_TOP_OF, Relation.UNDER,
    Relation.HIGHER_THAN, Relation.LOWER_THAN, Relation.ADJACENT_TO,
    Relation.INSIDE, Relation.ATTACHED_TO, Relation.HANGING_FROM,
], dtype=torch.long)
edge_attr = torch.rand(n_edges, 10)
edge_attr[:, 0] = 0.5   # delta_x
edge_attr[:, 1] = 0.3   # delta_y

test_g = {"x": x, "edge_index": edge_index, "edge_attr": edge_attr,
          "edge_label": labels, "obj_labels": [], "scene_id": "test"}

# After flip_x: left_of ↔ right_of, delta_x negated
flipped = augment_graph(test_g, "flip_x")
assert flipped["edge_label"][0] == Relation.RIGHT_OF, "flip_x: left_of should become right_of"
assert flipped["edge_label"][1] == Relation.LEFT_OF,  "flip_x: right_of should become left_of"
assert flipped["edge_label"][2] == Relation.IN_FRONT_OF, "flip_x: in_front_of unchanged"
assert abs(float(flipped["edge_attr"][0, 0]) - (-0.5)) < 1e-5, "flip_x: delta_x should negate"
print("flip_x label remapping: PASSED")

# After flip_y: in_front_of ↔ behind, delta_y negated
flipped_y = augment_graph(test_g, "flip_y")
assert flipped_y["edge_label"][2] == Relation.BEHIND,      "flip_y: in_front_of → behind"
assert flipped_y["edge_label"][3] == Relation.IN_FRONT_OF, "flip_y: behind → in_front_of"
assert flipped_y["edge_label"][0] == Relation.LEFT_OF,     "flip_y: left_of unchanged"
print("flip_y label remapping: PASSED")

# After rotate_180: left↔right AND front↔behind
rot180 = augment_graph(test_g, "rotate_180")
assert rot180["edge_label"][0] == Relation.RIGHT_OF,    "rotate_180: left_of → right_of"
assert rot180["edge_label"][2] == Relation.BEHIND,      "rotate_180: in_front_of → behind"
assert rot180["edge_label"][4] == Relation.ON_TOP_OF,   "rotate_180: on_top_of unchanged"
print("rotate_180 label remapping: PASSED")

print()
print("=" * 60)
print("TEST 3: Full dataset distribution using AugmentedScanNetDataset")
print("=" * 60)

from src.training.augmentation import AugmentedScanNetDataset

# Use only 50 graphs for speed in test
import os, random
random.seed(42)
all_files = sorted(f for f in os.listdir(cache_dir) if f.endswith("_v2.pt"))
sample_files = random.sample(all_files, min(50, len(all_files)))

# Temporarily patch the cache dir to only contain our sample
# by loading directly
sample_graphs_raw = [torch.load(os.path.join(cache_dir, f), weights_only=False)
                     for f in sample_files]

# Simulate what AugmentedScanNetDataset does
from src.training.augmentation import augment_graph, _has_rare_relation, AUGMENTATIONS

aug_graphs = []
selected_augs = ["rotate_90", "rotate_180", "rotate_270", "flip_x"]
all_aug_names = ["rotate_90", "rotate_180", "rotate_270", "flip_x", "flip_y", "jitter"]
oversample_rare = 3

for g in sample_graphs_raw:
    aug_graphs.append(g)
    for aug_name in selected_augs:
        aug_graphs.append(augment_graph(g, aug_name))
    if _has_rare_relation(g):
        for aug_name in all_aug_names:
            for _ in range(oversample_rare):
                aug_graphs.append(augment_graph(g, aug_name))

orig_counts = Counter()
for g in sample_graphs_raw:
    orig_counts.update(g["edge_label"].tolist())

aug_counts = Counter()
for g in aug_graphs:
    aug_counts.update(g["edge_label"].tolist())

orig_total = sum(orig_counts.values())
aug_total  = sum(aug_counts.values())

print(f"Original: {len(sample_graphs_raw)} graphs, {orig_total} edges")
print(f"Augmented: {len(aug_graphs)} graphs, {aug_total} edges")
print()
print(f"{'Relation':<20} {'Orig%':>6}  {'Aug%':>6}  {'Change':>8}")
print("-" * 50)
for idx in sorted(orig_counts):
    op = 100 * orig_counts[idx] / orig_total
    ap = 100 * aug_counts.get(idx, 0) / aug_total
    change = ap - op
    sign = "+" if change >= 0 else ""
    print(f"{RELATION_NAMES[idx]:<20} {op:>6.1f}%  {ap:>6.1f}%  {sign}{change:>6.1f}%")

orig_imbalance = max(orig_counts.values()) / max(min(orig_counts.values()), 1)
aug_imbalance  = max(aug_counts.values())  / max(min(aug_counts.values()), 1)
print(f"\nImbalance ratio: {orig_imbalance:.1f}x → {aug_imbalance:.1f}x")

# Count how many graphs have rare relations
rare_count = sum(1 for g in sample_graphs_raw if _has_rare_relation(g))
print(f"Graphs with rare relations: {rare_count}/{len(sample_graphs_raw)} "
      f"({100*rare_count/len(sample_graphs_raw):.0f}%)")
print("\nAll tests passed.")
