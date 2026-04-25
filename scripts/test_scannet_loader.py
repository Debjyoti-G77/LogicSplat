"""
Quick smoke-test for the ScanNet loader.
Loads the first scene from D:/scannet/scans/ and prints graph stats.

Usage:
    python scripts/test_scannet_loader.py
"""
import sys
sys.path.insert(0, ".")

from src.dataset.loader_scannet import SceneGraphDatasetScanNet

SCANNET_DIR = "D:/scannet/scans"

print(f"Loading from: {SCANNET_DIR}")
ds = SceneGraphDatasetScanNet(scannet_dir=SCANNET_DIR, max_scenes=5, verbose=True)

if len(ds) > 0:
    g = ds[0]
    print(f"\nSample graph:")
    print(f"  scene_id  : {g['scene_id']}")
    print(f"  x         : {g['x'].shape}  (nodes × node_feat_dim)")
    print(f"  edge_attr : {g['edge_attr'].shape}  (edges × edge_feat_dim)")
    print(f"  edge_label: {g['edge_label'].shape}")
    print(f"  objects   : {g['obj_labels']}")

    # check feature ranges — all should be roughly [0, 1]
    print(f"\nNode feature ranges:")
    for i in range(g['x'].shape[1]):
        col = g['x'][:, i]
        print(f"  feat[{i}]: min={col.min():.3f}  max={col.max():.3f}")

    print(f"\nEdge feature ranges:")
    for i in range(g['edge_attr'].shape[1]):
        col = g['edge_attr'][:, i]
        print(f"  feat[{i}]: min={col.min():.3f}  max={col.max():.3f}")

    from collections import Counter
    from src.relations.schema import RELATION_NAMES
    counts = Counter(g['edge_label'].tolist())
    print(f"\nRelation distribution in sample scene:")
    for idx, cnt in sorted(counts.items()):
        print(f"  {RELATION_NAMES[idx]:20s}  {cnt}")
else:
    print("No scenes loaded. Check that D:/scannet/scans/ contains scene folders.")
