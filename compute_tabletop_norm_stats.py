"""
Compute tabletop edge feature normalization statistics from scenes 1-5.
Uses the FULL current pipeline (filter -> prune -> RANSAC -> HDBSCAN).
Saves updated tabletop_feat_mean_v3.npy / tabletop_feat_std_v3.npy.
"""
import sys, os, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import torch

from src.gaussian.loader import (
    load_gaussian_ply, filter_gaussians,
    prune_isolated_gaussians, remove_table_background,
)
from src.gaussian.clustering import gaussian_to_objects, extract_gaussian_node_features
from scripts.build_3rscan_graphs import extract_3rscan_edge_features

DATA_DIR = "D:/logicsplat_data/processed"
SCENES   = [f"scene_{i:02d}" for i in range(1, 14)]  # all 13 scenes (no GT used)
OUT_DIR  = "models"

def cluster_scene(ply_path, n_hint=4):
    cloud = load_gaussian_ply(ply_path)
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)
    cloud = remove_table_background(cloud)

    objects, params = gaussian_to_objects(
        cloud,
        target_min=n_hint,
        target_max=n_hint + 3,
        n_exact=n_hint,
    )
    return objects

def build_edge_features(objects):
    """Build pairwise edge features for all directed object pairs."""
    n = len(objects)
    if n < 2:
        return None

    all_mins = np.stack([o.bbox_min for o in objects])
    all_maxs = np.stack([o.bbox_max for o in objects])
    scene_extent = np.maximum(all_maxs.max(axis=0) - all_mins.min(axis=0), 1e-6)

    edge_feats = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            edge_attr = extract_3rscan_edge_features(objects[i], objects[j], scene_extent)
            edge_feats.append(edge_attr)

    return np.stack(edge_feats, axis=0)  # (E, D)

all_feats = []

print("Computing tabletop edge feature statistics from all 13 scenes with RANSAC pipeline")
print("=" * 70)

for scene_name in SCENES:
    ply_path = os.path.join(DATA_DIR, scene_name, "splat.ply")
    if not os.path.exists(ply_path):
        print(f"  {scene_name}: MISSING — skipped")
        continue

    objects = cluster_scene(ply_path, n_hint=4)
    n = len(objects)
    feats = build_edge_features(objects)
    if feats is None:
        print(f"  {scene_name}: only {n} objects, skipped")
        continue

    all_feats.append(feats)
    print(f"  {scene_name}: {n} objects, {len(feats)} edges, feat_dim={feats.shape[1]}")

if not all_feats:
    print("ERROR: no features collected")
    sys.exit(1)

all_feats_np = np.concatenate(all_feats, axis=0)  # (total_E, D)
print(f"\nTotal edges: {len(all_feats_np)}")

tt_mean = all_feats_np.mean(axis=0).astype(np.float32)
tt_std  = all_feats_np.std(axis=0).clip(min=1e-6).astype(np.float32)

out_mean = os.path.join(OUT_DIR, "tabletop_feat_mean_v4.npy")
out_std  = os.path.join(OUT_DIR, "tabletop_feat_std_v4.npy")
np.save(out_mean, tt_mean)
np.save(out_std,  tt_std)

print(f"\nSaved: {out_mean}")
print(f"Saved: {out_std}")

# Compare with old stats (v2)
old_mean_path = os.path.join(OUT_DIR, "tabletop_feat_mean_v2.npy")
if os.path.exists(old_mean_path):
    old_mean = np.load(old_mean_path)
    old_std  = np.load(os.path.join(OUT_DIR, "tabletop_feat_std_v2.npy"))
    print(f"\nDim-by-dim comparison (v2 vs v3 mean):")
    print(f"  {'dim':>4}  {'v2_mean':>10}  {'v3_mean':>10}  {'v2_std':>10}  {'v3_std':>10}")
    for i, (om, nm, os_, ns) in enumerate(zip(old_mean, tt_mean, old_std, tt_std)):
        delta = abs(nm - om)
        marker = " <-- changed" if delta > 0.05 else ""
        print(f"  {i:>4}  {om:>10.4f}  {nm:>10.4f}  {os_:>10.4f}  {ns:>10.4f}{marker}")

print("\nDone.")
