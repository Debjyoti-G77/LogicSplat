"""
Diagnostic script for clustering failures in scenes 09 and 10.
Uses subsampling to keep HDBSCAN fast on large point clouds.
"""
import sys
sys.path.insert(0, ".")
import warnings
warnings.filterwarnings("ignore")

from src.gaussian.loader import load_gaussian_ply, filter_gaussians
from src.gaussian.clustering import auto_sat_threshold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
import numpy as np

MAX_PTS = 8000  # subsample ceiling for fast sweeps


def build_features(cloud, sat_mult=1.0, color_weight=0.3):
    sat = cloud.rgb.astype(float).max(axis=1) - cloud.rgb.astype(float).min(axis=1)
    thresh = auto_sat_threshold(cloud.rgb) * sat_mult
    mask = sat > thresh
    if mask.sum() < 0.05 * cloud.num_gaussians:
        mask = sat > thresh * 0.5
    if mask.sum() < 0.05 * cloud.num_gaussians:
        mask = np.ones(cloud.num_gaussians, dtype=bool)
    xyz = cloud.xyz[mask]
    rgb = cloud.rgb[mask]
    color_norm = rgb.astype(np.float32) / 255.0 * color_weight
    X = np.concatenate([xyz, color_norm], axis=1)
    # subsample
    if len(X) > MAX_PTS:
        idx = np.random.default_rng(42).choice(len(X), MAX_PTS, replace=False)
        X = X[idx]
    return StandardScaler().fit_transform(X), mask.sum()


def sweep(X_scaled, label, methods=("eom",), mcs_list=(10, 20, 30, 50, 80, 100, 150, 200), ms=3):
    print(f"  [{label}]")
    for method in methods:
        for mcs in mcs_list:
            labels = HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                             cluster_selection_method=method, copy=False).fit_predict(X_scaled)
            n = len(set(labels)) - (1 if -1 in labels else 0)
            noise_pct = 100 * (labels == -1).sum() / len(labels)
            print(f"    {method} mcs={mcs:3d} ms={ms}: {n} clusters  noise={noise_pct:.0f}%")


for scene in ["scene_09", "scene_10"]:
    print(f"\n{'='*55}")
    print(f"  {scene}")
    print(f"{'='*55}")

    cloud = load_gaussian_ply(f"D:/logicsplat_data/processed/{scene}/splat.ply")
    filtered = filter_gaussians(cloud, opacity_threshold=0.1)
    filtered05 = filter_gaussians(cloud, opacity_threshold=0.05)
    print(f"  raw={cloud.num_gaussians}  op>0.1={filtered.num_gaussians}  op>0.05={filtered05.num_gaussians}")

    thresh = auto_sat_threshold(filtered.rgb)
    sat = filtered.rgb.astype(float).max(axis=1) - filtered.rgb.astype(float).min(axis=1)
    mask = sat > thresh
    print(f"  Otsu sat thresh={thresh:.1f}  colored={mask.sum()} ({100*mask.sum()/filtered.num_gaussians:.1f}%)")
    mask2 = sat > thresh * 0.5
    print(f"  0.5x sat thresh={thresh*0.5:.1f}  colored={mask2.sum()} ({100*mask2.sum()/filtered.num_gaussians:.1f}%)")

    # Baseline: op>0.1, color_weight=0.3, eom
    X, n_pts = build_features(filtered, sat_mult=1.0, color_weight=0.3)
    print(f"\n  -- op>0.1, cw=0.3, sat=1.0x  (pts after filter+subsample: {n_pts} -> {len(X)}) --")
    sweep(X, "eom", methods=["eom"], mcs_list=[10, 20, 30, 50, 80, 100, 150, 200])
    sweep(X, "leaf", methods=["leaf"], mcs_list=[10, 20, 30, 50, 80, 100, 150, 200])
    sweep(X, "eom ms=1", methods=["eom"], mcs_list=[10, 20, 30, 50, 80, 100], ms=1)

    # Higher color weight
    X2, _ = build_features(filtered, sat_mult=1.0, color_weight=0.5)
    print(f"\n  -- op>0.1, cw=0.5, sat=1.0x --")
    sweep(X2, "eom", methods=["eom"], mcs_list=[10, 20, 30, 50, 80, 100, 150, 200])
    sweep(X2, "leaf", methods=["leaf"], mcs_list=[10, 20, 30, 50, 80, 100, 150, 200])

    # Lower opacity threshold
    X3, n3 = build_features(filtered05, sat_mult=1.0, color_weight=0.3)
    print(f"\n  -- op>0.05, cw=0.3, sat=1.0x  (pts after filter+subsample: {n3} -> {len(X3)}) --")
    sweep(X3, "eom", methods=["eom"], mcs_list=[10, 20, 30, 50, 80, 100, 150, 200])

    # Lower sat threshold
    X4, n4 = build_features(filtered, sat_mult=0.5, color_weight=0.3)
    print(f"\n  -- op>0.1, cw=0.3, sat=0.5x  (pts after filter+subsample: {n4} -> {len(X4)}) --")
    sweep(X4, "eom", methods=["eom"], mcs_list=[10, 20, 30, 50, 80, 100, 150, 200])

    # Best combo: op>0.05, cw=0.5, leaf
    X5, n5 = build_features(filtered05, sat_mult=1.0, color_weight=0.5)
    print(f"\n  -- op>0.05, cw=0.5, sat=1.0x  (pts: {n5} -> {len(X5)}) --")
    sweep(X5, "eom+leaf", methods=["eom", "leaf"], mcs_list=[10, 20, 30, 50, 80, 100, 150, 200])

print("\nDone.")
