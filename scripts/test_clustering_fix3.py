"""
Find the exact mcs/method that gives 4 clusters on the 6k subsample
for scenes 09 and 10, using seed=42.
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

MAX_PTS = 6000
rng = np.random.default_rng(42)

for scene in ["scene_09", "scene_10"]:
    print(f"\n=== {scene} ===")
    cloud = load_gaussian_ply(f"D:/logicsplat_data/processed/{scene}/splat.ply")
    filtered = filter_gaussians(cloud, opacity_threshold=0.1)

    sat = filtered.rgb.astype(float).max(axis=1) - filtered.rgb.astype(float).min(axis=1)
    thresh = auto_sat_threshold(filtered.rgb)
    mask = sat > thresh
    if mask.sum() < 0.05 * filtered.num_gaussians:
        mask = sat > thresh * 0.5
    if mask.sum() < 0.05 * filtered.num_gaussians:
        mask = np.ones(filtered.num_gaussians, dtype=bool)

    xyz = filtered.xyz[mask]
    rgb = filtered.rgb[mask]
    color_norm = rgb.astype(np.float32) / 255.0 * 0.3
    X = np.concatenate([xyz, color_norm], axis=1)
    X_scaled = StandardScaler().fit_transform(X)

    # Use the SAME subsample as auto_min_cluster_size will use
    idx = np.random.default_rng(42).choice(len(X_scaled), MAX_PTS, replace=False)
    X_sub = X_scaled[idx]
    print(f"  Subsample size: {len(X_sub)}")

    print("  Full sweep on subsample:")
    for method in ["eom", "leaf"]:
        for divisor in [3, 5, 8, 10, 15, 20, 30, 50, 80, 100, 150, 200, 300, 500]:
            mcs = max(10, len(X_sub) // divisor)
            labels = HDBSCAN(min_cluster_size=mcs, min_samples=3,
                             cluster_selection_method=method, copy=False).fit_predict(X_sub)
            n = len(set(labels)) - (1 if -1 in labels else 0)
            marker = " ← TARGET" if n == 4 else ""
            print(f"    {method} mcs={mcs:4d}: {n} clusters{marker}")

print("\nDone.")
