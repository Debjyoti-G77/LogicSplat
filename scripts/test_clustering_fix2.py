"""Debug: trace what happens with hint=4 for scene_09."""
import sys
sys.path.insert(0, ".")
import warnings
warnings.filterwarnings("ignore")

from src.gaussian.loader import load_gaussian_ply, filter_gaussians
from src.gaussian.clustering import auto_sat_threshold, auto_min_cluster_size
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
    print(f"  Full colored pts: {len(X_scaled)}")

    # Subsample
    idx = rng.choice(len(X_scaled), MAX_PTS, replace=False)
    X_sub = X_scaled[idx]

    # Find mcs/method
    mcs, method = auto_min_cluster_size(X_scaled, target_min=3, target_max=5, methods=("eom", "leaf"))
    print(f"  auto_min_cluster_size → mcs={mcs}, method={method}")

    # Verify on subsample
    labels_sub = HDBSCAN(min_cluster_size=mcs, min_samples=3,
                         cluster_selection_method=method, copy=False).fit_predict(X_sub)
    n_sub = len(set(labels_sub)) - (1 if -1 in labels_sub else 0)
    print(f"  On subsample ({len(X_sub)} pts): {n_sub} clusters")

    # Nearest-centroid assignment
    cluster_ids = sorted(set(labels_sub) - {-1})
    print(f"  Cluster IDs from subsample: {cluster_ids}")
    if cluster_ids:
        centroids = np.stack([X_sub[labels_sub == cid].mean(axis=0) for cid in cluster_ids])
        diffs = X_scaled[:, None, :] - centroids[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        assigned = dists.argmin(axis=1)
        counts = {cid: (assigned == i).sum() for i, cid in enumerate(cluster_ids)}
        print(f"  After nearest-centroid assignment on full data: {counts}")
        print(f"  → {len(cluster_ids)} clusters total")

print("\nDone.")
