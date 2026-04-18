"""
Gaussian Splatting object clustering.

Auto-tunes parameters from the data — no hardcoding.
Uses Otsu's method for saturation threshold and
elbow detection for min_cluster_size.
"""
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
from src.gaussian.loader import GaussianCloud
from src.graph.definitions import Object3D


# ── auto-parameter estimation ─────────────────────────────────────────────────

def auto_sat_threshold(rgb: np.ndarray) -> float:
    """Otsu's method on saturation histogram."""
    sat = rgb.astype(float).max(axis=1) - rgb.astype(float).min(axis=1)
    counts, bin_edges = np.histogram(sat, bins=256, range=(0, 255))
    total = counts.sum()
    best_thresh, best_var = 0.0, 0.0
    w0, sum0 = 0.0, 0.0
    total_mean = np.sum(np.arange(256) * counts) / total
    for t in range(256):
        w0 += counts[t]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += t * counts[t]
        mu0 = sum0 / w0
        mu1 = (total_mean * total - sum0) / w1
        var_between = w0 * w1 * (mu0 - mu1) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = float(bin_edges[t + 1])
    return best_thresh


def auto_min_cluster_size(pts: np.ndarray, target_min: int = 5, target_max: int = 15) -> int:
    """
    Find min_cluster_size that gives between target_min and target_max clusters.
    Tries values from N/5 down to N/100.
    """
    N = len(pts)
    best_mcs = max(10, N // 20)
    best_n = 0

    for divisor in [5, 8, 10, 15, 20, 30, 50, 80, 100]:
        mcs = max(10, N // divisor)
        labels = HDBSCAN(
            min_cluster_size=mcs, min_samples=3,
            cluster_selection_method="eom"
        ).fit_predict(pts)
        n = len(set(labels)) - (1 if -1 in labels else 0)
        if target_min <= n <= target_max:
            return mcs
        if abs(n - (target_min + target_max) // 2) < abs(best_n - (target_min + target_max) // 2):
            best_n = n
            best_mcs = mcs

    return best_mcs


# ── main clustering ───────────────────────────────────────────────────────────

def gaussian_to_objects(
    cloud: GaussianCloud,
    min_cluster_size: int = None,
    min_samples: int = 3,
    sat_threshold: float = None,
    color_weight: float = 0.3,
) -> Tuple[List[Object3D], dict]:
    """
    Cluster Gaussians into Object3D instances with auto-tuned parameters.

    Steps:
    1. Auto-detect saturation threshold (Otsu) to remove white/grey background
    2. Auto-detect min_cluster_size to get 5-15 objects
    3. HDBSCAN on position + color features
    4. Extract per-cluster Object3D with Gaussian attributes

    Returns: (objects, params)
    """
    N = cloud.num_gaussians
    if N < 10:
        return [], {"error": "too few Gaussians"}

    # ── step 1: background filtering ─────────────────────────────────────────
    sat_thresh = sat_threshold if sat_threshold is not None else auto_sat_threshold(cloud.rgb)
    sat = cloud.rgb.astype(float).max(axis=1) - cloud.rgb.astype(float).min(axis=1)
    sat_mask = sat > sat_thresh

    # fallback if too aggressive
    if sat_mask.sum() < 0.05 * N:
        sat_thresh = sat_thresh * 0.5
        sat_mask = sat > sat_thresh
    if sat_mask.sum() < 0.05 * N:
        sat_mask = np.ones(N, dtype=bool)

    xyz = cloud.xyz[sat_mask]
    rgb = cloud.rgb[sat_mask]
    opacity = cloud.opacity[sat_mask]
    cov = cloud.covariance[sat_mask]

    # ── step 2: build feature matrix ─────────────────────────────────────────
    color_norm = rgb.astype(np.float32) / 255.0 * color_weight
    X = np.concatenate([xyz, color_norm], axis=1)
    X_scaled = StandardScaler().fit_transform(X)

    # ── step 3: auto min_cluster_size ────────────────────────────────────────
    mcs = min_cluster_size if min_cluster_size is not None else auto_min_cluster_size(X_scaled)

    labels = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=min_samples,
        cluster_selection_method="eom",
    ).fit_predict(X_scaled)

    # ── step 4: build Object3D per cluster ───────────────────────────────────
    objects: List[Object3D] = []
    uid = 0
    for label in sorted(set(labels)):
        if label == -1:
            continue
        mask = labels == label
        cluster_xyz = xyz[mask]
        cluster_rgb = rgb[mask]
        cluster_opacity = opacity[mask]
        cluster_cov = cov[mask]

        weights = cluster_opacity / max(cluster_opacity.sum(), 1e-9)
        centroid = (cluster_xyz * weights[:, None]).sum(axis=0)
        mean_color = (cluster_rgb.astype(float) * weights[:, None]).sum(axis=0).astype(np.uint8)
        mean_opacity = float(cluster_opacity.mean())

        mean_cov = cluster_cov.mean(axis=0)
        cov_matrix = np.array([
            [mean_cov[0], mean_cov[1], mean_cov[2]],
            [mean_cov[1], mean_cov[3], mean_cov[4]],
            [mean_cov[2], mean_cov[4], mean_cov[5]],
        ])
        eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(cov_matrix)))[::-1]

        obj = Object3D(
            uid=uid,
            centroid=centroid,
            bbox_min=cluster_xyz.min(axis=0),
            bbox_max=cluster_xyz.max(axis=0),
            color=mean_color,
            point_count=int(mask.sum()),
        )
        obj._mean_opacity = mean_opacity
        obj._eigenvalues  = eigenvalues
        obj._mean_cov     = mean_cov
        objects.append(obj)
        uid += 1

    params = {
        "sat_threshold":    round(sat_thresh, 2),
        "min_cluster_size": mcs,
        "n_gaussians_raw":  N,
        "n_after_filter":   int(sat_mask.sum()),
        "n_clusters":       len(objects),
        "noise_fraction":   round(float((labels == -1).sum() / len(xyz)), 3),
    }
    return objects, params


def extract_gaussian_node_features(obj: Object3D, scene_extent: np.ndarray) -> np.ndarray:
    """13-dim geometric node features from a Gaussian-clustered object."""
    norm = np.maximum(scene_extent, 1e-6)
    size = obj.size
    centroid_norm = (obj.centroid - obj.bbox_min) / norm
    size_norm = size / norm
    vol = max(obj.volume, 1e-6)
    vol_norm = min(vol / (norm[0] * norm[1] * norm[2]), 1.0)
    h_ratio = float(size[2] / max(max(size[0], size[1]), 1e-6))
    opacity = getattr(obj, '_mean_opacity', 0.5)
    eigs = getattr(obj, '_eigenvalues', np.array([1.0, 0.5, 0.1]))
    eig_norm = eigs / max(eigs[0], 1e-6)
    pts_log = float(np.log1p(obj.point_count) / np.log1p(10000))
    return np.array([
        centroid_norm[0], centroid_norm[1], centroid_norm[2],
        size_norm[0], size_norm[1], size_norm[2],
        vol_norm, h_ratio, opacity,
        eig_norm[0], eig_norm[1], eig_norm[2],
        pts_log,
    ], dtype=np.float32)
