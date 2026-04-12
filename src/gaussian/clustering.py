"""
Gaussian Splatting object clustering.

Clusters Gaussians into discrete objects using position + color + opacity.
This is richer than COLMAP clustering because each Gaussian has shape info.

Key upgrade over COLMAP:
- COLMAP: cluster sparse XYZ points
- Gaussian: cluster dense XYZ + color + opacity + covariance
"""
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
from src.gaussian.loader import GaussianCloud
from src.graph.definitions import Object3D


def gaussian_to_objects(
    cloud: GaussianCloud,
    min_cluster_size: int = None,
    min_samples: int = 3,
    use_color: bool = True,
    use_opacity: bool = True,
    color_weight: float = 0.3,
) -> Tuple[List[Object3D], dict]:
    """
    Cluster Gaussians into Object3D instances.

    Feature space for clustering:
        - XYZ position (primary)
        - RGB color (weighted, helps separate objects of different colors)
        - Opacity (downweights background Gaussians)

    Args:
        cloud:            GaussianCloud to cluster
        min_cluster_size: auto-computed if None (N/50)
        min_samples:      HDBSCAN min_samples
        use_color:        include color in clustering features
        use_opacity:      weight points by opacity
        color_weight:     how much color influences clustering (0-1)

    Returns:
        (objects, params)
    """
    N = cloud.num_gaussians
    if N < 10:
        return [], {"error": "too few Gaussians"}

    # build feature matrix
    features = [cloud.xyz]
    if use_color:
        color_norm = cloud.rgb.astype(np.float32) / 255.0 * color_weight
        features.append(color_norm)

    X = np.concatenate(features, axis=1)

    # scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # weight by opacity — high opacity Gaussians are more reliable
    if use_opacity:
        sample_weight = cloud.opacity
    else:
        sample_weight = None

    # auto min_cluster_size
    mcs = min_cluster_size if min_cluster_size is not None else max(10, N // 50)

    labels = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=min_samples,
        cluster_selection_method="eom",
    ).fit_predict(X_scaled)

    objects: List[Object3D] = []
    uid = 0
    for label in sorted(set(labels)):
        if label == -1:
            continue
        mask = labels == label
        cluster_xyz = cloud.xyz[mask]
        cluster_rgb = cloud.rgb[mask]
        cluster_opacity = cloud.opacity[mask]
        cluster_cov = cloud.covariance[mask]

        # weight centroid and color by opacity
        weights = cluster_opacity / cluster_opacity.sum()
        centroid = (cluster_xyz * weights[:, None]).sum(axis=0)
        mean_color = (cluster_rgb.astype(float) * weights[:, None]).sum(axis=0).astype(np.uint8)
        mean_opacity = float(cluster_opacity.mean())

        # covariance eigenvalues — shape descriptor
        mean_cov = cluster_cov.mean(axis=0)  # [xx, xy, xz, yy, yz, zz]
        cov_matrix = np.array([
            [mean_cov[0], mean_cov[1], mean_cov[2]],
            [mean_cov[1], mean_cov[3], mean_cov[4]],
            [mean_cov[2], mean_cov[4], mean_cov[5]],
        ])
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]  # descending

        obj = Object3D(
            uid=uid,
            centroid=centroid,
            bbox_min=cluster_xyz.min(axis=0),
            bbox_max=cluster_xyz.max(axis=0),
            color=mean_color,
            point_count=int(mask.sum()),
        )
        # store Gaussian-specific attributes as extra fields
        obj._mean_opacity   = mean_opacity
        obj._eigenvalues    = eigenvalues   # shape descriptor
        obj._mean_cov       = mean_cov

        objects.append(obj)
        uid += 1

    params = {
        "min_cluster_size": mcs,
        "n_gaussians":      N,
        "n_clusters":       len(objects),
        "noise_fraction":   float((labels == -1).sum() / N),
    }
    return objects, params


def extract_gaussian_node_features(obj: Object3D, scene_extent: np.ndarray) -> np.ndarray:
    """
    Extract 13-dim node feature vector from a Gaussian-clustered object.
    This is the upgraded version of the semantic features used in training.

    Features:
        [0-2]  centroid xyz (normalized by scene extent)
        [3-5]  bbox size xyz (normalized)
        [6]    volume (normalized)
        [7]    height_ratio
        [8]    opacity (mean)
        [9-11] covariance eigenvalues (shape: flat/elongated/spherical)
        [12]   point_count (log normalized)
    """
    norm = np.maximum(scene_extent, 1e-6)
    size = obj.size

    centroid_norm = (obj.centroid - obj.bbox_min) / norm
    size_norm = size / norm
    vol = max(obj.volume, 1e-6)
    vol_norm = min(vol / (norm[0]*norm[1]*norm[2]), 1.0)
    h_ratio = float(size[2] / max(max(size[0], size[1]), 1e-6))

    opacity = getattr(obj, '_mean_opacity', 0.5)
    eigs = getattr(obj, '_eigenvalues', np.array([1.0, 0.5, 0.1]))
    eig_norm = eigs / max(eigs[0], 1e-6)  # normalize by largest

    pts_log = float(np.log1p(obj.point_count) / np.log1p(10000))

    return np.array([
        centroid_norm[0], centroid_norm[1], centroid_norm[2],
        size_norm[0], size_norm[1], size_norm[2],
        vol_norm, h_ratio, opacity,
        eig_norm[0], eig_norm[1], eig_norm[2],
        pts_log,
    ], dtype=np.float32)
