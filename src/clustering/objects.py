"""Point cloud clustering to discover 3D objects."""
import numpy as np
from sklearn.cluster import HDBSCAN
from typing import List
from src.graph.definitions import Object3D


# ── preprocessing ─────────────────────────────────────────────────────────────

def auto_sat_threshold(colors: np.ndarray) -> float:
    """Otsu's method on saturation histogram to separate background from objects."""
    sat = np.max(colors.astype(float), axis=1) - np.min(colors.astype(float), axis=1)
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


def remove_outliers(points: np.ndarray, colors: np.ndarray, sigma: float = 2.5):
    mean = points.mean(axis=0)
    std  = points.std(axis=0)
    std[std == 0] = 1e-9
    mask = np.all(np.abs(points - mean) < sigma * std, axis=1)
    return points[mask], colors[mask]


def filter_background(points: np.ndarray, colors: np.ndarray, sat_threshold: float):
    sat = np.max(colors.astype(float), axis=1) - np.min(colors.astype(float), axis=1)
    mask = sat > sat_threshold
    if mask.sum() < 0.1 * len(points):
        return points, colors
    return points[mask], colors[mask]


# ── table plane inference ─────────────────────────────────────────────────────

def infer_table_object(points: np.ndarray, uid: int) -> Object3D:
    """
    Fit a virtual table object from the lowest Z points in the full point cloud.
    COLMAP rarely reconstructs white/textureless surfaces well, so we infer
    the support plane from the bottom 5th percentile of Z values.
    """
    z_floor = np.percentile(points[:, 2], 5)
    z_ceil  = np.percentile(points[:, 2], 15)
    mask = (points[:, 2] >= z_floor) & (points[:, 2] <= z_ceil)
    plane_pts = points[mask]

    if len(plane_pts) < 3:
        plane_pts = points

    return Object3D(
        uid=uid,
        centroid=plane_pts.mean(axis=0),
        bbox_min=np.array([points[:, 0].min(), points[:, 1].min(), z_floor]),
        bbox_max=np.array([points[:, 0].max(), points[:, 1].max(), z_ceil]),
        color=np.array([220, 220, 220], dtype=np.uint8),  # white
        point_count=int(mask.sum()),
        label="table",
    )


# ── main entry point ──────────────────────────────────────────────────────────

def cluster_to_objects(
    points: np.ndarray,
    colors: np.ndarray,
    min_cluster_size: int = None,
    min_samples: int = 3,
    sat_threshold: float = None,
    infer_table: bool = True,
) -> tuple[List[Object3D], dict]:
    """
    Full pipeline: outlier removal → background filter → HDBSCAN → optional table inference.
    min_cluster_size auto-set to points/30 if not provided, ensuring ~6-10 objects.
    """
    pts_full, cols_full = remove_outliers(points, colors)

    sat_thresh = sat_threshold if sat_threshold is not None else auto_sat_threshold(cols_full)
    pts, cols = filter_background(pts_full, cols_full, sat_thresh)

    # auto min_cluster_size: target ~20-30 objects max, so pts/30 as floor
    mcs = min_cluster_size if min_cluster_size is not None else max(5, len(pts) // 30)

    labels = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=min_samples,
        cluster_selection_method="eom",
    ).fit_predict(pts)

    objects: List[Object3D] = []
    uid = 0
    for label in sorted(set(labels)):
        if label == -1:
            continue
        mask = labels == label
        cluster_pts = pts[mask]
        cluster_col = cols[mask]
        objects.append(Object3D(
            uid=uid,
            centroid=cluster_pts.mean(axis=0),
            bbox_min=cluster_pts.min(axis=0),
            bbox_max=cluster_pts.max(axis=0),
            color=cluster_col.mean(axis=0).astype(np.uint8),
            point_count=int(mask.sum()),
        ))
        uid += 1

    # inject virtual table from full (unfiltered) point cloud
    if infer_table:
        table = infer_table_object(pts_full, uid=uid)
        objects.append(table)

    params = {
        "min_cluster_size": mcs,
        "min_samples": min_samples,
        "sat_threshold": round(sat_thresh, 2),
        "points_after_filter": len(pts),
        "clusters_found": len(objects) - (1 if infer_table else 0),
    }
    return objects, params
