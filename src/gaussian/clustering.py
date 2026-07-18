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


def auto_min_cluster_size(
    pts: np.ndarray,
    target_min: int = 4,
    target_max: int = 12,
    methods: tuple = ("eom",),
) -> tuple:
    """
    Find (min_cluster_size, cluster_selection_method) that gives between
    target_min and target_max clusters.

    Sweeps mcs values across all requested HDBSCAN selection methods.
    The input pts are subsampled to at most _MAX_TUNE_PTS before sweeping so
    that the auto-tuner stays fast even on large point clouds.  The chosen mcs
    is used as-is (not scaled) for the final HDBSCAN call — HDBSCAN's cluster
    structure is density-based and the same mcs works on the subsample and the
    full data when the subsample preserves the spatial density distribution.

    Returns the first (mcs, method) combo that lands in [target_min, target_max];
    if none does, returns the combo whose cluster count is closest to the midpoint.

    Args:
        pts:        scaled feature matrix (already StandardScaler-transformed)
        target_min: minimum acceptable cluster count
        target_max: maximum acceptable cluster count
        methods:    HDBSCAN cluster_selection_method values to try

    Returns:
        (mcs, method) tuple
    """
    _MAX_TUNE_PTS = 6000  # subsample ceiling for fast sweeps

    N_full = len(pts)
    if N_full > _MAX_TUNE_PTS:
        idx = np.random.default_rng(42).choice(N_full, _MAX_TUNE_PTS, replace=False)
        pts_tune = pts[idx]
    else:
        pts_tune = pts
    N_tune = len(pts_tune)

    mid = (target_min + target_max) // 2
    best_mcs = max(10, N_tune // 20)
    best_method = methods[0]
    best_n = 0
    seen = set()
    candidates = []  # (abs_dist_to_mid, n, mcs, method)

    # Wider divisor range: include very large mcs (N//3) for small target counts
    divisors = [3, 5, 8, 10, 15, 20, 30, 50, 80, 100, 150, 200, 300, 500]

    for method in methods:
        for divisor in divisors:
            mcs = max(10, N_tune // divisor)
            key = (mcs, method)
            if key in seen:
                continue
            seen.add(key)
            labels = HDBSCAN(
                min_cluster_size=mcs, min_samples=3,
                cluster_selection_method=method, copy=False,
            ).fit_predict(pts_tune)
            n = len(set(labels)) - (1 if -1 in labels else 0)
            if target_min <= n <= target_max:
                candidates.append((abs(n - mid), n, mcs, method))
            if abs(n - mid) < abs(best_n - mid):
                best_n = n
                best_mcs = mcs
                best_method = method

    # Pick the in-range candidate closest to the midpoint
    if candidates:
        candidates.sort()
        _, _, best_mcs, best_method = candidates[0]

    return best_mcs, best_method


# ── main clustering ───────────────────────────────────────────────────────────

def gaussian_to_objects(
    cloud: GaussianCloud,
    min_cluster_size: int = None,
    min_samples: int = 3,
    sat_threshold: float = None,
    color_weight: float = 0.3,
    cluster_selection_method: str = None,
    target_min: int = 4,
    target_max: int = 12,
    n_exact: int = None,
) -> Tuple[List[Object3D], dict]:
    """
    Cluster Gaussians into Object3D instances with auto-tuned parameters.

    Steps:
    1. Auto-detect saturation threshold (Otsu) to remove white/grey background
    2. Auto-detect min_cluster_size to get target_min–target_max objects
    3. HDBSCAN on position + color features
    4. Extract per-cluster Object3D with Gaussian attributes

    Args:
        cloud:                    filtered GaussianCloud
        min_cluster_size:         override auto-tuning (None = auto)
        min_samples:              HDBSCAN min_samples
        sat_threshold:            override Otsu saturation threshold (None = auto)
        color_weight:             weight for RGB features relative to XYZ
        cluster_selection_method: override HDBSCAN method (None = auto-selected)
        target_min:               minimum cluster count for auto-tuner
        target_max:               maximum cluster count for auto-tuner
        n_exact:                  if provided, always keep exactly this many most-compact
                                  clusters (set target_max = n_exact*4 to give HDBSCAN
                                  room to isolate background into separate clusters)

    Returns: (objects, params)
    """
    N = cloud.num_gaussians
    if N < 10:
        return [], {"error": "too few Gaussians"}

    # When n_exact is given, allow HDBSCAN to over-split (find more clusters than
    # needed) so background structures end up in separate clusters; the n_exact
    # most compact clusters are kept at the end.
    if n_exact is not None:
        target_min = max(n_exact, target_min)
        target_max = max(n_exact * 4, target_max)

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
    _MAX_CLUSTER_PTS = 6000  # subsample ceiling — same as auto-tuner

    if min_cluster_size is not None:
        mcs = min_cluster_size
        method = cluster_selection_method or "eom"
    else:
        mcs, method = auto_min_cluster_size(
            X_scaled,
            target_min=target_min,
            target_max=target_max,
            methods=("eom", "leaf") if cluster_selection_method is None else (cluster_selection_method,),
        )
        if cluster_selection_method is not None:
            method = cluster_selection_method

    # Run HDBSCAN on a subsample (same size as auto-tuner used) so that the
    # mcs found by auto_min_cluster_size produces the expected cluster count.
    # Non-sampled points are then assigned to their nearest cluster centroid
    # in the scaled feature space.
    N_pts = len(X_scaled)
    if N_pts > _MAX_CLUSTER_PTS:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(N_pts, _MAX_CLUSTER_PTS, replace=False)
        X_sample = X_scaled[sample_idx]
        labels_sample = HDBSCAN(
            min_cluster_size=mcs,
            min_samples=min_samples,
            cluster_selection_method=method,
            copy=False,
        ).fit_predict(X_sample)

        # Compute centroid of each cluster in feature space
        cluster_ids = sorted(set(labels_sample) - {-1})
        if len(cluster_ids) == 0:
            # All noise — fall back to full data with eom
            labels = HDBSCAN(
                min_cluster_size=max(10, N_pts // 20),
                min_samples=min_samples,
                cluster_selection_method="eom",
                copy=False,
            ).fit_predict(X_scaled)
        else:
            cluster_centroids = np.stack([
                X_sample[labels_sample == cid].mean(axis=0)
                for cid in cluster_ids
            ])  # shape (K, D)

            # Estimate max allowed distance from the MOST COMPACT cluster's spread.
            # Background clusters are large → using all clusters inflates D_max.
            # The most compact cluster is almost certainly a real object, so its
            # 90th-percentile intra-cluster distance gives a tight reference radius.
            per_cluster_p90 = []
            for cid in cluster_ids:
                c_pts = X_sample[labels_sample == cid]
                c_cen = c_pts.mean(axis=0)
                c_dists = np.linalg.norm(c_pts - c_cen, axis=1)
                per_cluster_p90.append(float(np.percentile(c_dists, 90)))
            # Use the smallest cluster radius × 4 as the assignment threshold.
            # Background Gaussians are far from ALL object-cluster centroids.
            D_max = (min(per_cluster_p90) * 4.0) if per_cluster_p90 else np.inf

            # Assign each point to nearest centroid, but only if within D_max.
            # Points outside D_max are background Gaussians → noise (-1).
            diffs = X_scaled[:, None, :] - cluster_centroids[None, :, :]  # (N, K, D)
            dists_to_clusters = np.linalg.norm(diffs, axis=2)  # (N, K)
            min_dists  = dists_to_clusters.min(axis=1)
            best_idx   = dists_to_clusters.argmin(axis=1)
            labels = np.where(
                min_dists <= D_max,
                np.array([cluster_ids[i] for i in best_idx]),
                -1,  # noise: too far from any object cluster
            )
    else:
        labels = HDBSCAN(
            min_cluster_size=mcs,
            min_samples=min_samples,
            cluster_selection_method=method,
            copy=False,
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

        # Robust bbox: exclude outlier Gaussians that are far from the cluster
        # centroid (e.g. background/wall Gaussians mis-assigned by K-means).
        # Use distance-based trimming: keep points within 2.5σ of centroid.
        dists = np.linalg.norm(cluster_xyz - centroid, axis=1)
        dist_thresh = dists.mean() + 2.5 * dists.std()
        tight_xyz = cluster_xyz[dists <= dist_thresh]
        if len(tight_xyz) < max(5, len(cluster_xyz) // 5):
            tight_xyz = cluster_xyz  # fallback: too few points after trimming

        obj = Object3D(
            uid=uid,
            centroid=centroid,
            bbox_min=tight_xyz.min(axis=0),
            bbox_max=tight_xyz.max(axis=0),
            color=mean_color,
            point_count=int(mask.sum()),
        )
        obj._mean_opacity = mean_opacity
        obj._eigenvalues  = eigenvalues
        obj._mean_cov     = mean_cov
        objects.append(obj)
        uid += 1

    # ── step 5: remove background clusters ───────────────────────────────────
    # When n_exact is set, keep the n_exact clusters with the MOST GAUSSIANS.
    # Objects have more Gaussians than residual noise/surface fragments.
    # Compact-first ordering was discarded because it kept tiny sub-fragments.
    # Without n_exact, only trim if we over-produced beyond target_max.
    n_keep = n_exact if (n_exact is not None and len(objects) >= n_exact) else None
    if n_keep is None and len(objects) > target_max:
        n_keep = target_max
    if n_keep is not None and len(objects) > n_keep:
        counts  = np.array([o.point_count for o in objects])
        keep_idx = np.argsort(counts)[::-1][:n_keep]  # most Gaussians first
        objects  = [objects[i] for i in sorted(keep_idx)]
        for new_uid, o in enumerate(objects):
            o.uid = new_uid

    params = {
        "sat_threshold":    round(sat_thresh, 2),
        "min_cluster_size": mcs,
        "cluster_method":   method,
        "n_gaussians_raw":  N,
        "n_after_filter":   int(sat_mask.sum()),
        "n_clusters":       len(objects),
        "noise_fraction":   round(float((labels == -1).sum() / len(labels)), 3),
    }
    return objects, params


def extract_gaussian_node_features(
    obj: Object3D,
    scene_extent: np.ndarray,
    scene_min: np.ndarray = None,
    scene_mean_diag: float = None,
    scene_median_volume: float = None,
    z_rank: float = 0.5,
) -> np.ndarray:
    """
    Extract 10-dim scale-invariant node features from a Gaussian-clustered object.

    All features are normalized by object-level statistics (own diagonal, scene
    mean diagonal) rather than scene extent. This makes features consistent
    across room-scale and tabletop-scale scenes — the key fix for cross-domain
    failure on vertical relations (on_top_of, under, higher_than, lower_than).

    Features:
        [0]  (c_x - scene_min_x) / scene_mean_diag, clipped [0,1]
        [1]  (c_y - scene_min_y) / scene_mean_diag, clipped [0,1]
        [2]  (c_z - scene_min_z) / scene_mean_diag, clipped [0,1]
             → "how many object-widths above the floor" — consistent across scales
        [3]  size_x / self_diagonal  (shape aspect ratio, scale-invariant)
        [4]  size_y / self_diagonal
        [5]  size_z / self_diagonal
        [6]  log(volume / scene_median_volume + 1) / 5, clipped [0,1]
             → relative size within the scene
        [7]  height_ratio = size_z / max(size_x, size_y)  (elongation)
        [8]  point_density  (pts / volume, normalized)
        [9]  z_rank: ordinal rank of centroid_z among all scene objects [0,1]
             → 0.0 = lowest object, 1.0 = highest object in scene
             → completely scale-invariant: captures relative vertical ordering
    """
    if scene_min is None:
        scene_min = obj.bbox_min

    size = np.maximum(obj.size, 1e-6)
    self_diag = max(float(np.linalg.norm(size)), 1e-6)

    # Use scene_mean_diag if provided, else fall back to scene_extent mean
    if scene_mean_diag is None or scene_mean_diag < 1e-6:
        mean_diag = max(float(np.mean(np.maximum(scene_extent, 1e-6))), 1e-6)
    else:
        mean_diag = scene_mean_diag

    # [0-2] Centroid position in object-diagonal units — clipped to [0,1]
    max_diag_units = 8.0  # clip at 8 object-diagonals from scene corner
    cx = float(np.clip((obj.centroid[0] - scene_min[0]) / mean_diag, 0.0, max_diag_units) / max_diag_units)
    cy = float(np.clip((obj.centroid[1] - scene_min[1]) / mean_diag, 0.0, max_diag_units) / max_diag_units)
    cz = float(np.clip((obj.centroid[2] - scene_min[2]) / mean_diag, 0.0, max_diag_units) / max_diag_units)

    # [3-5] Shape aspect ratios (size / self_diagonal) — scale-invariant
    sx = float(size[0] / self_diag)
    sy = float(size[1] / self_diag)
    sz = float(size[2] / self_diag)

    # [6] Relative volume in scene (log-ratio vs median)
    volume = float(size[0] * size[1] * size[2])
    if scene_median_volume is None or scene_median_volume < 1e-9:
        scene_median_volume = volume  # fallback: neutral (log = 0)
    vol_log = float(np.clip(np.log(volume / max(scene_median_volume, 1e-9) + 1) / 5.0, 0.0, 1.0))

    # [7] Shape elongation
    height_ratio = float(size[2] / max(size[0], size[1]))

    # [8] Point density
    density = min(obj.point_count / max(volume, 1e-6), 1000.0) / 1000.0

    # [9] Ordinal vertical rank [0,1] — provided by caller
    z_rank_val = float(np.clip(z_rank, 0.0, 1.0))

    return np.array([
        cx, cy, cz,
        sx, sy, sz,
        vol_log, height_ratio, density, z_rank_val,
    ], dtype=np.float32)


def extract_gaussian_edge_features(obj_a: Object3D, obj_b: Object3D, scene_extent: np.ndarray) -> np.ndarray:
    """
    Extract 19-dim geometric edge features between two Gaussian-clustered objects.
    Identical feature layout and normalization to extract_3rscan_edge_features in
    scripts/build_3rscan_graphs.py — any change to one must be mirrored in the other.

    Features [0-4, 8]: avg_diag normalized (scale-invariant across room/tabletop).
    Features [17-18]: 3D containment features for inside detection.

    See build_3rscan_graphs.extract_3rscan_edge_features for full feature documentation.
    """
    c_a = obj_a.centroid
    c_b = obj_b.centroid
    min_a, max_a = obj_a.bbox_min, obj_a.bbox_max
    min_b, max_b = obj_b.bbox_min, obj_b.bbox_max
    size_a = np.maximum(max_a - min_a, 1e-6)
    size_b = np.maximum(max_b - min_b, 1e-6)

    diag_a = float(np.linalg.norm(size_a))
    diag_b = float(np.linalg.norm(size_b))
    avg_diag = max((diag_a + diag_b) / 2.0, 1e-6)

    # [0-4] avg_diag normalized directional features
    delta_x = float(np.clip((c_a[0] - c_b[0]) / avg_diag, -3.0, 3.0) / 3.0)
    # Negate delta_y: 3DSSG convention -Y = front, positive = in_front_of
    delta_y = float(np.clip(-(c_a[1] - c_b[1]) / avg_diag, -3.0, 3.0) / 3.0)
    delta_z = float(np.clip((c_a[2] - c_b[2]) / avg_diag, -3.0, 3.0) / 3.0)
    xy_dist = float(np.clip(np.linalg.norm(c_a[:2] - c_b[:2]) / avg_diag, 0.0, 3.0) / 3.0)
    dist_3d = float(np.clip(np.linalg.norm(c_a - c_b) / avg_diag, 0.0, 3.0) / 3.0)

    # [5-7, 9] scale-invariant
    ix = max(0.0, min(max_a[0], max_b[0]) - max(min_a[0], min_b[0]))
    iy = max(0.0, min(max_a[1], max_b[1]) - max(min_a[1], min_b[1]))
    area_a = max((max_a[0] - min_a[0]) * (max_a[1] - min_a[1]), 1e-9)
    bbox_overlap = (ix * iy) / area_a

    vol_a = float(np.prod(size_a))
    vol_b = float(np.prod(size_b))
    vol_ratio = min(vol_a, vol_b) / max(vol_a, vol_b)

    h_ratio = float(np.clip(np.log1p(size_a[2] / max(size_b[2], 1e-6)), -3.0, 3.0) / 3.0)

    size_ratio_xy = float(np.clip(
        np.log1p(np.linalg.norm(size_a[:2]) / max(np.linalg.norm(size_b[:2]), 1e-6)),
        -3.0, 3.0,
    ) / 3.0)

    # [8] vertical gap, avg_height normalized
    mean_height = max((size_a[2] + size_b[2]) / 2.0, 1e-6)
    vert_gap = float(np.clip((min_a[2] - max_b[2]) / mean_height, -3.0, 3.0) / 3.0)

    # [10-13] contact features
    vert_gap_obj_norm = float(np.clip((min_a[2] - max_b[2]) / mean_height, -3.0, 3.0) / 3.0)
    vert_gap_a_relative = float(np.clip((min_a[2] - max_b[2]) / size_a[2], -3.0, 3.0) / 3.0)

    a_above_b = float(c_a[2] > c_b[2])
    raw_gap = float(min_a[2] - max_b[2])
    contact_score = float(
        np.exp(-abs(raw_gap) / max(mean_height * 0.3, 1e-6)) * bbox_overlap * a_above_b
    )

    area_b = max((max_b[0] - min_b[0]) * (max_b[1] - min_b[1]), 1e-9)
    support_overlap_b = float((ix * iy) / area_b)

    # [14-16] rule-margin features
    centroid_z_diff = float(c_a[2] - c_b[2])
    centroid_xy_dist = float(np.linalg.norm(c_a[:2] - c_b[:2]))

    z_min_thresh = max(avg_diag * 0.05, 1e-6)
    ontop_z_margin = float(np.clip(
        (centroid_z_diff - z_min_thresh) / z_min_thresh, -3.0, 3.0
    ) / 3.0)

    b_xy_size = float(np.linalg.norm(size_b[:2]))
    xy_limit = max(b_xy_size * 0.8, 1e-6)
    ontop_xy_margin = float(np.clip(
        (xy_limit - centroid_xy_dist) / xy_limit, -3.0, 3.0
    ) / 3.0)

    z_dom_margin = float(np.clip(
        (centroid_z_diff - centroid_xy_dist * 0.5) / max(abs(centroid_z_diff) + 1e-6, 1e-6),
        -3.0, 3.0
    ) / 3.0)

    # [17-18] 3D containment features for inside detection
    iz = max(0.0, min(max_a[2], max_b[2]) - max(min_a[2], min_b[2]))
    vol_overlap = ix * iy * iz
    containment_3d = float(min(vol_overlap / max(vol_a, 1e-9), 1.0))

    centroid_inside_b = float(
        min_b[0] <= c_a[0] <= max_b[0] and
        min_b[1] <= c_a[1] <= max_b[1] and
        min_b[2] <= c_a[2] <= max_b[2]
    )

    # [19] vz_ratio: vertical/horizontal displacement ratio.
    # Stacked objects (on_top_of): |delta_z| >> xy_dist → ratio >> 1.
    # Side-by-side (adjacent_to): delta_z ≈ 0 → ratio ≈ 0.
    # Dimensionless and scale-invariant — directly encodes stacking geometry.
    xy_dist_raw = float(np.linalg.norm(c_a[:2] - c_b[:2]))
    vz_ratio = float(np.clip(
        (c_a[2] - c_b[2]) / max(xy_dist_raw, avg_diag * 0.05), -3.0, 3.0
    ) / 3.0)

    # [20-21] Per-axis containment fractions (complement containment_3d).
    # containment_3d can be high from 2D XY overlap without full 3D containment.
    # Per-axis fractions give the model clearer signal for inside detection.
    containment_x = float(np.clip(ix / size_a[0], 0.0, 1.0))
    containment_y = float(np.clip(iy / size_a[1], 0.0, 1.0))

    return np.array([
        delta_x, delta_y, delta_z, xy_dist, dist_3d,
        bbox_overlap, vol_ratio, h_ratio, vert_gap, size_ratio_xy,
        vert_gap_obj_norm, vert_gap_a_relative, contact_score, support_overlap_b,
        ontop_z_margin, ontop_xy_margin, z_dom_margin,
        containment_3d, centroid_inside_b,
        vz_ratio, containment_x, containment_y,
    ], dtype=np.float32)
