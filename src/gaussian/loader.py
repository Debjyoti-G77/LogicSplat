"""
Gaussian Splatting .ply loader.

Reads a Gaussian Splat .ply file and extracts per-Gaussian attributes:
    - position (xyz)
    - color (rgb from spherical harmonics DC component)
    - opacity
    - covariance (3D shape descriptor from scale + rotation)

This is the core of our original plan — using Gaussian attributes
as rich geometric features for object clustering and relation inference.
"""
import numpy as np
from plyfile import PlyData
from dataclasses import dataclass
from typing import Optional


@dataclass
class GaussianCloud:
    """Container for a loaded Gaussian Splat."""
    xyz:        np.ndarray   # (N, 3) positions
    rgb:        np.ndarray   # (N, 3) colors [0-255]
    opacity:    np.ndarray   # (N,)   opacity [0-1]
    scales:     np.ndarray   # (N, 3) log-scale per axis
    rotations:  np.ndarray   # (N, 4) quaternion [w, x, y, z]
    covariance: np.ndarray   # (N, 6) upper triangle of 3D covariance matrix

    @property
    def num_gaussians(self) -> int:
        return len(self.xyz)

    def __repr__(self):
        return f"<GaussianCloud: {self.num_gaussians:,} Gaussians>"


def load_gaussian_ply(path: str) -> GaussianCloud:
    """
    Load a Gaussian Splatting .ply file.
    Compatible with 3DGS standard format (Kerbl et al. 2023).

    Args:
        path: path to .ply file

    Returns:
        GaussianCloud with all attributes
    """
    ply = PlyData.read(path)
    v = ply['vertex']
    names = [p.name for p in v.properties]

    # ── positions ─────────────────────────────────────────────────────────────
    xyz = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)

    # ── colors from DC spherical harmonics component ──────────────────────────
    # 3DGS stores colors as SH coefficients: f_dc_0, f_dc_1, f_dc_2
    if 'f_dc_0' in names:
        # convert SH DC to RGB: C0 = 0.28209479177387814
        C0 = 0.28209479177387814
        r = (v['f_dc_0'] * C0 + 0.5).clip(0, 1)
        g = (v['f_dc_1'] * C0 + 0.5).clip(0, 1)
        b = (v['f_dc_2'] * C0 + 0.5).clip(0, 1)
        rgb = np.stack([r, g, b], axis=1)
        rgb = (rgb * 255).astype(np.uint8)
    elif 'red' in names:
        rgb = np.stack([v['red'], v['green'], v['blue']], axis=1).astype(np.uint8)
    else:
        rgb = np.ones((len(xyz), 3), dtype=np.uint8) * 128

    # ── opacity ───────────────────────────────────────────────────────────────
    if 'opacity' in names:
        # stored as logit, convert to probability
        raw_opacity = np.array(v['opacity'], dtype=np.float32)
        opacity = 1.0 / (1.0 + np.exp(-raw_opacity))
    else:
        opacity = np.ones(len(xyz), dtype=np.float32)

    # ── scales ────────────────────────────────────────────────────────────────
    if 'scale_0' in names:
        scales = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1).astype(np.float32)
    else:
        scales = np.zeros((len(xyz), 3), dtype=np.float32)

    # ── rotations (quaternion) ────────────────────────────────────────────────
    if 'rot_0' in names:
        rotations = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=1).astype(np.float32)
        # normalize quaternions
        norms = np.linalg.norm(rotations, axis=1, keepdims=True)
        rotations = rotations / np.maximum(norms, 1e-8)
    else:
        rotations = np.tile([1, 0, 0, 0], (len(xyz), 1)).astype(np.float32)

    # ── covariance from scale + rotation ─────────────────────────────────────
    covariance = _compute_covariance(scales, rotations)

    cloud = GaussianCloud(
        xyz=xyz,
        rgb=rgb,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
        covariance=covariance,
    )

    # guard against NaN/Inf from malformed PLY files
    valid = (
        np.isfinite(cloud.xyz).all(axis=1) &
        np.isfinite(cloud.opacity) &
        np.isfinite(cloud.scales).all(axis=1)
    )
    if not valid.all():
        n_bad = int((~valid).sum())
        import warnings
        warnings.warn(
            f"load_gaussian_ply: removed {n_bad} Gaussians with NaN/Inf values "
            f"({100*n_bad/len(xyz):.1f}% of total)",
            RuntimeWarning,
            stacklevel=2,
        )
        cloud = GaussianCloud(
            xyz=cloud.xyz[valid],
            rgb=cloud.rgb[valid],
            opacity=cloud.opacity[valid],
            scales=cloud.scales[valid],
            rotations=cloud.rotations[valid],
            covariance=cloud.covariance[valid],
        )

    return cloud


def _compute_covariance(scales: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """
    Compute 3D covariance matrices from scale and rotation.
    Returns upper triangle: [cov_xx, cov_xy, cov_xz, cov_yy, cov_yz, cov_zz]

    Σ = R · diag(exp(s))² · Rᵀ  where s = log-scale, R = rotation matrix

    Eigenvalues of Σ describe object shape:
        flat (table): λ₁ ≈ λ₂ >> λ₃
        elongated (bottle): λ₁ >> λ₂ ≈ λ₃
        spherical (ball): λ₁ ≈ λ₂ ≈ λ₃

    Vectorized — no Python loop over N Gaussians.
    """
    N = len(scales)

    # scale matrix: S = exp(log_scale) per axis
    S = np.exp(scales)  # (N, 3)

    # rotation matrix from quaternion [w, x, y, z]
    w = rotations[:, 0]
    x = rotations[:, 1]
    y = rotations[:, 2]
    z = rotations[:, 3]

    # build (N, 3, 3) rotation matrices — fully vectorized
    R = np.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], axis=1).reshape(N, 3, 3)  # (N, 3, 3)

    # RS = R * diag(S)  →  RS[i] = R[i] @ diag(S[i])
    # Equivalent to R * S[:, None, :] broadcast
    RS = R * S[:, np.newaxis, :]  # (N, 3, 3)

    # Σ = RS @ RSᵀ  →  vectorized matmul
    sigma = RS @ RS.transpose(0, 2, 1)  # (N, 3, 3)

    # extract upper triangle
    cov = np.stack([
        sigma[:, 0, 0], sigma[:, 0, 1], sigma[:, 0, 2],
        sigma[:, 1, 1], sigma[:, 1, 2], sigma[:, 2, 2],
    ], axis=1).astype(np.float32)  # (N, 6)

    return cov


def filter_gaussians(
    cloud: GaussianCloud,
    opacity_threshold: float = 0.1,
    max_scale: float = 1.0,
) -> GaussianCloud:
    """
    Remove low-opacity and oversized Gaussians (background/floaters).

    Args:
        opacity_threshold: remove Gaussians below this opacity
        max_scale: remove Gaussians larger than this in any axis (log scale)
    """
    mask = cloud.opacity > opacity_threshold
    if max_scale < float('inf'):
        scale_mask = np.all(cloud.scales < max_scale, axis=1)
        mask = mask & scale_mask

    return GaussianCloud(
        xyz=cloud.xyz[mask],
        rgb=cloud.rgb[mask],
        opacity=cloud.opacity[mask],
        scales=cloud.scales[mask],
        rotations=cloud.rotations[mask],
        covariance=cloud.covariance[mask],
    )


def remove_table_background(
    cloud: GaussianCloud,
    min_inlier_frac: float = 0.05,
    above_quantile: float = 0.97,
    margin_frac: float = 0.05,
    n_iterations: int = 1000,
    floor_inlier_thresh: float = 0.60,  # unused, kept for API compat
) -> GaussianCloud:
    """Remove floor/wall/ceiling Gaussians via single-pass RANSAC.

    Finds the dominant horizontal plane and keeps only the object region:
    Gaussians within [−margin, above_quantile-th height] above that plane.
    Falls back to the unchanged cloud if no dominant plane is found.
    """
    import warnings

    xyz = cloud.xyz
    N   = len(xyz)

    bbox_diag   = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    ransac_dist = max(bbox_diag * 0.01, 1e-4)

    n_sample = min(N, 50_000)
    rng = np.random.default_rng(42)
    idx = rng.choice(N, n_sample, replace=False) if N > n_sample else np.arange(N)
    xyz_s = xyz[idx].astype(np.float64)

    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_s)
        plane_eq, inliers_s = pcd.segment_plane(
            distance_threshold=float(ransac_dist),
            ransac_n=3,
            num_iterations=n_iterations,
            seed=42,
        )
    except Exception:
        return cloud

    if len(inliers_s) < min_inlier_frac * n_sample:
        return cloud

    a, b, c, d = [float(v) for v in plane_eq]
    norm_mag = float(np.sqrt(a*a + b*b + c*c))
    if norm_mag < 1e-6:
        return cloud

    n_vec       = np.array([a, b, c], dtype=np.float32) / norm_mag
    signed_dists = (cloud.xyz @ n_vec) + (d / norm_mag)

    # Objects sit on one side of the plane. Pick the side with smaller q95
    # (objects are bounded, background extends far on the other side).
    pos = signed_dists[signed_dists > 0]
    neg = -signed_dists[signed_dists < 0]
    pos_q95 = float(np.quantile(pos, 0.95)) if len(pos) >= 10 else np.inf
    neg_q95 = float(np.quantile(neg, 0.95)) if len(neg) >= 10 else np.inf
    heights = signed_dists if pos_q95 <= neg_q95 else -signed_dists

    positive = heights[heights > 0]
    if len(positive) < 100:
        return cloud

    height_cutoff = float(np.quantile(positive, above_quantile))
    margin        = height_cutoff * margin_frac
    mask          = (heights >= -margin) & (heights <= height_cutoff)

    if mask.sum() < max(int(0.10 * N), 50):
        warnings.warn(
            "remove_table_background: result < 10% of cloud, skipping",
            RuntimeWarning, stacklevel=2,
        )
        return cloud

    warnings.warn(
        f"remove_table_background: kept {mask.sum():,}/{N:,} "
        f"({100*mask.sum()/N:.1f}%)",
        RuntimeWarning, stacklevel=2,
    )
    return GaussianCloud(
        xyz=cloud.xyz[mask],
        rgb=cloud.rgb[mask],
        opacity=cloud.opacity[mask],
        scales=cloud.scales[mask],
        rotations=cloud.rotations[mask],
        covariance=cloud.covariance[mask],
    )


def prune_isolated_gaussians(
    cloud: GaussianCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> GaussianCloud:
    """Remove spatially isolated Gaussians using Statistical Outlier Removal (SOR).

    SOR computes for each Gaussian the mean distance to its nb_neighbors nearest
    neighbors. Gaussians whose mean distance exceeds mean + std_ratio * std of
    all mean distances are considered isolated floaters and removed.

    This is the standard approach in point cloud processing (PCL, Open3D).
    It removes floaters that survive opacity filtering because they are spatially
    isolated from the main scene geometry.

    Args:
        cloud:        GaussianCloud after opacity filtering
        nb_neighbors: number of nearest neighbors to consider (default 20)
        std_ratio:    threshold in standard deviations (default 2.0)

    Returns:
        GaussianCloud with isolated Gaussians removed
    """
    if cloud.num_gaussians < nb_neighbors + 1:
        return cloud

    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud.xyz.astype(np.float64))
        _, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
        mask = np.zeros(cloud.num_gaussians, dtype=bool)
        mask[ind] = True
    except ImportError:
        # Fallback: manual SOR using sklearn NearestNeighbors
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=nb_neighbors + 1).fit(cloud.xyz)
        distances, _ = nbrs.kneighbors(cloud.xyz)
        mean_dists = distances[:, 1:].mean(axis=1)  # exclude self (dist=0)
        threshold = mean_dists.mean() + std_ratio * mean_dists.std()
        mask = mean_dists <= threshold

    n_removed = int((~mask).sum())
    if n_removed > 0:
        import warnings
        warnings.warn(
            f"prune_isolated_gaussians: removed {n_removed} isolated Gaussians "
            f"({100*n_removed/cloud.num_gaussians:.1f}%)",
            RuntimeWarning,
            stacklevel=2,
        )

    return GaussianCloud(
        xyz=cloud.xyz[mask],
        rgb=cloud.rgb[mask],
        opacity=cloud.opacity[mask],
        scales=cloud.scales[mask],
        rotations=cloud.rotations[mask],
        covariance=cloud.covariance[mask],
    )
