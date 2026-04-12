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

    return GaussianCloud(
        xyz=xyz,
        rgb=rgb,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
        covariance=covariance,
    )


def _compute_covariance(scales: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """
    Compute 3D covariance matrices from scale and rotation.
    Returns upper triangle: [cov_xx, cov_xy, cov_xz, cov_yy, cov_yz, cov_zz]

    This is the key geometric descriptor — eigenvalues of the covariance
    tell you object shape: flat (table), elongated (bottle), spherical (ball).
    """
    N = len(scales)
    # scale matrix S = diag(exp(scale))
    S = np.exp(scales)  # (N, 3)

    # rotation matrix from quaternion [w, x, y, z]
    w, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    R = np.stack([
        1-2*(y*y+z*z),  2*(x*y-w*z),    2*(x*z+w*y),
        2*(x*y+w*z),    1-2*(x*x+z*z),  2*(y*z-w*x),
        2*(x*z-w*y),    2*(y*z+w*x),    1-2*(x*x+y*y),
    ], axis=1).reshape(N, 3, 3)  # (N, 3, 3)

    # Sigma = R * S^2 * R^T
    cov = np.zeros((N, 6), dtype=np.float32)
    for i in range(N):
        RS = R[i] * S[i]  # R * diag(S)
        sigma = RS @ RS.T  # (3, 3)
        cov[i] = [sigma[0,0], sigma[0,1], sigma[0,2],
                  sigma[1,1], sigma[1,2], sigma[2,2]]

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
