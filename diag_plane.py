"""Quick diagnostic: what plane does RANSAC find for each scene?"""
import sys, os, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np

from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians

DATA_DIR = "D:/logicsplat_data/processed"
SCENES = [f"scene_{i:02d}" for i in range(6, 14)]

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False
    print("open3d not available")

print(f"{'Scene':<12} {'Plane normal':>30}  {'Inliers%':>8}  {'Axis-align':>10}  {'Kept%':>6}")
print("-" * 75)

for sname in SCENES:
    ply = os.path.join(DATA_DIR, sname, "splat.ply")
    if not os.path.exists(ply):
        print(f"{sname:<12}  MISSING")
        continue

    cloud = load_gaussian_ply(ply)
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)
    xyz = cloud.xyz
    N   = len(xyz)

    bbox_diag = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    ransac_dist = max(bbox_diag * 0.01, 1e-4)

    n_sample = min(N, 50_000)
    rng = np.random.default_rng(42)
    idx = rng.choice(N, n_sample, replace=False)
    xyz_s = xyz[idx].astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_s)
    plane, inliers_s = pcd.segment_plane(
        distance_threshold=float(ransac_dist),
        ransac_n=3,
        num_iterations=1000,
    )
    a, b, c, d = [float(v) for v in plane]
    norm_mag = float(np.sqrt(a*a + b*b + c*c))
    n_vec = np.array([a, b, c]) / norm_mag
    inlier_pct = 100.0 * len(inliers_s) / n_sample
    axis_align = float(np.max(np.abs(n_vec)))  # how close to any axis

    # Compute what percentage RANSAC would keep
    signed_dists = (xyz @ n_vec.astype(np.float32)) + (d / norm_mag)
    pos = signed_dists[signed_dists > 0]
    neg = -signed_dists[signed_dists < 0]
    pos_q95 = float(np.quantile(pos, 0.95)) if len(pos) >= 10 else np.inf
    neg_q95 = float(np.quantile(neg, 0.95)) if len(neg) >= 10 else np.inf
    heights = signed_dists if pos_q95 <= neg_q95 else -signed_dists
    positive = heights[heights > 0]
    h_cut = float(np.quantile(positive, 0.85))
    margin = h_cut * 0.08
    kept = int(((heights >= -margin) & (heights <= h_cut)).sum())
    kept_pct = 100.0 * kept / N

    normal_str = f"({n_vec[0]:+.3f}, {n_vec[1]:+.3f}, {n_vec[2]:+.3f})"
    print(f"{sname:<12}  {normal_str:>30}  {inlier_pct:>7.1f}%  {axis_align:>10.3f}  {kept_pct:>5.1f}%")
