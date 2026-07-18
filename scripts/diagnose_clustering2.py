"""
Deep sweep for scene_10 — find any combo that gives 4 clusters.
Also verify scene_09 leaf=200 gives correct spatial layout.
"""
import sys
sys.path.insert(0, ".")
import warnings
warnings.filterwarnings("ignore")

from src.gaussian.loader import load_gaussian_ply, filter_gaussians
from src.gaussian.clustering import auto_sat_threshold, gaussian_to_objects
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
import numpy as np

MAX_PTS = 8000
rng = np.random.default_rng(42)


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
    if len(X) > MAX_PTS:
        idx = rng.choice(len(X), MAX_PTS, replace=False)
        X = X[idx]
    return StandardScaler().fit_transform(X), mask.sum()


# ── scene_09: verify leaf mcs=200 gives 4 clusters with correct centroids ────
print("=== scene_09: verify leaf mcs=200 ===")
cloud09 = load_gaussian_ply("D:/logicsplat_data/processed/scene_09/splat.ply")
f09 = filter_gaussians(cloud09, opacity_threshold=0.1)
X09, _ = build_features(f09)
labels = HDBSCAN(min_cluster_size=200, min_samples=3,
                 cluster_selection_method="leaf", copy=False).fit_predict(X09)
n = len(set(labels)) - (1 if -1 in labels else 0)
print(f"  leaf mcs=200: {n} clusters")

# Also try full data (no subsample) with leaf mcs scaled to full size
# mcs=200 on 8000 pts → proportional mcs on full 18273 pts = 200 * 18273/8000 ≈ 457
sat09 = f09.rgb.astype(float).max(axis=1) - f09.rgb.astype(float).min(axis=1)
thresh09 = auto_sat_threshold(f09.rgb)
mask09 = sat09 > thresh09
if mask09.sum() < 0.05 * f09.num_gaussians:
    mask09 = sat09 > thresh09 * 0.5
xyz09 = f09.xyz[mask09]
rgb09 = f09.rgb[mask09]
cn09 = rgb09.astype(np.float32) / 255.0 * 0.3
X09_full = np.concatenate([xyz09, cn09], axis=1)
X09_full_scaled = StandardScaler().fit_transform(X09_full)
print(f"  Full data size: {len(X09_full_scaled)}")
for mcs in [200, 300, 400, 457, 500, 600, 800]:
    labels = HDBSCAN(min_cluster_size=mcs, min_samples=3,
                     cluster_selection_method="leaf", copy=False).fit_predict(X09_full_scaled)
    n = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = 100 * (labels == -1).sum() / len(labels)
    print(f"  leaf mcs={mcs} (full): {n} clusters  noise={noise_pct:.0f}%")

# ── scene_10: deep sweep ──────────────────────────────────────────────────────
print("\n=== scene_10: deep sweep ===")
cloud10 = load_gaussian_ply("D:/logicsplat_data/processed/scene_10/splat.ply")
f10 = filter_gaussians(cloud10, opacity_threshold=0.1)

sat10 = f10.rgb.astype(float).max(axis=1) - f10.rgb.astype(float).min(axis=1)
thresh10 = auto_sat_threshold(f10.rgb)
mask10 = sat10 > thresh10
if mask10.sum() < 0.05 * f10.num_gaussians:
    mask10 = sat10 > thresh10 * 0.5
xyz10 = f10.xyz[mask10]
rgb10 = f10.rgb[mask10]
cn10 = rgb10.astype(np.float32) / 255.0 * 0.3
X10_full = np.concatenate([xyz10, cn10], axis=1)
X10_full_scaled = StandardScaler().fit_transform(X10_full)
print(f"  Full colored data size: {len(X10_full_scaled)}")

print("  leaf sweep (full data):")
for mcs in [200, 300, 400, 500, 600, 800, 1000, 1500, 2000]:
    labels = HDBSCAN(min_cluster_size=mcs, min_samples=3,
                     cluster_selection_method="leaf", copy=False).fit_predict(X10_full_scaled)
    n = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = 100 * (labels == -1).sum() / len(labels)
    print(f"    leaf mcs={mcs}: {n} clusters  noise={noise_pct:.0f}%")

print("  eom sweep (full data):")
for mcs in [200, 300, 400, 500, 600, 800, 1000, 1500, 2000]:
    labels = HDBSCAN(min_cluster_size=mcs, min_samples=3,
                     cluster_selection_method="eom", copy=False).fit_predict(X10_full_scaled)
    n = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = 100 * (labels == -1).sum() / len(labels)
    print(f"    eom mcs={mcs}: {n} clusters  noise={noise_pct:.0f}%")

# Try XY-only (ignore Z) — router is ON TOP of agaro_box, so Z separates them
# but they may be spatially close in XY
print("  XY-only + color (eom, full):")
cn10_xy = rgb10.astype(np.float32) / 255.0 * 0.3
X10_xy = np.concatenate([xyz10[:, :2], cn10_xy], axis=1)
X10_xy_scaled = StandardScaler().fit_transform(X10_xy)
for mcs in [100, 200, 300, 400, 500]:
    labels = HDBSCAN(min_cluster_size=mcs, min_samples=3,
                     cluster_selection_method="eom", copy=False).fit_predict(X10_xy_scaled)
    n = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = 100 * (labels == -1).sum() / len(labels)
    print(f"    eom mcs={mcs}: {n} clusters  noise={noise_pct:.0f}%")

# Try XYZ + higher color weight to separate router (grey) from agaro_box (pink)
print("  XYZ + color_weight=1.0 (eom, full):")
cn10_hi = rgb10.astype(np.float32) / 255.0 * 1.0
X10_hi = np.concatenate([xyz10, cn10_hi], axis=1)
X10_hi_scaled = StandardScaler().fit_transform(X10_hi)
for mcs in [100, 200, 300, 400, 500, 800]:
    labels = HDBSCAN(min_cluster_size=mcs, min_samples=3,
                     cluster_selection_method="eom", copy=False).fit_predict(X10_hi_scaled)
    n = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = 100 * (labels == -1).sum() / len(labels)
    print(f"    eom mcs={mcs}: {n} clusters  noise={noise_pct:.0f}%")

print("  XYZ + color_weight=1.0 (leaf, full):")
for mcs in [100, 200, 300, 400, 500, 800]:
    labels = HDBSCAN(min_cluster_size=mcs, min_samples=3,
                     cluster_selection_method="leaf", copy=False).fit_predict(X10_hi_scaled)
    n = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = 100 * (labels == -1).sum() / len(labels)
    print(f"    leaf mcs={mcs}: {n} clusters  noise={noise_pct:.0f}%")

print("\nDone.")
