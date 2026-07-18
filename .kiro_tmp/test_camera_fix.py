"""Quick test to verify the camera fix produces points at positive Z in camera space."""
import sys
sys.path.insert(0, ".")

import numpy as np
from scripts.segment_3rscan import generate_hemisphere_cameras
from src.gaussian.loader import load_gaussian_ply, filter_gaussians
import os

# Load a real scene
SPLATS_DIR = "D:/3rscan_splats"
scenes = []
for d in os.listdir(SPLATS_DIR):
    ply = os.path.join(SPLATS_DIR, d, "ckpts", "point_cloud_30000.ply")
    if os.path.exists(ply):
        scenes.append(d)
        break
if not scenes:
    print("ERROR: No valid scenes found")
    sys.exit(1)
scene_id = scenes[0]
ply_path = os.path.join(SPLATS_DIR, scene_id, "ckpts", "point_cloud_30000.ply")
print(f"Testing with scene: {scene_id}")
print(f"PLY path: {ply_path}")

cloud = load_gaussian_ply(ply_path)
cloud = filter_gaussians(cloud, opacity_threshold=0.05)
print(f"Gaussians: {cloud.num_gaussians}")
print(f"XYZ min: {cloud.xyz.min(axis=0)}")
print(f"XYZ max: {cloud.xyz.max(axis=0)}")
print(f"XYZ mean: {cloud.xyz.mean(axis=0)}")

scene_center = cloud.xyz.mean(axis=0)
scene_extent = cloud.xyz.max(axis=0) - cloud.xyz.min(axis=0)
radius = float(np.linalg.norm(scene_extent)) * 1.2
print(f"Scene center: {scene_center}")
print(f"Radius: {radius:.2f}")

cameras = generate_hemisphere_cameras(scene_center, radius, n_views=24)
print(f"\nGenerated {len(cameras)} cameras")

n_valid = 0
for i, (c2w, K) in enumerate(cameras):
    w2c = np.linalg.inv(c2w)
    # Transform first 100 points to camera space
    pts_cam = (w2c[:3, :3] @ cloud.xyz[:100].T).T + w2c[:3, 3]
    n_front = (pts_cam[:, 2] > 0).sum()
    
    # Also check all points
    pts_cam_all = (w2c[:3, :3] @ cloud.xyz.T).T + w2c[:3, 3]
    n_front_all = (pts_cam_all[:, 2] > 0.01).sum()
    
    if i < 5 or n_front < 50:
        print(f"  Camera {i:2d}: {n_front}/100 sample pts in front, "
              f"{n_front_all}/{cloud.num_gaussians} total in front, "
              f"Z range: [{pts_cam[:, 2].min():.2f}, {pts_cam[:, 2].max():.2f}]")
    
    if n_front_all > 0:
        n_valid += 1

print(f"\n{'='*60}")
print(f"RESULT: {n_valid}/24 cameras have points in front (should be 24)")
if n_valid == 24:
    print("SUCCESS: Camera fix is working correctly!")
else:
    print("FAILURE: Some cameras still have no points in front")

# Sanity check assertion from the task
for c2w, K in cameras[:3]:
    w2c = np.linalg.inv(c2w)
    pts_cam = (w2c[:3, :3] @ cloud.xyz[:100].T).T + w2c[:3, 3]
    assert (pts_cam[:, 2] > 0).sum() > 50, "Camera looking wrong way!"
print("\nAll sanity assertions passed!")
