"""Test that the software renderer produces valid views after the camera fix."""
import sys
sys.path.insert(0, ".")

import os
import time
import numpy as np
from scripts.segment_3rscan import generate_hemisphere_cameras, render_gaussians_software
from src.gaussian.loader import load_gaussian_ply, filter_gaussians

SPLATS_DIR = "D:/3rscan_splats"

# Find first valid scene
scene_id = None
for d in os.listdir(SPLATS_DIR):
    ply = os.path.join(SPLATS_DIR, d, "ckpts", "point_cloud_30000.ply")
    if os.path.exists(ply):
        scene_id = d
        break

if not scene_id:
    print("ERROR: No valid scenes found")
    sys.exit(1)

print(f"Scene: {scene_id}")
ply_path = os.path.join(SPLATS_DIR, scene_id, "ckpts", "point_cloud_30000.ply")

cloud = load_gaussian_ply(ply_path)
cloud = filter_gaussians(cloud, opacity_threshold=0.05)
print(f"Gaussians: {cloud.num_gaussians}")

scene_center = cloud.xyz.mean(axis=0)
scene_extent = cloud.xyz.max(axis=0) - cloud.xyz.min(axis=0)
radius = float(np.linalg.norm(scene_extent)) * 1.2

cameras = generate_hemisphere_cameras(scene_center, radius, n_views=24)
print(f"Generated {len(cameras)} cameras, radius={radius:.2f}")

n_valid = 0
t0 = time.time()
for i, (c2w, K) in enumerate(cameras):
    view = render_gaussians_software(cloud, c2w, K)
    if view is not None:
        n_pixels = (view.rgb.sum(axis=2) > 0).sum()
        n_valid += 1
        if i < 5:
            print(f"  View {i:2d}: {n_pixels} non-zero pixels, "
                  f"depth range [{view.depth[view.depth > 0].min():.2f}, {view.depth[view.depth > 0].max():.2f}]")
    else:
        print(f"  View {i:2d}: NONE (no valid points)")

elapsed = time.time() - t0
print(f"\nRendered 24 views in {elapsed:.1f}s")
print(f"Valid views: {n_valid}/24")

if n_valid == 24:
    print("\nSUCCESS: All 24 views produced valid renders!")
else:
    print(f"\nFAILURE: Only {n_valid}/24 views are valid")
