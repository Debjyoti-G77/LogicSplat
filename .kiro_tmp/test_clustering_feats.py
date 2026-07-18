"""Verify clustering.py edge features produce 14-dim output."""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.gaussian.clustering import extract_gaussian_edge_features, Object3D

# Create two objects: A on top of B
obj_a = Object3D(uid=0, centroid=np.array([0.0, 0.0, 1.1]),
                 bbox_min=np.array([-0.05, -0.05, 1.0]),
                 bbox_max=np.array([0.1, 0.1, 1.15]),
                 color=np.array([128, 128, 128], dtype=np.uint8), point_count=100)
obj_b = Object3D(uid=1, centroid=np.array([0.0, 0.0, 0.5]),
                 bbox_min=np.array([-0.1, -0.1, 0.0]),
                 bbox_max=np.array([0.2, 0.2, 1.0]),
                 color=np.array([128, 128, 128], dtype=np.uint8), point_count=200)
scene_extent = np.array([5.0, 5.0, 3.0], dtype=np.float32)

feats = extract_gaussian_edge_features(obj_a, obj_b, scene_extent)
print(f"Output dim = {len(feats)}")
print(f"[10] vert_gap_obj_norm = {feats[10]:.4f}")
print(f"[11] vert_gap_abs      = {feats[11]:.4f}")
print(f"[12] contact_score     = {feats[12]:.4f}")
print(f"[13] support_overlap_b = {feats[13]:.4f}")
assert len(feats) == 14, f"Expected 14, got {len(feats)}"
print("\n✓ clustering.py: 14-dim edge features working.")
