"""Quick sanity check: verify edge features produce 14-dim output."""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.dataset.loader_scannet import _edge_features, EDGE_FEATURE_DIM

# Simulate two objects: A sitting on top of B
pts_a = np.array([[0.0, 0.0, 1.05], [0.1, 0.1, 1.15], [-0.05, -0.05, 1.0]], dtype=np.float32)
pts_b = np.array([[0.0, 0.0, 0.0], [0.2, 0.2, 1.0], [-0.1, -0.1, 0.0]], dtype=np.float32)
scene_extent = np.array([5.0, 5.0, 3.0], dtype=np.float32)

feats = _edge_features(pts_a, pts_b, scene_extent)
print(f"EDGE_FEATURE_DIM = {EDGE_FEATURE_DIM}")
print(f"Actual output dim = {len(feats)}")
print(f"Features: {feats}")
print(f"\n[10] vert_gap_obj_norm = {feats[10]:.4f}")
print(f"[11] vert_gap_abs      = {feats[11]:.4f}")
print(f"[12] contact_score     = {feats[12]:.4f}")
print(f"[13] support_overlap_b = {feats[13]:.4f}")
assert len(feats) == 14, f"Expected 14, got {len(feats)}"
print("\n✓ All good! 14-dim edge features working.")
