"""
Investigate coordinate convention: does applied_transform affect projection?
"""
import sys
sys.path.insert(0, ".")
import json, os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from src.gaussian.loader import load_gaussian_ply, filter_gaussians
from src.gaussian.clustering import gaussian_to_objects

cloud = load_gaussian_ply("D:/logicsplat_data/processed/scene_01/splat.ply")
filtered = filter_gaussians(cloud, opacity_threshold=0.1)
objects, _ = gaussian_to_objects(filtered)

with open("D:/logicsplat_data/processed/scene_01/ns_data/transforms.json") as f:
    t = json.load(f)

frames = t["frames"]
w = int(t["w"]); h = int(t["h"])
fx = float(t["fl_x"]); fy = float(t["fl_y"])
cx = float(t["cx"]); cy = float(t["cy"])

# applied_transform from transforms.json
applied_transform = np.array(t["applied_transform"], dtype=np.float64)
print("applied_transform (3x4):")
print(applied_transform)
# Pad to 4x4
AT = np.vstack([applied_transform, [0, 0, 0, 1]])
print("applied_transform (4x4):")
print(AT)
print()

# The camera poses in transforms.json are ALREADY in the applied space.
# The Gaussians from splat.ply are in the ORIGINAL nerfstudio training space.
# nerfstudio applies this transform to the cameras BEFORE training.
# So the Gaussians are in the SAME space as the cameras (both post-transform).
# BUT: let's verify by checking if applying AT to a centroid changes projection.

# Test with obj 5 (best frame = frame_00075, should project near center)
obj5 = objects[5]
frame = None
for f in frames:
    if "frame_00075" in f["file_path"]:
        frame = f
        break

print(f"Testing Obj 5 centroid: {obj5.centroid}")
print(f"Frame: frame_00075")
tm = np.array(frame["transform_matrix"])
w2c = np.linalg.inv(tm)

def project(point, w2c, fx, fy, cx, cy, w, h, flip_x=False):
    p_h = np.array([point[0], point[1], point[2], 1.0])
    p_cam = w2c @ p_h
    x = p_cam[0] if not flip_x else -p_cam[0]
    y = -p_cam[1]
    z = -p_cam[2]
    if z <= 0:
        return None
    u = fx * x / z + cx
    v = fy * y / z + cy
    if 0 <= u < w and 0 <= v < h:
        return (u, v)
    return None

# Current projection
uv = project(obj5.centroid, w2c, fx, fy, cx, cy, w, h)
print(f"  Current projection: {uv}")

# With X flipped
uv_flip = project(obj5.centroid, w2c, fx, fy, cx, cy, w, h, flip_x=True)
print(f"  X-flipped projection: {uv_flip}")

# Apply AT to centroid first, then project
p_h = np.array([obj5.centroid[0], obj5.centroid[1], obj5.centroid[2], 1.0])
p_at = (AT @ p_h)[:3]
print(f"  Centroid after AT: {p_at}")
uv_at = project(p_at, w2c, fx, fy, cx, cy, w, h)
print(f"  Projection after AT: {uv_at}")

# Apply AT inverse to centroid
AT_inv = np.linalg.inv(AT)
p_at_inv = (AT_inv @ p_h)[:3]
print(f"  Centroid after AT_inv: {p_at_inv}")
uv_at_inv = project(p_at_inv, w2c, fx, fy, cx, cy, w, h)
print(f"  Projection after AT_inv: {uv_at_inv}")

print()
print("YOLO detections in frame_00075 (from previous debug):")
print("  book conf=0.47 [0,176,202,269]")
print("  cup  conf=0.30 [62,235,158,387]")
print("  bottle conf=0.29 [63,235,159,387]")
print()
print("Expected: Obj 5 should project into one of these bboxes")
print("The objects are on the LEFT side of the image (x: 0-202)")
print("But current projection gives u=412 (right side)")
print()

# Try all 8 objects with X-flip
print("All objects with X-flipped projection in frame_00075:")
for obj in objects:
    uv = project(obj.centroid, w2c, fx, fy, cx, cy, w, h, flip_x=True)
    print(f"  Obj {obj.uid}: {uv}")
