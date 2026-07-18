"""Debug projection for scene_01 — STEP 2 (fixed frame sync)"""
import sys
sys.path.insert(0, ".")
import json
import os
import numpy as np
from src.gaussian.loader import load_gaussian_ply, filter_gaussians
from src.gaussian.clustering import gaussian_to_objects
from src.labeling.yolo_labeler import project_point

import warnings
warnings.filterwarnings("ignore")

# ── Load objects ──────────────────────────────────────────────────────────────
print("Loading splat.ply and clustering...")
cloud = load_gaussian_ply("D:/logicsplat_data/processed/scene_01/splat.ply")
filtered = filter_gaussians(cloud, opacity_threshold=0.1)
objects, _ = gaussian_to_objects(filtered)
print(f"Found {len(objects)} objects")

# ── Load transforms ───────────────────────────────────────────────────────────
with open("D:/logicsplat_data/processed/scene_01/ns_data/transforms.json") as f:
    t = json.load(f)

frames = t["frames"]
w  = int(t["w"]);   h  = int(t["h"])
fx = float(t["fl_x"]); fy = float(t["fl_y"])
cx = float(t["cx"]); cy = float(t["cy"])

print(f"\nIntrinsics: w={w} h={h} fx={fx:.1f} fy={fy:.1f} cx={cx} cy={cy}")
print(f"Total frames: {len(frames)}")

# ── Find a frame near the middle of the video (by frame number) ───────────────
# transforms.json is in COLMAP order, not sequential — find frame_00167 (middle)
target_frame_num = 167
best_frame = None
best_diff = 9999
for f in frames:
    fname = os.path.basename(f["file_path"])
    try:
        num = int(fname.replace("frame_", "").replace(".png", ""))
        diff = abs(num - target_frame_num)
        if diff < best_diff:
            best_diff = diff
            best_frame = f
    except ValueError:
        pass

print(f"\nUsing frame closest to middle: {best_frame['file_path']}")
tm = best_frame["transform_matrix"]
cam_pos = np.array(tm)[:3, 3]
print(f"Camera position: {cam_pos}")

print("\nProjecting all object centroids:")
for obj in objects:
    uv = project_point(obj.centroid, tm, fx, fy, cx, cy, w, h)
    print(f"  Obj {obj.uid} centroid={np.round(obj.centroid, 3)}: uv={uv}")

# ── Scan all frames for each object ──────────────────────────────────────────
print("\nProjection coverage across ALL 334 frames:")
for obj in objects:
    hits = 0
    for frame in frames:
        uv = project_point(obj.centroid, frame["transform_matrix"],
                           fx, fy, cx, cy, w, h)
        if uv is not None:
            hits += 1
    print(f"  Obj {obj.uid} centroid={np.round(obj.centroid, 3)}: "
          f"{hits}/{len(frames)} frames ({100*hits/len(frames):.0f}%)")

# ── YOLO on the middle frame ──────────────────────────────────────────────────
images_dir = "D:/logicsplat_data/processed/scene_01/ns_data/images"
frame_file = os.path.basename(best_frame["file_path"])
frame_path = os.path.join(images_dir, frame_file)

if os.path.exists(frame_path):
    print(f"\nRunning YOLO on {frame_path}...")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    preds = model(frame_path, device="cpu", verbose=False, conf=0.25)
    print("YOLO detections:")
    for r in preds:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = model.names[int(box.cls)]
            conf = float(box.conf)
            print(f"  {label} conf={conf:.2f} bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

    # Check which centroids fall inside detections
    print("\nCentroid-to-detection matches:")
    for obj in objects:
        uv = project_point(obj.centroid, tm, fx, fy, cx, cy, w, h)
        if uv is None:
            print(f"  Obj {obj.uid}: no projection in this frame")
            continue
        u, v = uv
        matched = []
        for r in preds:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = model.names[int(box.cls)]
                if x1 <= u <= x2 and y1 <= v <= y2:
                    matched.append(f"{label}({float(box.conf):.2f})")
        if matched:
            print(f"  Obj {obj.uid} at ({u:.0f},{v:.0f}): MATCHED {matched}")
        else:
            print(f"  Obj {obj.uid} at ({u:.0f},{v:.0f}): no match")
else:
    print(f"\nFrame not found: {frame_path}")
