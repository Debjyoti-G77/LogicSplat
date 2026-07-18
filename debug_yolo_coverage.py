"""
Debug: check what YOLO detects across all frames where objects project,
and why only 3/8 objects got labels.
"""
import sys
sys.path.insert(0, ".")
import json, os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from src.gaussian.loader import load_gaussian_ply, filter_gaussians
from src.gaussian.clustering import gaussian_to_objects
from src.labeling.yolo_labeler import project_point

# Load
cloud = load_gaussian_ply("D:/logicsplat_data/processed/scene_01/splat.ply")
filtered = filter_gaussians(cloud, opacity_threshold=0.1)
objects, _ = gaussian_to_objects(filtered)

with open("D:/logicsplat_data/processed/scene_01/ns_data/transforms.json") as f:
    t = json.load(f)

frames = t["frames"]
w = int(t["w"]); h = int(t["h"])
fx = float(t["fl_x"]); fy = float(t["fl_y"])
cx = float(t["cx"]); cy = float(t["cy"])

# Find frames where objects 3-7 project (they got 0 votes)
print("Finding frames where unlabeled objects (3-7) project...")
target_objs = [o for o in objects if o.uid in [3, 4, 5, 6, 7]]

good_frames = []
for frame in frames:
    tm = frame["transform_matrix"]
    projections = {}
    for obj in target_objs:
        uv = project_point(obj.centroid, tm, fx, fy, cx, cy, w, h)
        if uv is not None:
            projections[obj.uid] = uv
    if len(projections) >= 3:  # frame shows at least 3 of the target objects
        good_frames.append((frame, projections))

print(f"Found {len(good_frames)} frames showing 3+ unlabeled objects")

# Run YOLO on first 5 such frames
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
images_dir = "D:/logicsplat_data/processed/scene_01/ns_data/images"

print("\nYOLO results on frames with good coverage:")
for i, (frame, projections) in enumerate(good_frames[:8]):
    fname = os.path.basename(frame["file_path"])
    fpath = os.path.join(images_dir, fname)
    if not os.path.exists(fpath):
        continue

    preds = model(fpath, device="cpu", verbose=False, conf=0.20)
    detections = []
    for r in preds:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = model.names[int(box.cls)]
            conf = float(box.conf)
            detections.append((label, conf, x1, y1, x2, y2))

    print(f"\n  Frame {fname}:")
    if detections:
        for label, conf, x1, y1, x2, y2 in detections:
            print(f"    YOLO: {label} conf={conf:.2f} [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
    else:
        print("    YOLO: no detections")

    for uid, (u, v) in projections.items():
        matched = [f"{d[0]}({d[1]:.2f})" for d in detections
                   if d[2] <= u <= d[4] and d[3] <= v <= d[5]]
        print(f"    Obj {uid} at ({u:.0f},{v:.0f}): {matched if matched else 'no match'}")
