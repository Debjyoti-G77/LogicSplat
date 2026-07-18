"""
Find frames where objects 3-7 project to the CENTER of the image (not edges).
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

cloud = load_gaussian_ply("D:/logicsplat_data/processed/scene_01/splat.ply")
filtered = filter_gaussians(cloud, opacity_threshold=0.1)
objects, _ = gaussian_to_objects(filtered)

with open("D:/logicsplat_data/processed/scene_01/ns_data/transforms.json") as f:
    t = json.load(f)

frames = t["frames"]
w = int(t["w"]); h = int(t["h"])
fx = float(t["fl_x"]); fy = float(t["fl_y"])
cx = float(t["cx"]); cy = float(t["cy"])

# For each object, find the frame where it projects closest to image center
print("Best frames for each object (closest to image center):")
for obj in objects:
    best_frame = None
    best_dist = 9999
    best_uv = None
    for frame in frames:
        uv = project_point(obj.centroid, frame["transform_matrix"],
                           fx, fy, cx, cy, w, h)
        if uv is None:
            continue
        u, v = uv
        dist = ((u - cx)**2 + (v - cy)**2)**0.5
        if dist < best_dist:
            best_dist = dist
            best_frame = frame
            best_uv = uv
    if best_frame:
        fname = os.path.basename(best_frame["file_path"])
        print(f"  Obj {obj.uid}: best frame={fname} uv=({best_uv[0]:.0f},{best_uv[1]:.0f}) "
              f"dist_from_center={best_dist:.0f}px")
    else:
        print(f"  Obj {obj.uid}: no valid projection found")

# Run YOLO on the best frame for obj 5
print("\n\nRunning YOLO on best frames for objects 3-7...")
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
images_dir = "D:/logicsplat_data/processed/scene_01/ns_data/images"

for obj in objects:
    if obj.uid not in [3, 4, 5, 6, 7]:
        continue
    best_frame = None
    best_dist = 9999
    best_uv = None
    for frame in frames:
        uv = project_point(obj.centroid, frame["transform_matrix"],
                           fx, fy, cx, cy, w, h)
        if uv is None:
            continue
        u, v = uv
        dist = ((u - cx)**2 + (v - cy)**2)**0.5
        if dist < best_dist:
            best_dist = dist
            best_frame = frame
            best_uv = uv

    if best_frame is None:
        continue

    fname = os.path.basename(best_frame["file_path"])
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

    u, v = best_uv
    matched = [f"{d[0]}({d[1]:.2f})" for d in detections
               if d[2] <= u <= d[4] and d[3] <= v <= d[5]]

    print(f"\n  Obj {obj.uid} best frame={fname} uv=({u:.0f},{v:.0f}):")
    for d in detections:
        print(f"    YOLO: {d[0]} conf={d[1]:.2f} [{d[2]:.0f},{d[3]:.0f},{d[4]:.0f},{d[5]:.0f}]")
    print(f"    -> Matched: {matched if matched else 'none'}")
