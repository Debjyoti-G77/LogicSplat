"""
Visualize projected centroids overlaid on the actual image to verify alignment.
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

# Find frame_00075
frame = None
for f in frames:
    if "frame_00075" in f["file_path"]:
        frame = f
        break

tm = frame["transform_matrix"]

# Project all objects
print("Projections in frame_00075:")
for obj in objects:
    uv = project_point(obj.centroid, tm, fx, fy, cx, cy, w, h)
    print(f"  Obj {obj.uid}: {uv}")

# Try to save annotated image
try:
    from PIL import Image, ImageDraw, ImageFont
    img_path = "D:/logicsplat_data/processed/scene_01/ns_data/images/frame_00075.png"
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)

    colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]
    for obj in objects:
        uv = project_point(obj.centroid, tm, fx, fy, cx, cy, w, h)
        if uv:
            u, v = uv
            r = 8
            draw.ellipse([u-r, v-r, u+r, v+r], fill=colors[obj.uid % len(colors)])
            draw.text((u+10, v-10), f"Obj{obj.uid}", fill=colors[obj.uid % len(colors)])

    # Also draw YOLO bboxes
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    preds = model(img_path, device="cpu", verbose=False, conf=0.20)
    for r in preds:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = model.names[int(box.cls)]
            conf = float(box.conf)
            draw.rectangle([x1, y1, x2, y2], outline="white", width=2)
            draw.text((x1, y1-15), f"{label} {conf:.2f}", fill="white")

    out_path = "debug_frame_00075_annotated.png"
    img.save(out_path)
    print(f"\nSaved annotated image to {out_path}")
    print("Open this image to see where centroids project vs YOLO detections")
except ImportError:
    print("\nPillow not available — skipping image visualization")
    print("Install with: pip install Pillow")
