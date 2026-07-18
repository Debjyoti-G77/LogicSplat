"""Debug YOLO labeling — check detections and projections."""
import sys
sys.path.insert(0, '.')
import json
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.gaussian.loader import load_gaussian_ply, filter_gaussians
from src.gaussian.clustering import gaussian_to_objects
from ultralytics import YOLO

# load scene
cloud = load_gaussian_ply('D:/logicsplat_data/processed/scene_01/splat.ply')
cf = filter_gaussians(cloud, opacity_threshold=0.1)
objects, params = gaussian_to_objects(cf)

print(f"Objects: {len(objects)}")
for o in objects:
    print(f"  Obj {o.uid} centroid={o.centroid.round(3)}")

# load transforms
with open('D:/logicsplat_data/processed/scene_01/ns_data/transforms.json') as f:
    transforms = json.load(f)

print(f"\nTransforms keys: {list(transforms.keys())[:10]}")
print(f"w={transforms.get('w')} h={transforms.get('h')}")
print(f"fl_x={transforms.get('fl_x')} fl_y={transforms.get('fl_y')}")
print(f"cx={transforms.get('cx')} cy={transforms.get('cy')}")
print(f"Total frames: {len(transforms.get('frames', []))}")

# check first frame
frame = transforms['frames'][100]
print(f"\nFrame 100: {frame['file_path']}")
print(f"Transform matrix:\n{np.array(frame['transform_matrix']).round(3)}")

# run YOLO on one frame
images_dir = 'D:/logicsplat_data/processed/scene_01/ns_data/images'
frames = sorted(os.listdir(images_dir))
fname = frames[100]
model = YOLO("yolov8n.pt")
preds = model(os.path.join(images_dir, fname), device='cpu', verbose=False)
print(f"\nYOLO detections on {fname}:")
for r in preds:
    if len(r.boxes) == 0:
        print("  No detections")
    for box in r.boxes:
        print(f"  {model.names[int(box.cls)]} conf={float(box.conf):.2f} bbox={box.xyxy[0].tolist()}")
