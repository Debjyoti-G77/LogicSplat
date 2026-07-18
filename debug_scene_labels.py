"""
Show detailed YOLO vote breakdown for scene_01 and scene_02.
Runs with n_frames=60, conf=0.15 to maximize coverage.
"""
import sys
sys.path.insert(0, ".")
import json, os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from src.gaussian.loader import load_gaussian_ply, filter_gaussians
from src.gaussian.clustering import gaussian_to_objects
from src.labeling.yolo_labeler import (
    load_transforms, project_point, run_yolo_on_frames,
    _load_dataparser_transform, point_in_bbox
)

def detailed_label(scene_id, n_frames=60, confidence=0.15):
    scene_dir = f"D:/logicsplat_data/processed/{scene_id}"
    transforms_path = os.path.join(scene_dir, "ns_data", "transforms.json")
    images_dir = os.path.join(scene_dir, "ns_data", "images")

    print(f"\n{'='*60}")
    print(f"  {scene_id}  (n_frames={n_frames}, conf>={confidence})")
    print(f"{'='*60}")

    cloud = load_gaussian_ply(os.path.join(scene_dir, "splat.ply"))
    from src.gaussian.loader import filter_gaussians
    filtered = filter_gaussians(cloud, opacity_threshold=0.1)
    objects, _ = gaussian_to_objects(filtered)

    transforms = load_transforms(transforms_path)
    frames_data = transforms.get("frames", [])
    w  = int(transforms.get("w", 848))
    h  = int(transforms.get("h", 480))
    fx = float(transforms.get("fl_x", w * 1.2))
    fy = float(transforms.get("fl_y", h * 1.2))
    cx = float(transforms.get("cx", w / 2))
    cy = float(transforms.get("cy", h / 2))

    dataparser_transform = _load_dataparser_transform(scene_dir)

    total = len(frames_data)
    step = max(1, total // n_frames)
    sampled_frames = frames_data[::step][:n_frames]
    frame_names = [os.path.basename(f["file_path"]) for f in sampled_frames]

    print(f"  Sampling {len(sampled_frames)} of {total} frames")
    print(f"  Running YOLO...")

    yolo_results = run_yolo_on_frames(images_dir, frame_names, confidence)

    # Count total detections
    all_labels = {}
    for fname, dets in yolo_results.items():
        for d in dets:
            if d["conf"] >= confidence:
                all_labels[d["label"]] = all_labels.get(d["label"], 0) + 1
    print(f"\n  YOLO detected across all frames (label: count):")
    for label, count in sorted(all_labels.items(), key=lambda x: -x[1]):
        print(f"    {label}: {count}")

    # Vote per object
    label_votes = {o.uid: {} for o in objects}
    projection_hits = {o.uid: 0 for o in objects}

    for frame_data, fname in zip(sampled_frames, frame_names):
        tm = frame_data["transform_matrix"]
        detections = yolo_results.get(fname, [])
        for obj in objects:
            uv = project_point(obj.centroid, tm, fx, fy, cx, cy, w, h,
                               dataparser_transform=dataparser_transform)
            if uv is None:
                continue
            projection_hits[obj.uid] += 1
            u, v = uv
            for det in detections:
                if det["conf"] >= confidence and point_in_bbox(u, v, det["bbox"]):
                    lbl = det["label"]
                    label_votes[obj.uid][lbl] = label_votes[obj.uid].get(lbl, 0) + 1

    print(f"\n  Per-object results:")
    print(f"  {'Obj':>4}  {'Proj%':>6}  {'Top label':>12}  Votes")
    for obj in objects:
        hits = projection_hits[obj.uid]
        pct = 100 * hits / max(len(sampled_frames), 1)
        votes = label_votes[obj.uid]
        if votes:
            top = max(votes, key=votes.get)
            vote_str = ", ".join(f"{k}:{v}" for k, v in
                                 sorted(votes.items(), key=lambda x: -x[1]))
        else:
            top = "object"
            vote_str = "(no matches)"
        print(f"  {obj.uid:>4}  {pct:>5.0f}%  {top:>12}  {vote_str}")
        print(f"         centroid=({obj.centroid[0]:.2f}, {obj.centroid[1]:.2f}, {obj.centroid[2]:.2f})")

detailed_label("scene_01")
detailed_label("scene_02")
