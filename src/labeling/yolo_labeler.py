"""
YOLO-based semantic labeling for Gaussian clusters.

Projects 3D Gaussian cluster centroids into video frames using
camera poses from transforms.json, then matches to YOLO detections.
Majority vote across frames assigns a semantic label to each cluster.

Coordinate convention fix (TASK 3):
    nerfstudio transforms.json stores camera-to-world (c2w) matrices in
    OpenGL / NeRF convention:
        +X = right,  +Y = up,  +Z = backward (camera looks down -Z)

    To project into pixel space we need OpenCV convention:
        +X = right,  +Y = down,  +Z = forward (camera looks down +Z)

    The correct axis flip after applying w2c is:
        x_cv =  x_gl          (X unchanged)
        y_cv = -y_gl          (flip Y: up → down)
        z_cv = -z_gl          (flip Z: backward → forward)

    This is what the code already does — the real bug was that many
    Gaussian centroids were in world-space coordinates that had been
    shifted/scaled by nerfstudio's scene normalisation.  We now read
    the dataparser_transforms.json (if present) and undo that transform
    before projecting, which is what makes most centroids project correctly.
"""
import json
import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from src.graph.definitions import Object3D


# ── transforms.json helpers ───────────────────────────────────────────────────

def load_transforms(transforms_path: str) -> Dict:
    with open(transforms_path) as f:
        return json.load(f)


def _load_dataparser_transform(scene_dir: str) -> Optional[np.ndarray]:
    """
    Load nerfstudio's dataparser_transforms.json if it exists.
    This file records the 4×4 world-space normalisation applied to the scene
    during training.  We need its inverse to bring Gaussian centroids back
    into the same coordinate frame as the camera poses in transforms.json.

    Returns a 4×4 float64 matrix, or None if the file doesn't exist.
    """
    candidates = [
        os.path.join(scene_dir, "dataparser_transforms.json"),
        os.path.join(scene_dir, "ns_data", "dataparser_transforms.json"),
        os.path.join(scene_dir, "ns_data", "colmap", "dataparser_transforms.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            # nerfstudio stores it as {"transform": [[...]], "scale": float}
            T = np.array(data["transform"], dtype=np.float64)
            if T.shape == (3, 4):
                # pad to 4×4
                T = np.vstack([T, [0, 0, 0, 1]])
            scale = float(data.get("scale", 1.0))
            # full transform: first scale, then rotate/translate
            S = np.eye(4, dtype=np.float64)
            S[:3, :3] *= scale
            return T @ S
    return None


def _apply_dataparser_inverse(
    point_world: np.ndarray,
    dataparser_transform: Optional[np.ndarray],
) -> np.ndarray:
    """
    Undo nerfstudio's dataparser normalisation so the point is in the same
    coordinate frame as the camera poses stored in transforms.json.
    """
    if dataparser_transform is None:
        return point_world
    T_inv = np.linalg.inv(dataparser_transform)
    p_h = np.array([point_world[0], point_world[1], point_world[2], 1.0])
    return (T_inv @ p_h)[:3]


# ── projection ────────────────────────────────────────────────────────────────

def project_point(
    point_3d: np.ndarray,
    transform_matrix: list,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    dataparser_transform: Optional[np.ndarray] = None,
) -> Optional[Tuple[float, float]]:
    """
    Project a 3D world-space point into 2D pixel coordinates.

    Handles nerfstudio's OpenGL camera convention (Y-up, Z-backward).
    Optionally undoes dataparser normalisation so Gaussian centroids
    (which live in nerfstudio's normalised space) project correctly.

    Returns (u, v) in pixels, or None if the point is behind the camera
    or outside the image bounds.
    """
    # ── step 1: undo dataparser normalisation if needed ───────────────────
    p_world = _apply_dataparser_inverse(np.asarray(point_3d, dtype=np.float64),
                                        dataparser_transform)

    # ── step 2: world → camera (OpenGL convention) ────────────────────────
    c2w = np.array(transform_matrix, dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    p_h = np.array([p_world[0], p_world[1], p_world[2], 1.0])
    p_cam = w2c @ p_h

    # ── step 3: OpenGL → OpenCV axis flip ────────────────────────────────
    # OpenGL: +Y up, +Z backward  →  OpenCV: +Y down, +Z forward
    x =  p_cam[0]
    y = -p_cam[1]
    z = -p_cam[2]

    if z <= 1e-6:
        return None

    # ── step 4: perspective projection ───────────────────────────────────
    u = fx * x / z + cx
    v = fy * y / z + cy

    if 0 <= u < width and 0 <= v < height:
        return (float(u), float(v))
    return None


# ── YOLO inference ────────────────────────────────────────────────────────────

def run_yolo_on_frames(
    images_dir: str,
    frame_indices: List[int],
    confidence: float = 0.3,
) -> Dict[str, List[Dict]]:
    """Run YOLO on selected frames. Returns {fname: [detections]}."""
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    frames = sorted(os.listdir(images_dir))
    selected = [frames[i] for i in frame_indices if i < len(frames)]
    results = {}
    for fname in selected:
        fpath = os.path.join(images_dir, fname)
        preds = model(fpath, device="cpu", verbose=False)
        detections = []
        for r in preds:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "label": model.names[int(box.cls)],
                    "bbox":  [x1, y1, x2, y2],
                    "conf":  float(box.conf),
                })
        results[fname] = detections
    return results


def point_in_bbox(u: float, v: float, bbox: List[float]) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= u <= x2 and y1 <= v <= y2


# ── main entry point ──────────────────────────────────────────────────────────

def label_objects_with_yolo(
    objects: List[Object3D],
    transforms_path: str,
    images_dir: str,
    n_frames: int = 30,
    confidence: float = 0.3,
    scene_dir: Optional[str] = None,
) -> List[Object3D]:
    """
    Assign semantic labels to Gaussian clusters using YOLO + camera projection.
    Uses majority vote across sampled frames.

    Args:
        objects:         list of Object3D with centroids in nerfstudio world space
        transforms_path: path to transforms.json (nerfstudio output)
        images_dir:      directory containing the video frames
        n_frames:        number of frames to sample for voting
        confidence:      YOLO detection confidence threshold
        scene_dir:       scene root directory — used to find
                         dataparser_transforms.json for coordinate correction.
                         Defaults to the directory containing transforms_path.
    """
    transforms = load_transforms(transforms_path)
    frames_data = transforms.get("frames", [])
    if not frames_data:
        print("No frames in transforms.json")
        return objects

    # intrinsics — prefer per-frame values, fall back to top-level
    w = int(transforms.get("w", 848))
    h = int(transforms.get("h", 480))
    fx = float(transforms.get("fl_x", transforms.get("fx", w * 1.2)))
    fy = float(transforms.get("fl_y", transforms.get("fy", h * 1.2)))
    cx = float(transforms.get("cx", w / 2))
    cy = float(transforms.get("cy", h / 2))

    # load dataparser normalisation (may be None)
    if scene_dir is None:
        scene_dir = os.path.dirname(transforms_path)
    dataparser_transform = _load_dataparser_transform(scene_dir)
    if dataparser_transform is not None:
        print("  Loaded dataparser_transforms.json — applying coordinate correction")
    else:
        print("  No dataparser_transforms.json found — assuming centroids are "
              "already in transforms.json world space")

    # sample frames evenly
    total = len(frames_data)
    step = max(1, total // n_frames)
    sampled_frames = frames_data[::step][:n_frames]
    frame_names = [os.path.basename(f["file_path"]) for f in sampled_frames]
    frame_indices = list(range(0, total, step))[:n_frames]

    print(f"Running YOLO on {len(sampled_frames)} frames...")
    yolo_results = run_yolo_on_frames(images_dir, frame_indices, confidence)

    label_votes: Dict[int, Dict[str, int]] = {o.uid: {} for o in objects}
    projection_hits: Dict[int, int] = {o.uid: 0 for o in objects}

    for frame_data, fname in zip(sampled_frames, frame_names):
        transform_matrix = frame_data["transform_matrix"]

        # per-frame intrinsics override if present
        frame_fx = float(frame_data.get("fl_x", fx))
        frame_fy = float(frame_data.get("fl_y", fy))
        frame_cx = float(frame_data.get("cx", cx))
        frame_cy = float(frame_data.get("cy", cy))
        frame_w  = int(frame_data.get("w", w))
        frame_h  = int(frame_data.get("h", h))

        detections = yolo_results.get(fname, [])

        for obj in objects:
            uv = project_point(
                obj.centroid,
                transform_matrix,
                frame_fx, frame_fy,
                frame_cx, frame_cy,
                frame_w, frame_h,
                dataparser_transform=dataparser_transform,
            )
            if uv is None:
                continue

            projection_hits[obj.uid] += 1
            u, v = uv

            for det in detections:
                if det["conf"] >= confidence and point_in_bbox(u, v, det["bbox"]):
                    label = det["label"]
                    label_votes[obj.uid][label] = (
                        label_votes[obj.uid].get(label, 0) + 1
                    )

    # report projection coverage
    total_frames = len(sampled_frames)
    for obj in objects:
        hits = projection_hits[obj.uid]
        coverage = hits / max(total_frames, 1)
        if coverage < 0.05:
            print(f"  ⚠ Obj {obj.uid}: only {hits}/{total_frames} frames "
                  f"projected — centroid may be outside scene bounds")

    # assign labels by majority vote
    for obj in objects:
        votes = label_votes[obj.uid]
        if votes:
            obj.label = max(votes, key=votes.get)
            print(f"  Obj {obj.uid}: {obj.label} (votes: {votes})")
        else:
            obj.label = "object"

    return objects


# ── debug helper ──────────────────────────────────────────────────────────────

def debug_projection(
    transforms_path: str,
    test_point: Optional[np.ndarray] = None,
    scene_dir: Optional[str] = None,
) -> None:
    """
    Debug projection by testing a known world-space point against all frames.
    Prints how many frames the point projects into and the pixel coordinates.

    Usage:
        from src.labeling.yolo_labeler import debug_projection
        debug_projection("data/processed/scene_01/ns_data/transforms.json",
                         test_point=np.array([0.0, 0.0, 0.0]))
    """
    transforms = load_transforms(transforms_path)
    frames_data = transforms.get("frames", [])
    w = int(transforms.get("w", 848))
    h = int(transforms.get("h", 480))
    fx = float(transforms.get("fl_x", transforms.get("fx", w * 1.2)))
    fy = float(transforms.get("fl_y", transforms.get("fy", h * 1.2)))
    cx = float(transforms.get("cx", w / 2))
    cy = float(transforms.get("cy", h / 2))

    if scene_dir is None:
        scene_dir = os.path.dirname(transforms_path)
    dataparser_transform = _load_dataparser_transform(scene_dir)

    if test_point is None:
        # use scene centroid as default test point
        test_point = np.zeros(3)

    hits = 0
    for i, frame in enumerate(frames_data):
        uv = project_point(
            test_point,
            frame["transform_matrix"],
            fx, fy, cx, cy, w, h,
            dataparser_transform=dataparser_transform,
        )
        if uv is not None:
            hits += 1
            if hits <= 5:
                print(f"  Frame {i:04d}: ({uv[0]:.1f}, {uv[1]:.1f})")

    print(f"\nPoint {test_point} projects into {hits}/{len(frames_data)} frames")
    if hits == 0:
        print("  → Try passing scene_dir so dataparser_transforms.json is found,")
        print("    or check that the point is within the scene bounding box.")
