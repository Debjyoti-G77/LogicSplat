"""
Grounding DINO semantic labeling for Gaussian clusters.

Why Grounding DINO instead of YOLO:
    YOLO is limited to 80 COCO classes — it will never say "router",
    "perfume bottle", or "coke can".  Grounding DINO is open-vocabulary:
    you give it a text prompt with the object names you expect, and it
    finds them in the image.  This is much better for thesis-quality
    labeling of real household scenes.

How it works:
    1. Project each Gaussian cluster centroid into video frames using
       the same camera projection as yolo_labeler.py.
    2. Run Grounding DINO on each frame with a scene-specific text prompt.
    3. Match projected centroids to detected bounding boxes.
    4. Majority vote across frames → final label per object.
    5. Save to dino_labels.json (preferred over yolo_labels.json).

Installation:
    pip install groundingdino-py
    # weights are auto-downloaded on first run (~700 MB)

Usage:
    python src/labeling/grounding_dino_labeler.py --scene scene_01
    python src/labeling/grounding_dino_labeler.py --scene scene_01 --prompt "router, box, coke can, comb, perfume, hair dryer"
    python src/labeling/grounding_dino_labeler.py --scene scene_01 --force
"""
import json
import os
import sys
import argparse
import numpy as np
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, ".")
from src.graph.definitions import Object3D
from src.labeling.yolo_labeler import (
    load_transforms,
    project_point,
    point_in_bbox,
    _load_dataparser_transform,
)

# ── default prompts per scene ─────────────────────────────────────────────────
# These are your best guesses at what's in each scene.
# Grounding DINO will find whichever ones are actually visible.
# Separate items with " . " (period-space) — that's the DINO prompt format.
DEFAULT_PROMPTS: Dict[str, str] = {
    "scene_01": "router . box . coke can . comb . perfume bottle . hair dryer . book . remote control",
    "scene_02": "book . cup . bottle . scissors . toothbrush . box . remote control",
    "scene_03": "chair . table . monitor . keyboard . mouse . laptop . bottle . cup",
    "scene_04": "chair . table . monitor . keyboard . mouse . laptop . bottle . cup",
    "scene_05": "book . bottle . cup . box . remote control . phone",
}
FALLBACK_PROMPT = (
    "router . box . coke can . comb . perfume bottle . hair dryer . book . "
    "remote control . cup . bottle . chair . table . laptop . phone . keyboard"
)

# ── Grounding DINO weights ────────────────────────────────────────────────────
DINO_CONFIG_URL  = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_WEIGHTS_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
DINO_WEIGHTS_PATH = "models/groundingdino_swint_ogc.pth"
DINO_CONFIG_PATH  = "models/GroundingDINO_SwinT_OGC.py"


def _ensure_dino_weights():
    """Download Grounding DINO weights if not already present."""
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(DINO_WEIGHTS_PATH):
        print(f"  Downloading Grounding DINO weights (~700 MB)...")
        import urllib.request
        urllib.request.urlretrieve(DINO_WEIGHTS_URL, DINO_WEIGHTS_PATH)
        print(f"  Saved to {DINO_WEIGHTS_PATH}")

    if not os.path.exists(DINO_CONFIG_PATH):
        print(f"  Downloading Grounding DINO config...")
        import urllib.request
        urllib.request.urlretrieve(DINO_CONFIG_URL, DINO_CONFIG_PATH)
        print(f"  Saved to {DINO_CONFIG_PATH}")


def _load_dino_model(device: str = "cpu"):
    """Load Grounding DINO model. Raises ImportError if not installed."""
    try:
        from groundingdino.util.inference import load_model
    except ImportError:
        raise ImportError(
            "groundingdino not installed.\n"
            "Install with: pip install groundingdino-py\n"
            "Then re-run this script."
        )
    _ensure_dino_weights()
    model = load_model(DINO_CONFIG_PATH, DINO_WEIGHTS_PATH, device=device)
    return model


def run_dino_on_frames(
    images_dir: str,
    frame_filenames: List[str],
    prompt: str,
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
    device: str = "cpu",
) -> Dict[str, List[Dict]]:
    """
    Run Grounding DINO on selected frames.

    Returns {fname: [{"label": str, "bbox": [x1,y1,x2,y2], "conf": float}]}
    """
    from groundingdino.util.inference import predict
    from PIL import Image
    import torchvision.transforms as T

    model = _load_dino_model(device)

    transform = T.Compose([
        T.Resize((800, 1333)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    results = {}
    for fname in frame_filenames:
        fpath = os.path.join(images_dir, fname)
        if not os.path.exists(fpath):
            results[fname] = []
            continue

        image_pil = Image.open(fpath).convert("RGB")
        orig_w, orig_h = image_pil.size
        image_tensor = transform(image_pil)

        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )

        detections = []
        for box, logit, phrase in zip(boxes, logits, phrases):
            # boxes are in [cx, cy, w, h] normalized — convert to pixel xyxy
            cx, cy, bw, bh = box.tolist()
            x1 = (cx - bw / 2) * orig_w
            y1 = (cy - bh / 2) * orig_h
            x2 = (cx + bw / 2) * orig_w
            y2 = (cy + bh / 2) * orig_h
            detections.append({
                "label": phrase.strip(),
                "bbox":  [x1, y1, x2, y2],
                "conf":  float(logit),
            })
        results[fname] = detections

    return results


def label_objects_with_dino(
    objects: List[Object3D],
    transforms_path: str,
    images_dir: str,
    prompt: Optional[str] = None,
    n_frames: int = 30,
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
    scene_dir: Optional[str] = None,
    device: str = "cpu",
) -> List[Object3D]:
    """
    Assign open-vocabulary semantic labels to Gaussian clusters using
    Grounding DINO + camera projection. Majority vote across frames.

    Args:
        objects:         list of Object3D with centroids in nerfstudio world space
        transforms_path: path to transforms.json
        images_dir:      directory containing video frames
        prompt:          text prompt e.g. "router . box . coke can . comb"
                         If None, uses DEFAULT_PROMPTS for the scene or FALLBACK_PROMPT
        n_frames:        number of frames to sample
        box_threshold:   Grounding DINO box confidence threshold
        text_threshold:  Grounding DINO text matching threshold
        scene_dir:       scene root directory (for caching and dataparser transform)
        device:          "cpu" or "cuda"
    """
    if scene_dir is None:
        scene_dir = os.path.dirname(transforms_path)

    # ── load cached labels if available ──────────────────────────────────
    labels_path = os.path.join(scene_dir, "dino_labels.json")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            cached = json.load(f)
        print(f"  Loaded cached Grounding DINO labels from {labels_path}")
        for obj in objects:
            lbl = cached.get(str(obj.uid))
            if lbl:
                obj.label = lbl
                print(f"  Obj {obj.uid}: {obj.label} (cached)")
        return objects

    # ── resolve prompt ────────────────────────────────────────────────────
    if prompt is None:
        scene_id = os.path.basename(scene_dir)
        prompt = DEFAULT_PROMPTS.get(scene_id, FALLBACK_PROMPT)
    print(f"  Prompt: {prompt}")

    # ── load transforms ───────────────────────────────────────────────────
    transforms = load_transforms(transforms_path)
    frames_data = transforms.get("frames", [])
    if not frames_data:
        print("  No frames in transforms.json")
        return objects

    w  = int(transforms.get("w", 848))
    h  = int(transforms.get("h", 480))
    fx = float(transforms.get("fl_x", transforms.get("fx", w * 1.2)))
    fy = float(transforms.get("fl_y", transforms.get("fy", h * 1.2)))
    cx = float(transforms.get("cx", w / 2))
    cy = float(transforms.get("cy", h / 2))

    dataparser_transform = _load_dataparser_transform(scene_dir)

    # ── sample frames evenly ──────────────────────────────────────────────
    total = len(frames_data)
    step  = max(1, total // n_frames)
    sampled_frames = frames_data[::step][:n_frames]
    frame_names    = [os.path.basename(f["file_path"]) for f in sampled_frames]

    print(f"  Running Grounding DINO on {len(sampled_frames)} frames...")
    dino_results = run_dino_on_frames(
        images_dir, frame_names, prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    # ── vote ──────────────────────────────────────────────────────────────
    label_votes:     Dict[int, Dict[str, int]] = {o.uid: {} for o in objects}
    projection_hits: Dict[int, int]            = {o.uid: 0  for o in objects}

    for frame_data, fname in zip(sampled_frames, frame_names):
        tm = frame_data["transform_matrix"]
        frame_fx = float(frame_data.get("fl_x", fx))
        frame_fy = float(frame_data.get("fl_y", fy))
        frame_cx = float(frame_data.get("cx", cx))
        frame_cy = float(frame_data.get("cy", cy))
        frame_w  = int(frame_data.get("w", w))
        frame_h  = int(frame_data.get("h", h))

        detections = dino_results.get(fname, [])

        for obj in objects:
            uv = project_point(
                obj.centroid, tm,
                frame_fx, frame_fy, frame_cx, frame_cy,
                frame_w, frame_h,
                dataparser_transform=dataparser_transform,
            )
            if uv is None:
                continue
            projection_hits[obj.uid] += 1
            u, v = uv
            for det in detections:
                if point_in_bbox(u, v, det["bbox"]):
                    lbl = det["label"]
                    label_votes[obj.uid][lbl] = label_votes[obj.uid].get(lbl, 0) + 1

    # ── assign by majority vote ───────────────────────────────────────────
    total_frames = len(sampled_frames)
    for obj in objects:
        hits  = projection_hits[obj.uid]
        votes = label_votes[obj.uid]
        if hits / max(total_frames, 1) < 0.05:
            print(f"  ⚠ Obj {obj.uid}: only {hits}/{total_frames} frames projected")
        if votes:
            obj.label = max(votes, key=votes.get)
            print(f"  Obj {obj.uid}: '{obj.label}'  (votes: {votes})")
        else:
            obj.label = "object"
            print(f"  Obj {obj.uid}: no match in any frame")

    # ── cache ─────────────────────────────────────────────────────────────
    label_data = {str(obj.uid): obj.label for obj in objects}
    with open(labels_path, "w") as f:
        json.dump(label_data, f, indent=2)
    print(f"  Labels saved to {labels_path}")

    return objects


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
        description="Label Gaussian clusters with Grounding DINO (open-vocabulary)"
    )
    parser.add_argument("--scene", default="scene_01",
                        help="Scene name under data/processed/")
    parser.add_argument("--prompt", default=None,
                        help='Text prompt e.g. "router . box . coke can . comb"')
    parser.add_argument("--n-frames", type=int, default=30,
                        help="Number of frames to sample (default: 30)")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if cached labels exist")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    scene_dir       = f"D:/logicsplat_data/processed/{args.scene}"
    transforms_path = os.path.join(scene_dir, "ns_data", "transforms.json")
    images_dir      = os.path.join(scene_dir, "ns_data", "images")
    splat_path      = os.path.join(scene_dir, "splat.ply")

    if not os.path.exists(splat_path):
        print(f"splat.ply not found: {splat_path}")
        sys.exit(1)
    if not os.path.exists(transforms_path):
        print(f"transforms.json not found: {transforms_path}")
        sys.exit(1)

    # remove cache if --force
    if args.force:
        for fname in ("dino_labels.json", "yolo_labels.json"):
            p = os.path.join(scene_dir, fname)
            if os.path.exists(p):
                os.remove(p)
                print(f"  Removed {p}")

    from src.gaussian.loader import load_gaussian_ply, filter_gaussians
    from src.gaussian.clustering import gaussian_to_objects

    print(f"Loading {splat_path}...")
    cloud    = load_gaussian_ply(splat_path)
    filtered = filter_gaussians(cloud, opacity_threshold=0.1)
    objects, _ = gaussian_to_objects(filtered)
    print(f"Found {len(objects)} objects")

    objects = label_objects_with_dino(
        objects, transforms_path, images_dir,
        prompt=args.prompt,
        n_frames=args.n_frames,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        scene_dir=scene_dir,
        device=args.device,
    )

    print(f"\nFinal labels for {args.scene}:")
    for obj in objects:
        print(f"  Obj {obj.uid}: {obj.label}  "
              f"centroid=({obj.centroid[0]:.2f}, {obj.centroid[1]:.2f}, {obj.centroid[2]:.2f})")
