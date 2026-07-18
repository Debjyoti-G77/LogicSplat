"""
Render-and-Lift Segmentation for 3RScan Gaussian Splats using Grounded-SAM2.

Assigns per-Gaussian instance labels by:
    1. Rendering the Gaussian splat from multiple hemisphere viewpoints
    2. Running Grounding DINO + SAM2 on each rendered view
    3. Back-projecting 2D masks onto Gaussians via depth/visibility buffer
    4. Accumulating votes across views for final per-Gaussian instance assignment

Output per scene: D:/3rscan_splats/{scene_id}/instance_labels.npz
    - labels:      (N,) int array — instance ID per Gaussian (-1 = unassigned)
    - confidences: (N,) float array — assignment confidence
    - object_map:  dict {int_id: label_string}

Hardware: GPU with >=8GB VRAM (SAM2 inference)
Estimated: ~1-3 min/scene × 566 scenes = 10-30 hours total

Usage:
    python scripts/segment_3rscan.py
    python scripts/segment_3rscan.py --max-scenes 10 --n-views 12
    python scripts/segment_3rscan.py --scene-id <uuid>
    python scripts/segment_3rscan.py --verify-only

Author: LogicSplat Team
"""
import sys
sys.path.insert(0, ".")

import os
import json
import math
import warnings
import argparse
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.gaussian.loader import load_gaussian_ply, filter_gaussians, GaussianCloud


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

SPLATS_DIR = "D:/3rscan_splats"
OBJECTS_PATH = "data/3DSSG/objects.json"
RELATIONSHIPS_PATH = "data/3DSSG/relationships.json"

RENDER_WIDTH = 640
RENDER_HEIGHT = 480
N_VIEWS = 24
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Grounding DINO thresholds
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25

# Minimum confidence to accept a Gaussian assignment
MIN_VOTE_CONFIDENCE = 0.2


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RenderedView:
    """A single rendered view of the Gaussian splat."""
    rgb: np.ndarray           # (H, W, 3) uint8
    depth: np.ndarray         # (H, W) float32 — depth per pixel
    gaussian_ids: np.ndarray  # (H, W) int32 — frontmost Gaussian index per pixel
    c2w: np.ndarray           # (4, 4) camera-to-world matrix
    K: np.ndarray             # (3, 3) intrinsic matrix


@dataclass
class MaskDetection:
    """A detected + segmented object in a single view."""
    mask: np.ndarray      # (H, W) bool
    label: str            # detected class label
    confidence: float     # detection confidence
    bbox: List[float]     # [x1, y1, x2, y2]


# ══════════════════════════════════════════════════════════════════════════════
# Camera generation — Fibonacci hemisphere sampling
# ══════════════════════════════════════════════════════════════════════════════

def generate_hemisphere_cameras(
    scene_center: np.ndarray,
    radius: float,
    n_views: int = 24,
    img_width: int = RENDER_WIDTH,
    img_height: int = RENDER_HEIGHT,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate camera poses on upper hemisphere looking at scene center.
    Uses Fibonacci sphere sampling for uniform angular coverage.

    Args:
        scene_center: (3,) center of the scene
        radius:       distance from center to camera
        n_views:      number of viewpoints
        img_width:    render width
        img_height:   render height

    Returns:
        List of (c2w, K) tuples where c2w is 4x4 camera-to-world
        and K is 3x3 intrinsic matrix.
    """
    cameras = []
    golden_ratio = (1 + 5**0.5) / 2

    # Reasonable FOV for indoor scenes
    fov_x = math.radians(60)
    fx = img_width / (2 * math.tan(fov_x / 2))
    fy = fx  # square pixels
    cx = img_width / 2.0
    cy = img_height / 2.0
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], dtype=np.float32)

    for i in range(n_views):
        # Fibonacci hemisphere: polar angle from 0 to ~75 degrees
        # (avoid looking straight down which gives poor coverage)
        theta = math.acos(1 - 0.75 * (i + 0.5) / n_views)
        phi = 2 * math.pi * i / golden_ratio

        # Camera position on hemisphere
        x = radius * math.sin(theta) * math.cos(phi) + scene_center[0]
        y = radius * math.sin(theta) * math.sin(phi) + scene_center[1]
        z = radius * math.cos(theta) + scene_center[2]
        cam_pos = np.array([x, y, z], dtype=np.float32)

        # Look-at: camera Z axis points from camera toward scene center
        forward = scene_center - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        # Up vector (world Z up, with fallback)
        world_up = np.array([0, 0, 1], dtype=np.float32)
        if abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0, 1, 0], dtype=np.float32)

        # Right = forward × up
        right = np.cross(forward, world_up)
        right = right / (np.linalg.norm(right) + 1e-8)

        # Recompute up = right × forward (orthogonal to both right and forward)
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-8)

        # Camera-to-world: columns are [right, up, forward, position]
        # Convention: camera looks along +Z so that w2c puts scene at positive Z
        # (the software renderer checks pts_cam[:, 2] > 0)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward  # camera +Z = look direction
        c2w[:3, 3] = cam_pos

        cameras.append((c2w, K))

    return cameras


# ══════════════════════════════════════════════════════════════════════════════
# Gaussian Splatting Renderer (using gsplat)
# ══════════════════════════════════════════════════════════════════════════════

def render_gaussians(
    cloud: GaussianCloud,
    c2w: np.ndarray,
    K: np.ndarray,
    img_width: int = RENDER_WIDTH,
    img_height: int = RENDER_HEIGHT,
    device: str = DEVICE,
) -> Optional[RenderedView]:
    """
    Render a Gaussian splat from a given camera pose using gsplat.

    Returns a RenderedView with RGB, depth, and per-pixel Gaussian IDs,
    or None if rendering fails.
    """
    try:
        from gsplat import rasterization
    except ImportError:
        raise ImportError(
            "gsplat not installed. Install with:\n"
            "  pip install gsplat\n"
            "See https://docs.gsplat.studio/main/install.html"
        )

    N = cloud.num_gaussians

    # Convert to torch tensors
    means = torch.tensor(cloud.xyz, dtype=torch.float32, device=device)
    scales = torch.exp(torch.tensor(cloud.scales, dtype=torch.float32, device=device))
    quats = torch.tensor(cloud.rotations, dtype=torch.float32, device=device)
    opacities = torch.tensor(cloud.opacity, dtype=torch.float32, device=device)

    # Colors from RGB (normalized to [0,1])
    colors = torch.tensor(
        cloud.rgb.astype(np.float32) / 255.0,
        dtype=torch.float32, device=device,
    )

    # Camera matrices — gsplat expects viewmats as (C, 4, 4) full w2c
    w2c = np.linalg.inv(c2w)
    viewmat = torch.tensor(w2c, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 4, 4)
    Kt = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 3, 3)

    try:
        # gsplat rasterization returns (rendered_colors, rendered_alphas, meta)
        render_colors, render_alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=Kt,
            width=img_width,
            height=img_height,
            render_mode="RGB+ED",  # RGB + expected depth
        )
    except Exception as e:
        warnings.warn(f"gsplat rasterization failed: {e}", RuntimeWarning)
        return None

    # Extract RGB image (H, W, 3) and depth (H, W)
    rgb_tensor = render_colors[0, :, :, :3]  # (H, W, 3)
    rgb_np = (rgb_tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    # Depth from the 4th channel (expected depth)
    if render_colors.shape[-1] > 3:
        depth_np = render_colors[0, :, :, 3].cpu().numpy()
    else:
        depth_np = np.zeros((img_height, img_width), dtype=np.float32)

    # Per-pixel Gaussian IDs: approximate by finding nearest Gaussian
    # to each pixel's 3D point (from depth + camera ray)
    gaussian_ids = _compute_gaussian_ids(
        cloud, depth_np, c2w, K, img_width, img_height, device
    )

    return RenderedView(
        rgb=rgb_np,
        depth=depth_np,
        gaussian_ids=gaussian_ids,
        c2w=c2w,
        K=K,
    )


def _compute_gaussian_ids(
    cloud: GaussianCloud,
    depth: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    img_width: int,
    img_height: int,
    device: str,
) -> np.ndarray:
    """
    Compute per-pixel Gaussian index by back-projecting pixels to 3D
    and finding the nearest Gaussian to each 3D point.

    Uses a KD-tree for efficient nearest-neighbor lookup.
    Only processes pixels with valid depth (> 0).
    """
    from scipy.spatial import cKDTree

    # Build KD-tree of Gaussian positions
    tree = cKDTree(cloud.xyz)

    # Generate pixel grid
    u_coords = np.arange(img_width)
    v_coords = np.arange(img_height)
    uu, vv = np.meshgrid(u_coords, v_coords)  # (H, W) each

    # Back-project to 3D: p_cam = K^{-1} * [u, v, 1]^T * depth
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Camera-space 3D points
    z = depth.flatten()
    valid_mask = z > 0.01  # skip invalid depth

    x_cam = ((uu.flatten() - cx) / fx) * z
    y_cam = ((vv.flatten() - cy) / fy) * z
    pts_cam = np.stack([x_cam, y_cam, z], axis=1)  # (H*W, 3)

    # Transform to world space
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    pts_world = (R @ pts_cam.T).T + t  # (H*W, 3)

    # Query nearest Gaussian for valid pixels
    gaussian_ids = np.full(img_height * img_width, -1, dtype=np.int32)
    if valid_mask.sum() > 0:
        valid_pts = pts_world[valid_mask]
        # Use max_distance to avoid assigning far-away Gaussians
        dists, indices = tree.query(valid_pts, k=1, workers=-1)
        # Only assign if within reasonable distance (3x median Gaussian scale)
        median_scale = np.median(np.exp(cloud.scales).max(axis=1))
        max_dist = max(median_scale * 5, 0.05)
        close_mask = dists < max_dist
        valid_indices = np.where(valid_mask)[0]
        gaussian_ids[valid_indices[close_mask]] = indices[close_mask]

    return gaussian_ids.reshape(img_height, img_width)


# ══════════════════════════════════════════════════════════════════════════════
# Fallback renderer (software rasterization for when gsplat is unavailable)
# ══════════════════════════════════════════════════════════════════════════════

def render_gaussians_software(
    cloud: GaussianCloud,
    c2w: np.ndarray,
    K: np.ndarray,
    img_width: int = RENDER_WIDTH,
    img_height: int = RENDER_HEIGHT,
) -> Optional[RenderedView]:
    """
    Software-based approximate rendering by projecting Gaussian centers.
    Much faster than full splatting but lower quality.
    Used as fallback when gsplat is not available.

    Uses vectorized numpy operations with argsort-based z-buffering
    for performance on large point clouds.
    """
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3]
    t = w2c[:3, 3]

    # Project all Gaussians to camera space
    pts_cam = (R @ cloud.xyz.T).T + t  # (N, 3)

    # Only keep points in front of camera
    valid = pts_cam[:, 2] > 0.01
    if valid.sum() == 0:
        return None

    # Project to pixel coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Work only with valid points for efficiency
    valid_indices = np.where(valid)[0]
    pts_valid = pts_cam[valid_indices]

    u = (pts_valid[:, 0] * fx / pts_valid[:, 2] + cx).astype(np.int32)
    v = (pts_valid[:, 1] * fy / pts_valid[:, 2] + cy).astype(np.int32)
    z = pts_valid[:, 2]

    # Filter to points within image bounds
    in_bounds = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds]
    orig_indices = valid_indices[in_bounds]

    if len(u) == 0:
        return None

    # Initialize buffers
    rgb = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    depth = np.full((img_height, img_width), np.inf, dtype=np.float32)
    gaussian_ids = np.full((img_height, img_width), -1, dtype=np.int32)

    # Z-buffer rendering: sort back-to-front so closer points overwrite farther ones
    # (last write wins — closer points are written last)
    order = np.argsort(-z)  # back to front (descending z)
    u = u[order]
    v = v[order]
    z_sorted = z[order]
    orig_indices = orig_indices[order]

    # Vectorized write: since we sorted back-to-front, later (closer) writes
    # overwrite earlier (farther) ones at the same pixel — correct z-buffering
    depth[v, u] = z_sorted
    rgb[v, u] = cloud.rgb[orig_indices]
    gaussian_ids[v, u] = orig_indices

    # Replace inf with 0 for invalid depth
    depth[depth == np.inf] = 0

    return RenderedView(
        rgb=rgb,
        depth=depth,
        gaussian_ids=gaussian_ids,
        c2w=c2w,
        K=K,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Grounded-SAM2 Segmentation
# ══════════════════════════════════════════════════════════════════════════════

_GROUNDING_DINO_MODEL = None
_SAM2_PREDICTOR = None


def _load_grounding_dino(device: str = DEVICE):
    """Load Grounding DINO model (cached singleton)."""
    global _GROUNDING_DINO_MODEL
    if _GROUNDING_DINO_MODEL is not None:
        return _GROUNDING_DINO_MODEL

    try:
        from groundingdino.util.inference import load_model
    except ImportError:
        raise ImportError(
            "groundingdino not installed.\n"
            "Install: pip install groundingdino-py"
        )

    # Use existing weights path from the project
    config_path = "models/GroundingDINO_SwinT_OGC.py"
    weights_path = "models/groundingdino_swint_ogc.pth"

    if not os.path.exists(weights_path):
        print("Downloading Grounding DINO weights (~700 MB)...")
        import urllib.request
        os.makedirs("models", exist_ok=True)
        urllib.request.urlretrieve(
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
            weights_path,
        )
    if not os.path.exists(config_path):
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            config_path,
        )

    _GROUNDING_DINO_MODEL = load_model(config_path, weights_path, device=device)
    return _GROUNDING_DINO_MODEL


def _load_sam2(device: str = DEVICE):
    """Load SAM2 predictor (cached singleton). Falls back to SAM v1."""
    global _SAM2_PREDICTOR
    if _SAM2_PREDICTOR is not None:
        return _SAM2_PREDICTOR

    # Try SAM2 first
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # SAM2 checkpoint — use the smallest model for speed
        checkpoint = "models/sam2_hiera_small.pt"
        config = "sam2_hiera_s.yaml"
        if os.path.exists(checkpoint):
            sam2_model = build_sam2(config, checkpoint, device=device)
            _SAM2_PREDICTOR = SAM2ImagePredictor(sam2_model)
            print("  Loaded SAM2 (hiera_small)")
            return _SAM2_PREDICTOR
    except ImportError:
        pass

    # Fallback to SAM v1
    try:
        from segment_anything import sam_model_registry, SamPredictor

        checkpoint = "models/sam_vit_b_01ec64.pth"
        if not os.path.exists(checkpoint):
            print("  Downloading SAM ViT-B checkpoint (~375 MB)...")
            import urllib.request
            os.makedirs("models", exist_ok=True)
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                checkpoint,
            )
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
        sam.to(device)
        _SAM2_PREDICTOR = SamPredictor(sam)
        print("  Loaded SAM v1 (ViT-B)")
        return _SAM2_PREDICTOR
    except ImportError:
        raise ImportError(
            "Neither SAM2 nor SAM v1 is installed.\n"
            "Install one of:\n"
            "  pip install segment-anything-2\n"
            "  pip install segment-anything"
        )


def segment_view(
    rgb: np.ndarray,
    text_prompts: List[str],
    device: str = DEVICE,
    box_threshold: float = BOX_THRESHOLD,
    text_threshold: float = TEXT_THRESHOLD,
) -> List[MaskDetection]:
    """
    Run Grounding DINO + SAM on a single rendered view.

    Args:
        rgb:           (H, W, 3) uint8 image
        text_prompts:  list of object labels to detect
        device:        torch device
        box_threshold: Grounding DINO box confidence
        text_threshold: Grounding DINO text confidence

    Returns:
        List of MaskDetection instances
    """
    from groundingdino.util.inference import predict
    from PIL import Image
    import torchvision.transforms as T

    # Build DINO prompt: "label1 . label2 . label3"
    prompt = " . ".join(text_prompts)

    # Prepare image for Grounding DINO
    image_pil = Image.fromarray(rgb)
    transform = T.Compose([
        T.Resize((800, 1333)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image_pil)

    # Run Grounding DINO
    dino_model = _load_grounding_dino(device)
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_tensor,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    if len(boxes) == 0:
        return []

    # Convert boxes from [cx, cy, w, h] normalized to [x1, y1, x2, y2] pixel
    H, W = rgb.shape[:2]
    boxes_pixel = []
    for box in boxes:
        cx, cy, bw, bh = box.tolist()
        x1 = (cx - bw / 2) * W
        y1 = (cy - bh / 2) * H
        x2 = (cx + bw / 2) * W
        y2 = (cy + bh / 2) * H
        boxes_pixel.append([x1, y1, x2, y2])
    boxes_np = np.array(boxes_pixel, dtype=np.float32)

    # Run SAM with detected boxes as prompts
    sam_predictor = _load_sam2(device)
    sam_predictor.set_image(rgb)

    detections = []
    for i, (box, logit, phrase) in enumerate(zip(boxes_np, logits, phrases)):
        try:
            # SAM expects boxes as (1, 4) tensor
            input_box = torch.tensor(box, dtype=torch.float32, device=device).unsqueeze(0)
            masks, scores, _ = sam_predictor.predict(
                box=input_box.cpu().numpy() if hasattr(sam_predictor, 'predict') else input_box,
                multimask_output=False,
            )
            # Take the best mask
            if len(masks) > 0:
                mask = masks[0] if masks.ndim == 3 else masks
                detections.append(MaskDetection(
                    mask=mask.astype(bool),
                    label=phrase.strip(),
                    confidence=float(logit),
                    bbox=box.tolist(),
                ))
        except Exception as e:
            warnings.warn(f"SAM prediction failed for box {i}: {e}", RuntimeWarning)
            # Fallback: use box as mask
            mask = np.zeros((H, W), dtype=bool)
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            mask[y1:y2, x1:x2] = True
            detections.append(MaskDetection(
                mask=mask,
                label=phrase.strip(),
                confidence=float(logit) * 0.5,  # lower confidence for box-only
                bbox=box.tolist(),
            ))

    return detections


# ══════════════════════════════════════════════════════════════════════════════
# Mask-to-Object Matching (across views)
# ══════════════════════════════════════════════════════════════════════════════

def match_masks_to_objects(
    masks_per_view: List[List[MaskDetection]],
    views: List[RenderedView],
    gt_objects: List[dict],
) -> Dict[int, List[Tuple[np.ndarray, float]]]:
    """
    Match detected masks to 3DSSG object IDs using label matching
    and spatial consistency via Gaussian overlap.

    Strategy:
        1. Group detections by label
        2. For each label, if only one GT object has that label → direct match
        3. If multiple GT objects share a label → use Gaussian centroid overlap
           to disambiguate across views

    Args:
        masks_per_view: list of mask detections per view
        views:          corresponding RenderedView instances
        gt_objects:     3DSSG objects for this scene

    Returns:
        {gt_object_id: [(mask, confidence), ...]} — masks assigned to each GT object
    """
    # Build label → GT object ID mapping
    label_to_gt_ids: Dict[str, List[str]] = {}
    for obj in gt_objects:
        label = obj["label"].lower().strip()
        if label not in label_to_gt_ids:
            label_to_gt_ids[label] = []
        label_to_gt_ids[label].append(obj["id"])

    # Collect all (view_idx, detection_idx, gaussian_set) per detected label
    label_detections: Dict[str, List[Tuple[int, MaskDetection, set]]] = {}

    for view_idx, (dets, view) in enumerate(zip(masks_per_view, views)):
        for det in dets:
            label = det.label.lower().strip()
            # Get set of Gaussian IDs covered by this mask
            masked_gids = view.gaussian_ids[det.mask]
            gid_set = set(masked_gids[masked_gids >= 0].tolist())
            if label not in label_detections:
                label_detections[label] = []
            label_detections[label].append((view_idx, det, gid_set))

    # Assign detections to GT objects
    # result: gt_object_id → list of (view_gaussian_ids_map, mask, confidence)
    assignments: Dict[str, List[Tuple[RenderedView, MaskDetection]]] = {
        obj["id"]: [] for obj in gt_objects
    }

    for label, detections in label_detections.items():
        # Find matching GT objects
        matching_gt_ids = label_to_gt_ids.get(label, [])

        # Also try partial label matching (e.g., "chair" matches "office chair")
        if not matching_gt_ids:
            for gt_label, gt_ids in label_to_gt_ids.items():
                if label in gt_label or gt_label in label:
                    matching_gt_ids.extend(gt_ids)

        if not matching_gt_ids:
            continue

        if len(matching_gt_ids) == 1:
            # Unique label → direct assignment
            gt_id = matching_gt_ids[0]
            for view_idx, det, gid_set in detections:
                assignments[gt_id].append((views[view_idx], det))
        else:
            # Multiple GT objects with same label → cluster by Gaussian overlap
            # Group detections into clusters based on shared Gaussians
            clusters = _cluster_detections_by_overlap(detections)

            # Assign each cluster to the nearest GT object (by centroid)
            # Sort GT objects and clusters spatially for matching
            for cluster_gids, cluster_dets in clusters:
                if not cluster_gids:
                    continue
                # Find which GT object this cluster is closest to
                # (would need GT positions — use ID order as proxy)
                # Simple heuristic: assign in order
                if matching_gt_ids:
                    gt_id = matching_gt_ids[0]
                    matching_gt_ids = matching_gt_ids[1:] + [matching_gt_ids[0]]
                    for view_idx, det, _ in cluster_dets:
                        assignments[gt_id].append((views[view_idx], det))

    return assignments


def _cluster_detections_by_overlap(
    detections: List[Tuple[int, MaskDetection, set]],
    min_overlap: float = 0.3,
) -> List[Tuple[set, List]]:
    """
    Cluster detections by Gaussian ID overlap across views.
    Two detections are in the same cluster if they share >= min_overlap
    fraction of their Gaussian IDs.

    Returns list of (merged_gid_set, list_of_detections) per cluster.
    """
    if not detections:
        return []

    # Union-find clustering
    n = len(detections)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Compare all pairs
    for i in range(n):
        for j in range(i + 1, n):
            gids_i = detections[i][2]
            gids_j = detections[j][2]
            if not gids_i or not gids_j:
                continue
            overlap = len(gids_i & gids_j) / min(len(gids_i), len(gids_j))
            if overlap >= min_overlap:
                union(i, j)

    # Group by cluster
    clusters_map: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in clusters_map:
            clusters_map[root] = []
        clusters_map[root].append(i)

    result = []
    for indices in clusters_map.values():
        merged_gids = set()
        cluster_dets = []
        for idx in indices:
            merged_gids |= detections[idx][2]
            cluster_dets.append(detections[idx])
        result.append((merged_gids, cluster_dets))

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Back-projection: Masks → Per-Gaussian Instance Labels (Vote Accumulation)
# ══════════════════════════════════════════════════════════════════════════════

def backproject_to_gaussians(
    assignments: Dict[str, List[Tuple[RenderedView, MaskDetection]]],
    n_gaussians: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Back-project mask assignments to per-Gaussian instance labels via voting.

    For each pixel in each assigned mask, look up which Gaussian is frontmost
    (from the gaussian_ids map) and accumulate a vote for that object ID.

    Final assignment: each Gaussian gets the object_id with the most votes.
    Confidence: max_votes / total_votes per Gaussian.

    Args:
        assignments: {gt_object_id: [(view, mask_detection), ...]}
        n_gaussians: total number of Gaussians in the scene

    Returns:
        labels:      (N,) int — instance ID per Gaussian (-1 for unassigned)
        confidences: (N,) float — assignment confidence [0, 1]
        object_map:  {int_id: label_string}
    """
    # Build integer ID mapping for GT objects
    gt_ids = sorted(assignments.keys(), key=lambda x: int(x))
    gt_id_to_int = {gt_id: i for i, gt_id in enumerate(gt_ids)}

    # Vote accumulation: (N, num_objects) count matrix
    num_objects = len(gt_ids)
    if num_objects == 0:
        return (
            np.full(n_gaussians, -1, dtype=np.int32),
            np.zeros(n_gaussians, dtype=np.float32),
            {},
        )

    votes = np.zeros((n_gaussians, num_objects), dtype=np.int32)

    for gt_id, view_dets in assignments.items():
        obj_int_id = gt_id_to_int[gt_id]
        for view, det in view_dets:
            # Get Gaussian IDs under this mask
            masked_gids = view.gaussian_ids[det.mask]
            valid_gids = masked_gids[masked_gids >= 0]

            # Weight by detection confidence
            weight = max(1, int(det.confidence * 10))
            for gid in valid_gids:
                if 0 <= gid < n_gaussians:
                    votes[gid, obj_int_id] += weight

    # Assign labels by majority vote
    total_votes = votes.sum(axis=1)  # (N,)
    max_votes = votes.max(axis=1)    # (N,)

    labels = np.full(n_gaussians, -1, dtype=np.int32)
    confidences = np.zeros(n_gaussians, dtype=np.float32)

    has_votes = total_votes > 0
    labels[has_votes] = votes[has_votes].argmax(axis=1)
    confidences[has_votes] = max_votes[has_votes] / total_votes[has_votes]

    # Filter low-confidence assignments
    low_conf = confidences < MIN_VOTE_CONFIDENCE
    labels[low_conf] = -1
    confidences[low_conf] = 0.0

    # Build object map: int_id → label string
    # (We need the GT objects to get labels — pass through assignments keys)
    object_map = {i: gt_id for gt_id, i in gt_id_to_int.items()}

    return labels, confidences, object_map


# ══════════════════════════════════════════════════════════════════════════════
# 3DSSG Annotation Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_3dssg_annotations(
    objects_path: str = OBJECTS_PATH,
    relationships_path: str = RELATIONSHIPS_PATH,
) -> Tuple[Dict[str, List[dict]], Dict[str, List[list]]]:
    """Load 3DSSG objects and relationships indexed by scan_id."""
    with open(objects_path, "r") as f:
        objects_data = json.load(f)
    with open(relationships_path, "r") as f:
        rels_data = json.load(f)

    objects_by_scan = {}
    for scan_entry in objects_data["scans"]:
        scan_id = scan_entry["scan"]
        objects_by_scan[scan_id] = scan_entry["objects"]

    rels_by_scan = {}
    for scan_entry in rels_data["scans"]:
        scan_id = scan_entry["scan"]
        rels_by_scan[scan_id] = scan_entry["relationships"]

    return objects_by_scan, rels_by_scan


# ══════════════════════════════════════════════════════════════════════════════
# Main Processing Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def process_scene(
    scene_id: str,
    gt_objects: List[dict],
    splats_dir: str = SPLATS_DIR,
    n_views: int = N_VIEWS,
    device: str = DEVICE,
    use_gsplat: bool = True,
) -> Optional[Dict]:
    """
    Process a single 3RScan scene: render → segment → back-project.

    Args:
        scene_id:    3RScan scan UUID
        gt_objects:  3DSSG objects for this scene
        splats_dir:  root directory containing splat files
        n_views:     number of hemisphere viewpoints
        device:      torch device
        use_gsplat:  whether to use gsplat (True) or software renderer (False)

    Returns:
        Dict with "labels", "confidences", "object_map", "stats"
        or None if processing fails.
    """
    ply_path = os.path.join(splats_dir, scene_id, "ckpts", "point_cloud_30000.ply")
    if not os.path.exists(ply_path):
        return None

    # ── 1. Load Gaussian splat ────────────────────────────────────────────────
    try:
        cloud = load_gaussian_ply(ply_path)
    except Exception as e:
        warnings.warn(f"Failed to load {scene_id}: {e}", RuntimeWarning)
        return None

    if cloud.num_gaussians < 100:
        return None

    # Filter low-opacity Gaussians
    cloud_filtered = filter_gaussians(cloud, opacity_threshold=0.05)
    if cloud_filtered.num_gaussians < 50:
        cloud_filtered = cloud  # use unfiltered if too aggressive

    # ── 2. Compute scene bounds and camera radius ─────────────────────────────
    scene_center = cloud_filtered.xyz.mean(axis=0)
    scene_extent = cloud_filtered.xyz.max(axis=0) - cloud_filtered.xyz.min(axis=0)
    radius = float(np.linalg.norm(scene_extent)) * 1.2  # 1.2x bounding sphere

    # ── 3. Generate camera poses ──────────────────────────────────────────────
    cameras = generate_hemisphere_cameras(scene_center, radius, n_views)

    # ── 4. Render from all viewpoints ─────────────────────────────────────────
    views: List[RenderedView] = []
    render_fn = render_gaussians if use_gsplat else render_gaussians_software

    for c2w, K in cameras:
        view = render_fn(cloud_filtered, c2w, K)
        # Fallback to software renderer if gsplat fails for this view
        if view is None and use_gsplat:
            view = render_gaussians_software(cloud_filtered, c2w, K)
        if view is not None:
            views.append(view)

    if len(views) < 3:
        warnings.warn(f"Scene {scene_id}: only {len(views)} valid views", RuntimeWarning)
        return None

    # ── 5. Get text prompts from GT object labels ─────────────────────────────
    labels_set = list(set(obj["label"].lower().strip() for obj in gt_objects))
    # Remove generic labels that confuse detection
    skip_labels = {"object", "otherprop", "otherfurniture", "otherstructure", "misc"}
    text_prompts = [l for l in labels_set if l not in skip_labels]
    if not text_prompts:
        text_prompts = labels_set  # use all if filtering removed everything

    # ── 6. Segment each view ──────────────────────────────────────────────────
    masks_per_view: List[List[MaskDetection]] = []
    for view in views:
        # Skip views with mostly empty content
        # Use a lower threshold for software renderer (sparse point projection)
        coverage = (view.rgb.sum(axis=2) > 0).mean()
        if coverage < 0.01:  # at least 1% pixel coverage
            masks_per_view.append([])
            continue
        try:
            masks = segment_view(view.rgb, text_prompts, device=device)
            masks_per_view.append(masks)
        except Exception as e:
            warnings.warn(f"Segmentation failed for a view: {e}", RuntimeWarning)
            masks_per_view.append([])

    # Check we got some detections
    total_detections = sum(len(m) for m in masks_per_view)
    if total_detections == 0:
        warnings.warn(f"Scene {scene_id}: no detections in any view", RuntimeWarning)
        return None

    # ── 7. Match masks to GT objects ──────────────────────────────────────────
    assignments = match_masks_to_objects(masks_per_view, views, gt_objects)

    # ── 8. Back-project to Gaussians ──────────────────────────────────────────
    labels, confidences, object_map = backproject_to_gaussians(
        assignments, cloud_filtered.num_gaussians
    )

    # ── 9. Compute stats ──────────────────────────────────────────────────────
    n_assigned = int((labels >= 0).sum())
    n_total = cloud_filtered.num_gaussians
    n_unique = len(set(labels[labels >= 0].tolist()))
    n_gt = len(gt_objects)

    stats = {
        "n_gaussians": n_total,
        "n_assigned": n_assigned,
        "pct_assigned": round(100 * n_assigned / max(n_total, 1), 1),
        "n_instances": n_unique,
        "n_gt_objects": n_gt,
        "n_views_rendered": len(views),
        "n_total_detections": total_detections,
        "mean_confidence": float(confidences[labels >= 0].mean()) if n_assigned > 0 else 0,
    }

    return {
        "labels": labels,
        "confidences": confidences,
        "object_map": object_map,
        "stats": stats,
    }


def save_instance_labels(scene_id: str, result: Dict, splats_dir: str = SPLATS_DIR):
    """Save per-Gaussian instance labels to .npz file."""
    out_path = os.path.join(splats_dir, scene_id, "instance_labels.npz")
    np.savez_compressed(
        out_path,
        labels=result["labels"],
        confidences=result["confidences"],
        object_map=json.dumps(result["object_map"]),
        stats=json.dumps(result["stats"]),
    )
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Batch Processing
# ══════════════════════════════════════════════════════════════════════════════

def find_available_scenes(
    splats_dir: str,
    objects_by_scan: Dict,
) -> List[str]:
    """Find scene IDs that have both splat files and 3DSSG annotations."""
    available = []
    if not os.path.isdir(splats_dir):
        print(f"ERROR: Splats directory not found: {splats_dir}")
        return []

    for scene_id in os.listdir(splats_dir):
        ply_path = os.path.join(splats_dir, scene_id, "ckpts", "point_cloud_30000.ply")
        if not os.path.exists(ply_path):
            continue
        if scene_id not in objects_by_scan:
            continue
        if len(objects_by_scan[scene_id]) < 2:
            continue
        available.append(scene_id)

    return sorted(available)


def process_all_scenes(
    splats_dir: str = SPLATS_DIR,
    max_scenes: Optional[int] = None,
    n_views: int = N_VIEWS,
    device: str = DEVICE,
    use_gsplat: bool = True,
    skip_existing: bool = True,
):
    """
    Process all available 3RScan scenes.

    Args:
        splats_dir:    root directory with splat files
        max_scenes:    limit number of scenes (for testing)
        n_views:       viewpoints per scene
        device:        torch device
        use_gsplat:    use gsplat renderer (vs software fallback)
        skip_existing: skip scenes that already have instance_labels.npz
    """
    print("=" * 70)
    print("  3RScan Gaussian Splat Segmentation via Grounded-SAM2")
    print("=" * 70)
    print(f"  Splats dir:  {splats_dir}")
    print(f"  Views/scene: {n_views}")
    print(f"  Device:      {device}")
    print(f"  Renderer:    {'gsplat' if use_gsplat else 'software'}")
    print()

    # Load annotations
    print("Loading 3DSSG annotations...")
    objects_by_scan, rels_by_scan = load_3dssg_annotations()
    print(f"  {len(objects_by_scan)} scenes in objects.json")

    # Find available scenes
    scenes = find_available_scenes(splats_dir, objects_by_scan)
    print(f"  {len(scenes)} scenes with splats + annotations")

    if skip_existing:
        scenes_to_process = []
        for s in scenes:
            out_path = os.path.join(splats_dir, s, "instance_labels.npz")
            if not os.path.exists(out_path):
                scenes_to_process.append(s)
        n_skipped = len(scenes) - len(scenes_to_process)
        if n_skipped > 0:
            print(f"  Skipping {n_skipped} already-processed scenes")
        scenes = scenes_to_process

    if max_scenes:
        scenes = scenes[:max_scenes]

    print(f"\n  Processing {len(scenes)} scenes...\n")

    # Track results
    n_success = 0
    n_failed = 0
    failed_scenes = []
    all_stats = []

    try:
        from tqdm import tqdm
        iterator = tqdm(scenes, desc="Segmenting", unit="scene")
    except ImportError:
        iterator = scenes

    for scene_id in iterator:
        gt_objects = objects_by_scan[scene_id]

        try:
            result = process_scene(
                scene_id=scene_id,
                gt_objects=gt_objects,
                splats_dir=splats_dir,
                n_views=n_views,
                device=device,
                use_gsplat=use_gsplat,
            )
        except Exception as e:
            warnings.warn(f"Scene {scene_id} crashed: {e}", RuntimeWarning)
            result = None

        if result is None:
            n_failed += 1
            failed_scenes.append(scene_id)
            continue

        # Save
        out_path = save_instance_labels(scene_id, result, splats_dir)
        n_success += 1
        all_stats.append(result["stats"])

        # Progress info
        if not hasattr(iterator, 'set_postfix'):
            if n_success % 10 == 0:
                print(f"  [{n_success + n_failed}/{len(scenes)}] "
                      f"success={n_success} failed={n_failed}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SEGMENTATION COMPLETE")
    print("=" * 70)
    print(f"  Success: {n_success}/{len(scenes)}")
    print(f"  Failed:  {n_failed}/{len(scenes)}")

    if all_stats:
        avg_pct = np.mean([s["pct_assigned"] for s in all_stats])
        avg_conf = np.mean([s["mean_confidence"] for s in all_stats])
        avg_inst = np.mean([s["n_instances"] for s in all_stats])
        print(f"\n  Average stats across successful scenes:")
        print(f"    Gaussians assigned: {avg_pct:.1f}%")
        print(f"    Mean confidence:    {avg_conf:.3f}")
        print(f"    Instances found:    {avg_inst:.1f}")

    if failed_scenes:
        # Save failed scene list for debugging
        failed_path = os.path.join(splats_dir, "segmentation_failed.json")
        with open(failed_path, "w") as f:
            json.dump(failed_scenes, f, indent=2)
        print(f"\n  Failed scenes saved to: {failed_path}")

    return n_success, n_failed


# ══════════════════════════════════════════════════════════════════════════════
# Verification
# ══════════════════════════════════════════════════════════════════════════════

def verify_segmentation(
    splats_dir: str = SPLATS_DIR,
    n_samples: int = 10,
):
    """
    Verify segmentation quality on a random sample of processed scenes.

    Checks:
        - What % of Gaussians got assigned an instance label
        - Does the number of unique instance IDs match 3DSSG object count
        - Mean confidence of assignments
        - Flags scenes with potential issues
    """
    print("\n" + "=" * 70)
    print("  SEGMENTATION VERIFICATION")
    print("=" * 70)

    # Find processed scenes
    processed = []
    if not os.path.isdir(splats_dir):
        print(f"  ERROR: {splats_dir} not found")
        return

    for scene_id in os.listdir(splats_dir):
        npz_path = os.path.join(splats_dir, scene_id, "instance_labels.npz")
        if os.path.exists(npz_path):
            processed.append(scene_id)

    print(f"  Found {len(processed)} processed scenes")
    if not processed:
        print("  No scenes to verify. Run segmentation first.")
        return

    # Load annotations for comparison
    objects_by_scan, _ = load_3dssg_annotations()

    # Sample scenes
    rng = np.random.default_rng(42)
    sample = rng.choice(processed, size=min(n_samples, len(processed)), replace=False)

    print(f"\n  Verifying {len(sample)} scenes:\n")
    print(f"  {'Scene ID':<40} {'Assigned%':>9} {'Instances':>9} "
          f"{'GT Objs':>8} {'Conf':>6} {'Status':>8}")
    print(f"  {'-'*40} {'-'*9} {'-'*9} {'-'*8} {'-'*6} {'-'*8}")

    issues = []
    for scene_id in sample:
        npz_path = os.path.join(splats_dir, scene_id, "instance_labels.npz")
        data = np.load(npz_path, allow_pickle=True)

        labels = data["labels"]
        confidences = data["confidences"]
        stats = json.loads(str(data["stats"]))

        n_total = len(labels)
        n_assigned = int((labels >= 0).sum())
        pct_assigned = 100 * n_assigned / max(n_total, 1)
        n_instances = len(set(labels[labels >= 0].tolist()))
        n_gt = len(objects_by_scan.get(scene_id, []))
        mean_conf = float(confidences[labels >= 0].mean()) if n_assigned > 0 else 0

        # Quality checks
        status = "OK"
        if pct_assigned < 30:
            status = "LOW_COV"
            issues.append((scene_id, "Low coverage (<30%)"))
        elif abs(n_instances - n_gt) > n_gt * 0.5:
            status = "MISMATCH"
            issues.append((scene_id, f"Instance count mismatch: {n_instances} vs {n_gt} GT"))
        elif mean_conf < 0.3:
            status = "LOW_CONF"
            issues.append((scene_id, "Low mean confidence (<0.3)"))

        print(f"  {scene_id:<40} {pct_assigned:>8.1f}% {n_instances:>9} "
              f"{n_gt:>8} {mean_conf:>6.3f} {status:>8}")

    # Summary
    print(f"\n  Issues found: {len(issues)}/{len(sample)}")
    for scene_id, issue in issues:
        print(f"    {scene_id}: {issue}")

    # Overall stats
    print(f"\n  Recommendation:")
    clean_count = len(processed) - len(issues) * (len(processed) / len(sample))
    print(f"    Estimated clean scenes: ~{int(clean_count)} / {len(processed)}")
    if clean_count >= 400:
        print(f"    [OK] Sufficient for training (target: >=400)")
    else:
        print(f"    [WARN] May need to relax quality thresholds or re-process failed scenes")


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Segment 3RScan Gaussian splats using Grounded-SAM2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all scenes (skip already-done)
    python scripts/segment_3rscan.py

    # Process first 10 scenes (for testing)
    python scripts/segment_3rscan.py --max-scenes 10

    # Process a specific scene
    python scripts/segment_3rscan.py --scene-id 0988ea72-eb32-2e61-8344-99e2283c2728

    # Use fewer views (faster but lower quality)
    python scripts/segment_3rscan.py --n-views 12

    # Use software renderer (no gsplat needed)
    python scripts/segment_3rscan.py --software-render

    # Verify quality of processed scenes
    python scripts/segment_3rscan.py --verify-only

    # Re-process all (ignore existing results)
    python scripts/segment_3rscan.py --force
        """,
    )
    parser.add_argument("--splats-dir", default=SPLATS_DIR,
                        help=f"Root directory with splat files (default: {SPLATS_DIR})")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Limit number of scenes to process")
    parser.add_argument("--scene-id", default=None,
                        help="Process a single specific scene")
    parser.add_argument("--n-views", type=int, default=N_VIEWS,
                        help=f"Number of hemisphere viewpoints (default: {N_VIEWS})")
    parser.add_argument("--device", default=DEVICE,
                        choices=["cuda", "cpu"],
                        help=f"Torch device (default: {DEVICE})")
    parser.add_argument("--software-render", action="store_true",
                        help="Use software renderer instead of gsplat")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if instance_labels.npz exists")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only run verification on existing results")
    parser.add_argument("--verify-samples", type=int, default=10,
                        help="Number of scenes to verify (default: 10)")

    args = parser.parse_args()

    if args.verify_only:
        verify_segmentation(args.splats_dir, n_samples=args.verify_samples)
        return

    if args.scene_id:
        # Process single scene
        print(f"Processing single scene: {args.scene_id}")
        objects_by_scan, _ = load_3dssg_annotations()
        if args.scene_id not in objects_by_scan:
            print(f"ERROR: Scene {args.scene_id} not found in 3DSSG annotations")
            sys.exit(1)

        gt_objects = objects_by_scan[args.scene_id]
        result = process_scene(
            scene_id=args.scene_id,
            gt_objects=gt_objects,
            splats_dir=args.splats_dir,
            n_views=args.n_views,
            device=args.device,
            use_gsplat=not args.software_render,
        )

        if result is None:
            print("FAILED: Scene processing returned None")
            sys.exit(1)

        out_path = save_instance_labels(args.scene_id, result, args.splats_dir)
        print(f"\nSaved to: {out_path}")
        print(f"Stats: {json.dumps(result['stats'], indent=2)}")
    else:
        # Batch processing
        process_all_scenes(
            splats_dir=args.splats_dir,
            max_scenes=args.max_scenes,
            n_views=args.n_views,
            device=args.device,
            use_gsplat=not args.software_render,
            skip_existing=not args.force,
        )

        # Run verification after batch processing
        print("\nRunning verification...")
        verify_segmentation(args.splats_dir, n_samples=args.verify_samples)


if __name__ == "__main__":
    main()
