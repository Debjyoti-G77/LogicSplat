"""
ScanNet Geometry Extractor — TASK 1.

Reads ScanNet scene files to extract per-object 3D bounding boxes,
semantic labels, and geometric features for GNN training.

ScanNet file structure per scene (scene<id>_<ver>/):
    scene<id>_<ver>_vh_clean_2.ply              ← mesh vertices (XYZ)
    scene<id>_<ver>_vh_clean_2.labels.ply       ← per-vertex NYU40 label IDs
    scene<id>_<ver>.aggregation.json            ← object groupings (segGroups)
    scene<id>_<ver>_vh_clean_2.0.010000.segs.json ← per-vertex segment IDs

Node features (10 dims) — matches loader_3rscan.NODE_FEATURE_DIM_GEO:
    [0]  centroid_x (normalized by scene extent)
    [1]  centroid_y
    [2]  centroid_z
    [3]  size_x (bbox width, normalized)
    [4]  size_y (bbox depth, normalized)
    [5]  size_z (bbox height, normalized)
    [6]  volume (normalized)
    [7]  height_ratio (size_z / max(size_x, size_y))
    [8]  point_density (pts / volume, normalized)
    [9]  z_relative (centroid_z relative to scene floor)

Edge features (8 dims) — matches loader_3rscan.EDGE_FEATURE_DIM_GEO:
    [0]  delta_z (centroid Z difference, normalized)
    [1]  xy_distance (normalized)
    [2]  dist_3d (normalized)
    [3]  bbox_overlap_xy (fraction of A's footprint overlapping B)
    [4]  volume_ratio (min/max)
    [5]  height_ratio (a.size_z / b.size_z)
    [6]  vertical_gap (a.bottom - b.top, normalized)
    [7]  size_ratio_xy (a XY diagonal / b XY diagonal)
"""
import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset
import torch

from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, DSSG_TO_SCHEMA
from src.relations.geometry import derive_relations

# ── constants ─────────────────────────────────────────────────────────────────

NODE_FEATURE_DIM = 10
EDGE_FEATURE_DIM = 17

# NYU40 label ID → human-readable name (subset used in ScanNet)
NYU40_NAMES = {
    1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
    6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
    11: "picture", 12: "counter", 13: "blinds", 14: "desk", 15: "shelves",
    16: "curtain", 17: "dresser", 18: "pillow", 19: "mirror", 20: "floor_mat",
    21: "clothes", 22: "ceiling", 23: "books", 24: "fridge", 25: "tv",
    26: "paper", 27: "towel", 28: "shower_curtain", 29: "box", 30: "whiteboard",
    31: "person", 32: "nightstand", 33: "toilet", 34: "sink", 35: "lamp",
    36: "bathtub", 37: "bag", 38: "otherstructure", 39: "otherfurniture",
    40: "otherprop",
}

# Labels to skip — structural elements that aren't objects
SKIP_LABELS = {"wall", "floor", "ceiling"}


# ── PLY reader ────────────────────────────────────────────────────────────────
# Try open3d first (C++ backend, ~10× faster than plyfile).
# Fall back to plyfile if open3d is not installed.

def _read_ply_vertices(ply_path: str) -> np.ndarray:
    """
    Read XYZ vertex positions from a PLY file.
    Returns (N, 3) float32 array.
    Uses open3d when available (~10× faster than plyfile).
    """
    try:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(ply_path)
        return np.asarray(mesh.vertices, dtype=np.float32)
    except Exception:
        pass
    # fallback: pure-Python plyfile
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)


def _read_ply_label_ids(ply_path: str) -> np.ndarray:
    """
    Read per-vertex NYU40 label IDs from a labels PLY file.
    Returns (N,) int32 array.
    Uses open3d point-cloud reader (labels PLY has no faces).
    """
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply_path)
        # open3d doesn't expose custom scalar fields directly;
        # fall through to plyfile for the label field
    except Exception:
        pass
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    # ScanNet uses 'label' field for NYU40 IDs
    for field in ("label", "scalar_label", "nyu40id"):
        if field in v.data.dtype.names:
            return np.array(v[field], dtype=np.int32)
    raise ValueError(f"No label field found in {ply_path}. "
                     f"Available: {v.data.dtype.names}")


# ── scene file discovery ──────────────────────────────────────────────────────

def _scene_files(scene_dir: str, scene_id: str) -> Dict[str, str]:
    """
    Build paths for all four ScanNet files for a given scene.
    Raises FileNotFoundError if any required file is missing.
    """
    paths = {
        "mesh":  os.path.join(scene_dir, f"{scene_id}_vh_clean_2.ply"),
        "labels": os.path.join(scene_dir, f"{scene_id}_vh_clean_2.labels.ply"),
        "agg":   os.path.join(scene_dir, f"{scene_id}.aggregation.json"),
        "segs":  os.path.join(scene_dir, f"{scene_id}_vh_clean_2.0.010000.segs.json"),
    }
    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing ScanNet file [{key}]: {path}")
    return paths


# ── geometric feature extraction ──────────────────────────────────────────────

def _node_features(
    obj_points: np.ndarray,
    scene_min: np.ndarray,
    scene_extent: np.ndarray,
) -> np.ndarray:
    """
    Compute 10-dim geometric node features for one object.
    Identical feature layout to loader_3rscan.extract_geometric_node_features.
    """
    if len(obj_points) == 0:
        return np.zeros(NODE_FEATURE_DIM, dtype=np.float32)

    norm = np.maximum(scene_extent, 1e-6)
    centroid = obj_points.mean(axis=0)
    bbox_min = obj_points.min(axis=0)
    bbox_max = obj_points.max(axis=0)
    size = np.maximum(bbox_max - bbox_min, 1e-6)

    centroid_norm = (centroid - scene_min) / norm
    volume = float(size[0] * size[1] * size[2])
    scene_vol = float(norm[0] * norm[1] * norm[2])
    volume_norm = min(volume / max(scene_vol, 1e-9), 1.0)
    height_ratio = float(size[2] / max(size[0], size[1]))
    density = min(len(obj_points) / max(volume, 1e-6), 1000.0) / 1000.0
    z_relative = float((centroid[2] - scene_min[2]) / norm[2])

    return np.array([
        centroid_norm[0], centroid_norm[1], centroid_norm[2],
        size[0] / norm[0], size[1] / norm[1], size[2] / norm[2],
        volume_norm, height_ratio, density, z_relative,
    ], dtype=np.float32)


def _edge_features(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    scene_extent: np.ndarray,
) -> np.ndarray:
    """
    Compute 17-dim geometric edge features between two objects.

    Features [0-1] are the KEY fix for directional relations:
      [0]  delta_x  — signed X displacement (A centroid - B centroid), normalized
      [1]  delta_y  — signed Y displacement, normalized
    These give the model the actual direction vector, not just distance.
    Without them, left_of/right_of/in_front_of/behind all look identical
    (same xy_dist, same dist_3d, same near-zero delta_z).

      [2]  delta_z  — signed Z displacement, normalized
      [3]  xy_distance (unsigned, normalized)
      [4]  dist_3d (unsigned, normalized)
      [5]  bbox_overlap_xy (fraction of A's footprint overlapping B)
      [6]  volume_ratio (min/max, 0-1)
      [7]  height_ratio (log-normalized)
      [8]  vertical_gap (a.bottom - b.top, normalized by scene extent, clamped)
      [9]  size_ratio_xy (log-normalized)

    Contact-specific features:
      [10] vert_gap_obj_norm — vertical gap normalized by mean object height
      [11] vert_gap_abs — absolute vertical gap in meters, clipped [-0.5, 0.5]
      [12] contact_score — exp(-|gap|/threshold) * bbox_overlap * a_above_b
      [13] support_overlap_b — XY intersection area / B's footprint area

    Rule-margin features:
      [14] ontop_z_margin — how far above z_min_threshold is the Z diff
      [15] ontop_xy_margin — how far inside B's footprint is A's centroid
      [16] z_dominance_margin — does Z diff dominate XY distance
    """
    if len(pts_a) == 0 or len(pts_b) == 0:
        return np.zeros(EDGE_FEATURE_DIM, dtype=np.float32)

    norm = np.maximum(scene_extent, 1e-6)
    c_a = pts_a.mean(axis=0)
    c_b = pts_b.mean(axis=0)
    min_a, max_a = pts_a.min(axis=0), pts_a.max(axis=0)
    min_b, max_b = pts_b.min(axis=0), pts_b.max(axis=0)
    size_a = np.maximum(max_a - min_a, 1e-6)
    size_b = np.maximum(max_b - min_b, 1e-6)

    # signed displacement — critical for left/right/front/behind
    delta_x = float(np.clip((c_a[0] - c_b[0]) / norm[0], -1.0, 1.0))
    delta_y = float(np.clip((c_a[1] - c_b[1]) / norm[1], -1.0, 1.0))
    delta_z = float(np.clip((c_a[2] - c_b[2]) / norm[2], -1.0, 1.0))

    xy_dist = float(np.linalg.norm(c_a[:2] - c_b[:2]) / np.linalg.norm(norm[:2]))
    dist_3d = float(np.linalg.norm(c_a - c_b) / np.linalg.norm(norm))

    # XY bbox overlap fraction (relative to A's footprint)
    ix = max(0.0, min(max_a[0], max_b[0]) - max(min_a[0], min_b[0]))
    iy = max(0.0, min(max_a[1], max_b[1]) - max(min_a[1], min_b[1]))
    area_a = max((max_a[0] - min_a[0]) * (max_a[1] - min_a[1]), 1e-9)
    bbox_overlap = (ix * iy) / area_a

    vol_a = float(np.prod(size_a))
    vol_b = float(np.prod(size_b))
    vol_ratio = min(vol_a, vol_b) / max(vol_a, vol_b)  # already 0-1

    h_ratio = float(np.clip(np.log1p(size_a[2] / max(size_b[2], 1e-6)), -3.0, 3.0) / 3.0)
    vert_gap = float(np.clip((min_a[2] - max_b[2]) / norm[2], -1.0, 1.0))
    size_ratio_xy = float(np.clip(
        np.log1p(np.linalg.norm(size_a[:2]) / max(np.linalg.norm(size_b[:2]), 1e-6)),
        -3.0, 3.0,
    ) / 3.0)

    # ── Contact-specific features [10-13] ──
    # [10] vert_gap_obj_norm: vertical gap normalized by mean object height
    mean_height = max((size_a[2] + size_b[2]) / 2.0, 1e-6)
    vert_gap_obj_norm = float(np.clip((min_a[2] - max_b[2]) / mean_height, -3.0, 3.0) / 3.0)

    # [11] vert_gap_abs: absolute vertical gap in meters, clipped
    vert_gap_abs = float(np.clip(min_a[2] - max_b[2], -0.5, 0.5))

    # [12] contact_score: high when small gap AND high XY overlap AND A above B
    a_above_b = float(c_a[2] > c_b[2])
    raw_gap = float(min_a[2] - max_b[2])
    contact_score = float(
        np.exp(-abs(raw_gap) / max(mean_height * 0.3, 0.01)) * bbox_overlap * a_above_b
    )

    # [13] support_overlap_b: XY intersection area / B's footprint area
    area_b = max((max_b[0] - min_b[0]) * (max_b[1] - min_b[1]), 1e-9)
    support_overlap_b = float((ix * iy) / area_b)

    # ── Rule-margin features [14-16] ──
    # These encode how close each pair is to the on_top_of rule boundary,
    # helping the GNN learn the decision surface directly.

    # [14] ontop_z_margin: how far above z_min_threshold is the Z diff?
    #   Positive = above threshold (rule would fire), negative = below
    z_min_thresh = max(norm[2] * 0.02, 0.01)
    centroid_z_diff = float(c_a[2] - c_b[2])
    ontop_z_margin = float(np.clip(
        (centroid_z_diff - z_min_thresh) / max(z_min_thresh, 0.01), -3.0, 3.0
    ) / 3.0)

    # [15] ontop_xy_margin: how far inside B's footprint is A's centroid?
    #   Positive = inside footprint (rule would fire), negative = outside
    b_xy_size = float(np.linalg.norm(size_b[:2]))
    xy_limit = max(b_xy_size * 0.8, 0.1)
    centroid_xy_dist = float(np.linalg.norm(c_a[:2] - c_b[:2]))
    ontop_xy_margin = float(np.clip(
        (xy_limit - centroid_xy_dist) / max(xy_limit, 0.01), -3.0, 3.0
    ) / 3.0)

    # [16] z_dominance_margin: does Z diff dominate XY distance?
    #   on_top_of requires centroid_z_diff > centroid_xy_dist * z_dominance_factor
    ontop_z_dom = 0.5
    z_dom_margin = float(np.clip(
        (centroid_z_diff - centroid_xy_dist * ontop_z_dom) / max(abs(centroid_z_diff) + 0.01, 0.01),
        -3.0, 3.0
    ) / 3.0)

    return np.array([
        delta_x, delta_y, delta_z, xy_dist, dist_3d,
        bbox_overlap, vol_ratio, h_ratio, vert_gap, size_ratio_xy,
        vert_gap_obj_norm, vert_gap_abs, contact_score, support_overlap_b,
        ontop_z_margin, ontop_xy_margin, z_dom_margin,
    ], dtype=np.float32)


def _read_axis_alignment(scene_dir: str, scene_id: str) -> np.ndarray:
    """
    Read the axisAlignment 4×4 matrix from the ScanNet per-scene .txt file.
    Returns a (4,4) float32 rotation matrix, or identity if file not found.

    The axisAlignment matrix rotates the raw scan coordinate system to a
    canonical gravity-aligned orientation where:
      - Z axis = up (gravity direction)
      - X/Y axes = consistent horizontal orientation across scenes

    Without this, delta_y in edge features is arbitrary per scan, making
    in_front_of/behind unlearnable (same geometry, opposite labels).
    """
    txt_path = os.path.join(scene_dir, f"{scene_id}.txt")
    if not os.path.exists(txt_path):
        return np.eye(4, dtype=np.float32)
    try:
        with open(txt_path) as f:
            for line in f:
                if line.startswith("axisAlignment"):
                    vals = list(map(float, line.split("=")[1].strip().split()))
                    return np.array(vals, dtype=np.float32).reshape(4, 4)
    except Exception:
        pass
    return np.eye(4, dtype=np.float32)


def _apply_axis_alignment(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply a 4×4 axisAlignment matrix to (N, 3) point array.
    Uses homogeneous coordinates: [x, y, z, 1] @ M.T
    """
    if np.allclose(matrix, np.eye(4)):
        return points  # identity — skip transform
    ones = np.ones((len(points), 1), dtype=np.float32)
    pts_h = np.concatenate([points, ones], axis=1)  # (N, 4)
    pts_aligned = (matrix @ pts_h.T).T             # (N, 4)
    return pts_aligned[:, :3].astype(np.float32)


# ── scene loader ──────────────────────────────────────────────────────────────

def load_scannet_scene(
    scene_id: str,
    scannet_dir: str = "data/scannet/scans",
) -> Optional[Dict]:
    """
    Load one ScanNet scene and extract per-object geometry.

    Returns dict with keys:
        objects:       {obj_id_str: {'points', 'label', 'node_feat'}}
        scene_extent:  (3,) scene bounding box size
        all_points:    (N, 3) all mesh vertices
        scene_id:      str

    Returns None if any required file is missing or parsing fails.
    """
    scene_dir = os.path.join(scannet_dir, scene_id)
    try:
        paths = _scene_files(scene_dir, scene_id)
    except FileNotFoundError as e:
        print(f"  [skip] {e}")
        return None

    # ── load mesh vertices ────────────────────────────────────────────────────
    try:
        all_points = _read_ply_vertices(paths["mesh"])
    except Exception as e:
        print(f"  [skip] {scene_id}: failed to read mesh: {e}")
        return None

    # ── apply axisAlignment to canonical orientation ──────────────────────────
    # Rotates the scan so Y-axis is consistent across all scenes, making
    # in_front_of/behind learnable. Falls back to identity if .txt not found.
    axis_matrix = _read_axis_alignment(scene_dir, scene_id)
    all_points = _apply_axis_alignment(all_points, axis_matrix)

    scene_min = all_points.min(axis=0)
    scene_max = all_points.max(axis=0)
    scene_extent = np.maximum(scene_max - scene_min, 1e-6)

    # ── load per-vertex segment IDs ───────────────────────────────────────────
    try:
        with open(paths["segs"]) as f:
            segs_data = json.load(f)
        seg_indices = np.array(segs_data["segIndices"], dtype=np.int32)
    except Exception as e:
        print(f"  [skip] {scene_id}: failed to read segs: {e}")
        return None

    if len(seg_indices) != len(all_points):
        print(f"  [skip] {scene_id}: seg_indices length mismatch "
              f"({len(seg_indices)} vs {len(all_points)} vertices)")
        return None

    # ── load per-vertex NYU40 label IDs ───────────────────────────────────────
    try:
        label_ids = _read_ply_label_ids(paths["labels"])
    except Exception as e:
        print(f"  [skip] {scene_id}: failed to read labels: {e}")
        return None

    if len(label_ids) != len(all_points):
        print(f"  [skip] {scene_id}: label_ids length mismatch "
              f"({len(label_ids)} vs {len(all_points)} vertices)")
        return None

    # ── load object groupings ─────────────────────────────────────────────────
    try:
        with open(paths["agg"]) as f:
            agg_data = json.load(f)
        seg_groups = agg_data["segGroups"]
    except Exception as e:
        print(f"  [skip] {scene_id}: failed to read aggregation: {e}")
        return None

    # ── build segment → vertex mask lookup ───────────────────────────────────
    # seg_indices[i] = segment ID for vertex i
    # We need: segment_id → array of vertex indices
    # Keys are plain Python int so they match the int IDs from aggregation JSON
    unique_segs = np.unique(seg_indices)
    seg_to_verts: Dict[int, np.ndarray] = {
        int(seg_id): np.where(seg_indices == seg_id)[0]
        for seg_id in unique_segs
    }

    # ── extract per-object geometry ───────────────────────────────────────────
    objects: Dict[str, Dict] = {}

    for group in seg_groups:
        obj_id = str(group["objectId"])
        raw_label = group.get("label", "unknown").lower().strip()

        # skip structural elements
        if raw_label in SKIP_LABELS:
            continue

        # collect all vertex indices for this object's segments
        vert_indices = []
        for seg_id in group.get("segments", []):
            key = int(seg_id)  # ensure plain int — JSON may give int or float
            if key in seg_to_verts:
                vert_indices.append(seg_to_verts[key])

        if not vert_indices:
            continue

        vert_idx = np.concatenate(vert_indices)
        obj_points = all_points[vert_idx]

        if len(obj_points) < 3:
            continue

        # derive NYU40 label from majority vote over vertex label IDs
        obj_label_ids = label_ids[vert_idx]
        unique, counts = np.unique(obj_label_ids, return_counts=True)
        majority_nyu40 = int(unique[counts.argmax()])
        nyu40_name = NYU40_NAMES.get(majority_nyu40, raw_label)

        node_feat = _node_features(obj_points, scene_min, scene_extent)

        objects[obj_id] = {
            "points":    obj_points,
            "label":     nyu40_name,
            "raw_label": raw_label,
            "node_feat": node_feat,
        }

    if not objects:
        print(f"  [skip] {scene_id}: no valid objects after filtering")
        return None

    return {
        "scene_id":     scene_id,
        "objects":      objects,
        "scene_extent": scene_extent,
        "all_points":   all_points,
    }


# ── scene graph builder ───────────────────────────────────────────────────────

def build_scannet_scene_graph_geometric(scene_data: Dict) -> Optional[Dict]:
    """
    Build a PyG-compatible scene graph from ScanNet geometry using
    derive_relations() from geometry.py to auto-label every object pair.

    Multi-label mode: every edge carries a multi-hot vector of shape
    (NUM_RELATIONS,) where each entry is 1.0 if that relation holds between
    the pair, 0.0 otherwise.  This matches the GT annotation convention where
    multiple relations can hold simultaneously (e.g. higher_than AND on_top_of).

    No human annotation needed — relations are derived purely from 3D bboxes.

    Returns dict with keys:
        x:           (N, 10) node features
        edge_index:  (2, E) directed edges (all pairs that have ≥1 relation)
        edge_attr:   (E, 10) geometric edge features
        edge_label:  (E, NUM_RELATIONS) multi-hot float tensor
        obj_labels:  list of semantic label strings
        scene_id:    str
    Returns None if fewer than 2 objects.
    """
    objects = scene_data["objects"]
    scene_extent = scene_data["scene_extent"]

    obj_ids = sorted(objects.keys(), key=lambda x: int(x))
    if len(obj_ids) < 2:
        return None

    # node features
    x = np.stack([objects[oid]["node_feat"] for oid in obj_ids])
    obj_labels = [objects[oid]["label"] for oid in obj_ids]

    src_list, dst_list, edge_feat_list = [], [], []
    label_list: List[np.ndarray] = []

    for i, oid_a in enumerate(obj_ids):
        pts_a = objects[oid_a]["points"]
        min_a = pts_a.min(axis=0)
        max_a = pts_a.max(axis=0)

        for j, oid_b in enumerate(obj_ids):
            if i == j:
                continue

            pts_b = objects[oid_b]["points"]
            min_b = pts_b.min(axis=0)
            max_b = pts_b.max(axis=0)

            # derive ALL applicable relation labels from geometry
            rels = derive_relations(min_a, max_a, min_b, max_b)

            # build multi-hot label vector (all-zeros if no relation)
            multi_hot = np.zeros(NUM_RELATIONS, dtype=np.float32)
            for r in rels:
                multi_hot[int(r)] = 1.0

            # Always include the edge — even if multi_hot is all zeros.
            # This gives the GNN negative examples (pairs with NO relation)
            # so it learns when NOT to predict a relation.
            edge_feat = _edge_features(pts_a, pts_b, scene_extent)

            src_list.append(i)
            dst_list.append(j)
            label_list.append(multi_hot)
            edge_feat_list.append(edge_feat)

    if not src_list:
        return None

    return {
        "x":          torch.tensor(x, dtype=torch.float32),
        "edge_index": torch.tensor([src_list, dst_list], dtype=torch.long),
        "edge_attr":  torch.tensor(np.stack(edge_feat_list), dtype=torch.float32),
        "edge_label": torch.tensor(np.stack(label_list), dtype=torch.float32),
        "obj_labels": obj_labels,
        "scene_id":   scene_data["scene_id"],
    }


def build_scannet_scene_graph(
    scene_data: Dict,
    relationships: List[list],
    dssg_obj_ids: Optional[List[str]] = None,
) -> Optional[Dict]:
    """
    Legacy builder kept for compatibility.
    For new training use build_scannet_scene_graph_geometric() instead.
    """
    return build_scannet_scene_graph_geometric(scene_data)


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class SceneGraphDatasetScanNet(Dataset):
    """
    PyTorch Dataset: ScanNet geometry + geometrically-derived relation labels.

    Performance optimisations vs. naive version:
      1. Disk cache — each processed graph is saved as a .pt file in
         `cache_dir` on first load. Subsequent runs skip all PLY/JSON parsing
         and load in ~2ms per scene instead of ~1.5s.
      2. open3d PLY reader — C++ backend, ~10× faster than plyfile for the
         mesh PLY (labels PLY still uses plyfile for custom scalar fields).
      3. Vectorised seg_to_verts — built with a single np.unique pass.

    First run:  ~36 min to process 1468 scenes (bottleneck: PLY I/O)
    Subsequent: ~30 sec to load 1468 cached .pt files

    Node features (10-dim): centroid, size, volume, density, z_relative
    Edge features (8-dim):  delta_z, xy_dist, dist_3d, overlap, vol_ratio,
                            height_ratio, vertical_gap, size_ratio_xy
    """

    # bump this if the feature schema changes — invalidates old cache files
    CACHE_VERSION = "v7_margins"

    def __init__(
        self,
        scannet_dir: str = "data/scannet/scans",
        cache_dir: str = "D:/logicsplat_data/scannet_cache",
        max_scenes: Optional[int] = None,
        verbose: bool = True,
    ):
        self.graphs: List[Dict] = []
        skipped = 0
        loaded = 0
        cache_hits = 0

        if not os.path.isdir(scannet_dir):
            if verbose:
                print(f"SceneGraphDatasetScanNet: directory not found: {scannet_dir}")
            return

        os.makedirs(cache_dir, exist_ok=True)

        scene_ids = sorted([
            name for name in os.listdir(scannet_dir)
            if os.path.isdir(os.path.join(scannet_dir, name))
        ])

        try:
            from tqdm import tqdm
            _tqdm = tqdm
        except ImportError:
            _tqdm = None

        total = min(len(scene_ids), max_scenes) if max_scenes else len(scene_ids)
        if verbose:
            print(f"SceneGraphDatasetScanNet: processing {total} scenes "
                  f"(cache: {cache_dir})")

        scene_iter = scene_ids[:total]
        if _tqdm is not None and verbose:
            scene_iter = _tqdm(
                scene_iter,
                desc="Building cache",
                unit="scene",
                dynamic_ncols=True,
            )

        for scene_id in scene_iter:
            if max_scenes is not None and loaded >= max_scenes:
                break

            cache_path = os.path.join(
                cache_dir, f"{scene_id}_{self.CACHE_VERSION}.pt"
            )

            # ── try cache first ───────────────────────────────────────────────
            if os.path.exists(cache_path):
                try:
                    graph = torch.load(cache_path, weights_only=False)
                    self.graphs.append(graph)
                    loaded += 1
                    cache_hits += 1
                    if _tqdm is not None and verbose:
                        scene_iter.set_postfix(
                            loaded=loaded, cache=cache_hits, skip=skipped
                        )
                    continue
                except Exception:
                    os.remove(cache_path)

            # ── process from raw files ────────────────────────────────────────
            scene_data = load_scannet_scene(scene_id, scannet_dir)
            if scene_data is None:
                skipped += 1
                if _tqdm is not None and verbose:
                    scene_iter.set_postfix(
                        loaded=loaded, cache=cache_hits, skip=skipped
                    )
                continue

            graph = build_scannet_scene_graph_geometric(scene_data)
            if graph is None:
                skipped += 1
                if _tqdm is not None and verbose:
                    scene_iter.set_postfix(
                        loaded=loaded, cache=cache_hits, skip=skipped
                    )
                continue

            torch.save(graph, cache_path)
            self.graphs.append(graph)
            loaded += 1

            if _tqdm is not None and verbose:
                scene_iter.set_postfix(
                    loaded=loaded, cache=cache_hits, skip=skipped
                )
            elif verbose and loaded % 50 == 0:
                print(f"  ... {loaded}/{total} scenes processed "
                      f"({cache_hits} cached, {skipped} skipped)")

        if verbose:
            print(f"SceneGraphDatasetScanNet: loaded {loaded} scenes "
                  f"({cache_hits} from cache, {skipped} skipped)")
            if self.graphs:
                self._print_stats()

    def _print_stats(self):
        from collections import Counter
        # edge_label is now (E, NUM_RELATIONS) multi-hot — sum positives per class
        import torch as _torch
        pos_counts = _torch.zeros(NUM_RELATIONS)
        total_edges = 0
        for g in self.graphs:
            if "edge_label" in g:
                lbl = g["edge_label"]
                pos_counts += lbl.sum(dim=0)
                total_edges += lbl.shape[0]
        if total_edges == 0:
            return
        print(f"Relation distribution (ScanNet, {total_edges} edges, multi-label):")
        max_count = pos_counts.max().item()
        for idx in range(NUM_RELATIONS):
            count = int(pos_counts[idx].item())
            bar = "#" * int(30 * count / max(max_count, 1))
            print(f"  {RELATION_NAMES[idx]:20s} {count:6d}  {bar}")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Dict:
        return self.graphs[idx]
