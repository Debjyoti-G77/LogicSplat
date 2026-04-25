"""
3RScan Geometric Feature Extractor.

When 3RScan point clouds arrive, this replaces the semantic node features
in the 3DSSG loader with real 3D geometry — closing the train/test domain gap.

3RScan structure (after download):
    data/3RScan/
        <scan_id>/
            mesh.refined.v2.obj      ← full scene mesh
            mesh.refined.v2.ply      ← full scene point cloud (XYZ + RGB)
            labels.instances.annotated.v2.ply  ← per-point instance labels

Each point in labels.ply has:
    - x, y, z
    - red, green, blue
    - objectId  ← matches object IDs in 3DSSG objects.json
"""
import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch


# ── PLY reader ────────────────────────────────────────────────────────────────

def read_ply_with_labels(ply_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a 3RScan annotated PLY file.
    Returns:
        points: (N, 3) XYZ
        colors: (N, 3) RGB
        labels: (N,)  object instance IDs
    """
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    points = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
    colors = np.stack([v['red'], v['green'], v['blue']], axis=1).astype(np.uint8)
    # objectId field name varies — try common names
    for field in ['objectId', 'object_id', 'label', 'instance_id']:
        if field in v.data.dtype.names:
            labels = np.array(v[field])
            break
    else:
        labels = np.zeros(len(points), dtype=np.int32)
    return points, colors, labels


# ── geometric feature extraction ──────────────────────────────────────────────

NODE_FEATURE_DIM_GEO = 10  # geometric node features

def extract_geometric_node_features(
    obj_points: np.ndarray,
    scene_points: np.ndarray,
) -> np.ndarray:
    """
    Extract geometric node features for one object from its point cloud.

    Features (10 dims):
        [0]  centroid_x (normalized by scene extent)
        [1]  centroid_y
        [2]  centroid_z
        [3]  size_x (bbox width)
        [4]  size_y (bbox depth)
        [5]  size_z (bbox height)
        [6]  volume (normalized)
        [7]  height_ratio (size_z / max(size_x, size_y))
        [8]  point_density (pts / volume)
        [9]  z_relative (centroid_z relative to scene floor)
    """
    if len(obj_points) == 0:
        return np.zeros(NODE_FEATURE_DIM_GEO, dtype=np.float32)

    # scene normalization
    scene_min = scene_points.min(axis=0)
    scene_max = scene_points.max(axis=0)
    scene_extent = np.maximum(scene_max - scene_min, 1e-6)

    centroid = obj_points.mean(axis=0)
    bbox_min = obj_points.min(axis=0)
    bbox_max = obj_points.max(axis=0)
    size = np.maximum(bbox_max - bbox_min, 1e-6)

    centroid_norm = (centroid - scene_min) / scene_extent
    volume = float(size[0] * size[1] * size[2])
    volume_norm = min(volume / (scene_extent[0] * scene_extent[1] * scene_extent[2]), 1.0)
    height_ratio = float(size[2] / max(size[0], size[1]))
    density = min(len(obj_points) / max(volume, 1e-6), 1000.0) / 1000.0
    z_relative = float((centroid[2] - scene_min[2]) / scene_extent[2])

    return np.array([
        centroid_norm[0], centroid_norm[1], centroid_norm[2],
        size[0] / scene_extent[0], size[1] / scene_extent[1], size[2] / scene_extent[2],
        volume_norm, height_ratio, density, z_relative,
    ], dtype=np.float32)


EDGE_FEATURE_DIM_GEO = 8  # geometric edge features

def extract_geometric_edge_features(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    scene_extent: np.ndarray,
) -> np.ndarray:
    """
    Extract geometric edge features between two objects.
    These are the same features as extract_pair_features but normalized.

    Features (8 dims):
        [0]  delta_z (centroid Z difference, normalized)
        [1]  xy_distance (normalized)
        [2]  dist_3d (normalized)
        [3]  bbox_overlap_xy (fraction)
        [4]  volume_ratio (min/max)
        [5]  height_ratio (a/b)
        [6]  vertical_gap (a.bottom - b.top, normalized)
        [7]  size_ratio_xy
    """
    if len(pts_a) == 0 or len(pts_b) == 0:
        return np.zeros(EDGE_FEATURE_DIM_GEO, dtype=np.float32)

    c_a = pts_a.mean(axis=0)
    c_b = pts_b.mean(axis=0)
    min_a, max_a = pts_a.min(axis=0), pts_a.max(axis=0)
    min_b, max_b = pts_b.min(axis=0), pts_b.max(axis=0)
    size_a = np.maximum(max_a - min_a, 1e-6)
    size_b = np.maximum(max_b - min_b, 1e-6)
    norm = np.maximum(scene_extent, 1e-6)

    delta_z    = float((c_a[2] - c_b[2]) / norm[2])
    xy_dist    = float(np.linalg.norm(c_a[:2] - c_b[:2]) / np.linalg.norm(norm[:2]))
    dist_3d    = float(np.linalg.norm(c_a - c_b) / np.linalg.norm(norm))

    ix = max(0.0, min(max_a[0], max_b[0]) - max(min_a[0], min_b[0]))
    iy = max(0.0, min(max_a[1], max_b[1]) - max(min_a[1], min_b[1]))
    area_a = max((max_a[0]-min_a[0]) * (max_a[1]-min_a[1]), 1e-9)
    bbox_overlap = (ix * iy) / area_a

    vol_a = float(np.prod(size_a))
    vol_b = float(np.prod(size_b))
    vol_ratio = min(vol_a, vol_b) / max(vol_a, vol_b)

    h_ratio = float(np.clip(np.log1p(size_a[2] / max(size_b[2], 1e-6)), -3.0, 3.0) / 3.0)
    vert_gap = float(np.clip((min_a[2] - max_b[2]) / norm[2], -1.0, 1.0))
    size_ratio_xy = float(np.clip(
        np.log1p(np.linalg.norm(size_a[:2]) / max(np.linalg.norm(size_b[:2]), 1e-6)),
        -3.0, 3.0,
    ) / 3.0)

    return np.array([
        delta_z, xy_dist, dist_3d, bbox_overlap,
        vol_ratio, h_ratio, vert_gap, size_ratio_xy,
    ], dtype=np.float32)


# ── scene loader ──────────────────────────────────────────────────────────────

def load_scan_geometry(
    scan_id: str,
    rscan_dir: str = "data/3RScan",
) -> Optional[Dict]:
    """
    Load geometric features for all objects in a 3RScan scene.

    Returns dict:
        {obj_id: {'points': np.ndarray, 'node_feat': np.ndarray}}
    or None if scan not found.
    """
    ply_path = os.path.join(rscan_dir, scan_id,
                            "labels.instances.annotated.v2.ply")
    if not os.path.exists(ply_path):
        return None

    try:
        points, colors, labels = read_ply_with_labels(ply_path)
    except Exception as e:
        print(f"  Failed to read {scan_id}: {e}")
        return None

    scene_extent = np.maximum(
        points.max(axis=0) - points.min(axis=0), 1e-6
    )

    obj_data = {}
    for obj_id in np.unique(labels):
        if obj_id == 0:  # background
            continue
        mask = labels == obj_id
        obj_pts = points[mask]
        node_feat = extract_geometric_node_features(obj_pts, points)
        obj_data[str(obj_id)] = {
            "points":    obj_pts,
            "node_feat": node_feat,
        }

    return {"objects": obj_data, "scene_extent": scene_extent, "all_points": points}


# ── upgraded dataset builder ──────────────────────────────────────────────────

def build_geometric_scene_graph(
    objects_3dssg: List[dict],
    relationships: List[list],
    scan_geometry: Dict,
    class_index: Dict[str, int],
    num_classes: int,
) -> Optional[Dict]:
    """
    Build scene graph using GEOMETRIC features from 3RScan point clouds.
    Falls back to semantic features if geometry not available for an object.

    This is the drop-in upgrade for loader_3dssg.build_scene_graph.
    """
    from src.dataset.loader_3dssg import encode_object
    from src.relations.schema import DSSG_TO_SCHEMA

    id_to_idx = {obj["id"]: i for i, obj in enumerate(objects_3dssg)}
    obj_geo = scan_geometry["objects"]
    scene_extent = scan_geometry["scene_extent"]

    # node features — geometric if available, semantic fallback
    node_feats = []
    for obj in objects_3dssg:
        obj_id = obj["id"]
        if obj_id in obj_geo:
            node_feats.append(obj_geo[obj_id]["node_feat"])
        else:
            # fallback to semantic
            sem = encode_object(obj, class_index, num_classes)
            # pad to geometric dim
            padded = np.zeros(NODE_FEATURE_DIM_GEO, dtype=np.float32)
            padded[:len(sem)] = sem
            node_feats.append(padded)

    x = np.stack(node_feats)

    # edge features and labels
    src, dst, labels, edge_feats = [], [], [], []
    for rel in relationships:
        subj_id = str(rel[0])
        obj_id  = str(rel[1])
        rel_name = rel[3]

        if rel_name not in DSSG_TO_SCHEMA:
            continue
        if subj_id not in id_to_idx or obj_id not in id_to_idx:
            continue

        i = id_to_idx[subj_id]
        j = id_to_idx[obj_id]

        # geometric edge features
        pts_a = obj_geo[subj_id]["points"] if subj_id in obj_geo else np.zeros((1, 3))
        pts_b = obj_geo[obj_id]["points"]  if obj_id  in obj_geo else np.zeros((1, 3))
        edge_feat = extract_geometric_edge_features(pts_a, pts_b, scene_extent)

        src.append(i)
        dst.append(j)
        labels.append(int(DSSG_TO_SCHEMA[rel_name]))
        edge_feats.append(edge_feat)

    if len(src) == 0:
        return None

    return {
        "x":          torch.tensor(x, dtype=torch.float32),
        "edge_index": torch.tensor([src, dst], dtype=torch.long),
        "edge_attr":  torch.tensor(np.stack(edge_feats), dtype=torch.float32),
        "edge_label": torch.tensor(labels, dtype=torch.long),
        "obj_labels": [obj["label"] for obj in objects_3dssg],
    }
