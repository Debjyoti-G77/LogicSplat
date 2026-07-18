"""
Build PyG-compatible training graphs from instance-labeled Gaussians + 3DSSG annotations.

For each scene with instance_labels.npz:
  1. Load Gaussian splat + instance labels
  2. Group Gaussians by instance ID (skip -1/unlabeled)
  3. For each instance group, compute Object3D-like representation
  4. Extract 10-dim node features (extract_gaussian_node_features)
  5. Extract 22-dim edge features (extract_3rscan_edge_features) for ALL directed pairs
  6. Load 3DSSG relations and build multi-hot edge labels (E × 12)
  7. Save to D:/logicsplat_data/3rscan_graph_cache/{scene_id}.pt

CRITICAL: 3DSSG object IDs directly correspond to instance IDs in the mesh
segmentation — no matching needed.

Usage:
    python scripts/build_3rscan_graphs.py
    python scripts/build_3rscan_graphs.py --max-scenes 10
    python scripts/build_3rscan_graphs.py --force
"""

import argparse
import json
import os
import sys
import time
import warnings
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gaussian.loader import load_gaussian_ply
from src.gaussian.clustering import extract_gaussian_node_features
from src.graph.definitions import Object3D
from src.relations.schema import DSSG_TO_SCHEMA, NUM_RELATIONS, Relation
from src.relations.geometry import derive_relations, compute_scene_context


# === Rule-based label injection settings ===
# Relations injected at full confidence (geometry IS the ground truth)
INJECT_FULL_CONF = {
    int(Relation.LEFT_OF),
    int(Relation.RIGHT_OF),
    int(Relation.IN_FRONT_OF),
    int(Relation.BEHIND),
    int(Relation.HIGHER_THAN),
    int(Relation.LOWER_THAN),
}
# Relations injected at soft confidence (rule is good but can have FPs)
INJECT_SOFT_CONF = {
    int(Relation.ON_TOP_OF):   0.75,
    int(Relation.UNDER):       0.75,
    # adjacent_to: injected with a STRICT proximity (0.8x avg_diagonal, not default 1.5x)
    # to avoid the annotation-inconsistency problem. Confidence 0.55 — low enough to
    # not override human labels, but enough to give the model training signal (was F1=0.25).
    int(Relation.ADJACENT_TO): 0.55,
    # attached_to: bbox gap < 5% avg_size is valid for wall-mounted/built-in objects
    # (the dominant case in 3DSSG). Confidence 0.60 — moderate to handle false positives
    # from objects placed close together. Was oscillating 0.31-0.46 with only 7,348 positives.
    int(Relation.ATTACHED_TO): 0.60,
}
# Proximity factor for adjacent_to injection (overrides default 1.5x).
# 0.8x avg_diagonal means objects within ~0.8 object-widths of each other — tight enough
# to avoid firing on distant pairs but loose enough to catch nearby objects.
ADJACENT_TO_PROXIMITY_FACTOR = 0.8


# === Inverse relation mapping for label injection (Fix 2) ===
INVERSE_RELATIONS = {
    'on_top_of': 'under',
    'under': 'on_top_of',
    'higher_than': 'lower_than',
    'lower_than': 'higher_than',
    'left_of': 'left',          # 3DSSG uses "left" not "left_of"
    'left': 'right',
    'right': 'left',
    'right_of': 'left_of',
    'in_front_of': 'behind',
    'front': 'behind',          # 3DSSG uses "front" not "in_front_of"
    'behind': 'front',
}

# Map from our schema relation names to their inverse schema index
SCHEMA_INVERSE_PAIRS = {
    int(Relation.ON_TOP_OF): int(Relation.UNDER),
    int(Relation.UNDER): int(Relation.ON_TOP_OF),
    int(Relation.HIGHER_THAN): int(Relation.LOWER_THAN),
    int(Relation.LOWER_THAN): int(Relation.HIGHER_THAN),
    int(Relation.LEFT_OF): int(Relation.RIGHT_OF),
    int(Relation.RIGHT_OF): int(Relation.LEFT_OF),
    int(Relation.IN_FRONT_OF): int(Relation.BEHIND),
    int(Relation.BEHIND): int(Relation.IN_FRONT_OF),
}


# === Configuration ===
SPLATS_DIR = Path(r"D:\3rscan_splats")
OUTPUT_DIR = Path("D:/logicsplat_data/3rscan_graph_cache")
RELATIONSHIPS_JSON = Path(__file__).parent.parent / "data" / "3DSSG" / "relationships.json"
OBJECTS_JSON = Path(__file__).parent.parent / "data" / "3DSSG" / "objects.json"

MIN_INSTANCES = 3
MIN_COVERAGE = 0.50


def inject_rule_labels(
    objects: List,
    src_list: List[int],
    dst_list: List[int],
    edge_label: np.ndarray,
    scene_context: dict,
) -> None:
    """
    Inject geometric rule labels for unannotated relations on every edge.

    Per-relation injection: if 3DSSG already labeled a relation on this edge,
    the 3DSSG label wins (not overwritten). Rule labels are injected only where
    edge_label[pos, rel_idx] == 0.

    Directional (left/right/front/behind/higher/lower): confidence=1.0
    on_top_of / under: confidence=0.75
    adjacent_to / attached_to: skipped (noisy / inaccurate proxy)
    """
    for pos, (src_idx, dst_idx) in enumerate(zip(src_list, dst_list)):
        obj_a = objects[src_idx]
        obj_b = objects[dst_idx]

        derived = derive_relations(
            obj_a.bbox_min, obj_a.bbox_max,
            obj_b.bbox_min, obj_b.bbox_max,
            proximity_factor=ADJACENT_TO_PROXIMITY_FACTOR,
            scene_context=scene_context,
        )

        for rel in derived:
            rel_idx = int(rel)
            if rel_idx >= NUM_RELATIONS:
                continue  # safety: filter any sentinel values
            if edge_label[pos, rel_idx] != 0:
                continue  # 3DSSG or Fix-2 inverse label takes priority
            if rel_idx in INJECT_FULL_CONF:
                edge_label[pos, rel_idx] = 1.0
            elif rel_idx in INJECT_SOFT_CONF:
                edge_label[pos, rel_idx] = INJECT_SOFT_CONF[rel_idx]


def load_3dssg_annotations() -> tuple:
    """
    Load 3DSSG objects and relationships indexed by scan_id.

    Returns:
        objects_by_scan: {scan_id: [obj_dict, ...]}
        rels_by_scan:    {scan_id: [[subj_id, obj_id, rel_type_id, rel_name], ...]}
    """
    with open(OBJECTS_JSON, "r") as f:
        objects_data = json.load(f)
    with open(RELATIONSHIPS_JSON, "r") as f:
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


def build_object3d_from_gaussians(
    instance_id: int,
    xyz: np.ndarray,
    rgb: np.ndarray,
    opacity: np.ndarray,
    covariance: np.ndarray,
) -> Object3D:
    """
    Build an Object3D-like representation from a group of Gaussians
    belonging to the same instance.

    Args:
        instance_id: the instance label
        xyz: (M, 3) positions of Gaussians in this instance
        rgb: (M, 3) colors
        opacity: (M,) opacities
        covariance: (M, 6) upper-triangle covariance

    Returns:
        Object3D with computed attributes + extra fields for feature extraction
    """
    weights = opacity / max(opacity.sum(), 1e-9)
    centroid = (xyz * weights[:, None]).sum(axis=0)
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    mean_color = (rgb.astype(float) * weights[:, None]).sum(axis=0).astype(np.uint8)
    mean_opacity = float(opacity.mean())

    # Compute eigenvalues from mean covariance
    mean_cov = covariance.mean(axis=0)
    cov_matrix = np.array([
        [mean_cov[0], mean_cov[1], mean_cov[2]],
        [mean_cov[1], mean_cov[3], mean_cov[4]],
        [mean_cov[2], mean_cov[4], mean_cov[5]],
    ])
    eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(cov_matrix)))[::-1]

    obj = Object3D(
        uid=instance_id,
        centroid=centroid,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        color=mean_color,
        point_count=len(xyz),
    )
    # Attach extra attributes used by extract_gaussian_node_features
    obj._mean_opacity = mean_opacity
    obj._eigenvalues = eigenvalues
    obj._mean_cov = mean_cov

    return obj


def extract_3rscan_edge_features(obj_a: Object3D, obj_b: Object3D, scene_extent: np.ndarray) -> np.ndarray:
    """
    Extract 19-dim geometric edge features using OBJECT-DIAGONAL normalization
    for directional features [0-4] and [8]. This makes features scale-invariant
    across room-scale and tabletop domains.

    Key design decisions:
    - Features [0-4, 8]: normalized by avg_diag (avg of both objects' diagonals)
      → "one object-width apart" always maps to ~1.0 regardless of scene scale
    - Feature [2] delta_z: uses centroid z diff (not bbox_top) to better separate
      higher_than (large centroid diff) from on_top_of (moderate centroid diff)
    - Features [5-7, 9]: scale-invariant ratios, unchanged
    - Features [10-13]: object-height normalized, unchanged
    - Features [14-16]: relative margin features using avg_diag for threshold
    - Features [17-18]: NEW 3D containment features for inside detection

    Features:
        [0]  delta_x  (centroid, avg_diag norm)  ← left_of / right_of
        [1]  delta_y  (centroid, avg_diag norm, negated for 3DSSG convention)
        [2]  delta_z  (centroid, avg_diag norm)  ← higher_than / lower_than
        [3]  xy_dist  (avg_diag norm)
        [4]  dist_3d  (avg_diag norm)
        [5]  bbox_overlap_xy
        [6]  vol_ratio (symmetric)
        [7]  h_ratio (log)
        [8]  vert_gap (min_a[2]-max_b[2], avg_height norm)  ← on_top_of / under
        [9]  size_ratio_xy (log)
        [10] vert_gap_obj_norm (gap / mean_height)
        [11] vert_gap_a_relative (gap / size_a[2])
        [12] contact_score
        [13] support_overlap_b
        [14] ontop_z_margin (centroid_z_diff vs avg_diag threshold)
        [15] ontop_xy_margin
        [16] z_dominance_margin
        [17] containment_3d  ← NEW: fraction of A's volume inside B's bbox
        [18] centroid_inside_b  ← NEW: 1 if A's centroid is inside B's bbox
    """
    c_a = obj_a.centroid
    c_b = obj_b.centroid
    min_a, max_a = obj_a.bbox_min, obj_a.bbox_max
    min_b, max_b = obj_b.bbox_min, obj_b.bbox_max
    size_a = np.maximum(max_a - min_a, 1e-6)
    size_b = np.maximum(max_b - min_b, 1e-6)

    # Object-diagonal normalization: scale-invariant across room/tabletop domains.
    # "One object-width apart" always maps to ~1.0, regardless of scene scale.
    diag_a = float(np.linalg.norm(size_a))
    diag_b = float(np.linalg.norm(size_b))
    avg_diag = max((diag_a + diag_b) / 2.0, 1e-6)

    # ── Features [0-4]: avg_diag normalized directional features ──
    delta_x = float(np.clip((c_a[0] - c_b[0]) / avg_diag, -3.0, 3.0) / 3.0)
    # 3DSSG convention: -Y = front. Negate so positive delta_y = "in_front_of".
    delta_y = float(np.clip(-(c_a[1] - c_b[1]) / avg_diag, -3.0, 3.0) / 3.0)
    # Use centroid z diff (not bbox_top): better separates higher_than (large diff)
    # from on_top_of (moderate diff where A sits just above B's centroid).
    delta_z = float(np.clip((c_a[2] - c_b[2]) / avg_diag, -3.0, 3.0) / 3.0)
    xy_dist = float(np.clip(np.linalg.norm(c_a[:2] - c_b[:2]) / avg_diag, 0.0, 3.0) / 3.0)
    dist_3d = float(np.clip(np.linalg.norm(c_a - c_b) / avg_diag, 0.0, 3.0) / 3.0)

    # ── Features [5-7, 9]: Scale-invariant ──
    ix = max(0.0, min(max_a[0], max_b[0]) - max(min_a[0], min_b[0]))
    iy = max(0.0, min(max_a[1], max_b[1]) - max(min_a[1], min_b[1]))
    area_a = max((max_a[0] - min_a[0]) * (max_a[1] - min_a[1]), 1e-9)
    bbox_overlap = (ix * iy) / area_a

    vol_a = float(np.prod(size_a))
    vol_b = float(np.prod(size_b))
    vol_ratio = min(vol_a, vol_b) / max(vol_a, vol_b)

    h_ratio = float(np.clip(np.log1p(size_a[2] / max(size_b[2], 1e-6)), -3.0, 3.0) / 3.0)

    size_ratio_xy = float(np.clip(
        np.log1p(np.linalg.norm(size_a[:2]) / max(np.linalg.norm(size_b[:2]), 1e-6)),
        -3.0, 3.0,
    ) / 3.0)

    # ── Feature [8]: Vertical gap, avg_height normalized ──
    mean_height = max((size_a[2] + size_b[2]) / 2.0, 1e-6)
    vert_gap = float(np.clip((min_a[2] - max_b[2]) / mean_height, -3.0, 3.0) / 3.0)

    # ── Contact features [10-13] ──
    # [10] gap / mean_height (same normalization as vert_gap, kept separate for model)
    vert_gap_obj_norm = float(np.clip((min_a[2] - max_b[2]) / mean_height, -3.0, 3.0) / 3.0)

    # [11] gap / A's own height (A-relative, distinct from mean-height normalization)
    vert_gap_a_relative = float(np.clip((min_a[2] - max_b[2]) / size_a[2], -3.0, 3.0) / 3.0)

    # [12] contact_score: high when A rests on B (small gap, XY overlap, A above B)
    a_above_b = float(c_a[2] > c_b[2])
    raw_gap = float(min_a[2] - max_b[2])
    contact_score = float(
        np.exp(-abs(raw_gap) / max(mean_height * 0.3, 1e-6)) * bbox_overlap * a_above_b
    )

    # [13] support_overlap_b: XY overlap relative to B's footprint
    area_b = max((max_b[0] - min_b[0]) * (max_b[1] - min_b[1]), 1e-9)
    support_overlap_b = float((ix * iy) / area_b)

    # ── Rule-margin features [14-16] ──
    centroid_z_diff = float(c_a[2] - c_b[2])
    centroid_xy_dist = float(np.linalg.norm(c_a[:2] - c_b[:2]))

    # [14] ontop_z_margin: centroid z diff vs avg_diag threshold (scale-invariant)
    z_min_thresh = max(avg_diag * 0.05, 1e-6)
    ontop_z_margin = float(np.clip(
        (centroid_z_diff - z_min_thresh) / z_min_thresh, -3.0, 3.0
    ) / 3.0)

    # [15] ontop_xy_margin: A's centroid inside B's XY footprint
    b_xy_size = float(np.linalg.norm(size_b[:2]))
    xy_limit = max(b_xy_size * 0.8, 1e-6)
    ontop_xy_margin = float(np.clip(
        (xy_limit - centroid_xy_dist) / xy_limit, -3.0, 3.0
    ) / 3.0)

    # [16] z_dominance_margin: does Z diff dominate XY distance?
    z_dom_margin = float(np.clip(
        (centroid_z_diff - centroid_xy_dist * 0.5) / max(abs(centroid_z_diff) + 1e-6, 1e-6),
        -3.0, 3.0
    ) / 3.0)

    # ── [17-18] 3D containment features for inside detection ──
    iz = max(0.0, min(max_a[2], max_b[2]) - max(min_a[2], min_b[2]))
    vol_overlap = ix * iy * iz
    containment_3d = float(min(vol_overlap / max(vol_a, 1e-9), 1.0))

    centroid_inside_b = float(
        min_b[0] <= c_a[0] <= max_b[0] and
        min_b[1] <= c_a[1] <= max_b[1] and
        min_b[2] <= c_a[2] <= max_b[2]
    )

    # ── [19-21] NEW features for stacking and per-axis containment ──
    # [19] vz_ratio: vertical/horizontal displacement ratio.
    # Stacked (on_top_of): |delta_z| >> xy_dist → large positive value.
    # Side-by-side (adjacent_to): delta_z ≈ 0 → near zero.
    # Dimensionless — scale-invariant key signal for contact relations.
    xy_dist_raw = float(np.linalg.norm(c_a[:2] - c_b[:2]))
    vz_ratio = float(np.clip(
        (c_a[2] - c_b[2]) / max(xy_dist_raw, avg_diag * 0.05), -3.0, 3.0
    ) / 3.0)

    # [20-21] Per-axis containment fractions (complement containment_3d).
    containment_x = float(np.clip(ix / size_a[0], 0.0, 1.0))
    containment_y = float(np.clip(iy / size_a[1], 0.0, 1.0))

    return np.array([
        delta_x, delta_y, delta_z, xy_dist, dist_3d,
        bbox_overlap, vol_ratio, h_ratio, vert_gap, size_ratio_xy,
        vert_gap_obj_norm, vert_gap_a_relative, contact_score, support_overlap_b,
        ontop_z_margin, ontop_xy_margin, z_dom_margin,
        containment_3d, centroid_inside_b,
        vz_ratio, containment_x, containment_y,
    ], dtype=np.float32)


def process_scene(
    scene_id: str,
    splats_dir: Path,
    gt_relationships: List[list],
    gt_objects: List[dict],
) -> Optional[Dict]:
    """
    Process a single scene: load splat + instance labels → build graph.

    Args:
        scene_id:         3RScan scan UUID
        splats_dir:       root directory containing splat files
        gt_relationships: 3DSSG relationships for this scene
        gt_objects:       3DSSG objects for this scene

    Returns:
        Graph dict or None if processing fails.
    """
    # Paths
    splat_ply = splats_dir / scene_id / "ckpts" / "point_cloud_30000.ply"
    labels_npz = splats_dir / scene_id / "instance_labels.npz"

    if not splat_ply.exists() or not labels_npz.exists():
        return None

    # Load Gaussian splat
    try:
        cloud = load_gaussian_ply(str(splat_ply))
    except Exception as e:
        warnings.warn(f"Failed to load splat {scene_id}: {e}", RuntimeWarning)
        return None

    # Load instance labels
    try:
        npz = np.load(str(labels_npz))
        labels = npz["labels"]
        coverage = float(npz["coverage"])
        n_instances = int(npz["n_instances"])
    except Exception as e:
        warnings.warn(f"Failed to load labels {scene_id}: {e}", RuntimeWarning)
        return None

    # Validate
    if len(labels) != cloud.num_gaussians:
        warnings.warn(
            f"{scene_id}: label count ({len(labels)}) != Gaussian count "
            f"({cloud.num_gaussians}), skipping",
            RuntimeWarning,
        )
        return None

    # Skip scenes with low coverage or too few instances
    if coverage < MIN_COVERAGE:
        return None
    if n_instances < MIN_INSTANCES:
        return None

    # Group Gaussians by instance ID (skip -1 = unlabeled)
    unique_instances = sorted(set(labels[labels != -1].tolist()))

    if len(unique_instances) < MIN_INSTANCES:
        return None

    # Build Object3D for each instance
    objects: List[Object3D] = []
    instance_id_to_idx: Dict[int, int] = {}

    for idx, inst_id in enumerate(unique_instances):
        mask = labels == inst_id
        inst_xyz = cloud.xyz[mask]
        inst_rgb = cloud.rgb[mask]
        inst_opacity = cloud.opacity[mask]
        inst_cov = cloud.covariance[mask]

        if len(inst_xyz) < 3:
            continue

        obj = build_object3d_from_gaussians(
            instance_id=inst_id,
            xyz=inst_xyz,
            rgb=inst_rgb,
            opacity=inst_opacity,
            covariance=inst_cov,
        )
        instance_id_to_idx[inst_id] = len(objects)
        objects.append(obj)

    if len(objects) < MIN_INSTANCES:
        return None

    # Compute scene extent
    all_labeled_mask = labels != -1
    all_xyz = cloud.xyz[all_labeled_mask]
    scene_min = all_xyz.min(axis=0)
    scene_max = all_xyz.max(axis=0)
    scene_extent = np.maximum(scene_max - scene_min, 1e-6)

    # ── Scale-invariant scene statistics + scene context for rule injection ──
    obj_diags = [float(np.linalg.norm(np.maximum(o.size, 1e-6))) for o in objects]
    scene_mean_diag = float(np.mean(obj_diags)) if obj_diags else 1.0
    obj_volumes = [float(np.prod(np.maximum(o.size, 1e-6))) for o in objects]
    scene_median_volume = float(np.median(obj_volumes)) if obj_volumes else 1.0

    # Scene context for geometric rule label injection
    all_bbox_mins = np.array([o.bbox_min for o in objects])
    all_bbox_maxs = np.array([o.bbox_max for o in objects])
    scene_ctx = compute_scene_context(all_bbox_mins, all_bbox_maxs)

    # z_rank: ordinal rank [0,1] of each object's centroid_z among all objects.
    # 0 = lowest, 1 = highest. Completely scale-invariant — captures relative
    # vertical ordering that's consistent across room and tabletop domains.
    centroid_zs = np.array([o.centroid[2] for o in objects])
    sorted_z_idx = np.argsort(centroid_zs)
    z_ranks = np.zeros(len(objects))
    if len(objects) > 1:
        for rank, obj_idx in enumerate(sorted_z_idx):
            z_ranks[obj_idx] = rank / (len(objects) - 1)
    else:
        z_ranks[0] = 0.5

    # Extract node features (10-dim, scale-invariant)
    node_features = []
    for idx, obj in enumerate(objects):
        feat = extract_gaussian_node_features(
            obj, scene_extent, scene_min,
            scene_mean_diag=scene_mean_diag,
            scene_median_volume=scene_median_volume,
            z_rank=float(z_ranks[idx]),
        )
        node_features.append(feat)
    x = np.stack(node_features)  # (N, 10)

    # Build ALL directed edges + extract edge features (17-dim)
    # Fix 1: Use scene-extent normalization (matching ScanNet)
    n_objects = len(objects)
    src_list, dst_list, edge_feats = [], [], []

    for i in range(n_objects):
        for j in range(n_objects):
            if i == j:
                continue
            feat = extract_3rscan_edge_features(objects[i], objects[j], scene_extent)
            src_list.append(i)
            dst_list.append(j)
            edge_feats.append(feat)

    edge_index = np.array([src_list, dst_list])  # (2, E)
    edge_attr = np.stack(edge_feats)  # (E, 22)

    # Build multi-hot edge labels from 3DSSG annotations
    # Create mapping from (src_idx, dst_idx) → edge position
    edge_to_pos = {}
    for pos, (i, j) in enumerate(zip(src_list, dst_list)):
        edge_to_pos[(i, j)] = pos

    n_edges = len(src_list)
    edge_label = np.zeros((n_edges, NUM_RELATIONS), dtype=np.float32)

    # 3DSSG object IDs directly correspond to instance IDs — no matching needed
    for rel in gt_relationships:
        # Format: [subject_id, object_id, rel_type_id, "relation_name"]
        if len(rel) < 4:
            continue

        subj_id = int(rel[0])
        obj_id = int(rel[1])
        rel_name = rel[3] if isinstance(rel[3], str) else str(rel[3])

        # Map relation name to our schema
        if rel_name not in DSSG_TO_SCHEMA:
            continue

        relation_idx = int(DSSG_TO_SCHEMA[rel_name])

        # Map instance IDs to object indices
        subj_idx = instance_id_to_idx.get(subj_id)
        obj_idx = instance_id_to_idx.get(obj_id)

        if subj_idx is None or obj_idx is None:
            continue

        edge_pos = edge_to_pos.get((subj_idx, obj_idx))
        if edge_pos is not None:
            edge_label[edge_pos, relation_idx] = 1.0

    # ── Fix 2: Inject inverse relation labels ──
    # For each labeled edge (subj→obj, rel), add the inverse label to (obj→subj)
    # This fixes "under = 0 positives" and boosts directional relations
    for rel in gt_relationships:
        if len(rel) < 4:
            continue

        subj_id = int(rel[0])
        obj_id = int(rel[1])
        rel_name = rel[3] if isinstance(rel[3], str) else str(rel[3])

        if rel_name not in DSSG_TO_SCHEMA:
            continue

        relation_idx = int(DSSG_TO_SCHEMA[rel_name])

        # Check if this relation has an inverse
        if relation_idx not in SCHEMA_INVERSE_PAIRS:
            continue

        inverse_idx = SCHEMA_INVERSE_PAIRS[relation_idx]

        # Map instance IDs to object indices (reversed: obj→subj)
        subj_idx = instance_id_to_idx.get(subj_id)
        obj_idx = instance_id_to_idx.get(obj_id)

        if subj_idx is None or obj_idx is None:
            continue

        # Add inverse label to the reverse edge (obj_idx → subj_idx)
        reverse_edge_pos = edge_to_pos.get((obj_idx, subj_idx))
        if reverse_edge_pos is not None:
            edge_label[reverse_edge_pos, inverse_idx] = 1.0

    # ── Rule-based label injection (v4) ──
    # Inject geometric labels for all unannotated relations using derive_relations().
    # Directional: inject at confidence=1.0 (geometry IS ground truth for these).
    # on_top_of/under: inject at confidence=0.75 (good rule but some FP risk).
    # 3DSSG and Fix-2 inverse labels always take priority (not overwritten).
    inject_rule_labels(objects, src_list, dst_list, edge_label, scene_ctx)

    # Build object label list from 3DSSG annotations
    gt_obj_map = {int(o["id"]): o.get("label", "object") for o in gt_objects}
    obj_labels = [gt_obj_map.get(obj.uid, "object") for obj in objects]

    return {
        "x": torch.tensor(x, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
        "edge_attr": torch.tensor(edge_attr, dtype=torch.float32),
        "edge_label": torch.tensor(edge_label, dtype=torch.float32),
        "scene_id": scene_id,
        "obj_labels": obj_labels,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build PyG-compatible training graphs from instance-labeled Gaussians"
    )
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Limit number of scenes to process")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if .pt already exists")
    parser.add_argument("--splats-dir", type=str, default=None,
                        help="Override splats directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    splats_dir = Path(args.splats_dir) if args.splats_dir else SPLATS_DIR
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Build 3RScan Graph Cache from Instance-Labeled Gaussians")
    print("=" * 60)
    print(f"Splats dir:   {splats_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Min instances: {MIN_INSTANCES}")
    print(f"Min coverage:  {MIN_COVERAGE*100:.0f}%")
    print()

    # Load 3DSSG annotations
    print("[1/4] Loading 3DSSG annotations...")
    objects_by_scan, rels_by_scan = load_3dssg_annotations()
    print(f"  Objects: {len(objects_by_scan)} scenes")
    print(f"  Relations: {len(rels_by_scan)} scenes")

    # Find scenes with instance_labels.npz
    print("\n[2/4] Finding scenes with instance labels...")
    scene_ids = []
    for scene_id in sorted(rels_by_scan.keys()):
        labels_path = splats_dir / scene_id / "instance_labels.npz"
        if labels_path.exists():
            scene_ids.append(scene_id)

    print(f"  Found {len(scene_ids)} scenes with instance_labels.npz")

    if args.max_scenes:
        scene_ids = scene_ids[:args.max_scenes]
        print(f"  Limited to {args.max_scenes} scenes")

    # Process scenes
    print(f"\n[3/4] Building graph tensors...")
    start_time = time.time()

    n_processed = 0
    n_skipped_exists = 0
    n_skipped_coverage = 0
    n_skipped_instances = 0
    n_failed = 0
    n_no_labels = 0
    positive_edges = 0
    total_edges = 0

    for i, scene_id in enumerate(scene_ids):
        output_path = output_dir / f"{scene_id}.pt"

        # Resume support
        if output_path.exists() and not args.force:
            n_skipped_exists += 1
            continue

        gt_rels = rels_by_scan.get(scene_id, [])
        gt_objects = objects_by_scan.get(scene_id, [])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph = process_scene(scene_id, splats_dir, gt_rels, gt_objects)

        if graph is None:
            n_failed += 1
            continue

        # Check that we have at least some positive labels
        n_positive = int(graph["edge_label"].sum().item())
        if n_positive == 0:
            n_no_labels += 1
            continue

        # Save
        try:
            torch.save(graph, str(output_path))
            n_processed += 1
            positive_edges += n_positive
            total_edges += graph["edge_label"].shape[0]

            if n_processed % 20 == 0 or n_processed == 1:
                elapsed = time.time() - start_time
                rate = n_processed / elapsed if elapsed > 0 else 0
                n_nodes = graph["x"].shape[0]
                n_e = graph["edge_label"].shape[0]
                print(
                    f"  [{i+1}/{len(scene_ids)}] Processed {n_processed} graphs "
                    f"({rate:.1f}/s) | Last: {n_nodes} nodes, {n_e} edges, "
                    f"{n_positive} positive labels"
                )
        except Exception as e:
            n_failed += 1
            print(f"  ERROR saving {scene_id}: {e}")

    elapsed = time.time() - start_time

    # Report
    print(f"\n[4/4] Summary")
    print("=" * 60)
    print(f"Total scenes with labels:  {len(scene_ids)}")
    print(f"Processed (new):           {n_processed}")
    print(f"Skipped (already exists):  {n_skipped_exists}")
    print(f"Skipped (no positive rels):{n_no_labels}")
    print(f"Failed (coverage/instances/other): {n_failed}")
    print(f"Time:                      {elapsed:.1f}s")

    if n_processed > 0:
        print(f"\nGraph stats (newly processed):")
        print(f"  Total edges:    {total_edges}")
        print(f"  Positive edges: {positive_edges}")
        print(f"  Positive rate:  {positive_edges/max(total_edges,1)*100:.1f}%")

    # Print relation distribution from all cached graphs
    all_cached = sorted(output_dir.glob("*.pt"))
    if all_cached:
        print(f"\nRelation distribution across {len(all_cached)} cached graphs:")
        from src.relations.schema import RELATION_NAMES
        pos_counts = torch.zeros(NUM_RELATIONS)
        total_e = 0
        for pt_path in all_cached:
            try:
                g = torch.load(str(pt_path), weights_only=False)
                pos_counts += g["edge_label"].sum(dim=0)
                total_e += g["edge_label"].shape[0]
            except Exception:
                continue
        if total_e > 0:
            max_count = pos_counts.max().item()
            for idx in range(NUM_RELATIONS):
                count = int(pos_counts[idx].item())
                bar = "#" * int(30 * count / max(max_count, 1))
                name = RELATION_NAMES.get(idx, f"rel_{idx}")
                print(f"  {name:20s} {count:6d}  {bar}")


if __name__ == "__main__":
    main()
