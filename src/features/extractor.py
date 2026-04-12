"""Extract geometric feature vectors for object pairs."""
import numpy as np
from typing import List, Tuple
from src.graph.definitions import Object3D


def extract_pair_features(a: Object3D, b: Object3D) -> np.ndarray:
    """
    Extract a geometric feature vector for an ordered object pair (a, b).

    Features:
        0  - delta_z: centroid Z difference (b.z - a.z)
        1  - xy_distance: XY centroid distance
        2  - dist_3d: full 3D centroid distance
        3  - bbox_overlap_xy: fraction of a's XY footprint overlapping b
        4  - volume_ratio: min/max volume ratio
        5  - height_ratio: a.height / b.height
        6  - size_ratio_xy: a.xy_footprint / b.xy_footprint
        7  - color_distance: Euclidean RGB distance (normalized 0-1)
        8  - a_bottom_z: absolute Z of a's bottom face
        9  - b_top_z: absolute Z of b's top face
        10 - vertical_gap: a.bottom_z - b.top_z
        11 - a_point_count_log: log of a's point count
        12 - b_point_count_log: log of b's point count
    """
    # Z difference
    delta_z = float(b.centroid[2] - a.centroid[2])

    # XY and 3D distances
    xy_dist = float(np.linalg.norm(a.centroid[:2] - b.centroid[:2]))
    dist_3d = float(np.linalg.norm(a.centroid - b.centroid))

    # XY bbox overlap (intersection area / a's area)
    ax_min, ax_max = a.bbox_min[0], a.bbox_max[0]
    ay_min, ay_max = a.bbox_min[1], a.bbox_max[1]
    bx_min, bx_max = b.bbox_min[0], b.bbox_max[0]
    by_min, by_max = b.bbox_min[1], b.bbox_max[1]

    ix = max(0.0, min(ax_max, bx_max) - max(ax_min, bx_min))
    iy = max(0.0, min(ay_max, by_max) - max(ay_min, by_min))
    intersection = ix * iy
    a_area = max((ax_max - ax_min) * (ay_max - ay_min), 1e-9)
    bbox_overlap_xy = intersection / a_area

    # Volume ratio
    vol_a = max(a.volume, 1e-9)
    vol_b = max(b.volume, 1e-9)
    volume_ratio = min(vol_a, vol_b) / max(vol_a, vol_b)

    # Height ratio
    h_a = max(a.size[2], 1e-9)
    h_b = max(b.size[2], 1e-9)
    height_ratio = h_a / h_b

    # XY footprint ratio
    fp_a = max(float(np.linalg.norm(a.size[:2])), 1e-9)
    fp_b = max(float(np.linalg.norm(b.size[:2])), 1e-9)
    size_ratio_xy = fp_a / fp_b

    # Color distance (normalized)
    color_dist = float(np.linalg.norm(
        a.color.astype(float) - b.color.astype(float)
    )) / 441.67  # max possible = sqrt(3 * 255^2)

    # Vertical geometry
    a_bottom_z = float(a.bbox_min[2])
    b_top_z = float(b.bbox_max[2])
    vertical_gap = a_bottom_z - b_top_z

    # Point counts (log scale)
    a_pts_log = float(np.log1p(a.point_count))
    b_pts_log = float(np.log1p(b.point_count))

    return np.array([
        delta_z, xy_dist, dist_3d, bbox_overlap_xy,
        volume_ratio, height_ratio, size_ratio_xy,
        color_dist, a_bottom_z, b_top_z, vertical_gap,
        a_pts_log, b_pts_log,
    ], dtype=np.float32)


FEATURE_NAMES = [
    "delta_z", "xy_distance", "dist_3d", "bbox_overlap_xy",
    "volume_ratio", "height_ratio", "size_ratio_xy",
    "color_distance", "a_bottom_z", "b_top_z", "vertical_gap",
    "a_pts_log", "b_pts_log",
]


def extract_all_pairs(
    objects: List[Object3D],
    scene_id: str,
) -> List[dict]:
    """
    Extract feature vectors for all ordered pairs in a scene.
    Returns a list of dicts ready to write to CSV.
    """
    rows = []
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i == j:
                continue
            feats = extract_pair_features(a, b)
            row = {
                "scene_id": scene_id,
                "obj_a": a.uid,
                "obj_b": b.uid,
                "a_color": f"rgb({a.color[0]},{a.color[1]},{a.color[2]})",
                "b_color": f"rgb({b.color[0]},{b.color[1]},{b.color[2]})",
                "a_pts": a.point_count,
                "b_pts": b.point_count,
                "relation": "",  # to be filled manually
            }
            for name, val in zip(FEATURE_NAMES, feats):
                row[name] = round(float(val), 4)
            rows.append(row)
    return rows
