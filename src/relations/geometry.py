"""
Geometric relation derivation from 3D bounding boxes.

Given two objects A and B with known 3D bounding boxes,
derive which relations hold between them purely from geometry.
All thresholds are relative to object size — scale invariant.
"""
import numpy as np
from typing import List, Tuple
from src.relations.schema import Relation


def derive_relations(
    a_min: np.ndarray, a_max: np.ndarray,  # bbox of object A
    b_min: np.ndarray, b_max: np.ndarray,  # bbox of object B
    proximity_factor: float = 1.5,          # multiplier for adjacency threshold
    height_factor: float = 0.3,             # min Z diff fraction to count as higher/lower
) -> List[Relation]:
    """
    Derive all spatial relations between object A and object B
    from their 3D bounding boxes.

    Args:
        a_min, a_max: bbox corners of object A [x, y, z]
        b_min, b_max: bbox corners of object B [x, y, z]
        proximity_factor: how close = adjacent (relative to avg size)
        height_factor: min centroid Z diff to count as higher/lower

    Returns:
        List of Relation enums that hold for (A, B)
    """
    relations = []

    a_center = (a_min + a_max) / 2
    b_center = (b_min + b_max) / 2
    a_size = a_max - a_min
    b_size = b_max - b_min

    avg_size = (np.linalg.norm(a_size) + np.linalg.norm(b_size)) / 2
    avg_height = (a_size[2] + b_size[2]) / 2

    delta = a_center - b_center  # A relative to B
    dist_xy = float(np.linalg.norm(delta[:2]))
    dist_3d = float(np.linalg.norm(delta))

    # ── Physical relations ────────────────────────────────────────────────────

    # ON_TOP_OF: A's bottom is near B's top, A is above B, XY footprints overlap
    vertical_gap = float(a_min[2] - b_max[2])
    xy_overlap = _bbox_xy_overlap(a_min, a_max, b_min, b_max)
    if (a_center[2] > b_center[2] and
            -avg_height * 0.5 < vertical_gap <= avg_height * 0.4 and
            xy_overlap):
        relations.append(Relation.ON_TOP_OF)

    # UNDER: inverse of on_top_of
    vertical_gap_inv = float(b_min[2] - a_max[2])
    if (b_center[2] > a_center[2] and
            -avg_height * 0.5 < vertical_gap_inv <= avg_height * 0.4 and
            xy_overlap):
        relations.append(Relation.UNDER)

    # INSIDE: A's bbox fully contained within B's bbox, A smaller
    a_vol = float(np.prod(np.maximum(a_size, 1e-6)))
    b_vol = float(np.prod(np.maximum(b_size, 1e-6)))
    if (a_vol < b_vol * 0.7 and
            np.all(a_min >= b_min - 0.05) and
            np.all(a_max <= b_max + 0.05)):
        relations.append(Relation.INSIDE)

    # HANGING_FROM: A is above B, A's bottom is near B's top, no XY support
    if (a_center[2] > b_center[2] and
            0 < vertical_gap <= avg_height * 0.3 and
            not xy_overlap):
        relations.append(Relation.HANGING_FROM)

    # ATTACHED_TO: very close in 3D, bboxes nearly touching or overlapping
    min_gap = _min_bbox_gap(a_min, a_max, b_min, b_max)
    if min_gap < avg_size * 0.05 and not Relation.ON_TOP_OF in relations:
        relations.append(Relation.ATTACHED_TO)

    # ── Proximity relations ───────────────────────────────────────────────────

    # ADJACENT_TO: close in XY, similar Z height, not physically stacked
    z_diff = abs(float(delta[2]))
    threshold = avg_size * proximity_factor
    if (dist_xy < threshold and
            z_diff < avg_height * 1.5 and
            Relation.ON_TOP_OF not in relations and
            Relation.INSIDE not in relations):
        relations.append(Relation.ADJACENT_TO)

    # ── Directional relations ─────────────────────────────────────────────────
    # Only add if objects are meaningfully separated (not overlapping)

    if dist_3d > avg_size * 0.3:
        # LEFT_OF / RIGHT_OF — X axis
        if abs(delta[0]) > abs(delta[1]) and abs(delta[0]) > abs(delta[2]):
            if delta[0] < 0:
                relations.append(Relation.LEFT_OF)
            else:
                relations.append(Relation.RIGHT_OF)

        # IN_FRONT_OF / BEHIND — Y axis (depth)
        if abs(delta[1]) > abs(delta[0]) and abs(delta[1]) > abs(delta[2]):
            if delta[1] < 0:
                relations.append(Relation.IN_FRONT_OF)
            else:
                relations.append(Relation.BEHIND)

    # ── Comparative relations ─────────────────────────────────────────────────

    z_centroid_diff = float(delta[2])
    if z_centroid_diff > avg_height * height_factor:
        relations.append(Relation.HIGHER_THAN)
    elif z_centroid_diff < -avg_height * height_factor:
        relations.append(Relation.LOWER_THAN)

    return relations


# ── helpers ───────────────────────────────────────────────────────────────────

def _bbox_xy_overlap(a_min, a_max, b_min, b_max) -> bool:
    return (a_min[0] < b_max[0] and a_max[0] > b_min[0] and
            a_min[1] < b_max[1] and a_max[1] > b_min[1])


def _xy_contains(outer_min, outer_max, inner_min, inner_max) -> bool:
    return (outer_min[0] <= inner_min[0] and outer_max[0] >= inner_max[0] and
            outer_min[1] <= inner_min[1] and outer_max[1] >= inner_max[1])


def _min_bbox_gap(a_min, a_max, b_min, b_max) -> float:
    """Minimum gap between two bboxes in any axis. Negative = overlap."""
    gaps = []
    for i in range(3):
        gaps.append(max(a_min[i] - b_max[i], b_min[i] - a_max[i], 0.0))
    return float(np.linalg.norm(gaps))
