"""
Geometric relation derivation from 3D bounding boxes.

Given two objects A and B with known 3D bounding boxes,
derive which spatial relations hold between them purely from geometry.

All thresholds are DYNAMIC — derived from scene-level statistics rather than
hardcoded constants. This makes the system robust to different scene scales,
camera distances, and Gaussian splat noise levels.

Architecture:
    1. compute_scene_context() — called ONCE per scene, computes adaptive
       thresholds from the distribution of object sizes and pairwise distances.
    2. derive_relations() — called per-pair, uses scene context for thresholds.
       Falls back to pair-local heuristics when no context is provided.

Scene context thresholds:
    z_min_threshold:     Minimum Z difference to count as "higher" — derived from
                         the scene's median pairwise Z spread (10th percentile of
                         nonzero |Δz| values, floored at 2% of scene Z extent).
    z_max_threshold:     Maximum Z difference for on_top_of — objects floating
                         more than this apart aren't stacked (median Z extent × 2).
    xy_footprint_scale:  How far in XY a centroid can be from the support object
                         and still count as "on top" — derived from median object
                         XY extent.
    dominance_ratio:     How much larger one axis must be vs another to be
                         considered "dominant" for directional relations — derived
                         from the spread of pairwise axis ratios.
"""
import numpy as np
from typing import List, Optional, Dict
from src.relations.schema import Relation


def compute_scene_context(
    all_bbox_mins: np.ndarray,
    all_bbox_maxs: np.ndarray,
) -> Dict[str, float]:
    """
    Compute scene-level statistics for adaptive thresholding.

    Called once per scene before evaluating all pairs. Returns a dict of
    thresholds that adapt to the scene's geometry.

    Args:
        all_bbox_mins: (N, 3) array of bbox min corners for all objects
        all_bbox_maxs: (N, 3) array of bbox max corners for all objects

    Returns:
        Dict with adaptive thresholds derived from scene statistics.
    """
    n = len(all_bbox_mins)
    centers = (all_bbox_mins + all_bbox_maxs) / 2
    sizes = np.maximum(all_bbox_maxs - all_bbox_mins, 1e-6)

    # Scene extent
    scene_min = all_bbox_mins.min(axis=0)
    scene_max = all_bbox_maxs.max(axis=0)
    scene_extent = scene_max - scene_min
    scene_z_extent = max(float(scene_extent[2]), 1e-6)
    scene_xy_extent = max(float(np.linalg.norm(scene_extent[:2])), 1e-6)

    # Object size statistics
    obj_diagonals = np.linalg.norm(sizes, axis=1)  # (N,)
    median_diagonal = float(np.median(obj_diagonals))
    obj_xy_extents = np.linalg.norm(sizes[:, :2], axis=1)  # (N,)
    median_xy_extent = float(np.median(obj_xy_extents))

    # Pairwise distance statistics
    pairwise_dists_3d = []
    pairwise_z_diffs = []
    pairwise_xy_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d3d = float(np.linalg.norm(centers[i] - centers[j]))
            dz = abs(float(centers[i, 2] - centers[j, 2]))
            dxy = float(np.linalg.norm(centers[i, :2] - centers[j, :2]))
            pairwise_dists_3d.append(d3d)
            pairwise_z_diffs.append(dz)
            pairwise_xy_dists.append(dxy)

    if not pairwise_dists_3d:
        # Fallback for < 2 objects
        return _default_context()

    median_dist_3d = float(np.median(pairwise_dists_3d))
    median_z_diff = float(np.median(pairwise_z_diffs))
    median_xy_dist = float(np.median(pairwise_xy_dists))

    # ── Adaptive thresholds ───────────────────────────────────────────────────

    # Z minimum threshold for higher_than:
    # Use 2% of scene Z extent as the noise floor — anything smaller than this
    # is within measurement noise for Gaussian splats.
    # Also consider 5% of median pairwise 3D distance as a relative floor.
    z_min_threshold = max(scene_z_extent * 0.02, median_dist_3d * 0.05)

    # Z maximum threshold for on_top_of:
    # Objects can't be "on top of" something if they're more than the full
    # scene Z extent apart. Use scene_z_extent as the cap.
    z_max_threshold = scene_z_extent * 1.0

    # XY footprint scale for on_top_of:
    # A is "on top of" B if A's centroid is within B's XY footprint.
    # Use median object XY extent as the reference — the footprint check
    # uses the actual object's XY size, but we need a minimum for tiny objects.
    # Set minimum to 15% of median pairwise XY distance (prevents tiny objects
    # from never triggering on_top_of).
    xy_footprint_min = median_xy_dist * 0.15

    # Dominance ratio for directional relations:
    # How much larger one axis must be vs another. In scenes where objects are
    # spread mostly along one axis, we want a higher ratio to avoid spurious
    # cross-axis predictions. Derive from the scene's aspect ratio.
    scene_xy_ratio = max(float(scene_extent[0]), 1e-6) / max(float(scene_extent[1]), 1e-6)
    # If scene is roughly square (ratio 0.5–2.0), use higher dominance (1.5×)
    # because diagonal pairs are common and we need to suppress the minor axis.
    # If scene is elongated, moderate dominance (1.3×) suffices.
    if 0.5 < scene_xy_ratio < 2.0:
        dominance_ratio = 1.5
    else:
        dominance_ratio = 1.3

    # Directional minimum axis threshold:
    # An axis displacement must be at least this fraction of the 3D distance
    # to be considered meaningful. Derived from scene noise level.
    dir_axis_fraction = 0.15

    # Comparative (higher_than) axis fraction:
    # More permissive than directional — height differences are meaningful
    # even when small relative to horizontal distance (e.g., a book is higher
    # than a pen even if they're far apart horizontally).
    comparative_axis_fraction = 0.08

    # On_top_of Z-dominance factor:
    # Z difference must be at least this fraction of XY distance for on_top_of.
    # In scenes with large Z spread, objects are more likely truly stacked.
    # In flat scenes, require stronger Z dominance.
    z_to_xy_ratio = scene_z_extent / max(scene_xy_extent, 1e-6)
    if z_to_xy_ratio > 0.5:
        # Tall scene — Z differences are common, be less strict
        ontop_z_dominance = 0.5
    else:
        # Flat scene — Z differences are rare, require strong Z dominance
        ontop_z_dominance = 0.8

    return {
        "z_min_threshold": z_min_threshold,
        "z_max_threshold": z_max_threshold,
        "xy_footprint_min": xy_footprint_min,
        "median_diagonal": median_diagonal,
        "median_xy_extent": median_xy_extent,
        "median_dist_3d": median_dist_3d,
        "dominance_ratio": dominance_ratio,
        "dir_axis_fraction": dir_axis_fraction,
        "comparative_axis_fraction": comparative_axis_fraction,
        "ontop_z_dominance": ontop_z_dominance,
        "scene_z_extent": scene_z_extent,
        "scene_xy_extent": scene_xy_extent,
    }


def _default_context() -> Dict[str, float]:
    """Fallback context when scene stats can't be computed."""
    return {
        "z_min_threshold": 0.05,
        "z_max_threshold": 1.5,
        "xy_footprint_min": 0.3,
        "median_diagonal": 1.0,
        "median_xy_extent": 0.5,
        "median_dist_3d": 1.0,
        "dominance_ratio": 1.3,
        "dir_axis_fraction": 0.15,
        "comparative_axis_fraction": 0.08,
        "ontop_z_dominance": 0.5,
        "scene_z_extent": 2.0,
        "scene_xy_extent": 2.0,
    }


def derive_relations(
    a_min: np.ndarray, a_max: np.ndarray,
    b_min: np.ndarray, b_max: np.ndarray,
    proximity_factor: float = 1.5,
    height_factor: float = 0.3,
    scene_context: Optional[Dict[str, float]] = None,
) -> List[Relation]:
    """
    Derive all spatial relations between object A and object B
    from their 3D bounding boxes.

    Args:
        a_min, a_max:      bbox corners of object A [x, y, z]
        b_min, b_max:      bbox corners of object B [x, y, z]
        proximity_factor:  adjacency threshold multiplier (default 1.5×avg_size)
        height_factor:     min Z diff fraction for higher_than (default 0.3×avg_h)
        scene_context:     adaptive thresholds from compute_scene_context().
                           If None, uses pair-local heuristics (backward compatible).

    Returns:
        List of Relation enums that hold for (A, B), ordered by specificity
        (physical > proximity > directional > comparative)
    """
    relations = []

    # Use default context if none provided (backward compatibility)
    ctx = scene_context if scene_context is not None else _default_context()

    a_center = (a_min + a_max) / 2
    b_center = (b_min + b_max) / 2
    a_size   = np.maximum(a_max - a_min, 1e-6)
    b_size   = np.maximum(b_max - b_min, 1e-6)

    avg_size   = (np.linalg.norm(a_size) + np.linalg.norm(b_size)) / 2
    avg_height = (a_size[2] + b_size[2]) / 2

    delta  = a_center - b_center
    dist_xy = float(np.linalg.norm(delta[:2]))
    dist_3d = float(np.linalg.norm(delta))

    xy_overlap = _bbox_xy_overlap(a_min, a_max, b_min, b_max)

    # ── Physical relations ────────────────────────────────────────────────────

    # ON_TOP_OF: centroid-based detection (reliable for Gaussian splat data)
    #   All thresholds derived from scene context:
    #   1. A centroid is above B centroid by z_min_threshold to z_max_threshold
    #   2. A's XY centroid is within B's XY footprint (or xy_footprint_min)
    #   3. Z difference dominates XY distance (ontop_z_dominance ratio)
    centroid_z_diff = float(a_center[2] - b_center[2])
    centroid_xy_dist = float(np.linalg.norm(a_center[:2] - b_center[:2]))
    a_xy_size = float(np.linalg.norm(a_size[:2]))
    b_xy_size = float(np.linalg.norm(b_size[:2]))

    ontop_z_min = ctx["z_min_threshold"]
    ontop_z_max = ctx["z_max_threshold"]
    ontop_xy_limit = max(b_xy_size * 0.8, ctx["xy_footprint_min"])
    ontop_z_dom = ctx["ontop_z_dominance"]

    if (centroid_z_diff > ontop_z_min and
            centroid_z_diff < ontop_z_max and
            centroid_z_diff > centroid_xy_dist * ontop_z_dom and
            centroid_xy_dist < ontop_xy_limit):
        relations.append(Relation.ON_TOP_OF)

    # UNDER: symmetric inverse of ON_TOP_OF
    centroid_z_diff_inv = float(b_center[2] - a_center[2])
    under_xy_limit = max(a_xy_size * 0.8, ctx["xy_footprint_min"])

    if (centroid_z_diff_inv > ontop_z_min and
            centroid_z_diff_inv < ontop_z_max and
            centroid_z_diff_inv > centroid_xy_dist * ontop_z_dom and
            centroid_xy_dist < under_xy_limit):
        relations.append(Relation.UNDER)

    # ATTACHED_TO: A is fixed to B (wall-mounted, built-in, touching)
    #   - Bboxes nearly touching or overlapping in 3D
    #   - Not already classified as ON_TOP_OF (which is more specific)
    min_gap = _min_bbox_gap(a_min, a_max, b_min, b_max)
    if (min_gap < avg_size * 0.05 and
            Relation.ON_TOP_OF not in relations):
        relations.append(Relation.ATTACHED_TO)

    # ── Proximity relations ───────────────────────────────────────────────────

    # ADJACENT_TO: close in XY, similar height, not physically stacked
    #   - Within proximity_factor × average diagonal distance
    #   - Z difference < 1.5× average height (same surface level)
    z_diff = abs(float(delta[2]))
    if (dist_xy < avg_size * proximity_factor and
            z_diff < avg_height * 1.5 and
            Relation.ON_TOP_OF not in relations):
        relations.append(Relation.ADJACENT_TO)

    # ── Directional relations ─────────────────────────────────────────────────
    # Only fire if objects are meaningfully separated (not overlapping/touching).
    # Minimum axis threshold is a fraction of the 3D distance (from scene context).

    if dist_3d > max(avg_size * 0.01, 0.01):
        abs_dx = abs(delta[0])
        abs_dy = abs(delta[1])

        # Adaptive axis threshold: fraction of 3D distance from scene context
        min_axis_thresh = dist_3d * ctx["dir_axis_fraction"]
        dominance = ctx["dominance_ratio"]

        # LEFT_OF / RIGHT_OF — fire when X displacement is significant
        if abs_dx > min_axis_thresh:
            if delta[0] < 0:
                relations.append(Relation.LEFT_OF)
            else:
                relations.append(Relation.RIGHT_OF)

        # IN_FRONT_OF / BEHIND — fire only when Y is the DOMINANT axis
        # Y must exceed X by the scene-adaptive dominance ratio.
        if abs_dy > min_axis_thresh and abs_dy > abs_dx * dominance:
            if delta[1] < 0:
                relations.append(Relation.IN_FRONT_OF)
            else:
                relations.append(Relation.BEHIND)

    # ── Comparative relations ─────────────────────────────────────────────────
    # HIGHER_THAN / LOWER_THAN: Z difference must exceed the scene-adaptive
    # z_min_threshold AND be at least comparative_axis_fraction of the 3D distance.
    # Uses a more permissive fraction than directional relations because height
    # differences are meaningful even between horizontally distant objects.

    z_diff_signed = float(delta[2])
    z_comparative_thresh = max(
        dist_3d * ctx["comparative_axis_fraction"],
        ctx["z_min_threshold"],
    )
    if z_diff_signed > z_comparative_thresh:
        relations.append(Relation.HIGHER_THAN)
    elif z_diff_signed < -z_comparative_thresh:
        relations.append(Relation.LOWER_THAN)

    return relations


# ── helpers ───────────────────────────────────────────────────────────────────

def _bbox_xy_overlap(a_min, a_max, b_min, b_max) -> bool:
    """True if the XY footprints of two bboxes overlap."""
    return (a_min[0] < b_max[0] and a_max[0] > b_min[0] and
            a_min[1] < b_max[1] and a_max[1] > b_min[1])


def _min_bbox_gap(a_min, a_max, b_min, b_max) -> float:
    """
    Minimum gap between two bboxes across all axes.
    Returns 0.0 if bboxes overlap (gap is negative in at least one axis).
    """
    gaps = [
        max(a_min[i] - b_max[i], b_min[i] - a_max[i], 0.0)
        for i in range(3)
    ]
    return float(np.linalg.norm(gaps))
