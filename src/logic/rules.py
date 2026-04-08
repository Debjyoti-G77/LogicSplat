"""Geometric relation inference between 3D objects."""
import numpy as np
from typing import List
from src.graph.definitions import Object3D, Relation, SceneGraph


# ── helpers ───────────────────────────────────────────────────────────────────

def _xy_dist(a: Object3D, b: Object3D) -> float:
    return float(np.linalg.norm(a.centroid[:2] - b.centroid[:2]))

def _bbox_xy_overlap(a: Object3D, b: Object3D, tol: float = 0.0) -> bool:
    return (
        a.bbox_min[0] - tol < b.bbox_max[0] + tol and
        a.bbox_max[0] + tol > b.bbox_min[0] - tol and
        a.bbox_min[1] - tol < b.bbox_max[1] + tol and
        a.bbox_max[1] + tol > b.bbox_min[1] - tol
    )

def _bbox_overlap_3d(a: Object3D, b: Object3D) -> bool:
    return (
        _bbox_xy_overlap(a, b) and
        a.bbox_min[2] < b.bbox_max[2] and
        a.bbox_max[2] > b.bbox_min[2]
    )

def _avg_footprint(a: Object3D, b: Object3D) -> float:
    """Average XY footprint size of two objects — used to scale adjacency threshold."""
    size_a = float(np.linalg.norm(a.size[:2]))
    size_b = float(np.linalg.norm(b.size[:2]))
    return (size_a + size_b) / 2.0


# ── relation checks ───────────────────────────────────────────────────────────

def check_support(upper: Object3D, lower: Object3D) -> bool:
    """
    upper is on_top_of lower when:
    - upper is not the table
    - upper centroid Z is meaningfully above lower centroid Z
      (at least 40% of lower's height — rules out side-by-side objects)
    - their XY footprints overlap with tolerance
    """
    if upper.label == "table":
        return False
    lower_height = max(lower.size[2], 0.01)
    z_centroid_diff = upper.centroid[2] - lower.centroid[2]
    # must be clearly above — not just noise or side-by-side
    if z_centroid_diff < lower_height * 0.4:
        return False
    tol = max(upper.size[0], upper.size[1]) * 0.3
    return _bbox_xy_overlap(upper, lower, tol=tol)


def check_containment(inner: Object3D, outer: Object3D) -> bool:
    """inner is fully enclosed by outer's bbox and is smaller."""
    return (
        inner.volume < outer.volume * 0.5 and
        inner.bbox_min[0] >= outer.bbox_min[0] and inner.bbox_max[0] <= outer.bbox_max[0] and
        inner.bbox_min[1] >= outer.bbox_min[1] and inner.bbox_max[1] <= outer.bbox_max[1] and
        inner.bbox_min[2] >= outer.bbox_min[2] and inner.bbox_max[2] <= outer.bbox_max[2]
    )


def check_occlusion(front: Object3D, back: Object3D) -> bool:
    """
    front occludes back when:
    - neither is the table (virtual plane)
    - front is meaningfully closer in Z (depth difference > 0.15)
    - their XY bboxes actually overlap
    - they do NOT have a support relation (support takes priority)
    """
    if front.label == "table" or back.label == "table":
        return False
    z_diff = back.centroid[2] - front.centroid[2]
    if z_diff <= 0.15:
        return False
    if check_support(back, front) or check_support(front, back):
        return False
    return _bbox_xy_overlap(front, back)


def check_adjacency(a: Object3D, b: Object3D) -> bool:
    """
    Two objects are adjacent when they sit at roughly the same height
    (both on the table surface) and are nearby in XY.
    Threshold: 3x the average of their XY footprint diagonals.
    Table is excluded — it's the support surface, not an adjacent peer.
    """
    if a.label == "table" or b.label == "table":
        return False
    z_diff = abs(a.centroid[2] - b.centroid[2])
    max_height = max(a.size[2], b.size[2])
    # objects at very different heights are stacked, not adjacent
    if z_diff > max_height * 2.0:
        return False
    xy_dist = _xy_dist(a, b)
    diag_a = float(np.linalg.norm(a.size[:2]))
    diag_b = float(np.linalg.norm(b.size[:2]))
    threshold = (diag_a + diag_b) * 3.0
    return xy_dist < threshold and not _bbox_overlap_3d(a, b)


# ── scene-level inference ─────────────────────────────────────────────────────

def infer_relations(objects: List[Object3D]) -> List[Relation]:
    """
    Infer all spatial relations between objects.
    Priority: support > containment > occlusion > adjacency
    (occlusion is skipped if support already exists between the pair)
    """
    relations: List[Relation] = []
    support_pairs = set()

    # pass 1: support and containment
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i == j:
                continue
            if check_support(a, b):
                relations.append(Relation(a.uid, "on_top_of", b.uid))
                support_pairs.add((a.uid, b.uid))
                support_pairs.add((b.uid, a.uid))
            if check_containment(a, b):
                relations.append(Relation(a.uid, "inside", b.uid))

    # pass 2: occlusion (skip if support pair)
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i == j:
                continue
            if (a.uid, b.uid) in support_pairs:
                continue
            if check_occlusion(a, b):
                relations.append(Relation(a.uid, "occludes", b.uid))

    # pass 3: adjacency (symmetric, skip if already support/containment)
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i], objects[j]
            if (a.uid, b.uid) in support_pairs:
                continue
            if check_adjacency(a, b):
                relations.append(Relation(a.uid, "adjacent_to", b.uid))

    return relations


def build_scene_graph(scene_id: str, objects: List[Object3D]) -> SceneGraph:
    relations = infer_relations(objects)
    return SceneGraph(scene_id=scene_id, objects=objects, relations=relations)
