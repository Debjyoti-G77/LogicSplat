"""Tests for geometric relation inference."""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.graph.definitions import Object3D, SceneGraph
from src.logic.rules import infer_relations, check_support


def make_obj(uid, cx, cy, cz, sx=0.2, sy=0.2, sz=0.2, label="obj"):
    c = np.array([cx, cy, cz])
    return Object3D(
        uid=uid, centroid=c,
        bbox_min=c - np.array([sx/2, sy/2, sz/2]),
        bbox_max=c + np.array([sx/2, sy/2, sz/2]),
        color=np.array([128, 128, 128], dtype=np.uint8),
        point_count=100, label=label,
    )


def test_support_detected():
    table = make_obj(0, 0, 0, 0.5, sx=2, sy=2, sz=1.0, label="table")
    cup   = make_obj(1, 0, 0, 1.1, sx=0.1, sy=0.1, sz=0.2, label="cup")
    rels  = infer_relations([table, cup])
    rel_types = [(r.subject_id, r.relation, r.object_id) for r in rels]
    assert (1, "on_top_of", 0) in rel_types, f"Expected cup on_top_of table, got {rel_types}"
    print("✓ support detected")


def test_floating_object_no_support():
    table    = make_obj(0, 0, 0, 0.5, sx=2, sy=2, sz=1.0, label="table")
    floating = make_obj(1, 0, 0, 5.0, label="ball")
    rels = infer_relations([table, floating])
    support_rels = [r for r in rels if r.relation == "on_top_of" and r.subject_id == 1]
    assert len(support_rels) == 0, "Floating ball should have no support relation"
    print("✓ floating object has no support")


def test_adjacency():
    a = make_obj(0, 0, 0, 0)
    b = make_obj(1, 0.5, 0, 0)
    rels = infer_relations([a, b])
    adj = [r for r in rels if r.relation == "adjacent_to"]
    assert len(adj) > 0, "Close objects should be adjacent"
    print("✓ adjacency detected")


if __name__ == "__main__":
    test_support_detected()
    test_floating_object_no_support()
    test_adjacency()
    print("\n✓ All tests passed.")
