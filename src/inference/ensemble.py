"""
Hybrid ensemble: Geometric Rules + GNN.

Rules are strong for vertical relations (on_top_of, under, inside).
GNN is stronger for directional/proximity relations (left_of, adjacent_to etc.)

Strategy:
- Run both systems on every object pair
- For relations where rules have high precision (vertical), trust rules
- For everything else, trust GNN predictions above confidence threshold
- Merge and deduplicate
"""
import sys
sys.path.insert(0, ".")

import os
import argparse
import torch
import numpy as np
from typing import List, Tuple

from src.colmap.loader import load_scene_points
from src.clustering.objects import cluster_to_objects
from src.logic.rules import infer_relations as geometric_infer
from src.models.relation_gnn import RelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, RELATION_DESCRIPTIONS, Relation
from src.graph.definitions import Object3D, Relation as GraphRelation
from src.inference.predict import (
    object_to_node_features,
    objects_to_edge_features,
    build_full_graph,
)

DATA_DIR   = "D:/logicsplat_data/processed"
MODEL_PATH = "models/relation_gnn_gat_edge_v2.pt"

# Relations where geometric rules are trusted over GNN
# These have strong geometric signals that rules capture well
RULE_DOMINANT = {
    Relation.ON_TOP_OF,
    Relation.UNDER,
    Relation.INSIDE,
    Relation.HANGING_FROM,
}


def run_ensemble(
    objects: List[Object3D],
    model: RelationGNN,
    gnn_threshold: float = 0.30,
    scene_id: str = "scene",
) -> List[dict]:
    """
    Run hybrid ensemble on a list of Object3D instances.

    Returns list of dicts:
        {subject_id, relation, object_id, confidence, source}
    """
    results = []
    seen = set()

    # ── Step 1: Geometric rules ───────────────────────────────────────────────
    from src.graph.definitions import SceneGraph
    scene = SceneGraph(scene_id=scene_id, objects=objects)
    rule_relations = geometric_infer(objects)

    for r in rule_relations:
        rel_enum = _map_rule_to_schema(r.relation)
        if rel_enum is None:
            continue
        key = (r.subject_id, r.object_id, rel_enum)
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "subject_id": r.subject_id,
            "relation":   rel_enum,
            "object_id":  r.object_id,
            "confidence": 1.0,
            "source":     "rules",
        })

    # ── Step 2: GNN predictions ───────────────────────────────────────────────
    if len(objects) < 2:
        return results

    x, edge_index, edge_attr = build_full_graph(objects)
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, edge_attr)
        probs  = torch.softmax(logits, dim=-1)
        preds  = logits.argmax(dim=-1)

    src_nodes = edge_index[0].tolist()
    dst_nodes = edge_index[1].tolist()

    for s, d, pred, prob in zip(src_nodes, dst_nodes, preds.tolist(), probs):
        conf = prob[pred].item()
        rel_enum = Relation(pred)

        if conf < gnn_threshold:
            continue

        # skip if rules already covered this pair with a dominant relation
        rule_covered = any(
            r["subject_id"] == s and r["object_id"] == d and r["relation"] in RULE_DOMINANT
            for r in results
        )
        if rule_covered and rel_enum in RULE_DOMINANT:
            continue

        key = (s, d, rel_enum)
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "subject_id": s,
            "relation":   rel_enum,
            "object_id":  d,
            "confidence": conf,
            "source":     "gnn",
        })

    return results


def _map_rule_to_schema(rel_name: str):
    """Map geometric rule relation names to our Relation enum."""
    mapping = {
        "on_top_of":   Relation.ON_TOP_OF,
        "inside":      Relation.INSIDE,
        "occludes":    Relation.OCCLUDES,
        "adjacent_to": Relation.ADJACENT_TO,
        "attached_to": Relation.ATTACHED_TO,
    }
    return mapping.get(rel_name)


def print_results(results: List[dict], objects: List[Object3D]):
    """Pretty print ensemble results."""
    rule_rels = [r for r in results if r["source"] == "rules"]
    gnn_rels  = [r for r in results if r["source"] == "gnn"]

    print(f"\nEnsemble Results: {len(results)} relations "
          f"({len(rule_rels)} from rules, {len(gnn_rels)} from GNN)")

    print("\n[Rules]")
    for r in sorted(rule_rels, key=lambda x: x["subject_id"]):
        desc = RELATION_DESCRIPTIONS.get(r["relation"], r["relation"].name)
        print(f"  Object_{r['subject_id']} {desc} Object_{r['object_id']}")

    print("\n[GNN]")
    for r in sorted(gnn_rels, key=lambda x: -x["confidence"]):
        desc = RELATION_DESCRIPTIONS.get(r["relation"], r["relation"].name)
        print(f"  Object_{r['subject_id']} {desc} Object_{r['object_id']}  "
              f"(conf={r['confidence']:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",     default="scene_01")
    parser.add_argument("--model",     default=MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=0.30)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(1)

    model = RelationGNN(node_feat_dim=8, edge_feat_dim=4, hidden_dim=128)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    print(f"Loaded: {args.model}")

    scene_path = os.path.join(DATA_DIR, args.scene)
    points, colors = load_scene_points(scene_path)
    objects, params = cluster_to_objects(points, colors, infer_table=True)

    print(f"\nScene: {args.scene} | {len(objects)} objects")
    for o in objects:
        print(f"  Obj {o.uid} [{o.label}] pts={o.point_count} z={o.centroid[2]:.2f}")

    results = run_ensemble(objects, model, gnn_threshold=args.threshold, scene_id=args.scene)
    print_results(results, objects)
