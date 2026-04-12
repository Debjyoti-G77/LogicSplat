"""
Run trained RelationGNN on your COLMAP scenes.

This is the bridge between:
  - Training: 3DSSG clean data
  - Testing: your sparse COLMAP reconstructions

Usage:
    python src/inference/predict.py --scene scene_01
"""
import sys
sys.path.insert(0, ".")

import os
import argparse
import torch
import numpy as np

from src.colmap.loader import load_scene_points
from src.clustering.objects import cluster_to_objects
from src.logic.rules import build_scene_graph as build_geometric_graph
from src.models.relation_gnn import RelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, RELATION_DESCRIPTIONS, Relation
from src.graph.definitions import Object3D


DATA_DIR   = "data/processed"
MODEL_PATH = "models/relation_gnn_gat_edge_v2.pt"


def object_to_node_features(obj: Object3D, num_classes: int = 500) -> torch.Tensor:
    """
    Convert a COLMAP Object3D into the same 8-dim feature vector
    used during 3DSSG training.

    Since COLMAP objects have no semantic labels, we use geometric
    proxies for the semantic features:
        [0] size_norm: normalized volume (proxy for class)
        [1] has_color: always 1 (we have RGB)
        [2] is_flat: height/width ratio < 0.3
        [3] is_tall: height/width ratio > 2.0
        [4] aff_place: always 1 (objects can have things placed on them)
        [5] aff_sit: 0 (unknown)
        [6] aff_hang: 0 (unknown)
        [7] aff_store: is_hollow proxy (large volume, few points)
    """
    size = obj.size
    vol = max(obj.volume, 1e-6)
    vol_norm = min(vol / 10.0, 1.0)  # normalize to ~0-1

    height_ratio = size[2] / max(max(size[0], size[1]), 1e-6)
    is_flat = 1.0 if height_ratio < 0.3 else 0.0
    is_tall = 1.0 if height_ratio > 2.0 else 0.0

    pts_per_vol = obj.point_count / vol
    is_hollow = 1.0 if pts_per_vol < 10 else 0.0

    return torch.tensor([
        vol_norm, 1.0, is_flat, is_tall,
        1.0, 0.0, 0.0, is_hollow,
    ], dtype=torch.float32)


def objects_to_edge_features(obj_a: Object3D, obj_b: Object3D) -> torch.Tensor:
    """
    Semantic edge features matching training format:
    [same_class, class_a_norm, class_b_norm, shared_affordance]
    Since we have no class labels, use size similarity as proxy.
    """
    vol_a = max(obj_a.volume, 1e-6)
    vol_b = max(obj_b.volume, 1e-6)
    size_sim = min(vol_a, vol_b) / max(vol_a, vol_b)  # 1.0 = same size
    size_a_norm = min(vol_a / 10.0, 1.0)
    size_b_norm = min(vol_b / 10.0, 1.0)
    same_size = 1.0 if size_sim > 0.8 else 0.0

    return torch.tensor([same_size, size_a_norm, size_b_norm, size_sim], dtype=torch.float32)


def build_full_graph(objects):
    """Build complete directed graph over all object pairs."""
    n = len(objects)
    src, dst, edge_attrs = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            src.append(i)
            dst.append(j)
            edge_attrs.append(objects_to_edge_features(objects[i], objects[j]))

    x = torch.stack([object_to_node_features(o) for o in objects])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.stack(edge_attrs)
    return x, edge_index, edge_attr


def predict_scene(scene_id: str, model: RelationGNN, threshold: float = 0.5):
    """
    Run full pipeline on a COLMAP scene and predict relations using GNN.
    """
    scene_path = os.path.join(DATA_DIR, scene_id)
    points, colors = load_scene_points(scene_path)
    objects, params = cluster_to_objects(points, colors, infer_table=True)

    print(f"\nScene: {scene_id}")
    print(f"Objects found: {len(objects)} | params: {params}")
    for o in objects:
        print(f"  Obj {o.uid} [{o.label}] pts={o.point_count} "
              f"z={o.centroid[2]:.2f} size={o.size[0]:.2f}x{o.size[1]:.2f}x{o.size[2]:.2f}")

    if len(objects) < 2:
        print("Not enough objects for relation inference.")
        return

    x, edge_index, edge_attr = build_full_graph(objects)

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, edge_attr)
        probs  = torch.softmax(logits, dim=-1)
        preds  = logits.argmax(dim=-1)

    # collect predicted relations
    print("\nPredicted Relations (GNN):")
    src_nodes, dst_nodes = edge_index[0].tolist(), edge_index[1].tolist()
    seen = set()
    for idx, (s, d, pred, prob) in enumerate(zip(src_nodes, dst_nodes, preds.tolist(), probs)):
        conf = prob[pred].item()
        rel_name = RELATION_NAMES[pred]
        if conf < threshold:
            continue
        key = (s, d, pred)
        if key in seen:
            continue
        seen.add(key)
        desc = RELATION_DESCRIPTIONS.get(Relation(pred), rel_name)
        print(f"  Object_{s} {desc} Object_{d}  (conf={conf:.2f})")

    # also show geometric baseline for comparison
    print("\nGeometric Baseline (rules):")
    graph = build_geometric_graph(scene_id, objects)
    for r in graph.relations:
        print(f"  {r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scene_01")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Run training first: python src/training/train.py")
        sys.exit(1)

    model = RelationGNN(node_feat_dim=8, edge_feat_dim=4, hidden_dim=128)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    print(f"Loaded model: {args.model}")

    predict_scene(args.scene, model, threshold=args.threshold)
