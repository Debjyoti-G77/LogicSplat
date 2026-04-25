"""
Run trained RelationGNN on Gaussian Splat scenes.

Usage:
    python src/inference/predict.py --scene scene_01
    python src/inference/predict.py --scene scene_01 --model models/relation_gnn_gat_scannet_geometry.pt
"""
import sys
sys.path.insert(0, ".")

import os
import argparse
import torch
import numpy as np

from src.gaussian.loader import load_gaussian_ply, filter_gaussians
from src.gaussian.clustering import gaussian_to_objects
from src.logic.rules import infer_relations
from src.models.relation_gnn import RelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, RELATION_DESCRIPTIONS, Relation
from src.graph.definitions import Object3D, SceneGraph


DATA_DIR   = "data/processed"
MODEL_PATH = "models/relation_gnn_gat_scannet_geometry.pt"


def object_to_node_features(obj: Object3D, scene_extent: np.ndarray = None) -> torch.Tensor:
    """
    Convert a Gaussian-clustered Object3D into a 10-dim geometric feature vector
    matching the ScanNet training format.

    Features (10-dim):
        [0-2] centroid xyz (normalised by scene extent)
        [3-5] size xyz (normalised)
        [6]   volume (normalised)
        [7]   height_ratio
        [8]   point_density
        [9]   z_relative
    """
    size = np.maximum(obj.size, 1e-6)
    if scene_extent is None:
        scene_extent = np.array([10.0, 10.0, 3.0])
    norm = np.maximum(scene_extent, 1e-6)

    centroid_norm = obj.centroid / norm
    vol = float(np.prod(size))
    scene_vol = float(np.prod(norm))
    vol_norm = min(vol / max(scene_vol, 1e-9), 1.0)
    height_ratio = float(size[2] / max(size[0], size[1]))
    density = min(obj.point_count / max(vol, 1e-6), 1000.0) / 1000.0
    z_relative = float(obj.centroid[2] / norm[2])

    return torch.tensor([
        float(centroid_norm[0]), float(centroid_norm[1]), float(centroid_norm[2]),
        float(size[0] / norm[0]), float(size[1] / norm[1]), float(size[2] / norm[2]),
        vol_norm, height_ratio, density, z_relative,
    ], dtype=torch.float32)


def objects_to_edge_features(obj_a: Object3D, obj_b: Object3D,
                              scene_extent: np.ndarray = None) -> torch.Tensor:
    """
    8-dim geometric edge features matching ScanNet training format.
    """
    if scene_extent is None:
        scene_extent = np.array([10.0, 10.0, 3.0])
    norm = np.maximum(scene_extent, 1e-6)

    c_a, c_b = obj_a.centroid, obj_b.centroid
    min_a, max_a = obj_a.bbox_min, obj_a.bbox_max
    min_b, max_b = obj_b.bbox_min, obj_b.bbox_max
    size_a = np.maximum(max_a - min_a, 1e-6)
    size_b = np.maximum(max_b - min_b, 1e-6)

    delta_z = float((c_a[2] - c_b[2]) / norm[2])
    xy_dist = float(np.linalg.norm(c_a[:2] - c_b[:2]) / np.linalg.norm(norm[:2]))
    dist_3d = float(np.linalg.norm(c_a - c_b) / np.linalg.norm(norm))

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
        -3.0, 3.0) / 3.0)

    return torch.tensor([
        delta_z, xy_dist, dist_3d, bbox_overlap,
        vol_ratio, h_ratio, vert_gap, size_ratio_xy,
    ], dtype=torch.float32)


def build_full_graph(objects):
    """Build complete directed graph over all object pairs."""
    n = len(objects)
    # compute scene extent from all object bboxes
    all_pts = np.stack([o.bbox_min for o in objects] + [o.bbox_max for o in objects])
    scene_extent = np.maximum(all_pts.max(axis=0) - all_pts.min(axis=0), 1e-6)

    src, dst, edge_attrs = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            src.append(i)
            dst.append(j)
            edge_attrs.append(objects_to_edge_features(objects[i], objects[j], scene_extent))

    x = torch.stack([object_to_node_features(o, scene_extent) for o in objects])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.stack(edge_attrs)
    return x, edge_index, edge_attr


def predict_scene(scene_id: str, model: RelationGNN, threshold: float = 0.4):
    """Run full pipeline on a Gaussian Splat scene."""
    splat_path = os.path.join(DATA_DIR, scene_id, "splat.ply")
    if not os.path.exists(splat_path):
        print(f"splat.ply not found: {splat_path}")
        return

    cloud = load_gaussian_ply(splat_path)
    filtered = filter_gaussians(cloud, opacity_threshold=0.1)
    objects, params = gaussian_to_objects(filtered)

    print(f"\nScene: {scene_id}")
    print(f"Objects found: {len(objects)} | params: {params}")
    for o in objects:
        print(f"  Obj {o.uid} [{o.label}] pts={o.point_count} "
              f"centroid=[{o.centroid[0]:.2f},{o.centroid[1]:.2f},{o.centroid[2]:.2f}]")

    if len(objects) < 2:
        print("Not enough objects for relation inference.")
        return

    x, edge_index, edge_attr = build_full_graph(objects)

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, edge_attr)
        probs  = torch.softmax(logits, dim=-1)
        preds  = logits.argmax(dim=-1)

    print("\nPredicted Relations (GNN):")
    src_nodes = edge_index[0].tolist()
    dst_nodes = edge_index[1].tolist()
    seen = set()
    for s, d, pred, prob in zip(src_nodes, dst_nodes, preds.tolist(), probs):
        conf = float(prob[pred])
        if conf < threshold:
            continue
        key = (s, d, pred)
        if key in seen:
            continue
        seen.add(key)
        desc = RELATION_DESCRIPTIONS.get(Relation(pred), RELATION_NAMES[pred])
        print(f"  {objects[s].label} {desc} {objects[d].label}  (conf={conf:.2f})")

    print("\nGeometric Baseline (rules):")
    rule_rels = infer_relations(objects)
    for r in rule_rels:
        subj = next(o for o in objects if o.uid == r.subject_id)
        obj  = next(o for o in objects if o.uid == r.object_id)
        print(f"  {subj.label} --[{r.relation}]--> {obj.label}")


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
