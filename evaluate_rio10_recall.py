"""
Evaluate Predicate Recall@K on RIO10 test scenes.

Computes Predicate Recall@3 and Recall@5 matching ReLaGS's evaluation protocol.

For each RIO10 test scene:
  1. Load the Gaussian splat (from D:\\3rscan_test\\{scene_id}\\splat\\point_cloud.ply)
  2. Transfer instance labels (from D:\\3rscan_test\\{scene_id}\\labels.instances.annotated.v2.ply)
  3. Group Gaussians by instance, extract features (same as build_3rscan_graphs.py)
  4. Run GNN inference → get sigmoid scores for all 12 relations per edge
  5. Load GT relations from relationships.json for this scene
  6. For each GT relation triple: check if the GT predicate is in the model's top-K predictions

Reports:
  - Predicate Recall@3 and Recall@5
  - Per-relation breakdown
  - Comparison table format matching ReLaGS Table 1

Usage:
    python evaluate_rio10_recall.py
    python evaluate_rio10_recall.py --model models/relation_gnn_v8_3rscan_rio10_excluded.pt
    python evaluate_rio10_recall.py --top-k 3 5 10
"""

import argparse
import json
import os
import sys
import warnings
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from src.gaussian.loader import load_gaussian_ply
from src.gaussian.clustering import extract_gaussian_node_features
from src.graph.definitions import Object3D
from src.models.relation_gnn import RelationGNN
from src.relations.schema import (
    DSSG_TO_SCHEMA, NUM_RELATIONS, RELATION_NAMES, Relation,
)
from scripts.build_3rscan_graphs import (
    extract_3rscan_edge_features,
    build_object3d_from_gaussians,
)


# === Configuration ===
INPUT_DIR = Path(r"D:\3rscan_test")
RIO10_TEST_SCENES = Path(__file__).parent / "data" / "3DSSG" / "rio10_test_scenes.txt"
RELATIONSHIPS_JSON = Path(__file__).parent / "data" / "3DSSG" / "relationships.json"
OBJECTS_JSON = Path(__file__).parent / "data" / "3DSSG" / "objects.json"
DEFAULT_MODEL = Path(__file__).parent / "models" / "geokan_relation_v1.pt"
DEFAULT_THRESHOLDS = Path(__file__).parent / "models" / "geokan_relation_v1_thresholds.json"

MIN_GAUSSIANS_PER_INSTANCE = 3


def load_test_scene_ids() -> list:
    """Load the 46 RIO10 test scene IDs (handles UTF-16 BOM)."""
    if not RIO10_TEST_SCENES.exists():
        print(f"ERROR: {RIO10_TEST_SCENES} not found!")
        sys.exit(1)
    raw = RIO10_TEST_SCENES.read_bytes()
    if raw[:2] == b'\xff\xfe':
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig")
    return [line.strip() for line in text.splitlines() if line.strip()]


def load_3dssg_annotations():
    """Load 3DSSG objects and relationships indexed by scan_id."""
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


def load_instance_labels(scene_dir: Path) -> np.ndarray:
    """
    Load instance labels for a scene.
    Tries instance_labels.npz first, then labels.instances.annotated.v2.ply.
    """
    # Try npz format first
    npz_path = scene_dir / "instance_labels.npz"
    if npz_path.exists():
        npz = np.load(str(npz_path))
        return npz["labels"]

    # Try PLY format (from label transfer)
    ply_path = scene_dir / "labels.instances.annotated.v2.ply"
    if ply_path.exists():
        from plyfile import PlyData
        ply = PlyData.read(str(ply_path))
        vertex = ply['vertex']
        if 'instance_id' in [p.name for p in vertex.properties]:
            return np.array(vertex['instance_id'])
        elif 'label' in [p.name for p in vertex.properties]:
            return np.array(vertex['label'])

    return None


def build_scene_graph(cloud, labels, scene_id):
    """
    Build graph tensors from a Gaussian cloud + instance labels.
    Returns (x, edge_index, edge_attr, instance_id_to_idx) or None.
    """
    if labels is None or len(labels) != cloud.num_gaussians:
        return None

    # Group Gaussians by instance ID (skip -1 = unlabeled)
    unique_instances = sorted(set(labels[labels >= 0].tolist()))

    if len(unique_instances) < 2:
        return None

    # Build Object3D for each instance
    objects = []
    instance_id_to_idx = {}

    for inst_id in unique_instances:
        mask = labels == inst_id
        inst_xyz = cloud.xyz[mask]
        inst_rgb = cloud.rgb[mask]
        inst_opacity = cloud.opacity[mask]
        inst_cov = cloud.covariance[mask]

        if len(inst_xyz) < MIN_GAUSSIANS_PER_INSTANCE:
            continue

        obj = build_object3d_from_gaussians(
            instance_id=int(inst_id),
            xyz=inst_xyz,
            rgb=inst_rgb,
            opacity=inst_opacity,
            covariance=inst_cov,
        )
        instance_id_to_idx[int(inst_id)] = len(objects)
        objects.append(obj)

    if len(objects) < 2:
        return None

    # Compute scene extent
    all_labeled_mask = labels >= 0
    all_xyz = cloud.xyz[all_labeled_mask]
    scene_min = all_xyz.min(axis=0)
    scene_max = all_xyz.max(axis=0)
    scene_extent = np.maximum(scene_max - scene_min, 1e-6)

    # Extract node features (10-dim)
    node_features = []
    for obj in objects:
        feat = extract_gaussian_node_features(obj, scene_extent, scene_min)
        node_features.append(feat)
    x = np.stack(node_features)

    # Build ALL directed edges + extract edge features (17-dim)
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

    edge_index = np.array([src_list, dst_list])
    edge_attr = np.stack(edge_feats)

    return {
        "x": torch.tensor(x, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
        "edge_attr": torch.tensor(edge_attr, dtype=torch.float32),
        "instance_id_to_idx": instance_id_to_idx,
        "n_objects": n_objects,
    }


def get_gt_triples(scene_id, rels_by_scan, instance_id_to_idx):
    """
    Get ground-truth relation triples for a scene that are representable
    (both subject and object have been mapped to graph indices).

    Returns list of (src_idx, dst_idx, relation_idx) tuples.
    """
    gt_rels = rels_by_scan.get(scene_id, [])
    triples = []

    for rel in gt_rels:
        if len(rel) < 4:
            continue
        subj_id = int(rel[0])
        obj_id = int(rel[1])
        rel_name = rel[3] if isinstance(rel[3], str) else str(rel[3])

        if rel_name not in DSSG_TO_SCHEMA:
            continue

        relation_idx = int(DSSG_TO_SCHEMA[rel_name])
        subj_idx = instance_id_to_idx.get(subj_id)
        obj_idx = instance_id_to_idx.get(obj_id)

        if subj_idx is None or obj_idx is None:
            continue

        triples.append((subj_idx, obj_idx, relation_idx))

    return triples


def compute_recall_at_k(model, device, scene_ids, input_dir, rels_by_scan,
                        top_k_values=(3, 5)):
    """
    Compute Predicate Recall@K across all scenes.

    For each GT triple (subj, obj, predicate):
      - Get the model's sigmoid scores for the edge (subj→obj)
      - Check if the GT predicate is in the top-K predicted predicates

    Returns dict with recall values and per-relation breakdown.
    """
    model.eval()

    # Accumulators
    total_gt_triples = 0
    hits_at_k = {k: 0 for k in top_k_values}
    per_relation_total = defaultdict(int)
    per_relation_hits = {k: defaultdict(int) for k in top_k_values}
    scenes_processed = 0
    scenes_skipped = 0

    for scene_id in scene_ids:
        scene_dir = input_dir / scene_id
        splat_path = scene_dir / "splat" / "point_cloud.ply"

        if not splat_path.exists():
            # Try alternative paths
            alt_path = scene_dir / "ckpts" / "point_cloud_30000.ply"
            if alt_path.exists():
                splat_path = alt_path
            else:
                scenes_skipped += 1
                continue

        # Load Gaussian splat
        try:
            cloud = load_gaussian_ply(str(splat_path))
        except Exception as e:
            warnings.warn(f"Failed to load splat {scene_id}: {e}")
            scenes_skipped += 1
            continue

        # Load instance labels
        labels = load_instance_labels(scene_dir)
        if labels is None:
            scenes_skipped += 1
            continue

        # Build graph
        graph_data = build_scene_graph(cloud, labels, scene_id)
        if graph_data is None:
            scenes_skipped += 1
            continue

        instance_id_to_idx = graph_data["instance_id_to_idx"]

        # Get GT triples for this scene
        gt_triples = get_gt_triples(scene_id, rels_by_scan, instance_id_to_idx)
        if not gt_triples:
            scenes_skipped += 1
            continue

        # Run GNN inference
        x = graph_data["x"].to(device)
        edge_index = graph_data["edge_index"].to(device)
        edge_attr = graph_data["edge_attr"].to(device)

        with torch.no_grad():
            logits = model(x, edge_index, edge_attr)
            probs = torch.sigmoid(logits).cpu()  # (E, 12)

        # Build edge position lookup
        n_objects = graph_data["n_objects"]
        edge_to_pos = {}
        src_arr = graph_data["edge_index"][0].numpy()
        dst_arr = graph_data["edge_index"][1].numpy()
        for pos in range(len(src_arr)):
            edge_to_pos[(int(src_arr[pos]), int(dst_arr[pos]))] = pos

        # Evaluate each GT triple
        for subj_idx, obj_idx, gt_rel_idx in gt_triples:
            edge_pos = edge_to_pos.get((subj_idx, obj_idx))
            if edge_pos is None:
                continue

            total_gt_triples += 1
            per_relation_total[gt_rel_idx] += 1

            # Get model's scores for this edge
            edge_scores = probs[edge_pos]  # (12,)

            # Get top-K predicted relation indices
            for k in top_k_values:
                top_k_indices = torch.topk(edge_scores, min(k, NUM_RELATIONS)).indices.tolist()
                if gt_rel_idx in top_k_indices:
                    hits_at_k[k] += 1
                    per_relation_hits[k][gt_rel_idx] += 1

        scenes_processed += 1

    # Compute recall values
    results = {
        "scenes_processed": scenes_processed,
        "scenes_skipped": scenes_skipped,
        "total_gt_triples": total_gt_triples,
        "recall": {},
        "per_relation": {},
    }

    for k in top_k_values:
        recall = hits_at_k[k] / max(total_gt_triples, 1)
        results["recall"][k] = recall

    for rel_idx in range(NUM_RELATIONS):
        rel_name = RELATION_NAMES[rel_idx]
        total = per_relation_total[rel_idx]
        results["per_relation"][rel_name] = {
            "total": total,
        }
        for k in top_k_values:
            hits = per_relation_hits[k][rel_idx]
            recall = hits / max(total, 1) if total > 0 else 0.0
            results["per_relation"][rel_name][f"recall@{k}"] = recall

    return results


def print_results(results, top_k_values):
    """Print results in a table format matching ReLaGS Table 1."""
    print(f"\n{'='*70}")
    print("PREDICATE RECALL EVALUATION (RIO10 Test Set)")
    print(f"{'='*70}")
    print(f"  Scenes processed: {results['scenes_processed']}")
    print(f"  Scenes skipped:   {results['scenes_skipped']}")
    print(f"  Total GT triples: {results['total_gt_triples']}")

    # Overall recall
    print(f"\n{'─'*70}")
    print("OVERALL PREDICATE RECALL")
    print(f"{'─'*70}")
    for k in top_k_values:
        recall = results["recall"].get(k, 0.0)
        print(f"  Recall@{k}: {recall:.4f} ({recall*100:.1f}%)")

    # Per-relation breakdown
    print(f"\n{'─'*70}")
    print("PER-RELATION BREAKDOWN")
    print(f"{'─'*70}")

    # Header
    k_headers = "".join(f"  R@{k:>2}" for k in top_k_values)
    print(f"  {'Relation':<20} {'Total':>6}{k_headers}")
    print(f"  {'-'*20} {'-'*6}" + "  -----" * len(top_k_values))

    for rel_idx in range(NUM_RELATIONS):
        rel_name = RELATION_NAMES[rel_idx]
        info = results["per_relation"].get(rel_name, {})
        total = info.get("total", 0)
        recalls = []
        for k in top_k_values:
            r = info.get(f"recall@{k}", 0.0)
            recalls.append(f"  {r:.3f}")
        recalls_str = "".join(recalls)
        print(f"  {rel_name:<20} {total:>6}{recalls_str}")

    # Comparison table (ReLaGS format)
    print(f"\n{'─'*70}")
    print("COMPARISON TABLE (ReLaGS Table 1 format)")
    print(f"{'─'*70}")
    print(f"  {'Method':<25} {'R@3':>8} {'R@5':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8}")

    r3 = results["recall"].get(3, 0.0)
    r5 = results["recall"].get(5, 0.0)
    print(f"  {'LogicSplat (ours)':<25} {r3*100:>7.1f}% {r5*100:>7.1f}%")
    # Reference values from ReLaGS paper (approximate)
    print(f"  {'ReLaGS (reported)':<25} {'--':>8} {'--':>8}")
    print(f"  {'3DSSG (reported)':<25} {'--':>8} {'--':>8}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Predicate Recall@K on RIO10 test scenes"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained GNN model (.pt)")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Path to thresholds JSON (unused for recall, kept for API)")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Override input directory for test splats")
    parser.add_argument("--top-k", type=int, nargs="+", default=[3, 5],
                        help="Top-K values to evaluate (default: 3 5)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else INPUT_DIR
    model_path = Path(args.model) if args.model else DEFAULT_MODEL
    device = args.device if torch.cuda.is_available() else "cpu"
    top_k_values = sorted(args.top_k)

    print("=" * 70)
    print("RIO10 Predicate Recall@K Evaluation")
    print("=" * 70)
    print(f"  Model:      {model_path}")
    print(f"  Input dir:  {input_dir}")
    print(f"  Device:     {device}")
    print(f"  Top-K:      {top_k_values}")
    print()

    # Load model
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    # Detect model type from filename
    if "geokan" in str(model_path).lower():
        sys.path.insert(0, str(Path(__file__).parent))
        from geokan_relation import GeoKANRelationGNN
        model = GeoKANRelationGNN(
            node_feat_dim=10,
            edge_feat_dim=17,
            hidden_dim=128,
            num_relations=NUM_RELATIONS,
            dropout=0.2,
        ).to(device)
    else:
        model = RelationGNN(
            node_feat_dim=10,
            edge_feat_dim=17,
            hidden_dim=256,
            num_relations=NUM_RELATIONS,
            dropout=0.3,
        ).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load scene IDs
    scene_ids = load_test_scene_ids()
    print(f"  RIO10 test scenes: {len(scene_ids)}")

    # Load 3DSSG annotations
    print("  Loading 3DSSG annotations...")
    _, rels_by_scan = load_3dssg_annotations()
    print(f"  Annotations loaded for {len(rels_by_scan)} scenes")

    # Check how many test scenes have annotations
    annotated = [s for s in scene_ids if s in rels_by_scan]
    print(f"  Test scenes with GT annotations: {len(annotated)}/{len(scene_ids)}")

    # Run evaluation
    print(f"\n{'─'*70}")
    print("Running inference on test scenes...")
    print(f"{'─'*70}")

    results = compute_recall_at_k(
        model=model,
        device=device,
        scene_ids=scene_ids,
        input_dir=input_dir,
        rels_by_scan=rels_by_scan,
        top_k_values=top_k_values,
    )

    # Print results
    print_results(results, top_k_values)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {output_path}")
    else:
        # Default save location
        output_path = Path("results") / "rio10_recall_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
