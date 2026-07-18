"""
Build PyG-compatible TEST graphs from instance-labeled Gaussians + 3DSSG annotations.

Same logic as build_3rscan_graphs.py but for the 46 RIO10 test scenes only.
These graphs are used for evaluation (not training).

For each test scene with instance_labels.npz:
  1. Load Gaussian splat + instance labels
  2. Group Gaussians by instance ID (skip -1/unlabeled)
  3. For each instance group, compute Object3D-like representation
  4. Extract 10-dim node features + 17-dim edge features
  5. Load 3DSSG relations and build multi-hot edge labels
  6. Save to data/3rscan_test_graph_cache/{scene_id}.pt

Usage:
    python scripts/build_3rscan_test_graphs.py
    python scripts/build_3rscan_test_graphs.py --max-scenes 10
    python scripts/build_3rscan_test_graphs.py --force
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
from src.relations.schema import DSSG_TO_SCHEMA, NUM_RELATIONS, Relation, RELATION_NAMES

# Import shared functions from build_3rscan_graphs
from scripts.build_3rscan_graphs import (
    build_object3d_from_gaussians,
    extract_3rscan_edge_features,
    build_support_groups,
    are_same_support_level,
    DIRECTIONAL_RELATIONS,
    SCHEMA_INVERSE_PAIRS,
    MIN_INSTANCES,
    MIN_COVERAGE,
)


# === Configuration ===
INPUT_DIR = Path(r"D:\3rscan_test")
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "3rscan_test_graph_cache"
RELATIONSHIPS_JSON = Path(__file__).parent.parent / "data" / "3DSSG" / "relationships.json"
OBJECTS_JSON = Path(__file__).parent.parent / "data" / "3DSSG" / "objects.json"
RIO10_TEST_SCENES = Path(__file__).parent.parent / "data" / "3DSSG" / "rio10_test_scenes.txt"


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


def load_3dssg_annotations() -> tuple:
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


def process_test_scene(
    scene_id: str,
    input_dir: Path,
    gt_relationships: List[list],
    gt_objects: List[dict],
) -> Optional[Dict]:
    """
    Process a single test scene: load splat + instance labels → build graph.

    Directory structure:
        input_dir/{scene_id}/splat/point_cloud.ply
        input_dir/{scene_id}/instance_labels.npz

    Returns:
        Graph dict or None if processing fails.
    """
    scene_dir = input_dir / scene_id
    splat_ply = scene_dir / "splat" / "point_cloud.ply"
    labels_npz = scene_dir / "instance_labels.npz"

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

    # Extract node features (10-dim)
    node_features = []
    for obj in objects:
        feat = extract_gaussian_node_features(obj, scene_extent, scene_min)
        node_features.append(feat)
    x = np.stack(node_features)  # (N, 10)

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

    edge_index = np.array([src_list, dst_list])  # (2, E)
    edge_attr = np.stack(edge_feats)  # (E, 17)

    # Build multi-hot edge labels from 3DSSG annotations
    edge_to_pos = {}
    for pos, (i, j) in enumerate(zip(src_list, dst_list)):
        edge_to_pos[(i, j)] = pos

    n_edges = len(src_list)
    edge_label = np.zeros((n_edges, NUM_RELATIONS), dtype=np.float32)

    # Assign labels from 3DSSG
    for rel in gt_relationships:
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

        edge_pos = edge_to_pos.get((subj_idx, obj_idx))
        if edge_pos is not None:
            edge_label[edge_pos, relation_idx] = 1.0

    # Inject inverse relation labels (same as training)
    for rel in gt_relationships:
        if len(rel) < 4:
            continue

        subj_id = int(rel[0])
        obj_id = int(rel[1])
        rel_name = rel[3] if isinstance(rel[3], str) else str(rel[3])

        if rel_name not in DSSG_TO_SCHEMA:
            continue

        relation_idx = int(DSSG_TO_SCHEMA[rel_name])

        if relation_idx not in SCHEMA_INVERSE_PAIRS:
            continue

        inverse_idx = SCHEMA_INVERSE_PAIRS[relation_idx]
        subj_idx = instance_id_to_idx.get(subj_id)
        obj_idx = instance_id_to_idx.get(obj_id)

        if subj_idx is None or obj_idx is None:
            continue

        reverse_edge_pos = edge_to_pos.get((obj_idx, subj_idx))
        if reverse_edge_pos is not None:
            edge_label[reverse_edge_pos, inverse_idx] = 1.0

    # Mask directional supervision to same-support pairs only
    support_groups = build_support_groups(gt_relationships, instance_id_to_idx)

    for edge_pos, (src_idx, dst_idx) in enumerate(zip(src_list, dst_list)):
        same_support = are_same_support_level(src_idx, dst_idx, support_groups)

        for rel_idx in DIRECTIONAL_RELATIONS:
            if not same_support and edge_label[edge_pos, rel_idx] == 0:
                edge_label[edge_pos, rel_idx] = -1.0  # ignore in evaluation

    # Build object label list
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
        description="Build PyG-compatible TEST graphs for RIO10 test scenes"
    )
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Limit number of scenes")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if .pt already exists")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Override input directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else INPUT_DIR
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Build 3RScan TEST Graph Cache (RIO10 Test Scenes)")
    print("=" * 70)
    print(f"  Input dir:   {input_dir}")
    print(f"  Output dir:  {output_dir}")
    print()

    # Load 3DSSG annotations
    print("[1/4] Loading 3DSSG annotations...")
    objects_by_scan, rels_by_scan = load_3dssg_annotations()
    print(f"  Objects: {len(objects_by_scan)} scenes")
    print(f"  Relations: {len(rels_by_scan)} scenes")

    # Load test scene IDs
    print("\n[2/4] Loading test scene IDs...")
    scene_ids = load_test_scene_ids()
    if args.max_scenes:
        scene_ids = scene_ids[:args.max_scenes]
    print(f"  Test scenes: {len(scene_ids)}")

    # Check which scenes have annotations
    scenes_with_annotations = [s for s in scene_ids if s in rels_by_scan]
    print(f"  With 3DSSG annotations: {len(scenes_with_annotations)}")

    # Process scenes
    print(f"\n[3/4] Building test graph tensors...")
    start_time = time.time()

    n_processed = 0
    n_skipped_exists = 0
    n_skipped_missing = 0
    n_no_labels = 0
    n_failed = 0
    positive_edges = 0
    total_edges = 0

    for i, scene_id in enumerate(scenes_with_annotations):
        output_path = output_dir / f"{scene_id}.pt"

        # Resume support — validate existing file integrity
        if output_path.exists() and not args.force:
            try:
                # Verify the .pt is not corrupt
                test_graph = torch.load(str(output_path), weights_only=False)
                assert "x" in test_graph and "edge_label" in test_graph
                n_skipped_exists += 1
                continue
            except Exception:
                # Corrupt file from interrupted write — delete and redo
                output_path.unlink(missing_ok=True)

        gt_rels = rels_by_scan.get(scene_id, [])
        gt_objects = objects_by_scan.get(scene_id, [])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph = process_test_scene(scene_id, input_dir, gt_rels, gt_objects)

        if graph is None:
            n_skipped_missing += 1
            continue

        # Check for positive labels
        n_positive = int(graph["edge_label"].sum().item())
        if n_positive == 0:
            n_no_labels += 1
            continue

        # Save atomically — write to temp then rename
        try:
            tmp_path = output_path.with_suffix(".pt.tmp")
            torch.save(graph, str(tmp_path))
            # Atomic rename
            if output_path.exists():
                output_path.unlink()
            tmp_path.rename(output_path)
            n_processed += 1
            positive_edges += n_positive
            total_edges += graph["edge_label"].shape[0]

            n_nodes = graph["x"].shape[0]
            n_e = graph["edge_label"].shape[0]
            print(
                f"  [{i+1}/{len(scenes_with_annotations)}] {scene_id}: "
                f"{n_nodes} nodes, {n_e} edges, {n_positive} positive labels"
            )
        except Exception as e:
            n_failed += 1
            print(f"  ERROR saving {scene_id}: {e}")

    elapsed = time.time() - start_time

    # Summary
    print(f"\n[4/4] Summary")
    print("=" * 70)
    print(f"  Test scenes with annotations: {len(scenes_with_annotations)}")
    print(f"  Processed (new):              {n_processed}")
    print(f"  Skipped (already exists):     {n_skipped_exists}")
    print(f"  Skipped (missing splat/labels):{n_skipped_missing}")
    print(f"  Skipped (no positive rels):   {n_no_labels}")
    print(f"  Failed:                       {n_failed}")
    print(f"  Time:                         {elapsed:.1f}s")

    if n_processed > 0:
        print(f"\n  Graph stats:")
        print(f"    Total edges:    {total_edges}")
        print(f"    Positive edges: {positive_edges}")
        print(f"    Positive rate:  {positive_edges/max(total_edges,1)*100:.1f}%")

    # Print relation distribution
    all_cached = sorted(output_dir.glob("*.pt"))
    if all_cached:
        print(f"\n  Relation distribution across {len(all_cached)} test graphs:")
        pos_counts = torch.zeros(NUM_RELATIONS)
        total_e = 0
        for pt_path in all_cached:
            try:
                g = torch.load(str(pt_path), weights_only=False)
                # Only count non-ignored labels
                labels = g["edge_label"]
                pos_counts += (labels > 0).float().sum(dim=0)
                total_e += labels.shape[0]
            except Exception:
                continue
        if total_e > 0:
            max_count = pos_counts.max().item()
            for idx in range(NUM_RELATIONS):
                count = int(pos_counts[idx].item())
                bar = "#" * int(30 * count / max(max_count, 1))
                name = RELATION_NAMES.get(idx, f"rel_{idx}")
                print(f"    {name:20s} {count:6d}  {bar}")

    print(f"\n  Output directory: {output_dir}")
    print(f"  Total cached test graphs: {len(all_cached)}")


if __name__ == "__main__":
    main()
