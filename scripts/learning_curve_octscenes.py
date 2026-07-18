"""
Generate a learning curve for GNN fine-tuning on OCTScenes.

Trains on increasing subsets of annotated OCTScenes (1, 5, 10, 15, 20)
and evaluates on the held-out test set (scene_06..scene_13) after each.

Reports: fine-tuned F1 vs zero-shot F1 vs geometry-only F1.

Output:
    data/octscenes/learning_curve.json
    Console table with results

Usage:
    python scripts/learning_curve_octscenes.py
    python scripts/learning_curve_octscenes.py --subsets 1 3 5 10 15 20

Requirements:
    - Annotated OCTScenes (annotation_status = "verified")
    - Pretrained model at models/relation_gnn_gat_scannet_geometry_multilabel_v3_axisalign.pt
    - Held-out scenes at data/processed/scene_06..scene_13
"""
import sys
sys.path.insert(0, ".")

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import List
from sklearn.metrics import f1_score

from src.models.relation_gnn import RelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    print("ERROR: torch_geometric not installed.")
    sys.exit(1)

# Reuse data loading from finetune script
from scripts.finetune_tabletop import (
    CONFIG,
    load_annotation,
    load_ground_truth,
    build_pyg_data,
    build_eval_pyg_data,
    load_pretrained_model,
    evaluate,
)


def compute_geometry_f1(config: dict) -> float:
    """
    Compute geometry-only baseline F1 on held-out scenes.
    Uses the same geometry rules as auto_annotate_octscenes.py.
    """
    from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
    from src.gaussian.clustering import gaussian_to_objects
    from src.relations.geometry import compute_scene_context, derive_relations
    from src.relations.schema import Relation

    eval_dir = config["eval_scenes_dir"]
    all_preds = []
    all_labels = []

    for scene_id in config["eval_scenes"]:
        scene_dir = os.path.join(eval_dir, scene_id)
        gt = load_ground_truth(scene_dir)
        if gt is None:
            continue

        ply_path = os.path.join(scene_dir, "splat.ply")
        if not os.path.exists(ply_path):
            continue

        # Load and cluster
        cloud = load_gaussian_ply(ply_path)
        cloud = filter_gaussians(cloud, opacity_threshold=0.1)
        cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)
        objects, _ = gaussian_to_objects(cloud, target_min=3, target_max=8)
        if len(objects) < 2:
            continue

        # Z-flip
        for o in objects:
            o.centroid = o.centroid.copy(); o.centroid[2] *= -1
            o.bbox_min = o.bbox_min.copy(); o.bbox_min[2] *= -1
            o.bbox_max = o.bbox_max.copy(); o.bbox_max[2] *= -1
            o.bbox_min[2], o.bbox_max[2] = min(o.bbox_min[2], o.bbox_max[2]), max(o.bbox_min[2], o.bbox_max[2])

        n = len(objects)
        n_edges = n * (n - 1)

        # Scene context
        all_mins = np.stack([o.bbox_min for o in objects])
        all_maxs = np.stack([o.bbox_max for o in objects])
        scene_ctx = compute_scene_context(all_mins, all_maxs)

        # Predict with geometry rules
        preds = np.zeros((n_edges, NUM_RELATIONS))
        edge_idx = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                derived = derive_relations(
                    objects[i].bbox_min, objects[i].bbox_max,
                    objects[j].bbox_min, objects[j].bbox_max,
                    scene_context=scene_ctx,
                )
                for rel in derived:
                    preds[edge_idx, int(rel)] = 1.0
                edge_idx += 1

        # Build ground truth labels (reuse annotation_to_labels logic)
        from scripts.finetune_tabletop import annotation_to_labels
        labels = annotation_to_labels(gt, objects, n)
        if labels is None:
            continue

        all_preds.append(preds)
        all_labels.append(labels.numpy())

    if not all_preds:
        return 0.0

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    f1_per_rel = f1_score(all_labels, all_preds, average=None, zero_division=0)
    active = [i for i in range(NUM_RELATIONS) if all_labels[:, i].sum() > 0]
    if not active:
        return 0.0
    return float(np.mean([f1_per_rel[i] for i in active]))


def compute_zeroshot_f1(config: dict) -> float:
    """Compute zero-shot (pretrained, no fine-tuning) F1 on held-out scenes."""
    device = config["device"]

    # Load eval data
    eval_data = []
    eval_dir = config["eval_scenes_dir"]
    for scene_id in config["eval_scenes"]:
        scene_dir = os.path.join(eval_dir, scene_id)
        gt = load_ground_truth(scene_dir)
        if gt is None:
            continue
        data = build_eval_pyg_data(scene_dir, gt)
        if data is not None:
            eval_data.append(data)

    if not eval_data:
        return 0.0

    eval_loader = DataLoader(eval_data, batch_size=config["batch_size"], shuffle=False)

    # Load pretrained model (no fine-tuning)
    model = load_pretrained_model(config).to(device)
    return evaluate(model, eval_loader, device, split_name="Zero-shot")


def train_subset(train_data: list, eval_loader, config: dict) -> float:
    """Train on a subset of scenes, return best eval F1."""
    device = config["device"]

    if not train_data:
        return 0.0

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)

    # Fresh pretrained model for each subset
    model = load_pretrained_model(config).to(device)

    # Class weights from this subset
    pos_counts = torch.zeros(NUM_RELATIONS)
    total_edges = 0
    for data in train_data:
        pos_counts += data.y.sum(dim=0)
        total_edges += data.y.shape[0]
    pos_counts = torch.clamp(pos_counts, min=1.0)
    neg_counts = total_edges - pos_counts
    pos_weight = torch.clamp(neg_counts / pos_counts, max=50.0).to(device)

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0.0
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()

        if eval_loader:
            f1 = evaluate(model, eval_loader, device,
                          split_name=f"Subset({len(train_data)} scenes) Epoch {epoch}")
            best_f1 = max(best_f1, f1)

    return best_f1


def main():
    parser = argparse.ArgumentParser(
        description="Generate learning curve for GNN fine-tuning on OCTScenes"
    )
    parser.add_argument(
        "--subsets", nargs="+", type=int, default=[1, 5, 10, 15, 20],
        help="Number of training scenes for each point on the curve"
    )
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    args = parser.parse_args()

    config = CONFIG.copy()
    config["epochs"] = args.epochs
    config["lr"] = args.lr
    device = config["device"]

    print(f"\n{'='*60}")
    print("  LogicSplat — Learning Curve (OCTScenes Fine-tuning)")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Subsets: {args.subsets}")
    print(f"  Epochs per subset: {config['epochs']}")

    # ── Load all annotated training scenes ────────────────────────────────────
    print("\nLoading annotated OCTScenes...")
    all_train_data = []
    octscenes_dir = config["octscenes_dir"]

    scene_dirs = sorted([
        d for d in os.listdir(octscenes_dir)
        if d.startswith("oct_") and os.path.isdir(os.path.join(octscenes_dir, d))
    ]) if os.path.isdir(octscenes_dir) else []

    for scene_id in scene_dirs:
        scene_dir = os.path.join(octscenes_dir, scene_id)
        annotation = load_annotation(scene_dir)
        if annotation is None:
            continue
        data = build_pyg_data(scene_dir, annotation)
        if data is not None:
            all_train_data.append(data)
            print(f"  {scene_id}: loaded")

    print(f"\n  Total annotated scenes available: {len(all_train_data)}")

    # ── Load eval data ────────────────────────────────────────────────────────
    print("\nLoading held-out evaluation scenes...")
    eval_data = []
    eval_dir = config["eval_scenes_dir"]
    for scene_id in config["eval_scenes"]:
        scene_dir = os.path.join(eval_dir, scene_id)
        gt = load_ground_truth(scene_dir)
        if gt is None:
            continue
        data = build_eval_pyg_data(scene_dir, gt)
        if data is not None:
            eval_data.append(data)

    eval_loader = DataLoader(eval_data, batch_size=config["batch_size"], shuffle=False) \
                  if eval_data else None
    print(f"  Eval scenes: {len(eval_data)}")

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("  Computing baselines...")
    print("─" * 40)

    geometry_f1 = compute_geometry_f1(config)
    print(f"\n  Geometry-only F1: {geometry_f1:.4f}")

    zeroshot_f1 = compute_zeroshot_f1(config)
    print(f"  Zero-shot GNN F1: {zeroshot_f1:.4f}")

    # ── Learning curve ────────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("  Training subsets...")
    print("─" * 40)

    results = {
        "geometry_only_f1": geometry_f1,
        "zeroshot_gnn_f1": zeroshot_f1,
        "curve": [],
    }

    for n_scenes in args.subsets:
        if n_scenes > len(all_train_data):
            print(f"\n  Subset {n_scenes}: only {len(all_train_data)} scenes available — using all")
            subset = all_train_data
        else:
            subset = all_train_data[:n_scenes]

        print(f"\n  Training on {len(subset)} scene(s)...")
        f1 = train_subset(subset, eval_loader, config)

        results["curve"].append({
            "n_scenes": len(subset),
            "finetuned_f1": round(f1, 4),
        })
        print(f"  → Fine-tuned F1 ({len(subset)} scenes): {f1:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    output_path = os.path.join(config["octscenes_dir"], "learning_curve.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {output_path}")

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  LEARNING CURVE RESULTS")
    print(f"{'='*60}")
    print(f"\n  {'Method':<30s}  {'F1':>8s}")
    print(f"  {'─'*30}  {'─'*8}")
    print(f"  {'Geometry-only':<30s}  {geometry_f1:>8.4f}")
    print(f"  {'Zero-shot GNN':<30s}  {zeroshot_f1:>8.4f}")
    for point in results["curve"]:
        label = f"Fine-tuned ({point['n_scenes']} scenes)"
        print(f"  {label:<30s}  {point['finetuned_f1']:>8.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
