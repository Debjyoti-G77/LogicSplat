"""
Fine-tune the RelationGNN on manually-annotated OCTScenes tabletop data.

Run this AFTER manually filling in the annotation_template.json files
for all 20 OCTScenes.

Strategy:
    - Load pretrained model (ScanNet-trained, v3 axisalign)
    - Freeze all parameters EXCEPT edge_classifier
    - Train on annotated OCTScenes (oct_01 to oct_20)
    - Evaluate on held-out scenes (scene_06 to scene_13) after each epoch
    - Save best model based on held-out macro F1

Usage:
    python scripts/finetune_tabletop.py
    python scripts/finetune_tabletop.py --epochs 50 --lr 5e-5
    python scripts/finetune_tabletop.py --eval-only  # just evaluate current model

Requirements:
    - Annotated OCTScenes at data/octscenes/oct_XX/annotation_template.json
    - Pretrained model at models/relation_gnn_gat_scannet_geometry_multilabel_v3_axisalign.pt
    - Held-out scenes at data/processed/scene_06..scene_13 with ground_truth_relations.json
"""
import sys
sys.path.insert(0, ".")

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from sklearn.metrics import f1_score

from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
from src.gaussian.clustering import (
    gaussian_to_objects,
    extract_gaussian_node_features,
    extract_gaussian_edge_features,
)
from src.models.relation_gnn import RelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, Relation
from src.graph.definitions import Object3D

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
except ImportError:
    print("ERROR: torch_geometric not installed.")
    print("  Install with: pip install torch-geometric")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it


# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "pretrained_model": "models/relation_gnn_v7_dualhead.pt",
    "save_path":        "models/relation_gnn_finetuned_tabletop.pt",
    "octscenes_dir":    "D:/logicsplat_data/octscenes",
    "eval_scenes_dir":  "D:/logicsplat_data/processed",
    "eval_scenes":      [f"scene_{i:02d}" for i in range(6, 14)],  # scene_06..scene_13
    "node_feat_dim":    10,
    "edge_feat_dim":    17,
    "hidden_dim":       256,
    "dropout":          0.3,
    "lr":               1e-4,
    "weight_decay":     1e-4,
    "epochs":           30,
    "batch_size":       4,
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
}

# Map annotation relation names to our schema indices
RELATION_NAME_TO_IDX = {
    "on_top_of":        Relation.ON_TOP_OF,
    "under":            Relation.UNDER,
    "inside":           Relation.INSIDE,
    "attached_to":      Relation.ATTACHED_TO,
    "hanging_from":     Relation.HANGING_FROM,
    "adjacent_to":      Relation.ADJACENT_TO,
    "to_the_left_of":   Relation.LEFT_OF,
    "left_of":          Relation.LEFT_OF,
    "to_the_right_of":  Relation.RIGHT_OF,
    "right_of":         Relation.RIGHT_OF,
    "in_front_of":      Relation.IN_FRONT_OF,
    "behind":           Relation.BEHIND,
    "higher_than":      Relation.HIGHER_THAN,
    "lower_than":       Relation.LOWER_THAN,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_annotation(scene_dir: str) -> Optional[dict]:
    """Load a completed annotation_template.json."""
    path = os.path.join(scene_dir, "annotation_template.json")
    if not os.path.exists(path):
        return None

    with open(path) as f:
        data = json.load(f)

    # Check if annotation is complete (no FILL_IN names)
    for obj in data.get("objects", []):
        if obj.get("name", "FILL_IN") == "FILL_IN":
            return None  # not yet annotated

    if not data.get("relations"):
        return None  # no relations annotated

    return data


def load_ground_truth(scene_dir: str) -> Optional[dict]:
    """Load ground_truth_relations.json for eval scenes."""
    path = os.path.join(scene_dir, "ground_truth_relations.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_graph_from_splat(scene_dir: str) -> Optional[Tuple[List[Object3D], np.ndarray, np.ndarray]]:
    """
    Load splat.ply, cluster, Z-flip, extract features.
    Returns (objects, node_features, edge_features) or None on failure.
    """
    ply_path = os.path.join(scene_dir, "splat.ply")
    if not os.path.exists(ply_path):
        return None

    cloud = load_gaussian_ply(ply_path)
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)

    objects, params = gaussian_to_objects(cloud, target_min=3, target_max=8)
    if len(objects) < 2:
        return None

    # Z-flip
    for o in objects:
        o.centroid = o.centroid.copy()
        o.centroid[2] *= -1
        o.bbox_min = o.bbox_min.copy()
        o.bbox_min[2] *= -1
        o.bbox_max = o.bbox_max.copy()
        o.bbox_max[2] *= -1
        o.bbox_min[2], o.bbox_max[2] = min(o.bbox_min[2], o.bbox_max[2]), max(o.bbox_min[2], o.bbox_max[2])

    # Scene extent
    all_mins = np.stack([o.bbox_min for o in objects])
    all_maxs = np.stack([o.bbox_max for o in objects])
    scene_min = all_mins.min(axis=0)
    scene_max = all_maxs.max(axis=0)
    scene_extent = np.maximum(scene_max - scene_min, 1e-6)

    # Node features
    x = np.stack([
        extract_gaussian_node_features(o, scene_extent, scene_min)
        for o in objects
    ])

    # Edge features (all directed pairs)
    edge_feats = []
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i != j:
                edge_feats.append(extract_gaussian_edge_features(a, b, scene_extent))

    edge_feats = np.stack(edge_feats)
    return objects, x, edge_feats


def annotation_to_labels(
    annotation: dict,
    objects: List[Object3D],
    n_objects: int,
) -> Optional[torch.Tensor]:
    """
    Convert annotation JSON relations to multi-hot edge labels.

    Returns tensor of shape (n_edges, NUM_RELATIONS) where n_edges = n*(n-1).
    """
    n_edges = n_objects * (n_objects - 1)
    labels = torch.zeros(n_edges, NUM_RELATIONS)

    # Build name→id mapping from annotation
    name_to_id = {}
    for obj in annotation["objects"]:
        name_to_id[obj["name"]] = obj["id"]

    for rel in annotation["relations"]:
        subj_name = rel["subject"]
        obj_name = rel["object"]
        rel_name = rel["relation"]

        subj_id = name_to_id.get(subj_name)
        obj_id = name_to_id.get(obj_name)
        rel_idx = RELATION_NAME_TO_IDX.get(rel_name)

        if subj_id is None or obj_id is None or rel_idx is None:
            continue
        if subj_id >= n_objects or obj_id >= n_objects:
            continue

        # Edge index in flattened all-pairs: for pair (i,j) where i!=j
        # edge_idx = i * (n_objects - 1) + (j if j < i else j - 1)
        edge_idx = subj_id * (n_objects - 1) + (obj_id if obj_id < subj_id else obj_id - 1)
        labels[edge_idx, int(rel_idx)] = 1.0

    return labels


def build_pyg_data(scene_dir: str, annotation: dict) -> Optional[Data]:
    """Build a PyG Data object from a scene's splat + annotation."""
    result = build_graph_from_splat(scene_dir)
    if result is None:
        return None

    objects, x, edge_feats = result
    n = len(objects)

    # Build edge_index (all directed pairs)
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    edge_attr = torch.tensor(edge_feats, dtype=torch.float32)

    # Build labels
    labels = annotation_to_labels(annotation, objects, n)
    if labels is None:
        return None

    return Data(
        x=x_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels,
    )


def build_eval_pyg_data(scene_dir: str, gt: dict) -> Optional[Data]:
    """Build a PyG Data object from an eval scene's splat + ground truth."""
    result = build_graph_from_splat(scene_dir)
    if result is None:
        return None

    objects, x, edge_feats = result
    n = len(objects)

    # Build edge_index
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    edge_attr = torch.tensor(edge_feats, dtype=torch.float32)

    # Build labels from ground truth (same format as annotation)
    labels = annotation_to_labels(gt, objects, n)
    if labels is None:
        return None

    return Data(
        x=x_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def load_pretrained_model(config: dict) -> RelationGNN:
    """Load pretrained model and freeze all except edge_classifier."""
    model = RelationGNN(
        node_feat_dim=config["node_feat_dim"],
        edge_feat_dim=config["edge_feat_dim"],
        hidden_dim=config["hidden_dim"],
        num_relations=NUM_RELATIONS,
        dropout=config["dropout"],
    )

    model_path = config["pretrained_model"]
    if not os.path.exists(model_path):
        print(f"ERROR: Pretrained model not found: {model_path}")
        sys.exit(1)

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    print(f"  Loaded pretrained model: {model_path}")

    # Freeze all except the two heads and contact encoder
    frozen_count = 0
    trainable_count = 0
    trainable_parts = ("head_directional", "head_contact", "contact_encoder")
    for name, param in model.named_parameters():
        if any(part in name for part in trainable_parts):
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    print(f"  Frozen parameters: {frozen_count:,}")
    print(f"  Trainable parameters (heads + contact_encoder): {trainable_count:,}")

    return model


def evaluate(model: RelationGNN, loader: DataLoader, device: str,
             split_name: str = "eval") -> float:
    """Evaluate model, return macro F1."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            preds = (torch.sigmoid(logits) >= 0.5).float().cpu()
            all_preds.append(preds)
            all_labels.append(batch.y.cpu())

    if not all_preds:
        return 0.0

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    f1_per_rel = f1_score(all_labels, all_preds, average=None, zero_division=0)
    # Only compute macro over relations that have at least one positive sample
    active = [i for i in range(NUM_RELATIONS) if all_labels[:, i].sum() > 0]
    if not active:
        return 0.0

    macro_f1 = float(np.mean([f1_per_rel[i] for i in active]))

    print(f"\n  {split_name} — Macro F1: {macro_f1:.4f} (over {len(active)} active relations)")
    for i in range(NUM_RELATIONS):
        pos = int(all_labels[:, i].sum())
        if pos > 0:
            print(f"    {RELATION_NAMES[i]:20s}  F1={f1_per_rel[i]:.3f}  pos={pos}")

    return macro_f1


def train(config: dict):
    """Main fine-tuning loop."""
    device = config["device"]
    print(f"\n{'='*60}")
    print("  LogicSplat — Fine-tune on OCTScenes")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  LR: {config['lr']}")
    print(f"  Pretrained: {config['pretrained_model']}")

    # ── Load training data (annotated OCTScenes) ──────────────────────────────
    print("\nLoading annotated OCTScenes...")
    train_data = []
    octscenes_dir = config["octscenes_dir"]

    if not os.path.isdir(octscenes_dir):
        print(f"ERROR: OCTScenes directory not found: {octscenes_dir}")
        print("  Run: python scripts/download_octscenes.py")
        sys.exit(1)

    scene_dirs = sorted([
        d for d in os.listdir(octscenes_dir)
        if d.startswith("oct_") and os.path.isdir(os.path.join(octscenes_dir, d))
    ])

    for scene_id in scene_dirs:
        scene_dir = os.path.join(octscenes_dir, scene_id)
        annotation = load_annotation(scene_dir)
        if annotation is None:
            print(f"  {scene_id}: not annotated — skipping")
            continue

        data = build_pyg_data(scene_dir, annotation)
        if data is None:
            print(f"  {scene_id}: failed to build graph — skipping")
            continue

        train_data.append(data)
        print(f"  {scene_id}: {data.x.shape[0]} objects, "
              f"{data.edge_index.shape[1]} edges, "
              f"{int(data.y.sum())} positive labels")

    if not train_data:
        print("\nERROR: No annotated scenes found.")
        print("  Fill in annotation_template.json files first.")
        print("  Templates at: data/octscenes/oct_XX/annotation_template.json")
        sys.exit(1)

    print(f"\n  Training scenes: {len(train_data)}")

    # ── Load eval data (held-out scenes 06-13) ────────────────────────────────
    print("\nLoading held-out evaluation scenes...")
    eval_data = []
    eval_dir = config["eval_scenes_dir"]

    for scene_id in config["eval_scenes"]:
        scene_dir = os.path.join(eval_dir, scene_id)
        gt = load_ground_truth(scene_dir)
        if gt is None:
            print(f"  {scene_id}: no ground truth — skipping")
            continue

        data = build_eval_pyg_data(scene_dir, gt)
        if data is None:
            print(f"  {scene_id}: failed to build graph — skipping")
            continue

        eval_data.append(data)
        print(f"  {scene_id}: {data.x.shape[0]} objects, "
              f"{int(data.y.sum())} positive labels")

    print(f"\n  Eval scenes: {len(eval_data)}")

    # ── Build data loaders ────────────────────────────────────────────────────
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=config["batch_size"], shuffle=False) \
                  if eval_data else None

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading pretrained model...")
    model = load_pretrained_model(config).to(device)

    # ── Compute class weights ─────────────────────────────────────────────────
    pos_counts = torch.zeros(NUM_RELATIONS)
    total_edges = 0
    for data in train_data:
        pos_counts += data.y.sum(dim=0)
        total_edges += data.y.shape[0]

    pos_counts = torch.clamp(pos_counts, min=1.0)
    neg_counts = total_edges - pos_counts
    pos_weight = torch.clamp(neg_counts / pos_counts, max=50.0).to(device)

    print("\n  Class weights:")
    for i in range(NUM_RELATIONS):
        if pos_counts[i] > 1:
            print(f"    {RELATION_NAMES[i]:20s}  weight={pos_weight[i]:.1f}  "
                  f"pos={int(pos_counts[i])}")

    # ── Optimizer (only edge_classifier params) ───────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Training loop ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
    best_f1 = 0.0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"\n  Epoch {epoch:02d}/{config['epochs']} — Loss: {avg_loss:.4f}")

        # Evaluate every epoch (small dataset, fast eval)
        if eval_loader:
            macro_f1 = evaluate(model, eval_loader, device, split_name="Held-out")
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                torch.save(model.state_dict(), config["save_path"])
                print(f"    [SAVED] New best model (F1={best_f1:.4f})")
        else:
            # No eval data — save based on training loss
            if epoch == config["epochs"]:
                torch.save(model.state_dict(), config["save_path"])
                print(f"    [SAVED] Final model (no eval data available)")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Fine-tuning Complete")
    print(f"{'='*60}")
    print(f"  Best held-out Macro F1: {best_f1:.4f}")
    print(f"  Model saved to: {config['save_path']}")
    print(f"\n  To use the fine-tuned model for inference:")
    print(f"    python src/inference/gaussian_inference.py --scene scene_XX "
          f"--model {config['save_path']}")


def eval_only(config: dict):
    """Just evaluate the fine-tuned model on held-out scenes."""
    device = config["device"]
    model_path = config["save_path"]

    if not os.path.exists(model_path):
        print(f"ERROR: Fine-tuned model not found: {model_path}")
        print("  Run fine-tuning first: python scripts/finetune_tabletop.py")
        sys.exit(1)

    print(f"\nLoading fine-tuned model: {model_path}")
    model = RelationGNN(
        node_feat_dim=config["node_feat_dim"],
        edge_feat_dim=config["edge_feat_dim"],
        hidden_dim=config["hidden_dim"],
        num_relations=NUM_RELATIONS,
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Load eval scenes
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
        print("No eval scenes available.")
        return

    eval_loader = DataLoader(eval_data, batch_size=config["batch_size"], shuffle=False)
    evaluate(model, eval_loader, device, split_name="Held-out (eval-only)")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune RelationGNN on annotated OCTScenes"
    )
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate the fine-tuned model")
    args = parser.parse_args()

    config = CONFIG.copy()
    config["epochs"] = args.epochs
    config["lr"] = args.lr
    config["batch_size"] = args.batch_size

    if args.eval_only:
        eval_only(config)
    else:
        train(config)


if __name__ == "__main__":
    main()
