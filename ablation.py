"""
Ablation table — TASK 5.

Loads all trained models from models/ folder, evaluates each on the
3DSSG test set, and prints a comparison table.

Usage:
    python ablation.py
    python ablation.py --models_dir models --data_dir data/3DSSG
"""
import sys
sys.path.insert(0, ".")

import os
import argparse
import glob
import torch
import numpy as np
from collections import Counter
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from src.dataset.loader_3dssg import SceneGraphDataset3DSSG
from src.models.relation_gnn import RelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES


# ── known model configs ───────────────────────────────────────────────────────
# Maps model filename stem → (node_feat_dim, edge_feat_dim, notes)
# Add entries here when new model variants are trained.
MODEL_CONFIGS = {
    # original SAGEConv semantic baseline (no edge features)
    "relation_gnn_sage_semantic_baseline": (8, 4, "SAGEConv semantic baseline"),
    # GATConv + edge features, first proper test split
    "relation_gnn_gat_edge_v2":            (8, 4, "GATConv + edge features"),
    # GATConv + LR warmup
    "relation_gnn_gat_geo_gpu_b8":         (8, 4, "GATConv + LR warmup"),
    # ScanNet geometry (10-dim nodes, 8-dim edges)
    "relation_gnn_gat_scannet_geometry":   (10, 8, "+ ScanNet geometry"),
    # ScanNet geometry fallback (padded 3DSSG features)
    "relation_gnn_gat_scannet_geometry_3dssg_padded": (10, 8, "+ ScanNet geo (3DSSG padded)"),
}

# Fallback: auto-detect feature dims from model weights
def _infer_dims_from_weights(state_dict: dict):
    """Infer node_feat_dim and edge_feat_dim from saved weight shapes."""
    node_feat_dim = None
    edge_feat_dim = None
    for key, tensor in state_dict.items():
        if "node_encoder.0.weight" in key:
            node_feat_dim = tensor.shape[1]
        if "edge_classifier.0.weight" in key:
            # edge_input_dim = 2 * hidden_dim + edge_feat_dim
            # hidden_dim inferred from node_encoder output
            break
    # infer edge_feat_dim from edge_classifier input
    for key, tensor in state_dict.items():
        if "node_encoder.0.weight" in key:
            hidden_dim = tensor.shape[0]
        if "edge_classifier.0.weight" in key:
            edge_input_dim = tensor.shape[1]
            if node_feat_dim is not None:
                edge_feat_dim = edge_input_dim - 2 * hidden_dim
            break
    return node_feat_dim, edge_feat_dim


# ── dataset helpers ───────────────────────────────────────────────────────────

class PyGDataset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.data = [subset[i] for i in range(len(subset))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        g = self.data[idx]
        return Data(
            x=g["x"],
            edge_index=g["edge_index"],
            edge_attr=g["edge_attr"],
            y=g["edge_label"],
        )


def _pad_graph(g: dict, node_dim: int, edge_dim: int) -> dict:
    """Zero-pad node/edge features to match model input dims."""
    import torch.nn.functional as F
    x = g["x"]
    e = g["edge_attr"]
    if x.shape[1] < node_dim:
        x = F.pad(x, (0, node_dim - x.shape[1]))
    if e.shape[1] < edge_dim:
        e = F.pad(e, (0, edge_dim - e.shape[1]))
    return {**g, "x": x, "edge_attr": e}


class PaddedPyGDataset(torch.utils.data.Dataset):
    def __init__(self, subset, node_dim: int, edge_dim: int):
        self.data = [
            _pad_graph(subset[i], node_dim, edge_dim)
            for i in range(len(subset))
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        g = self.data[idx]
        return Data(
            x=g["x"],
            edge_index=g["edge_index"],
            edge_attr=g["edge_attr"],
            y=g["edge_label"],
        )


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    model: RelationGNN,
    loader: DataLoader,
    device: str,
) -> dict:
    """
    Evaluate model on a DataLoader.
    Returns dict with macro_f1 and per-relation F1 scores.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            preds = logits.argmax(dim=-1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch.y.cpu().tolist())

    present = sorted(set(all_labels))
    per_class_f1 = f1_score(
        all_labels, all_preds, labels=present, average=None, zero_division=0
    )
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    per_relation = {}
    for i, cls_idx in enumerate(present):
        per_relation[RELATION_NAMES[cls_idx]] = float(per_class_f1[i])

    return {"macro_f1": float(macro_f1), "per_relation": per_relation}


# ── main ──────────────────────────────────────────────────────────────────────

def run_ablation(
    models_dir: str = "models",
    data_dir: str = "data/3DSSG",
    batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Device: {device}")
    print(f"Loading 3DSSG dataset from {data_dir}...")

    full_dataset = SceneGraphDataset3DSSG(data_dir=data_dir)

    # reproducible 70/15/15 split — same seed as train.py
    test_size  = int(len(full_dataset) * 0.15)
    val_size   = int(len(full_dataset) * 0.15)
    train_size = len(full_dataset) - val_size - test_size
    train_raw, val_raw, test_raw = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Split — Train: {train_size} | Val: {val_size} | Test: {test_size}\n")

    # discover model files
    model_files = sorted(glob.glob(os.path.join(models_dir, "relation_gnn_*.pt")))
    if not model_files:
        print(f"No model files found in {models_dir}/")
        print("Train a model first: python src/training/train.py")
        return

    results = []

    for model_path in model_files:
        stem = os.path.splitext(os.path.basename(model_path))[0]
        print(f"Evaluating: {stem}")

        # load weights
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            continue

        # determine feature dims
        if stem in MODEL_CONFIGS:
            node_dim, edge_dim, notes = MODEL_CONFIGS[stem]
        else:
            node_dim, edge_dim = _infer_dims_from_weights(state_dict)
            if node_dim is None or edge_dim is None:
                print(f"  ✗ Could not infer feature dims — skipping")
                continue
            notes = "unknown variant"
            print(f"  Auto-detected dims: node={node_dim}, edge={edge_dim}")

        # build model
        try:
            model = RelationGNN(
                node_feat_dim=node_dim,
                edge_feat_dim=edge_dim,
                hidden_dim=128,
                num_relations=NUM_RELATIONS,
                dropout=0.0,  # no dropout at eval time
            ).to(device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"  ✗ Model load failed: {e}")
            continue

        # build padded datasets for this model's dims
        val_ds  = PaddedPyGDataset(val_raw,  node_dim, edge_dim)
        test_ds = PaddedPyGDataset(test_raw, node_dim, edge_dim)
        val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        val_metrics  = evaluate_model(model, val_loader,  device)
        test_metrics = evaluate_model(model, test_loader, device)

        results.append({
            "name":      stem.replace("relation_gnn_", ""),
            "notes":     notes,
            "val_f1":    val_metrics["macro_f1"],
            "test_f1":   test_metrics["macro_f1"],
            "val_per":   val_metrics["per_relation"],
            "test_per":  test_metrics["per_relation"],
            "node_dim":  node_dim,
            "edge_dim":  edge_dim,
        })
        print(f"  Val F1: {val_metrics['macro_f1']:.4f}  "
              f"Test F1: {test_metrics['macro_f1']:.4f}")

    if not results:
        print("No models evaluated successfully.")
        return

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ABLATION TABLE")
    print("=" * 80)
    header = f"{'Model':<40} {'Val F1':>8} {'Test F1':>9}  Notes"
    print(header)
    print("-" * 80)

    # sort by test F1 descending
    results.sort(key=lambda r: r["test_f1"], reverse=True)
    for r in results:
        name = r["name"][:39]
        print(f"{name:<40} {r['val_f1']:>8.4f} {r['test_f1']:>9.4f}  {r['notes']}")

    print("=" * 80)

    # ── per-relation breakdown for best model ─────────────────────────────────
    best = results[0]
    print(f"\nPer-relation F1 for best model: {best['name']}")
    print(f"{'Relation':<22} {'Val F1':>8} {'Test F1':>9}")
    print("-" * 42)
    all_rels = sorted(
        set(best["val_per"]) | set(best["test_per"])
    )
    for rel in all_rels:
        v = best["val_per"].get(rel, 0.0)
        t = best["test_per"].get(rel, 0.0)
        print(f"  {rel:<20} {v:>8.3f} {t:>9.3f}")

    # ── directional relations highlight ───────────────────────────────────────
    directional = ["left_of", "right_of", "in_front_of", "behind"]
    print(f"\nDirectional relations (key metric for geometry upgrade):")
    print(f"{'Relation':<22}", end="")
    for r in results:
        print(f"  {r['name'][:12]:>12}", end="")
    print()
    print("-" * (22 + 14 * len(results)))
    for rel in directional:
        print(f"  {rel:<20}", end="")
        for r in results:
            v = r["test_per"].get(rel, float("nan"))
            if np.isnan(v):
                print(f"  {'—':>12}", end="")
            else:
                print(f"  {v:>12.3f}", end="")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LogicSplat ablation table")
    parser.add_argument("--models_dir", default="models",
                        help="Directory containing .pt model files")
    parser.add_argument("--data_dir", default="data/3DSSG",
                        help="Path to 3DSSG data directory")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_ablation(
        models_dir=args.models_dir,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=args.device,
    )
