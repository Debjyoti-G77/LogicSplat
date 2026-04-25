"""
Train RelationGNN on ScanNet geometry — TASK 2.

Upgrades from semantic-only 3DSSG features to real 3D geometry from ScanNet:
  node_feat_dim: 8  → 10  (adds centroid, size, volume, density, z_relative)
  edge_feat_dim: 4  → 8   (adds delta_z, xy_dist, overlap, vol_ratio, ...)

Expected improvement: directional relations (left_of, right_of, in_front_of,
behind) should jump from F1~0.15 to F1~0.5+ because the model now has actual
3D positions rather than semantic proxies.

Usage:
    python src/training/train_scannet.py
"""
import sys
sys.path.insert(0, ".")

import os
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from src.dataset.loader_scannet import SceneGraphDatasetScanNet
from src.models.relation_gnn import RelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES


# ── config ────────────────────────────────────────────────────────────────────

CONFIG = {
    "model_name":    "gat_scannet_geometry",
    "node_feat_dim": 10,   # geometric node features (up from 8)
    "edge_feat_dim": 10,   # 8 + delta_x + delta_y (fixes directional relations)
    "hidden_dim":    128,
    "dropout":       0.3,
    "lr":            1e-3,
    "epochs":        60,
    "batch_size":    8,
    "save_dir":      "models",
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
}


# ── PyG dataset wrapper ───────────────────────────────────────────────────────

class PyGDataset(torch.utils.data.Dataset):
    """Wraps dict-based scene graphs into PyTorch Geometric Data objects."""

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


# ── class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(subset) -> torch.Tensor:
    """Inverse-frequency weighting to handle class imbalance."""
    counts = Counter()
    for i in range(len(subset)):
        counts.update(subset[i]["edge_label"].tolist())
    total = sum(counts.values())
    weights = torch.zeros(NUM_RELATIONS)
    for cls_idx, count in counts.items():
        weights[cls_idx] = total / (NUM_RELATIONS * count)
    # normalize so mean weight = 1
    weights = weights / weights.mean()
    return weights


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, split_name="val"):
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
    f1 = f1_score(all_labels, all_preds, labels=present, average=None, zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print(f"\n{split_name} — Macro F1: {macro_f1:.4f}")
    for i, cls_idx in enumerate(present):
        print(f"  {RELATION_NAMES[cls_idx]:20s}  F1={f1[i]:.3f}")

    return macro_f1


# ── training loop ─────────────────────────────────────────────────────────────

def train():
    device = CONFIG["device"]
    print(f"Device: {device}")
    print(f"Node feat dim: {CONFIG['node_feat_dim']}  "
          f"Edge feat dim: {CONFIG['edge_feat_dim']}")

    # ScanNet data lives at D:/scannet/scans/ (not inside the project folder)
    SCANNET_DIR = "D:/scannet/scans"
    CACHE_DIR   = "data/scannet_cache"
    full_dataset = SceneGraphDatasetScanNet(
        scannet_dir=SCANNET_DIR,
        cache_dir=CACHE_DIR,
    )

    if len(full_dataset) == 0:
        print(f"\nNo ScanNet scenes loaded from: {SCANNET_DIR}")
        print("Check that the directory contains scene folders (scene0000_00, ...).")
        return

    # 70/15/15 split
    test_size  = max(1, int(len(full_dataset) * 0.15))
    val_size   = max(1, int(len(full_dataset) * 0.15))
    train_size = len(full_dataset) - val_size - test_size
    if train_size < 1:
        print(f"Not enough scenes ({len(full_dataset)}) for a train/val/test split.")
        return

    train_raw, val_raw, test_raw = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"\nTrain: {train_size} | Val: {val_size} | Test: {test_size}")

    train_ds = PyGDataset(train_raw)
    val_ds   = PyGDataset(val_raw)
    test_ds  = PyGDataset(test_raw)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False)

    # class weights from training split
    weights = compute_class_weights(train_raw).to(device)
    print("\nClass weights:")
    for i, w in enumerate(weights):
        print(f"  {RELATION_NAMES[i]:20s}  {w:.3f}")

    # model with updated feature dims
    model = RelationGNN(
        node_feat_dim=CONFIG["node_feat_dim"],
        edge_feat_dim=CONFIG["edge_feat_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_relations=NUM_RELATIONS,
        dropout=CONFIG["dropout"],
    ).to(device)
    print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / (CONFIG["epochs"] - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(weight=weights)

    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    save_path = os.path.join(
        CONFIG["save_dir"], f"relation_gnn_{CONFIG['model_name']}.pt"
    )

    best_f1 = 0.0

    for epoch in range(1, CONFIG["epochs"] + 1):
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

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d}/{CONFIG['epochs']} — "
              f"Loss: {avg_loss:.4f}  LR: {lr:.6f}")

        if epoch % 5 == 0:
            macro_f1 = evaluate(model, val_loader, device, split_name="Val")
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                torch.save(model.state_dict(), save_path)
                print(f"  ✓ Saved best model (F1={best_f1:.4f})")

    print(f"\nTraining complete. Best Val Macro F1: {best_f1:.4f}")
    print(f"Model saved to: {save_path}")

    print("\n" + "=" * 50)
    print("FINAL TEST SET EVALUATION")
    print("=" * 50)
    model.load_state_dict(torch.load(save_path, weights_only=True))
    evaluate(model, test_loader, device, split_name="Test")


def _train_3dssg_fallback(device: str):
    """
    Fallback: train with 3DSSG semantic features when ScanNet isn't available.
    Uses node_feat_dim=10 / edge_feat_dim=8 with zero-padded semantic features
    so the model architecture matches the ScanNet geometry model.
    """
    from src.dataset.loader_3dssg import SceneGraphDataset3DSSG
    import torch.nn.functional as F

    print("Loading 3DSSG dataset with padded features "
          f"(node={CONFIG['node_feat_dim']}, edge={CONFIG['edge_feat_dim']})...")

    raw_ds = SceneGraphDataset3DSSG()

    # pad features to match ScanNet dims
    padded = []
    for g in raw_ds.graphs:
        x_pad = F.pad(g["x"], (0, CONFIG["node_feat_dim"] - g["x"].shape[1]))
        e_pad = F.pad(g["edge_attr"], (0, CONFIG["edge_feat_dim"] - g["edge_attr"].shape[1]))
        padded.append({**g, "x": x_pad, "edge_attr": e_pad})

    class PaddedDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, i):
            return self.items[i]

    full_dataset = PaddedDataset(padded)
    test_size  = int(len(full_dataset) * 0.15)
    val_size   = int(len(full_dataset) * 0.15)
    train_size = len(full_dataset) - val_size - test_size
    train_raw, val_raw, test_raw = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_ds = PyGDataset(train_raw)
    val_ds   = PyGDataset(val_raw)
    test_ds  = PyGDataset(test_raw)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False)

    weights = compute_class_weights(train_raw).to(device)

    model = RelationGNN(
        node_feat_dim=CONFIG["node_feat_dim"],
        edge_feat_dim=CONFIG["edge_feat_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_relations=NUM_RELATIONS,
        dropout=CONFIG["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / (CONFIG["epochs"] - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(weight=weights)

    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    save_path = os.path.join(
        CONFIG["save_dir"], f"relation_gnn_{CONFIG['model_name']}_3dssg_padded.pt"
    )
    best_f1 = 0.0

    for epoch in range(1, CONFIG["epochs"] + 1):
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
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d}/{CONFIG['epochs']} — "
              f"Loss: {avg_loss:.4f}  LR: {lr:.6f}")
        if epoch % 5 == 0:
            macro_f1 = evaluate(model, val_loader, device, split_name="Val")
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                torch.save(model.state_dict(), save_path)
                print(f"  ✓ Saved best model (F1={best_f1:.4f})")

    print(f"\nFallback training complete. Best Val Macro F1: {best_f1:.4f}")
    model.load_state_dict(torch.load(save_path, weights_only=True))
    evaluate(model, test_loader, device, split_name="Test")


if __name__ == "__main__":
    train()
