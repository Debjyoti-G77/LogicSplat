"""
Training loop for RelationGNN on 3DSSG dataset.

Improvements:
- PyTorch Geometric DataLoader (batch_size=8, 8x faster, stable gradients)
- Cosine annealing LR schedule (smoother convergence)
- 60 epochs (model wasn't converged at 30)
- Proper train/val/test split
- Per-relation F1 evaluation
"""
import sys
sys.path.insert(0, ".")

import os
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from torch.utils.data import random_split
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from src.dataset.loader_3dssg import SceneGraphDataset3DSSG
from src.models.relation_gnn import RelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES


# ── config ────────────────────────────────────────────────────────────────────

CONFIG = {
    "model_name":    "gat_geo_gpu_b8",
    "node_feat_dim": 8,
    "edge_feat_dim": 4,
    "hidden_dim":    128,
    "dropout":       0.3,
    "lr":            1e-3,
    "epochs":        60,
    "batch_size":    8,
    "save_dir":      "models",
    "device":        "cuda",
}


# ── convert dict graphs to PyG Data objects ───────────────────────────────────

class PyGDataset(torch.utils.data.Dataset):
    """Wraps our dict-based dataset into PyTorch Geometric Data objects."""
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
    counts = Counter()
    for i in range(len(subset)):
        counts.update(subset[i]["edge_label"].tolist())
    total = sum(counts.values())
    weights = torch.zeros(NUM_RELATIONS)
    for cls_idx, count in counts.items():
        weights[cls_idx] = total / (NUM_RELATIONS * count)
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

    full_dataset = SceneGraphDataset3DSSG()

    # 70/15/15 split
    test_size  = int(len(full_dataset) * 0.15)
    val_size   = int(len(full_dataset) * 0.15)
    train_size = len(full_dataset) - val_size - test_size
    train_raw, val_raw, test_raw = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"\nTrain: {train_size} | Val: {val_size} | Test: {test_size}")

    # wrap in PyG datasets and DataLoaders
    train_ds = PyGDataset(train_raw)
    val_ds   = PyGDataset(val_raw)
    test_ds  = PyGDataset(test_raw)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False)

    # class weights
    weights = compute_class_weights(train_raw).to(device)
    print("\nClass weights:")
    for i, w in enumerate(weights):
        print(f"  {RELATION_NAMES[i]:20s}  {w:.3f}")

    # model
    model = RelationGNN(
        node_feat_dim=CONFIG["node_feat_dim"],
        edge_feat_dim=CONFIG["edge_feat_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_relations=NUM_RELATIONS,
        dropout=CONFIG["dropout"],
    ).to(device)
    print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    # warmup for 5 epochs then cosine annealing
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / (CONFIG["epochs"] - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(weight=weights)

    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    save_path = os.path.join(CONFIG["save_dir"], f"relation_gnn_{CONFIG['model_name']}.pt")

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
        print(f"Epoch {epoch:02d}/{CONFIG['epochs']} — Loss: {avg_loss:.4f}  LR: {lr:.6f}")

        if epoch % 5 == 0:
            macro_f1 = evaluate(model, val_loader, device, split_name="Val")
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                torch.save(model.state_dict(), save_path)
                print(f"  ✓ Saved best model (F1={best_f1:.4f})")

    print(f"\nTraining complete. Best Val Macro F1: {best_f1:.4f}")
    print(f"Model saved to: {save_path}")

    print("\n" + "="*50)
    print("FINAL TEST SET EVALUATION")
    print("="*50)
    model.load_state_dict(torch.load(save_path, weights_only=True))
    evaluate(model, test_loader, device, split_name="Test")


if __name__ == "__main__":
    train()
