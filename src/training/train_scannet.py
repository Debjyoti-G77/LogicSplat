"""
Train RelationGNN on ScanNet geometry.

v7 improvements over v6:
  - Dual-head architecture (directional + contact) — separate gradient paths
  - Asymmetric focal loss for contact relations — down-weights easy negatives
  - Rule-margin features (14→17 dim) — encode proximity to decision boundary
  - Per-relation threshold tuning on validation set
  - Hierarchical constraint enforcement at inference time
  - edge_feat_dim 14 → 17

Usage:
    python src/training/train_scannet.py
"""
import sys
sys.path.insert(0, ".")

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from src.dataset.loader_scannet import SceneGraphDatasetScanNet
from src.models.relation_gnn import RelationGNN, CONTACT_INDICES, DIRECTIONAL_INDICES
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, Relation

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def _tqdm(it, **kwargs):
        return it


# ── config ────────────────────────────────────────────────────────────────────

CONFIG = {
    "model_name":    "v7_dualhead",
    "node_feat_dim": 10,
    "edge_feat_dim": 17,
    "hidden_dim":    256,
    "dropout":       0.3,
    "lr":            1e-3,
    "weight_decay":  1e-4,
    "epochs":        80,
    "batch_size":    8,
    "save_dir":      "models",
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
    "use_augmentation":  True,
    "augment_factor":    4,
    "augment_jitter":    True,
    "exclude_relations": [],
    # Focal loss params for contact head
    "focal_gamma_neg":   2.0,
    "focal_gamma_pos":   0.0,
}


# ── Asymmetric Focal BCE Loss ─────────────────────────────────────────────────

class AsymmetricFocalBCELoss(nn.Module):
    """
    Focal loss applied to contact relations only.
    Standard BCE for directional relations (they don't need it).

    gamma_neg=2: down-weights easy negatives (pairs far apart, obviously not on_top_of)
    gamma_pos=0: standard treatment for positives (no down-weighting)
    """

    def __init__(self, pos_weight, contact_indices, gamma_neg=2.0, gamma_pos=0.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.contact_indices = contact_indices
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos

    def forward(self, logits, labels):
        # Standard BCE with pos_weight
        bce = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=self.pos_weight, reduction='none'
        )

        # Focal modulation for contact relations only
        probs = torch.sigmoid(logits)

        # Compute focal weights
        focal_weight = torch.where(
            labels == 1,
            (1 - probs) ** self.gamma_pos,  # positive: standard (gamma=0 → weight=1)
            probs ** self.gamma_neg          # negative: down-weight easy negatives
        )

        # Apply focal only to contact relation columns
        mask = torch.zeros_like(logits)
        for idx in self.contact_indices:
            mask[:, idx] = 1.0

        # For contact columns: use focal weight; for directional: weight=1.0
        weight = 1.0 + mask * (focal_weight - 1.0)

        return (bce * weight).mean()


# ── PyG dataset wrapper ───────────────────────────────────────────────────────

class PyGDataset(torch.utils.data.Dataset):
    """Wraps dict-based scene graphs into PyTorch Geometric Data objects."""

    def __init__(self, subset, exclude_indices=None):
        self.data = [subset[i] for i in range(len(subset))]
        self.exclude_indices = exclude_indices or []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        g = self.data[idx]
        y = g["edge_label"].clone()   # (E, NUM_RELATIONS) multi-hot float
        # zero out excluded relation columns so they don't contribute to loss
        for i in self.exclude_indices:
            y[:, i] = 0.0
        return Data(
            x=g["x"],
            edge_index=g["edge_index"],
            edge_attr=g["edge_attr"],
            y=y,
        )


# ── class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(subset, exclude_indices=None) -> torch.Tensor:
    """
    Inverse-frequency pos_weight for BCEWithLogitsLoss.
    Capped at 100 to prevent gradient explosion on extremely rare classes.
    """
    pos_counts = torch.zeros(NUM_RELATIONS)
    total_edges = 0
    for i in range(len(subset)):
        labels = subset[i]["edge_label"]
        pos_counts += labels.sum(dim=0)
        total_edges += labels.shape[0]
    pos_counts = torch.clamp(pos_counts, min=1.0)
    neg_counts = total_edges - pos_counts
    weights = neg_counts / pos_counts
    weights = torch.clamp(weights, max=100.0)
    if exclude_indices:
        for idx in exclude_indices:
            weights[idx] = 0.0
    return weights


# ── Hierarchical constraints ──────────────────────────────────────────────────

def apply_hierarchical_constraints(preds, probs):
    """
    Enforce physical constraints on predictions:
    1. NOT higher_than(A,B) → cannot have on_top_of(A,B) [suppression]
    2. NOT lower_than(A,B) → cannot have under(A,B) [suppression]
    3. on_top_of(A,B) → higher_than(A,B) must also be true [implication]
    4. under(A,B) → lower_than(A,B) must also be true [implication]
    5. left_of(A,B) conflicts with right_of(A,B) — keep higher confidence
    6. in_front_of(A,B) conflicts with behind(A,B) — keep higher confidence

    Order matters: suppressions first, then implications.
    This ensures that on_top_of is only kept when higher_than was independently predicted.
    """
    IDX_ON_TOP_OF = int(Relation.ON_TOP_OF)
    IDX_UNDER = int(Relation.UNDER)
    IDX_HIGHER_THAN = int(Relation.HIGHER_THAN)
    IDX_LOWER_THAN = int(Relation.LOWER_THAN)
    IDX_LEFT_OF = int(Relation.LEFT_OF)
    IDX_RIGHT_OF = int(Relation.RIGHT_OF)
    IDX_IN_FRONT_OF = int(Relation.IN_FRONT_OF)
    IDX_BEHIND = int(Relation.BEHIND)

    preds = preds.clone()

    # ── Suppressions first (hard physical constraints) ────────────────────────
    # NOT higher_than → NOT on_top_of (can't be on top without being higher)
    mask_not_higher = preds[:, IDX_HIGHER_THAN] == 0
    preds[mask_not_higher, IDX_ON_TOP_OF] = 0

    # NOT lower_than → NOT under (can't be under without being lower)
    mask_not_lower = preds[:, IDX_LOWER_THAN] == 0
    preds[mask_not_lower, IDX_UNDER] = 0

    # ── Implications (ensure consistency) ─────────────────────────────────────
    # on_top_of → higher_than (if on_top_of survived suppression, ensure higher_than)
    mask_ontop = preds[:, IDX_ON_TOP_OF] == 1
    preds[mask_ontop, IDX_HIGHER_THAN] = 1

    # under → lower_than
    mask_under = preds[:, IDX_UNDER] == 1
    preds[mask_under, IDX_LOWER_THAN] = 1

    # ── Exclusivity constraints ───────────────────────────────────────────────
    # left_of vs right_of — keep higher prob
    conflict_lr = (preds[:, IDX_LEFT_OF] == 1) & (preds[:, IDX_RIGHT_OF] == 1)
    if conflict_lr.any():
        keep_left = probs[conflict_lr, IDX_LEFT_OF] >= probs[conflict_lr, IDX_RIGHT_OF]
        conflict_indices = conflict_lr.nonzero(as_tuple=True)[0]
        for i, idx in enumerate(conflict_indices):
            if keep_left[i]:
                preds[idx, IDX_RIGHT_OF] = 0
            else:
                preds[idx, IDX_LEFT_OF] = 0

    # in_front_of vs behind — keep higher prob
    conflict_fb = (preds[:, IDX_IN_FRONT_OF] == 1) & (preds[:, IDX_BEHIND] == 1)
    if conflict_fb.any():
        keep_front = probs[conflict_fb, IDX_IN_FRONT_OF] >= probs[conflict_fb, IDX_BEHIND]
        conflict_indices = conflict_fb.nonzero(as_tuple=True)[0]
        for i, idx in enumerate(conflict_indices):
            if keep_front[i]:
                preds[idx, IDX_BEHIND] = 0
            else:
                preds[idx, IDX_IN_FRONT_OF] = 0

    return preds


# ── Per-relation threshold tuning ─────────────────────────────────────────────

def tune_thresholds(model, val_loader, device):
    """
    Find optimal sigmoid threshold per relation on validation set.
    Returns dict {relation_idx: optimal_threshold}
    """
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            all_logits.append(logits.cpu())
            all_labels.append(batch.y.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()

    thresholds = {}
    print("\n  Per-relation threshold tuning (validation set):")
    for rel_idx in range(NUM_RELATIONS):
        best_f1 = 0.0
        best_thresh = 0.5
        probs = torch.sigmoid(all_logits[:, rel_idx]).numpy()

        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (probs >= thresh).astype(int)
            f1 = f1_score(all_labels[:, rel_idx], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)

        thresholds[rel_idx] = best_thresh
        print(f"    {RELATION_NAMES[rel_idx]:20s} optimal_threshold={best_thresh:.2f} "
              f"val_F1={best_f1:.3f}")

    return thresholds


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, split_name="val", exclude_indices=None,
             thresholds=None, use_constraints=False):
    """
    Multi-label evaluation with optional per-relation thresholds and
    hierarchical constraints.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_labels.append(batch.y.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Apply per-relation thresholds
    if thresholds:
        all_preds = torch.zeros_like(all_probs)
        for rel_idx in range(NUM_RELATIONS):
            thresh = thresholds.get(rel_idx, 0.5)
            all_preds[:, rel_idx] = (all_probs[:, rel_idx] >= thresh).float()
    else:
        all_preds = (all_probs >= 0.5).float()

    # Apply hierarchical constraints
    if use_constraints:
        all_preds = apply_hierarchical_constraints(all_preds, all_probs)

    all_preds_np = all_preds.numpy()
    all_labels_np = all_labels.numpy()

    exclude_set = set(exclude_indices or [])
    active_cols = [i for i in range(NUM_RELATIONS) if i not in exclude_set]

    f1_per_rel = f1_score(all_labels_np, all_preds_np, average=None, zero_division=0)
    macro_f1 = float(np.mean([f1_per_rel[i] for i in active_cols]))
    micro_f1 = f1_score(
        all_labels_np[:, active_cols], all_preds_np[:, active_cols],
        average="micro", zero_division=0,
    )

    constraint_tag = " [+constraints]" if use_constraints else ""
    thresh_tag = " [+tuned_thresh]" if thresholds else ""
    print(f"\n{split_name}{thresh_tag}{constraint_tag} -- "
          f"Macro F1: {macro_f1:.4f}  Micro F1: {micro_f1:.4f}"
          f"  (over {len(active_cols)} active relations)")
    for i, f1 in enumerate(f1_per_rel):
        pos = int(all_labels_np[:, i].sum())
        tag = " [excluded]" if i in exclude_set else ""
        print(f"  {RELATION_NAMES[i]:20s}  F1={f1:.3f}  pos={pos}{tag}")

    return macro_f1


# ── training loop ─────────────────────────────────────────────────────────────

def train():
    device = CONFIG["device"]
    print(f"Device: {device}")
    print(f"Model: {CONFIG['model_name']}")
    print(f"hidden_dim={CONFIG['hidden_dim']}  dropout={CONFIG['dropout']}"
          f"  weight_decay={CONFIG['weight_decay']}")
    print(f"edge_feat_dim={CONFIG['edge_feat_dim']}  (17 = 10 base + 4 contact + 3 rule-margin)")
    print(f"Focal loss: gamma_neg={CONFIG['focal_gamma_neg']}, gamma_pos={CONFIG['focal_gamma_pos']}")
    print(f"Excluded relations: {CONFIG['exclude_relations']}")

    SCANNET_DIR = "D:/scannet/scans"
    CACHE_DIR   = "D:/logicsplat_data/scannet_cache"

    # resolve excluded relation indices
    exclude_indices = [
        idx for idx, name in RELATION_NAMES.items()
        if name in CONFIG["exclude_relations"]
    ]
    print(f"Excluded indices: {exclude_indices} -> "
          f"{[RELATION_NAMES[i] for i in exclude_indices]}")

    # ── Load raw scenes ───────────────────────────────────────────────────────
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pt")] \
                  if os.path.isdir(CACHE_DIR) else []
    if not cache_files:
        print("Cache empty -- regenerating from raw ScanNet files...")
        SceneGraphDatasetScanNet(scannet_dir=SCANNET_DIR, cache_dir=CACHE_DIR)

    raw_dataset = SceneGraphDatasetScanNet(
        scannet_dir=SCANNET_DIR, cache_dir=CACHE_DIR,
    )

    if len(raw_dataset) == 0:
        print(f"No scenes loaded. Check cache dir: {CACHE_DIR}")
        return

    # ── Split by scene (no leakage) ──────────────────────────────────────────
    n_scenes = len(raw_dataset)
    test_size  = max(1, int(n_scenes * 0.15))
    val_size   = max(1, int(n_scenes * 0.15))
    train_size = n_scenes - val_size - test_size

    train_raw, val_raw, test_raw = random_split(
        raw_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"\nRaw scenes — Train: {train_size} | Val: {val_size} | Test: {test_size}")

    # ── Augment ONLY training split ───────────────────────────────────────────
    if CONFIG["use_augmentation"]:
        print(f"\nAugmenting TRAINING split only (factor={CONFIG['augment_factor']}, "
              f"jitter={CONFIG['augment_jitter']})")
        from src.training.augmentation import augment_graph, AUGMENTATIONS
        aug_names = ["rotate_90", "rotate_180", "rotate_270", "flip_x", "flip_y"]
        if CONFIG["augment_jitter"]:
            aug_names.append("jitter")
        selected_augs = aug_names[:CONFIG["augment_factor"]]

        train_graphs = []
        for i in range(len(train_raw)):
            g = train_raw[i]
            train_graphs.append(g)
            for aug_name in selected_augs:
                train_graphs.append(augment_graph(g, aug_name))

        print(f"  Training graphs after augmentation: {len(train_graphs)} "
              f"({train_size} original × {1 + len(selected_augs)} = "
              f"{train_size * (1 + len(selected_augs))})")
    else:
        train_graphs = [train_raw[i] for i in range(len(train_raw))]

    val_graphs = [val_raw[i] for i in range(len(val_raw))]
    test_graphs = [test_raw[i] for i in range(len(test_raw))]

    train_ds = PyGDataset(train_graphs, exclude_indices=exclude_indices)
    val_ds   = PyGDataset(val_graphs,   exclude_indices=exclude_indices)
    test_ds  = PyGDataset(test_graphs,  exclude_indices=exclude_indices)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False)

    weights = compute_class_weights(train_graphs, exclude_indices=exclude_indices).to(device)
    print("\nClass weights (multi-label BCE, clamped):")
    for i, w in enumerate(weights):
        tag = " [excluded]" if i in exclude_indices else ""
        print(f"  {RELATION_NAMES[i]:20s}  {w:.3f}{tag}")

    model = RelationGNN(
        node_feat_dim=CONFIG["node_feat_dim"],
        edge_feat_dim=CONFIG["edge_feat_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_relations=NUM_RELATIONS,
        dropout=CONFIG["dropout"],
    ).to(device)
    print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / (CONFIG["epochs"] - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Use asymmetric focal loss instead of plain BCE
    criterion = AsymmetricFocalBCELoss(
        pos_weight=weights,
        contact_indices=CONTACT_INDICES,
        gamma_neg=CONFIG["focal_gamma_neg"],
        gamma_pos=CONFIG["focal_gamma_pos"],
    ).to(device)

    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    save_path = os.path.join(
        CONFIG["save_dir"], f"relation_gnn_{CONFIG['model_name']}.pt"
    )

    best_f1 = 0.0
    patience = 10
    patience_counter = 0

    epoch_iter = _tqdm(range(1, CONFIG["epochs"] + 1), desc="Training",
                       unit="epoch", dynamic_ncols=True) if HAS_TQDM \
                 else range(1, CONFIG["epochs"] + 1)

    for epoch in epoch_iter:
        model.train()
        total_loss = 0.0

        batch_iter = _tqdm(train_loader, desc=f"Ep {epoch:02d}", leave=False,
                           unit="batch", dynamic_ncols=True) if HAS_TQDM \
                     else train_loader

        for batch in batch_iter:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if HAS_TQDM:
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        lr = scheduler.get_last_lr()[0]

        if HAS_TQDM:
            epoch_iter.set_postfix(
                loss=f"{avg_loss:.4f}", lr=f"{lr:.6f}", best_f1=f"{best_f1:.4f}"
            )
        else:
            print(f"Epoch {epoch:02d}/{CONFIG['epochs']} -- "
                  f"Loss: {avg_loss:.4f}  LR: {lr:.6f}")

        if epoch % 5 == 0:
            macro_f1 = evaluate(model, val_loader, device,
                                split_name="Val", exclude_indices=exclude_indices)
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
                print(f"  [SAVED] best model (F1={best_f1:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience} patience)")
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

    print(f"\nTraining complete. Best Val Macro F1: {best_f1:.4f}")
    print(f"Model saved to: {save_path}")

    # ── Threshold tuning on validation set ────────────────────────────────────
    print("\n" + "=" * 50)
    print("THRESHOLD TUNING (validation set)")
    print("=" * 50)
    model.load_state_dict(torch.load(save_path, weights_only=True))
    thresholds = tune_thresholds(model, val_loader, device)

    # Save thresholds alongside model
    thresholds_path = save_path.replace(".pt", "_thresholds.json")
    import json
    with open(thresholds_path, "w") as f:
        json.dump({str(k): v for k, v in thresholds.items()}, f, indent=2)
    print(f"  Thresholds saved to: {thresholds_path}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("FINAL TEST SET EVALUATION")
    print("=" * 50)

    # Baseline: fixed 0.5 threshold, no constraints
    print("\n--- Baseline (threshold=0.5, no constraints) ---")
    evaluate(model, test_loader, device,
             split_name="Test", exclude_indices=exclude_indices)

    # With tuned thresholds
    print("\n--- With tuned thresholds ---")
    evaluate(model, test_loader, device,
             split_name="Test", exclude_indices=exclude_indices,
             thresholds=thresholds)

    # With tuned thresholds + hierarchical constraints
    print("\n--- With tuned thresholds + hierarchical constraints ---")
    evaluate(model, test_loader, device,
             split_name="Test", exclude_indices=exclude_indices,
             thresholds=thresholds, use_constraints=True)


if __name__ == "__main__":
    train()
