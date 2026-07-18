"""
Train RelationGNN v7 on 3RScan Gaussian Splats with 3DSSG annotations.

Pipeline:
    1. Load/build cache of 3RScan splat scene graphs
    2. Split: 400 train / 80 val / remaining test
    3. Augment training split (rotations, flips, jitter)
    4. Train dual-head GNN with AsymmetricFocalBCELoss
    5. Tune per-relation thresholds on val set
    6. Evaluate with hierarchical constraints + symbolic repair
    7. Cross-domain evaluation on custom tabletop scenes

Usage:
    python scripts/train_3rscan.py
"""
import sys
sys.path.insert(0, ".")

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from src.dataset.loader_3rscan_splat import Dataset3RScanSplat
from src.models.relation_gnn import RelationGNN, CONTACT_INDICES, DIRECTIONAL_INDICES
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, Relation
from src.training.augmentation import augment_graph
from src.repair.symbolic_repair import SceneGraphRepair

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def _tqdm(it, **kwargs):
        return it


# ── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    "model_name":       "v7_3rscan_splat",
    "node_feat_dim":    10,
    "edge_feat_dim":    17,
    "hidden_dim":       256,
    "dropout":          0.3,
    "lr":               1e-3,
    "weight_decay":     1e-4,
    "epochs":           80,
    "batch_size":       8,
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
    "splats_dir":       "D:/3rscan_splats",
    "cache_dir":        "D:/logicsplat_data/3rscan_cache",
    "save_dir":         "models",
    "focal_gamma_neg":  2.0,
    "focal_gamma_pos":  0.0,
    "use_augmentation": True,
    "augment_factor":   4,
    "augment_jitter":   True,
    "train_size":       400,
    "val_size":         80,
}


# ── Asymmetric Focal BCE Loss (same as train_scannet.py) ──────────────────────

class AsymmetricFocalBCELoss(nn.Module):
    """
    Focal loss for contact relations, standard BCE for directional.
    gamma_neg=2: down-weights easy negatives
    gamma_pos=0: standard treatment for positives
    """

    def __init__(self, pos_weight, contact_indices, gamma_neg=2.0, gamma_pos=0.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.contact_indices = contact_indices
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos

    def forward(self, logits, labels):
        bce = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=self.pos_weight, reduction='none'
        )
        probs = torch.sigmoid(logits)
        focal_weight = torch.where(
            labels == 1,
            (1 - probs) ** self.gamma_pos,
            probs ** self.gamma_neg
        )
        mask = torch.zeros_like(logits)
        for idx in self.contact_indices:
            mask[:, idx] = 1.0
        weight = 1.0 + mask * (focal_weight - 1.0)
        return (bce * weight).mean()


# ── PyG dataset wrapper ───────────────────────────────────────────────────────

class PyGDataset(torch.utils.data.Dataset):
    """Wraps dict-based scene graphs into PyTorch Geometric Data objects."""

    def __init__(self, graphs):
        self.data = graphs

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


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(graphs) -> torch.Tensor:
    """Inverse-frequency pos_weight for BCEWithLogitsLoss, capped at 100."""
    pos_counts = torch.zeros(NUM_RELATIONS)
    total_edges = 0
    for g in graphs:
        labels = g["edge_label"]
        pos_counts += labels.sum(dim=0)
        total_edges += labels.shape[0]
    pos_counts = torch.clamp(pos_counts, min=1.0)
    neg_counts = total_edges - pos_counts
    weights = neg_counts / pos_counts
    weights = torch.clamp(weights, max=100.0)
    return weights


# ── Hierarchical constraints ──────────────────────────────────────────────────

def apply_hierarchical_constraints(preds, probs):
    """Enforce physical constraints on predictions."""
    IDX_ON_TOP_OF = int(Relation.ON_TOP_OF)
    IDX_UNDER = int(Relation.UNDER)
    IDX_HIGHER_THAN = int(Relation.HIGHER_THAN)
    IDX_LOWER_THAN = int(Relation.LOWER_THAN)
    IDX_LEFT_OF = int(Relation.LEFT_OF)
    IDX_RIGHT_OF = int(Relation.RIGHT_OF)
    IDX_IN_FRONT_OF = int(Relation.IN_FRONT_OF)
    IDX_BEHIND = int(Relation.BEHIND)

    preds = preds.clone()

    # Suppressions
    mask_not_higher = preds[:, IDX_HIGHER_THAN] == 0
    preds[mask_not_higher, IDX_ON_TOP_OF] = 0
    mask_not_lower = preds[:, IDX_LOWER_THAN] == 0
    preds[mask_not_lower, IDX_UNDER] = 0

    # Implications
    mask_ontop = preds[:, IDX_ON_TOP_OF] == 1
    preds[mask_ontop, IDX_HIGHER_THAN] = 1
    mask_under = preds[:, IDX_UNDER] == 1
    preds[mask_under, IDX_LOWER_THAN] = 1

    # Exclusivity
    conflict_lr = (preds[:, IDX_LEFT_OF] == 1) & (preds[:, IDX_RIGHT_OF] == 1)
    if conflict_lr.any():
        keep_left = probs[conflict_lr, IDX_LEFT_OF] >= probs[conflict_lr, IDX_RIGHT_OF]
        conflict_indices = conflict_lr.nonzero(as_tuple=True)[0]
        for i, idx in enumerate(conflict_indices):
            if keep_left[i]:
                preds[idx, IDX_RIGHT_OF] = 0
            else:
                preds[idx, IDX_LEFT_OF] = 0

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
    """Find optimal sigmoid threshold per relation on validation set."""
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
        print(f"    {RELATION_NAMES[rel_idx]:20s} threshold={best_thresh:.2f} "
              f"val_F1={best_f1:.3f}")

    return thresholds


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, split_name="val",
             thresholds=None, use_constraints=False):
    """Multi-label evaluation with optional thresholds and constraints."""
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

    if thresholds:
        all_preds = torch.zeros_like(all_probs)
        for rel_idx in range(NUM_RELATIONS):
            thresh = thresholds.get(rel_idx, 0.5)
            all_preds[:, rel_idx] = (all_probs[:, rel_idx] >= thresh).float()
    else:
        all_preds = (all_probs >= 0.5).float()

    if use_constraints:
        all_preds = apply_hierarchical_constraints(all_preds, all_probs)

    all_preds_np = all_preds.numpy()
    all_labels_np = all_labels.numpy()

    f1_per_rel = f1_score(all_labels_np, all_preds_np, average=None, zero_division=0)
    macro_f1 = float(np.mean(f1_per_rel))
    micro_f1 = f1_score(all_labels_np, all_preds_np, average="micro", zero_division=0)

    constraint_tag = " [+constraints]" if use_constraints else ""
    thresh_tag = " [+tuned_thresh]" if thresholds else ""
    print(f"\n{split_name}{thresh_tag}{constraint_tag} -- "
          f"Macro F1: {macro_f1:.4f}  Micro F1: {micro_f1:.4f}")
    for i, f1 in enumerate(f1_per_rel):
        pos = int(all_labels_np[:, i].sum())
        print(f"  {RELATION_NAMES[i]:20s}  F1={f1:.3f}  pos={pos}")

    return macro_f1


# ── Symbolic repair evaluation ────────────────────────────────────────────────

def evaluate_with_repair(model, loader, device, thresholds):
    """Evaluate with symbolic repair applied after GNN predictions."""
    model.eval()
    repair = SceneGraphRepair(verbose=False)

    all_preds_repaired = []
    all_labels_all = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            probs = torch.sigmoid(logits).cpu()
            labels = batch.y.cpu()

            # Apply thresholds
            preds = torch.zeros_like(probs)
            for rel_idx in range(NUM_RELATIONS):
                thresh = thresholds.get(rel_idx, 0.5)
                preds[:, rel_idx] = (probs[:, rel_idx] >= thresh).float()

            # Apply hierarchical constraints
            preds = apply_hierarchical_constraints(preds, probs)

            # Convert to symbolic repair format and back
            edge_index = batch.edge_index.cpu()
            predictions_for_repair = []
            for e in range(preds.shape[0]):
                src_idx = edge_index[0, e].item()
                dst_idx = edge_index[1, e].item()
                for rel_idx in range(NUM_RELATIONS):
                    if preds[e, rel_idx] == 1:
                        conf = float(probs[e, rel_idx])
                        predictions_for_repair.append((
                            f"obj_{src_idx}",
                            RELATION_NAMES[rel_idx],
                            f"obj_{dst_idx}",
                            conf,
                        ))

            # Run symbolic repair
            if predictions_for_repair:
                repaired, stats = repair.repair(predictions_for_repair)

                # Convert back to tensor format
                preds_repaired = torch.zeros_like(preds)
                for subj, rel_name, obj, conf in repaired:
                    src_id = int(subj.split("_")[1])
                    dst_id = int(obj.split("_")[1])
                    rel_idx = next(
                        (k for k, v in RELATION_NAMES.items() if v == rel_name),
                        None
                    )
                    if rel_idx is None:
                        continue
                    # Find edge position
                    edge_mask = (edge_index[0] == src_id) & (edge_index[1] == dst_id)
                    positions = edge_mask.nonzero(as_tuple=True)[0]
                    if len(positions) > 0:
                        preds_repaired[positions[0], rel_idx] = 1.0

                all_preds_repaired.append(preds_repaired)
            else:
                all_preds_repaired.append(preds)

            all_labels_all.append(labels)

    all_preds_repaired = torch.cat(all_preds_repaired, dim=0).numpy()
    all_labels_all = torch.cat(all_labels_all, dim=0).numpy()

    f1_per_rel = f1_score(all_labels_all, all_preds_repaired, average=None, zero_division=0)
    macro_f1 = float(np.mean(f1_per_rel))
    micro_f1 = f1_score(all_labels_all, all_preds_repaired, average="micro", zero_division=0)

    print(f"\nTest [+tuned_thresh +constraints +symbolic_repair] -- "
          f"Macro F1: {macro_f1:.4f}  Micro F1: {micro_f1:.4f}")
    for i, f1 in enumerate(f1_per_rel):
        pos = int(all_labels_all[:, i].sum())
        print(f"  {RELATION_NAMES[i]:20s}  F1={f1:.3f}  pos={pos}")

    return macro_f1


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    device = CONFIG["device"]
    print("=" * 60)
    print("  3RScan Gaussian Splat GNN Training (v7 dual-head)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {CONFIG['model_name']}")
    print(f"hidden_dim={CONFIG['hidden_dim']}  dropout={CONFIG['dropout']}")
    print(f"edge_feat_dim={CONFIG['edge_feat_dim']}  node_feat_dim={CONFIG['node_feat_dim']}")
    print(f"Focal loss: gamma_neg={CONFIG['focal_gamma_neg']}, "
          f"gamma_pos={CONFIG['focal_gamma_pos']}")
    print(f"Splats dir: {CONFIG['splats_dir']}")
    print(f"Cache dir: {CONFIG['cache_dir']}")
    print()

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset = Dataset3RScanSplat(
        splats_dir=CONFIG["splats_dir"],
        cache_dir=CONFIG["cache_dir"],
        verbose=True,
    )

    if len(dataset) == 0:
        print("ERROR: No scenes loaded. Check splats_dir and 3DSSG annotations.")
        return

    print(f"\nTotal scenes available: {len(dataset)}")

    # ── Split by scene ────────────────────────────────────────────────────────
    n_scenes = len(dataset)
    train_size = min(CONFIG["train_size"], n_scenes - CONFIG["val_size"] - 10)
    val_size = min(CONFIG["val_size"], n_scenes - train_size - 5)
    test_size = n_scenes - train_size - val_size

    if train_size < 10 or val_size < 5 or test_size < 5:
        # Fallback split ratios
        train_size = int(n_scenes * 0.7)
        val_size = int(n_scenes * 0.15)
        test_size = n_scenes - train_size - val_size

    train_raw, val_raw, test_raw = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Split — Train: {train_size} | Val: {val_size} | Test: {test_size}")

    # ── Augment training split ────────────────────────────────────────────────
    if CONFIG["use_augmentation"]:
        print(f"\nAugmenting training split (factor={CONFIG['augment_factor']}, "
              f"jitter={CONFIG['augment_jitter']})")
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
              f"({train_size} × {1 + len(selected_augs)})")
    else:
        train_graphs = [train_raw[i] for i in range(len(train_raw))]

    val_graphs = [val_raw[i] for i in range(len(val_raw))]
    test_graphs = [test_raw[i] for i in range(len(test_raw))]

    # ── Create PyG datasets ───────────────────────────────────────────────────
    train_ds = PyGDataset(train_graphs)
    val_ds = PyGDataset(val_graphs)
    test_ds = PyGDataset(test_graphs)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    # ── Class weights ─────────────────────────────────────────────────────────
    weights = compute_class_weights(train_graphs).to(device)
    print("\nClass weights (multi-label BCE, capped at 100):")
    for i, w in enumerate(weights):
        print(f"  {RELATION_NAMES[i]:20s}  {w:.3f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RelationGNN(
        node_feat_dim=CONFIG["node_feat_dim"],
        edge_feat_dim=CONFIG["edge_feat_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_relations=NUM_RELATIONS,
        dropout=CONFIG["dropout"],
    ).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    # Cosine LR with warmup
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / (CONFIG["epochs"] - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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

    # ── Training ──────────────────────────────────────────────────────────────
    best_f1 = 0.0
    patience = 10
    patience_counter = 0

    print("\n" + "=" * 60)
    print("  TRAINING")
    print("=" * 60)

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
            if HAS_TQDM and hasattr(batch_iter, 'set_postfix'):
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)
        lr = scheduler.get_last_lr()[0]

        if HAS_TQDM and hasattr(epoch_iter, 'set_postfix'):
            epoch_iter.set_postfix(
                loss=f"{avg_loss:.4f}", lr=f"{lr:.6f}", best=f"{best_f1:.4f}"
            )
        elif not HAS_TQDM:
            print(f"Epoch {epoch:02d}/{CONFIG['epochs']} -- "
                  f"Loss: {avg_loss:.4f}  LR: {lr:.6f}")

        # Validate every 5 epochs
        if epoch % 5 == 0:
            macro_f1 = evaluate(model, val_loader, device, split_name="Val")
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
                print(f"  [SAVED] best model (F1={best_f1:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

    print(f"\nTraining complete. Best Val Macro F1: {best_f1:.4f}")
    print(f"Model saved to: {save_path}")

    # ── Threshold tuning ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  THRESHOLD TUNING (validation set)")
    print("=" * 60)
    model.load_state_dict(torch.load(save_path, weights_only=True))
    thresholds = tune_thresholds(model, val_loader, device)

    # Save thresholds
    thresholds_path = save_path.replace(".pt", "_thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump({str(k): v for k, v in thresholds.items()}, f, indent=2)
    print(f"  Thresholds saved to: {thresholds_path}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL TEST SET EVALUATION")
    print("=" * 60)

    # Baseline
    print("\n--- Baseline (threshold=0.5, no constraints) ---")
    evaluate(model, test_loader, device, split_name="Test")

    # With tuned thresholds
    print("\n--- With tuned thresholds ---")
    evaluate(model, test_loader, device, split_name="Test", thresholds=thresholds)

    # With tuned thresholds + hierarchical constraints
    print("\n--- With tuned thresholds + hierarchical constraints ---")
    evaluate(model, test_loader, device, split_name="Test",
             thresholds=thresholds, use_constraints=True)

    # With symbolic repair
    print("\n--- With tuned thresholds + constraints + symbolic repair ---")
    evaluate_with_repair(model, test_loader, device, thresholds)

    # ── Cross-domain evaluation on custom scenes ──────────────────────────────
    print("\n" + "=" * 60)
    print("  CROSS-DOMAIN EVALUATION (custom tabletop scenes)")
    print("=" * 60)
    cross_domain_eval(model, thresholds, device)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Geometry rules baseline:          0.478")
    print(f"  GNN v7 ScanNet:                   0.281")
    print(f"  GNN v7 fine-tuned synthetic:      0.322")
    print(f"  GNN v7 3RScan splat (this run):   {best_f1:.3f} (val)")
    print("=" * 60)


# ── Cross-domain evaluation ───────────────────────────────────────────────────

def cross_domain_eval(model, thresholds, device):
    """Evaluate the 3RScan-trained model on custom tabletop scenes."""
    # Try to load custom scene cache
    custom_cache_dir = "D:/logicsplat_data/scannet_cache"  # custom scenes use same cache format
    if not os.path.isdir(custom_cache_dir):
        print("  Custom scene cache not found. Skipping cross-domain eval.")
        return

    cache_files = sorted(f for f in os.listdir(custom_cache_dir) if f.endswith(".pt"))
    if not cache_files:
        print("  No cached custom scenes found. Skipping cross-domain eval.")
        return

    # Load custom scene graphs
    custom_graphs = []
    for fname in cache_files:
        try:
            g = torch.load(os.path.join(custom_cache_dir, fname), weights_only=False)
            if g["x"].shape[1] == 10 and g["edge_attr"].shape[1] == 17:
                if g["edge_label"].dim() == 2 and g["edge_label"].shape[1] == NUM_RELATIONS:
                    custom_graphs.append(g)
        except Exception:
            continue

    if not custom_graphs:
        print("  No valid custom scene graphs found. Skipping.")
        return

    print(f"  Loaded {len(custom_graphs)} custom tabletop scenes")

    custom_ds = PyGDataset(custom_graphs)
    custom_loader = DataLoader(custom_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    # Evaluate
    print("\n  --- Custom scenes: tuned thresholds + constraints ---")
    evaluate(model, custom_loader, device, split_name="Custom",
             thresholds=thresholds, use_constraints=True)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
