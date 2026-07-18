"""
GeoKAN Ablation Study: Train and compare all 3 variants on the same data.

Variants:
  1. GeoKAN-RBF (baseline): Full MetricNet MLP + RBF basis
  2. GeoKAN-γ (lightest): Per-dimension learnable scalar metric + RBF basis
  3. GeoKAN-Wavelet: Full MetricNet MLP + Mexican hat wavelet basis

All variants share:
  - Same GATv2 backbone (node encoder + 2× GATv2Conv)
  - Same pair projection
  - Same dual-head architecture (contact + directional)
  - Same training hyperparameters, loss, augmentation, schedule
  - Same train/val split (seed=42)

Reports: Macro F1, Pred R@3, Pred R@5, parameter count per variant.

Usage:
    python train_geokan_ablation.py
"""
import sys
sys.path.insert(0, ".")

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, Relation, LEGACY_12_TO_10_COLS
from src.training.augmentation import augment_graph

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def _tqdm(it, **kwargs):
        return it

# Import all 3 model variants
from geokan_relation import GeoKANRelationGNN, CONTACT_INDICES, DIRECTIONAL_INDICES
from geokan_gamma_relation import GeoKANGammaRelationGNN
from geokan_wavelet_relation import GeoKANWaveletRelationGNN


# ── Config (shared across all variants) ──────────────────────────────────────

CONFIG = {
    "node_feat_dim":    10,
    "edge_feat_dim":    22,
    "hidden_dim":       128,
    "dropout":          0.2,
    "lr":               2e-3,
    "weight_decay":     1e-4,
    "epochs":           100,
    "batch_size":       8,
    "device":           "cuda",
    "save_dir":         "models/ablation",
    "cache_dir":        "D:/logicsplat_data/3rscan_graph_cache",
    "train_frac":       0.85,
    "augment_factor":   6,
    "augment_jitter":   True,
    "focal_gamma_neg":  4.0,
    "focal_gamma_pos":  0.0,
    "asl_prob_margin":  0.0,
    "patience":         15,
    "metric_reg_weight": 1e-4,
    "max_grad_norm":    2.0,
    "warmup_epochs":    5,
    "pos_weight_cap":   8.0,
}


# ── Inverse relation pairs ────────────────────────────────────────────────────
INVERSE_PAIRS = [
    (int(Relation.ON_TOP_OF),   int(Relation.UNDER)),
    (int(Relation.UNDER),       int(Relation.ON_TOP_OF)),
    (int(Relation.HIGHER_THAN), int(Relation.LOWER_THAN)),
    (int(Relation.LOWER_THAN),  int(Relation.HIGHER_THAN)),
    (int(Relation.LEFT_OF),     int(Relation.RIGHT_OF)),
    (int(Relation.RIGHT_OF),    int(Relation.LEFT_OF)),
    (int(Relation.IN_FRONT_OF), int(Relation.BEHIND)),
    (int(Relation.BEHIND),      int(Relation.IN_FRONT_OF)),
]

RELATION_LOSS_WEIGHTS = {
    int(Relation.ATTACHED_TO):  1.8,
    int(Relation.ADJACENT_TO):  1.5,
    int(Relation.ON_TOP_OF):    1.2,
    int(Relation.UNDER):        1.2,
}

LABEL_SMOOTHING = {
    int(Relation.ADJACENT_TO):  0.12,
    int(Relation.IN_FRONT_OF):  0.08,
    int(Relation.BEHIND):       0.08,
    int(Relation.LEFT_OF):      0.07,
    int(Relation.RIGHT_OF):     0.07,
    int(Relation.ON_TOP_OF):    0.04,
    int(Relation.UNDER):        0.04,
    int(Relation.ATTACHED_TO):  0.05,
    int(Relation.HIGHER_THAN):  0.04,
    int(Relation.LOWER_THAN):   0.04,
}


# ── Loss function ─────────────────────────────────────────────────────────────

class AsymmetricFocalBCELoss(nn.Module):
    """ASL loss (Baruch et al. 2021) — same as train_geokan.py."""

    def __init__(self, pos_weight, gamma_neg=4.0, gamma_pos=0.0,
                 prob_margin=0.0, relation_weights=None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.prob_margin = prob_margin
        if relation_weights is not None:
            self.register_buffer("relation_weights", relation_weights)
        else:
            self.relation_weights = None

    def forward(self, logits, labels):
        mask = (labels >= 0).float()
        labels_clamped = labels.clamp(min=0.0)
        probs = torch.sigmoid(logits)

        xs_pos = probs
        loss_pos = F.binary_cross_entropy_with_logits(
            logits, torch.ones_like(logits),
            pos_weight=self.pos_weight, reduction='none'
        )
        focal_pos = (1.0 - xs_pos) ** self.gamma_pos

        xs_neg = torch.clamp(probs - self.prob_margin, min=0.0)
        loss_neg = -torch.log(torch.clamp(1.0 - xs_neg, min=1e-8))
        focal_neg = xs_neg ** self.gamma_neg

        loss = torch.where(
            labels_clamped >= 0.5,
            loss_pos * focal_pos,
            loss_neg * focal_neg,
        )
        masked_loss = loss * mask
        if self.relation_weights is not None:
            masked_loss = masked_loss * self.relation_weights.unsqueeze(0)
        n_valid = mask.sum()
        if n_valid > 0:
            return masked_loss.sum() / n_valid
        return masked_loss.sum()


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class PyGGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs):
        self.data_list = []
        for g in graphs:
            data = Data(
                x=g["x"],
                edge_index=g["edge_index"],
                edge_attr=g["edge_attr"],
                y=g["edge_label"].clone(),
            )
            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# ── Utility functions ─────────────────────────────────────────────────────────

def load_rio10_test_scene_ids():
    rio10_path = os.path.join("data", "3DSSG", "rio10_test_scenes.txt")
    if not os.path.exists(rio10_path):
        return set()
    raw = open(rio10_path, "rb").read()
    if raw[:2] == b'\xff\xfe':
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig")
    return set(line.strip() for line in text.splitlines() if line.strip())


def load_all_graphs(cache_dir, exclude_scene_ids=None):
    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
    graphs = []
    n_excluded = 0
    for fname in _tqdm(files, desc="Loading", unit="graph"):
        if exclude_scene_ids:
            scene_id = fname.replace("_v1_3rscan_splat.pt", "").replace(".pt", "")
            if scene_id in exclude_scene_ids:
                n_excluded += 1
                continue
        g = torch.load(os.path.join(cache_dir, fname), weights_only=False)
        if g["edge_label"].shape[1] == 12:
            g["edge_label"] = g["edge_label"][:, LEGACY_12_TO_10_COLS]
        graphs.append(g)
    print(f"Loaded {len(graphs)} graphs (excluded {n_excluded} RIO10 test scenes).")
    return graphs


def compute_class_weights(graphs, cap=8.0) -> torch.Tensor:
    pos_counts = torch.zeros(NUM_RELATIONS)
    valid_counts = torch.zeros(NUM_RELATIONS)
    for g in graphs:
        labels = g["edge_label"]
        for rel_idx in range(NUM_RELATIONS):
            col = labels[:, rel_idx]
            valid_mask = col >= 0
            pos_counts[rel_idx] += (col[valid_mask] >= 0.5).sum().float()
            valid_counts[rel_idx] += valid_mask.sum().float()
    pos_counts = torch.clamp(pos_counts, min=1.0)
    neg_counts = torch.clamp(valid_counts - pos_counts, min=1.0)
    weights = neg_counts / pos_counts
    return torch.clamp(weights, max=cap)


def split_by_scene(graphs, train_frac=0.85, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(graphs))
    rng.shuffle(indices)
    train_size = int(len(graphs) * train_frac)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    return [graphs[i] for i in train_idx], [graphs[i] for i in val_idx]


def augment_training(train_graphs, augment_factor=6, jitter=True):
    aug_names = ["rotate_90", "rotate_180", "rotate_270", "flip_x", "flip_y"]
    if jitter:
        aug_names.append("jitter")
    selected = aug_names[:augment_factor]
    augmented = []
    for g in train_graphs:
        augmented.append(g)
        for aug_name in selected:
            augmented.append(augment_graph(g, aug_name))
    # Rare-relation oversampling
    rare_indices = [int(Relation.ATTACHED_TO), int(Relation.ADJACENT_TO)]
    extra_augs = ["flip_x", "flip_y"]
    for g in train_graphs:
        labels = g["edge_label"]
        is_rare = any(
            (labels[:, ri] == 1).any().item()
            for ri in rare_indices if labels.shape[1] > ri
        )
        if is_rare:
            for aug_name in extra_augs:
                augmented.append(augment_graph(g, aug_name))
    print(f"  Training: {len(train_graphs)} original -> {len(augmented)} augmented")
    return augmented


def subsample_edges(edge_label, target_pos_rate=0.30):
    pos_mask = (edge_label >= 0.5).any(dim=-1)
    neg_mask = ~pos_mask
    n_pos = pos_mask.sum().item()
    if n_pos == 0:
        return torch.arange(edge_label.shape[0])
    n_neg_target = int(n_pos * (1 - target_pos_rate) / target_pos_rate)
    neg_indices = neg_mask.nonzero(as_tuple=True)[0]
    if len(neg_indices) <= n_neg_target:
        return torch.arange(edge_label.shape[0])
    perm = torch.randperm(len(neg_indices))[:n_neg_target]
    pos_indices = pos_mask.nonzero(as_tuple=True)[0]
    return torch.cat([pos_indices, neg_indices[perm]])


def apply_label_smoothing(labels):
    smoothed = labels.clone()
    for rel_idx, smooth in LABEL_SMOOTHING.items():
        col = smoothed[:, rel_idx]
        valid_mask = col >= 0
        col_valid = col[valid_mask]
        smoothed[valid_mask, rel_idx] = col_valid * (1 - smooth) + (1 - col_valid) * smooth
    return smoothed


def precompute_reverse_edge_map(edge_index):
    src, dst = edge_index[0], edge_index[1]
    n_edges = src.shape[0]
    max_node = int(max(src.max().item(), dst.max().item())) + 1
    fwd_keys = src.long() * max_node + dst.long()
    rev_keys = dst.long() * max_node + src.long()
    sorted_order = torch.argsort(fwd_keys)
    sorted_keys = fwd_keys[sorted_order]
    positions = torch.searchsorted(sorted_keys, rev_keys).clamp(max=n_edges - 1)
    matched = sorted_keys[positions] == rev_keys
    rev_map = sorted_order[positions]
    if not matched.all():
        rev_map[~matched] = torch.arange(n_edges, device=src.device)[~matched]
    return rev_map


def inverse_consistency_loss(logits, edge_index, inverse_pairs, rev_edge_map):
    if rev_edge_map is None:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    probs = torch.sigmoid(logits)
    probs_rev = probs[rev_edge_map]
    rel_indices = torch.tensor([p[0] for p in inverse_pairs], device=logits.device)
    inv_indices = torch.tensor([p[1] for p in inverse_pairs], device=logits.device)
    p_fwd = probs[:, rel_indices]
    p_rev = probs_rev[:, inv_indices]
    return ((p_fwd - p_rev) ** 2).mean() * 0.1


# ── Scheduler ─────────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1.0 + np.cos(np.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale


# ── Recall@K computation ─────────────────────────────────────────────────────

def compute_recall_at_k(probs, labels, k):
    """
    For each edge with at least one positive GT relation,
    check if each positive GT relation is in the top-K predictions.
    """
    hits = 0
    total = 0
    for edge_idx in range(probs.shape[0]):
        edge_labels = labels[edge_idx]
        edge_probs = probs[edge_idx]
        positive_rels = np.where(edge_labels >= 0.5)[0]
        if len(positive_rels) == 0:
            continue
        # Exclude masked (-1) relations from ranking
        valid_mask = edge_labels >= 0
        if not valid_mask.any():
            continue
        # Rank only among valid relations
        valid_indices = np.where(valid_mask)[0]
        valid_probs = edge_probs[valid_indices]
        top_k_valid = valid_indices[np.argsort(valid_probs)[::-1][:k]]
        for rel_idx in positive_rels:
            total += 1
            if rel_idx in top_k_valid:
                hits += 1
    return hits / max(total, 1), hits, total


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, val_loader, device):
    """Evaluate model: returns macro_f1, per-relation F1, R@3, R@5."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_labels.append(batch.y.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Per-relation F1 at threshold 0.5
    f1_per_rel = np.zeros(NUM_RELATIONS)
    for rel_idx in range(NUM_RELATIONS):
        valid_mask = all_labels[:, rel_idx] >= 0
        if valid_mask.sum() == 0:
            continue
        preds_valid = (all_probs[valid_mask, rel_idx] >= 0.5).numpy().astype(int)
        labels_valid = (all_labels[valid_mask, rel_idx] >= 0.5).numpy().astype(int)
        f1_per_rel[rel_idx] = f1_score(labels_valid, preds_valid, zero_division=0)

    macro_f1 = float(np.mean(f1_per_rel))

    # Recall@K
    probs_np = all_probs.numpy()
    labels_np = all_labels.numpy()
    r3, _, _ = compute_recall_at_k(probs_np, labels_np, 3)
    r5, _, _ = compute_recall_at_k(probs_np, labels_np, 5)

    return macro_f1, f1_per_rel, r3, r5


# ── Single variant training loop ─────────────────────────────────────────────

def train_variant(variant_name, model, train_loader, val_loader, device,
                  pos_weight, save_path):
    """Train a single model variant. Returns best val metrics."""
    print(f"\n{'='*70}")
    print(f"  Training: {variant_name}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"{'='*70}")

    model = model.to(device)

    # Per-relation loss weight tensor
    rel_weights = torch.ones(NUM_RELATIONS)
    for rel_idx, w in RELATION_LOSS_WEIGHTS.items():
        rel_weights[rel_idx] = w
    rel_weights = rel_weights.to(device)

    criterion = AsymmetricFocalBCELoss(
        pos_weight=pos_weight,
        gamma_neg=CONFIG["focal_gamma_neg"],
        gamma_pos=CONFIG["focal_gamma_pos"],
        prob_margin=CONFIG["asl_prob_margin"],
        relation_weights=rel_weights,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = WarmupCosineScheduler(
        optimizer, CONFIG["warmup_epochs"], CONFIG["epochs"]
    )

    best_val_f1 = 0.0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in _tqdm(train_loader, desc=f"  Ep {epoch+1:02d}", unit="batch",
                           leave=False):
            batch = batch.to(device)

            # Subsample edges
            keep_idx = subsample_edges(batch.y)
            if len(keep_idx) < batch.y.shape[0]:
                edge_mask = torch.zeros(batch.y.shape[0], dtype=torch.bool)
                edge_mask[keep_idx] = True
                sub_edge_index = batch.edge_index[:, edge_mask]
                sub_edge_attr = batch.edge_attr[edge_mask]
                sub_labels = batch.y[edge_mask]
            else:
                sub_edge_index = batch.edge_index
                sub_edge_attr = batch.edge_attr
                sub_labels = batch.y

            sub_labels_smooth = apply_label_smoothing(sub_labels)

            # Forward
            logits = model(batch.x, sub_edge_index, sub_edge_attr)

            # Losses
            bce_loss = criterion(logits, sub_labels_smooth)
            rev_map = precompute_reverse_edge_map(sub_edge_index)
            inv_loss = inverse_consistency_loss(
                logits, sub_edge_index, INVERSE_PAIRS, rev_map
            )
            metric_loss = model.metric_reg()
            total_loss = bce_loss + inv_loss + CONFIG["metric_reg_weight"] * metric_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches += 1

        scheduler.step(epoch)
        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation (every epoch)
        val_f1, _, _, _ = evaluate_model(model, val_loader, device)

        if (epoch + 1) % 10 == 0 or val_f1 > best_val_f1:
            print(f"  Epoch {epoch+1:02d}: loss={avg_loss:.4f}  val_F1={val_f1:.4f}"
                  f"  best={best_val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    elapsed = time.time() - start_time
    print(f"  Training complete in {elapsed:.0f}s. Best val F1: {best_val_f1:.4f}")

    # Load best model and compute final metrics
    model.load_state_dict(torch.load(save_path, weights_only=True))
    macro_f1, f1_per_rel, r3, r5 = evaluate_model(model, val_loader, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "variant": variant_name,
        "macro_f1": macro_f1,
        "r_at_3": r3,
        "r_at_5": r5,
        "params": n_params,
        "f1_per_rel": f1_per_rel,
        "train_time_s": elapsed,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = CONFIG["device"]
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    print(f"{'='*70}")
    print("GeoKAN ABLATION STUDY — 3 Variants on 3RScan")
    print(f"{'='*70}")
    print(f"Config: {json.dumps(CONFIG, indent=2, default=str)}")

    # ── Load data ─────────────────────────────────────────────────────────────
    rio10_ids = load_rio10_test_scene_ids()
    graphs = load_all_graphs(CONFIG["cache_dir"], exclude_scene_ids=rio10_ids)
    if len(graphs) == 0:
        print("ERROR: No graphs loaded.")
        return

    train_graphs, val_graphs = split_by_scene(graphs, CONFIG["train_frac"])
    print(f"  Split: {len(train_graphs)} train / {len(val_graphs)} val")

    train_graphs_aug = augment_training(
        train_graphs, CONFIG["augment_factor"], CONFIG["augment_jitter"]
    )

    pos_weight = compute_class_weights(train_graphs, CONFIG["pos_weight_cap"]).to(device)

    train_dataset = PyGGraphDataset(train_graphs_aug)
    val_dataset = PyGGraphDataset(val_graphs)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
                            shuffle=False)

    # ── Define variants ───────────────────────────────────────────────────────
    rbf_ablation_path = os.path.join(CONFIG["save_dir"], "geokan_rbf_ablation.pt")

    variants = [
        (
            "GeoKAN-RBF",
            GeoKANRelationGNN(
                node_feat_dim=CONFIG["node_feat_dim"],
                edge_feat_dim=CONFIG["edge_feat_dim"],
                hidden_dim=CONFIG["hidden_dim"],
                num_relations=NUM_RELATIONS,
                dropout=CONFIG["dropout"],
            ),
            rbf_ablation_path,
        ),
        (
            "GeoKAN-Gamma",
            GeoKANGammaRelationGNN(
                node_feat_dim=CONFIG["node_feat_dim"],
                edge_feat_dim=CONFIG["edge_feat_dim"],
                hidden_dim=CONFIG["hidden_dim"],
                num_relations=NUM_RELATIONS,
                dropout=CONFIG["dropout"],
            ),
            os.path.join(CONFIG["save_dir"], "geokan_gamma_ablation.pt"),
        ),
        (
            "GeoKAN-Wavelet",
            GeoKANWaveletRelationGNN(
                node_feat_dim=CONFIG["node_feat_dim"],
                edge_feat_dim=CONFIG["edge_feat_dim"],
                hidden_dim=CONFIG["hidden_dim"],
                num_relations=NUM_RELATIONS,
                dropout=CONFIG["dropout"],
            ),
            os.path.join(CONFIG["save_dir"], "geokan_wavelet_ablation.pt"),
        ),
    ]

    # ── Train each variant (with resume support) ────────────────────────────
    results = []
    for variant_name, model, save_path in variants:
        # Check if this variant already has a completed checkpoint
        done_marker = save_path + ".done"
        if os.path.exists(done_marker) and os.path.exists(save_path):
            print(f"\n{'='*70}")
            print(f"  SKIPPING: {variant_name} (already completed)")
            print(f"  Loading checkpoint from: {save_path}")
            print(f"{'='*70}")
            model = model.to(device)
            model.load_state_dict(torch.load(save_path, weights_only=True))
            macro_f1, f1_per_rel, r3, r5 = evaluate_model(model, val_loader, device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # Load saved training time
            meta_path = save_path + ".meta.json"
            train_time = 0.0
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    train_time = json.load(f).get("train_time_s", 0.0)
            results.append({
                "variant": variant_name,
                "macro_f1": macro_f1,
                "r_at_3": r3,
                "r_at_5": r5,
                "params": n_params,
                "f1_per_rel": f1_per_rel,
                "train_time_s": train_time,
            })
            continue

        result = train_variant(
            variant_name, model, train_loader, val_loader,
            device, pos_weight, save_path
        )
        results.append(result)

        # Mark as done so we can resume if interrupted
        with open(done_marker, "w") as f:
            f.write("done")
        # Save metadata
        meta_path = save_path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump({"train_time_s": result["train_time_s"]}, f)

    # ── Print ablation table ──────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("ABLATION RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Variant':<20} | {'Macro F1':>9} | {'Pred R@3':>9} | "
          f"{'Pred R@5':>9} | {'Params':>10} | {'Time':>6}")
    print(f"{'-'*20}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}-+-{'-'*6}")

    for r in results:
        print(f"{r['variant']:<20} | {r['macro_f1']:>9.4f} | "
              f"{r['r_at_3']*100:>8.1f}% | {r['r_at_5']*100:>8.1f}% | "
              f"{r['params']:>10,} | {r['train_time_s']:>5.0f}s")

    # ── Per-relation breakdown ────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("PER-RELATION F1 BREAKDOWN")
    print(f"{'='*70}")
    header = f"{'Relation':<20}"
    for r in results:
        header += f" | {r['variant']:>14}"
    print(header)
    print("-" * len(header))

    for rel_idx in range(NUM_RELATIONS):
        row = f"{RELATION_NAMES[rel_idx]:<20}"
        for r in results:
            row += f" | {r['f1_per_rel'][rel_idx]:>14.3f}"
        print(row)

    # Macro row
    row = f"{'MACRO F1':<20}"
    for r in results:
        row += f" | {r['macro_f1']:>14.4f}"
    print("-" * len(header))
    print(row)

    # ── Save results to JSON ──────────────────────────────────────────────────
    results_json = []
    for r in results:
        results_json.append({
            "variant": r["variant"],
            "macro_f1": r["macro_f1"],
            "pred_r_at_3": r["r_at_3"],
            "pred_r_at_5": r["r_at_5"],
            "params": r["params"],
            "train_time_s": r["train_time_s"],
            "f1_per_rel": {RELATION_NAMES[i]: float(r["f1_per_rel"][i])
                           for i in range(NUM_RELATIONS)},
        })

    json_path = os.path.join(CONFIG["save_dir"], "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    print(f"\n{'='*70}")
    print("Ablation study complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
