"""
Train GeoKAN-based Relation Predictor on 3RScan graph cache.

Architecture: GeoKANRelationGNN (GATv2 context + dual GeoKAN heads)
Dataset: Pre-cached 3RScan scene graphs (D:/logicsplat_data/3rscan_graph_cache/)
Split: 400 train / 80 val / rest test (seed=42, scene-level split)
Augmentation: rotations + flips (training only)
Loss: ASL (Asymmetric Loss, prob_margin=0.05) + 1e-4 * metric_reg
Schedule: 5-epoch warmup + cosine decay
Early stopping: patience=15 on val macro F1

Usage:
    python train_geokan.py
"""
import sys
sys.path.insert(0, ".")

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from geokan_relation import GeoKANRelationGNN, CONTACT_INDICES, DIRECTIONAL_INDICES
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, Relation, LEGACY_12_TO_10_COLS
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
    "model_name":       "geokan_v4",
    "node_feat_dim":    10,
    "edge_feat_dim":    22,
    "hidden_dim":       128,
    "dropout":          0.2,
    "lr":               2e-3,
    "weight_decay":     1e-4,
    "epochs":           100,
    "batch_size":       8,
    "device":           "cuda",
    "save_dir":         "models",
    "cache_dir":        "D:/logicsplat_data/3rscan_graph_cache",
    "train_frac":       0.85,
    "augment_factor":   6,
    "augment_jitter":   True,
    # ASL loss (Baruch et al. 2021): gamma_pos=0 gives full gradient to all positives
    # (critical for extreme imbalance); gamma_neg=4 aggressively down-weights easy
    # negatives. prob_margin=0 (was 0.05 in v3 — removing it improves training stability).
    "focal_gamma_neg":  4.0,
    "focal_gamma_pos":  0.0,
    "asl_prob_margin":  0.0,
    "patience":         15,
    "metric_reg_weight": 1e-4,
    "max_grad_norm":    2.0,
    "warmup_epochs":    5,
    "pos_weight_cap":   8.0,
}


# ── Inverse relation pairs for consistency loss ───────────────────────────────
# Uses new 10-class indices (no INSIDE/HANGING_FROM)
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

# ── Per-relation loss weights (multipliers on top of pos_weight) ──────────────
# INSIDE and HANGING_FROM removed from schema — no longer needed here.
RELATION_LOSS_WEIGHTS = {
    # attached_to and adjacent_to now have soft injection but remain the hardest
    # classes — keep elevated weights to compensate for lower confidence labels.
    int(Relation.ATTACHED_TO):  1.8,   # injected at 0.60, still sparse — highest boost
    int(Relation.ADJACENT_TO):  1.5,   # injected at 0.55 strict threshold — needs boost
    int(Relation.ON_TOP_OF):    1.2,   # soft-injected at 0.75
    int(Relation.UNDER):        1.2,
}

# ── Per-relation label smoothing values ───────────────────────────────────────
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


# ── Asymmetric Focal BCE Loss ─────────────────────────────────────────────────

class AsymmetricFocalBCELoss(nn.Module):
    """
    Asymmetric Loss (ASL, Baruch et al. 2021) with:
      - Separate focal exponents for positives (gamma_pos) and negatives (gamma_neg)
      - Probability shift m for negatives: easy negatives (p < m) are discarded
        before computing the focal weight. This is critical for 1:1000+ imbalance
        (e.g. inside class: 90 pos / 154980 valid) — it prevents gradient from
        being completely dominated by easy-negative background.
      - Masked labels (-1 = ignore, used for directional relation masking)
      - Per-relation loss weight multipliers (RELATION_LOSS_WEIGHTS)

    Reference: https://arxiv.org/abs/2009.14119
    Recommended settings for severe imbalance: gamma_pos=0, gamma_neg=4, m=0.05
    """

    def __init__(self, pos_weight, contact_indices, gamma_neg=4.0, gamma_pos=0.0,
                 prob_margin=0.05, relation_weights=None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.contact_indices = contact_indices
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

        # ── Positive side ──
        # Standard focal loss for positive examples
        xs_pos = probs
        loss_pos = F.binary_cross_entropy_with_logits(
            logits, torch.ones_like(logits), pos_weight=self.pos_weight, reduction='none'
        )
        focal_pos = (1.0 - xs_pos) ** self.gamma_pos

        # ── Negative side with probability shift ──
        # Shift: discard easy negatives (p < m) by clipping to max(p - m, 0).
        # This prevents the ~99.9% of trivially-correct negatives from
        # swamping the gradient for rare positive classes like `inside`.
        xs_neg = torch.clamp(probs - self.prob_margin, min=0.0)
        loss_neg = -torch.log(torch.clamp(1.0 - xs_neg, min=1e-8))
        focal_neg = xs_neg ** self.gamma_neg

        # ── Combine ──
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


# ── PyG Dataset wrapper ───────────────────────────────────────────────────────

class PyGGraphDataset(torch.utils.data.Dataset):
    """Wraps list of graph dicts into PyG Data objects."""

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


# ── Utilities ─────────────────────────────────────────────────────────────────

def subsample_edges(edge_label, target_pos_rate=0.30):
    """
    Negative edge subsampling during training.
    Keep ALL positive edges, randomly sample negatives to achieve ~30% positive rate.
    Handles -1 (ignore) labels. Treats soft labels (e.g. 0.75) as positive.
    """
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
    keep = torch.cat([pos_indices, neg_indices[perm]])
    return keep


def apply_label_smoothing(labels):
    """Per-relation label smoothing. Preserves -1 (ignore) labels."""
    smoothed = labels.clone()
    for rel_idx, smooth in LABEL_SMOOTHING.items():
        col = smoothed[:, rel_idx]
        valid_mask = col >= 0
        col_valid = col[valid_mask]
        smoothed[valid_mask, rel_idx] = col_valid * (1 - smooth) + (1 - col_valid) * smooth
    return smoothed


def inverse_consistency_loss(logits, edge_index, inverse_pairs, rev_edge_map):
    """
    Inverse consistency regularization using precomputed reverse-edge mapping.
    MSE between p(R(A,B)) and p(R_inv(B,A)), lambda=0.1.
    """
    if rev_edge_map is None:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    probs = torch.sigmoid(logits)
    probs_rev = probs[rev_edge_map]

    rel_indices = torch.tensor([p[0] for p in inverse_pairs], device=logits.device)
    inv_indices = torch.tensor([p[1] for p in inverse_pairs], device=logits.device)

    p_fwd = probs[:, rel_indices]
    p_rev = probs_rev[:, inv_indices]

    loss = ((p_fwd - p_rev) ** 2).mean()
    return loss * 0.1


def precompute_reverse_edge_map(edge_index):
    """
    Precompute a mapping from each edge to its reverse edge.
    Returns (E,) long tensor or None if not all edges have reverses.
    """
    src, dst = edge_index[0], edge_index[1]
    n_edges = src.shape[0]

    max_node = int(max(src.max().item(), dst.max().item())) + 1
    fwd_keys = src.long() * max_node + dst.long()
    rev_keys = dst.long() * max_node + src.long()

    sorted_order = torch.argsort(fwd_keys)
    sorted_keys = fwd_keys[sorted_order]

    positions = torch.searchsorted(sorted_keys, rev_keys)
    positions = positions.clamp(max=n_edges - 1)

    matched = sorted_keys[positions] == rev_keys
    if not matched.all():
        rev_map = sorted_order[positions]
        rev_map[~matched] = torch.arange(n_edges, device=src.device)[~matched]
    else:
        rev_map = sorted_order[positions]

    return rev_map


def load_rio10_test_scene_ids():
    """Load the 46 RIO10 test scene IDs for exclusion."""
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
    """Load all .pt graph files from cache directory."""
    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
    graphs = []
    n_excluded = 0
    print(f"Loading graphs from {cache_dir} ({len(files)} files)...")
    if exclude_scene_ids:
        print(f"  Excluding {len(exclude_scene_ids)} RIO10 test scene IDs")
    for fname in _tqdm(files, desc="Loading", unit="graph"):
        if exclude_scene_ids:
            scene_id = fname.replace("_v1_3rscan_splat.pt", "").replace(".pt", "")
            if scene_id in exclude_scene_ids:
                n_excluded += 1
                continue
        g = torch.load(os.path.join(cache_dir, fname), weights_only=False)
        assert g["x"].shape[1] == 10, f"{fname}: node feats {g['x'].shape[1]} != 10"
        assert g["edge_attr"].shape[1] == 22, f"{fname}: edge feats {g['edge_attr'].shape[1]} != 22"
        # Convert legacy 12-dim labels (with INSIDE=col2, HANGING_FROM=col4)
        # to current 10-dim schema by dropping those two columns.
        if g["edge_label"].shape[1] == 12:
            g["edge_label"] = g["edge_label"][:, LEGACY_12_TO_10_COLS]
        assert g["edge_label"].shape[1] == NUM_RELATIONS, \
            f"{fname}: edge_label dim {g['edge_label'].shape[1]} != {NUM_RELATIONS}"
        graphs.append(g)
    print(f"Loaded {len(graphs)} graphs (excluded {n_excluded} RIO10 test scenes).")
    return graphs


def compute_class_weights(graphs, cap=8.0) -> torch.Tensor:
    """
    Compute per-relation pos_weight.
    Cap raised from 5.0 → 8.0 to allow stronger signal for rare classes
    (inside, hanging_from) without completely dominating the gradient.
    """
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
    weights = torch.clamp(weights, max=cap)
    return weights


def _binarize_labels(labels: np.ndarray) -> np.ndarray:
    """Threshold soft labels (e.g. 0.75 rule-injected) to binary for F1 computation."""
    return (labels >= 0.5).astype(int)


def split_by_scene(graphs, train_frac=0.85, seed=42):
    """Split graphs into train/val only (no test — RIO10 is external test set).
    
    Returns train, val lists.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(graphs))
    rng.shuffle(indices)

    train_size = int(len(graphs) * train_frac)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train = [graphs[i] for i in train_idx]
    val = [graphs[i] for i in val_idx]
    return train, val


def augment_training(train_graphs, augment_factor=6, jitter=True):
    """
    Apply rotations + flips to training graphs.

    augment_factor controls how many augmentations are selected from the list:
      6 = rotate_90, rotate_180, rotate_270, flip_x, flip_y, jitter  (7× expansion)
      4 = rotate_180, flip_x, flip_y, jitter  (5× expansion)

    Also oversamples graphs containing rare relations (inside, hanging_from) with
    extra augmented copies to give the model more positive examples for those classes.
    """
    aug_names = ["rotate_90", "rotate_180", "rotate_270", "flip_x", "flip_y"]
    if jitter:
        aug_names.append("jitter")
    selected = aug_names[:augment_factor]

    augmented = []
    for g in _tqdm(train_graphs, desc="Augmenting", unit="graph"):
        augmented.append(g)
        for aug_name in selected:
            augmented.append(augment_graph(g, aug_name))

    # Rare-relation oversampling: extra augmented copies for graphs with
    # attached_to or adjacent_to labels (hardest to predict in 10-class schema).
    rare_indices = [int(Relation.ATTACHED_TO), int(Relation.ADJACENT_TO)]
    extra_augs = ["flip_x", "flip_y"]
    n_rare_extra = 0
    for g in train_graphs:
        labels = g["edge_label"]
        is_rare = any(
            (labels[:, ri] == 1).any().item()
            for ri in rare_indices
            if labels.shape[1] > ri
        )
        if is_rare:
            for aug_name in extra_augs:
                augmented.append(augment_graph(g, aug_name))
                n_rare_extra += 1

    print(f"  Training: {len(train_graphs)} original -> {len(augmented)} after augmentation "
          f"(x{1 + len(selected)}, +{n_rare_extra} rare-relation extras)")
    return augmented


# ── Hierarchical constraints (fixed: no on_top_of suppression bug) ────────────

def apply_hierarchical_constraints(preds, probs):
    """
    Enforce physical constraints.
    
    NOTE: The "NOT higher_than → NOT on_top_of" rule is DISABLED because it
    kills on_top_of recall. In 3DSSG, objects can be on_top_of without being
    annotated as higher_than (e.g., a book on a table at the same height).
    
    Active constraints:
    1. left_of vs right_of — keep higher confidence
    2. in_front_of vs behind — keep higher confidence
    3. higher_than vs lower_than — keep higher confidence
    """
    IDX = {r: int(r) for r in Relation}
    preds = preds.clone()

    # DISABLED: NOT higher_than → NOT on_top_of (kills on_top_of recall)
    # mask_not_higher = preds[:, IDX[Relation.HIGHER_THAN]] == 0
    # preds[mask_not_higher, IDX[Relation.ON_TOP_OF]] = 0

    # DISABLED: NOT lower_than → NOT under (same issue)
    # mask_not_lower = preds[:, IDX[Relation.LOWER_THAN]] == 0
    # preds[mask_not_lower, IDX[Relation.UNDER]] = 0

    # Exclusivity: left vs right
    conflict_lr = ((preds[:, IDX[Relation.LEFT_OF]] == 1) &
                   (preds[:, IDX[Relation.RIGHT_OF]] == 1))
    if conflict_lr.any():
        keep_left = (probs[conflict_lr, IDX[Relation.LEFT_OF]] >=
                     probs[conflict_lr, IDX[Relation.RIGHT_OF]])
        idx_conflict = conflict_lr.nonzero(as_tuple=True)[0]
        for i, ci in enumerate(idx_conflict):
            if keep_left[i]:
                preds[ci, IDX[Relation.RIGHT_OF]] = 0
            else:
                preds[ci, IDX[Relation.LEFT_OF]] = 0

    # Exclusivity: front vs behind
    conflict_fb = ((preds[:, IDX[Relation.IN_FRONT_OF]] == 1) &
                   (preds[:, IDX[Relation.BEHIND]] == 1))
    if conflict_fb.any():
        keep_front = (probs[conflict_fb, IDX[Relation.IN_FRONT_OF]] >=
                      probs[conflict_fb, IDX[Relation.BEHIND]])
        idx_conflict = conflict_fb.nonzero(as_tuple=True)[0]
        for i, ci in enumerate(idx_conflict):
            if keep_front[i]:
                preds[ci, IDX[Relation.BEHIND]] = 0
            else:
                preds[ci, IDX[Relation.IN_FRONT_OF]] = 0

    # Exclusivity: higher vs lower
    conflict_hl = ((preds[:, IDX[Relation.HIGHER_THAN]] == 1) &
                   (preds[:, IDX[Relation.LOWER_THAN]] == 1))
    if conflict_hl.any():
        keep_higher = (probs[conflict_hl, IDX[Relation.HIGHER_THAN]] >=
                       probs[conflict_hl, IDX[Relation.LOWER_THAN]])
        idx_conflict = conflict_hl.nonzero(as_tuple=True)[0]
        for i, ci in enumerate(idx_conflict):
            if keep_higher[i]:
                preds[ci, IDX[Relation.LOWER_THAN]] = 0
            else:
                preds[ci, IDX[Relation.HIGHER_THAN]] = 0

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
    all_labels = torch.cat(all_labels, dim=0)

    thresholds = {}
    print("\n  Per-relation threshold tuning (validation set):")
    for rel_idx in range(NUM_RELATIONS):
        best_f1 = 0.0
        best_thresh = 0.5

        valid_mask = all_labels[:, rel_idx] >= 0
        if valid_mask.sum() == 0:
            thresholds[rel_idx] = 0.5
            print(f"    {RELATION_NAMES[rel_idx]:20s} thresh=0.50  F1=0.000 (no valid)")
            continue

        probs = torch.sigmoid(all_logits[valid_mask, rel_idx]).numpy()
        labels_valid = _binarize_labels(all_labels[valid_mask, rel_idx].numpy())

        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (probs >= thresh).astype(int)
            f1 = f1_score(labels_valid, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)

        thresholds[rel_idx] = best_thresh
        print(f"    {RELATION_NAMES[rel_idx]:20s} thresh={best_thresh:.2f}  F1={best_f1:.3f}")

    return thresholds


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, split_name="val", thresholds=None,
             use_constraints=False):
    """Multi-label evaluation. Returns (macro_f1, per_relation_f1_dict)."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_labels.append(batch.y.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Apply thresholds
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

    # Compute per-relation F1, masking out -1 (ignore) labels
    f1_per_rel = np.zeros(NUM_RELATIONS)
    for rel_idx in range(NUM_RELATIONS):
        valid_mask = all_labels[:, rel_idx] >= 0
        if valid_mask.sum() == 0:
            f1_per_rel[rel_idx] = 0.0
            continue
        preds_valid = all_preds[valid_mask, rel_idx].numpy()
        labels_valid = _binarize_labels(all_labels[valid_mask, rel_idx].numpy())
        f1_per_rel[rel_idx] = f1_score(labels_valid, preds_valid, zero_division=0)

    macro_f1 = float(np.mean(f1_per_rel))

    # Micro F1
    valid_mask_all = all_labels >= 0
    all_preds_valid = all_preds[valid_mask_all].numpy()
    all_labels_valid = _binarize_labels(all_labels[valid_mask_all].numpy())
    micro_f1 = (f1_score(all_labels_valid.flatten(), all_preds_valid.flatten(),
                         average="binary", zero_division=0)
                if len(all_labels_valid) > 0 else 0.0)

    tags = []
    if thresholds:
        tags.append("+tuned_thresh")
    if use_constraints:
        tags.append("+constraints")
    tag_str = f" [{', '.join(tags)}]" if tags else ""

    print(f"\n{split_name}{tag_str} -- Macro F1: {macro_f1:.4f}  Micro F1: {micro_f1:.4f}")
    for i, f1 in enumerate(f1_per_rel):
        valid_count = int((all_labels[:, i] >= 0).sum())
        pos = int((all_labels[:, i] >= 0.5).sum())
        print(f"  {RELATION_NAMES[i]:20s}  F1={f1:.3f}  pos={pos}  valid={valid_count}")

    return macro_f1, {RELATION_NAMES[i]: float(f1_per_rel[i]) for i in range(NUM_RELATIONS)}


# ── Warmup + Cosine schedule ─────────────────────────────────────────────────

class WarmupCosineScheduler:
    """Linear warmup for warmup_epochs, then cosine decay to 0."""

    def __init__(self, optimizer, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1.0 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale


# ── Main training loop ────────────────────────────────────────────────────────

def train():
    device = CONFIG["device"]

    print(f"{'='*60}")
    print("GeoKAN Relation Predictor - Training on 3RScan")
    print(f"{'='*60}")
    print(f"Config: {json.dumps(CONFIG, indent=2, default=str)}")

    # Load data (exclude RIO10 test scenes)
    rio10_ids = load_rio10_test_scene_ids()
    graphs = load_all_graphs(CONFIG["cache_dir"], exclude_scene_ids=rio10_ids)

    if len(graphs) == 0:
        print("ERROR: No graphs loaded. Check cache_dir path.")
        return

    # Split: 85/15 train/val (no test — RIO10 scenes are the external test set)
    train_graphs, val_graphs = split_by_scene(graphs, CONFIG["train_frac"])
    print(f"  Split: {len(train_graphs)} train / {len(val_graphs)} val "
          f"(no test split — RIO10 is external test set)")

    # Augment training data
    train_graphs_aug = augment_training(
        train_graphs, CONFIG["augment_factor"], CONFIG["augment_jitter"]
    )

    # Compute class weights from original training data
    pos_weight = compute_class_weights(
        train_graphs, cap=CONFIG["pos_weight_cap"]
    ).to(device)
    print(f"  Pos weights: {pos_weight.tolist()}")

    # Create datasets and loaders
    train_dataset = PyGGraphDataset(train_graphs_aug)
    val_dataset = PyGGraphDataset(val_graphs)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
                            shuffle=False)

    # Initialize model
    model = GeoKANRelationGNN(
        node_feat_dim=CONFIG["node_feat_dim"],
        edge_feat_dim=CONFIG["edge_feat_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_relations=NUM_RELATIONS,
        dropout=CONFIG["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model: GeoKANRelationGNN ({n_params:,} parameters)")

    # Per-relation loss weight tensor
    rel_weights = torch.ones(NUM_RELATIONS)
    for rel_idx, w in RELATION_LOSS_WEIGHTS.items():
        rel_weights[rel_idx] = w
    rel_weights = rel_weights.to(device)

    # Loss, optimizer, scheduler
    criterion = AsymmetricFocalBCELoss(
        pos_weight=pos_weight,
        contact_indices=CONTACT_INDICES,
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
        optimizer,
        warmup_epochs=CONFIG["warmup_epochs"],
        total_epochs=CONFIG["epochs"],
    )

    # Early stopping
    best_val_f1 = 0.0
    patience_counter = 0
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    save_path = os.path.join(CONFIG["save_dir"], "geokan_relation_v4.pt")

    # ── Training epochs ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")

    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_bce_loss = 0.0
        epoch_inv_loss = 0.0
        epoch_metric_loss = 0.0
        n_batches = 0

        for batch in _tqdm(train_loader, desc=f"Epoch {epoch+1:02d}", unit="batch"):
            batch = batch.to(device)

            # Subsample edges for class balance
            keep_idx = subsample_edges(batch.y)
            if len(keep_idx) < batch.y.shape[0]:
                # Remap edge_index for subsampled edges
                edge_mask = torch.zeros(batch.y.shape[0], dtype=torch.bool)
                edge_mask[keep_idx] = True
                sub_edge_index = batch.edge_index[:, edge_mask]
                sub_edge_attr = batch.edge_attr[edge_mask]
                sub_labels = batch.y[edge_mask]
            else:
                sub_edge_index = batch.edge_index
                sub_edge_attr = batch.edge_attr
                sub_labels = batch.y

            # Apply label smoothing
            sub_labels_smooth = apply_label_smoothing(sub_labels)

            # Forward pass
            logits = model(batch.x, sub_edge_index, sub_edge_attr)

            # BCE + focal loss
            bce_loss = criterion(logits, sub_labels_smooth)

            # Inverse consistency loss
            rev_map = precompute_reverse_edge_map(sub_edge_index)
            inv_loss = inverse_consistency_loss(
                logits, sub_edge_index, INVERSE_PAIRS, rev_map
            )

            # Metric regularization
            metric_loss = model.metric_reg()

            # Total loss
            total_loss = (bce_loss + inv_loss +
                          CONFIG["metric_reg_weight"] * metric_loss)

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), CONFIG["max_grad_norm"]
            )
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_bce_loss += bce_loss.item()
            epoch_inv_loss += inv_loss.item()
            epoch_metric_loss += metric_loss.item()
            n_batches += 1

        # Step scheduler
        scheduler.step(epoch)
        lr = optimizer.param_groups[0]['lr']

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_bce = epoch_bce_loss / max(n_batches, 1)
        avg_inv = epoch_inv_loss / max(n_batches, 1)
        avg_met = epoch_metric_loss / max(n_batches, 1)

        print(f"\n  Epoch {epoch+1:02d}/{CONFIG['epochs']} - "
              f"loss={avg_loss:.4f} (bce={avg_bce:.4f} inv={avg_inv:.4f} "
              f"metric={avg_met:.4f}) lr={lr:.6f}")

        # Validation
        val_f1, _ = evaluate(model, val_loader, device, split_name="  Val")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  * New best val F1: {best_val_f1:.4f} - saved to {save_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{CONFIG['patience']})")
            if patience_counter >= CONFIG["patience"]:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    # ── Post-training evaluation ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Post-training evaluation (best model)")
    print(f"{'='*60}")

    # Load best model
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()

    # Tune thresholds on validation set
    thresholds = tune_thresholds(model, val_loader, device)

    # Save thresholds
    thresh_path = os.path.join(CONFIG["save_dir"], "geokan_relation_v4_thresholds.json")
    with open(thresh_path, "w") as f:
        json.dump({str(k): v for k, v in thresholds.items()}, f, indent=2)
    print(f"  Thresholds saved to {thresh_path}")

    # Evaluate on val set with tuned thresholds + constraints
    print("\n  Val set (tuned thresholds):")
    evaluate(model, val_loader, device, split_name="  Val",
             thresholds=thresholds)

    print("\n  Val set (tuned + constraints):")
    val_f1_final, val_per_rel = evaluate(
        model, val_loader, device, split_name="  Val",
        thresholds=thresholds, use_constraints=True
    )

    # ── Compare with current GAT model ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Comparison with current GAT model (on same val set)")
    print(f"{'='*60}")

    gat_model_path = os.path.join(CONFIG["save_dir"], "relation_gnn_v8_3rscan_instance.pt")
    gat_thresh_path = os.path.join(CONFIG["save_dir"],
                                   "relation_gnn_v8_3rscan_instance_thresholds.json")

    if os.path.exists(gat_model_path) and os.path.exists(gat_thresh_path):
        from src.models.relation_gnn import RelationGNN

        gat_model = RelationGNN(
            node_feat_dim=CONFIG["node_feat_dim"],
            edge_feat_dim=CONFIG["edge_feat_dim"],
            hidden_dim=256,
            num_relations=NUM_RELATIONS,
            dropout=0.3,
        ).to(device)
        gat_model.load_state_dict(torch.load(gat_model_path, weights_only=True))

        with open(gat_thresh_path) as f:
            gat_thresholds = {int(k): v for k, v in json.load(f).items()}

        print("\n  GAT model (val set, tuned + constraints):")
        gat_f1, gat_per_rel = evaluate(
            gat_model, val_loader, device, split_name="  GAT Val",
            thresholds=gat_thresholds, use_constraints=True
        )

        # Side-by-side comparison
        print(f"\n  {'Relation':<20s} {'GAT F1':>8s} {'GeoKAN F1':>10s} {'Delta':>8s}")
        print(f"  {'-'*48}")
        for i in range(NUM_RELATIONS):
            name = RELATION_NAMES[i]
            gat_v = gat_per_rel.get(name, 0.0)
            geo_v = val_per_rel.get(name, 0.0)
            delta = geo_v - gat_v
            marker = "^" if delta > 0.01 else ("v" if delta < -0.01 else "~")
            print(f"  {name:<20s} {gat_v:>8.3f} {geo_v:>10.3f} {delta:>+7.3f} {marker}")
        print(f"  {'-'*48}")
        print(f"  {'MACRO':<20s} {gat_f1:>8.4f} {val_f1_final:>10.4f} "
              f"{val_f1_final - gat_f1:>+7.4f}")
    else:
        print("  GAT model not found - skipping comparison.")
        print(f"  (Expected: {gat_model_path})")

    print(f"\n{'='*60}")
    print(f"Training complete. Model saved to: {save_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
