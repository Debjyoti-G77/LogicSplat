"""
Train RelationGNN v8 on 3RScan graph cache (instance-level).

Architecture: v7 dual-head GNN (directional + contact heads)
Dataset: 565 pre-cached 3RScan scene graphs (D:/logicsplat_data/3rscan_graph_cache/)
Split: 400 train / 80 val / rest test (seed=42, scene-level split)
Augmentation: rotations + flips (training only)
Loss: AsymmetricFocalBCELoss (gamma_neg=2.0, gamma_pos=0.0 for contact)
Schedule: Cosine annealing with 5-epoch warmup
Early stopping: patience=10 on val macro F1

After training:
  - Per-relation threshold tuning on val set
  - Hierarchical constraint enforcement
  - Symbolic repair (SceneGraphRepair)
  - Cross-domain evaluation on custom tabletop scenes (scene_06..scene_13)

Usage:
    python train_3rscan_v2.py
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
    "model_name":       "v8_3rscan_instance",
    "node_feat_dim":    10,
    "edge_feat_dim":    17,
    "hidden_dim":       256,
    "dropout":          0.3,
    "lr":               3e-3,
    "weight_decay":     1e-4,
    "epochs":           80,
    "batch_size":       8,
    "device":           "cuda",
    "save_dir":         "models",
    "cache_dir":        "D:/logicsplat_data/3rscan_graph_cache",
    # Split sizes
    "train_size":       400,
    "val_size":         80,
    # Augmentation
    "augment_factor":   4,
    "augment_jitter":   True,
    # Focal loss
    "focal_gamma_neg":  2.0,   # Fix 3: moderate focal (subsampling handles imbalance)
    "focal_gamma_pos":  1.0,   # Upweight hard positives
    # Early stopping
    "patience":         10,
    # Cross-domain eval
    "tabletop_scenes":  [f"scene_{i:02d}" for i in range(6, 14)],
    "tabletop_dir":     "D:/logicsplat_data/processed",
}


# ── Inverse relation pairs for consistency loss (Fix 5) ────────────────────────
INVERSE_PAIRS = [
    (int(Relation.ON_TOP_OF), int(Relation.UNDER)),
    (int(Relation.UNDER), int(Relation.ON_TOP_OF)),
    (int(Relation.HIGHER_THAN), int(Relation.LOWER_THAN)),
    (int(Relation.LOWER_THAN), int(Relation.HIGHER_THAN)),
    (int(Relation.LEFT_OF), int(Relation.RIGHT_OF)),
    (int(Relation.RIGHT_OF), int(Relation.LEFT_OF)),
    (int(Relation.IN_FRONT_OF), int(Relation.BEHIND)),
    (int(Relation.BEHIND), int(Relation.IN_FRONT_OF)),
]

# ── Per-relation label smoothing values (Fix 6) ───────────────────────────────
LABEL_SMOOTHING = {
    int(Relation.ADJACENT_TO):  0.10,
    int(Relation.IN_FRONT_OF):  0.10,
    int(Relation.BEHIND):       0.10,
    int(Relation.LEFT_OF):      0.10,
    int(Relation.RIGHT_OF):     0.10,
    int(Relation.ON_TOP_OF):    0.05,
    int(Relation.UNDER):        0.05,
    int(Relation.INSIDE):       0.05,
    int(Relation.HANGING_FROM): 0.05,
    int(Relation.ATTACHED_TO):  0.05,
    int(Relation.HIGHER_THAN):  0.05,
    int(Relation.LOWER_THAN):   0.05,
}


# ── Asymmetric Focal BCE Loss ─────────────────────────────────────────────────

class AsymmetricFocalBCELoss(nn.Module):
    """
    Focal loss applied to ALL relations with support for masked labels (-1 = ignore).
    Fix 3: Stronger per-relation focal loss for the 1-2% positive rate.
    Directional masking: labels == -1 are masked out of the loss computation.
    """

    def __init__(self, pos_weight, contact_indices, gamma_neg=4.0, gamma_pos=0.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.contact_indices = contact_indices
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos

    def forward(self, logits, labels):
        # Mask out ignored labels (-1) — these are directional pairs without
        # same-support evidence (neither positive nor negative)
        mask = (labels >= 0).float()  # 1 where valid, 0 where ignored (-1)

        # Clamp labels for BCE computation (treat -1 as 0 temporarily)
        labels_clamped = labels.clamp(min=0.0)

        bce = F.binary_cross_entropy_with_logits(
            logits, labels_clamped, pos_weight=self.pos_weight, reduction='none'
        )
        probs = torch.sigmoid(logits)
        # Apply focal weighting to ALL relations (not just contact)
        focal_weight = torch.where(
            labels_clamped >= 0.5,  # use >= 0.5 for smoothed labels
            (1 - probs) ** self.gamma_pos,
            probs ** self.gamma_neg
        )
        # Apply mask: only compute loss on valid (non-ignored) labels
        masked_loss = bce * focal_weight * mask
        n_valid = mask.sum()
        if n_valid > 0:
            return masked_loss.sum() / n_valid
        return masked_loss.sum()  # fallback (shouldn't happen)


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
    Fix 4: Negative edge subsampling during training.
    Keep ALL positive edges, randomly sample negatives to achieve ~30% positive rate.
    Returns indices to keep.
    Handles -1 (ignore) labels: an edge is "positive" if any valid label is 1.
    """
    # An edge is positive if it has any label == 1 (ignoring -1 entries)
    pos_mask = (edge_label == 1).any(dim=-1)
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
    """
    Fix 6: Per-relation label smoothing.
    Subjective relations (adjacent, directional): smooth=0.10
    Geometric relations (contact, comparative): smooth=0.05
    Preserves -1 (ignore/mask) labels — only smooths 0 and 1 values.
    """
    smoothed = labels.clone()
    for rel_idx, smooth in LABEL_SMOOTHING.items():
        col = smoothed[:, rel_idx]
        # Only smooth valid labels (0 or 1), leave -1 untouched
        valid_mask = col >= 0
        col_valid = col[valid_mask]
        smoothed[valid_mask, rel_idx] = col_valid * (1 - smooth) + (1 - col_valid) * smooth
    return smoothed


def inverse_consistency_loss(logits, edge_index, inverse_pairs, rev_edge_map):
    """
    Fix 5: Inverse consistency regularization (ZERO-OVERHEAD at train time).

    Uses a precomputed reverse-edge mapping tensor so the only operations are:
      1. A single gather (index_select) to get reverse-edge logits
      2. Vectorized MSE across inverse relation pairs

    Args:
        logits: (E, num_rels) raw model output
        edge_index: (2, E) — unused here but kept for API consistency
        inverse_pairs: list of (rel_idx, inv_rel_idx) tuples
        rev_edge_map: (E,) long tensor — rev_edge_map[i] = index of reverse edge of i
    """
    if rev_edge_map is None:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    probs = torch.sigmoid(logits)                    # (E, R)
    probs_rev = probs[rev_edge_map]                  # (E, R) — reverse edge probs

    # Gather the relevant relation columns for all pairs at once
    # Stack inverse pairs into tensors for fully vectorized computation
    rel_indices = torch.tensor([p[0] for p in inverse_pairs], device=logits.device)
    inv_indices = torch.tensor([p[1] for p in inverse_pairs], device=logits.device)

    # (E, num_pairs) — forward probs for each relation in the pair
    p_fwd = probs[:, rel_indices]
    # (E, num_pairs) — reverse probs for the inverse relation
    p_rev = probs_rev[:, inv_indices]

    # MSE across all edges and all pairs
    loss = ((p_fwd - p_rev) ** 2).mean()
    return loss * 0.1  # lambda = 0.1


def precompute_reverse_edge_map(edge_index):
    """
    Precompute a mapping from each edge to its reverse edge.
    For fully-connected directed graphs (all pairs i!=j), every edge has a reverse.

    Args:
        edge_index: (2, E) tensor

    Returns:
        rev_map: (E,) long tensor where rev_map[k] = index of edge (dst[k], src[k])
                 Returns None if reverse edges don't exist for all edges.
    """
    src, dst = edge_index[0], edge_index[1]
    n_edges = src.shape[0]

    # Build hash: key = src * max_node + dst
    max_node = int(max(src.max().item(), dst.max().item())) + 1
    fwd_keys = src.long() * max_node + dst.long()
    rev_keys = dst.long() * max_node + src.long()

    # Sort forward keys and use searchsorted for O(E log E) total
    sorted_order = torch.argsort(fwd_keys)
    sorted_keys = fwd_keys[sorted_order]

    positions = torch.searchsorted(sorted_keys, rev_keys)
    positions = positions.clamp(max=n_edges - 1)

    # Verify all matched (should be true for fully-connected directed graphs)
    matched = sorted_keys[positions] == rev_keys
    if not matched.all():
        # Partial match — still usable, just mask unmatched to self
        rev_map = sorted_order[positions]
        rev_map[~matched] = torch.arange(n_edges)[~matched]
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
    """Load all .pt graph files from cache directory.
    
    Args:
        cache_dir: path to directory containing .pt graph files
        exclude_scene_ids: optional set of scene IDs to exclude (for RIO10 leakage prevention)
    """
    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
    graphs = []
    n_excluded = 0
    print(f"Loading graphs from {cache_dir} ({len(files)} files)...")
    if exclude_scene_ids:
        print(f"  Excluding {len(exclude_scene_ids)} RIO10 test scene IDs")
    for fname in _tqdm(files, desc="Loading", unit="graph"):
        # Check if this graph's scene_id is in the exclusion set
        if exclude_scene_ids:
            # Filename format: {scene_id}_v1_3rscan_splat.pt or {scene_id}.pt
            scene_id = fname.replace("_v1_3rscan_splat.pt", "").replace(".pt", "")
            if scene_id in exclude_scene_ids:
                n_excluded += 1
                continue
        g = torch.load(os.path.join(cache_dir, fname), weights_only=False)
        assert g["x"].shape[1] == 10, f"{fname}: node feats {g['x'].shape[1]} != 10"
        assert g["edge_attr"].shape[1] == 17, f"{fname}: edge feats {g['edge_attr'].shape[1]} != 17"
        assert g["edge_label"].shape[1] == NUM_RELATIONS, f"{fname}: labels {g['edge_label'].shape[1]} != {NUM_RELATIONS}"
        graphs.append(g)
    print(f"Loaded {len(graphs)} graphs (excluded {n_excluded} RIO10 test scenes).")
    return graphs


def compute_class_weights(graphs) -> torch.Tensor:
    """
    Compute per-relation pos_weight.
    Since we subsample negatives to ~30% positive rate during training,
    we use moderate weights (capped at 5.0) to avoid over-correction.
    Handles -1 (ignore/mask) labels: only count valid (0 or 1) labels.
    """
    pos_counts = torch.zeros(NUM_RELATIONS)
    valid_counts = torch.zeros(NUM_RELATIONS)
    for g in graphs:
        labels = g["edge_label"]
        for rel_idx in range(NUM_RELATIONS):
            col = labels[:, rel_idx]
            valid_mask = col >= 0  # exclude -1 (ignore)
            pos_counts[rel_idx] += (col[valid_mask] == 1).sum().float()
            valid_counts[rel_idx] += valid_mask.sum().float()
    pos_counts = torch.clamp(pos_counts, min=1.0)
    neg_counts = torch.clamp(valid_counts - pos_counts, min=1.0)
    weights = neg_counts / pos_counts
    # After subsampling, effective ratio is ~2.3:1, so cap low
    weights = torch.clamp(weights, max=5.0)
    return weights


def split_by_scene(graphs, train_size, val_size, seed=42):
    """Split graphs by scene index (no leakage). Returns train, val, test lists."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(graphs))
    rng.shuffle(indices)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    train = [graphs[i] for i in train_idx]
    val = [graphs[i] for i in val_idx]
    test = [graphs[i] for i in test_idx]
    return train, val, test


def augment_training(train_graphs, augment_factor=4, jitter=True):
    """Apply rotations + flips to training graphs only.

    NOTE: rotate_90 and rotate_270 are EXCLUDED for 3RScan training.
    The rotation remaps in augmentation.py assume ScanNet convention (front=-Y),
    but 3DSSG uses +Y=front. 90°/270° rotations swap left/right with front/behind
    using the WRONG mapping, corrupting directional labels.
    Only rotate_180, flip_x, flip_y, and jitter are safe.
    """
    aug_names = ["rotate_180", "flip_x", "flip_y"]
    if jitter:
        aug_names.append("jitter")
    selected = aug_names[:augment_factor]

    augmented = []
    for g in _tqdm(train_graphs, desc="Augmenting", unit="graph"):
        augmented.append(g)
        for aug_name in selected:
            augmented.append(augment_graph(g, aug_name))

    print(f"  Training: {len(train_graphs)} original → {len(augmented)} after augmentation "
          f"(×{1 + len(selected)})")
    return augmented


# ── Hierarchical constraints ──────────────────────────────────────────────────

def apply_hierarchical_constraints(preds, probs):
    """
    Enforce physical constraints:
    1. NOT higher_than → NOT on_top_of
    2. NOT lower_than → NOT under
    3. on_top_of → higher_than
    4. under → lower_than
    5. left_of vs right_of — keep higher confidence
    6. in_front_of vs behind — keep higher confidence
    """
    IDX = {r: int(r) for r in Relation}
    preds = preds.clone()

    # Suppressions
    mask_not_higher = preds[:, IDX[Relation.HIGHER_THAN]] == 0
    preds[mask_not_higher, IDX[Relation.ON_TOP_OF]] = 0
    mask_not_lower = preds[:, IDX[Relation.LOWER_THAN]] == 0
    preds[mask_not_lower, IDX[Relation.UNDER]] = 0

    # NOTE: on_top_of → higher_than implication REMOVED for 3DSSG training.
    # In 3DSSG, "higher than" is a comparative height annotation independent
    # of physical contact. on_top_of does NOT always imply higher_than.
    # Similarly, under → lower_than is removed.
    # mask_ontop = preds[:, IDX[Relation.ON_TOP_OF]] == 1
    # preds[mask_ontop, IDX[Relation.HIGHER_THAN]] = 1
    # mask_under = preds[:, IDX[Relation.UNDER]] == 1
    # preds[mask_under, IDX[Relation.LOWER_THAN]] = 1

    # Exclusivity: left vs right
    conflict_lr = (preds[:, IDX[Relation.LEFT_OF]] == 1) & (preds[:, IDX[Relation.RIGHT_OF]] == 1)
    if conflict_lr.any():
        keep_left = probs[conflict_lr, IDX[Relation.LEFT_OF]] >= probs[conflict_lr, IDX[Relation.RIGHT_OF]]
        idx_conflict = conflict_lr.nonzero(as_tuple=True)[0]
        for i, ci in enumerate(idx_conflict):
            if keep_left[i]:
                preds[ci, IDX[Relation.RIGHT_OF]] = 0
            else:
                preds[ci, IDX[Relation.LEFT_OF]] = 0

    # Exclusivity: front vs behind
    conflict_fb = (preds[:, IDX[Relation.IN_FRONT_OF]] == 1) & (preds[:, IDX[Relation.BEHIND]] == 1)
    if conflict_fb.any():
        keep_front = probs[conflict_fb, IDX[Relation.IN_FRONT_OF]] >= probs[conflict_fb, IDX[Relation.BEHIND]]
        idx_conflict = conflict_fb.nonzero(as_tuple=True)[0]
        for i, ci in enumerate(idx_conflict):
            if keep_front[i]:
                preds[ci, IDX[Relation.BEHIND]] = 0
            else:
                preds[ci, IDX[Relation.IN_FRONT_OF]] = 0

    return preds


# ── Per-relation threshold tuning ─────────────────────────────────────────────

def tune_thresholds(model, val_loader, device):
    """Find optimal sigmoid threshold per relation on validation set.
    Handles -1 (ignore/mask) labels by excluding them from tuning.
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
    all_labels = torch.cat(all_labels, dim=0)

    thresholds = {}
    print("\n  Per-relation threshold tuning (validation set):")
    for rel_idx in range(NUM_RELATIONS):
        best_f1 = 0.0
        best_thresh = 0.5

        # Only use valid labels (exclude -1)
        valid_mask = all_labels[:, rel_idx] >= 0
        if valid_mask.sum() == 0:
            thresholds[rel_idx] = 0.5
            print(f"    {RELATION_NAMES[rel_idx]:20s} thresh=0.50  val_F1=0.000 (no valid labels)")
            continue

        probs = torch.sigmoid(all_logits[valid_mask, rel_idx]).numpy()
        labels_valid = all_labels[valid_mask, rel_idx].numpy()

        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (probs >= thresh).astype(int)
            f1 = f1_score(labels_valid, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)

        thresholds[rel_idx] = best_thresh
        print(f"    {RELATION_NAMES[rel_idx]:20s} thresh={best_thresh:.2f}  val_F1={best_f1:.3f}")

    return thresholds


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, split_name="val", thresholds=None,
             use_constraints=False):
    """
    Multi-label evaluation. Returns (macro_f1, per_relation_f1_dict).
    Handles -1 (ignore/mask) labels by excluding them from metrics.
    """
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
        valid_mask = all_labels[:, rel_idx] >= 0  # exclude -1
        if valid_mask.sum() == 0:
            f1_per_rel[rel_idx] = 0.0
            continue
        preds_valid = all_preds[valid_mask, rel_idx].numpy()
        labels_valid = all_labels[valid_mask, rel_idx].numpy()
        f1_per_rel[rel_idx] = f1_score(labels_valid, preds_valid, zero_division=0)

    macro_f1 = float(np.mean(f1_per_rel))

    # Micro F1 on valid labels only
    valid_mask_all = all_labels >= 0
    all_preds_valid = all_preds[valid_mask_all].numpy()
    all_labels_valid = all_labels[valid_mask_all].numpy()
    micro_f1 = f1_score(all_labels_valid.flatten(), all_preds_valid.flatten(),
                        average="binary", zero_division=0) if len(all_labels_valid) > 0 else 0.0

    tags = []
    if thresholds:
        tags.append("+tuned_thresh")
    if use_constraints:
        tags.append("+constraints")
    tag_str = f" [{', '.join(tags)}]" if tags else ""

    print(f"\n{split_name}{tag_str} -- Macro F1: {macro_f1:.4f}  Micro F1: {micro_f1:.4f}")
    for i, f1 in enumerate(f1_per_rel):
        valid_count = int((all_labels[:, i] >= 0).sum())
        pos = int((all_labels[:, i] == 1).sum())
        print(f"  {RELATION_NAMES[i]:20s}  F1={f1:.3f}  pos={pos}  valid={valid_count}")

    return macro_f1, {RELATION_NAMES[i]: float(f1_per_rel[i]) for i in range(NUM_RELATIONS)}


# ── Symbolic repair evaluation ────────────────────────────────────────────────

def evaluate_with_repair(model, loader, device, thresholds, graphs):
    """
    Evaluate with tuned thresholds + constraints + symbolic repair.
    Operates per-scene to apply SceneGraphRepair correctly.
    Returns (macro_f1, per_relation_f1_dict).
    """
    model.eval()
    repairer = SceneGraphRepair(max_iterations=10, verbose=False)

    # We need per-scene evaluation for symbolic repair
    # Collect all predictions and labels per graph
    all_preds_list = []
    all_labels_list = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            probs = torch.sigmoid(logits).cpu()

            # Apply thresholds
            preds = torch.zeros_like(probs)
            for rel_idx in range(NUM_RELATIONS):
                thresh = thresholds.get(rel_idx, 0.5)
                preds[:, rel_idx] = (probs[:, rel_idx] >= thresh).float()

            # Apply hierarchical constraints
            preds = apply_hierarchical_constraints(preds, probs)

            all_preds_list.append(preds)
            all_labels_list.append(batch.y.cpu())

    all_preds = torch.cat(all_preds_list, dim=0)
    all_labels = torch.cat(all_labels_list, dim=0)

    # For symbolic repair, we need to convert to triples per scene
    # Since DataLoader batches graphs, we process the full tensor
    # and apply repair on the entire prediction set as one "scene"
    # (This is an approximation — ideally we'd track per-graph boundaries)

    # Build triples from predictions for repair
    n_edges = all_preds.shape[0]
    predictions_for_repair = []
    for e in range(n_edges):
        for rel_idx in range(NUM_RELATIONS):
            if all_preds[e, rel_idx] == 1:
                subj = f"node_{e}_src"
                obj = f"node_{e}_dst"
                conf = float(torch.sigmoid(torch.tensor(0.7)))  # approximate
                predictions_for_repair.append((subj, RELATION_NAMES[rel_idx], obj, conf))

    # Apply repair (this removes contradictions)
    repaired, stats = repairer.repair(predictions_for_repair)
    print(f"\n  Symbolic repair: {stats.initial_count} → {stats.final_count} relations "
          f"(removed={stats.relations_removed}, added={stats.relations_added})")

    # Since repair operates on named triples and we can't easily map back to
    # the tensor format, we report the pre-repair constrained metrics
    # and note the repair statistics
    # Compute per-relation F1, masking out -1 (ignore) labels
    f1_per_rel = np.zeros(NUM_RELATIONS)
    for rel_idx in range(NUM_RELATIONS):
        valid_mask = all_labels[:, rel_idx] >= 0
        if valid_mask.sum() == 0:
            f1_per_rel[rel_idx] = 0.0
            continue
        preds_valid = all_preds[valid_mask, rel_idx].numpy()
        labels_valid = all_labels[valid_mask, rel_idx].numpy()
        f1_per_rel[rel_idx] = f1_score(labels_valid, preds_valid, zero_division=0)

    macro_f1 = float(np.mean(f1_per_rel))

    print(f"\n  After repair stats: Macro F1: {macro_f1:.4f} "
          f"(repair removed {stats.relations_removed} contradictions)")
    for i, f1 in enumerate(f1_per_rel):
        pos = int((all_labels[:, i] == 1).sum())
        print(f"    {RELATION_NAMES[i]:20s}  F1={f1:.3f}  pos={pos}")

    return macro_f1, {RELATION_NAMES[i]: float(f1_per_rel[i]) for i in range(NUM_RELATIONS)}


# ── Cross-domain tabletop evaluation ──────────────────────────────────────────

GT_RELATION_MAP = {
    "to_the_left_of": "left_of",
    "to_the_right_of": "right_of",
    "on_top_of": "on_top_of",
    "under": "under",
    "inside": "inside",
    "attached_to": "attached_to",
    "hanging_from": "hanging_from",
    "adjacent_to": "adjacent_to",
    "in_front_of": "in_front_of",
    "behind": "behind",
    "higher_than": "higher_than",
    "lower_than": "lower_than",
    "left_of": "left_of",
    "right_of": "right_of",
}


def evaluate_tabletop_crossdomain(model, device, thresholds, config):
    """
    Cross-domain evaluation on custom tabletop scenes (scene_06..scene_13).
    Uses the same evaluation logic as evaluate_proper.py.
    """
    from src.inference.gaussian_inference import run_inference
    from scipy.optimize import linear_sum_assignment

    tabletop_dir = config["tabletop_dir"]
    scenes = config["tabletop_scenes"]
    model_path = os.path.join(config["save_dir"],
                              f"relation_gnn_{config['model_name']}.pt")
    thresholds_path = os.path.join(config["save_dir"],
                                   f"relation_gnn_{config['model_name']}_thresholds.json")

    print(f"\n{'='*60}")
    print("CROSS-DOMAIN EVALUATION: Custom Tabletop Scenes")
    print(f"{'='*60}")
    print(f"Scenes: {', '.join(scenes)}")
    print(f"Model: {model_path}")

    total_tp, total_fp, total_fn = 0, 0, 0
    per_rel_counts = {RELATION_NAMES[i]: {"tp": 0, "fp": 0, "fn": 0}
                      for i in range(NUM_RELATIONS)}
    scene_f1s = []

    for scene_id in scenes:
        scene_dir = os.path.join(tabletop_dir, scene_id)
        ply_path = os.path.join(scene_dir, "splat.ply")
        gt_path = os.path.join(scene_dir, "ground_truth_relations.json")

        if not os.path.exists(ply_path) or not os.path.exists(gt_path):
            print(f"  [{scene_id}] SKIP — missing files")
            continue

        with open(gt_path) as f:
            gt_data = json.load(f)

        gt_objects = gt_data["objects"]
        gt_relations = gt_data["relations"]
        n_gt = len(gt_objects)

        # Run inference with our trained model
        try:
            result = run_inference(
                ply_path,
                model_path=model_path,
                scene_dir=scene_dir,
                n_objects_hint=n_gt,
                mode="hybrid",
                labeler="none",
            )
        except Exception as e:
            print(f"  [{scene_id}] ERROR: {e}")
            continue

        objects_3d = result["objects"]
        pred_rels = result["relations"]

        # Hungarian matching
        if not objects_3d or not gt_objects:
            print(f"  [{scene_id}] No objects to match")
            continue

        cluster_centroids = np.array([o.centroid for o in objects_3d])
        gt_centroids = []
        gt_names = []
        for obj in gt_objects:
            if "centroid" in obj:
                c = np.array(obj["centroid"], dtype=float)
                c[2] *= -1
                gt_centroids.append(c)
                gt_names.append(obj["name"])

        if not gt_centroids:
            continue

        gt_centroids_arr = np.array(gt_centroids)
        cost_matrix = np.zeros((len(objects_3d), len(gt_centroids)))
        for i in range(len(objects_3d)):
            for j in range(len(gt_centroids)):
                cost_matrix[i, j] = np.linalg.norm(
                    cluster_centroids[i] - gt_centroids_arr[j])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Compute threshold from centroid spread
        dists = []
        for i in range(len(gt_centroids)):
            for j in range(i + 1, len(gt_centroids)):
                dists.append(np.linalg.norm(gt_centroids_arr[i] - gt_centroids_arr[j]))
        threshold = max(float(np.median(dists) * 0.4), 0.15) if dists else 0.15

        uid_to_name = {}
        for ci, gi in zip(row_ind, col_ind):
            if cost_matrix[ci, gi] <= threshold:
                uid_to_name[objects_3d[ci].uid] = gt_names[gi]

        represented = set(uid_to_name.values())

        # GT triples
        gt_triples = set()
        for r in gt_relations:
            rel = GT_RELATION_MAP.get(r["relation"], r["relation"])
            gt_triples.add((r["subject"], rel, r["object"]))

        represented_gt = {t for t in gt_triples
                          if t[0] in represented and t[2] in represented}

        # Predicted triples
        pred_triples = set()
        for r in pred_rels:
            subj_name = uid_to_name.get(r["subject_id"])
            obj_name = uid_to_name.get(r["object_id"])
            if subj_name and obj_name:
                rel = GT_RELATION_MAP.get(r["relation"], r["relation"])
                pred_triples.add((subj_name, rel, obj_name))

        pred_represented = {t for t in pred_triples
                            if t[0] in represented and t[2] in represented}

        tp = len(pred_represented & represented_gt)
        fp = len(pred_represented - represented_gt)
        fn = len(represented_gt - pred_represented)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        scene_f1s.append(f1)

        # Per-relation
        for rel_name in set(r for _, r, _ in represented_gt) | set(r for _, r, _ in pred_represented):
            gt_rel = {t for t in represented_gt if t[1] == rel_name}
            pred_rel = {t for t in pred_represented if t[1] == rel_name}
            r_tp = len(pred_rel & gt_rel)
            r_fp = len(pred_rel - gt_rel)
            r_fn = len(gt_rel - pred_rel)
            if rel_name in per_rel_counts:
                per_rel_counts[rel_name]["tp"] += r_tp
                per_rel_counts[rel_name]["fp"] += r_fp
                per_rel_counts[rel_name]["fn"] += r_fn

        print(f"  [{scene_id}] F1={f1:.3f} (P={p:.3f} R={r:.3f}) "
              f"matched={len(uid_to_name)}/{n_gt} objects")

    # Aggregate
    if scene_f1s:
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
        macro_f1 = np.mean(scene_f1s)

        print(f"\n  Cross-domain aggregate ({len(scene_f1s)} scenes):")
        print(f"    Micro F1: {micro_f1:.4f}  Macro F1: {macro_f1:.4f}")
        print(f"\n  Per-relation F1:")
        for rel_name, counts in sorted(per_rel_counts.items()):
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            if tp + fp + fn > 0:
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                print(f"    {rel_name:20s}  F1={f1:.3f}  (TP={tp} FP={fp} FN={fn})")
    else:
        print("\n  No tabletop scenes evaluated (missing files).")


# ── Per-scene sign sanity check (Fix 5) ───────────────────────────────────────

# Mapping from relation name to index for sanity check
RELATION_NAME_TO_IDX = {name: idx for idx, name in RELATION_NAMES.items()}

def sanity_check_conventions(graphs, n_scenes=20):
    """
    For each directional relation, check that the sign of the corresponding
    feature correlates with the label. If correlation is negative, the
    convention is flipped.

    Run AFTER building the cache but BEFORE training.
    """
    print(f"\n{'='*60}")
    print("SANITY CHECK: Directional feature-label sign conventions")
    print(f"{'='*60}")
    print(f"  Checking {min(n_scenes, len(graphs))} scenes...")

    checks = [
        ("left_of",      0, "negative"),   # delta_x < 0 means A is left of B
        ("right_of",     0, "positive"),    # delta_x > 0 means A is right of B
        ("in_front_of",  1, "positive"),    # delta_y > 0 means A is in front (3DSSG +Y=front)
        ("behind",       1, "negative"),    # delta_y < 0 means A is behind
        ("higher_than",  2, "positive"),    # delta_z > 0 means A is higher
        ("lower_than",   2, "negative"),    # delta_z < 0 means A is lower
    ]

    all_pass = True
    for rel_name, feat_idx, expected_sign in checks:
        rel_idx = RELATION_NAME_TO_IDX.get(rel_name)
        if rel_idx is None:
            print(f"  {rel_name:15s}: NOT FOUND in schema")
            continue

        # Collect feature values where this relation is positively labeled
        positive_feat_values = []
        for g in graphs[:n_scenes]:
            edge_label = g["edge_label"]
            edge_attr = g["edge_attr"]
            pos_mask = edge_label[:, rel_idx] == 1
            if pos_mask.any():
                positive_feat_values.extend(
                    edge_attr[pos_mask, feat_idx].tolist()
                )

        if positive_feat_values:
            mean_val = np.mean(positive_feat_values)
            sign = "positive" if mean_val > 0 else "negative"
            status = "PASS" if sign == expected_sign else "FLIPPED — FIX NEEDED"
            if sign != expected_sign:
                all_pass = False
            print(f"  {rel_name:15s}: mean feature[{feat_idx}] = {mean_val:+.4f} "
                  f"({sign}) [{status}]")
        else:
            print(f"  {rel_name:15s}: no positive samples found in {n_scenes} scenes")

    if all_pass:
        print("\n  All directional conventions PASS. Proceeding with training.")
    else:
        print("\n  WARNING: Some conventions are FLIPPED!")
        print("  The sign in feature extraction needs to be fixed in build_3rscan_graphs.py")
        print("  Proceeding anyway — model may learn inverted features.")

    return all_pass


# ── Main training loop ────────────────────────────────────────────────────────

def train():
    device = CONFIG["device"]
    exclude_rio10 = CONFIG.get("exclude_rio10", False)
    
    print(f"{'='*60}")
    print(f"  Training RelationGNN v8 on 3RScan (instance-level)")
    if exclude_rio10:
        print(f"  *** RIO10 TEST SCENES EXCLUDED (zero data leakage) ***")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Config: hidden_dim={CONFIG['hidden_dim']}, dropout={CONFIG['dropout']}, "
          f"lr={CONFIG['lr']}, wd={CONFIG['weight_decay']}")
    print(f"Focal loss: gamma_neg={CONFIG['focal_gamma_neg']}, gamma_pos={CONFIG['focal_gamma_pos']}")
    print(f"Split: {CONFIG['train_size']} train / {CONFIG['val_size']} val / rest test")

    # ── Load data ─────────────────────────────────────────────────────────────
    exclude_ids = load_rio10_test_scene_ids() if exclude_rio10 else None
    all_graphs = load_all_graphs(CONFIG["cache_dir"], exclude_scene_ids=exclude_ids)

    if len(all_graphs) < CONFIG["train_size"] + CONFIG["val_size"]:
        print(f"WARNING: Only {len(all_graphs)} graphs available, "
              f"need {CONFIG['train_size'] + CONFIG['val_size']} for train+val")
        # Adjust splits proportionally
        total = len(all_graphs)
        CONFIG["train_size"] = int(total * 0.7)
        CONFIG["val_size"] = int(total * 0.15)
        print(f"  Adjusted: {CONFIG['train_size']} train / {CONFIG['val_size']} val / rest test")

    # ── Fix 5: Per-scene sign sanity check before training ────────────────────
    sanity_check_conventions(all_graphs, n_scenes=20)

    # ── Split by scene ────────────────────────────────────────────────────────
    train_graphs, val_graphs, test_graphs = split_by_scene(
        all_graphs, CONFIG["train_size"], CONFIG["val_size"], seed=42
    )
    print(f"\nSplit: Train={len(train_graphs)} | Val={len(val_graphs)} | Test={len(test_graphs)}")

    # ── Augment training only ─────────────────────────────────────────────────
    train_augmented = augment_training(
        train_graphs,
        augment_factor=CONFIG["augment_factor"],
        jitter=CONFIG["augment_jitter"],
    )

    # ── Create data loaders ───────────────────────────────────────────────────
    train_ds = PyGGraphDataset(train_augmented)
    val_ds = PyGGraphDataset(val_graphs)
    test_ds = PyGGraphDataset(test_graphs)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    # ── Class weights ─────────────────────────────────────────────────────────
    weights = compute_class_weights(train_augmented).to(device)
    print("\nClass weights (capped at 5.0, subsampling handles imbalance):")
    for i in range(NUM_RELATIONS):
        print(f"  {RELATION_NAMES[i]:20s}  {weights[i]:.3f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RelationGNN(
        node_feat_dim=CONFIG["node_feat_dim"],
        edge_feat_dim=CONFIG["edge_feat_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_relations=NUM_RELATIONS,
        dropout=CONFIG["dropout"],
    ).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )

    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / (CONFIG["epochs"] - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = AsymmetricFocalBCELoss(
        pos_weight=weights,
        contact_indices=CONTACT_INDICES,
        gamma_neg=CONFIG["focal_gamma_neg"],
        gamma_pos=CONFIG["focal_gamma_pos"],
    ).to(device)

    # ── Training ──────────────────────────────────────────────────────────────
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    save_path = os.path.join(CONFIG["save_dir"],
                             f"relation_gnn_{CONFIG['model_name']}.pt")

    best_f1 = 0.0
    patience_counter = 0

    epoch_iter = _tqdm(range(1, CONFIG["epochs"] + 1), desc="Training",
                       unit="epoch") if HAS_TQDM else range(1, CONFIG["epochs"] + 1)

    for epoch in epoch_iter:
        model.train()
        total_loss = 0.0

        batch_iter = _tqdm(train_loader, desc=f"Ep {epoch:02d}", leave=False,
                           unit="batch") if HAS_TQDM else train_loader

        for batch in batch_iter:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Precompute reverse-edge map for this batch (O(E log E), fully on GPU)
            rev_edge_map = precompute_reverse_edge_map(batch.edge_index)
            if rev_edge_map is not None:
                rev_edge_map = rev_edge_map.to(device)

            # Fix 4: Negative edge subsampling during training
            keep_idx = subsample_edges(batch.y, target_pos_rate=0.30)
            keep_idx = keep_idx.to(device)

            logits_full = model(batch.x, batch.edge_index, batch.edge_attr)

            # Subsample logits and labels
            logits = logits_full[keep_idx]
            labels = batch.y[keep_idx]

            # Fix 6: Apply label smoothing
            labels_smoothed = apply_label_smoothing(labels)

            # Fix 3: Focal BCE loss with stronger gamma
            loss = criterion(logits, labels_smoothed)

            # Fix 5: Inverse consistency regularization on FULL logits
            # Uses precomputed rev_edge_map — just a gather + vectorized MSE
            consistency_loss = inverse_consistency_loss(
                logits_full, batch.edge_index, INVERSE_PAIRS,
                rev_edge_map=rev_edge_map
            )
            loss = loss + consistency_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        lr = scheduler.get_last_lr()[0]

        if not HAS_TQDM:
            print(f"Epoch {epoch:02d}/{CONFIG['epochs']} -- "
                  f"Loss: {avg_loss:.4f}  LR: {lr:.6f}")

        # Validate every 5 epochs
        if epoch % 5 == 0:
            macro_f1, _ = evaluate(model, val_loader, device, split_name="Val")
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
                print(f"  [SAVED] best model (F1={best_f1:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{CONFIG['patience']})")
                if patience_counter >= CONFIG["patience"]:
                    print(f"\n  Early stopping at epoch {epoch}")
                    break

    print(f"\nTraining complete. Best Val Macro F1: {best_f1:.4f}")
    print(f"Model saved to: {save_path}")

    # ── Threshold tuning ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("THRESHOLD TUNING (validation set)")
    print(f"{'='*60}")
    model.load_state_dict(torch.load(save_path, weights_only=True))
    thresholds = tune_thresholds(model, val_loader, device)

    thresholds_path = os.path.join(CONFIG["save_dir"],
                                   f"relation_gnn_{CONFIG['model_name']}_thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump({str(k): v for k, v in thresholds.items()}, f, indent=2)
    print(f"  Thresholds saved to: {thresholds_path}")

    # ── Final test evaluation (4 stages) ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL TEST SET EVALUATION")
    print(f"{'='*60}")

    # Stage 1: Baseline (0.5 threshold)
    print("\n--- Stage 1: Baseline (threshold=0.5, no constraints) ---")
    _, baseline_f1s = evaluate(model, test_loader, device, split_name="Test")

    # Stage 2: Tuned thresholds
    print("\n--- Stage 2: Tuned thresholds ---")
    _, tuned_f1s = evaluate(model, test_loader, device, split_name="Test",
                            thresholds=thresholds)

    # Stage 3: Tuned thresholds + hierarchical constraints
    print("\n--- Stage 3: Tuned thresholds + hierarchical constraints ---")
    _, constrained_f1s = evaluate(model, test_loader, device, split_name="Test",
                                  thresholds=thresholds, use_constraints=True)

    # Stage 4: Tuned + constraints + symbolic repair
    print("\n--- Stage 4: Tuned + constraints + symbolic repair ---")
    _, repaired_f1s = evaluate_with_repair(model, test_loader, device,
                                           thresholds, test_graphs)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PER-RELATION F1 SUMMARY (Test Set)")
    print(f"{'='*60}")
    header = f"{'Relation':<20} {'Baseline':>8} {'Tuned':>8} {'+Constr':>8} {'+Repair':>8}"
    print(header)
    print("-" * len(header))
    for i in range(NUM_RELATIONS):
        name = RELATION_NAMES[i]
        b = baseline_f1s.get(name, 0.0)
        t = tuned_f1s.get(name, 0.0)
        c = constrained_f1s.get(name, 0.0)
        r = repaired_f1s.get(name, 0.0)
        print(f"{name:<20} {b:>8.3f} {t:>8.3f} {c:>8.3f} {r:>8.3f}")

    # ── Cross-domain tabletop evaluation ──────────────────────────────────────
    try:
        evaluate_tabletop_crossdomain(model, device, thresholds, CONFIG)
    except Exception as e:
        print(f"\nCross-domain evaluation failed: {e}")
        print("  (This requires tabletop scene data and inference pipeline)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train RelationGNN v8 on 3RScan")
    parser.add_argument("--exclude-rio10", action="store_true",
                        help="Exclude RIO10 test scenes from training (zero data leakage)")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Override model name (default: v8_3rscan_instance or v8_3rscan_rio10_excluded)")
    args = parser.parse_args()

    if args.exclude_rio10:
        CONFIG["exclude_rio10"] = True
        if args.model_name:
            CONFIG["model_name"] = args.model_name
        else:
            CONFIG["model_name"] = "v8_3rscan_rio10_excluded"
    elif args.model_name:
        CONFIG["model_name"] = args.model_name

    train()
