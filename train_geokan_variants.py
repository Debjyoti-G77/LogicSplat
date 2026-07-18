"""
Train GeoKAN-Gamma and GeoKAN-Wavelet variants, then compare with RBF (v4).

Uses the EXACT same training config, data loading, augmentation, loss, and
evaluation logic as train_geokan.py. The ONLY difference is the basis function
in the GeoKAN layers:
  - RBF (existing):  phi_k = exp(-gamma * (z - c_k)^2)
  - Gamma:           MetricNet replaced by learnable scalar g_i = softplus(gamma_i)
  - Wavelet:         Mexican hat wavelet (1 - z^2) * exp(-z^2 / 2) at same centres

Usage:
    python train_geokan_variants.py
"""
import sys
sys.path.insert(0, ".")

import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from geokan_relation import (
    GeoKANRelationGNN, GeoKANHead, GeoKANLayer,
    CONTACT_INDICES, DIRECTIONAL_INDICES,
)
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, Relation, LEGACY_12_TO_10_COLS
from src.training.augmentation import augment_graph

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def _tqdm(it, **kwargs):
        return it


# ── Config (IDENTICAL to train_geokan.py) ─────────────────────────────────────

CONFIG = {
    "model_name":       "geokan_variants",
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
    "focal_gamma_neg":  4.0,
    "focal_gamma_pos":  0.0,
    "asl_prob_margin":  0.0,
    "patience":         15,
    "metric_reg_weight": 1e-4,
    "max_grad_norm":    2.0,
    "warmup_epochs":    5,
    "pos_weight_cap":   8.0,
}


# ── Inverse relation pairs (10-class schema) ─────────────────────────────────

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

# ── Per-relation loss weights ─────────────────────────────────────────────────

RELATION_LOSS_WEIGHTS = {
    int(Relation.ATTACHED_TO):  1.8,
    int(Relation.ADJACENT_TO):  1.5,
    int(Relation.ON_TOP_OF):    1.2,
    int(Relation.UNDER):        1.2,
}

# ── Per-relation label smoothing ──────────────────────────────────────────────

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


# ══════════════════════════════════════════════════════════════════════════════
# VARIANT 1: GeoKAN-Gamma Layer
# Replaces MetricNet MLP with a single learnable scalar per dimension:
#   g_i = softplus(gamma_i) where gamma is nn.Parameter(torch.zeros(in_dim))
# ══════════════════════════════════════════════════════════════════════════════

class GeoKANGammaLayer(nn.Module):
    """
    GeoKAN layer with Gamma metric: replaces the MetricNet MLP with a single
    learnable scalar per input dimension. g_i = softplus(gamma_i).
    """

    def __init__(self, in_dim: int, out_dim: int, n_bases: int = 12,
                 metric_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_bases = n_bases

        # Input normalization
        self.bn = nn.BatchNorm1d(in_dim)

        # Gamma metric: single learnable scalar per dimension (replaces MetricNet)
        # Initialize to zeros so softplus(0) ≈ 0.693 → sqrt ≈ 0.83 (near identity)
        self.gamma_param = nn.Parameter(torch.zeros(in_dim))

        # RBF centres: K evenly spaced in [-3, 3]
        centers = torch.linspace(-3.0, 3.0, n_bases)
        self.register_buffer("centers", centers)

        # RBF bandwidth (learned)
        self.gamma_rbf = nn.Parameter(torch.tensor(1.0))

        # Linear mix: [phi_flattened(in_dim * K) | u_original(in_dim)] → out_dim
        mix_in_dim = in_dim * n_bases + in_dim
        self.linear_mix = nn.Sequential(
            nn.Linear(mix_in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self._last_g: torch.Tensor | None = None

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        u_normed = self.bn(u)

        # Gamma metric: g_i = softplus(gamma_i) — input-independent diagonal metric
        g = F.softplus(self.gamma_param).unsqueeze(0) + 1e-6  # (1, in_dim)
        g = g.clamp(0.05, 16.0)
        self._last_g = g

        # Warp: z = u * sqrt(g)
        z = u_normed * torch.sqrt(g)  # (B, in_dim)

        # RBF basis expansion (same as original)
        z_expanded = z.unsqueeze(-1)  # (B, in_dim, 1)
        centers = self.centers.view(1, 1, -1)  # (1, 1, K)
        gamma_rbf = F.softplus(self.gamma_rbf)
        phi = torch.exp(-gamma_rbf * (z_expanded - centers).pow(2))  # (B, in_dim, K)
        phi_flat = phi.reshape(u.shape[0], -1)  # (B, in_dim * K)

        # Skip connection + linear mix
        features = torch.cat([phi_flat, u_normed], dim=-1)
        return self.linear_mix(features)

    def metric_regularization(self) -> torch.Tensor:
        if self._last_g is None:
            return torch.tensor(0.0)
        return self._last_g.log().pow(2).mean()


# ══════════════════════════════════════════════════════════════════════════════
# VARIANT 2: GeoKAN-Wavelet Layer
# Replaces RBF basis exp(-gamma*(z-c)^2) with Mexican hat wavelet:
#   psi(z) = (1 - z^2) * exp(-z^2 / 2)  evaluated at same centres
# ══════════════════════════════════════════════════════════════════════════════

class GeoKANWaveletLayer(nn.Module):
    """
    GeoKAN layer with Mexican hat wavelet basis instead of RBF.
    Keeps the full MetricNet MLP (same as original RBF variant).
    """

    def __init__(self, in_dim: int, out_dim: int, n_bases: int = 12,
                 metric_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_bases = n_bases

        # Input normalization
        self.bn = nn.BatchNorm1d(in_dim)

        # MetricNet (same as original RBF variant)
        self.metric_net = nn.Sequential(
            nn.Linear(in_dim, metric_hidden),
            nn.GELU(),
            nn.Linear(metric_hidden, in_dim),
        )
        softplus_inv_one = math.log(math.e - 1.0)
        nn.init.zeros_(self.metric_net[2].weight)
        nn.init.constant_(self.metric_net[2].bias, softplus_inv_one)

        # Wavelet centres: same as RBF centres
        centers = torch.linspace(-3.0, 3.0, n_bases)
        self.register_buffer("centers", centers)

        # Linear mix: [phi_flattened(in_dim * K) | u_original(in_dim)] → out_dim
        mix_in_dim = in_dim * n_bases + in_dim
        self.linear_mix = nn.Sequential(
            nn.Linear(mix_in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self._last_g: torch.Tensor | None = None

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        u_normed = self.bn(u)

        # MetricNet (same as RBF variant)
        g_raw = self.metric_net(u_normed)
        g = F.softplus(g_raw) + 1e-6
        g = g.clamp(0.05, 16.0)
        self._last_g = g

        # Warp: z = u * sqrt(g)
        z = u_normed * torch.sqrt(g)  # (B, in_dim)

        # Mexican hat wavelet: psi(t) = (1 - t^2) * exp(-t^2 / 2)
        # Evaluate at (z_i - c_k) for each centre c_k
        z_expanded = z.unsqueeze(-1)  # (B, in_dim, 1)
        centers = self.centers.view(1, 1, -1)  # (1, 1, K)
        t = z_expanded - centers  # (B, in_dim, K)
        t_sq = t.pow(2)
        phi = (1.0 - t_sq) * torch.exp(-t_sq / 2.0)  # (B, in_dim, K)
        phi_flat = phi.reshape(u.shape[0], -1)  # (B, in_dim * K)

        # Skip connection + linear mix
        features = torch.cat([phi_flat, u_normed], dim=-1)
        return self.linear_mix(features)

    def metric_regularization(self) -> torch.Tensor:
        if self._last_g is None:
            return torch.tensor(0.0)
        return self._last_g.log().pow(2).mean()


# ── Variant Head (stacked layers) ────────────────────────────────────────────

class GeoKANVariantHead(nn.Module):
    """Two stacked variant GeoKAN layers + linear output head."""

    def __init__(self, layer_cls, in_dim: int, hidden_dim: int = 128,
                 out_dim: int = 6, n_bases: int = 12, metric_hidden: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.layer1 = layer_cls(in_dim, hidden_dim, n_bases=n_bases,
                                metric_hidden=metric_hidden, dropout=dropout)
        self.layer2 = layer_cls(hidden_dim, hidden_dim, n_bases=n_bases,
                                metric_hidden=metric_hidden, dropout=dropout)
        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x)
        h = self.layer2(h)
        return self.output(h)

    def metric_regularization(self) -> torch.Tensor:
        return (self.layer1.metric_regularization() +
                self.layer2.metric_regularization())


# ── Variant GNN Model ─────────────────────────────────────────────────────────

class GeoKANVariantGNN(nn.Module):
    """
    Same architecture as GeoKANRelationGNN but with pluggable GeoKAN layer class.
    Shares: node encoder, GATv2Conv layers, pair projection.
    Differs: GeoKAN head layer type (Gamma or Wavelet).
    """

    def __init__(self, layer_cls, node_feat_dim=10, edge_feat_dim=22,
                 hidden_dim=128, num_relations=NUM_RELATIONS, dropout=0.2):
        super().__init__()
        from torch_geometric.nn import GATv2Conv

        self.edge_feat_dim = edge_feat_dim
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim

        # 1. Node encoder (identical)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 2. GATv2Conv layers (identical)
        self.conv1 = GATv2Conv(
            hidden_dim, hidden_dim, heads=4, concat=False,
            dropout=dropout, edge_dim=edge_feat_dim,
        )
        self.conv2 = GATv2Conv(
            hidden_dim, hidden_dim, heads=4, concat=False,
            dropout=dropout, edge_dim=edge_feat_dim,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 3. Pair projection (identical)
        pair_raw_dim = 4 * hidden_dim
        pair_dim = 64
        self.pair_proj = nn.Sequential(
            nn.Linear(pair_raw_dim, pair_dim),
            nn.LayerNorm(pair_dim),
            nn.GELU(),
        )

        # 4. Dual variant GeoKAN heads
        contact_in_dim = pair_dim + edge_feat_dim  # 64 + 22 = 86
        self.head_contact = GeoKANVariantHead(
            layer_cls, in_dim=contact_in_dim, hidden_dim=128,
            out_dim=len(CONTACT_INDICES), n_bases=12, metric_hidden=64,
            dropout=dropout,
        )

        directional_in_dim = pair_dim + 10  # 64 + 10 = 74
        self.head_directional = GeoKANVariantHead(
            layer_cls, in_dim=directional_in_dim, hidden_dim=128,
            out_dim=len(DIRECTIONAL_INDICES), n_bases=12, metric_hidden=64,
            dropout=dropout,
        )

    def forward(self, x, edge_index, edge_attr=None):
        # 1. Node encoding
        h = self.node_encoder(x)

        # 2. GATv2Conv with residual
        h1 = self.conv1(h, edge_index, edge_attr=edge_attr)
        h = self.norm1(F.gelu(h1) + h)
        h = self.dropout(h)

        h2 = self.conv2(h, edge_index, edge_attr=edge_attr)
        h = self.norm2(F.gelu(h2) + h)
        h = self.dropout(h)

        # 3. Pair representation
        src, dst = edge_index[0], edge_index[1]
        h_src, h_dst = h[src], h[dst]
        pair_raw = torch.cat([h_src, h_dst, h_src - h_dst, h_src * h_dst], dim=-1)
        pair = self.pair_proj(pair_raw)

        # 4. Edge features for each head
        if edge_attr is not None:
            all_edge_feats = edge_attr
            dir_edge_feats = edge_attr[:, :10]
        else:
            all_edge_feats = torch.zeros(pair.shape[0], self.edge_feat_dim,
                                         device=pair.device)
            dir_edge_feats = torch.zeros(pair.shape[0], 10, device=pair.device)

        contact_input = torch.cat([pair, all_edge_feats], dim=-1)
        out_contact = self.head_contact(contact_input)

        dir_input = torch.cat([pair, dir_edge_feats], dim=-1)
        out_dir = self.head_directional(dir_input)

        return torch.cat([out_contact, out_dir], dim=-1)

    def metric_reg(self) -> torch.Tensor:
        return (self.head_contact.metric_regularization() +
                self.head_directional.metric_regularization())


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING INFRASTRUCTURE (reused from train_geokan.py — identical logic)
# ══════════════════════════════════════════════════════════════════════════════

class AsymmetricFocalBCELoss(nn.Module):
    """ASL (Baruch et al. 2021) — identical to train_geokan.py."""

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

        xs_pos = probs
        loss_pos = F.binary_cross_entropy_with_logits(
            logits, torch.ones_like(logits), pos_weight=self.pos_weight,
            reduction='none'
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


# ── Utility functions (identical to train_geokan.py) ──────────────────────────

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


def precompute_reverse_edge_map(edge_index):
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
        assert g["edge_attr"].shape[1] == 22, f"{fname}: edge feats != 22"
        if g["edge_label"].shape[1] == 12:
            g["edge_label"] = g["edge_label"][:, LEGACY_12_TO_10_COLS]
        assert g["edge_label"].shape[1] == NUM_RELATIONS
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


def _binarize_labels(labels: np.ndarray) -> np.ndarray:
    return (labels >= 0.5).astype(int)


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
    for g in _tqdm(train_graphs, desc="Augmenting", unit="graph"):
        augmented.append(g)
        for aug_name in selected:
            augmented.append(augment_graph(g, aug_name))

    # Rare-relation oversampling
    rare_indices = [int(Relation.ATTACHED_TO), int(Relation.ADJACENT_TO)]
    extra_augs = ["flip_x", "flip_y"]
    n_rare_extra = 0
    for g in train_graphs:
        labels = g["edge_label"]
        is_rare = any(
            (labels[:, ri] == 1).any().item()
            for ri in rare_indices if labels.shape[1] > ri
        )
        if is_rare:
            for aug_name in extra_augs:
                augmented.append(augment_graph(g, aug_name))
                n_rare_extra += 1

    print(f"  Training: {len(train_graphs)} -> {len(augmented)} after augmentation "
          f"(x{1 + len(selected)}, +{n_rare_extra} rare extras)")
    return augmented


# ── Hierarchical constraints (identical to train_geokan.py) ───────────────────

def apply_hierarchical_constraints(preds, probs):
    IDX = {r: int(r) for r in Relation}
    preds = preds.clone()

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


# ── Threshold tuning ──────────────────────────────────────────────────────────

def tune_thresholds(model, val_loader, device):
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
    for rel_idx in range(NUM_RELATIONS):
        best_f1, best_thresh = 0.0, 0.5
        valid_mask = all_labels[:, rel_idx] >= 0
        if valid_mask.sum() == 0:
            thresholds[rel_idx] = 0.5
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
    return thresholds


# ── Evaluation with Macro F1 ─────────────────────────────────────────────────

def evaluate(model, loader, device, thresholds=None, use_constraints=False):
    """Returns (macro_f1, per_relation_f1_dict, all_probs, all_labels)."""
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

    if thresholds:
        all_preds = torch.zeros_like(all_probs)
        for rel_idx in range(NUM_RELATIONS):
            thresh = thresholds.get(rel_idx, 0.5)
            all_preds[:, rel_idx] = (all_probs[:, rel_idx] >= thresh).float()
    else:
        all_preds = (all_probs >= 0.5).float()

    if use_constraints:
        all_preds = apply_hierarchical_constraints(all_preds, all_probs)

    f1_per_rel = np.zeros(NUM_RELATIONS)
    for rel_idx in range(NUM_RELATIONS):
        valid_mask = all_labels[:, rel_idx] >= 0
        if valid_mask.sum() == 0:
            continue
        preds_valid = all_preds[valid_mask, rel_idx].numpy()
        labels_valid = _binarize_labels(all_labels[valid_mask, rel_idx].numpy())
        f1_per_rel[rel_idx] = f1_score(labels_valid, preds_valid, zero_division=0)

    macro_f1 = float(np.mean(f1_per_rel))
    per_rel = {RELATION_NAMES[i]: float(f1_per_rel[i]) for i in range(NUM_RELATIONS)}
    return macro_f1, per_rel, all_probs.numpy(), all_labels.numpy()


# ── Recall@K computation ─────────────────────────────────────────────────────

def compute_recall_at_k(probs, labels, k):
    """
    For each edge with at least one positive GT relation,
    check if each positive GT relation is in the top-K predictions.
    """
    hits, total = 0, 0
    for edge_idx in range(probs.shape[0]):
        edge_labels = labels[edge_idx]
        edge_probs = probs[edge_idx]
        positive_rels = np.where(edge_labels >= 0.5)[0]
        if len(positive_rels) == 0:
            continue
        top_k_indices = np.argsort(edge_probs)[::-1][:k]
        for rel_idx in positive_rels:
            total += 1
            if rel_idx in top_k_indices:
                hits += 1
    return hits / max(total, 1)


# ── Warmup + Cosine schedule ─────────────────────────────────────────────────

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


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP (trains one variant)
# ══════════════════════════════════════════════════════════════════════════════

def train_variant_full(variant_name, model, train_loader, val_loader,
                       pos_weight, device, save_path,
                       lr_override=None, warmup_override=None):
    """Full training loop for a variant. Returns (best_val_f1, model)."""
    lr = lr_override if lr_override is not None else CONFIG["lr"]
    warmup = warmup_override if warmup_override is not None else CONFIG["warmup_epochs"]

    print(f"\n{'='*60}")
    print(f"Training: {variant_name}")
    print(f"  lr={lr}, warmup={warmup} epochs")
    print(f"{'='*60}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    rel_weights = torch.ones(NUM_RELATIONS).to(device)
    for rel_idx, w in RELATION_LOSS_WEIGHTS.items():
        rel_weights[rel_idx] = w

    criterion = AsymmetricFocalBCELoss(
        pos_weight=pos_weight,
        contact_indices=CONTACT_INDICES,
        gamma_neg=CONFIG["focal_gamma_neg"],
        gamma_pos=CONFIG["focal_gamma_pos"],
        prob_margin=CONFIG["asl_prob_margin"],
        relation_weights=rel_weights,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=CONFIG["weight_decay"]
    )
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=warmup,
        total_epochs=CONFIG["epochs"]
    )

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in _tqdm(train_loader, desc=f"  {variant_name} E{epoch+1:02d}",
                           unit="batch"):
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
        lr = optimizer.param_groups[0]['lr']
        avg_loss = epoch_loss / max(n_batches, 1)

        # Quick val F1 (no threshold tuning during training)
        val_f1, _, _, _ = evaluate(model, val_loader, device)

        print(f"  Epoch {epoch+1:02d}/{CONFIG['epochs']} - "
              f"loss={avg_loss:.4f} lr={lr:.6f} val_F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"    * New best: {best_val_f1:.4f} -> saved {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    # Reload best
    model.load_state_dict(torch.load(save_path, weights_only=True))
    return best_val_f1, model


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: Train both variants, evaluate all three (RBF + Gamma + Wavelet)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    device = CONFIG["device"]
    if not torch.cuda.is_available():
        device = "cpu"
        print("WARNING: CUDA not available, using CPU (will be slow)")

    print(f"{'='*70}")
    print("GeoKAN Variant Ablation: RBF vs Gamma vs Wavelet")
    print(f"{'='*70}")
    print(f"Config: {json.dumps(CONFIG, indent=2, default=str)}")

    # ── Load data (same split as train_geokan.py) ─────────────────────────────
    rio10_ids = load_rio10_test_scene_ids()
    graphs = load_all_graphs(CONFIG["cache_dir"], exclude_scene_ids=rio10_ids)
    if len(graphs) == 0:
        print("ERROR: No graphs loaded.")
        return

    train_graphs, val_graphs = split_by_scene(graphs, CONFIG["train_frac"])
    print(f"  Split: {len(train_graphs)} train / {len(val_graphs)} val")

    # Augment training data (same augmentation for both variants)
    train_graphs_aug = augment_training(
        train_graphs, CONFIG["augment_factor"], CONFIG["augment_jitter"]
    )

    # Compute class weights from original training data
    pos_weight = compute_class_weights(train_graphs, cap=CONFIG["pos_weight_cap"]).to(device)
    print(f"  Pos weights: {pos_weight.tolist()}")

    # Create datasets and loaders
    train_dataset = PyGGraphDataset(train_graphs_aug)
    val_dataset = PyGGraphDataset(val_graphs)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Train GeoKAN-Gamma
    # Uses lower peak lr (5e-4) because the Gamma metric is a single parameter
    # vector — much more sensitive to lr than the full MetricNet MLP.
    # Also uses longer warmup (10 epochs) for stability.
    # ══════════════════════════════════════════════════════════════════════════
    gamma_model = GeoKANVariantGNN(
        layer_cls=GeoKANGammaLayer,
        node_feat_dim=CONFIG["node_feat_dim"],
        edge_feat_dim=CONFIG["edge_feat_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_relations=NUM_RELATIONS,
        dropout=CONFIG["dropout"],
    ).to(device)

    gamma_save = os.path.join(CONFIG["save_dir"], "geokan_relation_gamma.pt")
    _, gamma_model = train_variant_full(
        "GeoKAN-Gamma", gamma_model, train_loader, val_loader,
        pos_weight, device, gamma_save,
        lr_override=5e-4, warmup_override=10,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Train GeoKAN-Wavelet
    # ══════════════════════════════════════════════════════════════════════════
    wavelet_model = GeoKANVariantGNN(
        layer_cls=GeoKANWaveletLayer,
        node_feat_dim=CONFIG["node_feat_dim"],
        edge_feat_dim=CONFIG["edge_feat_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_relations=NUM_RELATIONS,
        dropout=CONFIG["dropout"],
    ).to(device)

    wavelet_save = os.path.join(CONFIG["save_dir"], "geokan_relation_wavelet.pt")
    _, wavelet_model = train_variant_full(
        "GeoKAN-Wavelet", wavelet_model, train_loader, val_loader,
        pos_weight, device, wavelet_save
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Load existing RBF model (geokan_relation_v4.pt)
    # ══════════════════════════════════════════════════════════════════════════
    rbf_save = os.path.join(CONFIG["save_dir"], "geokan_relation_v4.pt")
    rbf_model = GeoKANRelationGNN(
        node_feat_dim=CONFIG["node_feat_dim"],
        edge_feat_dim=CONFIG["edge_feat_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_relations=NUM_RELATIONS,
        dropout=CONFIG["dropout"],
    ).to(device)

    if os.path.exists(rbf_save):
        rbf_model.load_state_dict(torch.load(rbf_save, map_location=device,
                                             weights_only=True))
        print(f"\n  Loaded existing RBF model from {rbf_save}")
    else:
        print(f"\n  WARNING: RBF model not found at {rbf_save}")
        print("  Skipping RBF evaluation.")
        rbf_model = None

    # ══════════════════════════════════════════════════════════════════════════
    # EVALUATION: Tune thresholds + compute Macro F1, R@3, R@5 for all three
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("FINAL EVALUATION (val set, tuned thresholds + constraints)")
    print(f"{'='*70}")

    results = {}

    models_to_eval = [
        ("GeoKAN-Gamma", gamma_model),
        ("GeoKAN-Wavelet", wavelet_model),
    ]
    if rbf_model is not None:
        models_to_eval.append(("GeoKAN-RBF (v4)", rbf_model))

    for name, model in models_to_eval:
        print(f"\n{'─'*60}")
        print(f"  Evaluating: {name}")
        print(f"{'─'*60}")

        # Tune thresholds on val set
        thresholds = tune_thresholds(model, val_loader, device)

        # Print per-relation thresholds
        print(f"  Tuned thresholds:")
        for rel_idx in range(NUM_RELATIONS):
            print(f"    {RELATION_NAMES[rel_idx]:20s} = {thresholds[rel_idx]:.2f}")

        # Evaluate with tuned thresholds + constraints
        macro_f1, per_rel, all_probs, all_labels = evaluate(
            model, val_loader, device, thresholds=thresholds, use_constraints=True
        )

        # Compute R@3 and R@5
        r_at_3 = compute_recall_at_k(all_probs, all_labels, 3)
        r_at_5 = compute_recall_at_k(all_probs, all_labels, 5)

        results[name] = {
            "macro_f1": macro_f1,
            "r_at_3": r_at_3,
            "r_at_5": r_at_5,
            "per_rel": per_rel,
            "thresholds": thresholds,
        }

        print(f"\n  {name} Results:")
        print(f"    Macro F1:  {macro_f1:.4f}")
        print(f"    R@3:       {r_at_3:.4f}")
        print(f"    R@5:       {r_at_5:.4f}")
        print(f"    Per-relation F1:")
        for rel_name, f1_val in per_rel.items():
            print(f"      {rel_name:20s}  {f1_val:.3f}")

        # Save thresholds
        thresh_path = os.path.join(
            CONFIG["save_dir"],
            f"geokan_relation_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_thresholds.json"
        )
        with open(thresh_path, "w") as f:
            json.dump({str(k): v for k, v in thresholds.items()}, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("3-WAY COMPARISON TABLE")
    print(f"{'='*70}")

    header = f"  {'Variant':<22s} {'Macro F1':>10s} {'R@3':>8s} {'R@5':>8s}"
    print(header)
    print(f"  {'─'*50}")

    for name in ["GeoKAN-RBF (v4)", "GeoKAN-Gamma", "GeoKAN-Wavelet"]:
        if name in results:
            r = results[name]
            print(f"  {name:<22s} {r['macro_f1']:>10.4f} "
                  f"{r['r_at_3']*100:>7.1f}% {r['r_at_5']*100:>7.1f}%")

    # Per-relation comparison
    print(f"\n  {'Relation':<20s}", end="")
    variant_names = [n for n in ["GeoKAN-RBF (v4)", "GeoKAN-Gamma", "GeoKAN-Wavelet"]
                     if n in results]
    for vn in variant_names:
        print(f" {vn:>16s}", end="")
    print()
    print(f"  {'─'*20}" + "─"*17*len(variant_names))

    for rel_idx in range(NUM_RELATIONS):
        rel_name = RELATION_NAMES[rel_idx]
        print(f"  {rel_name:<20s}", end="")
        for vn in variant_names:
            f1_val = results[vn]["per_rel"].get(rel_name, 0.0)
            print(f" {f1_val:>16.3f}", end="")
        print()

    # Save full results JSON
    results_path = os.path.join(CONFIG["save_dir"], "geokan_variant_comparison.json")
    # Convert numpy floats for JSON serialization
    json_results = {}
    for name, r in results.items():
        json_results[name] = {
            "macro_f1": float(r["macro_f1"]),
            "r_at_3": float(r["r_at_3"]),
            "r_at_5": float(r["r_at_5"]),
            "per_rel": {k: float(v) for k, v in r["per_rel"].items()},
            "thresholds": {str(k): float(v) for k, v in r["thresholds"].items()},
        }
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Full results saved to: {results_path}")

    print(f"\n{'='*70}")
    print("Done. Models saved:")
    print(f"  GeoKAN-Gamma:   {gamma_save}")
    print(f"  GeoKAN-Wavelet: {wavelet_save}")
    print(f"  GeoKAN-RBF:     {rbf_save} (pre-existing)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
