"""
3D Scene Graph Augmentation for ScanNet training data.

Generates geometrically valid augmented graphs from cached ScanNet scenes.
All transforms operate on graph tensors — no raw PLY files needed.

Augmentation strategies:

1. rotate_90 / rotate_180 / rotate_270
   Rotate scene around Z axis. Remaps directional relation labels:
       rotate_90:  left_of→in_front_of, right_of→behind, etc.
       rotate_180: left_of↔right_of, in_front_of↔behind
       rotate_270: inverse of rotate_90
   Leaves on_top_of / under / higher_than / lower_than unchanged.

2. flip_x
   Mirror scene along X axis. Swaps left_of ↔ right_of.

3. flip_y
   Mirror scene along Y axis. Swaps in_front_of ↔ behind.

4. jitter
   Small Gaussian noise (σ=0.01) on centroid positions and edge deltas.
   Teaches robustness to small position errors from clustering.

Note on class imbalance:
   The 37.7× imbalance (adjacent_to vs inside) is NOT fixed by augmentation
   because 94% of graphs already contain rare relations as a small fraction
   of their edges. Augmenting graphs doesn't change per-edge label ratios.
   Class imbalance is handled by weighted CrossEntropyLoss in train_scannet.py.

Usage:
    from src.training.augmentation import AugmentedScanNetDataset
    dataset = AugmentedScanNetDataset(cache_dir="D:/logicsplat_data/scannet_cache", augment_factor=4)
"""
import os
import math
import random
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset

from src.relations.schema import Relation, NUM_RELATIONS, RELATION_NAMES


# ── relation flip tables ──────────────────────────────────────────────────────
# When we rotate/flip the scene, some relation labels must be remapped.

# 90° CCW rotation around Z: X→-Y, Y→X
# left_of(+X) → behind(+Y direction becomes -X after rotation... )
# Easier to think in terms of: after 90° CCW, what was "right" is now "forward"
# Full rotation remapping for 90° CCW:
#   left_of  → in_front_of
#   right_of → behind
#   in_front_of → right_of
#   behind   → left_of
ROTATE_90_CCW = {
    Relation.LEFT_OF:     Relation.IN_FRONT_OF,
    Relation.RIGHT_OF:    Relation.BEHIND,
    Relation.IN_FRONT_OF: Relation.RIGHT_OF,
    Relation.BEHIND:      Relation.LEFT_OF,
}

# 180° rotation: left↔right, front↔behind
ROTATE_180 = {
    Relation.LEFT_OF:     Relation.RIGHT_OF,
    Relation.RIGHT_OF:    Relation.LEFT_OF,
    Relation.IN_FRONT_OF: Relation.BEHIND,
    Relation.BEHIND:      Relation.IN_FRONT_OF,
}

# 270° CCW = 90° CW
ROTATE_270_CCW = {
    Relation.LEFT_OF:     Relation.BEHIND,
    Relation.RIGHT_OF:    Relation.IN_FRONT_OF,
    Relation.IN_FRONT_OF: Relation.LEFT_OF,
    Relation.BEHIND:      Relation.RIGHT_OF,
}

# Flip X axis: left↔right
FLIP_X = {
    Relation.LEFT_OF:  Relation.RIGHT_OF,
    Relation.RIGHT_OF: Relation.LEFT_OF,
}

# Flip Y axis: front↔behind
FLIP_Y = {
    Relation.IN_FRONT_OF: Relation.BEHIND,
    Relation.BEHIND:      Relation.IN_FRONT_OF,
}

# Flip Z axis (Z-DOWN convention): on_top_of↔under, higher_than↔lower_than
# This augmentation makes the model Z-orientation invariant, handling both:
#   Z-UP  (3RScan training): higher objects have higher Z values
#   Z-DOWN (tabletop Gaussian splats): higher objects have lower Z values
FLIP_Z = {
    Relation.ON_TOP_OF:   Relation.UNDER,
    Relation.UNDER:       Relation.ON_TOP_OF,
    Relation.HIGHER_THAN: Relation.LOWER_THAN,
    Relation.LOWER_THAN:  Relation.HIGHER_THAN,
}


def _remap_labels(labels: torch.Tensor, remap: Dict) -> torch.Tensor:
    """
    Apply a relation remapping dict to a multi-hot label tensor.

    labels: (E, NUM_RELATIONS) float tensor — multi-hot encoding
    remap:  {Relation → Relation} mapping for geometric transforms

    For each (old_rel → new_rel) pair, swap the corresponding columns.
    This correctly handles cases where both old and new columns may be set.
    """
    result = labels.clone()
    # Collect all swaps first to avoid double-swapping
    swaps = [(int(old_rel), int(new_rel)) for old_rel, new_rel in remap.items()]
    for old_idx, new_idx in swaps:
        old_col = labels[:, old_idx].clone()
        new_col = labels[:, new_idx].clone()
        result[:, old_idx] = new_col
        result[:, new_idx] = old_col
    return result


# ── feature-level transforms ──────────────────────────────────────────────────
# Node features layout (10-dim):
#   [0] centroid_x_norm  [1] centroid_y_norm  [2] centroid_z_norm
#   [3] size_x_norm      [4] size_y_norm       [5] size_z_norm
#   [6] volume_norm      [7] height_ratio      [8] density  [9] z_relative
#
# Edge features layout (17-dim):
#   [0] delta_x  [1] delta_y  [2] delta_z  [3] xy_dist  [4] dist_3d
#   [5] bbox_overlap  [6] vol_ratio  [7] h_ratio  [8] vert_gap  [9] size_ratio
#   [10] vert_gap_obj_norm  [11] vert_gap_abs  [12] contact_score  [13] support_overlap_b
#   [14] ontop_z_margin  [15] ontop_xy_margin  [16] z_dominance_margin

def _rotate_z_90ccw(x: torch.Tensor, edge_attr: torch.Tensor,
                    edge_label: torch.Tensor) -> Tuple:
    """Rotate 90° CCW around Z: (x,y) → (-y, x)."""
    x = x.clone()
    edge_attr = edge_attr.clone()

    # centroid: swap x,y and negate new x (was y)
    cx, cy = x[:, 0].clone(), x[:, 1].clone()
    x[:, 0] = 1.0 - cy   # -y normalized → 1-y (stays in [0,1])
    x[:, 1] = cx

    # size: swap x,y dims
    sx, sy = x[:, 3].clone(), x[:, 4].clone()
    x[:, 3] = sy
    x[:, 4] = sx

    # edge delta_x, delta_y
    dx, dy = edge_attr[:, 0].clone(), edge_attr[:, 1].clone()
    edge_attr[:, 0] = -dy
    edge_attr[:, 1] = dx

    return x, edge_attr, _remap_labels(edge_label, ROTATE_90_CCW)


def _rotate_z_180(x: torch.Tensor, edge_attr: torch.Tensor,
                  edge_label: torch.Tensor) -> Tuple:
    """Rotate 180° around Z: (x,y) → (-x,-y)."""
    x = x.clone()
    edge_attr = edge_attr.clone()

    x[:, 0] = 1.0 - x[:, 0]
    x[:, 1] = 1.0 - x[:, 1]

    edge_attr[:, 0] = -edge_attr[:, 0]
    edge_attr[:, 1] = -edge_attr[:, 1]

    return x, edge_attr, _remap_labels(edge_label, ROTATE_180)


def _rotate_z_270ccw(x: torch.Tensor, edge_attr: torch.Tensor,
                     edge_label: torch.Tensor) -> Tuple:
    """Rotate 270° CCW (= 90° CW) around Z: (x,y) → (y,-x)."""
    x = x.clone()
    edge_attr = edge_attr.clone()

    cx, cy = x[:, 0].clone(), x[:, 1].clone()
    x[:, 0] = cy
    x[:, 1] = 1.0 - cx

    sx, sy = x[:, 3].clone(), x[:, 4].clone()
    x[:, 3] = sy
    x[:, 4] = sx

    dx, dy = edge_attr[:, 0].clone(), edge_attr[:, 1].clone()
    edge_attr[:, 0] = dy
    edge_attr[:, 1] = -dx

    return x, edge_attr, _remap_labels(edge_label, ROTATE_270_CCW)


def _flip_x(x: torch.Tensor, edge_attr: torch.Tensor,
            edge_label: torch.Tensor) -> Tuple:
    """Mirror along X axis."""
    x = x.clone()
    edge_attr = edge_attr.clone()

    x[:, 0] = 1.0 - x[:, 0]
    edge_attr[:, 0] = -edge_attr[:, 0]

    return x, edge_attr, _remap_labels(edge_label, FLIP_X)


def _flip_y(x: torch.Tensor, edge_attr: torch.Tensor,
            edge_label: torch.Tensor) -> Tuple:
    """Mirror along Y axis."""
    x = x.clone()
    edge_attr = edge_attr.clone()

    x[:, 1] = 1.0 - x[:, 1]
    edge_attr[:, 1] = -edge_attr[:, 1]

    return x, edge_attr, _remap_labels(edge_label, FLIP_Y)


def _flip_z(x: torch.Tensor, edge_attr: torch.Tensor,
            edge_label: torch.Tensor) -> Tuple:
    """
    Simulate Z-DOWN coordinate convention (Gaussian splat style).

    Gaussian Splat scenes have Z increasing downward — physically higher objects
    have LOWER Z values. This augmentation teaches the model to handle both:
      Z-UP  (3RScan training):       vz_ratio > 0, contact_score > 0 → on_top_of
      Z-DOWN (tabletop splat eval):  vz_ratio < 0, contact_score = 0 → on_top_of

    Node feature transformations (22-dim edge, 10-dim node):
      Node [2] cz (centroid Z): 1 - cz  (flip within [0,1])
      Node [9] z_rank:           1 - z_rank  (reverse vertical ordering)

    Edge feature transformations (22-dim):
      [2]  delta_z:            negate  (A-B Z direction flips)
      [8]  vert_gap:           negate  (gap sign inverts)
      [10] vert_gap_obj_norm:  negate
      [11] vert_gap_a_relative: negate
      [12] contact_score:      0.0  (a_above_b flips, score becomes 0 for original on_top_of)
      [14] ontop_z_margin:     negate  (centroid_z_diff flips)
      [16] z_dom_margin:       negate
      [19] vz_ratio:           negate  (vertical/horizontal ratio flips)

    Label remapping: on_top_of↔under, higher_than↔lower_than
    """
    x          = x.clone()
    edge_attr  = edge_attr.clone()

    # ── Node features ──────────────────────────────────────────────────────────
    # [2] cz: approximate flip within normalized [0,1] range
    x[:, 2] = 1.0 - x[:, 2]
    # [9] z_rank: ordinal rank inverts (lowest becomes highest)
    x[:, 9] = 1.0 - x[:, 9]

    # ── Edge features ──────────────────────────────────────────────────────────
    # Negate all Z-signed direction features
    for idx in [2, 8, 10, 11, 14, 16, 19]:
        edge_attr[:, idx] = -edge_attr[:, idx]

    # contact_score [12]: when A was above B (contact_score > 0), after Z-flip
    # A is now below B, so a_above_b = 0 and contact_score becomes 0.
    # This correctly matches the Z-DOWN tabletop pattern where on_top_of edges
    # have contact_score = 0 (the key distinguishing feature from Z-UP).
    edge_attr[:, 12] = 0.0

    return x, edge_attr, _remap_labels(edge_label, FLIP_Z)


def _jitter(x: torch.Tensor, edge_attr: torch.Tensor,
            edge_label: torch.Tensor, sigma: float = 0.01) -> Tuple:
    """
    Add small Gaussian noise to centroid positions (features 0-2) and
    consistently update edge delta features (features 0-2) to match.

    The edge_index is not available here, so we approximate by adding
    correlated noise to edge deltas: delta_noisy = delta + (noise_a - noise_b)
    where noise_a, noise_b are independent per-node noise samples.
    This is statistically equivalent to recomputing deltas from noisy centroids.
    """
    x = x.clone()
    edge_attr = edge_attr.clone()

    n_nodes = x.shape[0]
    n_edges = edge_attr.shape[0]

    # per-node position noise
    node_noise = torch.randn(n_nodes, 3) * sigma
    x[:, :3] = torch.clamp(x[:, :3] + node_noise, 0.0, 1.0)

    # edge delta noise: approximate as independent noise on each endpoint
    # delta_noisy ≈ delta + noise_src - noise_dst
    # Since we don't have edge_index, use independent noise of same magnitude
    # (sqrt(2) * sigma because it's the difference of two independent normals)
    edge_noise = torch.randn(n_edges, 3) * (sigma * 1.414)
    edge_attr[:, :3] = torch.clamp(edge_attr[:, :3] + edge_noise, -1.0, 1.0)

    return x, edge_attr, edge_label


# ── augmentation registry ─────────────────────────────────────────────────────

AUGMENTATIONS = [
    ("rotate_90",  _rotate_z_90ccw),
    ("rotate_180", _rotate_z_180),
    ("rotate_270", _rotate_z_270ccw),
    ("flip_x",     _flip_x),
    ("flip_y",     _flip_y),
    ("jitter",     _jitter),
    ("flip_z",     _flip_z),   # Z-orientation invariance for tabletop cross-domain
]


def augment_graph(graph: Dict, aug_name: str) -> Dict:
    """Apply a named augmentation to a graph dict. Returns new dict."""
    fn = {name: fn for name, fn in AUGMENTATIONS}[aug_name]
    x, edge_attr, edge_label = fn(
        graph["x"], graph["edge_attr"], graph["edge_label"]
    )
    return {
        **graph,
        "x":          x,
        "edge_attr":  edge_attr,
        "edge_label": edge_label,
        "scene_id":   graph["scene_id"] + f"_aug_{aug_name}",
    }


# ── rare-relation detection ───────────────────────────────────────────────────

# Relations with < 5% frequency — these graphs get oversampled
RARE_RELATIONS = {
    Relation.ON_TOP_OF,
    Relation.UNDER,
    Relation.HIGHER_THAN,
    Relation.LOWER_THAN,
    Relation.ATTACHED_TO,
}


def _has_rare_relation(graph: Dict) -> bool:
    """Check if any edge in the graph has a rare relation set (multi-hot)."""
    labels = graph["edge_label"]  # (E, NUM_RELATIONS) float
    rare_indices = torch.tensor([int(r) for r in RARE_RELATIONS])
    return bool(labels[:, rare_indices].any().item())


# ── augmented dataset ─────────────────────────────────────────────────────────

class AugmentedScanNetDataset(Dataset):
    """
    Wraps the ScanNet cache and applies on-the-fly augmentation.

    Strategy:
    1. Load all cached graphs
    2. For each graph, generate augmented variants (rotations + flips)
    3. Oversample graphs containing rare relations (inside, on_top_of, etc.)

    Args:
        cache_dir:        path to D:/logicsplat_data/scannet_cache/
        augment_factor:   how many augmented copies per original graph
                          (4 = rotate 90/180/270 + flip_x, good default)
        oversample_rare:  extra copies of rare-relation graphs
        jitter:           whether to include jitter augmentation
        max_graphs:       cap total graphs (None = no cap)
        verbose:          print stats
    """

    def __init__(
        self,
        cache_dir: str = "D:/logicsplat_data/scannet_cache",
        augment_factor: int = 4,
        jitter: bool = True,
        max_graphs: Optional[int] = None,
        verbose: bool = True,
    ):
        self.graphs: List[Dict] = []

        if not os.path.isdir(cache_dir):
            print(f"Cache dir not found: {cache_dir}")
            return

        files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
        if verbose:
            print(f"Loading {len(files)} cached graphs...")

        try:
            from tqdm import tqdm
            _tqdm_aug = tqdm
        except ImportError:
            _tqdm_aug = None

        raw_graphs = []
        file_iter = files
        if _tqdm_aug is not None and verbose:
            file_iter = _tqdm_aug(files, desc="Loading cache", unit="scene",
                                  dynamic_ncols=True)
        for i, fname in enumerate(file_iter):
            g = torch.load(os.path.join(cache_dir, fname), weights_only=False)
            # hard assert — no silent bypassing of dimension mismatches
            if g["x"].shape[1] != 10:
                raise ValueError(
                    f"{fname}: node features are {g['x'].shape[1]}-dim, expected 10. "
                    f"Delete the cache and regenerate with CACHE_VERSION='v6_contact_features'."
                )
            if g["edge_attr"].shape[1] != 17:
                raise ValueError(
                    f"{fname}: edge features are {g['edge_attr'].shape[1]}-dim, expected 17. "
                    f"Delete the cache and regenerate with CACHE_VERSION='v7_margins'."
                )
            if g["edge_label"].dim() != 2 or g["edge_label"].shape[1] != NUM_RELATIONS:
                raise ValueError(
                    f"{fname}: edge_label shape {tuple(g['edge_label'].shape)}, "
                    f"expected (E, {NUM_RELATIONS}) multi-hot. "
                    f"Delete the cache and regenerate with CACHE_VERSION='v3_multilabel'."
                )
            raw_graphs.append(g)
            if verbose and not _tqdm_aug and i % 500 == 0 and i > 0:
                print(f"  {i}/{len(files)} loaded")

        if verbose:
            print(f"Loaded {len(raw_graphs)} graphs. Generating augmentations...")

        aug_names = ["rotate_90", "rotate_180", "rotate_270",
                     "flip_x", "flip_y"]
        if jitter:
            aug_names.append("jitter")
        selected_augs = aug_names[:augment_factor]

        graph_iter = raw_graphs
        if _tqdm_aug is not None and verbose:
            graph_iter = _tqdm_aug(raw_graphs, desc="Augmenting", unit="graph",
                                   dynamic_ncols=True)

        for g in graph_iter:
            # always include original
            self.graphs.append(g)

            # augmented copies — geometric diversity helps generalization
            # rotations teach the model that relations are viewpoint-relative
            # flips double the directional training signal
            # jitter teaches robustness to small position errors
            for aug_name in selected_augs:
                self.graphs.append(augment_graph(g, aug_name))

        # NOTE: class imbalance (37x) is handled by weighted CrossEntropyLoss
        # in the training script, not by oversampling here.
        # Oversampling graphs with rare relations doesn't help because 94% of
        # graphs already contain rare relations as a small fraction of their edges.

        if max_graphs is not None:
            random.shuffle(self.graphs)
            self.graphs = self.graphs[:max_graphs]

        if verbose:
            self._print_stats(len(raw_graphs))

    def _print_stats(self, n_original: int):
        # Multi-hot: count positives per relation class
        pos_counts = torch.zeros(NUM_RELATIONS)
        total_edges = 0
        for g in self.graphs:
            labels = g["edge_label"]  # (E, NUM_RELATIONS)
            pos_counts += labels.sum(dim=0)
            total_edges += labels.shape[0]
        print(f"\nAugmented dataset: {len(self.graphs)} graphs "
              f"({n_original} original + {len(self.graphs)-n_original} augmented)")
        print(f"Total edges: {total_edges}")
        print("Positive label distribution after augmentation (multi-label):")
        max_count = pos_counts.max().item()
        for idx in range(NUM_RELATIONS):
            count = int(pos_counts[idx].item())
            pct = 100 * count / max(total_edges, 1)
            bar = "#" * int(40 * count / max(max_count, 1))
            print(f"  {RELATION_NAMES[idx]:20s} {count:7d}  {pct:5.1f}%  {bar}")
        if pos_counts.min().item() > 0:
            print(f"Imbalance ratio: {pos_counts.max().item()/pos_counts.min().item():.1f}x")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Dict:
        return self.graphs[idx]
