"""
GeoKAN-based Spatial Relation Predictor for 3D Scene Graphs.

Inspired by Sen et al., 2026 (arXiv:2605.06740) — learns a diagonal Riemannian
metric that warps input features before RBF basis expansion.

Architecture:
  1. Node encoder: Linear(10 → 128) + LayerNorm + GELU
  2. 2× GATv2Conv layers (4 heads, edge-aware) with residual connections
     - These provide neighborhood context (GeoKAN alone can't aggregate)
  3. Pair representation: [h_src | h_dst | h_src - h_dst | h_src * h_dst] → project to 64-dim
  4. Dual GeoKAN heads:
     - Contact head: GeoKAN on [pair(64) + all_edge_features(22)] = 86-dim → 4 outputs
       (on_top_of, under, attached_to, adjacent_to)
     - Directional head: GeoKAN on [pair(64) + directional_edge_features(10)] = 74-dim → 6 outputs
       (left_of, right_of, in_front_of, behind, higher_than, lower_than)

Edge features (22-dim, avg_diag/object-relative normalized for scale invariance):
  [0-4]  delta_x/y/z, xy_dist, dist_3d  (avg_diag normalized)
  [5-9]  bbox_overlap, vol_ratio, h_ratio, vert_gap, size_ratio_xy
  [10-16] contact/support/margin features (object-height normalized)
  [17]   containment_3d  (fraction of A inside B's bbox — key for inside)
  [18]   centroid_inside_b  (binary: A's centroid inside B's bbox)
  [19]   vz_ratio  (vertical/horizontal displacement ratio — stacking signal)
  [20]   containment_x  (fraction of A's X width inside B)
  [21]   containment_y  (fraction of A's Y depth inside B)

GeoKAN Layer internals:
  - BatchNorm on input
  - MetricNet: Linear(in_dim → metric_hidden) → GELU → Linear(metric_hidden → in_dim)
    Output: g = softplus(MetricNet(u)) + eps  (positive diagonal metric)
    Clamp g to [0.05, 16.0] for stability
  - Warp: z = u * sqrt(g)  (geometry-adapted coordinates)
  - RBF basis expansion: phi_k = exp(-gamma * (z_i - c_k)^2) for K=12 centers in [-3, 3]
  - Feature vector: [phi_flattened | u_original] (skip connection)
  - Linear mix: Linear(in_dim * K + in_dim → out_dim) → GELU → Dropout

MetricNet initialization:
  - Zero-init the last layer weights
  - Bias init to softplus_inv(1.0) so initial metric ≈ 1.0 (identity warp at start)

Metric regularization:
  - Each GeoKAN layer tracks: reg = log(g).pow(2).mean()
  - Total model.metric_reg() = sum of all layer regs
  - Add to loss: loss += 1e-4 * model.metric_reg()
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from src.relations.schema import NUM_RELATIONS, Relation


# ── Relation index groups (10-class schema) ───────────────────────────────────
# Contact head: 4 outputs (on_top_of, under, attached_to, adjacent_to)
# Directional head: 6 outputs (left_of, right_of, in_front_of, behind, higher_than, lower_than)

CONTACT_INDICES = [
    int(Relation.ON_TOP_OF),    # 0
    int(Relation.UNDER),        # 1
    int(Relation.ATTACHED_TO),  # 2
    int(Relation.ADJACENT_TO),  # 3
]

DIRECTIONAL_INDICES = [
    int(Relation.LEFT_OF),      # 4
    int(Relation.RIGHT_OF),     # 5
    int(Relation.IN_FRONT_OF),  # 6
    int(Relation.BEHIND),       # 7
    int(Relation.HIGHER_THAN),  # 8
    int(Relation.LOWER_THAN),   # 9
]


# ── GeoKAN Layer ──────────────────────────────────────────────────────────────

class GeoKANLayer(nn.Module):
    """
    Single GeoKAN layer: learns a per-edge adaptive Riemannian metric,
    warps features, expands via RBF basis, then linearly mixes.
    """

    def __init__(self, in_dim: int, out_dim: int, n_bases: int = 12,
                 metric_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_bases = n_bases

        # Input normalization
        self.bn = nn.BatchNorm1d(in_dim)

        # MetricNet: learns diagonal Riemannian metric g(u)
        # Output is in_dim-dimensional (one metric coefficient per input dimension)
        self.metric_net = nn.Sequential(
            nn.Linear(in_dim, metric_hidden),
            nn.GELU(),
            nn.Linear(metric_hidden, in_dim),
        )

        # Initialize MetricNet so initial metric ≈ 1.0 (identity warp)
        # softplus(x) = 1.0 when x = log(e - 1) ≈ 0.5414
        softplus_inv_one = math.log(math.e - 1.0)
        nn.init.zeros_(self.metric_net[2].weight)
        nn.init.constant_(self.metric_net[2].bias, softplus_inv_one)

        # RBF centers: K evenly spaced in [-3, 3]. BatchNorm normalizes inputs
        # to ~N(0,1) before the metric warp, so centers cover >3σ of the distribution.
        # n_bases=12 gives finer resolution than the original 8.
        centers = torch.linspace(-3.0, 3.0, n_bases)
        self.register_buffer("centers", centers)  # (K,)

        # RBF bandwidth (learned)
        self.gamma = nn.Parameter(torch.tensor(1.0))

        # Linear mix: [phi_flattened(in_dim * K) | u_original(in_dim)] → out_dim
        mix_in_dim = in_dim * n_bases + in_dim
        self.linear_mix = nn.Sequential(
            nn.Linear(mix_in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Store last metric for regularization
        self._last_g: torch.Tensor | None = None

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (B, in_dim) input features

        Returns:
            out: (B, out_dim) transformed features
        """
        # Normalize input
        u_normed = self.bn(u)

        # Compute adaptive metric: g = softplus(MetricNet(u)) + eps
        g_raw = self.metric_net(u_normed)
        g = F.softplus(g_raw) + 1e-6
        g = g.clamp(0.05, 16.0)  # stability clamp

        # Store for regularization
        self._last_g = g

        # Warp: z = u * sqrt(g) — geometry-adapted coordinates
        z = u_normed * torch.sqrt(g)  # (B, in_dim)

        # RBF basis expansion: phi_k(z_i) = exp(-gamma * (z_i - c_k)^2)
        # z: (B, in_dim) → (B, in_dim, 1)
        # centers: (K,) → (1, 1, K)
        z_expanded = z.unsqueeze(-1)  # (B, in_dim, 1)
        centers = self.centers.view(1, 1, -1)  # (1, 1, K)
        gamma = F.softplus(self.gamma)

        phi = torch.exp(-gamma * (z_expanded - centers).pow(2))  # (B, in_dim, K)
        phi_flat = phi.reshape(u.shape[0], -1)  # (B, in_dim * K)

        # Concatenate with skip connection (original normalized input)
        features = torch.cat([phi_flat, u_normed], dim=-1)  # (B, in_dim*K + in_dim)

        # Linear mix to output
        out = self.linear_mix(features)
        return out

    def metric_regularization(self) -> torch.Tensor:
        """log(g)^2 regularization to keep metric near identity."""
        if self._last_g is None:
            return torch.tensor(0.0)
        return self._last_g.log().pow(2).mean()


# ── GeoKAN Head (stacked layers) ─────────────────────────────────────────────

class GeoKANHead(nn.Module):
    """
    Two stacked GeoKAN layers + linear output head.
    Specialized for either contact or directional relations.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 6,
                 n_bases: int = 12, metric_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.layer1 = GeoKANLayer(in_dim, hidden_dim, n_bases=n_bases,
                                  metric_hidden=metric_hidden, dropout=dropout)
        self.layer2 = GeoKANLayer(hidden_dim, hidden_dim, n_bases=n_bases,
                                  metric_hidden=metric_hidden, dropout=dropout)
        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x)
        h = self.layer2(h)
        return self.output(h)

    def metric_regularization(self) -> torch.Tensor:
        return (self.layer1.metric_regularization() +
                self.layer2.metric_regularization())


# ── Main Model ────────────────────────────────────────────────────────────────

class GeoKANRelationGNN(nn.Module):
    """
    GeoKAN-based relation predictor with GATv2 context aggregation.

    Same interface as RelationGNN:
        forward(x, edge_index, edge_attr) → (E, 12) logits
    """

    def __init__(
        self,
        node_feat_dim: int = 10,
        edge_feat_dim: int = 22,
        hidden_dim: int = 128,
        num_relations: int = NUM_RELATIONS,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.edge_feat_dim = edge_feat_dim
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim

        # ── 1. Node encoder ───────────────────────────────────────────────────
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ── 2. GATv2Conv layers (edge-aware, 4 heads) ────────────────────────
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

        # ── 3. Pair projection ────────────────────────────────────────────────
        # [h_src | h_dst | h_src - h_dst | h_src * h_dst] = 4 * hidden_dim
        pair_raw_dim = 4 * hidden_dim
        pair_dim = 64
        self.pair_proj = nn.Sequential(
            nn.Linear(pair_raw_dim, pair_dim),
            nn.LayerNorm(pair_dim),
            nn.GELU(),
        )

        # ── 4. Dual GeoKAN heads ─────────────────────────────────────────────
        # Contact head: pair(64) + all_edge_features(22) = 86 → 4 outputs
        # on_top_of, under, attached_to, adjacent_to
        contact_in_dim = pair_dim + edge_feat_dim  # 64 + 22 = 86
        self.head_contact = GeoKANHead(
            in_dim=contact_in_dim, hidden_dim=128, out_dim=len(CONTACT_INDICES),
            n_bases=12, metric_hidden=64, dropout=dropout,
        )

        # Directional head: pair(64) + directional_edge_features(10) = 74 → 6 outputs
        # left_of, right_of, in_front_of, behind, higher_than, lower_than
        directional_in_dim = pair_dim + 10  # 64 + 10 = 74
        self.head_directional = GeoKANHead(
            in_dim=directional_in_dim, hidden_dim=128, out_dim=len(DIRECTIONAL_INDICES),
            n_bases=12, metric_hidden=64, dropout=dropout,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (N, node_feat_dim) node features
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_feat_dim) edge features

        Returns:
            logits: (E, 12) relation logits in schema order
        """
        # 1. Node encoding
        h = self.node_encoder(x)

        # 2. GATv2Conv with residual connections (edge-aware)
        h1 = self.conv1(h, edge_index, edge_attr=edge_attr)
        h = self.norm1(F.gelu(h1) + h)
        h = self.dropout(h)

        h2 = self.conv2(h, edge_index, edge_attr=edge_attr)
        h = self.norm2(F.gelu(h2) + h)
        h = self.dropout(h)

        # 3. Pair representation
        src, dst = edge_index[0], edge_index[1]
        h_src = h[src]  # (E, hidden_dim)
        h_dst = h[dst]  # (E, hidden_dim)

        pair_raw = torch.cat([
            h_src,
            h_dst,
            h_src - h_dst,
            h_src * h_dst,
        ], dim=-1)  # (E, 4*hidden_dim)

        pair = self.pair_proj(pair_raw)  # (E, 64)

        # 4. Prepare edge features for each head
        if edge_attr is not None:
            all_edge_feats = edge_attr  # (E, 19)
            dir_edge_feats = edge_attr[:, :10]  # (E, 10) — features [0-9]
        else:
            all_edge_feats = torch.zeros(pair.shape[0], self.edge_feat_dim,
                                         device=pair.device)
            dir_edge_feats = torch.zeros(pair.shape[0], 10, device=pair.device)

        # Contact head input: [pair | all_edge_features]
        contact_input = torch.cat([pair, all_edge_feats], dim=-1)  # (E, 81)
        out_contact = self.head_contact(contact_input)  # (E, 6)

        # Directional head input: [pair | directional_edge_features]
        dir_input = torch.cat([pair, dir_edge_feats], dim=-1)  # (E, 74)
        out_dir = self.head_directional(dir_input)  # (E, 6)

        # 5. Assemble in schema order: contact[0-3] + directional[4-9]
        out = torch.cat([out_contact, out_dir], dim=-1)  # (E, 10)
        return out

    def metric_reg(self) -> torch.Tensor:
        """Total metric regularization across all GeoKAN layers."""
        return (self.head_contact.metric_regularization() +
                self.head_directional.metric_regularization())

    def predict(self, x, edge_index, edge_attr=None):
        """Convenience method for inference (argmax)."""
        with torch.no_grad():
            return self.forward(x, edge_index, edge_attr).argmax(dim=-1)
