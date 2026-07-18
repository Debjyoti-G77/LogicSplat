"""
GeoKAN-γ Variant: Separable Per-Dimension Metric (Lightest).

Instead of a coupled MetricNet MLP that takes the full input, this variant uses
a per-dimension learnable scalar γ_i (no MLP). The metric is simply:

    g_i = softplus(γ_i)

This is the simplest GeoKAN variant: it learns which dimensions to stretch or
compress, but the metric is input-independent (same warp for all samples).
Much fewer parameters than the full MetricNet.

Reference: Sen et al., 2026 (arXiv:2605.06740) — GeoKAN-γ variant.

Architecture is identical to GeoKANRelationGNN except GeoKANLayer is replaced
with GeoKANGammaLayer.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from src.relations.schema import NUM_RELATIONS, Relation


# ── Relation index groups (10-class schema) ───────────────────────────────────

CONTACT_INDICES = [
    int(Relation.ON_TOP_OF),
    int(Relation.UNDER),
    int(Relation.ATTACHED_TO),
    int(Relation.ADJACENT_TO),
]

DIRECTIONAL_INDICES = [
    int(Relation.LEFT_OF),
    int(Relation.RIGHT_OF),
    int(Relation.IN_FRONT_OF),
    int(Relation.BEHIND),
    int(Relation.HIGHER_THAN),
    int(Relation.LOWER_THAN),
]


# ── GeoKAN-γ Layer (separable metric) ────────────────────────────────────────

class GeoKANGammaLayer(nn.Module):
    """
    GeoKAN layer with per-dimension learnable scalar metric (no MLP).
    
    Metric: g_i = softplus(γ_i) — one learnable parameter per input dimension.
    The metric is input-independent: same warp applied to all samples.
    This is the lightest possible GeoKAN variant.
    """

    def __init__(self, in_dim: int, out_dim: int, n_bases: int = 12,
                 dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_bases = n_bases

        # Input normalization
        self.bn = nn.BatchNorm1d(in_dim)

        # Per-dimension learnable metric scalars γ_i
        # Initialize so softplus(γ_i) ≈ 1.0 (identity warp at start)
        softplus_inv_one = math.log(math.e - 1.0)
        self.gamma_params = nn.Parameter(
            torch.full((in_dim,), softplus_inv_one)
        )

        # RBF centers: K evenly spaced in [-3, 3]
        centers = torch.linspace(-3.0, 3.0, n_bases)
        self.register_buffer("centers", centers)

        # RBF bandwidth (learned)
        self.rbf_gamma = nn.Parameter(torch.tensor(1.0))

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

        # Compute separable metric: g_i = softplus(γ_i) — same for all samples
        g = F.softplus(self.gamma_params) + 1e-6  # (in_dim,)
        g = g.clamp(0.05, 16.0)

        # Store for regularization
        self._last_g = g

        # Warp: z = u * sqrt(g) — geometry-adapted coordinates
        z = u_normed * torch.sqrt(g).unsqueeze(0)  # (B, in_dim)

        # RBF basis expansion: phi_k(z_i) = exp(-gamma * (z_i - c_k)^2)
        z_expanded = z.unsqueeze(-1)  # (B, in_dim, 1)
        centers = self.centers.view(1, 1, -1)  # (1, 1, K)
        rbf_gamma = F.softplus(self.rbf_gamma)

        phi = torch.exp(-rbf_gamma * (z_expanded - centers).pow(2))  # (B, in_dim, K)
        phi_flat = phi.reshape(u.shape[0], -1)  # (B, in_dim * K)

        # Concatenate with skip connection
        features = torch.cat([phi_flat, u_normed], dim=-1)

        # Linear mix to output
        out = self.linear_mix(features)
        return out

    def metric_regularization(self) -> torch.Tensor:
        """log(g)^2 regularization to keep metric near identity."""
        if self._last_g is None:
            return torch.tensor(0.0)
        return self._last_g.log().pow(2).mean()


# ── GeoKAN-γ Head (stacked layers) ───────────────────────────────────────────

class GeoKANGammaHead(nn.Module):
    """Two stacked GeoKAN-γ layers + linear output head."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 6,
                 n_bases: int = 12, dropout: float = 0.2):
        super().__init__()
        self.layer1 = GeoKANGammaLayer(in_dim, hidden_dim, n_bases=n_bases,
                                       dropout=dropout)
        self.layer2 = GeoKANGammaLayer(hidden_dim, hidden_dim, n_bases=n_bases,
                                       dropout=dropout)
        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x)
        h = self.layer2(h)
        return self.output(h)

    def metric_regularization(self) -> torch.Tensor:
        return (self.layer1.metric_regularization() +
                self.layer2.metric_regularization())


# ── Main Model ────────────────────────────────────────────────────────────────

class GeoKANGammaRelationGNN(nn.Module):
    """
    GeoKAN-γ relation predictor: separable per-dimension metric (no MetricNet MLP).
    Same interface as GeoKANRelationGNN.
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
        pair_raw_dim = 4 * hidden_dim
        pair_dim = 64
        self.pair_proj = nn.Sequential(
            nn.Linear(pair_raw_dim, pair_dim),
            nn.LayerNorm(pair_dim),
            nn.GELU(),
        )

        # ── 4. Dual GeoKAN-γ heads ───────────────────────────────────────────
        contact_in_dim = pair_dim + edge_feat_dim  # 64 + 22 = 86
        self.head_contact = GeoKANGammaHead(
            in_dim=contact_in_dim, hidden_dim=128, out_dim=len(CONTACT_INDICES),
            n_bases=12, dropout=dropout,
        )

        directional_in_dim = pair_dim + 10  # 64 + 10 = 74
        self.head_directional = GeoKANGammaHead(
            in_dim=directional_in_dim, hidden_dim=128, out_dim=len(DIRECTIONAL_INDICES),
            n_bases=12, dropout=dropout,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        # 1. Node encoding
        h = self.node_encoder(x)

        # 2. GATv2Conv with residual connections
        h1 = self.conv1(h, edge_index, edge_attr=edge_attr)
        h = self.norm1(F.gelu(h1) + h)
        h = self.dropout(h)

        h2 = self.conv2(h, edge_index, edge_attr=edge_attr)
        h = self.norm2(F.gelu(h2) + h)
        h = self.dropout(h)

        # 3. Pair representation
        src, dst = edge_index[0], edge_index[1]
        h_src = h[src]
        h_dst = h[dst]

        pair_raw = torch.cat([
            h_src, h_dst, h_src - h_dst, h_src * h_dst,
        ], dim=-1)
        pair = self.pair_proj(pair_raw)

        # 4. Prepare edge features for each head
        if edge_attr is not None:
            all_edge_feats = edge_attr
            dir_edge_feats = edge_attr[:, :10]
        else:
            all_edge_feats = torch.zeros(pair.shape[0], self.edge_feat_dim,
                                         device=pair.device)
            dir_edge_feats = torch.zeros(pair.shape[0], 10, device=pair.device)

        # Contact head
        contact_input = torch.cat([pair, all_edge_feats], dim=-1)
        out_contact = self.head_contact(contact_input)

        # Directional head
        dir_input = torch.cat([pair, dir_edge_feats], dim=-1)
        out_dir = self.head_directional(dir_input)

        # 5. Assemble in schema order
        out = torch.cat([out_contact, out_dir], dim=-1)
        return out

    def metric_reg(self) -> torch.Tensor:
        """Total metric regularization across all GeoKAN-γ layers."""
        return (self.head_contact.metric_regularization() +
                self.head_directional.metric_regularization())

    def predict(self, x, edge_index, edge_attr=None):
        """Convenience method for inference."""
        with torch.no_grad():
            return self.forward(x, edge_index, edge_attr).argmax(dim=-1)
