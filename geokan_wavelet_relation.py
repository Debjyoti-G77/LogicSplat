"""
GeoKAN-Wavelet Variant: Mexican Hat Wavelet Basis.

Same architecture as GeoKAN-RBF but replaces the RBF (Gaussian) basis functions
with Mexican hat (Ricker) wavelets:

    ψ(z) = (1 - z²) * exp(-z²/2)

The Mexican hat wavelet has a zero-crossing structure that captures sharper
boundaries than smooth RBFs. This is hypothesized to be better for contact
detection (on_top_of/under) where the decision boundary is a sharp threshold
in vertical gap space.

Reference: Sen et al., 2026 (arXiv:2605.06740) — wavelet basis variant.
Ricker wavelet: https://en.wikipedia.org/wiki/Mexican_hat_wavelet

Architecture is identical to GeoKANRelationGNN except the basis expansion
uses Mexican hat wavelets instead of RBFs.
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


# ── GeoKAN-Wavelet Layer ─────────────────────────────────────────────────────

class GeoKANWaveletLayer(nn.Module):
    """
    GeoKAN layer with Mexican hat wavelet basis expansion.
    
    Same as GeoKANLayer but replaces:
        RBF: phi_k = exp(-gamma * (z - c_k)^2)
    with:
        Wavelet: psi_k = (1 - t^2) * exp(-t^2 / 2),  where t = (z - c_k) * sigma
    
    The Mexican hat wavelet has:
    - A positive peak at center
    - Negative lobes on either side
    - Zero crossings at t = ±1
    
    This creates sharper feature responses at decision boundaries compared to
    the smooth, always-positive RBF basis.
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
        self.metric_net = nn.Sequential(
            nn.Linear(in_dim, metric_hidden),
            nn.GELU(),
            nn.Linear(metric_hidden, in_dim),
        )

        # Initialize MetricNet so initial metric ≈ 1.0 (identity warp)
        softplus_inv_one = math.log(math.e - 1.0)
        nn.init.zeros_(self.metric_net[2].weight)
        nn.init.constant_(self.metric_net[2].bias, softplus_inv_one)

        # Wavelet centers: K evenly spaced in [-3, 3]
        centers = torch.linspace(-3.0, 3.0, n_bases)
        self.register_buffer("centers", centers)

        # Wavelet bandwidth/scale (learned) — controls width of the wavelet
        self.sigma = nn.Parameter(torch.tensor(1.0))

        # Linear mix: [psi_flattened(in_dim * K) | u_original(in_dim)] → out_dim
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
        g = g.clamp(0.05, 16.0)

        # Store for regularization
        self._last_g = g

        # Warp: z = u * sqrt(g) — geometry-adapted coordinates
        z = u_normed * torch.sqrt(g)  # (B, in_dim)

        # Mexican hat wavelet basis expansion:
        # ψ_k(z_i) = (1 - t²) * exp(-t²/2), where t = sigma * (z_i - c_k)
        z_expanded = z.unsqueeze(-1)  # (B, in_dim, 1)
        centers = self.centers.view(1, 1, -1)  # (1, 1, K)
        sigma = F.softplus(self.sigma)  # ensure positive

        t = sigma * (z_expanded - centers)  # (B, in_dim, K)
        t_sq = t.pow(2)
        psi = (1.0 - t_sq) * torch.exp(-t_sq / 2.0)  # (B, in_dim, K)

        psi_flat = psi.reshape(u.shape[0], -1)  # (B, in_dim * K)

        # Concatenate with skip connection
        features = torch.cat([psi_flat, u_normed], dim=-1)

        # Linear mix to output
        out = self.linear_mix(features)
        return out

    def metric_regularization(self) -> torch.Tensor:
        """log(g)^2 regularization to keep metric near identity."""
        if self._last_g is None:
            return torch.tensor(0.0)
        return self._last_g.log().pow(2).mean()


# ── GeoKAN-Wavelet Head (stacked layers) ─────────────────────────────────────

class GeoKANWaveletHead(nn.Module):
    """Two stacked GeoKAN-Wavelet layers + linear output head."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 6,
                 n_bases: int = 12, metric_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.layer1 = GeoKANWaveletLayer(in_dim, hidden_dim, n_bases=n_bases,
                                         metric_hidden=metric_hidden, dropout=dropout)
        self.layer2 = GeoKANWaveletLayer(hidden_dim, hidden_dim, n_bases=n_bases,
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

class GeoKANWaveletRelationGNN(nn.Module):
    """
    GeoKAN-Wavelet relation predictor: Mexican hat wavelet basis.
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

        # ── 4. Dual GeoKAN-Wavelet heads ─────────────────────────────────────
        contact_in_dim = pair_dim + edge_feat_dim  # 64 + 22 = 86
        self.head_contact = GeoKANWaveletHead(
            in_dim=contact_in_dim, hidden_dim=128, out_dim=len(CONTACT_INDICES),
            n_bases=12, metric_hidden=64, dropout=dropout,
        )

        directional_in_dim = pair_dim + 10  # 64 + 10 = 74
        self.head_directional = GeoKANWaveletHead(
            in_dim=directional_in_dim, hidden_dim=128, out_dim=len(DIRECTIONAL_INDICES),
            n_bases=12, metric_hidden=64, dropout=dropout,
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
        """Total metric regularization across all GeoKAN-Wavelet layers."""
        return (self.head_contact.metric_regularization() +
                self.head_directional.metric_regularization())

    def predict(self, x, edge_index, edge_attr=None):
        """Convenience method for inference."""
        with torch.no_grad():
            return self.forward(x, edge_index, edge_attr).argmax(dim=-1)
