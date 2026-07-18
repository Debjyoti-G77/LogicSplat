"""
RelationGNN — Graph Neural Network for spatial relation prediction.

Architecture (v7 dual-head):
  1. Node encoder MLP: raw features → hidden dim
  2. Two rounds of GATConv message passing with residual connections
  3. Two specialized edge classifier heads:
     - HEAD A (Directional): left_of, right_of, in_front_of, behind, higher_than, lower_than
       Uses only first 10 edge features (directional signals)
     - HEAD B (Contact): on_top_of, under, inside, attached_to, hanging_from, adjacent_to
       Uses all 17 edge features + dedicated contact encoder for features [10-16]

Rationale:
  Directional relations (67k samples) dominate gradients in a shared MLP,
  drowning contact relations (4k samples). Separate heads let each group
  learn its own decision boundary without interference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from src.relations.schema import NUM_RELATIONS, Relation


# Relation index groups (from schema.py)
# Contact head outputs (in order): on_top_of, under, inside, attached_to, hanging_from, adjacent_to
CONTACT_INDICES = [
    int(Relation.ON_TOP_OF),    # 0
    int(Relation.UNDER),        # 1
    int(Relation.INSIDE),       # 2
    int(Relation.ATTACHED_TO),  # 3
    int(Relation.HANGING_FROM), # 4
    int(Relation.ADJACENT_TO),  # 5
]

# Directional head outputs (in order): left_of, right_of, in_front_of, behind, higher_than, lower_than
DIRECTIONAL_INDICES = [
    int(Relation.LEFT_OF),      # 6
    int(Relation.RIGHT_OF),     # 7
    int(Relation.IN_FRONT_OF),  # 8
    int(Relation.BEHIND),       # 9
    int(Relation.HIGHER_THAN),  # 10
    int(Relation.LOWER_THAN),   # 11
]


class RelationGNN(nn.Module):

    def __init__(
        self,
        node_feat_dim: int = 10,
        edge_feat_dim: int = 17,
        hidden_dim: int = 256,
        num_relations: int = NUM_RELATIONS,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.edge_feat_dim = edge_feat_dim
        self.num_relations = num_relations

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GATConv: learns attention weights over neighbours
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # ── HEAD A: Directional + comparative (6 relations) ───────────────────
        # left_of, right_of, in_front_of, behind, higher_than, lower_than
        # Only needs features [0-9] (directional signals: delta_x, delta_y, delta_z, etc.)
        dir_input_dim = 2 * hidden_dim + 10  # node_pair(512) + first 10 edge feats
        self.head_directional = nn.Sequential(
            nn.Linear(dir_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 6),
        )

        # ── HEAD B: Contact + physical (6 relations) ─────────────────────────
        # on_top_of, under, inside, attached_to, hanging_from, adjacent_to
        # Uses ALL edge features + a dedicated contact encoder for features [10:]
        n_contact_feats = max(edge_feat_dim - 10, 4)  # features [10:] (contact-specific)
        self.contact_encoder = nn.Sequential(
            nn.Linear(n_contact_feats, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        # node_pair(512) + all edge feats + contact_encoded(64)
        contact_input_dim = 2 * hidden_dim + edge_feat_dim + 64
        self.head_contact = nn.Sequential(
            nn.Linear(contact_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 6),
        )

    def forward(self, x, edge_index, edge_attr=None):
        h = self.node_encoder(x)

        # GATConv with residual connections
        h = self.norm1(F.relu(self.conv1(h, edge_index)) + h)
        h = self.dropout(h)
        h = self.norm2(F.relu(self.conv2(h, edge_index)) + h)
        h = self.dropout(h)

        src, dst = edge_index[0], edge_index[1]
        node_pair = torch.cat([h[src], h[dst]], dim=-1)  # (E, 2*hidden_dim)

        # ── Directional head: only first 10 edge features ────────────────────
        dir_feats = edge_attr[:, :10] if edge_attr is not None else torch.zeros(
            node_pair.shape[0], 10, device=node_pair.device
        )
        dir_input = torch.cat([node_pair, dir_feats], dim=-1)
        out_dir = self.head_directional(dir_input)  # (E, 6)

        # ── Contact head: all edge features + dedicated contact encoding ─────
        if edge_attr is not None:
            contact_feats = edge_attr[:, 10:]  # features [10:] (contact-specific)
            contact_encoded = self.contact_encoder(contact_feats)  # (E, 64)
            contact_input = torch.cat([node_pair, edge_attr, contact_encoded], dim=-1)
        else:
            dummy_edge = torch.zeros(node_pair.shape[0], self.edge_feat_dim, device=node_pair.device)
            dummy_contact = torch.zeros(node_pair.shape[0], 64, device=node_pair.device)
            contact_input = torch.cat([node_pair, dummy_edge, dummy_contact], dim=-1)
        out_contact = self.head_contact(contact_input)  # (E, 6)

        # ── Assemble full 12-dim output in schema order ──────────────────────
        # Schema: [ON_TOP_OF=0, UNDER=1, INSIDE=2, ATTACHED_TO=3, HANGING_FROM=4,
        #          ADJACENT_TO=5, LEFT_OF=6, RIGHT_OF=7, IN_FRONT_OF=8, BEHIND=9,
        #          HIGHER_THAN=10, LOWER_THAN=11]
        # Contact head outputs indices 0-5, Directional head outputs indices 6-11
        out = torch.cat([out_contact, out_dir], dim=-1)  # (E, 12)
        return out

    def predict(self, x, edge_index, edge_attr=None):
        with torch.no_grad():
            return self.forward(x, edge_index, edge_attr).argmax(dim=-1)
