"""
RelationGNN — Graph Neural Network for spatial relation prediction.

Architecture:
  1. Node encoder MLP: raw features → hidden dim
  2. Two rounds of SAGEConv message passing
  3. Edge classifier MLP: [node_A | node_B | edge_features] → relation class
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from src.relations.schema import NUM_RELATIONS


class RelationGNN(nn.Module):

    def __init__(
        self,
        node_feat_dim: int = 8,
        edge_feat_dim: int = 4,
        hidden_dim: int = 128,
        num_relations: int = NUM_RELATIONS,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GATConv: learns attention weights over neighbours
        # heads=4 means 4 parallel attention mechanisms, concat=False averages them
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # [node_A | node_B | edge_features]
        edge_input_dim = 2 * hidden_dim + edge_feat_dim
        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_relations),
        )

    def forward(self, x, edge_index, edge_attr=None):
        h = self.node_encoder(x)

        h = self.conv1(h, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = F.relu(h)
        h = self.dropout(h)

        src, dst = edge_index[0], edge_index[1]
        parts = [h[src], h[dst]]
        if edge_attr is not None:
            parts.append(edge_attr)
        edge_features = torch.cat(parts, dim=-1)
        return self.edge_classifier(edge_features)

    def predict(self, x, edge_index, edge_attr=None):
        with torch.no_grad():
            return self.forward(x, edge_index, edge_attr).argmax(dim=-1)
