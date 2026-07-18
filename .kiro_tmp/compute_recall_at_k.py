"""
Compute Predicate Recall@K on the validation set using the GeoKAN model.
This gives us the metric that's comparable to ReLaGS's reported numbers.
"""
import sys
sys.path.insert(0, ".")

import os
import json
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from geokan_relation import GeoKANRelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES

# Load graphs
cache_dir = "D:/logicsplat_data/3rscan_graph_cache"
files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
print(f"Loading {len(files)} graphs...")

graphs = []
for fname in files:
    g = torch.load(os.path.join(cache_dir, fname), weights_only=False)
    graphs.append(g)

# Same split as training (85% train, 15% val)
rng = np.random.default_rng(42)
indices = np.arange(len(graphs))
rng.shuffle(indices)
train_size = int(len(graphs) * 0.85)
val_idx = indices[train_size:]
val_graphs = [graphs[i] for i in val_idx]
print(f"Val set: {len(val_graphs)} scenes")

# Build PyG dataset
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

val_ds = PyGGraphDataset(val_graphs)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GeoKANRelationGNN(
    node_feat_dim=10, edge_feat_dim=22, hidden_dim=128,
    num_relations=NUM_RELATIONS, dropout=0.2
).to(device)
model.load_state_dict(torch.load("models/geokan_relation_v4.pt", map_location=device, weights_only=True))
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# Compute Recall@K
all_probs = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        probs = torch.sigmoid(logits).cpu()
        all_probs.append(probs)
        all_labels.append(batch.y.cpu())

all_probs = torch.cat(all_probs, dim=0).numpy()
all_labels = torch.cat(all_labels, dim=0).numpy()

print(f"\nTotal edges: {all_probs.shape[0]}")
print(f"Total positive labels: {(all_labels > 0).sum()}")

# For each positive GT label, check if it's in the top-K predictions for that edge
def compute_recall_at_k(probs, labels, k):
    """
    For each edge that has at least one positive GT relation,
    for each positive GT relation on that edge,
    check if that relation is in the top-K predicted relations.
    """
    hits = 0
    total = 0
    
    for edge_idx in range(probs.shape[0]):
        edge_labels = labels[edge_idx]  # (12,)
        edge_probs = probs[edge_idx]    # (12,)
        
        # Find positive GT relations for this edge (excluding -1 masked labels)
        positive_rels = np.where(edge_labels == 1.0)[0]
        
        if len(positive_rels) == 0:
            continue
        
        # Get top-K predicted relations by score
        top_k_indices = np.argsort(edge_probs)[::-1][:k]
        
        # Check each GT relation
        for rel_idx in positive_rels:
            total += 1
            if rel_idx in top_k_indices:
                hits += 1
    
    return hits / max(total, 1), hits, total

# Compute for different K values
print(f"\n{'='*60}")
print("PREDICATE RECALL@K (Validation Set)")
print(f"{'='*60}")

for k in [1, 3, 5, 10]:
    recall, hits, total = compute_recall_at_k(all_probs, all_labels, k)
    print(f"  Recall@{k:2d}: {recall:.4f} ({hits}/{total} GT triples found in top-{k})")

# Per-relation breakdown for R@5
print(f"\n{'='*60}")
print("PER-RELATION RECALL@5")
print(f"{'='*60}")

for rel_idx in range(NUM_RELATIONS):
    hits = 0
    total = 0
    for edge_idx in range(all_probs.shape[0]):
        if all_labels[edge_idx, rel_idx] == 1.0:
            total += 1
            top_5 = np.argsort(all_probs[edge_idx])[::-1][:5]
            if rel_idx in top_5:
                hits += 1
    recall = hits / max(total, 1)
    print(f"  {RELATION_NAMES[rel_idx]:20s}  R@5={recall:.3f}  ({hits}/{total})")

# Comparison table
print(f"\n{'='*60}")
print("COMPARISON TABLE (ReLaGS format)")
print(f"{'='*60}")
r3, _, _ = compute_recall_at_k(all_probs, all_labels, 3)
r5, _, _ = compute_recall_at_k(all_probs, all_labels, 5)
print(f"  {'Method':<30} {'Pred R@3':>10} {'Pred R@5':>10}")
print(f"  {'-'*30} {'-'*10} {'-'*10}")
print(f"  {'LogicSplat GeoKAN (ours)':<30} {r3*100:>9.1f}% {r5*100:>9.1f}%")
print(f"  {'ReLaGS (reported)':<30} {'79.0%':>10} {'87.0%':>10}")
print(f"  {'ConceptGraphs (reported)':<30} {'74.0%':>10} {'79.0%':>10}")
print(f"  {'Open3DSG (reported)':<30} {'58.0%':>10} {'65.0%':>10}")
