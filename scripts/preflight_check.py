"""Pre-flight check before training v2."""
import sys
sys.path.insert(0, ".")
import warnings
warnings.filterwarnings("ignore")
import os
import torch

from src.models.relation_gnn import RelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES
from src.training.train_scannet import CONFIG, compute_class_weights

print("=" * 55)
print("PRE-FLIGHT CHECK — v2 training")
print("=" * 55)

# ── 1. Excluded relation indices ──────────────────────────────
exclude_indices = [
    idx for idx, name in RELATION_NAMES.items()
    if name in CONFIG["exclude_relations"]
]
print(f"\n[1] Excluded relations ({len(exclude_indices)}):")
for i in exclude_indices:
    print(f"    idx={i}  name={RELATION_NAMES[i]}")
active = [i for i in range(NUM_RELATIONS) if i not in exclude_indices]
print(f"    Active relations: {len(active)}/12")

# ── 2. Model architecture ─────────────────────────────────────
model = RelationGNN(
    node_feat_dim=CONFIG["node_feat_dim"],
    edge_feat_dim=CONFIG["edge_feat_dim"],
    hidden_dim=CONFIG["hidden_dim"],
    num_relations=NUM_RELATIONS,
    dropout=CONFIG["dropout"],
)
params = sum(p.numel() for p in model.parameters())
print(f"\n[2] Model: hidden_dim={CONFIG['hidden_dim']}  params={params:,}")
# v1 was 178,508 — v2 with hidden=256 should be ~700k
# With 7340 augmented scenes and weight_decay=1e-4 this is acceptable

# ── 3. Class weights from 20 cached graphs ────────────────────
cache_dir = "D:/logicsplat_data/scannet_cache"
files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))[:20]
graphs = [torch.load(os.path.join(cache_dir, f), weights_only=False) for f in files]

class FakeSubset:
    def __init__(self, gs):
        self.gs = gs
    def __len__(self):
        return len(self.gs)
    def __getitem__(self, i):
        return self.gs[i]

weights = compute_class_weights(FakeSubset(graphs), exclude_indices=exclude_indices)
print(f"\n[3] Class weights (neg/pos ratio, capped at 50):")
for i, w in enumerate(weights):
    tag = "  [EXCLUDED -> 0]" if i in exclude_indices else ""
    status = "OK" if w.item() > 0 or i in exclude_indices else "WARN: zero weight"
    print(f"    {RELATION_NAMES[i]:20s}  {w:.2f}  {status}{tag}")

# ── 4. Tensor shapes ──────────────────────────────────────────
g = graphs[0]
el_shape = tuple(g["edge_label"].shape)
ea_shape = tuple(g["edge_attr"].shape)
x_shape  = tuple(g["x"].shape)
print(f"\n[4] Tensor shapes (first cached graph):")
print(f"    x:          {x_shape}   expected (N, 10)")
print(f"    edge_attr:  {ea_shape}  expected (E, 10)")
print(f"    edge_label: {el_shape}  expected (E, {NUM_RELATIONS})")

x_ok  = x_shape[1] == 10
ea_ok = ea_shape[1] == 10
el_ok = len(el_shape) == 2 and el_shape[1] == NUM_RELATIONS
print(f"    x OK={x_ok}  edge_attr OK={ea_ok}  edge_label OK={el_ok}")

# ── 5. Config summary ─────────────────────────────────────────
print(f"\n[5] Training config:")
print(f"    epochs={CONFIG['epochs']}  batch={CONFIG['batch_size']}")
print(f"    lr={CONFIG['lr']}  weight_decay={CONFIG['weight_decay']}")
print(f"    dropout={CONFIG['dropout']}  augment_factor={CONFIG['augment_factor']}")
print(f"    patience=10 x 5 = 50 epochs without improvement -> early stop")

# ── 6. Cache version check ────────────────────────────────────
sample_name = files[0] if files else ""
expected_ver = "v3_multilabel"
ver_ok = expected_ver in sample_name
print(f"\n[6] Cache version: '{sample_name}' -> contains '{expected_ver}': {ver_ok}")
print(f"    Total cached files: {len(os.listdir(cache_dir))}")

# ── Summary ───────────────────────────────────────────────────
all_ok = x_ok and ea_ok and el_ok and ver_ok and len(exclude_indices) == 2
print(f"\n{'='*55}")
print(f"RESULT: {'READY TO TRAIN' if all_ok else 'ISSUES FOUND — fix before training'}")
print(f"{'='*55}")
