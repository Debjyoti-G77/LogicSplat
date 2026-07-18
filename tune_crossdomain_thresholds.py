"""Threshold tuning for geokan_crossdomain.pt on synthetic val set."""
import sys
sys.path.insert(0, ".")

import os, json, random
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from geokan_gamma_relation import GeoKANGammaRelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, Relation

SAVE_PATH   = "models/geokan_crossdomain.pt"
THRESH_PATH = "models/geokan_crossdomain_thresholds.json"
SYNTH_DIR   = "D:/logicsplat_data/synthetic_v2"
N_VAL       = 160
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

def load_synth_val(synth_dir, n_val=160, seed=42):
    files = sorted(f for f in os.listdir(synth_dir) if f.endswith(".pt"))
    random.Random(seed).shuffle(files)
    # same split as training: first 640 = train, next 160 = val
    val_files = files[640:640+n_val]
    val_list = []
    for fname in val_files:
        g = torch.load(os.path.join(synth_dir, fname), weights_only=False)
        if g["edge_attr"].shape[1] != 22 or g["edge_label"].shape[1] != NUM_RELATIONS:
            continue
        y = g["edge_label"].clone()
        y[:, int(Relation.ATTACHED_TO)] = -1.0
        val_list.append(Data(x=g["x"], edge_index=g["edge_index"],
                             edge_attr=g["edge_attr"], y=y))
    print(f"Val graphs: {len(val_list)}")
    return val_list

state = torch.load(SAVE_PATH, weights_only=False, map_location=DEVICE)
hidden = state["node_encoder.0.weight"].shape[0]
ef_dim = state["conv1.lin_edge.weight"].shape[1]
model = GeoKANGammaRelationGNN(
    node_feat_dim=10, edge_feat_dim=ef_dim,
    hidden_dim=hidden, num_relations=NUM_RELATIONS,
).to(DEVICE)
model.load_state_dict(state)
model.eval()
print(f"Loaded {SAVE_PATH}")

val_data = load_synth_val(SYNTH_DIR)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

all_s, all_g = [], []
with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(DEVICE)
        all_s.append(torch.sigmoid(model(batch.x, batch.edge_index, batch.edge_attr)).cpu().numpy())
        all_g.append(batch.y.cpu().numpy())
scores = np.concatenate(all_s)
gt     = np.concatenate(all_g)

thresholds = {}
for r in range(NUM_RELATIONS):
    col_gt = gt[:, r]; col_s = scores[:, r]; valid = col_gt >= 0
    if valid.sum() < 10 or (col_gt[valid] > 0.5).sum() == 0:
        thresholds[r] = 0.5; continue
    best_t, best_f1r = 0.5, -1.0
    for t in [i / 20 for i in range(1, 20)]:
        pred = (col_s[valid] >= t).astype(int)
        gtv  = (col_gt[valid] >= 0.5).astype(int)
        tp = int((pred & gtv).sum()); fp = int((pred & ~gtv).sum()); fn = int((~pred & gtv).sum())
        p  = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1r = 2 * p * rec / max(p + rec, 1e-9)
        if f1r > best_f1r: best_f1r, best_t = f1r, t
    thresholds[r] = best_t

print("\nTuned thresholds:")
for r, t in thresholds.items():
    print(f"  {RELATION_NAMES[r]:20s}: {t:.2f}")
with open(THRESH_PATH, "w") as f:
    json.dump({str(k): v for k, v in thresholds.items()}, f, indent=2)
print(f"\nSaved: {THRESH_PATH}")
