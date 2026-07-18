"""
Train GeoKAN-Gamma from scratch on 3RScan + synthetic tabletop data jointly.

WHY joint training beats fine-tuning:
  Fine-tuning: KAN basis functions already fixed on room-scale geometry.
  Contact features for tabletop fall outside the learned basis range.
  Result: on_top_of/under score very low → never in top-3.

  Joint training: basis functions are optimised to cover BOTH scales.
  The model learns a unified geometric representation.
  No catastrophic forgetting since both domains are present from epoch 0.

Training data (no real tabletop GT ever used):
  - 3RScan : D:/logicsplat_data/3rscan_graph_cache  (~1500 scenes)
  - Tabletop synthetic: D:/logicsplat_data/synthetic_v2 (800 Z-UP scenes)

Test data (held out completely):
  - scenes 06-13 evaluated via eval_geokan_tabletop.py

Output:
  models/geokan_crossdomain.pt
"""
import sys
sys.path.insert(0, ".")

import os, json, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from geokan_gamma_relation import GeoKANGammaRelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, Relation

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    "save_path":        "models/geokan_crossdomain.pt",
    "thresh_save_path": "models/geokan_crossdomain_thresholds.json",

    "rscan_dir":        "D:/logicsplat_data/3rscan_graph_cache",
    "synth_dir":        "D:/logicsplat_data/synthetic_v2",

    # Synthetic 80/20 train/val split; all 3RScan goes to train
    "n_synth_train":    640,
    "n_synth_val":      160,
    "n_rscan_train":    1200,   # cap to keep epoch length manageable

    # Training
    "lr":               2e-4,
    "lr_min":           1e-5,
    "weight_decay":     1e-4,
    "epochs":           150,
    "batch_size":       8,
    "patience":         30,
    "device":           "cuda" if torch.cuda.is_available() else "cpu",

    # Relation weights: boost contact (tabletop gap) without hurting directional
    "ontop_weight":     4.0,
    "under_weight":     4.0,
    "higher_weight":    2.0,
    "lower_weight":     2.0,

    # ASL (same proven settings)
    "focal_gamma_neg":  4.0,
    "focal_gamma_pos":  0.0,
    "asl_prob_margin":  0.0,
    "pos_weight_cap":   8.0,

    # Model dims
    "node_feat_dim":    10,
    "edge_feat_dim":    22,
    "hidden_dim":       128,
}

# ── Loss ──────────────────────────────────────────────────────────────────────

class ASLLoss(nn.Module):
    def __init__(self, pos_weight, gamma_neg=4.0, gamma_pos=0.0,
                 prob_margin=0.0, rel_weights=None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma_neg   = gamma_neg
        self.gamma_pos   = gamma_pos
        self.prob_margin = prob_margin
        if rel_weights is not None:
            self.register_buffer("rel_w", rel_weights)
        else:
            self.rel_w = None

    def forward(self, logits, labels):
        mask      = (labels >= 0).float()
        probs     = torch.sigmoid(logits)
        loss_pos  = F.binary_cross_entropy_with_logits(
            logits, torch.ones_like(logits), pos_weight=self.pos_weight, reduction="none"
        )
        focal_pos = (1.0 - probs) ** self.gamma_pos
        xs_neg    = torch.clamp(probs - self.prob_margin, min=0.0)
        loss_neg  = -torch.log(torch.clamp(1.0 - xs_neg, min=1e-8))
        focal_neg = xs_neg ** self.gamma_neg
        labels_c  = labels.clamp(min=0.0)
        loss      = torch.where(labels_c >= 0.5,
                                loss_pos * focal_pos, loss_neg * focal_neg)
        masked    = loss * mask
        if self.rel_w is not None:
            masked = masked * self.rel_w.unsqueeze(0)
        n_valid = mask.sum()
        return masked.sum() / n_valid if n_valid > 0 else masked.sum()

# ── Data ──────────────────────────────────────────────────────────────────────

def load_synth(synth_dir, n_train, n_val, seed=42):
    files = sorted(f for f in os.listdir(synth_dir) if f.endswith(".pt"))
    random.Random(seed).shuffle(files)
    files = files[:n_train + n_val]
    train_list, val_list = [], []
    for i, fname in enumerate(files):
        g = torch.load(os.path.join(synth_dir, fname), weights_only=False)
        if g["edge_attr"].shape[1] != 22 or g["edge_label"].shape[1] != NUM_RELATIONS:
            continue
        y = g["edge_label"].clone()
        y[:, int(Relation.ATTACHED_TO)] = -1.0   # never in real tabletop
        d = Data(x=g["x"], edge_index=g["edge_index"], edge_attr=g["edge_attr"], y=y)
        if i < n_train:
            train_list.append(d)
        else:
            val_list.append(d)
    print(f"  Synthetic : {len(train_list)} train  {len(val_list)} val")
    return train_list, val_list


def load_rscan(rscan_dir, n_max, seed=0):
    from src.relations.schema import LEGACY_12_TO_10_COLS
    files = sorted(f for f in os.listdir(rscan_dir) if f.endswith(".pt"))
    random.Random(seed).shuffle(files)
    files = files[:n_max]
    data_list = []
    for fname in files:
        g = torch.load(os.path.join(rscan_dir, fname), weights_only=False)
        if g["edge_attr"].shape[1] != 22:
            continue
        if g["edge_label"].shape[1] == 12:
            g["edge_label"] = g["edge_label"][:, LEGACY_12_TO_10_COLS]
        if g["edge_label"].shape[1] != NUM_RELATIONS:
            continue
        data_list.append(Data(x=g["x"], edge_index=g["edge_index"],
                              edge_attr=g["edge_attr"], y=g["edge_label"]))
    print(f"  3RScan    : {len(data_list)} train")
    return data_list


def compute_pos_weight(data_list, cap=8.0):
    pos   = torch.zeros(NUM_RELATIONS)
    valid = torch.zeros(NUM_RELATIONS)
    for d in data_list:
        for r in range(NUM_RELATIONS):
            col = d.y[:, r]; vm = col >= 0
            pos[r]   += (col[vm] >= 0.5).float().sum()
            valid[r] += vm.float().sum()
    return torch.clamp((valid - pos).clamp(min=1) / pos.clamp(min=1), max=cap)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--lr",     type=float, default=CONFIG["lr"])
    args = parser.parse_args()
    cfg  = {**CONFIG, "epochs": args.epochs, "lr": args.lr}
    dev  = cfg["device"]

    print("=" * 65)
    print("  GeoKAN-Gamma Cross-Domain — Joint Training from Scratch")
    print("=" * 65)
    print(f"  3RScan: {cfg['rscan_dir']}")
    print(f"  Synth : {cfg['synth_dir']}")
    print(f"  LR={cfg['lr']}  Epochs={cfg['epochs']}")
    print()

    print("Loading data...")
    synth_train, synth_val = load_synth(
        cfg["synth_dir"], cfg["n_synth_train"], cfg["n_synth_val"])
    rscan_train = load_rscan(cfg["rscan_dir"], cfg["n_rscan_train"])

    train_data = synth_train + rscan_train
    val_data   = synth_val
    print(f"  Total train: {len(train_data)}  Val: {len(val_data)}")

    pw = compute_pos_weight(train_data, cap=cfg["pos_weight_cap"]).to(dev)
    print(f"  Pos weights: {[round(x,2) for x in pw.tolist()]}")

    rel_w = torch.ones(NUM_RELATIONS, device=dev)
    rel_w[int(Relation.ON_TOP_OF)]   = cfg["ontop_weight"]
    rel_w[int(Relation.UNDER)]       = cfg["under_weight"]
    rel_w[int(Relation.HIGHER_THAN)] = cfg["higher_weight"]
    rel_w[int(Relation.LOWER_THAN)]  = cfg["lower_weight"]

    # Fresh model — no pretrained weights
    model = GeoKANGammaRelationGNN(
        node_feat_dim=cfg["node_feat_dim"],
        edge_feat_dim=cfg["edge_feat_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_relations=NUM_RELATIONS,
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_params:,} params  (training from scratch)")

    criterion = ASLLoss(pw, cfg["focal_gamma_neg"], cfg["focal_gamma_pos"],
                        cfg["asl_prob_margin"], rel_w)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr_min"])

    train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=cfg["batch_size"], shuffle=False)

    best_f1, patience_cnt = -1.0, 0
    print(f"\n{'Epoch':>5}  {'Loss':>8}  {'Val F1':>8}  {'best':>4}")
    print("-" * 35)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(dev)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss   = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validate on synthetic tabletop val
        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(dev)
                all_p.append(torch.sigmoid(model(batch.x, batch.edge_index, batch.edge_attr)).cpu())
                all_l.append(batch.y.cpu())
        probs  = torch.cat(all_p).numpy()
        labels = torch.cat(all_l).numpy()
        mask   = labels >= 0
        val_f1 = f1_score((labels[mask] >= 0.5).astype(int),
                          (probs[mask]  >= 0.5).astype(int),
                          average="macro", zero_division=0)

        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
            torch.save(model.state_dict(), cfg["save_path"])
            patience_cnt = 0
        else:
            patience_cnt += 1

        avg_loss = total_loss / max(len(train_loader), 1)
        if epoch % 5 == 0 or is_best:
            print(f"{epoch:>5}  {avg_loss:>8.4f}  {val_f1:>8.4f}  {'*' if is_best else ''}")

        if patience_cnt >= cfg["patience"]:
            print(f"  Early stop at epoch {epoch}")
            break

    print(f"\nBest val F1: {best_f1:.4f}  ->  {cfg['save_path']}")

    # Threshold tuning on synth val
    model.load_state_dict(torch.load(cfg["save_path"], weights_only=False, map_location=dev))
    model.eval()
    all_s, all_g = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(dev)
            all_s.append(torch.sigmoid(model(batch.x, batch.edge_index, batch.edge_attr)).cpu().numpy())
            all_g.append(batch.y.cpu().numpy())
    scores = np.concatenate(all_s); gt = np.concatenate(all_g)

    thresholds = {}
    for r in range(NUM_RELATIONS):
        col_gt = gt[:, r]; col_s = scores[:, r]; valid = col_gt >= 0
        if valid.sum() < 10 or (col_gt[valid] > 0.5).sum() == 0:
            thresholds[r] = 0.5; continue
        best_t, best_f1r = 0.5, -1.0
        for t in [i / 10 for i in range(1, 10)]:
            pred = (col_s[valid] >= t).astype(int)
            gtv  = (col_gt[valid] >= 0.5).astype(int)
            tp = (pred & gtv).sum(); fp = (pred & ~gtv).sum(); fn = (~pred & gtv).sum()
            p  = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            f1r = 2 * p * rec / max(p + rec, 1e-9)
            if f1r > best_f1r: best_f1r, best_t = f1r, t
        thresholds[r] = best_t

    print("\nTuned thresholds:")
    for r, t in thresholds.items():
        print(f"  {RELATION_NAMES[r]:20s}: {t:.2f}")
    with open(cfg["thresh_save_path"], "w") as f:
        json.dump({str(k): v for k, v in thresholds.items()}, f, indent=2)
    print(f"Saved: {cfg['thresh_save_path']}")


if __name__ == "__main__":
    main()
