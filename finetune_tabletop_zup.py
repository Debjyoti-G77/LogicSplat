"""
Fine-tune GeoKAN on Z-UP tabletop-scale synthetic data.

After Z-convention normalization (all 8 scenes are now Z-UP), the remaining
domain gap is SCALE: 3RScan contact features are calibrated for room-scale
geometry (3-8m), while tabletop scenes are 0.3-0.8m.

avg_diag normalization makes DIRECTIONAL features scale-invariant, but
CONTACT features (contact_score, vert_gap, ontop margins) still differ
because they encode absolute physics at different scales.

FIX: Fine-tune the FULL GeoKAN model on Z-UP tabletop-scale synthetic data
(D:/logicsplat_data/synthetic_v2, 800 scenes) mixed with 3RScan data.
EWC prevents catastrophic forgetting of 3RScan knowledge.

Output:
    models/geokan_relation_tabletop_znorm.pt
    models/geokan_relation_tabletop_znorm_thresholds.json

Usage:
    python finetune_tabletop_zup.py
    python finetune_tabletop_zup.py --epochs 80 --lr 1e-4
"""
import sys
sys.path.insert(0, ".")

import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from geokan_gamma_relation import GeoKANGammaRelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, Relation

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kw): return it


# ── Config ─────────────────────────────────────────────────────────────────────

CONFIG = {
    "pretrained_path":  "models/geokan_relation_gamma.pt",
    "save_path":        "models/geokan_relation_tabletop_gamma_znorm.pt",
    "thresh_save_path": "models/geokan_relation_tabletop_znorm_thresholds.json",
    # Z-UP synthetic tabletop data (normalized scenes)
    "synth_dir":        "D:/logicsplat_data/synthetic_v2",
    "rscan_cache_dir":  "D:/logicsplat_data/3rscan_graph_cache",
    # Synthetic: 640 train / 160 val (80/20)
    "n_synth_total":    800,
    "n_synth_train":    640,
    "n_synth_val":      160,
    # 3RScan mix to prevent forgetting
    "n_rscan_mix":      300,
    # Training
    "lr":               1e-4,
    "weight_decay":     1e-4,
    "epochs":           100,
    "batch_size":       8,
    "patience":         25,
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
    # EWC: stronger — preserve 3RScan directional knowledge
    "ewc_lambda":       1.0,
    # Relation weights: mild boost contact, do NOT distort directional
    "ontop_weight":     3.0,
    "under_weight":     3.0,
    "higher_weight":    1.5,
    "lower_weight":     1.5,
    # ASL
    "focal_gamma_neg":  4.0,
    "focal_gamma_pos":  0.0,
    "asl_prob_margin":  0.0,
    "pos_weight_cap":   8.0,
    # Model dims (must match pretrained)
    "node_feat_dim":    10,
    "edge_feat_dim":    22,
    "hidden_dim":       128,
}


# ── Loss ───────────────────────────────────────────────────────────────────────

class ASLLoss(nn.Module):
    def __init__(self, pos_weight, gamma_neg=4.0, gamma_pos=0.0,
                 prob_margin=0.0, relation_weights=None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma_neg  = gamma_neg
        self.gamma_pos  = gamma_pos
        self.prob_margin = prob_margin
        if relation_weights is not None:
            self.register_buffer("rel_w", relation_weights)
        else:
            self.rel_w = None

    def forward(self, logits, labels):
        mask     = (labels >= 0).float()
        probs    = torch.sigmoid(logits)

        loss_pos = F.binary_cross_entropy_with_logits(
            logits, torch.ones_like(logits), pos_weight=self.pos_weight, reduction="none"
        )
        focal_pos = (1.0 - probs) ** self.gamma_pos

        xs_neg   = torch.clamp(probs - self.prob_margin, min=0.0)
        loss_neg = -torch.log(torch.clamp(1.0 - xs_neg, min=1e-8))
        focal_neg = xs_neg ** self.gamma_neg

        labels_c = labels.clamp(min=0.0)
        loss     = torch.where(labels_c >= 0.5, loss_pos * focal_pos, loss_neg * focal_neg)
        masked   = loss * mask
        if self.rel_w is not None:
            masked = masked * self.rel_w.unsqueeze(0)

        n_valid = mask.sum()
        return masked.sum() / n_valid if n_valid > 0 else masked.sum()


# ── Data ───────────────────────────────────────────────────────────────────────

def load_synth_data(synth_dir, n_train, n_val, seed=42):
    """Load Z-UP synthetic tabletop scenes."""
    files = sorted(f for f in os.listdir(synth_dir) if f.endswith(".pt"))
    rng = random.Random(seed); rng.shuffle(files)
    files = files[:n_train + n_val]

    data_list = []
    for fname in files:
        g = torch.load(os.path.join(synth_dir, fname), weights_only=False)
        if g["edge_attr"].shape[1] != 22:
            continue
        if g["edge_label"].shape[1] != NUM_RELATIONS:
            continue
        y = g["edge_label"].clone()
        # Suppress attached_to from synthetic (never in tabletop GT)
        y[:, int(Relation.ATTACHED_TO)] = -1.0
        data_list.append(Data(
            x=g["x"], edge_index=g["edge_index"],
            edge_attr=g["edge_attr"], y=y,
        ))

    print(f"  Synthetic: loaded {len(data_list)} scenes")
    return data_list[:n_train], data_list[n_train:]


def load_rscan_subset(cache_dir, n_scenes, seed=999):
    """Load 3RScan scenes to prevent catastrophic forgetting."""
    from src.relations.schema import LEGACY_12_TO_10_COLS
    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
    rng = random.Random(seed); rng.shuffle(files)
    files = files[:n_scenes]

    data_list = []
    for fname in files:
        g = torch.load(os.path.join(cache_dir, fname), weights_only=False)
        if g["edge_attr"].shape[1] != 22:
            continue
        if g["edge_label"].shape[1] == 12:
            g["edge_label"] = g["edge_label"][:, LEGACY_12_TO_10_COLS]
        if g["edge_label"].shape[1] != NUM_RELATIONS:
            continue
        data_list.append(Data(
            x=g["x"], edge_index=g["edge_index"],
            edge_attr=g["edge_attr"], y=g["edge_label"],
        ))
    print(f"  3RScan: loaded {len(data_list)} scenes")
    return data_list


def compute_pos_weight(data_list, cap=8.0):
    pos   = torch.zeros(NUM_RELATIONS)
    valid = torch.zeros(NUM_RELATIONS)
    for d in data_list:
        for r in range(NUM_RELATIONS):
            col = d.y[:, r]; vm = col >= 0
            pos[r]   += (col[vm] >= 0.5).float().sum()
            valid[r] += vm.float().sum()
    return torch.clamp(torch.clamp(valid - pos, min=1.0) / torch.clamp(pos, min=1.0), max=cap)


# ── EWC ────────────────────────────────────────────────────────────────────────

def compute_fisher(model, data_list, device, n_samples=200, batch_size=8):
    """Estimate Fisher information (diagonal) for EWC."""
    model.train()
    loader = DataLoader(data_list[:n_samples], batch_size=batch_size, shuffle=True)
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        model.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        labels = getattr(batch, 'y', batch.edge_label) if hasattr(batch, 'edge_label') else batch.y
        loss   = F.binary_cross_entropy_with_logits(logits, labels.float().clamp(min=0.0))
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        n_batches += 1

    for n in fisher:
        fisher[n] /= max(n_batches, 1)
    return fisher


def ewc_loss(model, fisher, old_params, lam):
    """EWC penalty: sum of Fisher-weighted squared parameter deviations."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for n, p in model.named_parameters():
        if n in fisher and n in old_params:
            loss += (fisher[n] * (p - old_params[n]).pow(2)).sum()
    return lam * loss


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--lr",     type=float, default=CONFIG["lr"])
    parser.add_argument("--ewc-lambda", type=float, default=CONFIG["ewc_lambda"])
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg["epochs"]     = args.epochs
    cfg["lr"]         = args.lr
    cfg["ewc_lambda"] = args.ewc_lambda
    device = cfg["device"]

    print("=" * 65)
    print("  GeoKAN Tabletop Fine-tuning — Z-UP Synthetic Data")
    print("=" * 65)
    print(f"  Base model:  {cfg['pretrained_path']}")
    print(f"  Synth data:  {cfg['synth_dir']}  (Z-UP, all scenes normalized)")
    print(f"  LR={cfg['lr']}  EWC_lambda={cfg['ewc_lambda']}  epochs={cfg['epochs']}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    synth_train, synth_val = load_synth_data(
        cfg["synth_dir"], cfg["n_synth_train"], cfg["n_synth_val"]
    )
    rscan_data = load_rscan_subset(cfg["rscan_cache_dir"], cfg["n_rscan_mix"])

    # Combined training: synthetic + 3RScan mix
    train_data = synth_train + rscan_data
    val_data   = synth_val
    print(f"  Train: {len(train_data)} graphs  Val: {len(val_data)} graphs")

    # ── Compute pos weights from synth + rscan ─────────────────────────────────
    pw = compute_pos_weight(train_data, cap=cfg["pos_weight_cap"]).to(device)
    print(f"  Pos weights: {pw.tolist()}")

    # Relation loss weights
    rel_w = torch.ones(NUM_RELATIONS, device=device)
    rel_w[int(Relation.ON_TOP_OF)]   = cfg["ontop_weight"]
    rel_w[int(Relation.UNDER)]       = cfg["under_weight"]
    rel_w[int(Relation.HIGHER_THAN)] = cfg["higher_weight"]
    rel_w[int(Relation.LOWER_THAN)]  = cfg["lower_weight"]

    # ── Load model ────────────────────────────────────────────────────────────
    model = GeoKANGammaRelationGNN(
        node_feat_dim=cfg["node_feat_dim"],
        edge_feat_dim=cfg["edge_feat_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_relations=NUM_RELATIONS,
    ).to(device)
    state = torch.load(cfg["pretrained_path"], weights_only=False, map_location=device)
    # Remap old checkpoint key names → current class parameter names
    def _remap(k):
        if k.endswith(".gamma_param"):
            return k[:-len(".gamma_param")] + ".gamma_params"
        if k.endswith(".gamma_rbf"):
            return k[:-len(".gamma_rbf")] + ".rbf_gamma"
        return k
    state = {_remap(k): v for k, v in state.items()}
    model.load_state_dict(state)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model loaded: {n_params:,} trainable params")

    # ── EWC: compute Fisher on 3RScan data, freeze old params ─────────────────
    print("  Computing EWC Fisher matrix on 3RScan data...")
    fisher    = compute_fisher(model, rscan_data, device, n_samples=200)
    old_params = {n: p.detach().clone() for n, p in model.named_parameters()
                  if p.requires_grad}
    print(f"  Fisher computed on {len(rscan_data)} 3RScan scenes")

    # ── Loss and optimizer ─────────────────────────────────────────────────────
    criterion = ASLLoss(
        pos_weight=pw,
        gamma_neg=cfg["focal_gamma_neg"],
        gamma_pos=cfg["focal_gamma_pos"],
        prob_margin=cfg["asl_prob_margin"],
        relation_weights=rel_w,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.1
    )

    train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=cfg["batch_size"], shuffle=False)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1  = -1.0
    patience_cnt = 0

    print()
    print(f"{'Epoch':>5}  {'Train Loss':>11}  {'Val F1':>8}  {'best':>5}")
    print("-" * 40)

    for epoch in range(1, cfg["epochs"] + 1):
        # Train
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            labels = batch.y

            asl  = criterion(logits, labels)
            ewc  = ewc_loss(model, fisher, old_params, cfg["ewc_lambda"])
            loss = asl + ewc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / max(len(train_loader), 1)

        # Validate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.edge_attr)
                all_probs.append(torch.sigmoid(logits).cpu())
                all_labels.append(batch.y.cpu())

        probs  = torch.cat(all_probs).numpy()
        labels = torch.cat(all_labels).numpy()
        mask   = labels >= 0
        preds  = (probs >= 0.5).astype(float)
        val_f1 = f1_score(
            labels[mask].astype(int), preds[mask].astype(int),
            average="macro", zero_division=0
        )

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), cfg["save_path"])
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 5 == 0 or is_best:
            print(f"{epoch:>5}  {avg_loss:>11.4f}  {val_f1:>8.4f}  {'*' if is_best else ''}")

        if patience_cnt >= cfg["patience"]:
            print(f"  Early stop at epoch {epoch}")
            break

    print(f"\nBest val Macro F1: {best_val_f1:.4f}")
    print(f"Saved: {cfg['save_path']}")

    # ── Threshold tuning on synth val ─────────────────────────────────────────
    model.load_state_dict(torch.load(cfg["save_path"], weights_only=False, map_location=device))
    model.eval()

    all_scores, all_gt = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            probs  = torch.sigmoid(logits).cpu().numpy()
            labels = batch.y.cpu().numpy()
            for i in range(probs.shape[0]):
                all_scores.append(probs[i])
                all_gt.append(labels[i])

    scores_arr = np.stack(all_scores)
    gt_arr     = np.stack(all_gt)

    thresholds = {}
    for r in range(NUM_RELATIONS):
        col_gt = gt_arr[:, r]
        col_s  = scores_arr[:, r]
        valid  = col_gt >= 0
        if valid.sum() < 10 or (col_gt[valid] > 0.5).sum() == 0:
            thresholds[r] = 0.5
            continue
        best_t, best_f1 = 0.5, -1.0
        for t in [i / 10 for i in range(1, 10)]:
            pred = (col_s[valid] >= t).astype(int)
            gt_v = (col_gt[valid] >= 0.5).astype(int)
            tp = (pred & gt_v).sum()
            fp = (pred & ~gt_v).sum()
            fn = (~pred & gt_v).sum()
            p  = tp / max(tp + fp, 1)
            r_ = tp / max(tp + fn, 1)
            f1 = 2 * p * r_ / max(p + r_, 1e-9)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[r] = best_t

    print("\nTuned thresholds (on synth val):")
    for r, t in thresholds.items():
        print(f"  {RELATION_NAMES[r]:20s}: {t:.2f}")

    with open(cfg["thresh_save_path"], "w") as f:
        json.dump({str(k): v for k, v in thresholds.items()}, f, indent=2)
    print(f"Saved thresholds: {cfg['thresh_save_path']}")


if __name__ == "__main__":
    main()
