"""
Fine-tune GeoKAN contact head on Z-INVERTED synthetic tabletop data.

ROOT CAUSE: The 8 tabletop Gaussian splats use Z-DOWN convention
(camera above, Z increases toward camera = physically higher objects have LOWER Z).
The model trained on 3RScan (Z-UP) assigns contact_score=0 and vz_ratio<0 to GT
on_top_of edges, causing R@5=0% for on_top_of and under.

FIX: Fine-tune contact head on synthetic scenes with Z-INVERTED coordinates
(matching real tabletop splat convention). The contact head learns:
  vz_ratio<0, contact_score=0, negative vert_gap → on_top_of (Z-down convention)

Key training parameters tuned against the root cause:
  - ONLY z-inverted synthetic data (not normal-Z synthetic)
  - adjacent_to masked from synthetic loss (prevents adj_to regression on 3RScan)
  - Strong EWC (lambda=0.5) on contact-head weights to anchor 3RScan contact patterns
  - Equal mix with 3RScan (preserves full 3RScan contact behavior)
  - Longer training (70 epochs) with patience=25 for slow convergence

Usage:
    python finetune_geokan_tabletop.py
    python finetune_geokan_tabletop.py --epochs 70 --lr 3e-5
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

from geokan_relation import GeoKANRelationGNN, CONTACT_INDICES
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, Relation

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kwargs):
        return it


# ── Config ─────────────────────────────────────────────────────────────────────

CONFIG = {
    "pretrained_path":   "models/geokan_relation_v4.pt",
    "save_path":         "models/geokan_relation_tabletop_adapted.pt",
    "thresh_save_path":  "models/geokan_relation_tabletop_adapted_thresholds.json",
    # Z-INVERTED synthetic data — matches real tabletop Z convention
    # (400 scenes in synthetic_v2_zinv, correct path)
    "zinv_dir":          "D:/logicsplat_data/synthetic_v2_zinv",
    # Z-NORMAL for TTBN calibration only
    "znorm_dir":         "D:/logicsplat_data/synthetic_v2",
    "rscan_cache_dir":   "D:/logicsplat_data/3rscan_graph_cache",
    # Data split: 300 train + 100 val (from 400 available Z-inverted scenes)
    "n_zinv_train":      300,
    "n_zinv_val":        100,
    "n_rscan_mix":       300,
    # TTBN calibration sizes
    "n_ttbn_zinv":       300,
    "n_ttbn_znorm":      300,
    # Training: moderate EWC (0.1) allows MetricNet to adapt more
    "lr":                5e-5,
    "weight_decay":      1e-4,
    "epochs":            100,
    "batch_size":        8,
    "patience":          30,
    "device":            "cuda" if torch.cuda.is_available() else "cpu",
    # Aggressive on_top_of/under boosting — pushes scores above directional rels
    "ontop_weight":      8.0,
    "under_weight":      8.0,
    "suppress_adj_in_synth": True,
    # Moderate EWC: allows MetricNet adaptation while anchoring 3RScan
    "ewc_lambda":        0.1,
    "metric_reg_weight": 1e-4,
    # ASL params
    "focal_gamma_neg":   4.0,
    "focal_gamma_pos":   0.0,
    "asl_prob_margin":   0.0,
    "pos_weight_cap":    8.0,
    "node_feat_dim":     10,
    "edge_feat_dim":     22,
    "hidden_dim":        128,
}

# Relations to suppress from synthetic labels to prevent 3RScan regression
SUPPRESS_RELS_IN_SYNTH = [int(Relation.ADJACENT_TO), int(Relation.ATTACHED_TO)]


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
            self.register_buffer("relation_weights", relation_weights)
        else:
            self.relation_weights = None

    def forward(self, logits, labels):
        mask      = (labels >= 0).float()
        labels_c  = labels.clamp(min=0.0)
        probs     = torch.sigmoid(logits)

        loss_pos  = F.binary_cross_entropy_with_logits(
            logits, torch.ones_like(logits), pos_weight=self.pos_weight, reduction="none"
        )
        focal_pos = (1.0 - probs) ** self.gamma_pos

        xs_neg   = torch.clamp(probs - self.prob_margin, min=0.0)
        loss_neg = -torch.log(torch.clamp(1.0 - xs_neg, min=1e-8))
        focal_neg = xs_neg ** self.gamma_neg

        loss   = torch.where(labels_c >= 0.5, loss_pos * focal_pos, loss_neg * focal_neg)
        masked = loss * mask
        if self.relation_weights is not None:
            masked = masked * self.relation_weights.unsqueeze(0)

        n_valid = mask.sum()
        return masked.sum() / n_valid if n_valid > 0 else masked.sum()


# ── Data helpers ───────────────────────────────────────────────────────────────

def suppress_synth_labels(y: torch.Tensor, rels: list) -> torch.Tensor:
    """Set specified relations to -1 (ignore) in synthetic labels."""
    y2 = y.clone()
    for r in rels:
        y2[:, r] = -1.0
    return y2


def load_zinverted_data(zinv_dir: str, n_train: int, n_val: int, suppress_rels: list):
    files = sorted(f for f in os.listdir(zinv_dir) if f.endswith(".pt"))
    data_list = []
    for fname in files:
        g = torch.load(os.path.join(zinv_dir, fname), weights_only=False)
        if g["edge_attr"].shape[1] != 22: continue
        if g["edge_label"].shape[1] != NUM_RELATIONS: continue
        y = suppress_synth_labels(g["edge_label"], suppress_rels)
        data_list.append(Data(
            x=g["x"], edge_index=g["edge_index"], edge_attr=g["edge_attr"], y=y,
        ))
    print(f"  Loaded {len(data_list)} Z-inverted synthetic scenes "
          f"(on_top_of/under only active from these)")
    train = data_list[:n_train]
    val   = data_list[n_train: n_train + n_val]
    return train, val


def load_rscan_subset(cache_dir: str, n_scenes: int, seed: int = 999):
    from src.relations.schema import LEGACY_12_TO_10_COLS
    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
    rng = random.Random(seed); rng.shuffle(files)
    files = files[:n_scenes]
    data_list = []
    for fname in files:
        g = torch.load(os.path.join(cache_dir, fname), weights_only=False)
        if g["edge_attr"].shape[1] != 22: continue
        if g["edge_label"].shape[1] == 12:
            g["edge_label"] = g["edge_label"][:, LEGACY_12_TO_10_COLS]
        if g["edge_label"].shape[1] != NUM_RELATIONS: continue
        data_list.append(Data(
            x=g["x"], edge_index=g["edge_index"], edge_attr=g["edge_attr"], y=g["edge_label"],
        ))
    print(f"  Loaded {len(data_list)} 3RScan scenes for domain mixing")
    return data_list


def compute_pos_weight(data_list, cap=8.0):
    pos = torch.zeros(NUM_RELATIONS)
    valid = torch.zeros(NUM_RELATIONS)
    for d in data_list:
        for r in range(NUM_RELATIONS):
            col = d.y[:, r]
            vm  = col >= 0
            pos[r]   += (col[vm] >= 0.5).float().sum()
            valid[r] += vm.float().sum()
    pos   = torch.clamp(pos,   min=1.0)
    neg   = torch.clamp(valid - pos, min=1.0)
    return torch.clamp(neg / pos, max=cap)


# ── TTBN ───────────────────────────────────────────────────────────────────────

def apply_ttbn(model, zinv_data, znorm_data, device):
    """
    Test-Time BN: reset contact-head BatchNorm to tabletop statistics.
    Uses EXACT statistics from mixed Z-normal + Z-inverted tabletop data,
    computed via forward hooks — no moving averages, just one-shot exact stats.
    """
    print(f"\n[TTBN] Adapting contact-head BN to tabletop distribution "
          f"({len(zinv_data)} Z-inv + {len(znorm_data)} Z-norm)...")
    model.eval()
    captured = {"l1": [], "l2": []}

    h1 = model.head_contact.layer1.bn.register_forward_hook(
        lambda m, inp, out: captured["l1"].append(inp[0].detach().cpu()))
    h2 = model.head_contact.layer2.bn.register_forward_hook(
        lambda m, inp, out: captured["l2"].append(inp[0].detach().cpu()))

    ttbn_all = zinv_data + znorm_data
    loader = DataLoader(ttbn_all, batch_size=64, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _ = model(batch.x, batch.edge_index, batch.edge_attr)

    h1.remove(); h2.remove()

    all_l1 = torch.cat(captured["l1"], dim=0)
    all_l2 = torch.cat(captured["l2"], dim=0)

    with torch.no_grad():
        model.head_contact.layer1.bn.running_mean.copy_(all_l1.mean(0))
        model.head_contact.layer1.bn.running_var.copy_(all_l1.var(0, unbiased=False).clamp(1e-6))
        model.head_contact.layer2.bn.running_mean.copy_(all_l2.mean(0))
        model.head_contact.layer2.bn.running_var.copy_(all_l2.var(0, unbiased=False).clamp(1e-6))

    print(f"  TTBN done. Layer1 BN mean range: "
          f"[{model.head_contact.layer1.bn.running_mean.min():.3f}, "
          f"{model.head_contact.layer1.bn.running_mean.max():.3f}]")


# ── Model setup ────────────────────────────────────────────────────────────────

def load_and_prepare(config):
    state    = torch.load(config["pretrained_path"], weights_only=False, map_location="cpu")
    hd       = state["node_encoder.0.weight"].shape[0]
    ef       = state["conv1.lin_edge.weight"].shape[1]
    model    = GeoKANRelationGNN(node_feat_dim=config["node_feat_dim"],
                                  edge_feat_dim=ef, hidden_dim=hd)
    model.load_state_dict(state)

    frozen, trainable = 0, 0
    for name, p in model.named_parameters():
        if "head_contact" in name:
            p.requires_grad = True; trainable += p.numel()
        else:
            p.requires_grad = False; frozen += p.numel()

    print(f"  Frozen (backbone + directional head): {frozen:,}")
    print(f"  Trainable (contact head):             {trainable:,}")
    return model


def snapshot_weights(model):
    return {n: p.detach().clone()
            for n, p in model.named_parameters() if "head_contact" in n}


def ewc_loss(model, orig, lam):
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for n, p in model.named_parameters():
        if "head_contact" in n and n in orig:
            loss = loss + (p - orig[n].to(p.device)).pow(2).sum()
    return lam * loss


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, thresholds=None, split_name="val"):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(batch.y.cpu())

    all_probs  = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    thresh_dict = thresholds or {}
    all_preds   = torch.zeros_like(all_probs)
    for r in range(NUM_RELATIONS):
        all_preds[:, r] = (all_probs[:, r] >= thresh_dict.get(r, 0.5)).float()

    f1_per = np.zeros(NUM_RELATIONS)
    for r in range(NUM_RELATIONS):
        valid = all_labels[:, r] >= 0
        if valid.sum() == 0: continue
        f1_per[r] = f1_score(
            (all_labels[valid, r] >= 0.5).numpy().astype(int),
            all_preds[valid, r].numpy(),
            zero_division=0
        )

    macro = float(np.mean(f1_per))
    pos_rels = [r for r in range(NUM_RELATIONS) if (all_labels[:, r] >= 0.5).sum() > 0]
    macro_pos = float(np.mean([f1_per[r] for r in pos_rels])) if pos_rels else 0.0

    print(f"\n  {split_name}  Macro(all)={macro:.4f}  Macro(pos-only)={macro_pos:.4f}")
    for r in range(NUM_RELATIONS):
        pos_n = int((all_labels[:, r] >= 0.5).sum())
        if pos_n > 0:
            print(f"    {RELATION_NAMES[r]:20s}  F1={f1_per[r]:.3f}  pos={pos_n}")
    return macro_pos, {RELATION_NAMES[r]: float(f1_per[r]) for r in range(NUM_RELATIONS)}


def tune_thresholds(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            all_probs.append(torch.sigmoid(model(batch.x, batch.edge_index, batch.edge_attr)).cpu())
            all_labels.append(batch.y.cpu())

    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    thresholds = {}
    for r in range(NUM_RELATIONS):
        valid = all_labels[:, r] >= 0
        if valid.sum() == 0: thresholds[r] = 0.5; continue
        p_v  = all_probs[valid, r]
        l_v  = (all_labels[valid, r] >= 0.5).astype(int)
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.1, 0.91, 0.05):
            f1 = f1_score(l_v, (p_v >= t).astype(int), zero_division=0)
            if f1 > best_f1: best_f1, best_t = f1, float(t)
        thresholds[r] = best_t

    print("\n  Tuned thresholds:")
    for r in range(NUM_RELATIONS):
        print(f"    {RELATION_NAMES[r]:20s}: {thresholds[r]:.2f}")
    return thresholds


# ── Training loop ───────────────────────────────────────────────────────────────

def train(config):
    device = config["device"]
    print("=" * 65)
    print("GeoKAN Tabletop Adaptation v2 — Z-Inverted Contact Head FT")
    print("=" * 65)
    print(f"  Pretrained: {config['pretrained_path']}")
    print(f"  Output:     {config['save_path']}")
    print(f"  Z-inv dir:  {config['zinv_dir']}")

    suppress_rels = SUPPRESS_RELS_IN_SYNTH if config["suppress_adj_in_synth"] else []

    print("\n[1] Loading data...")
    train_zinv, val_zinv = load_zinverted_data(
        config["zinv_dir"], config["n_zinv_train"], config["n_zinv_val"],
        suppress_rels=suppress_rels,
    )
    rscan_mix = load_rscan_subset(config["rscan_cache_dir"], config["n_rscan_mix"])
    train_data = train_zinv + rscan_mix
    print(f"  Train: {len(train_zinv)} Z-inv synth + {len(rscan_mix)} 3RScan = {len(train_data)} total")
    print(f"  Val  : {len(val_zinv)} Z-inv synth")

    # Load Z-normal data for TTBN (not used in training, only for BN calibration)
    znorm_ttbn = load_zinverted_data(
        config.get("znorm_dir", config["zinv_dir"]),
        config.get("n_ttbn_znorm", 300), 0, suppress_rels=[]
    )[0]
    zinv_ttbn = load_zinverted_data(
        config["zinv_dir"], config.get("n_ttbn_zinv", 300), 0, suppress_rels=[]
    )[0]

    train_loader = DataLoader(train_data, batch_size=config["batch_size"],
                              shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_zinv,   batch_size=config["batch_size"], shuffle=False)

    print("\n[2] Loading model...")
    model  = load_and_prepare(config).to(device)
    orig_w = snapshot_weights(model)

    # TTBN: adapt contact-head BN to tabletop distribution BEFORE any weight updates
    apply_ttbn(model, zinv_ttbn, znorm_ttbn, device)

    print("\n[3] Setting up loss...")
    pos_weight = compute_pos_weight(train_data, cap=config["pos_weight_cap"]).to(device)

    rel_weights = torch.ones(NUM_RELATIONS)
    rel_weights[int(Relation.ON_TOP_OF)] = config["ontop_weight"]
    rel_weights[int(Relation.UNDER)]     = config["under_weight"]
    rel_weights = rel_weights.to(device)

    criterion = ASLLoss(
        pos_weight=pos_weight,
        gamma_neg=config["focal_gamma_neg"],
        gamma_pos=config["focal_gamma_pos"],
        prob_margin=config["asl_prob_margin"],
        relation_weights=rel_weights,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config["lr"] * 0.05
    )

    print("\n[4] Baseline evaluation...")
    baseline_f1, _ = evaluate(model, val_loader, device, split_name="Val-baseline")

    print("\n[5] Training...")
    best_f1 = 0.0
    patience_counter = 0
    os.makedirs("models", exist_ok=True)
    best_epoch = 0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in _tqdm(train_loader, desc=f"E{epoch:02d}", leave=False):
            batch = batch.to(device)
            logits    = model(batch.x, batch.edge_index, batch.edge_attr)
            bce_loss  = criterion(logits, batch.y)
            met_loss  = model.metric_reg()
            reg_loss  = ewc_loss(model, orig_w, config["ewc_lambda"])
            total     = bce_loss + config["metric_reg_weight"] * met_loss + reg_loss

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 2.0)
            optimizer.step()

            epoch_loss += total.item(); n_batches += 1

        scheduler.step()
        lr      = optimizer.param_groups[0]["lr"]
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"\n  Epoch {epoch:02d}/{config['epochs']}  loss={avg_loss:.4f}  lr={lr:.2e}")

        val_f1, _ = evaluate(model, val_loader, device, split_name="Val")

        if val_f1 > best_f1:
            best_f1 = val_f1; patience_counter = 0; best_epoch = epoch
            torch.save(model.state_dict(), config["save_path"])
            print(f"  * Best F1={best_f1:.4f} at epoch {epoch} -> saved")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config['patience']})")
            if patience_counter >= config["patience"]:
                print(f"\n  Early stop at epoch {epoch} (best epoch={best_epoch})")
                break

    if not os.path.exists(config["save_path"]) or best_f1 == 0.0:
        print("  Saving final model (no improvement found)")
        torch.save(model.state_dict(), config["save_path"])

    # Threshold tuning
    print("\n[6] Threshold tuning on Z-inv val set...")
    model.load_state_dict(torch.load(config["save_path"], map_location=device, weights_only=True))
    thresholds = tune_thresholds(model, val_loader, device)
    with open(config["thresh_save_path"], "w") as f:
        json.dump({str(k): v for k, v in thresholds.items()}, f, indent=2)
    print(f"  Thresholds saved: {config['thresh_save_path']}")

    print("\n[7] Final val eval (tuned thresholds):")
    evaluate(model, val_loader, device, thresholds=thresholds, split_name="Val-tuned")

    print(f"\n{'='*65}")
    print(f"  Baseline val F1: {baseline_f1:.4f}")
    print(f"  Best val F1:     {best_f1:.4f}  (epoch {best_epoch})")
    print(f"  Model:      {config['save_path']}")
    print(f"  Thresholds: {config['thresh_save_path']}")
    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int,   default=CONFIG["epochs"])
    parser.add_argument("--lr",        type=float, default=CONFIG["lr"])
    parser.add_argument("--ewc-lam",   type=float, default=CONFIG["ewc_lambda"])
    parser.add_argument("--zinv-dir",  type=str,   default=CONFIG["zinv_dir"])
    args = parser.parse_args()

    config = CONFIG.copy()
    config["epochs"]      = args.epochs
    config["lr"]          = args.lr
    config["ewc_lambda"]  = args.ewc_lam
    config["zinv_dir"]    = args.zinv_dir
    train(config)


if __name__ == "__main__":
    main()
