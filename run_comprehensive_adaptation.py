"""
Comprehensive GeoKAN Cross-Domain Adaptation for Tabletop Scenes.

Goal: Improve OVERALL cross-domain performance across ALL relations,
not just on_top_of/under.

Root cause: The tabletop splats use Z-DOWN convention, affecting ALL
Z-dependent relations: on_top_of, under, higher_than, lower_than.
Directional XY-only relations (left_of, right_of, in_front_of, behind)
already work well (67-86% R@5).

Strategy:
1. Fine-tune BOTH contact AND directional heads on Z-inverted synthetic data
   - Contact head: EWC=0.5 (allows significant on_top_of/under adaptation)
   - Directional head: EWC=1.5 (stronger to preserve 3RScan higher/lower)
   - adj_to/attached_to MASKED in synthetic (prevent reg.)
   - higher_than/lower_than in synthetic get loss weight 2.0

2. Apply Test-Time BatchNorm on ALL 4 BN layers (5 passes)
   - Calibrates BN stats to tabletop distribution for all heads
   - Improves relative ranking of all relations in tabletop

Saves: models/geokan_relation_tabletop_adapted.pt

Run: python run_comprehensive_adaptation.py
"""
import sys
sys.path.insert(0, ".")

import os, json, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from geokan_relation import GeoKANRelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, Relation, LEGACY_12_TO_10_COLS
from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
from src.gaussian.clustering import gaussian_to_objects, extract_gaussian_node_features
from scripts.build_3rscan_graphs import extract_3rscan_edge_features
from scipy.optimize import linear_sum_assignment

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kwargs): return it

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
ZINV_DIR = "D:/logicsplat_data/synthetic_zinv"
RSCAN_DIR = "D:/logicsplat_data/3rscan_graph_cache"
DATA_DIR  = "D:/logicsplat_data/processed"
SAVE_PATH = "models/geokan_relation_tabletop_adapted.pt"
THRESH_PATH = "models/geokan_relation_tabletop_adapted_thresholds.json"
SCENES    = [f"scene_{i:02d}" for i in range(6, 14)]

# Relations to suppress from synthetic labels (noisy in synthetic, anchor from 3RScan)
SUPPRESS_SYNTH = [int(Relation.ADJACENT_TO), int(Relation.ATTACHED_TO)]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_model_fresh():
    state = torch.load("models/geokan_relation_v4.pt", weights_only=False, map_location="cpu")
    hd = state["node_encoder.0.weight"].shape[0]
    ef = state["conv1.lin_edge.weight"].shape[1]
    m  = GeoKANRelationGNN(node_feat_dim=10, edge_feat_dim=ef, hidden_dim=hd)
    m.load_state_dict(state)
    return m


def freeze_backbone_only(model):
    """Freeze backbone (GATv2 + node encoder + pair proj) but unfreeze BOTH heads."""
    frozen = trainable = 0
    for name, p in model.named_parameters():
        if any(x in name for x in ["head_contact", "head_directional"]):
            p.requires_grad = True; trainable += p.numel()
        else:
            p.requires_grad = False; frozen += p.numel()
    print(f"  Frozen (backbone): {frozen:,}  Trainable (both heads): {trainable:,}")
    return model


def load_zinv(n_train, n_val):
    files = sorted(f for f in os.listdir(ZINV_DIR) if f.endswith(".pt"))
    dlist = []
    for f in files:
        g = torch.load(os.path.join(ZINV_DIR, f), weights_only=False)
        if g["edge_attr"].shape[1] != 22 or g["edge_label"].shape[1] != NUM_RELATIONS:
            continue
        y = g["edge_label"].clone()
        for r in SUPPRESS_SYNTH:
            y[:, r] = -1.0
        dlist.append(Data(x=g["x"], edge_index=g["edge_index"], edge_attr=g["edge_attr"], y=y))
    print(f"  Z-inverted synthetic: {len(dlist)} scenes loaded")
    return dlist[:n_train], dlist[n_train:n_train+n_val]


def load_rscan(n, seed=999):
    files = sorted(f for f in os.listdir(RSCAN_DIR) if f.endswith(".pt"))
    random.Random(seed).shuffle(files); files = files[:n]
    dlist = []
    for f in files:
        g = torch.load(os.path.join(RSCAN_DIR, f), weights_only=False)
        if g["edge_attr"].shape[1] != 22: continue
        if g["edge_label"].shape[1] == 12:
            g["edge_label"] = g["edge_label"][:, LEGACY_12_TO_10_COLS]
        if g["edge_label"].shape[1] != NUM_RELATIONS: continue
        dlist.append(Data(x=g["x"], edge_index=g["edge_index"], edge_attr=g["edge_attr"], y=g["edge_label"]))
    print(f"  3RScan mix: {len(dlist)} scenes loaded")
    return dlist


class ASL(nn.Module):
    def __init__(self, pw, gn=4., gp=0., rw=None):
        super().__init__()
        self.register_buffer("pw", pw); self.gn=gn; self.gp=gp
        if rw is not None: self.register_buffer("rw", rw)
        else: self.rw = None

    def forward(self, lg, lb):
        mask = (lb >= 0).float(); lbc = lb.clamp(0)
        pr = torch.sigmoid(lg)
        lp = F.binary_cross_entropy_with_logits(lg, torch.ones_like(lg), pos_weight=self.pw, reduction="none")
        fp = (1-pr)**self.gp
        xn = pr.clamp(0.)
        ln_loss = -torch.log((1-xn).clamp(1e-8)); fn = xn**self.gn
        loss = torch.where(lbc >= 0.5, lp*fp, ln_loss*fn) * mask
        if self.rw is not None: loss = loss * self.rw.unsqueeze(0)
        nv = mask.sum()
        return loss.sum() / nv if nv > 0 else loss.sum()


def snapshot_weights(model, prefixes):
    return {n: p.detach().clone()
            for n, p in model.named_parameters()
            if any(p_str in n for p_str in prefixes)}


def ewc_loss_selective(model, orig, lambdas):
    """EWC with different lambda per head prefix."""
    total = torch.tensor(0., device=next(model.parameters()).device)
    for n, p in model.named_parameters():
        if n not in orig: continue
        lam = 0.
        for prefix, l in lambdas.items():
            if prefix in n: lam = l; break
        if lam > 0:
            total = total + lam * (p - orig[n].to(p.device)).pow(2).sum()
    return total


def eval_model(model, loader, device):
    model.eval()
    ap, al = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            ap.append(torch.sigmoid(model(b.x, b.edge_index, b.edge_attr)).cpu())
            al.append(b.y.cpu())
    ap = torch.cat(ap); al = torch.cat(al)
    f1s = []
    for r in range(NUM_RELATIONS):
        v = al[:, r] >= 0
        if not v.any(): f1s.append(0.); continue
        f1s.append(f1_score((al[v,r]>=0.5).numpy().astype(int),
                             (ap[v,r]>=0.5).float().numpy(), zero_division=0))
    pos = [r for r in range(NUM_RELATIONS) if (al[:,r]>=0.5).sum()>0]
    mf1 = float(np.mean([f1s[r] for r in pos])) if pos else 0.
    print(f"\n  Val macro(pos): {mf1:.4f}")
    for r in range(NUM_RELATIONS):
        pn = int((al[:,r]>=0.5).sum())
        if pn>0: print(f"    {RELATION_NAMES[r]:20s} F1={f1s[r]:.3f} pos={pn}")
    return mf1


def tune_thresh(model, loader, device):
    model.eval()
    ap, al = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            ap.append(torch.sigmoid(model(b.x, b.edge_index, b.edge_attr)).cpu())
            al.append(b.y.cpu())
    ap = torch.cat(ap).numpy(); al = torch.cat(al).numpy()
    thr = {}
    for r in range(NUM_RELATIONS):
        v = al[:, r] >= 0
        if not v.any(): thr[r]=0.5; continue
        pv, lv = ap[v,r], (al[v,r]>=0.5).astype(int)
        bf, bt = 0., 0.5
        for t in np.arange(0.1, 0.91, 0.05):
            f = f1_score(lv, (pv>=t).astype(int), zero_division=0)
            if f>bf: bf, bt = f, float(t)
        thr[r] = bt
    return thr


def build_tabletop_graph(scene_dir):
    ply = os.path.join(scene_dir, "splat.ply")
    if not os.path.exists(ply): return None, None, None
    cloud = load_gaussian_ply(ply)
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)
    n_hint = len(json.load(open(os.path.join(scene_dir, "ground_truth_relations.json")))["objects"])
    objs, _ = gaussian_to_objects(cloud, target_min=max(2,n_hint-1), target_max=n_hint+1)
    if len(objs) < 2: return None, None, None
    mins = np.stack([o.bbox_min for o in objs]); maxs = np.stack([o.bbox_max for o in objs])
    se = np.maximum(maxs.max(0)-mins.min(0), 1e-6)
    sm = float(np.mean([np.linalg.norm(np.maximum(o.size,1e-6)) for o in objs]))
    sv = float(np.median([float(np.prod(np.maximum(o.size,1e-6))) for o in objs]))
    czs = np.array([o.centroid[2] for o in objs])
    zr = np.zeros(len(objs))
    if len(objs)>1:
        for rk, oi in enumerate(np.argsort(czs)): zr[oi]=rk/(len(objs)-1)
    x = np.stack([extract_gaussian_node_features(objs[i],se,mins.min(0),
                  scene_mean_diag=sm, scene_median_volume=sv, z_rank=float(zr[i]))
                  for i in range(len(objs))])
    ef_l, sl, dl = [], [], []
    for i in range(len(objs)):
        for j in range(len(objs)):
            if i!=j:
                ef_l.append(extract_3rscan_edge_features(objs[i],objs[j],se))
                sl.append(i); dl.append(j)
    return (torch.tensor(x, dtype=torch.float32),
            torch.tensor([sl,dl], dtype=torch.long),
            torch.tensor(np.stack(ef_l), dtype=torch.float32))


# ── Step 1: Comprehensive Fine-tuning ─────────────────────────────────────────

def run_finetune():
    print("="*65)
    print("STEP 1: Fine-tune BOTH heads on Z-inverted synthetic data")
    print("="*65)
    print("  Contact head EWC: 0.5  |  Directional head EWC: 1.5")

    train_zinv, val_zinv = load_zinv(700, 100)
    rscan_mix = load_rscan(350)
    train_data = train_zinv + rscan_mix
    print(f"  Train: {len(train_zinv)} Z-inv + {len(rscan_mix)} 3RScan = {len(train_data)} total")

    train_ld = DataLoader(train_data, batch_size=8, shuffle=True)
    val_ld   = DataLoader(val_zinv,   batch_size=8, shuffle=False)

    model = load_model_fresh()
    model = freeze_backbone_only(model)
    model = model.to(DEVICE)

    orig_w = snapshot_weights(model, ["head_contact", "head_directional"])

    # Class weights
    pos = torch.zeros(NUM_RELATIONS); valid = torch.zeros(NUM_RELATIONS)
    for d in train_data:
        for r in range(NUM_RELATIONS):
            col = d.y[:, r]; vm = col >= 0
            pos[r] += (col[vm] >= 0.5).float().sum(); valid[r] += vm.float().sum()
    pos = torch.clamp(pos, 1.)
    pw = torch.clamp((valid-pos)/pos, max=8.).to(DEVICE)

    # Relation weights: boost Z-dependent relations
    rw = torch.ones(NUM_RELATIONS)
    rw[int(Relation.ON_TOP_OF)]   = 4.0
    rw[int(Relation.UNDER)]       = 4.0
    rw[int(Relation.HIGHER_THAN)] = 2.0  # also boost higher/lower
    rw[int(Relation.LOWER_THAN)]  = 2.0
    rw = rw.to(DEVICE)

    criterion = ASL(pw, rw=rw).to(DEVICE)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=3e-5, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80, eta_min=3e-5*0.05)

    # Per-head EWC lambdas
    ewc_lambdas = {"head_contact": 0.5, "head_directional": 0.3}

    best_f1, patience_cnt, best_epoch = 0., 0, 0

    for epoch in range(1, 81):
        model.train()
        total = 0.; nb = 0
        for batch in _tqdm(train_ld, desc=f"E{epoch:02d}", leave=False):
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            bce = criterion(logits, batch.y)
            met = model.metric_reg()
            reg = ewc_loss_selective(model, orig_w, ewc_lambdas)
            loss = bce + 1e-4*met + reg
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 2.)
            opt.step(); total += loss.item(); nb += 1
        sch.step()

        print(f"\n  Epoch {epoch:02d}  loss={total/max(nb,1):.4f}  lr={opt.param_groups[0]['lr']:.2e}")
        vf1 = eval_model(model, val_ld, DEVICE)

        if vf1 > best_f1:
            best_f1 = vf1; patience_cnt = 0; best_epoch = epoch
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  * Saved (best={best_f1:.4f} @ epoch {epoch})")
        else:
            patience_cnt += 1
            if patience_cnt >= 25:
                print(f"  Early stop @ epoch {epoch} (best epoch {best_epoch})")
                break

    if not os.path.exists(SAVE_PATH):
        torch.save(model.state_dict(), SAVE_PATH)

    # Threshold tuning
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True))
    thr = tune_thresh(model, val_ld, DEVICE)
    with open(THRESH_PATH, "w") as f:
        json.dump({str(k): v for k,v in thr.items()}, f, indent=2)

    print(f"\n  Done. Best val F1={best_f1:.4f} @ epoch {best_epoch}")


# ── Step 2: Full TTBN (ALL 4 BN layers) ───────────────────────────────────────

def run_full_ttbn(n_passes=5):
    print("\n" + "="*65)
    print(f"STEP 2: Full TTBN — ALL BN layers (contact + directional), {n_passes} passes")
    print("="*65)

    full_state = torch.load("models/geokan_relation_v4.pt", weights_only=False, map_location="cpu")
    hd = full_state["node_encoder.0.weight"].shape[0]
    ef = full_state["conv1.lin_edge.weight"].shape[1]
    model = GeoKANRelationGNN(node_feat_dim=10, edge_feat_dim=ef, hidden_dim=hd).to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True))

    # Set ALL BN layers to train() mode with high momentum for fast adaptation
    model.eval()
    n_bn = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()
            module.momentum = 0.3
            n_bn += 1
            print(f"  Adapting: {name} (in_features={module.num_features})")

    # Build tabletop graphs
    graphs = []
    for sn in SCENES:
        x, ei, ea = build_tabletop_graph(os.path.join(DATA_DIR, sn))
        if x is None: continue
        graphs.append((x, ei, ea))
        print(f"  Loaded {sn}: {x.shape[0]} nodes")

    print(f"\n  Running {n_passes} passes on {len(graphs)} tabletop scenes...")
    with torch.no_grad():
        for p in range(n_passes):
            for x, ei, ea in graphs:
                model(x.to(DEVICE), ei.to(DEVICE), ea.to(DEVICE))

    print("  TTBN complete — all BN adapted to tabletop distribution")
    model.eval()

    # Re-tune thresholds after TTBN
    train_zinv, val_zinv = load_zinv(700, 100)
    val_ld = DataLoader(val_zinv, batch_size=8, shuffle=False)
    thr = tune_thresh(model, val_ld, DEVICE)
    with open(THRESH_PATH, "w") as f:
        json.dump({str(k): v for k,v in thr.items()}, f, indent=2)

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"  Saved model with full TTBN: {SAVE_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ft",       action="store_true")
    parser.add_argument("--skip-ttbn",     action="store_true")
    parser.add_argument("--ttbn-passes",   type=int, default=5)
    args = parser.parse_args()

    if not args.skip_ft:
        run_finetune()
    if not args.skip_ttbn:
        run_full_ttbn(n_passes=args.ttbn_passes)

    print(f"\n{'='*65}")
    print(f"  Saved: {SAVE_PATH}")
    print(f"  Thresholds: {THRESH_PATH}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
