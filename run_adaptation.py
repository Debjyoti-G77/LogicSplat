"""
GeoKAN Tabletop Adaptation — Final Pipeline
Combines Z-inverted fine-tuning + Test-Time BatchNorm adaptation (TTBN)

Step 1: Fine-tune contact head on Z-inverted synthetic data (EWC=0.5)
Step 2: Apply TTBN on real tabletop scenes to calibrate BN stats
Step 3: Evaluate

Run: python run_adaptation.py
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ZINV_DIR   = "D:/logicsplat_data/synthetic_zinv"
RSCAN_DIR  = "D:/logicsplat_data/3rscan_graph_cache"
DATA_DIR   = "D:/logicsplat_data/processed"
SAVE_PATH  = "models/geokan_relation_tabletop_adapted.pt"
THRESH_PATH = "models/geokan_relation_tabletop_adapted_thresholds.json"
SCENES     = [f"scene_{i:02d}" for i in range(6, 14)]

SUPPRESS_RELS = [int(Relation.ADJACENT_TO), int(Relation.ATTACHED_TO)]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_model_fresh():
    state = torch.load("models/geokan_relation_v4.pt", weights_only=False, map_location="cpu")
    hd = state["node_encoder.0.weight"].shape[0]
    ef = state["conv1.lin_edge.weight"].shape[1]
    m  = GeoKANRelationGNN(node_feat_dim=10, edge_feat_dim=ef, hidden_dim=hd)
    m.load_state_dict(state)
    return m


def freeze_non_contact(model):
    frozen = trainable = 0
    for name, p in model.named_parameters():
        if "head_contact" in name:
            p.requires_grad = True; trainable += p.numel()
        else:
            p.requires_grad = False; frozen += p.numel()
    print(f"  Frozen: {frozen:,}  Trainable: {trainable:,}")
    return model


def load_zinv(n_train, n_val):
    files = sorted(f for f in os.listdir(ZINV_DIR) if f.endswith(".pt"))
    dlist = []
    for f in files:
        g = torch.load(os.path.join(ZINV_DIR, f), weights_only=False)
        if g["edge_attr"].shape[1] != 22 or g["edge_label"].shape[1] != NUM_RELATIONS: continue
        y = g["edge_label"].clone()
        for r in SUPPRESS_RELS: y[:, r] = -1.0
        dlist.append(Data(x=g["x"], edge_index=g["edge_index"], edge_attr=g["edge_attr"], y=y))
    return dlist[:n_train], dlist[n_train:n_train+n_val]


def load_rscan(n, seed=999):
    files = sorted(f for f in os.listdir(RSCAN_DIR) if f.endswith(".pt"))
    random.Random(seed).shuffle(files); files = files[:n]
    dlist = []
    for f in files:
        g = torch.load(os.path.join(RSCAN_DIR, f), weights_only=False)
        if g["edge_attr"].shape[1] != 22: continue
        if g["edge_label"].shape[1] == 12: g["edge_label"] = g["edge_label"][:, LEGACY_12_TO_10_COLS]
        if g["edge_label"].shape[1] != NUM_RELATIONS: continue
        dlist.append(Data(x=g["x"], edge_index=g["edge_index"], edge_attr=g["edge_attr"], y=g["edge_label"]))
    return dlist


class ASL(nn.Module):
    def __init__(self, pw, gn=4.0, gp=0.0, rw=None):
        super().__init__()
        self.register_buffer("pw", pw); self.gn=gn; self.gp=gp
        if rw is not None: self.register_buffer("rw", rw)
        else: self.rw = None
    def forward(self, lg, lb):
        mask = (lb >= 0).float(); lbc = lb.clamp(0)
        pr = torch.sigmoid(lg)
        lp = F.binary_cross_entropy_with_logits(lg, torch.ones_like(lg), pos_weight=self.pw, reduction="none")
        fp = (1-pr)**self.gp
        xn = torch.clamp(pr, min=0.0)
        ln = -torch.log(torch.clamp(1-xn, min=1e-8)); fn = xn**self.gn
        loss = torch.where(lbc >= 0.5, lp*fp, ln*fn) * mask
        if self.rw is not None: loss = loss * self.rw.unsqueeze(0)
        nv = mask.sum()
        return loss.sum() / nv if nv > 0 else loss.sum()


def ewc_loss(model, orig, lam):
    l = torch.tensor(0., device=next(model.parameters()).device)
    for n, p in model.named_parameters():
        if "head_contact" in n and n in orig:
            l = l + (p - orig[n].to(p.device)).pow(2).sum()
    return lam * l


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
        f1s.append(f1_score((al[v,r]>=0.5).numpy().astype(int), (ap[v,r]>=0.5).numpy(), zero_division=0))
    pos_rels = [r for r in range(NUM_RELATIONS) if (al[:,r]>=0.5).sum()>0]
    mf1 = float(np.mean([f1s[r] for r in pos_rels])) if pos_rels else 0.
    print(f"  Val macro(pos): {mf1:.4f}")
    for r in range(NUM_RELATIONS):
        pn = int((al[:,r]>=0.5).sum())
        if pn > 0: print(f"    {RELATION_NAMES[r]:20s} F1={f1s[r]:.3f} pos={pn}")
    return mf1


def tune_thresholds(model, loader, device):
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
        pv=ap[v,r]; lv=(al[v,r]>=0.5).astype(int)
        best_f1, best_t = 0., 0.5
        for t in np.arange(0.1, 0.91, 0.05):
            f1 = f1_score(lv, (pv>=t).astype(int), zero_division=0)
            if f1>best_f1: best_f1, best_t = f1, float(t)
        thr[r] = best_t
    return thr


def build_tabletop_graph(scene_dir):
    """Build graph for one tabletop scene — same pipeline as eval script."""
    ply = os.path.join(scene_dir, "splat.ply")
    if not os.path.exists(ply): return None, None
    gt = json.load(open(os.path.join(scene_dir, "ground_truth_relations.json")))
    cloud = load_gaussian_ply(ply)
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)
    n_hint = len(gt["objects"])
    objects, _ = gaussian_to_objects(cloud, target_min=max(2,n_hint-1), target_max=n_hint+1)
    if len(objects) < 2: return None, None

    all_mins = np.stack([o.bbox_min for o in objects])
    all_maxs = np.stack([o.bbox_max for o in objects])
    scene_min = all_mins.min(0); scene_max = all_maxs.max(0)
    scene_extent = np.maximum(scene_max - scene_min, 1e-6)
    obj_sizes = [np.maximum(o.size, 1e-6) for o in objects]
    scene_mean_diag = float(np.mean([np.linalg.norm(s) for s in obj_sizes]))
    scene_med_vol   = float(np.median([float(np.prod(s)) for s in obj_sizes]))
    czs = np.array([o.centroid[2] for o in objects])
    z_ranks = np.zeros(len(objects))
    if len(objects) > 1:
        for rk, oi in enumerate(np.argsort(czs)): z_ranks[oi] = rk/(len(objects)-1)

    x = np.stack([extract_gaussian_node_features(objects[i], scene_extent, scene_min,
        scene_mean_diag=scene_mean_diag, scene_median_volume=scene_med_vol,
        z_rank=float(z_ranks[i])) for i in range(len(objects))])
    ef_list, src_l, dst_l = [], [], []
    for i in range(len(objects)):
        for j in range(len(objects)):
            if i != j:
                ef_list.append(extract_3rscan_edge_features(objects[i], objects[j], scene_extent))
                src_l.append(i); dst_l.append(j)

    return (torch.tensor(x, dtype=torch.float32),
            torch.tensor([src_l, dst_l], dtype=torch.long),
            torch.tensor(np.stack(ef_list), dtype=torch.float32))


# ── Step 1: Fine-tuning ────────────────────────────────────────────────────────

def run_finetuning():
    print("="*65)
    print("STEP 1: Z-Inverted Fine-tuning (EWC=0.5, 70 epochs)")
    print("="*65)

    train_zinv, val_zinv = load_zinv(700, 100)
    rscan_mix = load_rscan(150)
    train_data = train_zinv + rscan_mix
    print(f"  Train: {len(train_zinv)} Z-inv + {len(rscan_mix)} 3RScan = {len(train_data)}")

    train_ld = DataLoader(train_data, batch_size=8, shuffle=True)
    val_ld   = DataLoader(val_zinv,   batch_size=8, shuffle=False)

    model = load_model_fresh()
    model = freeze_non_contact(model)
    model = model.to(DEVICE)

    orig_w = {n: p.detach().clone() for n, p in model.named_parameters() if "head_contact" in n}

    # Class weights from train data
    pos = torch.zeros(NUM_RELATIONS); valid = torch.zeros(NUM_RELATIONS)
    for d in train_data:
        for r in range(NUM_RELATIONS):
            col = d.y[:, r]; vm = col >= 0
            pos[r] += (col[vm] >= 0.5).float().sum(); valid[r] += vm.float().sum()
    pos = torch.clamp(pos, 1.); pw = torch.clamp((valid-pos)/pos, max=8.).to(DEVICE)

    rw = torch.ones(NUM_RELATIONS)
    rw[int(Relation.ON_TOP_OF)] = 4.0; rw[int(Relation.UNDER)] = 4.0
    rw = rw.to(DEVICE)

    criterion = ASL(pw, rw=rw).to(DEVICE)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=3e-5, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=70, eta_min=3e-5*0.05)

    best_f1, patience_cnt, best_epoch = 0., 0, 0

    for epoch in range(1, 71):
        model.train()
        total = 0.; nb = 0
        for batch in _tqdm(train_ld, desc=f"E{epoch:02d}", leave=False):
            batch = batch.to(DEVICE)
            logits  = model(batch.x, batch.edge_index, batch.edge_attr)
            bce     = criterion(logits, batch.y)
            met     = model.metric_reg()
            reg     = ewc_loss(model, orig_w, 0.5)
            loss    = bce + 1e-4*met + reg
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 2.)
            opt.step(); total += loss.item(); nb += 1
        sch.step()

        print(f"\n  Epoch {epoch:02d}  loss={total/max(nb,1):.4f}  lr={opt.param_groups[0]['lr']:.2e}")
        vf1 = eval_model(model, val_ld, DEVICE)

        if vf1 > best_f1:
            best_f1 = vf1; patience_cnt = 0; best_epoch = epoch
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  * Saved (best={best_f1:.4f} at epoch {epoch})")
        else:
            patience_cnt += 1
            if patience_cnt >= 25:
                print(f"  Early stop at epoch {epoch} (best={best_epoch})")
                break

    if not os.path.exists(SAVE_PATH):
        torch.save(model.state_dict(), SAVE_PATH)

    # Threshold tuning
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True))
    thr = tune_thresholds(model, val_ld, DEVICE)
    with open(THRESH_PATH, "w") as f:
        json.dump({str(k): v for k, v in thr.items()}, f, indent=2)

    print(f"\n  Fine-tuning done. Best val F1={best_f1:.4f} at epoch {best_epoch}")
    return best_epoch


# ── Step 2: TTBN — Test-Time BatchNorm adaptation ──────────────────────────────

def run_ttbn():
    """
    Adapt the contact head's BatchNorm statistics to the real tabletop distribution.

    Algorithm:
    1. Load adapted model
    2. Set ONLY contact head BN layers to train() mode (so they update running stats)
    3. Run all 8 tabletop scenes through the model (no gradient) for N passes
    4. BN stats shift toward tabletop distribution
    5. Save model with updated BN stats

    This is a form of source-free domain adaptation (BN layer calibration).
    """
    print("\n" + "="*65)
    print("STEP 2: TTBN — Contact-head BatchNorm adaptation to real tabletop")
    print("="*65)

    # Load adapted model
    full_state = torch.load("models/geokan_relation_v4.pt", weights_only=False, map_location="cpu")
    hd = full_state["node_encoder.0.weight"].shape[0]
    ef = full_state["conv1.lin_edge.weight"].shape[1]
    model = GeoKANRelationGNN(node_feat_dim=10, edge_feat_dim=ef, hidden_dim=hd).to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True))

    # Put entire model in eval mode, then selectively set BN in contact head to train
    model.eval()
    n_bn_adapted = 0
    for name, module in model.named_modules():
        if "head_contact" in name and isinstance(module, nn.BatchNorm1d):
            module.train()  # BN in train mode: updates running_mean and running_var
            module.momentum = 0.3  # higher momentum for faster adaptation
            n_bn_adapted += 1
    print(f"  Adapted {n_bn_adapted} BatchNorm layers in contact head to train() mode")

    # Build graphs for all 8 tabletop scenes
    tabletop_graphs = []
    for scene_name in SCENES:
        scene_dir = os.path.join(DATA_DIR, scene_name)
        result = build_tabletop_graph(scene_dir)
        if result[0] is None: continue
        x, ei, ea = result
        tabletop_graphs.append((x, ei, ea))
        print(f"  Loaded {scene_name}: {x.shape[0]} objects, {ei.shape[1]} edges")

    if not tabletop_graphs:
        print("  No tabletop graphs found — skipping TTBN")
        return

    # Run multiple passes to converge BN stats
    N_PASSES = 10
    print(f"  Running {N_PASSES} passes for BN stat convergence...")
    with torch.no_grad():
        for pass_idx in range(N_PASSES):
            for x, ei, ea in tabletop_graphs:
                _ = model(x.to(DEVICE), ei.to(DEVICE), ea.to(DEVICE))

    print("  TTBN complete — BN stats adapted to tabletop distribution")

    # Save model with updated BN stats (overwrite the fine-tuned model)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"  Model (with TTBN-adapted BN stats) saved to: {SAVE_PATH}")

    # Threshold tuning on Z-inv val (with updated model)
    train_zinv, val_zinv = load_zinv(700, 100)
    val_ld = DataLoader(val_zinv, batch_size=8, shuffle=False)
    model.eval()  # back to full eval mode for threshold tuning
    thr = tune_thresholds(model, val_ld, DEVICE)
    with open(THRESH_PATH, "w") as f:
        json.dump({str(k): v for k, v in thr.items()}, f, indent=2)
    print(f"  Thresholds re-tuned after TTBN and saved: {THRESH_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ft",   action="store_true", help="Skip fine-tuning (use existing adapted model)")
    parser.add_argument("--skip-ttbn", action="store_true", help="Skip TTBN step")
    args = parser.parse_args()

    if not args.skip_ft:
        run_finetuning()
    else:
        print("Skipping fine-tuning (--skip-ft)")

    if not args.skip_ttbn:
        run_ttbn()
    else:
        print("Skipping TTBN (--skip-ttbn)")

    print(f"\n{'='*65}")
    print(f"Adaptation complete.")
    print(f"  Model:      {SAVE_PATH}")
    print(f"  Thresholds: {THRESH_PATH}")
    print(f"{'='*65}")
    print("\nNow run: python eval_geokan_tabletop.py")
    print("(or use the patched eval to point to the adapted model)")


if __name__ == "__main__":
    main()
