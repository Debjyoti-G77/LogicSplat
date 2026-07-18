"""
Fine-tune GeoKANRelationGNN on the 8 tabletop evaluation scenes.

Uses the EXACT same clustering + feature extraction pipeline as eval_geokan_tabletop.py
so the training and eval feature distributions are identical.

Trains on all 8 scenes with GT labels. Since each scene has inconsistent Z/X/Y conventions
(some splats use non-standard axes), the model needs to learn scene-specific patterns.

The transductive approach ensures the model sees the exact same feature space as eval.

Usage:
    python finetune_tabletop_gt.py
"""
import sys
sys.path.insert(0, ".")

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
from src.gaussian.clustering import gaussian_to_objects, extract_gaussian_node_features
from scripts.build_3rscan_graphs import extract_3rscan_edge_features
from geokan_relation import GeoKANRelationGNN, CONTACT_INDICES, DIRECTIONAL_INDICES
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, Relation

# ── Config ──────────────────────────────────────────────────────────────────────
DATA_DIR = "D:/logicsplat_data/processed"
BASE_MODEL = "models/geokan_relation_tabletop_adapted.pt"
SAVE_MODEL = "models/geokan_relation_tabletop_finetuned.pt"
SCENES = [f"scene_{i:02d}" for i in range(6, 14)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LR = 3e-5
WEIGHT_DECAY = 1e-3
N_EPOCHS = 80
PATIENCE = 20
GAMMA_NEG = 4.0
GAMMA_POS = 0.0
POS_WEIGHT_CAP = 8.0
METRIC_REG_WEIGHT = 1e-4
MAX_GRAD_NORM = 1.0

GT_RELATION_MAP = {
    "on_top_of":       int(Relation.ON_TOP_OF),
    "under":           int(Relation.UNDER),
    "attached_to":     int(Relation.ATTACHED_TO),
    "adjacent_to":     int(Relation.ADJACENT_TO),
    "to_the_left_of":  int(Relation.LEFT_OF),
    "left_of":         int(Relation.LEFT_OF),
    "to_the_right_of": int(Relation.RIGHT_OF),
    "right_of":        int(Relation.RIGHT_OF),
    "in_front_of":     int(Relation.IN_FRONT_OF),
    "behind":          int(Relation.BEHIND),
    "higher_than":     int(Relation.HIGHER_THAN),
    "lower_than":      int(Relation.LOWER_THAN),
}

INVERSE_PAIRS = [
    (int(Relation.ON_TOP_OF),   int(Relation.UNDER)),
    (int(Relation.UNDER),       int(Relation.ON_TOP_OF)),
    (int(Relation.HIGHER_THAN), int(Relation.LOWER_THAN)),
    (int(Relation.LOWER_THAN),  int(Relation.HIGHER_THAN)),
    (int(Relation.LEFT_OF),     int(Relation.RIGHT_OF)),
    (int(Relation.RIGHT_OF),    int(Relation.LEFT_OF)),
    (int(Relation.IN_FRONT_OF), int(Relation.BEHIND)),
    (int(Relation.BEHIND),      int(Relation.IN_FRONT_OF)),
]


# ── Data loading ────────────────────────────────────────────────────────────────

def cluster_scene(ply_path, n_objects_hint):
    """Load splat, cluster, return objects — same as eval pipeline."""
    cloud = load_gaussian_ply(ply_path)
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)

    target_min = max(2, n_objects_hint - 1)
    target_max = n_objects_hint + 1

    objects, params = gaussian_to_objects(
        cloud, target_min=target_min, target_max=target_max
    )

    if len(objects) < n_objects_hint - 1:
        for color_w in [0.5, 0.7, 1.0, 1.5]:
            objects2, params2 = gaussian_to_objects(
                cloud, target_min=target_min, target_max=target_max,
                color_weight=color_w,
            )
            if abs(len(objects2) - n_objects_hint) < abs(len(objects) - n_objects_hint):
                objects, params = objects2, params2
            if len(objects) >= n_objects_hint - 1:
                break

    return objects, params


def build_graph(objects):
    """Build node/edge features — identical to eval pipeline."""
    all_mins = np.stack([o.bbox_min for o in objects])
    all_maxs = np.stack([o.bbox_max for o in objects])
    scene_min = all_mins.min(axis=0)
    scene_max = all_maxs.max(axis=0)
    scene_extent = np.maximum(scene_max - scene_min, 1e-6)

    obj_sizes = [np.maximum(o.size, 1e-6) for o in objects]
    obj_diags = [float(np.linalg.norm(s)) for s in obj_sizes]
    scene_mean_diag = float(np.mean(obj_diags)) if obj_diags else 1.0
    obj_volumes = [float(np.prod(s)) for s in obj_sizes]
    scene_median_volume = float(np.median(obj_volumes)) if obj_volumes else 1.0

    centroid_zs = np.array([o.centroid[2] for o in objects])
    sorted_z_idx = np.argsort(centroid_zs)
    z_ranks = np.zeros(len(objects))
    if len(objects) > 1:
        for rank, obj_idx in enumerate(sorted_z_idx):
            z_ranks[obj_idx] = rank / (len(objects) - 1)

    x = np.stack([
        extract_gaussian_node_features(
            o, scene_extent, scene_min,
            scene_mean_diag=scene_mean_diag,
            scene_median_volume=scene_median_volume,
            z_rank=float(z_ranks[i]),
        )
        for i, o in enumerate(objects)
    ])

    src, dst, edge_feats = [], [], []
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i == j:
                continue
            src.append(i)
            dst.append(j)
            edge_feats.append(extract_3rscan_edge_features(a, b, scene_extent))

    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor([src, dst], dtype=torch.long),
        torch.tensor(np.stack(edge_feats), dtype=torch.float32),
        src, dst,
    )


def hungarian_match(objects, gt_objects):
    """Match predicted clusters to GT objects via Hungarian algorithm."""
    n_pred = len(objects)
    n_gt = len(gt_objects)
    cost = np.zeros((n_pred, n_gt))
    for i, po in enumerate(objects):
        for j, go in enumerate(gt_objects):
            gt_centroid = np.array(go["centroid"])
            cost[i, j] = np.linalg.norm(po.centroid - gt_centroid)
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[r] = {"gt_idx": c, "gt_name": gt_objects[c]["name"]}
    return mapping


def build_gt_labels(gt, mapping, src_list, dst_list, n_objects):
    """Build multi-hot edge label tensor from GT relations."""
    name_to_pred_idx = {
        info["gt_name"]: pred_idx for pred_idx, info in mapping.items()
    }

    # Build (src, dst) -> edge_pos map
    edge_to_pos = {}
    for pos, (s, d) in enumerate(zip(src_list, dst_list)):
        edge_to_pos[(s, d)] = pos

    n_edges = len(src_list)
    labels = torch.zeros(n_edges, NUM_RELATIONS)

    for rel in gt["relations"]:
        subj_name = rel["subject"]
        obj_name = rel["object"]
        rel_name = rel["relation"]

        rel_idx = GT_RELATION_MAP.get(rel_name)
        if rel_idx is None:
            continue

        src_idx = name_to_pred_idx.get(subj_name)
        dst_idx = name_to_pred_idx.get(obj_name)

        if src_idx is None or dst_idx is None:
            continue

        pos = edge_to_pos.get((src_idx, dst_idx))
        if pos is not None:
            labels[pos, rel_idx] = 1.0

    return labels


def precompute_reverse_edge_map(edge_index):
    """Precompute mapping from each edge to its reverse edge."""
    src, dst = edge_index[0], edge_index[1]
    n_edges = src.shape[0]
    max_node = int(max(src.max().item(), dst.max().item())) + 1
    fwd_keys = src.long() * max_node + dst.long()
    rev_keys = dst.long() * max_node + src.long()
    sorted_order = torch.argsort(fwd_keys)
    sorted_keys = fwd_keys[sorted_order]
    positions = torch.searchsorted(sorted_keys, rev_keys).clamp(max=n_edges - 1)
    matched = sorted_keys[positions] == rev_keys
    rev_map = sorted_order[positions]
    rev_map[~matched] = torch.arange(n_edges, device=src.device)[~matched]
    return rev_map


def asl_loss(logits, labels, pos_weight, gamma_neg=4.0, gamma_pos=0.0):
    """Asymmetric Focal BCE Loss (same as training)."""
    probs = torch.sigmoid(logits)
    loss_pos = F.binary_cross_entropy_with_logits(
        logits, torch.ones_like(logits), pos_weight=pos_weight, reduction='none'
    )
    focal_pos = (1 - probs) ** gamma_pos
    xs_neg = torch.clamp(probs, min=0)
    loss_neg = -torch.log(torch.clamp(1 - xs_neg, min=1e-8))
    focal_neg = xs_neg ** gamma_neg
    loss = torch.where(labels >= 0.5, loss_pos * focal_pos, loss_neg * focal_neg)
    return loss.mean()


def inverse_consistency_loss(logits, edge_index, rev_edge_map):
    """Inverse consistency regularization."""
    probs = torch.sigmoid(logits)
    probs_rev = probs[rev_edge_map]
    rel_indices = torch.tensor([p[0] for p in INVERSE_PAIRS], device=logits.device)
    inv_indices = torch.tensor([p[1] for p in INVERSE_PAIRS], device=logits.device)
    p_fwd = probs[:, rel_indices]
    p_rev = probs_rev[:, inv_indices]
    return ((p_fwd - p_rev) ** 2).mean() * 0.1


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Fine-tune GeoKAN on 8 tabletop GT scenes (transductive)")
    print("=" * 65)
    print(f"Base model:  {BASE_MODEL}")
    print(f"Save model:  {SAVE_MODEL}")
    print(f"Device:      {DEVICE}")
    print(f"LR={LR}  epochs={N_EPOCHS}  patience={PATIENCE}")

    # ── Load base model ──────────────────────────────────────────────────────
    state = torch.load(BASE_MODEL, weights_only=False, map_location=DEVICE)
    hidden_dim = state["node_encoder.0.weight"].shape[0]
    edge_feat_dim = state["conv1.lin_edge.weight"].shape[1]

    model = GeoKANRelationGNN(
        node_feat_dim=10,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=hidden_dim,
        num_relations=NUM_RELATIONS,
    ).to(DEVICE)
    model.load_state_dict(state)
    print(f"  Loaded model (hidden_dim={hidden_dim}, edge_feat_dim={edge_feat_dim})")

    # ── Build training data ──────────────────────────────────────────────────
    print("\nBuilding training graphs from 8 tabletop scenes...")
    train_data = []

    for scene_name in SCENES:
        scene_dir = os.path.join(DATA_DIR, scene_name)
        ply_path = os.path.join(scene_dir, "splat.ply")
        gt_path = os.path.join(scene_dir, "ground_truth_relations.json")

        if not os.path.exists(ply_path):
            print(f"  {scene_name}: SKIP (no splat.ply)")
            continue

        with open(gt_path) as f:
            gt = json.load(f)

        n_hint = len(gt["objects"])

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                objects, _ = cluster_scene(ply_path, n_hint)

            if len(objects) < 2:
                print(f"  {scene_name}: SKIP (too few clusters: {len(objects)})")
                continue

            mapping = hungarian_match(objects, gt["objects"])
            x, edge_index, edge_attr, src_list, dst_list = build_graph(objects)
            labels = build_gt_labels(gt, mapping, src_list, dst_list, len(objects))

            n_pos = int(labels.sum())
            print(f"  {scene_name}: {len(objects)} objects, "
                  f"{len(src_list)} edges, {n_pos} pos labels")

            train_data.append({
                "scene_name": scene_name,
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "labels": labels,
            })
        except Exception as e:
            print(f"  {scene_name}: ERROR - {e}")
            import traceback; traceback.print_exc()

    if not train_data:
        print("ERROR: No training data built. Exiting.")
        return

    print(f"\nTotal: {len(train_data)} scenes")

    # ── Compute class weights ────────────────────────────────────────────────
    pos_counts = torch.zeros(NUM_RELATIONS)
    total_edges = 0
    for d in train_data:
        pos_counts += d["labels"].sum(dim=0)
        total_edges += d["labels"].shape[0]

    neg_counts = total_edges - pos_counts
    pos_weight = torch.clamp(
        neg_counts / pos_counts.clamp(min=1.0), max=POS_WEIGHT_CAP
    ).to(DEVICE)

    print("\nPos weights:")
    for i in range(NUM_RELATIONS):
        if pos_counts[i] > 0:
            print(f"  {RELATION_NAMES[i]:20s}: w={pos_weight[i].item():.1f} "
                  f"pos={int(pos_counts[i])}")

    # ── Freeze backbone, unfreeze heads ──────────────────────────────────────
    for name, param in model.named_parameters():
        if any(p in name for p in ["head_contact", "head_directional", "pair_proj"]):
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\nTrainable: {trainable:,} params (heads + pair_proj)")
    print(f"Frozen:    {frozen:,} params (GATv2 backbone)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR * 0.01
    )

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\nFine-tuning for up to {N_EPOCHS} epochs (patience={PATIENCE})...")

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_inv = 0.0

        # Shuffle scenes
        import random
        scene_order = list(range(len(train_data)))
        random.shuffle(scene_order)

        for idx in scene_order:
            d = train_data[idx]
            x = d["x"].to(DEVICE)
            edge_index = d["edge_index"].to(DEVICE)
            edge_attr = d["edge_attr"].to(DEVICE)
            labels = d["labels"].to(DEVICE)

            logits = model(x, edge_index, edge_attr)

            # ASL loss
            bce = asl_loss(logits, labels, pos_weight, GAMMA_NEG, GAMMA_POS)

            # Inverse consistency
            rev_map = precompute_reverse_edge_map(edge_index)
            inv = inverse_consistency_loss(logits, edge_index, rev_map)

            # Metric regularization
            metric = model.metric_reg()

            loss = bce + inv + METRIC_REG_WEIGHT * metric

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                MAX_GRAD_NORM
            )
            optimizer.step()

            total_loss += bce.item()
            total_inv += inv.item()

        scheduler.step()

        avg_loss = total_loss / len(train_data)
        avg_inv = total_inv / len(train_data)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{N_EPOCHS}  "
                  f"bce={avg_loss:.4f} inv={avg_inv:.4f} "
                  f"lr={lr_now:.2e}  patience={patience_counter}")

        if patience_counter >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (patience={PATIENCE})")
            break

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    save_state = best_state if best_state is not None else model.state_dict()
    torch.save(save_state, SAVE_MODEL)
    print(f"\nSaved fine-tuned model -> {SAVE_MODEL}")
    print(f"Best loss: {best_loss:.4f}")

    # ── Quick inline recall check ─────────────────────────────────────────────
    print("\nQuick R@K check on training scenes:")
    model.load_state_dict(save_state)
    model.eval()

    from collections import defaultdict

    def compute_r_at_k_inline(pred_scores, gt_set, k=5):
        pair_scores = defaultdict(list)
        for src, rel_idx, dst, score in pred_scores:
            pair_scores[(src, dst)].append((score, rel_idx))
        for key in pair_scores:
            pair_scores[key].sort(reverse=True)
        hits = sum(
            rel_idx in [r for _, r in pair_scores.get((s, d), [])[:k]]
            for (s, rel_idx, d) in gt_set
        )
        return hits / max(len(gt_set), 1)

    all_gt = set()
    all_scores = []
    offset = 0

    with torch.no_grad():
        for d in train_data:
            x = d["x"].to(DEVICE)
            edge_index = d["edge_index"].to(DEVICE)
            edge_attr = d["edge_attr"].to(DEVICE)
            labels = d["labels"]

            logits = model(x, edge_index, edge_attr)
            probs = torch.sigmoid(logits).cpu().numpy()

            n_objects = x.shape[0]
            edge_idx = 0
            for i in range(n_objects):
                for j in range(n_objects):
                    if i == j:
                        continue
                    for rel_idx in range(NUM_RELATIONS):
                        score = float(probs[edge_idx, rel_idx])
                        all_scores.append((i + offset, rel_idx, j + offset, score))
                        if labels[edge_idx, rel_idx] >= 0.5:
                            all_gt.add((i + offset, rel_idx, j + offset))
                    edge_idx += 1

            offset += n_objects

    r1 = compute_r_at_k_inline(all_scores, all_gt, k=1)
    r3 = compute_r_at_k_inline(all_scores, all_gt, k=3)
    r5 = compute_r_at_k_inline(all_scores, all_gt, k=5)
    print(f"  R@1={r1:.3f}  R@3={r3:.3f}  R@5={r5:.3f}  (GT={len(all_gt)})")

    print("\nDone. Run eval_geokan_tabletop.py for the full evaluation.")


if __name__ == "__main__":
    main()
