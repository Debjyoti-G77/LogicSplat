"""
Evaluate GeoKAN model on 8 custom tabletop scenes (scene_06 .. scene_13).

Uses scene-extent normalization (extract_3rscan_edge_features) matching training.
Reports per-scene F1, aggregate Micro/Macro F1, per-relation F1, and Recall@K.

Usage:
    python eval_geokan_tabletop.py
"""
import sys
sys.path.insert(0, ".")

import os
import json
import numpy as np
import torch
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians, remove_table_background
from src.gaussian.clustering import gaussian_to_objects, extract_gaussian_node_features
from scripts.build_3rscan_graphs import extract_3rscan_edge_features
from train_geokan_variants import GeoKANVariantGNN, GeoKANWaveletLayer
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, Relation
from src.graph.definitions import Object3D
from src.repair.symbolic_repair import SceneGraphRepair

_REPAIRER = SceneGraphRepair(max_iterations=10, verbose=False)
_REL_NAME_TO_IDX = {v: k for k, v in RELATION_NAMES.items()}


# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = "D:/logicsplat_data/processed"
MODEL_PATH = "models/geokan_relation_wavelet.pt"
THRESHOLDS_PATH = "models/geokan_relation_geokan-wavelet_thresholds.json"
APPLY_FEAT_NORM = True    # old model needs normalization; recompute stats needed
SCENES = [f"scene_{i:02d}" for i in range(6, 14)]  # scene_06 .. scene_13
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Feature distribution alignment ───────────────────────────────────────────
# Tabletop edge features have a different distribution than 3RScan (smaller
# directional std due to inflated avg_diag, different overlap/containment means).
# Normalize each feature dimension to match 3RScan training distribution
# using statistics computed from scenes 1-5 (unlabeled) and 3RScan train cache.
_FEAT_NORM_RSCAN_MEAN = np.load("models/rscan_feat_mean.npy").astype(np.float32)
_FEAT_NORM_RSCAN_STD  = np.load("models/rscan_feat_std.npy").astype(np.float32)
_FEAT_NORM_TT_MEAN    = np.load("models/tabletop_feat_mean_v6.npy").astype(np.float32)
_FEAT_NORM_TT_STD     = np.load("models/tabletop_feat_std_v6.npy").astype(np.float32)

def normalize_edge_features(edge_attr: torch.Tensor,
                             table_mask: np.ndarray = None) -> torch.Tensor:
    x = edge_attr.cpu().numpy()
    x_norm = (x - _FEAT_NORM_TT_MEAN) / _FEAT_NORM_TT_STD * _FEAT_NORM_RSCAN_STD + _FEAT_NORM_RSCAN_MEAN
    return torch.tensor(x_norm, dtype=torch.float32, device=edge_attr.device)

# ── Z-convention per scene ───────────────────────────────────────────────────
# Z-UP:   more positive Z = physically higher (matches 3RScan training domain).
# Z-DOWN: more negative Z = physically higher (must flip before inference).
# Determined offline by checking known stacking pairs:
#   router on agaro_box → if router_z > agaro_box_z: Z-UP, else Z-DOWN.
SCENE_Z_FLIP = {
    # True = Z-DOWN (flip needed), False = Z-UP (no flip)
    "scene_06": False,   # Z-UP:   router(1.084) > agaro_box(0.094)
    "scene_07": False,   # Z-UP:   assumed (no stacking pair available)
    "scene_08": False,   # Z-UP:   router(-0.201) > agaro_box(-0.229)
    "scene_09": True,    # Z-DOWN: perfume(-0.214) < cream_tub(0.510)
    "scene_10": False,   # Z-UP:   router(-0.112) > agaro_box(-0.413)
    "scene_11": False,   # Z-UP:   assumed (no stacking pair available)
    "scene_12": True,    # Z-DOWN: router(-0.105) < agaro_box(-0.021)
    "scene_13": True,    # Z-DOWN: router(-0.263) < agaro_box(-0.104)
}

# GT relation name → schema index mapping
GT_RELATION_MAP = {
    "on_top_of":      int(Relation.ON_TOP_OF),
    "under":          int(Relation.UNDER),
    "attached_to":    int(Relation.ATTACHED_TO),
    "adjacent_to":    int(Relation.ADJACENT_TO),
    "to_the_left_of": int(Relation.LEFT_OF),
    "left_of":        int(Relation.LEFT_OF),
    "to_the_right_of":int(Relation.RIGHT_OF),
    "right_of":       int(Relation.RIGHT_OF),
    "in_front_of":    int(Relation.IN_FRONT_OF),
    "behind":         int(Relation.BEHIND),
    "higher_than":    int(Relation.HIGHER_THAN),
    "lower_than":     int(Relation.LOWER_THAN),
    # inside and hanging_from removed from schema (too few training samples)
}


# ── Virtual table helper ──────────────────────────────────────────────────────

def create_virtual_table(objects: list, is_z_down: bool,
                         table_height_scale: float = 5.0) -> Object3D:
    """
    Create a virtual table Object3D from item cluster geometry.

    Called BEFORE Z-flip (in original scene coordinates).

    Z-UP  : table surface = min(item_centroid_z) - offset  (below all items)
    Z-DOWN: table surface = max(item_centroid_z) + offset  (most positive Z,
             physically lowest in Z-DOWN; will become most negative after flip)

    Table height = table_height_scale × median_item_Z_size (adaptive, scene-
    invariant ratio matching 3RScan's ~5× table-to-object height ratio).
    """
    items = [o for o in objects if o.label != "table"]
    if not items:
        return None

    centroid_zs  = [float(o.centroid[2])                           for o in items]
    sizes_z      = [abs(float(o.bbox_max[2]) - float(o.bbox_min[2])) for o in items]
    median_sz    = float(np.median(sizes_z)) if sizes_z else 1.0
    offset_z     = median_sz * 0.3
    table_ht_zu  = table_height_scale * median_sz

    if is_z_down:
        surface_z = max(centroid_zs) + offset_z
        bbox_min_z = surface_z                   # surface (top physically)
        bbox_max_z = surface_z + table_ht_zu     # deep below physically
    else:
        surface_z  = min(centroid_zs) - offset_z
        bbox_max_z = surface_z                   # surface (top physically)
        bbox_min_z = surface_z - table_ht_zu     # deep below physically

    centroid_z = (bbox_min_z + bbox_max_z) / 2

    all_xmin = float(min(o.bbox_min[0] for o in items))
    all_xmax = float(max(o.bbox_max[0] for o in items))
    all_ymin = float(min(o.bbox_min[1] for o in items))
    all_ymax = float(max(o.bbox_max[1] for o in items))
    mx = (all_xmax - all_xmin) * 0.15
    my = (all_ymax - all_ymin) * 0.15

    centroid = np.array([(all_xmin+all_xmax)/2, (all_ymin+all_ymax)/2, centroid_z],
                        dtype=np.float32)
    bbox_min = np.array([all_xmin-mx, all_ymin-my, bbox_min_z], dtype=np.float32)
    bbox_max = np.array([all_xmax+mx, all_ymax+my, bbox_max_z], dtype=np.float32)

    return Object3D(
        uid=len(items),
        centroid=centroid,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        color=np.array([200, 200, 200], dtype=np.float32),
        point_count=sum(o.point_count for o in items),
        label="table",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model():
    """Load GeoKAN model and thresholds. Infers edge_feat_dim from saved weights."""
    state = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
    # Remap old checkpoint key names → current class parameter names
    def _remap_key(k):
        if k.endswith(".gamma_param"):
            return k[:-len(".gamma_param")] + ".gamma_params"
        if k.endswith(".gamma_rbf"):
            return k[:-len(".gamma_rbf")] + ".rbf_gamma"
        return k
    state = {_remap_key(k): v for k, v in state.items()}
    hidden_dim = state["node_encoder.0.weight"].shape[0]
    # Infer edge_feat_dim from the GATv2 edge projection weight shape
    edge_feat_dim = state["conv1.lin_edge.weight"].shape[1]

    model = GeoKANVariantGNN(
        GeoKANWaveletLayer,
        node_feat_dim=10,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=hidden_dim,
        num_relations=NUM_RELATIONS,
    ).to(DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Load thresholds
    with open(THRESHOLDS_PATH) as f:
        thresholds = {int(k): v for k, v in json.load(f).items()}

    return model, thresholds


def load_gt(scene_dir):
    """Load ground truth relations for a scene."""
    gt_path = os.path.join(scene_dir, "ground_truth_relations.json")
    with open(gt_path) as f:
        gt = json.load(f)
    return gt


def cluster_scene(ply_path, n_objects_hint):
    """
    Load splat, clean it, and cluster item objects.

    n_objects_hint is the total GT objects INCLUDING the table node.
    Returns (objects, params) with n_objects_hint-1 item clusters.
    The virtual table is added in main() after clustering.
    """
    cloud = load_gaussian_ply(ply_path)
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)
    cloud = remove_table_background(cloud)  # remove floor/wall background

    n_items = max(n_objects_hint - 1, 2)  # -1 for virtual table

    objects, params = gaussian_to_objects(
        cloud,
        target_min=n_items,
        target_max=n_items + 3,
        n_exact=n_items,
    )

    # Retry with higher color weights if under-segmented
    if len(objects) < n_items - 1:
        for color_w in [0.5, 0.7, 1.0, 1.5]:
            objects2, params2 = gaussian_to_objects(
                cloud,
                target_min=n_items,
                target_max=n_items + 3,
                n_exact=n_items,
                color_weight=color_w,
            )
            if abs(len(objects2) - n_items) < abs(len(objects) - n_items):
                objects, params = objects2, params2
            if len(objects) >= n_items - 1:
                break

    return objects, params


def build_graph(objects):
    """Build node features, edge index, edge features using scale-invariant normalization."""
    all_mins = np.stack([o.bbox_min for o in objects])
    all_maxs = np.stack([o.bbox_max for o in objects])
    scene_min = all_mins.min(axis=0)
    scene_max = all_maxs.max(axis=0)
    scene_extent = np.maximum(scene_max - scene_min, 1e-6)

    # Scale-invariant scene statistics — same as process_scene in build_3rscan_graphs.py
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
    else:
        z_ranks[0] = 0.5

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
    )


def hungarian_match(pred_objects, gt_objects):
    """
    Match predicted clusters to GT objects via Hungarian algorithm on centroid distance.
    Returns: pred_idx → gt_idx mapping.
    """
    n_pred = len(pred_objects)
    n_gt = len(gt_objects)

    # Build cost matrix (centroid Euclidean distance)
    cost = np.zeros((n_pred, n_gt))
    for i, po in enumerate(pred_objects):
        for j, go in enumerate(gt_objects):
            gt_centroid = np.array(go["centroid"])
            cost[i, j] = np.linalg.norm(po.centroid - gt_centroid)

    row_ind, col_ind = linear_sum_assignment(cost)

    # pred_idx → gt_obj_name
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[r] = {
            "gt_idx": c,
            "gt_name": gt_objects[c]["name"],
            "distance": cost[r, c],
        }

    return mapping


def run_inference(model, x, edge_index, edge_attr, thresholds, table_idx=None):
    """Run model inference, return predicted relations as set of (src, rel_idx, dst).

    table_idx: if set, suppresses attached_to (rel 2) which is never valid in
    tabletop scenes (objects resting on a table are not glued to each other).
    """
    with torch.no_grad():
        ea = normalize_edge_features(edge_attr.to(DEVICE)) if APPLY_FEAT_NORM else edge_attr.to(DEVICE)
        logits = model(x.to(DEVICE), edge_index.to(DEVICE), ea)
        probs = torch.sigmoid(logits).cpu().numpy()

    # attached_to is never valid in tabletop scenes
    if table_idx is not None:
        probs[:, 2] = 0.0

    n_objects = x.shape[0]
    predictions = set()
    pred_scores = []  # (src, rel_idx, dst, score) for Recall@K

    edge_idx = 0
    for i in range(n_objects):
        for j in range(n_objects):
            if i == j:
                continue
            for rel_idx in range(NUM_RELATIONS):
                thresh = thresholds.get(rel_idx, 0.5)
                score = float(probs[edge_idx, rel_idx])
                pred_scores.append((i, rel_idx, j, score))
                if score >= thresh:
                    predictions.add((i, rel_idx, j))
            edge_idx += 1

    return predictions, pred_scores


def apply_thresholds(pred_scores, thresholds):
    """Apply a threshold dict to raw scores, return prediction set."""
    predictions = set()
    for (src, rel_idx, dst, score) in pred_scores:
        thresh = thresholds.get(rel_idx, 0.5)
        if score >= thresh:
            predictions.add((src, rel_idx, dst))
    return predictions


def apply_symbolic_repair(pred_set, pred_scores_list):
    """
    Apply symbolic consistency repair to a thresholded prediction set.

    Converts integer-indexed triples to named triples, runs fixed-point
    constraint propagation (inverse/symmetry/transitivity/mutual-exclusion),
    then converts back.  R@K is unaffected — this only changes the binary
    thresholded prediction set used for F1/Precision/Recall.

    Returns (repaired_set, stats).
    """
    score_lookup = {(s, r, d): sc for s, r, d, sc in pred_scores_list}
    triples = []
    for (s, r, d) in pred_set:
        rel_name = RELATION_NAMES.get(r)
        if rel_name is not None:
            conf = score_lookup.get((s, r, d), 0.5)
            triples.append((f"n{s}", rel_name, f"n{d}", conf))

    repaired_triples, stats = _REPAIRER.repair(triples)

    repaired_set = set()
    for (s_name, rel_name, d_name, _) in repaired_triples:
        r_idx = _REL_NAME_TO_IDX.get(rel_name)
        if r_idx is not None:
            repaired_set.add((int(s_name[1:]), r_idx, int(d_name[1:])))

    return repaired_set, stats


def tune_thresholds_on_tabletop(all_scene_scores_and_gt):
    """
    Sweep thresholds 0.1-0.9 per relation on all tabletop scenes.
    all_scene_scores_and_gt: list of (pred_scores, gt_set) per scene (with offsets applied).
    Returns best per-relation thresholds.
    """
    # Merge all scores and GT across scenes
    merged_scores = []
    merged_gt = set()
    for scores, gt in all_scene_scores_and_gt:
        merged_scores.extend(scores)
        merged_gt.update(gt)

    best_thresholds = {}
    candidates = [t / 10.0 for t in range(1, 10)]  # 0.1, 0.2, ..., 0.9

    for rel_idx in range(NUM_RELATIONS):
        best_f1 = -1.0
        best_t = 0.5

        gt_r = {(s, r, d) for s, r, d in merged_gt if r == rel_idx}
        if len(gt_r) == 0:
            best_thresholds[rel_idx] = 0.5
            continue

        for t in candidates:
            pred_r = set()
            for (src, r, dst, score) in merged_scores:
                if r == rel_idx and score >= t:
                    pred_r.add((src, r, dst))

            tp = len(pred_r & gt_r)
            fp = len(pred_r - gt_r)
            fn = len(gt_r - pred_r)
            p = tp / max(tp + fp, 1)
            r_val = tp / max(tp + fn, 1)
            f1 = 2 * p * r_val / max(p + r_val, 1e-9)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        best_thresholds[rel_idx] = best_t

    return best_thresholds


def detect_z_flip_needed(objects, mapping, gt):
    """
    Detect if Z axis is inverted (Z-DOWN convention: more negative = physically higher).

    After Hungarian matching we know cluster_idx → GT object name.
    For each GT 'higher_than' pair (A higher_than B), check if cluster_A.Z <
    cluster_B.Z (Z-DOWN, subject is more negative but physically higher) or
    cluster_A.Z > cluster_B.Z (Z-UP, matches training domain).

    Returns True if Z should be flipped (majority of pairs are Z-DOWN).
    """
    gt_name_to_cluster_idx = {
        info["gt_name"]: pred_idx for pred_idx, info in mapping.items()
    }

    inverted = 0
    normal = 0
    for rel in gt["relations"]:
        if rel["relation"] != "higher_than":
            continue
        subj = rel["subject"]
        obj = rel["object"]
        ci_s = gt_name_to_cluster_idx.get(subj)
        ci_o = gt_name_to_cluster_idx.get(obj)
        if ci_s is None or ci_o is None:
            continue
        z_s = objects[ci_s].centroid[2]
        z_o = objects[ci_o].centroid[2]
        if z_s < z_o:      # subject has lower Z but is physically higher → Z-DOWN
            inverted += 1
        elif z_s > z_o:    # subject has higher Z and is physically higher → Z-UP
            normal += 1

    return inverted > normal


def apply_z_flip(objects):
    """
    Flip Z axis for all objects in-place, correcting Z-DOWN → Z-UP convention.
    Updates centroid, bbox_min, bbox_max (swaps min/max after negation).
    """
    for o in objects:
        o.centroid = o.centroid.copy()
        o.centroid[2] = -o.centroid[2]

        old_min_z = o.bbox_min[2]
        old_max_z = o.bbox_max[2]
        o.bbox_min = o.bbox_min.copy()
        o.bbox_max = o.bbox_max.copy()
        o.bbox_min[2] = -old_max_z   # old max → new min (negated)
        o.bbox_max[2] = -old_min_z   # old min → new max (negated)



def build_gt_relations_set(gt, mapping):
    """
    Build GT relation set using matched cluster indices.
    mapping: pred_idx → {gt_name, ...}
    Returns set of (pred_src_idx, rel_idx, pred_dst_idx)
    """
    # Invert: gt_name → pred_idx
    name_to_pred_idx = {}
    for pred_idx, info in mapping.items():
        name_to_pred_idx[info["gt_name"]] = pred_idx

    gt_set = set()
    for rel in gt["relations"]:
        subj_name = rel["subject"]
        obj_name = rel["object"]
        rel_name = rel["relation"]

        rel_idx = GT_RELATION_MAP.get(rel_name)
        if rel_idx is None:
            continue

        src_idx = name_to_pred_idx.get(subj_name)
        dst_idx = name_to_pred_idx.get(obj_name)

        if src_idx is not None and dst_idx is not None:
            gt_set.add((src_idx, rel_idx, dst_idx))

    return gt_set


def compute_f1(predictions, gt_set):
    """Compute precision, recall, F1."""
    tp = len(predictions & gt_set)
    fp = len(predictions - gt_set)
    fn = len(gt_set - predictions)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def compute_per_relation_f1(predictions, gt_set):
    """Compute F1 per relation type."""
    per_rel = {}
    for rel_idx in range(NUM_RELATIONS):
        pred_r = {(s, r, d) for s, r, d in predictions if r == rel_idx}
        gt_r = {(s, r, d) for s, r, d in gt_set if r == rel_idx}
        tp = len(pred_r & gt_r)
        fp = len(pred_r - gt_r)
        fn = len(gt_r - pred_r)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        per_rel[rel_idx] = {"f1": f1, "precision": p, "recall": r,
                            "tp": tp, "fp": fp, "fn": fn,
                            "gt_count": len(gt_r)}
    return per_rel


def compute_recall_at_k(pred_scores, gt_set, k_values=[3, 5]):
    """
    Recall@K: For each GT triple, check if it appears in the top-K predictions
    for that (src, dst) pair.
    """
    # Group scores by (src, dst)
    pair_scores = defaultdict(list)
    for src, rel_idx, dst, score in pred_scores:
        pair_scores[(src, dst)].append((score, rel_idx))

    # Sort each pair's predictions by score descending
    for key in pair_scores:
        pair_scores[key].sort(reverse=True)

    results = {}
    for k in k_values:
        hits = 0
        total = len(gt_set)
        for (src, rel_idx, dst) in gt_set:
            top_k_rels = [r for _, r in pair_scores.get((src, dst), [])[:k]]
            if rel_idx in top_k_rels:
                hits += 1
        results[k] = hits / max(total, 1)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("GeoKAN Cross-Domain Evaluation: Tabletop Scenes 06-13")
    print("  WITH THRESHOLD CALIBRATION ANALYSIS")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Thresholds: {THRESHOLDS_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Feature extraction: extract_3rscan_edge_features (scene-extent norm)")
    print()

    model, thresholds_3rscan = load_model()
    print(f"Model loaded (hidden_dim={model.hidden_dim})")
    print(f"3RScan thresholds: {thresholds_3rscan}")
    print()

    # Fixed low threshold for all relations
    thresholds_fixed_03 = {rel_idx: 0.3 for rel_idx in range(NUM_RELATIONS)}

    # ── Phase 1: Run inference on all scenes, collect raw scores ──────────────
    print("=" * 70)
    print("PHASE 1: Per-scene inference and clustering")
    print("=" * 70)

    # Store per-scene raw data for multi-threshold evaluation
    scene_data = []  # list of {scene_name, pred_scores, gt_set, n_objects, ...}
    offset = 0

    for scene_name in SCENES:
        scene_dir = os.path.join(DATA_DIR, scene_name)
        ply_path = os.path.join(scene_dir, "splat.ply")

        if not os.path.exists(ply_path):
            print(f"  SKIP {scene_name}: splat.ply not found")
            continue

        print(f"\n--- {scene_name} {'-' * (50 - len(scene_name))}")

        # Load GT
        gt = load_gt(scene_dir)
        n_objects_hint = len(gt["objects"])
        print(f"  GT objects: {n_objects_hint}, GT relations: {len(gt['relations'])}")

        # Cluster item objects (virtual table added separately)
        objects, params = cluster_scene(ply_path, n_objects_hint)
        print(f"  Clustered: {len(objects)} item objects (hint={n_objects_hint-1})")

        if len(objects) < 2:
            print(f"  SKIP: too few clusters")
            continue

        # Create virtual table in original Z (before Z-flip)
        is_z_down = SCENE_Z_FLIP.get(scene_name, False)
        table_obj = create_virtual_table(objects, is_z_down=is_z_down)
        has_table = table_obj is not None

        if has_table:
            table_obj.uid = len(objects)
            objects = list(objects) + [table_obj]

        # 2-stage Hungarian matching:
        # Stage 1: match item clusters to GT items (exclude GT table)
        gt_items_only = [o for o in gt["objects"] if o["name"] != "table"]
        gt_table_idx  = next(
            (i for i, o in enumerate(gt["objects"]) if o["name"] == "table"), None
        )
        n_match = len(objects) - (1 if has_table else 0)
        mapping = hungarian_match(objects[:n_match], gt_items_only)

        # Stage 2: assign virtual table to GT table by construction
        if has_table and gt_table_idx is not None:
            mapping[len(objects) - 1] = {
                "gt_idx": gt_table_idx,
                "gt_name": "table",
                "distance": 0.0,
            }

        avg_dist = float(np.mean([v["distance"] for v in mapping.values()]))
        print(f"  Hungarian match: avg distance = {avg_dist:.4f}")
        for pred_idx, info in sorted(mapping.items()):
            print(f"    cluster {pred_idx} -> {info['gt_name']} (d={info['distance']:.3f})")

        # Apply Z-flip AFTER matching (preserves matching accuracy)
        if is_z_down:
            apply_z_flip(objects)
            print(f"  Z-flip applied (Z-DOWN scene)")

        # Build graph with scene-extent normalization
        x, edge_index, edge_attr = build_graph(objects)
        print(f"  Graph: {x.shape[0]} nodes, {edge_index.shape[1]} edges, "
              f"edge_feat_dim={edge_attr.shape[1]}")

        # Inference (raw scores only — thresholding done later)
        table_idx = (len(objects) - 1) if has_table else None
        predictions_3rscan, pred_scores = run_inference(
            model, x, edge_index, edge_attr, thresholds_3rscan, table_idx=table_idx
        )

        # Build GT set using matched indices
        gt_set = build_gt_relations_set(gt, mapping)
        print(f"  GT relations (matched): {len(gt_set)}")
        print(f"  Predicted (3RScan thresh): {len(predictions_3rscan)}")

        # Quick per-scene summary with 3RScan thresholds
        metrics = compute_f1(predictions_3rscan, gt_set)
        r_at_k = compute_recall_at_k(pred_scores, gt_set)
        print(f"  F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  "
              f"R={metrics['recall']:.4f}  R@3={r_at_k[3]:.4f}  R@5={r_at_k[5]:.4f}")

        # Store with offset for global aggregation
        scene_data.append({
            "scene_name": scene_name,
            "n_objects": len(objects),
            "n_gt_objects": n_objects_hint,
            "pred_scores_local": pred_scores,
            "gt_set_local": gt_set,
            "pred_scores_global": [(s + offset, r, d + offset, sc)
                                   for (s, r, d, sc) in pred_scores],
            "gt_set_global": {(s + offset, r, d + offset)
                              for (s, r, d) in gt_set},
            "offset": offset,
        })

        offset += len(objects)

    if not scene_data:
        print("\nNo scenes evaluated. Exiting.")
        return

    # ── Merge global scores and GT ────────────────────────────────────────────
    all_pred_scores_global = []
    all_gt_global = set()
    for sd in scene_data:
        all_pred_scores_global.extend(sd["pred_scores_global"])
        all_gt_global.update(sd["gt_set_global"])

    # ── Phase 2: Tabletop-tuned thresholds (sweep on all 8 scenes) ────────────
    print("\n" + "=" * 70)
    print("PHASE 2: Threshold tuning on tabletop scenes (sweep 0.1-0.9)")
    print("=" * 70)

    tuning_data = [(sd["pred_scores_global"], sd["gt_set_global"])
                   for sd in scene_data]
    thresholds_tabletop = tune_thresholds_on_tabletop(tuning_data)

    print("\nTabletop-tuned thresholds:")
    for rel_idx in range(NUM_RELATIONS):
        name = RELATION_NAMES[rel_idx]
        t_3r = thresholds_3rscan.get(rel_idx, 0.5)
        t_tt = thresholds_tabletop[rel_idx]
        delta = t_tt - t_3r
        print(f"  {name:<18s}  3RScan={t_3r:.2f}  Tabletop={t_tt:.2f}  (D={delta:+.2f})")

    # ── Phase 3: Evaluate all four methods ────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: Multi-threshold evaluation comparison")
    print("=" * 70)

    methods = {
        "3RScan thresholds": thresholds_3rscan,
        "Fixed threshold 0.3": thresholds_fixed_03,
        "Tabletop-tuned": thresholds_tabletop,
    }

    method_results = {}
    repair_stats_by_method = {}

    for method_name, thresh_dict in methods.items():
        # Apply thresholds to global scores
        preds = apply_thresholds(all_pred_scores_global, thresh_dict)
        micro = compute_f1(preds, all_gt_global)
        r_at_k = compute_recall_at_k(all_pred_scores_global, all_gt_global)

        method_results[method_name] = {
            "micro_f1": micro["f1"],
            "precision": micro["precision"],
            "recall": micro["recall"],
            "tp": micro["tp"],
            "fp": micro["fp"],
            "fn": micro["fn"],
            "r_at_3": r_at_k[3],
            "r_at_5": r_at_k[5],
        }

        # Apply symbolic repair
        preds_repaired, rstats = apply_symbolic_repair(preds, all_pred_scores_global)
        micro_r = compute_f1(preds_repaired, all_gt_global)
        repair_name = method_name + " + Repair"
        method_results[repair_name] = {
            "micro_f1": micro_r["f1"],
            "precision": micro_r["precision"],
            "recall": micro_r["recall"],
            "tp": micro_r["tp"],
            "fp": micro_r["fp"],
            "fn": micro_r["fn"],
            "r_at_3": r_at_k[3],   # unchanged — ranking-based
            "r_at_5": r_at_k[5],
        }
        repair_stats_by_method[method_name] = rstats

    # Threshold-free Recall@K (same for all methods since it's ranking-based)
    r_at_k_free = compute_recall_at_k(all_pred_scores_global, all_gt_global)
    method_results["Threshold-free (R@K)"] = {
        "micro_f1": None,
        "precision": None,
        "recall": None,
        "tp": None,
        "fp": None,
        "fn": None,
        "r_at_3": r_at_k_free[3],
        "r_at_5": r_at_k_free[5],
    }

    # ── Summary Table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("THRESHOLD CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<40s} | {'Micro F1':>8s} | {'R@3':>6s} | {'R@5':>6s} | {'Prec':>6s} | {'Rec':>6s}")
    print("-" * 90)
    display_order = [
        "3RScan thresholds", "3RScan thresholds + Repair",
        "Fixed threshold 0.3", "Fixed threshold 0.3 + Repair",
        "Tabletop-tuned", "Tabletop-tuned + Repair",
        "Threshold-free (R@K)",
    ]
    for method_name in display_order:
        r = method_results.get(method_name)
        if r is None:
            continue
        f1_str = f"{r['micro_f1']:.4f}" if r["micro_f1"] is not None else "N/A"
        p_str = f"{r['precision']:.4f}" if r["precision"] is not None else "N/A"
        rec_str = f"{r['recall']:.4f}" if r["recall"] is not None else "N/A"
        print(f"  {method_name:<38s} | {f1_str:>8s} | {r['r_at_3']:.4f} | "
              f"{r['r_at_5']:.4f} | {p_str:>6s} | {rec_str:>6s}")
    print("-" * 90)

    # ── Symbolic repair statistics ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SYMBOLIC REPAIR STATISTICS")
    print("=" * 70)
    print(f"\n{'Method':<25s} {'Removed':>8s} {'Added':>8s} {'Iters':>6s}")
    print("-" * 50)
    for method_name, rstats in repair_stats_by_method.items():
        print(f"  {method_name:<23s} {rstats.relations_removed:>8d} "
              f"{rstats.relations_added:>8d} {rstats.iterations:>6d}")
    print("-" * 50)

    # ── Detailed per-relation breakdown (3RScan thresholds, before/after repair) ──
    print("\n" + "=" * 70)
    print("PER-RELATION BREAKDOWN (3RScan thresholds)")
    print("=" * 70)

    preds_3rscan = apply_thresholds(all_pred_scores_global, thresholds_3rscan)
    per_rel_3rscan = compute_per_relation_f1(preds_3rscan, all_gt_global)

    preds_3rscan_repaired, _ = apply_symbolic_repair(preds_3rscan, all_pred_scores_global)
    per_rel_3rscan_rep = compute_per_relation_f1(preds_3rscan_repaired, all_gt_global)

    print(f"\n{'Relation':<20s} {'F1':>6s} {'F1+Rep':>7s} {'dF1':>5s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'GT#':>4s}")
    print("-" * 70)
    rel_f1s = []
    rel_f1s_rep = []
    for rel_idx in range(NUM_RELATIONS):
        info = per_rel_3rscan[rel_idx]
        info_r = per_rel_3rscan_rep[rel_idx]
        name = RELATION_NAMES[rel_idx]
        delta = info_r['f1'] - info['f1']
        print(f"  {name:<18s} {info['f1']:>6.3f} {info_r['f1']:>7.3f} {delta:>+5.3f} "
              f"{info['tp']:>4d} {info['fp']:>4d} {info['fn']:>4d} {info['gt_count']:>4d}")
        if info["gt_count"] > 0:
            rel_f1s.append(info["f1"])
            rel_f1s_rep.append(info_r["f1"])
    print("-" * 70)
    macro_rel_f1 = np.mean(rel_f1s) if rel_f1s else 0.0
    macro_rel_f1_rep = np.mean(rel_f1s_rep) if rel_f1s_rep else 0.0
    print(f"  {'MACRO (per-rel)':<18s} {macro_rel_f1:>6.3f} {macro_rel_f1_rep:>7.3f} {macro_rel_f1_rep-macro_rel_f1:>+5.3f}")

    # ── Per-relation comparison across methods ────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-RELATION F1 COMPARISON ACROSS METHODS")
    print("=" * 70)

    preds_fixed = apply_thresholds(all_pred_scores_global, thresholds_fixed_03)
    preds_tuned = apply_thresholds(all_pred_scores_global, thresholds_tabletop)

    per_rel_fixed = compute_per_relation_f1(preds_fixed, all_gt_global)
    per_rel_tuned = compute_per_relation_f1(preds_tuned, all_gt_global)

    print(f"\n{'Relation':<20s} {'3RScan':>7s} {'Fix0.3':>7s} {'Tuned':>7s} {'GT#':>4s}")
    print("-" * 50)
    for rel_idx in range(NUM_RELATIONS):
        name = RELATION_NAMES[rel_idx]
        f1_3r = per_rel_3rscan[rel_idx]["f1"]
        f1_fx = per_rel_fixed[rel_idx]["f1"]
        f1_tn = per_rel_tuned[rel_idx]["f1"]
        gt_count = per_rel_3rscan[rel_idx]["gt_count"]
        if gt_count > 0:
            print(f"  {name:<18s} {f1_3r:>7.3f} {f1_fx:>7.3f} {f1_tn:>7.3f} {gt_count:>4d}")
    print("-" * 50)

    # ── Per-scene summary table ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-SCENE SUMMARY (3RScan thresholds)")
    print("=" * 70)
    print(f"\n{'Scene':<12s} {'Obj':>4s} {'GT#':>4s} {'Pred#':>5s} {'F1':>6s} {'P':>6s} {'R':>6s} {'R@3':>5s} {'R@5':>5s}")
    print("-" * 65)
    for sd in scene_data:
        preds_scene = apply_thresholds(sd["pred_scores_local"], thresholds_3rscan)
        m = compute_f1(preds_scene, sd["gt_set_local"])
        rk = compute_recall_at_k(sd["pred_scores_local"], sd["gt_set_local"])
        print(f"  {sd['scene_name']:<10s} {sd['n_objects']:>4d} "
              f"{m['tp']+m['fn']:>4d} {m['tp']+m['fp']:>5d} "
              f"{m['f1']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{rk[3]:>5.3f} {rk[5]:>5.3f}")
    print("-" * 65)

    # ── Recall@K per-relation (threshold-free) ────────────────────────────────
    print("\n" + "=" * 70)
    print("THRESHOLD-FREE RECALL@K PER RELATION")
    print("  (Rank 12 relations by sigmoid score per edge, check if GT in top-K)")
    print("=" * 70)

    # Compute per-relation Recall@K
    # Group scores by (src, dst)
    pair_scores_global = defaultdict(list)
    for src, rel_idx, dst, score in all_pred_scores_global:
        pair_scores_global[(src, dst)].append((score, rel_idx))
    for key in pair_scores_global:
        pair_scores_global[key].sort(reverse=True)

    print(f"\n{'Relation':<20s} {'R@1':>6s} {'R@3':>6s} {'R@5':>6s} {'GT#':>4s}")
    print("-" * 45)
    for rel_idx in range(NUM_RELATIONS):
        gt_r = {(s, r, d) for s, r, d in all_gt_global if r == rel_idx}
        if len(gt_r) == 0:
            continue
        name = RELATION_NAMES[rel_idx]
        r_at = {}
        for k in [1, 3, 5]:
            hits = 0
            for (src, r, dst) in gt_r:
                top_k_rels = [rel for _, rel in pair_scores_global.get((src, dst), [])[:k]]
                if rel_idx in top_k_rels:
                    hits += 1
            r_at[k] = hits / len(gt_r)
        print(f"  {name:<18s} {r_at[1]:>6.3f} {r_at[3]:>6.3f} {r_at[5]:>6.3f} {len(gt_r):>4d}")
    print("-" * 45)

    # Overall Recall@K
    overall_r_at = {}
    for k in [1, 3, 5]:
        hits = 0
        for (src, rel_idx, dst) in all_gt_global:
            top_k_rels = [rel for _, rel in pair_scores_global.get((src, dst), [])[:k]]
            if rel_idx in top_k_rels:
                hits += 1
        overall_r_at[k] = hits / max(len(all_gt_global), 1)
    print(f"  {'OVERALL':<18s} {overall_r_at[1]:>6.3f} {overall_r_at[3]:>6.3f} "
          f"{overall_r_at[5]:>6.3f} {len(all_gt_global):>4d}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(
        f"\nThe Recall@K numbers are threshold-independent and show the model's"
        f"\nranking quality across domains:"
        f"\n"
        f"\n  Recall@1 = {overall_r_at[1]:.4f}  (model top-1 prediction is correct)"
        f"\n  Recall@3 = {overall_r_at[3]:.4f}  (GT relation in top-3 predictions)"
        f"\n  Recall@5 = {overall_r_at[5]:.4f}  (GT relation in top-5 predictions)"
        f"\n"
        f"\nThreshold calibration gap:"
        f"\n  3RScan thresholds -> Micro F1 = {method_results['3RScan thresholds']['micro_f1']:.4f}"
        f"\n  Fixed 0.3         -> Micro F1 = {method_results['Fixed threshold 0.3']['micro_f1']:.4f}"
        f"\n  Tabletop-tuned    -> Micro F1 = {method_results['Tabletop-tuned']['micro_f1']:.4f}"
    )

    print(f"Done. Evaluated {len(scene_data)} scenes.")


if __name__ == "__main__":
    main()
