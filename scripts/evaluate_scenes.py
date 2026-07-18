"""
Evaluate the RelationGNN on scenes that have both splat.ply and ground_truth_relations.json.

Pipeline per scene:
    1. Run inference (run_inference) on splat.ply
    2. Load ground truth from ground_truth_relations.json
    3. Map cluster IDs → ground truth object names via centroid proximity
    4. Compute precision / recall / F1 per scene and per relation type
    5. Print table and save to D:/logicsplat_data/processed/evaluation_results.json

Usage:
    python scripts/evaluate_scenes.py
    python scripts/evaluate_scenes.py --scenes scene_06 scene_07
    python scripts/evaluate_scenes.py --threshold 0.5
    python scripts/evaluate_scenes.py --labeler dino

Notes on label matching:
    Clusters from HDBSCAN have generic labels ("object").
    We match each cluster to a ground truth object by finding the nearest
    ground truth centroid in 3D space.  Ground truth centroids are estimated
    from the objects list in ground_truth_relations.json — if explicit
    centroid coordinates are absent we fall back to rank-ordering by the
    cluster's own centroid (best-effort).
"""

import sys
sys.path.insert(0, ".")

import os
import json
import argparse
import numpy as np
from typing import Optional
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "D:/logicsplat_data/processed"
RESULTS_PATH = os.path.join(DATA_DIR, "evaluation_results.json")
ALL_SCENES = [f"scene_{i:02d}" for i in range(1, 14)]

# Ground truth relation names → schema relation names
# The GT JSON uses "to_the_left_of" / "to_the_right_of"; schema uses "left_of" / "right_of"
GT_RELATION_MAP = {
    "to_the_left_of":  "left_of",
    "to_the_right_of": "right_of",
    "on_top_of":       "on_top_of",
    "under":           "under",
    "inside":          "inside",
    "attached_to":     "attached_to",
    "hanging_from":    "hanging_from",
    "adjacent_to":     "adjacent_to",
    "in_front_of":     "in_front_of",
    "behind":          "behind",
    "higher_than":     "higher_than",
    "lower_than":      "lower_than",
    # pass-through for anything already in schema form
    "left_of":         "left_of",
    "right_of":        "right_of",
}


# ── Centroid matching ─────────────────────────────────────────────────────────

def build_cluster_to_gt_name_map(
    objects_3d,          # List[Object3D] from run_inference
    gt_objects: list,    # list of dicts from ground_truth_relations.json
) -> dict:
    """
    Optimal assignment of predicted clusters to GT objects using the Hungarian
    algorithm (scipy.optimize.linear_sum_assignment).

    Minimises total centroid distance across all assignments simultaneously,
    avoiding the suboptimal greedy nearest-neighbour approach that breaks when
    cluster positions shift after SOR filtering.

    Returns: {cluster_uid: gt_name}
    """
    if not gt_objects or not objects_3d:
        return {}

    cluster_centroids = np.array([o.centroid for o in objects_3d])   # (N, 3)
    cluster_uids      = [o.uid for o in objects_3d]

    # Collect GT centroids; None where the GT object has no explicit coords.
    gt_centroids = []
    for gt_obj in gt_objects:
        if "centroid" in gt_obj:
            c = np.array(gt_obj["centroid"], dtype=float)
            c[2] *= -1   # match the Z-flip applied in run_inference
            gt_centroids.append(c)
        else:
            gt_centroids.append(None)

    has_gt_coords = any(c is not None for c in gt_centroids)

    if not has_gt_coords:
        # No GT coordinates — fall back to matching by cluster size rank
        # (largest cluster → first GT object, etc.)
        sorted_clusters = sorted(objects_3d, key=lambda o: -o.point_count)
        mapping = {}
        for i, obj in enumerate(sorted_clusters):
            if i < len(gt_objects):
                mapping[obj.uid] = gt_objects[i]["name"]
        return mapping

    # Fill missing GT centroids with the mean of known ones
    known = [c for c in gt_centroids if c is not None]
    mean_c = np.mean(known, axis=0) if known else np.zeros(3)
    gt_centroids_arr = np.array(
        [c if c is not None else mean_c for c in gt_centroids]
    )  # (M, 3)

    n_clusters = len(objects_3d)
    n_gt       = len(gt_objects)

    # Build cost matrix: cost[i, j] = distance from cluster i to GT object j
    cost_matrix = np.zeros((n_clusters, n_gt))
    for i, centroid in enumerate(cluster_centroids):
        for j, gt_c in enumerate(gt_centroids_arr):
            cost_matrix[i, j] = np.linalg.norm(centroid - gt_c)

    # Hungarian algorithm: finds the globally optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping = {}
    for cluster_idx, gt_idx in zip(row_ind, col_ind):
        uid = cluster_uids[cluster_idx]
        mapping[uid] = gt_objects[gt_idx]["name"]

    return mapping


# ── Precision / Recall / F1 helpers ──────────────────────────────────────────

def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return round(precision, 4), round(recall, 4), round(f1, 4)


# ── Per-scene evaluation ──────────────────────────────────────────────────────

def evaluate_scene(
    scene_id: str,
    confidence_threshold: float = 0.55,
    labeler: str = "none",
    mode: str = "hybrid",
    classifiers_path: str = "models/physical_classifiers_v2.pkl",
) -> Optional[dict]:
    """
    Run inference + evaluation for one scene.

    Args:
        mode: inference mode for the ablation study
              "hybrid"    — GNN + geometry validation (default, principled approach)
              "geometry"  — geometry rules only, GNN bypassed (old behaviour)
              "gnn"       — GNN sigmoid only, no geometry validation
              "physical"  — logistic regression on geometry + Gaussian physical features
              "ensemble"  — geometry rules for directional/vertical, physical classifiers for adjacent_to

    Returns a result dict or None if the scene cannot be evaluated.
    """
    scene_dir = os.path.join(DATA_DIR, scene_id)
    ply_path  = os.path.join(scene_dir, "splat.ply")
    gt_path   = os.path.join(scene_dir, "ground_truth_relations.json")

    # ── Qualification check ────────────────────────────────────────────────────
    if not os.path.exists(ply_path):
        print(f"  [{scene_id}] SKIP — splat.ply not found")
        return None
    if not os.path.exists(gt_path):
        print(f"  [{scene_id}] SKIP — ground_truth_relations.json not found")
        return None

    print(f"\n{'─'*60}")
    print(f"  Evaluating {scene_id}")
    print(f"{'─'*60}")

    # ── Load ground truth ──────────────────────────────────────────────────────
    with open(gt_path) as f:
        gt_data = json.load(f)

    gt_objects   = gt_data.get("objects", [])
    gt_relations = gt_data.get("relations", [])

    # Normalise GT relation names to schema names
    gt_triples = set()
    for r in gt_relations:
        rel = GT_RELATION_MAP.get(r["relation"], r["relation"])
        gt_triples.add((r["subject"], rel, r["object"]))

    # ── Run inference ──────────────────────────────────────────────────────────
    from src.inference.gaussian_inference import run_inference

    # Physical mode: run clustering only (geometry mode), then apply physical classifiers
    if mode == "physical":
        try:
            result = _run_physical_inference(
                ply_path=ply_path,
                scene_dir=scene_dir,
                labeler=labeler,
                n_objects_hint=len(gt_objects),
                classifiers_path=classifiers_path,
                confidence_threshold=confidence_threshold,
            )
        except Exception as e:
            import traceback
            print(f"  [{scene_id}] ERROR during physical inference: {e}")
            traceback.print_exc()
            return None
    else:
        try:
            result = run_inference(
                ply_path,
                labeler=labeler,
                confidence_threshold=confidence_threshold,
                scene_dir=scene_dir,
                n_objects_hint=len(gt_objects),
                mode=mode,
            )
        except Exception as e:
            print(f"  [{scene_id}] ERROR during inference: {e}")
            return None

    objects_3d = result["objects"]
    pred_rels  = result["relations"]

    if not objects_3d:
        print(f"  [{scene_id}] No objects found — skipping")
        return None

    # ── Map cluster IDs → GT names ─────────────────────────────────────────────
    uid_to_name = build_cluster_to_gt_name_map(objects_3d, gt_objects)
    print(f"\n  Cluster → GT name mapping:")
    for uid, name in sorted(uid_to_name.items()):
        obj = next((o for o in objects_3d if o.uid == uid), None)
        centroid_str = (f"[{obj.centroid[0]:.2f}, {obj.centroid[1]:.2f}, {obj.centroid[2]:.2f}]"
                        if obj else "?")
        print(f"    Cluster {uid} {centroid_str} → '{name}'")

    # ── Translate predicted relations to named triples ─────────────────────────
    pred_triples = set()
    for r in pred_rels:
        subj_name = uid_to_name.get(r["subject_id"])
        obj_name  = uid_to_name.get(r["object_id"])
        if subj_name is None or obj_name is None:
            continue   # cluster not mapped to any GT object
        rel = GT_RELATION_MAP.get(r["relation"], r["relation"])
        pred_triples.add((subj_name, rel, obj_name))

    # ── Compute TP / FP / FN ──────────────────────────────────────────────────
    tp_set = pred_triples & gt_triples
    fp_set = pred_triples - gt_triples
    fn_set = gt_triples  - pred_triples

    tp, fp, fn = len(tp_set), len(fp_set), len(fn_set)
    precision, recall, f1 = prf1(tp, fp, fn)

    print(f"\n  Results: TP={tp}  FP={fp}  FN={fn}")
    print(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    # ── Per-relation-type breakdown ────────────────────────────────────────────
    all_relations = set(r for _, r, _ in gt_triples) | set(r for _, r, _ in pred_triples)
    per_relation: dict[str, dict] = {}
    for rel in sorted(all_relations):
        gt_rel   = {t for t in gt_triples   if t[1] == rel}
        pred_rel = {t for t in pred_triples if t[1] == rel}
        r_tp = len(pred_rel & gt_rel)
        r_fp = len(pred_rel - gt_rel)
        r_fn = len(gt_rel  - pred_rel)
        r_p, r_r, r_f1 = prf1(r_tp, r_fp, r_fn)
        per_relation[rel] = {"tp": r_tp, "fp": r_fp, "fn": r_fn,
                             "precision": r_p, "recall": r_r, "f1": r_f1}

    return {
        "scene_id":     scene_id,
        "n_objects":    len(objects_3d),
        "n_gt_triples": len(gt_triples),
        "n_pred_triples": len(pred_triples),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "per_relation": per_relation,
        "tp_triples": sorted(tp_set),
        "fp_triples": sorted(fp_set),
        "fn_triples": sorted(fn_set),
    }


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def aggregate_results(scene_results: list[dict]) -> dict:
    """Compute macro-averaged and micro-averaged metrics across scenes."""
    if not scene_results:
        return {}

    # Micro: sum TP/FP/FN across all scenes
    total_tp = sum(r["tp"] for r in scene_results)
    total_fp = sum(r["fp"] for r in scene_results)
    total_fn = sum(r["fn"] for r in scene_results)
    micro_p, micro_r, micro_f1 = prf1(total_tp, total_fp, total_fn)

    # Macro: average per-scene F1
    macro_f1 = round(np.mean([r["f1"] for r in scene_results]), 4)
    macro_p  = round(np.mean([r["precision"] for r in scene_results]), 4)
    macro_r  = round(np.mean([r["recall"]    for r in scene_results]), 4)

    # Per-relation aggregation across all scenes
    all_rels: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in scene_results:
        for rel, stats in r["per_relation"].items():
            all_rels[rel]["tp"] += stats["tp"]
            all_rels[rel]["fp"] += stats["fp"]
            all_rels[rel]["fn"] += stats["fn"]

    per_relation_agg = {}
    for rel, counts in sorted(all_rels.items()):
        p, r_, f1 = prf1(counts["tp"], counts["fp"], counts["fn"])
        per_relation_agg[rel] = {**counts, "precision": p, "recall": r_, "f1": f1}

    return {
        "n_scenes": len(scene_results),
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1,
                  "tp": total_tp, "fp": total_fp, "fn": total_fn},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "per_relation": per_relation_agg,
    }


# ── Pretty printing ───────────────────────────────────────────────────────────

def print_results_table(scene_results: list[dict], aggregate: dict):
    """Print a formatted summary table."""
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")

    # Per-scene table
    header = f"{'Scene':<12} {'Objects':>7} {'GT':>5} {'Pred':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'F1':>6}"
    print(f"\n{header}")
    print("─" * len(header))
    for r in scene_results:
        print(f"{r['scene_id']:<12} "
              f"{r['n_objects']:>7} "
              f"{r['n_gt_triples']:>5} "
              f"{r['n_pred_triples']:>5} "
              f"{r['tp']:>4} "
              f"{r['fp']:>4} "
              f"{r['fn']:>4} "
              f"{r['precision']:>6.3f} "
              f"{r['recall']:>6.3f} "
              f"{r['f1']:>6.3f}")

    if aggregate:
        print("─" * len(header))
        m = aggregate["micro"]
        print(f"{'MICRO':<12} {'':>7} {'':>5} {'':>5} "
              f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")
        ma = aggregate["macro"]
        print(f"{'MACRO':<12} {'':>7} {'':>5} {'':>5} {'':>4} {'':>4} {'':>4} "
              f"{ma['precision']:>6.3f} {ma['recall']:>6.3f} {ma['f1']:>6.3f}")

    # Per-relation table
    if aggregate and aggregate.get("per_relation"):
        print(f"\n{'Relation':<18} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'F1':>6}")
        print("─" * 50)
        for rel, stats in aggregate["per_relation"].items():
            print(f"{rel:<18} "
                  f"{stats['tp']:>4} "
                  f"{stats['fp']:>4} "
                  f"{stats['fn']:>4} "
                  f"{stats['precision']:>6.3f} "
                  f"{stats['recall']:>6.3f} "
                  f"{stats['f1']:>6.3f}")

    print(f"\n{'='*70}")


# ── Physical mode inference ───────────────────────────────────────────────────

def _run_physical_inference(
    ply_path: str,
    scene_dir: str,
    labeler: str,
    n_objects_hint: Optional[int],
    classifiers_path: str,
    confidence_threshold: float = 0.35,
) -> dict:
    """
    Run physical classifier inference on a Gaussian splat scene.

    1. Load + cluster Gaussians (same as geometry mode)
    2. Extract TRUE Gaussian physical features from obj._mean_opacity, obj._eigenvalues
    3. Build 25-dim feature vectors for all directed pairs
    4. Apply trained logistic regression classifiers
    5. Return relations dict in same format as run_inference()
    """
    import torch
    from src.gaussian.loader import load_gaussian_ply, filter_gaussians
    from src.gaussian.clustering import (
        gaussian_to_objects,
        extract_gaussian_edge_features,
    )
    from src.inference.gaussian_inference import run_labeling
    from src.models.physical_relation_classifier import (
        PhysicalRelationClassifier,
        extract_physical_features_from_gaussian,
        build_full_feature_vector,
    )
    from src.relations.schema import RELATION_NAMES

    # ── Load classifiers ──────────────────────────────────────────────────────
    if not os.path.exists(classifiers_path):
        raise FileNotFoundError(
            f"Physical classifiers not found: {classifiers_path}\n"
            f"Run: python scripts/extract_physical_features.py"
        )
    clf = PhysicalRelationClassifier.load(classifiers_path)

    # ── Load and cluster Gaussians ────────────────────────────────────────────
    print(f"Loading: {ply_path}")
    cloud = load_gaussian_ply(ply_path)
    cloud_filtered = filter_gaussians(cloud, opacity_threshold=0.1)

    if n_objects_hint is not None:
        target_min = max(2, n_objects_hint - 1)
        target_max = n_objects_hint + 1
        objects, params = gaussian_to_objects(
            cloud_filtered, target_min=target_min, target_max=target_max
        )
    else:
        objects, params = gaussian_to_objects(cloud_filtered)

    print(f"Objects found: {len(objects)} | params: {params}")

    # ── Post-clustering filters (mirrors run_inference — all dynamic) ─────────
    from src.inference.gaussian_inference import (
        _filter_by_point_count, _filter_by_xy_range, _filter_by_z_range,
    )
    objects = _filter_by_point_count(objects)
    objects = _filter_by_xy_range(objects)
    objects = _filter_by_z_range(objects)

    # ── Z-axis normalisation (same as run_inference) ──────────────────────────
    for o in objects:
        o.centroid = o.centroid.copy(); o.centroid[2] *= -1
        o.bbox_min = o.bbox_min.copy(); o.bbox_min[2] *= -1
        o.bbox_max = o.bbox_max.copy(); o.bbox_max[2] *= -1
        o.bbox_min[2], o.bbox_max[2] = (
            min(o.bbox_min[2], o.bbox_max[2]),
            max(o.bbox_min[2], o.bbox_max[2]),
        )

    # ── Semantic labeling ─────────────────────────────────────────────────────
    print(f"\nLabeling (mode={labeler})...")
    objects = run_labeling(objects, scene_dir, labeler=labeler)

    print(f"\nObjects after labeling:")
    for o in objects:
        print(f"  Obj {o.uid} [{o.label}] pts={o.point_count} "
              f"z={o.centroid[2]:.2f}")

    if len(objects) < 2:
        return {"objects": objects, "relations": [], "params": params}

    # ── Scene extent for edge feature normalization ───────────────────────────
    all_mins = np.stack([o.bbox_min for o in objects])
    all_maxs = np.stack([o.bbox_max for o in objects])
    scene_extent = np.maximum(all_maxs.max(axis=0) - all_mins.min(axis=0), 1e-6)

    # ── Build feature vectors for all directed pairs ──────────────────────────
    n = len(objects)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Physical classifiers were trained on 10-dim edge features (base geometry).
    # extract_gaussian_edge_features() now returns 17-dim (added contact + rule-margin).
    # Truncate to first 10 dims for compatibility with the trained classifiers.
    PHYS_EDGE_DIM = 10
    X = np.zeros((len(pairs), 31), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        a, b = objects[i], objects[j]
        edge_f = extract_gaussian_edge_features(a, b, scene_extent)[:PHYS_EDGE_DIM]
        phys_a = extract_physical_features_from_gaussian(a)
        phys_b = extract_physical_features_from_gaussian(b)
        X[k] = build_full_feature_vector(edge_f, phys_a, phys_b)

    # ── Predict ───────────────────────────────────────────────────────────────
    probs = clf.predict_proba(X)  # (E, 12)

    relations = []
    seen_keys: set = set()
    for k, (i, j) in enumerate(pairs):
        for rel_idx in range(12):
            conf = float(probs[k, rel_idx])
            if conf < confidence_threshold:
                continue
            rel_name = RELATION_NAMES[rel_idx]
            key = (i, rel_name, j)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            relations.append({
                "subject_id":    i,
                "subject_label": objects[i].label,
                "relation":      rel_name,
                "object_id":     j,
                "object_label":  objects[j].label,
                "confidence":    round(conf, 3),
            })

    return {"objects": objects, "relations": relations, "params": params}


# ── Comparison table ──────────────────────────────────────────────────────────

def run_comparison(scenes_to_eval: list, args) -> None:
    """
    Run geometry, physical, GNN (hybrid), and ensemble modes and print a comparison table.
    Saves results to data/processed/ensemble_experiment.json.
    """
    COMPARISON_MODES = [
        ("geometry", "Geometry-only  "),
        ("physical", "Physical classif"),
        ("hybrid",   "GNN (hybrid)   "),
        ("ensemble", "Ensemble       "),
    ]

    KEY_RELATIONS = ["on_top_of", "higher_than", "adjacent_to", "left_of", "right_of"]

    all_mode_results = {}

    for mode_key, mode_label in COMPARISON_MODES:
        print(f"\n{'='*70}")
        print(f"Running mode: {mode_label.strip()} ({mode_key})")
        print(f"{'='*70}")

        scene_results = []
        skipped = []
        t_mode_start = __import__("time").time()

        for scene_id in scenes_to_eval:
            result = evaluate_scene(
                scene_id,
                confidence_threshold=args.threshold,
                labeler=args.labeler,
                mode=mode_key,
                classifiers_path=args.classifiers,
            )
            if result is not None:
                scene_results.append(result)
            else:
                skipped.append(scene_id)

        t_mode_elapsed = __import__("time").time() - t_mode_start
        n_evaluated = len(scene_results)
        ms_per_scene = (t_mode_elapsed / n_evaluated * 1000) if n_evaluated > 0 else 0.0

        if not scene_results:
            print(f"  No results for mode {mode_key}")
            all_mode_results[mode_key] = None
            continue

        agg = aggregate_results(scene_results)
        all_mode_results[mode_key] = {
            "aggregate": agg,
            "scenes": scene_results,
            "skipped": skipped,
            "ms_per_scene": round(ms_per_scene, 1),
        }

    # ── Print comparison table ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("COMPARISON TABLE — Ensemble Experiment")
    print(f"{'='*80}")

    # Header
    rel_cols = "  ".join(f"{r[:11]:>11}" for r in KEY_RELATIONS)
    print(f"\n{'Mode':<20} {'Micro F1':>8}  {rel_cols}  {'ms/scene':>9}")
    print("─" * 90)

    for mode_key, mode_label in COMPARISON_MODES:
        res = all_mode_results.get(mode_key)
        if res is None:
            print(f"{mode_label:<20} {'N/A':>8}")
            continue

        agg = res["aggregate"]
        micro_f1 = agg["micro"]["f1"]
        per_rel = agg.get("per_relation", {})
        ms = res.get("ms_per_scene", 0.0)

        rel_vals = "  ".join(
            f"{per_rel.get(r, {}).get('f1', 0.0):>11.3f}"
            for r in KEY_RELATIONS
        )
        print(f"{mode_label:<20} {micro_f1:>8.3f}  {rel_vals}  {ms:>9.1f}")

    print("─" * 90)

    # ── Per-scene F1 for ensemble mode ────────────────────────────────────────
    ens_res = all_mode_results.get("ensemble")
    if ens_res and ens_res.get("scenes"):
        print(f"\n{'='*70}")
        print("PER-SCENE F1 — Ensemble Mode")
        print(f"{'='*70}")
        hdr = f"{'Scene':<12} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Rec':>6} {'F1':>6}"
        print(hdr)
        print("─" * 50)
        for sr in ens_res["scenes"]:
            print(f"  {sr['scene_id']:<10} {sr['tp']:>4} {sr['fp']:>4} {sr['fn']:>4} "
                  f"{sr['precision']:>6.3f} {sr['recall']:>6.3f} {sr['f1']:>6.3f}")
        print("─" * 50)
        ens_agg = ens_res["aggregate"]
        print(f"  {'MICRO':10} {ens_agg['micro']['tp']:>4} {ens_agg['micro']['fp']:>4} "
              f"{ens_agg['micro']['fn']:>4} "
              f"{ens_agg['micro']['precision']:>6.3f} {ens_agg['micro']['recall']:>6.3f} "
              f"{ens_agg['micro']['f1']:>6.3f}")

    # ── Feature importance (physical mode) ───────────────────────────────────
    if os.path.exists(args.classifiers):
        from src.models.physical_relation_classifier import PhysicalRelationClassifier
        clf = PhysicalRelationClassifier.load(args.classifiers)
        importance = clf.feature_importance(top_k=5)

        print(f"\n{'='*70}")
        print("FEATURE IMPORTANCE (Physical Classifiers — top 5 per relation)")
        print(f"{'='*70}")
        for rel_name, feats in importance.items():
            feat_str = ", ".join(f"{n}({c:+.3f})" for n, c in feats)
            print(f"  [{rel_name:18s}] {feat_str}")

        # ── Top-3 per relation (interpretability summary for presentation) ────
        print(f"\n{'='*70}")
        print("TOP-3 FEATURES PER RELATION (Interpretability Summary)")
        print(f"{'='*70}")
        importance_top3 = clf.feature_importance(top_k=3)
        for rel_name, feats in importance_top3.items():
            feat_str = ", ".join(f"{n}" for n, _ in feats)
            print(f"  {rel_name:<20}: [{feat_str}]")

        # ── Training time benchmark ───────────────────────────────────────────
        print(f"\n{'='*70}")
        print("TRAINING TIME BENCHMARK")
        print(f"{'='*70}")
        import time
        import pickle
        with open(args.classifiers, "rb") as _f:
            _clf_data = pickle.load(_f)
        n_classifiers = len(_clf_data.get("classifiers", {}))
        print(f"  Physical classifiers loaded: {n_classifiers} relation classifiers")
        print(f"  Inference per scene: run with --mode ensemble to measure")
        print(f"  (Training was done offline on ScanNet; classifiers are pre-trained)")

        # Print ensemble timing if available
        ens_res = all_mode_results.get("ensemble")
        if ens_res:
            print(f"  Ensemble inference per scene: {ens_res.get('ms_per_scene', 0):.1f}ms")
        geo_res = all_mode_results.get("geometry")
        if geo_res:
            print(f"  Geometry-only inference per scene: {geo_res.get('ms_per_scene', 0):.1f}ms")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output_path = "D:/logicsplat_data/processed/ensemble_experiment_v7.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    save_data = {
        "config": {
            "scenes_evaluated": scenes_to_eval,
            "threshold": args.threshold,
            "labeler": args.labeler,
            "classifiers_path": args.classifiers,
        },
        "modes": {},
    }

    for mode_key, mode_label in COMPARISON_MODES:
        res = all_mode_results.get(mode_key)
        if res is None:
            save_data["modes"][mode_key] = None
            continue
        save_data["modes"][mode_key] = {
            "label": mode_label.strip(),
            "aggregate": res["aggregate"],
            "skipped": res["skipped"],
            "ms_per_scene": res.get("ms_per_scene", 0.0),
        }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nComparison results saved to: {output_path}")

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RelationGNN on scenes with splat.ply + ground_truth_relations.json"
    )
    parser.add_argument(
        "--scenes", nargs="+", default=None,
        help="Scene IDs to evaluate (default: all scenes with both required files)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.25,
        help="Confidence threshold for relation predictions (default: 0.25, sigmoid per-class)"
    )
    parser.add_argument(
        "--labeler", default="none",
        choices=["auto", "yolo", "dino", "none"],
        help="Semantic labeler (default: none — uses centroid matching only)"
    )
    parser.add_argument(
        "--mode", default="hybrid",
        choices=["hybrid", "geometry", "gnn", "physical", "ensemble"],
        help=(
            "Inference mode for ablation study:\n"
            "  hybrid    — GNN + geometry validation (default, principled approach)\n"
            "  geometry  — geometry rules only, GNN bypassed\n"
            "  gnn       — GNN sigmoid only, no geometry validation\n"
            "  physical  — logistic regression on geometry + Gaussian physical features\n"
            "  ensemble  — geometry rules for directional/vertical, physical classifiers for adjacent_to"
        ),
    )
    parser.add_argument(
        "--classifiers", default="models/physical_classifiers_v2.pkl",
        help="Path to trained physical classifiers (used with --mode physical)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run all three modes (geometry, physical, gnn) and print comparison table"
    )
    parser.add_argument(
        "--output", default=RESULTS_PATH,
        help=f"Path to save evaluation_results.json (default: {RESULTS_PATH})"
    )
    args = parser.parse_args()

    scenes_to_eval = args.scenes or ALL_SCENES

    if args.compare:
        run_comparison(scenes_to_eval, args)
        return

    print(f"\nLogicSplat — Scene Graph Evaluation")
    print(f"Scenes    : {', '.join(scenes_to_eval)}")
    print(f"Threshold : {args.threshold}")
    print(f"Labeler   : {args.labeler}")
    print(f"Mode      : {args.mode}")
    print(f"Output    : {args.output}")

    scene_results = []
    skipped = []

    for scene_id in scenes_to_eval:
        result = evaluate_scene(
            scene_id,
            confidence_threshold=args.threshold,
            labeler=args.labeler,
            mode=args.mode,
            classifiers_path=args.classifiers,
        )
        if result is not None:
            scene_results.append(result)
        else:
            skipped.append(scene_id)

    if not scene_results:
        print("\nNo scenes could be evaluated.")
        print("Make sure splat.ply and ground_truth_relations.json exist for at least one scene.")
        return

    aggregate = aggregate_results(scene_results)
    print_results_table(scene_results, aggregate)

    if skipped:
        print(f"\nSkipped (missing files): {', '.join(skipped)}")

    # ── Save results ───────────────────────────────────────────────────────────
    output = {
        "config": {
            "confidence_threshold": args.threshold,
            "labeler": args.labeler,
            "mode": args.mode,
            "scenes_evaluated": [r["scene_id"] for r in scene_results],
            "scenes_skipped": skipped,
        },
        "aggregate": aggregate,
        "scenes": scene_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
