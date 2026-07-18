"""
Proper Evaluation Protocol for LogicSplat — 3D Scene Graph Relation Prediction

Separates clustering quality from relation prediction quality, following
3DSSG / SceneGraphFusion / SGFN evaluation standards.

Three evaluations per method:
  1. Relation-Only (Matched Pairs) — primary paper metric
  2. End-to-End (Fully Automatic) — deployment metric
  3. Clustering Quality — explains the gap between 1 and 2

Usage:
    python scripts/evaluate_proper.py

Author: LogicSplat Team
"""
import sys
sys.path.insert(0, ".")

import os
import json
import time
import numpy as np
from typing import Optional, Dict, List, Tuple, Set
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from src.inference.gaussian_inference import run_inference
from src.relations.schema import RELATION_NAMES, NUM_RELATIONS

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = "D:/logicsplat_data/processed"
RESULTS_PATH = "proper_evaluation_results.json"

TEST_SCENES = [f"scene_{i:02d}" for i in range(6, 14)]

METHODS = [
    {
        "name": "Geometry Rules",
        "mode": "geometry",
        "model_path": None,
        "thresholds_path": None,
    },
    {
        "name": "GNN v7 (ScanNet pretrained)",
        "mode": "hybrid",
        "model_path": "models/relation_gnn_v7_dualhead.pt",
        "thresholds_path": "models/relation_gnn_v7_dualhead_thresholds.json",
    },
    {
        "name": "GNN v7 (Tabletop fine-tuned)",
        "mode": "hybrid",
        "model_path": "models/relation_gnn_v7_finetuned_tabletop.pt",
        "thresholds_path": "models/relation_gnn_v7_finetuned_tabletop_thresholds.json",
    },
]

# GT relation name → schema relation name mapping
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
    "left_of":         "left_of",
    "right_of":        "right_of",
}


# ── Utility functions ─────────────────────────────────────────────────────────

def prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Compute precision, recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def bootstrap_ci(scene_f1_scores: List[float], n_bootstrap: int = 1000) -> Tuple[float, float]:
    """
    Bootstrap 95% confidence interval for mean F1.
    Resample scenes with replacement, compute mean F1 each time,
    return 2.5th and 97.5th percentile.
    """
    if len(scene_f1_scores) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    means = []
    arr = np.array(scene_f1_scores)
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_ground_truth(scene_id: str) -> Optional[dict]:
    """Load ground truth JSON for a scene."""
    gt_path = os.path.join(DATA_DIR, scene_id, "ground_truth_relations.json")
    if not os.path.exists(gt_path):
        return None
    with open(gt_path) as f:
        return json.load(f)


def compute_distance_threshold(gt_objects: List[dict]) -> float:
    """
    Compute matching distance threshold T.
    T = max(median object diagonal, 0.15m)
    """
    diagonals = []
    for obj in gt_objects:
        if "centroid" in obj:
            # Estimate diagonal from point_count as proxy for size
            # Better: use actual bbox if available. For now use a fixed
            # reasonable estimate based on the scene scale.
            pass
    # Since GT doesn't have bbox info, use a fixed threshold of 0.15
    # or compute from centroid spread
    if len(gt_objects) >= 2:
        centroids = []
        for obj in gt_objects:
            if "centroid" in obj:
                centroids.append(np.array(obj["centroid"]))
        if len(centroids) >= 2:
            # Median pairwise distance / 3 as a reasonable threshold
            dists = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    dists.append(np.linalg.norm(centroids[i] - centroids[j]))
            median_dist = np.median(dists)
            # Threshold: 40% of median pairwise distance (generous but not unlimited)
            return max(float(median_dist * 0.4), 0.15)
    return 0.15


def hungarian_match_with_threshold(
    objects_3d,          # List[Object3D] from inference
    gt_objects: list,    # GT objects with centroids
    threshold: float,    # max distance for valid match
) -> Dict[str, any]:
    """
    Hungarian match predicted clusters to GT objects with distance threshold.

    Returns:
        {
            "matched": [(cluster_uid, gt_name, distance), ...],
            "unmatched_clusters": [uid, ...],
            "unmatched_gt": [gt_name, ...],
            "represented_objects": set of gt_names with valid match,
        }
    """
    if not objects_3d or not gt_objects:
        return {
            "matched": [],
            "unmatched_clusters": [o.uid for o in objects_3d],
            "unmatched_gt": [g["name"] for g in gt_objects],
            "represented_objects": set(),
        }

    cluster_centroids = np.array([o.centroid for o in objects_3d])
    cluster_uids = [o.uid for o in objects_3d]

    gt_centroids = []
    gt_names = []
    for gt_obj in gt_objects:
        if "centroid" in gt_obj:
            c = np.array(gt_obj["centroid"], dtype=float)
            c[2] *= -1  # match Z-flip from run_inference
            gt_centroids.append(c)
            gt_names.append(gt_obj["name"])

    if not gt_centroids:
        return {
            "matched": [],
            "unmatched_clusters": [o.uid for o in objects_3d],
            "unmatched_gt": [g["name"] for g in gt_objects],
            "represented_objects": set(),
        }

    gt_centroids_arr = np.array(gt_centroids)
    n_clusters = len(objects_3d)
    n_gt = len(gt_centroids)

    # Build cost matrix
    cost_matrix = np.zeros((n_clusters, n_gt))
    for i in range(n_clusters):
        for j in range(n_gt):
            cost_matrix[i, j] = np.linalg.norm(
                cluster_centroids[i] - gt_centroids_arr[j]
            )

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched = []
    matched_cluster_uids = set()
    matched_gt_names = set()

    for ci, gi in zip(row_ind, col_ind):
        dist = cost_matrix[ci, gi]
        if dist <= threshold:
            matched.append((cluster_uids[ci], gt_names[gi], float(dist)))
            matched_cluster_uids.add(cluster_uids[ci])
            matched_gt_names.add(gt_names[gi])

    unmatched_clusters = [uid for uid in cluster_uids
                          if uid not in matched_cluster_uids]
    unmatched_gt = [name for name in gt_names
                    if name not in matched_gt_names]

    return {
        "matched": matched,
        "unmatched_clusters": unmatched_clusters,
        "unmatched_gt": unmatched_gt,
        "represented_objects": matched_gt_names,
    }


def compute_clustering_quality(
    objects_3d,
    gt_objects: list,
    threshold: float,
    match_result: dict,
) -> dict:
    """
    Compute clustering quality metrics for a single scene.
    """
    n_gt = len(gt_objects)
    n_pred = len(objects_3d)
    count_error = abs(n_pred - n_gt)

    matched = match_result["matched"]
    obj_recall = len(matched) / n_gt if n_gt > 0 else 0.0
    obj_precision = len(matched) / n_pred if n_pred > 0 else 0.0

    # Mean centroid distance for matched pairs
    if matched:
        mean_dist = np.mean([d for _, _, d in matched])
    else:
        mean_dist = float("nan")

    # Merge rate: how many GT objects share the same predicted cluster
    # (multiple GT objects matched to same cluster)
    cluster_to_gt = defaultdict(list)
    for uid, gt_name, _ in matched:
        cluster_to_gt[uid].append(gt_name)
    merges = sum(1 for gts in cluster_to_gt.values() if len(gts) > 1)

    # Split rate: how many predicted clusters map to the same GT object
    gt_to_clusters = defaultdict(list)
    for uid, gt_name, _ in matched:
        gt_to_clusters[gt_name].append(uid)
    splits = sum(1 for uids in gt_to_clusters.values() if len(uids) > 1)

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "count_error": count_error,
        "obj_recall": round(obj_recall, 3),
        "obj_precision": round(obj_precision, 3),
        "mean_dist": round(float(mean_dist), 4) if not np.isnan(mean_dist) else None,
        "merges": merges,
        "splits": splits,
    }


def run_inference_with_thresholds(
    ply_path: str,
    method: dict,
    scene_dir: str,
    n_objects_hint: Optional[int] = None,
) -> dict:
    """
    Run inference with method-specific model and thresholds.
    Handles the thresholds override for fine-tuned models.
    """
    import src.inference.gaussian_inference as gi

    # Save original thresholds path
    original_thresholds_path = gi.THRESHOLDS_PATH

    # Override thresholds path if method specifies one
    if method["thresholds_path"]:
        if os.path.exists(method["thresholds_path"]):
            gi.THRESHOLDS_PATH = method["thresholds_path"]
        else:
            # Use default 0.5 threshold (no per-relation tuning)
            gi.THRESHOLDS_PATH = "__nonexistent__"

    try:
        result = run_inference(
            ply_path,
            model_path=method["model_path"],
            scene_dir=scene_dir,
            n_objects_hint=n_objects_hint,
            mode=method["mode"],
            labeler="none",
        )
    finally:
        # Restore original thresholds path
        gi.THRESHOLDS_PATH = original_thresholds_path

    return result


def evaluate_relation_only(
    result: dict,
    gt_data: dict,
    threshold: float,
) -> dict:
    """
    Evaluation 1: Relation-Only (Matched Pairs).
    Only evaluates on represented pairs (both subject and object matched).
    """
    objects_3d = result["objects"]
    pred_rels = result["relations"]
    gt_objects = gt_data["objects"]
    gt_relations = gt_data["relations"]

    # Match clusters to GT
    match = hungarian_match_with_threshold(objects_3d, gt_objects, threshold)
    represented = match["represented_objects"]

    # Build uid → gt_name map
    uid_to_name = {uid: name for uid, name, _ in match["matched"]}

    # Object coverage
    n_gt_objects = len(gt_objects)
    obj_coverage = len(represented) / n_gt_objects if n_gt_objects > 0 else 0.0

    # Normalize GT relations
    gt_triples = set()
    for r in gt_relations:
        rel = GT_RELATION_MAP.get(r["relation"], r["relation"])
        gt_triples.add((r["subject"], rel, r["object"]))

    # Represented pairs: GT triples where BOTH subject and object are represented
    represented_gt = {
        t for t in gt_triples
        if t[0] in represented and t[2] in represented
    }
    total_gt_triples = len(gt_triples)
    pair_coverage = len(represented_gt) / total_gt_triples if total_gt_triples > 0 else 0.0

    # Translate predictions to named triples (only matched clusters)
    pred_triples = set()
    for r in pred_rels:
        subj_name = uid_to_name.get(r["subject_id"])
        obj_name = uid_to_name.get(r["object_id"])
        if subj_name is None or obj_name is None:
            continue  # ignore predictions involving unmatched clusters
        rel = GT_RELATION_MAP.get(r["relation"], r["relation"])
        pred_triples.add((subj_name, rel, obj_name))

    # Filter predictions to only represented pairs
    pred_represented = {
        t for t in pred_triples
        if t[0] in represented and t[2] in represented
    }

    # Compute metrics on represented pairs only
    tp_set = pred_represented & represented_gt
    fp_set = pred_represented - represented_gt
    fn_set = represented_gt - pred_represented

    tp, fp, fn = len(tp_set), len(fp_set), len(fn_set)
    precision, recall, f1 = prf1(tp, fp, fn)

    # Per-relation breakdown
    all_rels = set(r for _, r, _ in represented_gt) | set(r for _, r, _ in pred_represented)
    per_relation = {}
    for rel in sorted(all_rels):
        gt_rel = {t for t in represented_gt if t[1] == rel}
        pred_rel = {t for t in pred_represented if t[1] == rel}
        r_tp = len(pred_rel & gt_rel)
        r_fp = len(pred_rel - gt_rel)
        r_fn = len(gt_rel - pred_rel)
        r_p, r_r, r_f1 = prf1(r_tp, r_fp, r_fn)
        per_relation[rel] = {"tp": r_tp, "fp": r_fp, "fn": r_fn,
                             "precision": r_p, "recall": r_r, "f1": r_f1}

    return {
        "obj_coverage": round(obj_coverage, 3),
        "pair_coverage": round(pair_coverage, 3),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_relation": per_relation,
    }


def evaluate_end_to_end(
    result: dict,
    gt_data: dict,
    threshold: float,
) -> dict:
    """
    Evaluation 2: End-to-End (Fully Automatic).
    All GT relations with unmatched objects → FN.
    All predictions with unmatched clusters → FP.
    """
    objects_3d = result["objects"]
    pred_rels = result["relations"]
    gt_objects = gt_data["objects"]
    gt_relations = gt_data["relations"]

    # Match clusters to GT
    match = hungarian_match_with_threshold(objects_3d, gt_objects, threshold)
    represented = match["represented_objects"]
    uid_to_name = {uid: name for uid, name, _ in match["matched"]}

    # All GT triples
    gt_triples = set()
    for r in gt_relations:
        rel = GT_RELATION_MAP.get(r["relation"], r["relation"])
        gt_triples.add((r["subject"], rel, r["object"]))

    # All predicted triples (matched clusters only get named)
    pred_triples_matched = set()
    pred_unmatched_count = 0
    for r in pred_rels:
        subj_name = uid_to_name.get(r["subject_id"])
        obj_name = uid_to_name.get(r["object_id"])
        if subj_name is None or obj_name is None:
            pred_unmatched_count += 1  # counts as FP
            continue
        rel = GT_RELATION_MAP.get(r["relation"], r["relation"])
        pred_triples_matched.add((subj_name, rel, obj_name))

    # TP: predicted and in GT
    tp_set = pred_triples_matched & gt_triples
    # FP: predicted but not in GT + predictions involving unmatched clusters
    fp_set = pred_triples_matched - gt_triples
    fp = len(fp_set) + pred_unmatched_count
    # FN: in GT but not predicted (includes all GT triples with unmatched objects)
    fn_set = gt_triples - pred_triples_matched
    fn = len(fn_set)
    tp = len(tp_set)

    precision, recall, f1 = prf1(tp, fp, fn)

    # Per-relation breakdown
    all_rels = set(r for _, r, _ in gt_triples) | set(r for _, r, _ in pred_triples_matched)
    per_relation = {}
    for rel in sorted(all_rels):
        gt_rel = {t for t in gt_triples if t[1] == rel}
        pred_rel = {t for t in pred_triples_matched if t[1] == rel}
        r_tp = len(pred_rel & gt_rel)
        r_fp = len(pred_rel - gt_rel)
        r_fn = len(gt_rel - pred_rel)
        r_p, r_r, r_f1 = prf1(r_tp, r_fp, r_fn)
        per_relation[rel] = {"tp": r_tp, "fp": r_fp, "fn": r_fn,
                             "precision": r_p, "recall": r_r, "f1": r_f1}

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_relation": per_relation,
        "pred_unmatched_fp": pred_unmatched_count,
    }


# ── Main evaluation loop ─────────────────────────────────────────────────────

def run_full_evaluation() -> dict:
    """
    Run all three evaluations for all methods on all test scenes.
    Returns structured results dict.
    """
    results = {
        "methods": {},
        "clustering": {},
    }

    for method in METHODS:
        method_name = method["name"]
        print(f"\n{'='*70}")
        print(f"  METHOD: {method_name}")
        print(f"{'='*70}")

        # Check model exists
        if method["model_path"] and not os.path.exists(method["model_path"]):
            print(f"  SKIP — model not found: {method['model_path']}")
            continue

        method_results = {
            "relation_only": {"scenes": [], "aggregate": {}},
            "end_to_end": {"scenes": [], "aggregate": {}},
            "clustering": {"scenes": []},
        }

        for scene_id in TEST_SCENES:
            scene_dir = os.path.join(DATA_DIR, scene_id)
            ply_path = os.path.join(scene_dir, "splat.ply")

            if not os.path.exists(ply_path):
                print(f"  [{scene_id}] SKIP — no splat.ply")
                continue

            gt_data = load_ground_truth(scene_id)
            if gt_data is None:
                print(f"  [{scene_id}] SKIP — no ground_truth_relations.json")
                continue

            gt_objects = gt_data["objects"]
            n_gt = len(gt_objects)
            threshold = compute_distance_threshold(gt_objects)

            print(f"\n  [{scene_id}] GT objects: {n_gt}, threshold: {threshold:.3f}")

            # ── Evaluation 1: Relation-Only (with n_objects_hint) ─────────────
            try:
                result_hint = run_inference_with_thresholds(
                    ply_path, method, scene_dir,
                    n_objects_hint=n_gt,
                )
            except Exception as e:
                print(f"  [{scene_id}] ERROR (relation-only): {e}")
                continue

            rel_only = evaluate_relation_only(result_hint, gt_data, threshold)
            rel_only["scene_id"] = scene_id
            method_results["relation_only"]["scenes"].append(rel_only)

            # Clustering quality (from the hinted run)
            match_hint = hungarian_match_with_threshold(
                result_hint["objects"], gt_objects, threshold
            )
            clust_quality = compute_clustering_quality(
                result_hint["objects"], gt_objects, threshold, match_hint
            )
            clust_quality["scene_id"] = scene_id
            method_results["clustering"]["scenes"].append(clust_quality)

            print(f"    Rel-Only: F1={rel_only['f1']:.3f} "
                  f"(ObjCov={rel_only['obj_coverage']:.2f}, "
                  f"PairCov={rel_only['pair_coverage']:.2f})")

            # ── Evaluation 2: End-to-End (no hint) ────────────────────────────
            try:
                result_auto = run_inference_with_thresholds(
                    ply_path, method, scene_dir,
                    n_objects_hint=None,
                )
            except Exception as e:
                print(f"  [{scene_id}] ERROR (end-to-end): {e}")
                continue

            e2e = evaluate_end_to_end(result_auto, gt_data, threshold)
            e2e["scene_id"] = scene_id
            method_results["end_to_end"]["scenes"].append(e2e)

            print(f"    End2End: F1={e2e['f1']:.3f}")

        # ── Aggregate relation-only ───────────────────────────────────────────
        ro_scenes = method_results["relation_only"]["scenes"]
        if ro_scenes:
            total_tp = sum(s["tp"] for s in ro_scenes)
            total_fp = sum(s["fp"] for s in ro_scenes)
            total_fn = sum(s["fn"] for s in ro_scenes)
            micro_p, micro_r, micro_f1 = prf1(total_tp, total_fp, total_fn)
            macro_f1 = np.mean([s["f1"] for s in ro_scenes])
            avg_obj_cov = np.mean([s["obj_coverage"] for s in ro_scenes])
            avg_pair_cov = np.mean([s["pair_coverage"] for s in ro_scenes])

            # Bootstrap CI
            scene_f1s = [s["f1"] for s in ro_scenes]
            ci_low, ci_high = bootstrap_ci(scene_f1s)

            # Per-relation aggregation
            all_rels_agg = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
            for s in ro_scenes:
                for rel, stats in s["per_relation"].items():
                    all_rels_agg[rel]["tp"] += stats["tp"]
                    all_rels_agg[rel]["fp"] += stats["fp"]
                    all_rels_agg[rel]["fn"] += stats["fn"]
            per_rel_agg = {}
            for rel, counts in sorted(all_rels_agg.items()):
                p, r, f = prf1(counts["tp"], counts["fp"], counts["fn"])
                per_rel_agg[rel] = {**counts, "precision": p, "recall": r, "f1": f}

            method_results["relation_only"]["aggregate"] = {
                "micro_f1": round(micro_f1, 4),
                "macro_f1": round(float(macro_f1), 4),
                "micro_precision": round(micro_p, 4),
                "micro_recall": round(micro_r, 4),
                "avg_obj_coverage": round(float(avg_obj_cov), 3),
                "avg_pair_coverage": round(float(avg_pair_cov), 3),
                "ci_95": [round(ci_low, 4), round(ci_high, 4)],
                "per_relation": per_rel_agg,
            }

        # ── Aggregate end-to-end ──────────────────────────────────────────────
        e2e_scenes = method_results["end_to_end"]["scenes"]
        if e2e_scenes:
            total_tp = sum(s["tp"] for s in e2e_scenes)
            total_fp = sum(s["fp"] for s in e2e_scenes)
            total_fn = sum(s["fn"] for s in e2e_scenes)
            micro_p, micro_r, micro_f1 = prf1(total_tp, total_fp, total_fn)
            macro_f1 = np.mean([s["f1"] for s in e2e_scenes])

            scene_f1s = [s["f1"] for s in e2e_scenes]
            ci_low, ci_high = bootstrap_ci(scene_f1s)

            per_rel_agg = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
            for s in e2e_scenes:
                for rel, stats in s["per_relation"].items():
                    per_rel_agg[rel]["tp"] += stats["tp"]
                    per_rel_agg[rel]["fp"] += stats["fp"]
                    per_rel_agg[rel]["fn"] += stats["fn"]
            per_rel_final = {}
            for rel, counts in sorted(per_rel_agg.items()):
                p, r, f = prf1(counts["tp"], counts["fp"], counts["fn"])
                per_rel_final[rel] = {**counts, "precision": p, "recall": r, "f1": f}

            method_results["end_to_end"]["aggregate"] = {
                "micro_f1": round(micro_f1, 4),
                "macro_f1": round(float(macro_f1), 4),
                "micro_precision": round(micro_p, 4),
                "micro_recall": round(micro_r, 4),
                "ci_95": [round(ci_low, 4), round(ci_high, 4)],
                "per_relation": per_rel_final,
            }

        results["methods"][method_name] = method_results

    return results


# ── Pretty printing ───────────────────────────────────────────────────────────

def print_table1(results: dict):
    """TABLE 1: RELATION-ONLY (Matched Pairs) — Primary Paper Metric"""
    print(f"\n{'='*70}")
    print("TABLE 1: RELATION-ONLY (Matched Pairs) — Primary Paper Metric")
    print(f"{'='*70}")

    header = f"{'Method':<32} {'Obj Cov':>7} {'Pair Cov':>8} {'Micro F1':>9} {'Macro F1':>9} {'95% CI':>16}"
    print(f"\n{header}")
    print("─" * len(header))

    for method_name, mdata in results["methods"].items():
        agg = mdata["relation_only"].get("aggregate", {})
        if not agg:
            print(f"{method_name:<32} {'N/A':>7}")
            continue
        ci = agg.get("ci_95", [0, 0])
        print(f"{method_name:<32} "
              f"{agg['avg_obj_coverage']:>7.2f} "
              f"{agg['avg_pair_coverage']:>8.2f} "
              f"{agg['micro_f1']:>9.3f} "
              f"{agg['macro_f1']:>9.3f} "
              f"[{ci[0]:.3f}, {ci[1]:.3f}]")

    # Per-relation breakdown
    print(f"\nPer-relation F1 breakdown:")
    # Collect all relations across methods
    all_rels = set()
    for mdata in results["methods"].values():
        agg = mdata["relation_only"].get("aggregate", {})
        if agg and "per_relation" in agg:
            all_rels.update(agg["per_relation"].keys())
    all_rels = sorted(all_rels)

    method_names = list(results["methods"].keys())
    short_names = []
    for mn in method_names:
        if "Geometry" in mn:
            short_names.append("Geom")
        elif "ScanNet" in mn:
            short_names.append("GNN-SN")
        elif "Tabletop" in mn or "fine" in mn.lower():
            short_names.append("GNN-FT")
        else:
            short_names.append(mn[:8])

    col_width = 8
    header2 = f"{'Relation':<16}" + "".join(f"{sn:>{col_width}}" for sn in short_names)
    print(f"\n{header2}")
    print("─" * len(header2))

    for rel in all_rels:
        row = f"{rel:<16}"
        for method_name in method_names:
            agg = results["methods"][method_name]["relation_only"].get("aggregate", {})
            per_rel = agg.get("per_relation", {})
            if rel in per_rel:
                row += f"{per_rel[rel]['f1']:>{col_width}.3f}"
            else:
                row += f"{'—':>{col_width}}"
        print(row)


def print_table2(results: dict):
    """TABLE 2: END-TO-END (Fully Automatic) — Deployment Metric"""
    print(f"\n{'='*70}")
    print("TABLE 2: END-TO-END (Fully Automatic) — Deployment Metric")
    print(f"{'='*70}")

    header = f"{'Method':<32} {'Micro F1':>9} {'Macro F1':>9} {'95% CI':>16}"
    print(f"\n{header}")
    print("─" * len(header))

    for method_name, mdata in results["methods"].items():
        agg = mdata["end_to_end"].get("aggregate", {})
        if not agg:
            print(f"{method_name:<32} {'N/A':>9}")
            continue
        ci = agg.get("ci_95", [0, 0])
        print(f"{method_name:<32} "
              f"{agg['micro_f1']:>9.3f} "
              f"{agg['macro_f1']:>9.3f} "
              f"[{ci[0]:.3f}, {ci[1]:.3f}]")


def print_table3(results: dict):
    """TABLE 3: CLUSTERING QUALITY"""
    print(f"\n{'='*70}")
    print("TABLE 3: CLUSTERING QUALITY (from Relation-Only run with n_objects_hint)")
    print(f"{'='*70}")

    # Use clustering data from the first method (all use same clustering with hint)
    first_method = next(iter(results["methods"].values()), None)
    if not first_method:
        print("  No data available.")
        return

    scenes = first_method["clustering"]["scenes"]
    if not scenes:
        print("  No clustering data.")
        return

    header = (f"{'Scene':<12} {'GT_obj':>6} {'Pred_obj':>8} "
              f"{'Obj_Recall':>10} {'Obj_Prec':>8} "
              f"{'Mean_Dist':>9} {'Merges':>6} {'Splits':>6}")
    print(f"\n{header}")
    print("─" * len(header))

    for s in scenes:
        mean_d = f"{s['mean_dist']:.4f}" if s['mean_dist'] is not None else "N/A"
        print(f"{s['scene_id']:<12} "
              f"{s['n_gt']:>6} "
              f"{s['n_pred']:>8} "
              f"{s['obj_recall']:>10.3f} "
              f"{s['obj_precision']:>8.3f} "
              f"{mean_d:>9} "
              f"{s['merges']:>6} "
              f"{s['splits']:>6}")

    # Summary row
    avg_recall = np.mean([s["obj_recall"] for s in scenes])
    avg_prec = np.mean([s["obj_precision"] for s in scenes])
    valid_dists = [s["mean_dist"] for s in scenes if s["mean_dist"] is not None]
    avg_dist = np.mean(valid_dists) if valid_dists else float("nan")
    total_merges = sum(s["merges"] for s in scenes)
    total_splits = sum(s["splits"] for s in scenes)
    print("─" * len(header))
    dist_str = f"{avg_dist:.4f}" if not np.isnan(avg_dist) else "N/A"
    print(f"{'AVERAGE':<12} {'':>6} {'':>8} "
          f"{avg_recall:>10.3f} {avg_prec:>8.3f} "
          f"{dist_str:>9} {total_merges:>6} {total_splits:>6}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("LogicSplat — Proper Evaluation Protocol")
    print("Separating clustering quality from relation prediction quality")
    print("=" * 70)
    print(f"\nScenes: {', '.join(TEST_SCENES)}")
    print(f"Methods: {', '.join(m['name'] for m in METHODS)}")
    print(f"Evaluations: Relation-Only, End-to-End, Clustering Quality")
    print()

    t_start = time.time()
    results = run_full_evaluation()
    t_elapsed = time.time() - t_start

    print(f"\n\n{'#'*70}")
    print(f"  RESULTS SUMMARY (completed in {t_elapsed:.1f}s)")
    print(f"{'#'*70}")

    print_table1(results)
    print_table2(results)
    print_table3(results)

    # Save to JSON
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
