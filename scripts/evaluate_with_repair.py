"""
Evaluate Symbolic Consistency Repair on Scene Graph Predictions.

Runs the normal evaluation pipeline, then applies the SceneGraphRepair module
and reports before/after metrics to demonstrate the repair module's impact.

Uses relation-only evaluation mode (with n_objects_hint) to isolate the
repair module's impact from clustering errors.

Usage:
    python scripts/evaluate_with_repair.py
    python scripts/evaluate_with_repair.py --scenes scene_06 scene_07 scene_08
    python scripts/evaluate_with_repair.py --verbose

Output:
    - Console comparison table
    - repair_evaluation_results.json

Author: LogicSplat Team
"""
import sys
sys.path.insert(0, ".")

import os
import json
import time
import argparse
import numpy as np
from typing import Optional, Dict, List, Tuple, Set
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from src.inference.gaussian_inference import run_inference
from src.relations.schema import RELATION_NAMES, NUM_RELATIONS
from src.repair.symbolic_repair import SceneGraphRepair, compute_metrics, compute_per_relation_metrics

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = "D:/logicsplat_data/processed"
RESULTS_PATH = "repair_evaluation_results.json"

DEFAULT_SCENES = [f"scene_{i:02d}" for i in range(6, 14)]

METHODS = [
    {
        "name": "Geometry Rules",
        "short": "Geom",
        "mode": "geometry",
        "model_path": None,
        "thresholds_path": None,
    },
    {
        "name": "GNN v7 (ScanNet pretrained)",
        "short": "GNN-SN",
        "mode": "hybrid",
        "model_path": "models/relation_gnn_v7_dualhead.pt",
        "thresholds_path": "models/relation_gnn_v7_dualhead_thresholds.json",
    },
    {
        "name": "GNN v7 (Tabletop fine-tuned)",
        "short": "GNN-FT",
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


def load_ground_truth(scene_id: str) -> Optional[dict]:
    """Load ground truth JSON for a scene."""
    gt_path = os.path.join(DATA_DIR, scene_id, "ground_truth_relations.json")
    if not os.path.exists(gt_path):
        return None
    with open(gt_path) as f:
        return json.load(f)


def compute_distance_threshold(gt_objects: List[dict]) -> float:
    """Compute matching distance threshold from GT object centroids."""
    if len(gt_objects) >= 2:
        centroids = []
        for obj in gt_objects:
            if "centroid" in obj:
                centroids.append(np.array(obj["centroid"]))
        if len(centroids) >= 2:
            dists = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    dists.append(np.linalg.norm(centroids[i] - centroids[j]))
            median_dist = np.median(dists)
            return max(float(median_dist * 0.4), 0.15)
    return 0.15


def hungarian_match(objects_3d, gt_objects: list, threshold: float) -> dict:
    """Hungarian match predicted clusters to GT objects."""
    if not objects_3d or not gt_objects:
        return {
            "matched": [],
            "unmatched_clusters": [o.uid for o in objects_3d] if objects_3d else [],
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

    cost_matrix = np.zeros((n_clusters, n_gt))
    for i in range(n_clusters):
        for j in range(n_gt):
            cost_matrix[i, j] = np.linalg.norm(
                cluster_centroids[i] - gt_centroids_arr[j]
            )

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

    return {
        "matched": matched,
        "unmatched_clusters": [uid for uid in cluster_uids if uid not in matched_cluster_uids],
        "unmatched_gt": [name for name in gt_names if name not in matched_gt_names],
        "represented_objects": matched_gt_names,
    }


def run_inference_with_thresholds(
    ply_path: str,
    method: dict,
    scene_dir: str,
    n_objects_hint: Optional[int] = None,
) -> dict:
    """Run inference with method-specific model and thresholds."""
    import src.inference.gaussian_inference as gi

    original_thresholds_path = gi.THRESHOLDS_PATH

    if method["thresholds_path"]:
        if os.path.exists(method["thresholds_path"]):
            gi.THRESHOLDS_PATH = method["thresholds_path"]
        else:
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
        gi.THRESHOLDS_PATH = original_thresholds_path

    return result


# ── Core evaluation with repair ───────────────────────────────────────────────

def evaluate_scene_with_repair(
    scene_id: str,
    method: dict,
    repairer: SceneGraphRepair,
    verbose: bool = False,
) -> Optional[dict]:
    """
    Evaluate a single scene with and without symbolic repair.

    Returns dict with before/after metrics and repair statistics.
    """
    scene_dir = os.path.join(DATA_DIR, scene_id)
    ply_path = os.path.join(scene_dir, "splat.ply")

    if not os.path.exists(ply_path):
        print(f"  [{scene_id}] SKIP — no splat.ply")
        return None

    gt_data = load_ground_truth(scene_id)
    if gt_data is None:
        print(f"  [{scene_id}] SKIP — no ground_truth_relations.json")
        return None

    gt_objects = gt_data["objects"]
    gt_relations = gt_data["relations"]
    n_gt = len(gt_objects)
    threshold = compute_distance_threshold(gt_objects)

    # ── Run inference (relation-only mode with n_objects_hint) ─────────────────
    try:
        result = run_inference_with_thresholds(
            ply_path, method, scene_dir, n_objects_hint=n_gt
        )
    except Exception as e:
        print(f"  [{scene_id}] ERROR: {e}")
        return None

    objects_3d = result["objects"]
    pred_rels = result["relations"]

    # ── Match clusters to GT ──────────────────────────────────────────────────
    match = hungarian_match(objects_3d, gt_objects, threshold)
    represented = match["represented_objects"]
    uid_to_name = {uid: name for uid, name, _ in match["matched"]}

    # ── Build GT triple set (represented pairs only) ──────────────────────────
    gt_triples = set()
    for r in gt_relations:
        rel = GT_RELATION_MAP.get(r["relation"], r["relation"])
        gt_triples.add((r["subject"], rel, r["object"]))

    represented_gt = {
        t for t in gt_triples
        if t[0] in represented and t[2] in represented
    }

    # ── Convert predictions to named triples ──────────────────────────────────
    pred_named = []
    for r in pred_rels:
        subj_name = uid_to_name.get(r["subject_id"])
        obj_name = uid_to_name.get(r["object_id"])
        if subj_name is None or obj_name is None:
            continue
        rel = GT_RELATION_MAP.get(r["relation"], r["relation"])
        conf = r.get("confidence", 1.0)
        pred_named.append((subj_name, rel, obj_name, conf))

    # ── BEFORE repair: compute metrics ────────────────────────────────────────
    pred_triples_before = {(s, r, o) for s, r, o, _ in pred_named}
    # Filter to represented pairs only
    pred_represented_before = {
        t for t in pred_triples_before
        if t[0] in represented and t[2] in represented
    }
    metrics_before = compute_metrics(pred_represented_before, represented_gt)

    # ── Apply symbolic repair ─────────────────────────────────────────────────
    repaired, repair_stats = repairer.repair(pred_named)

    # ── AFTER repair: compute metrics ─────────────────────────────────────────
    pred_triples_after = {(s, r, o) for s, r, o, _ in repaired}
    pred_represented_after = {
        t for t in pred_triples_after
        if t[0] in represented and t[2] in represented
    }
    metrics_after = compute_metrics(pred_represented_after, represented_gt)

    # ── Per-relation metrics ──────────────────────────────────────────────────
    per_rel_before = compute_per_relation_metrics(pred_represented_before, represented_gt)
    per_rel_after = compute_per_relation_metrics(pred_represented_after, represented_gt)

    if verbose:
        print(f"  [{scene_id}] Objects: {n_gt} GT, {len(objects_3d)} pred, "
              f"{len(represented)} matched")
        print(f"    Before: F1={metrics_before['f1']:.3f} "
              f"(P={metrics_before['precision']:.3f}, R={metrics_before['recall']:.3f})")
        print(f"    After:  F1={metrics_after['f1']:.3f} "
              f"(P={metrics_after['precision']:.3f}, R={metrics_after['recall']:.3f})")
        print(f"    Repair: removed={repair_stats.relations_removed}, "
              f"added={repair_stats.relations_added}, "
              f"iters={repair_stats.iterations}")

    return {
        "scene_id": scene_id,
        "n_gt_objects": n_gt,
        "n_pred_objects": len(objects_3d),
        "n_matched": len(represented),
        "n_gt_relations_represented": len(represented_gt),
        "before": metrics_before,
        "after": metrics_after,
        "per_relation_before": per_rel_before,
        "per_relation_after": per_rel_after,
        "repair_stats": repair_stats.to_dict(),
    }


# ── Main evaluation loop ─────────────────────────────────────────────────────

def run_evaluation(scenes: List[str], verbose: bool = False) -> dict:
    """Run full evaluation with repair for all methods on specified scenes."""
    repairer = SceneGraphRepair(max_iterations=10, verbose=verbose)
    results = {"methods": {}, "scenes": scenes}

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
            "scenes": [],
            "aggregate_before": {},
            "aggregate_after": {},
            "aggregate_repair_stats": {},
        }

        for scene_id in scenes:
            scene_result = evaluate_scene_with_repair(
                scene_id, method, repairer, verbose=verbose
            )
            if scene_result is not None:
                method_results["scenes"].append(scene_result)

        # ── Aggregate metrics ─────────────────────────────────────────────────
        scene_results = method_results["scenes"]
        if scene_results:
            # Before repair
            total_tp_b = sum(s["before"]["tp"] for s in scene_results)
            total_fp_b = sum(s["before"]["fp"] for s in scene_results)
            total_fn_b = sum(s["before"]["fn"] for s in scene_results)
            micro_p_b, micro_r_b, micro_f1_b = prf1(total_tp_b, total_fp_b, total_fn_b)
            macro_f1_b = np.mean([s["before"]["f1"] for s in scene_results])

            # After repair
            total_tp_a = sum(s["after"]["tp"] for s in scene_results)
            total_fp_a = sum(s["after"]["fp"] for s in scene_results)
            total_fn_a = sum(s["after"]["fn"] for s in scene_results)
            micro_p_a, micro_r_a, micro_f1_a = prf1(total_tp_a, total_fp_a, total_fn_a)
            macro_f1_a = np.mean([s["after"]["f1"] for s in scene_results])

            # Repair stats aggregate
            total_removed = sum(s["repair_stats"]["relations_removed"] for s in scene_results)
            total_added = sum(s["repair_stats"]["relations_added"] for s in scene_results)
            total_contradictions = sum(s["repair_stats"]["contradictions_found"] for s in scene_results)
            avg_iterations = np.mean([s["repair_stats"]["iterations"] for s in scene_results])

            # Aggregate violations by type
            agg_violations = defaultdict(int)
            for s in scene_results:
                for vtype, count in s["repair_stats"]["violations_by_type"].items():
                    agg_violations[vtype] += count

            method_results["aggregate_before"] = {
                "micro_f1": round(micro_f1_b, 4),
                "macro_f1": round(float(macro_f1_b), 4),
                "micro_precision": round(micro_p_b, 4),
                "micro_recall": round(micro_r_b, 4),
                "tp": total_tp_b, "fp": total_fp_b, "fn": total_fn_b,
            }
            method_results["aggregate_after"] = {
                "micro_f1": round(micro_f1_a, 4),
                "macro_f1": round(float(macro_f1_a), 4),
                "micro_precision": round(micro_p_a, 4),
                "micro_recall": round(micro_r_a, 4),
                "tp": total_tp_a, "fp": total_fp_a, "fn": total_fn_a,
            }
            method_results["aggregate_repair_stats"] = {
                "total_contradictions": total_contradictions,
                "total_removed": total_removed,
                "total_added": total_added,
                "avg_iterations": round(float(avg_iterations), 1),
                "violations_by_type": dict(agg_violations),
            }

            # Per-relation aggregation (before and after)
            all_rels = set()
            for s in scene_results:
                all_rels.update(s["per_relation_before"].keys())
                all_rels.update(s["per_relation_after"].keys())

            per_rel_agg_before = {}
            per_rel_agg_after = {}
            for rel in sorted(all_rels):
                # Before
                tp_b = sum(s["per_relation_before"].get(rel, {}).get("tp", 0) for s in scene_results)
                fp_b = sum(s["per_relation_before"].get(rel, {}).get("fp", 0) for s in scene_results)
                fn_b = sum(s["per_relation_before"].get(rel, {}).get("fn", 0) for s in scene_results)
                p_b, r_b, f1_b = prf1(tp_b, fp_b, fn_b)
                per_rel_agg_before[rel] = {"tp": tp_b, "fp": fp_b, "fn": fn_b,
                                           "precision": round(p_b, 4), "recall": round(r_b, 4), "f1": round(f1_b, 4)}
                # After
                tp_a = sum(s["per_relation_after"].get(rel, {}).get("tp", 0) for s in scene_results)
                fp_a = sum(s["per_relation_after"].get(rel, {}).get("fp", 0) for s in scene_results)
                fn_a = sum(s["per_relation_after"].get(rel, {}).get("fn", 0) for s in scene_results)
                p_a, r_a, f1_a = prf1(tp_a, fp_a, fn_a)
                per_rel_agg_after[rel] = {"tp": tp_a, "fp": fp_a, "fn": fn_a,
                                          "precision": round(p_a, 4), "recall": round(r_a, 4), "f1": round(f1_a, 4)}

            method_results["per_relation_before"] = per_rel_agg_before
            method_results["per_relation_after"] = per_rel_agg_after

        results["methods"][method_name] = method_results

    return results


# ── Pretty printing ───────────────────────────────────────────────────────────

def print_summary_table(results: dict):
    """Print the main comparison table."""
    print(f"\n{'='*90}")
    print("SYMBOLIC CONSISTENCY REPAIR — EVALUATION RESULTS")
    print(f"{'='*90}")
    print(f"Scenes: {', '.join(results['scenes'])}")
    print(f"Mode: Relation-Only (with n_objects_hint)")

    # Main table
    header = (f"{'Method':<32} {'F1 before':>10} {'F1 after':>10} "
              f"{'Δ F1':>7} {'Contrad.':>9} {'Removed':>8} {'Added':>7} {'Iters':>6}")
    print(f"\n{header}")
    print("─" * len(header))

    for method_name, mdata in results["methods"].items():
        agg_b = mdata.get("aggregate_before", {})
        agg_a = mdata.get("aggregate_after", {})
        repair = mdata.get("aggregate_repair_stats", {})

        if not agg_b:
            print(f"{method_name:<32} {'N/A':>10}")
            continue

        f1_b = agg_b["micro_f1"]
        f1_a = agg_a["micro_f1"]
        delta = f1_a - f1_b
        delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"

        print(f"{method_name:<32} "
              f"{f1_b:>10.3f} "
              f"{f1_a:>10.3f} "
              f"{delta_str:>7} "
              f"{repair.get('total_contradictions', 0):>9} "
              f"{repair.get('total_removed', 0):>8} "
              f"{repair.get('total_added', 0):>7} "
              f"{repair.get('avg_iterations', 0):>6.1f}")

    # Precision/Recall breakdown
    print(f"\n{'─'*90}")
    print("Precision / Recall breakdown:")
    header2 = (f"{'Method':<32} {'P before':>9} {'P after':>9} "
               f"{'R before':>9} {'R after':>9}")
    print(f"\n{header2}")
    print("─" * len(header2))

    for method_name, mdata in results["methods"].items():
        agg_b = mdata.get("aggregate_before", {})
        agg_a = mdata.get("aggregate_after", {})
        if not agg_b:
            continue
        print(f"{method_name:<32} "
              f"{agg_b['micro_precision']:>9.3f} "
              f"{agg_a['micro_precision']:>9.3f} "
              f"{agg_b['micro_recall']:>9.3f} "
              f"{agg_a['micro_recall']:>9.3f}")


def print_per_relation_table(results: dict):
    """Print per-relation F1 before/after repair."""
    print(f"\n{'='*90}")
    print("PER-RELATION F1 (before → after repair)")
    print(f"{'='*90}")

    method_names = list(results["methods"].keys())

    # Collect all relations
    all_rels = set()
    for mdata in results["methods"].values():
        all_rels.update(mdata.get("per_relation_before", {}).keys())
        all_rels.update(mdata.get("per_relation_after", {}).keys())
    all_rels = sorted(all_rels)

    if not all_rels:
        print("  No per-relation data available.")
        return

    # Build short names
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

    col_width = 16
    header = f"{'Relation':<16}" + "".join(f"{sn:>{col_width}}" for sn in short_names)
    print(f"\n{header}")
    print("─" * len(header))

    for rel in all_rels:
        row = f"{rel:<16}"
        for method_name in method_names:
            mdata = results["methods"][method_name]
            before = mdata.get("per_relation_before", {}).get(rel, {})
            after = mdata.get("per_relation_after", {}).get(rel, {})
            f1_b = before.get("f1", 0.0)
            f1_a = after.get("f1", 0.0)
            cell = f"{f1_b:.3f}→{f1_a:.3f}"
            row += f"{cell:>{col_width}}"
        print(row)


def print_per_scene_table(results: dict):
    """Print per-scene F1 before/after for each method."""
    print(f"\n{'='*90}")
    print("PER-SCENE F1 (before → after repair)")
    print(f"{'='*90}")

    for method_name, mdata in results["methods"].items():
        scenes = mdata.get("scenes", [])
        if not scenes:
            continue

        print(f"\n  {method_name}:")
        header = (f"    {'Scene':<12} {'F1 before':>10} {'F1 after':>10} "
                  f"{'Δ F1':>7} {'Removed':>8} {'Added':>7}")
        print(header)
        print(f"    {'─'*56}")

        for s in scenes:
            f1_b = s["before"]["f1"]
            f1_a = s["after"]["f1"]
            delta = f1_a - f1_b
            delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            print(f"    {s['scene_id']:<12} "
                  f"{f1_b:>10.3f} "
                  f"{f1_a:>10.3f} "
                  f"{delta_str:>7} "
                  f"{s['repair_stats']['relations_removed']:>8} "
                  f"{s['repair_stats']['relations_added']:>7}")


def print_violation_breakdown(results: dict):
    """Print breakdown of violation types found."""
    print(f"\n{'='*90}")
    print("VIOLATION TYPE BREAKDOWN")
    print(f"{'='*90}")

    for method_name, mdata in results["methods"].items():
        repair = mdata.get("aggregate_repair_stats", {})
        violations = repair.get("violations_by_type", {})
        if not violations:
            continue
        print(f"\n  {method_name}:")
        for vtype, count in sorted(violations.items(), key=lambda x: -x[1]):
            print(f"    {vtype:<20} {count:>5}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate symbolic consistency repair on scene graph predictions"
    )
    parser.add_argument(
        "--scenes", nargs="+", default=DEFAULT_SCENES,
        help="Scene IDs to evaluate (default: scene_06 through scene_13)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed per-scene repair info"
    )
    parser.add_argument(
        "--output", "-o", default=RESULTS_PATH,
        help="Output JSON path (default: repair_evaluation_results.json)"
    )
    args = parser.parse_args()

    print("=" * 90)
    print("LogicSplat — Symbolic Consistency Repair Evaluation")
    print("Deterministic constraint propagation for scene graph denoising")
    print("=" * 90)
    print(f"\nScenes: {', '.join(args.scenes)}")
    print(f"Methods: {', '.join(m['name'] for m in METHODS)}")
    print(f"Mode: Relation-Only (n_objects_hint) — isolates repair impact")
    print()

    t_start = time.time()
    results = run_evaluation(args.scenes, verbose=args.verbose)
    t_elapsed = time.time() - t_start

    print(f"\n\n{'#'*90}")
    print(f"  RESULTS (completed in {t_elapsed:.1f}s)")
    print(f"{'#'*90}")

    print_summary_table(results)
    print_per_relation_table(results)
    print_per_scene_table(results)
    print_violation_breakdown(results)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
