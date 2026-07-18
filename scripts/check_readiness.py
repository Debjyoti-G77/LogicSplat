"""
Comprehensive readiness check for evaluation scenes (06-13).

Checks:
1. Files exist and are non-empty
2. Splat quality (Gaussian count, NaN/Inf)
3. Clustering produces correct object count
4. Ground truth validity
5. Centroid matching quality
6. Inference runs without errors

Usage:
    python scripts/check_readiness.py

Output:
    - Console table with per-scene status
    - data/processed/readiness_report.json
"""
import sys
sys.path.insert(0, ".")

import os
import json
import warnings
import traceback
import numpy as np
from typing import Optional

warnings.filterwarnings("ignore")

DATA_DIR = "D:/logicsplat_data/processed"
SCENES = [f"scene_{i:02d}" for i in range(6, 14)]
REPORT_PATH = os.path.join(DATA_DIR, "readiness_report.json")

# Valid relation types in our GT files (9 tabletop relations)
VALID_GT_RELATIONS = {
    "on_top_of", "under", "higher_than", "lower_than",
    "to_the_left_of", "to_the_right_of",
    "in_front_of", "behind", "adjacent_to",
}

# Inverse relation pairs
INVERSE_PAIRS = {
    "higher_than": "lower_than",
    "lower_than": "higher_than",
    "on_top_of": "under",
    "under": "on_top_of",
    "to_the_left_of": "to_the_right_of",
    "to_the_right_of": "to_the_left_of",
    "in_front_of": "behind",
    "behind": "in_front_of",
}

# Valid inference relation types (schema names)
VALID_INFERENCE_RELATIONS = {
    "on_top_of", "under", "inside", "attached_to", "hanging_from",
    "adjacent_to", "left_of", "right_of", "in_front_of", "behind",
    "higher_than", "lower_than",
}


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — Files exist
# ═══════════════════════════════════════════════════════════════════════════════

def check_files(scene_id: str) -> dict:
    """Verify all required files exist and meet size thresholds."""
    scene_dir = os.path.join(DATA_DIR, scene_id)
    result = {"pass": True, "details": {}}

    # splat.ply > 1 MB
    ply_path = os.path.join(scene_dir, "splat.ply")
    if os.path.exists(ply_path):
        size_mb = os.path.getsize(ply_path) / (1024 * 1024)
        result["details"]["splat_ply_mb"] = round(size_mb, 2)
        if size_mb < 1.0:
            result["pass"] = False
            result["details"]["splat_ply_error"] = f"Too small: {size_mb:.2f} MB"
    else:
        result["pass"] = False
        result["details"]["splat_ply_error"] = "MISSING"

    # transforms.json with > 10 frames
    transforms_path = os.path.join(scene_dir, "ns_data", "transforms.json")
    if os.path.exists(transforms_path):
        with open(transforms_path) as f:
            tf = json.load(f)
        n_frames = len(tf.get("frames", []))
        result["details"]["transforms_frames"] = n_frames
        if n_frames <= 10:
            result["pass"] = False
            result["details"]["transforms_error"] = f"Only {n_frames} frames"
    else:
        result["pass"] = False
        result["details"]["transforms_error"] = "MISSING"

    # ns_data/images/ with > 50 files
    images_dir = os.path.join(scene_dir, "ns_data", "images")
    if os.path.isdir(images_dir):
        n_images = len([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
        result["details"]["n_images"] = n_images
        if n_images <= 50:
            result["pass"] = False
            result["details"]["images_error"] = f"Only {n_images} images"
    else:
        result["pass"] = False
        result["details"]["images_error"] = "MISSING"

    # ground_truth_relations.json — valid JSON with objects + relations
    gt_path = os.path.join(scene_dir, "ground_truth_relations.json")
    if os.path.exists(gt_path):
        try:
            with open(gt_path) as f:
                gt = json.load(f)
            has_objects = "objects" in gt and len(gt["objects"]) > 0
            has_relations = "relations" in gt and len(gt["relations"]) > 0
            result["details"]["gt_objects"] = len(gt.get("objects", []))
            result["details"]["gt_relations"] = len(gt.get("relations", []))
            if not has_objects or not has_relations:
                result["pass"] = False
                result["details"]["gt_error"] = "Missing objects or relations"
        except json.JSONDecodeError as e:
            result["pass"] = False
            result["details"]["gt_error"] = f"Invalid JSON: {e}"
    else:
        result["pass"] = False
        result["details"]["gt_error"] = "MISSING"

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Splat quality
# ═══════════════════════════════════════════════════════════════════════════════

def check_splat_quality(scene_id: str) -> dict:
    """Load splat.ply and verify Gaussian counts and data integrity."""
    from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians

    scene_dir = os.path.join(DATA_DIR, scene_id)
    ply_path = os.path.join(scene_dir, "splat.ply")
    result = {"pass": True, "details": {}}

    try:
        cloud = load_gaussian_ply(ply_path)
        n_raw = cloud.num_gaussians
        result["details"]["n_raw"] = n_raw

        if n_raw < 50000:
            result["pass"] = False
            result["details"]["error_raw"] = f"Too few: {n_raw:,} (need > 50,000)"

        # Check NaN/Inf
        has_nan = not np.isfinite(cloud.xyz).all()
        result["details"]["has_nan_inf"] = has_nan
        if has_nan:
            result["pass"] = False
            result["details"]["error_nan"] = "NaN/Inf in xyz coordinates"

        # After opacity filter
        filtered = filter_gaussians(cloud, opacity_threshold=0.1)
        n_filtered = filtered.num_gaussians
        result["details"]["n_after_opacity"] = n_filtered

        if n_filtered < 30000:
            result["pass"] = False
            result["details"]["error_opacity"] = f"Too few after opacity: {n_filtered:,}"

        # After SOR pruning
        pruned = prune_isolated_gaussians(filtered, nb_neighbors=20, std_ratio=2.0)
        n_pruned = pruned.num_gaussians
        result["details"]["n_after_sor"] = n_pruned

        if n_pruned < 25000:
            result["pass"] = False
            result["details"]["error_sor"] = f"Too few after SOR: {n_pruned:,}"

        # SOR shouldn't remove more than 10%
        sor_removal_pct = (n_filtered - n_pruned) / max(n_filtered, 1) * 100
        result["details"]["sor_removal_pct"] = round(sor_removal_pct, 1)
        if sor_removal_pct > 10:
            result["details"]["warning_sor"] = f"SOR removed {sor_removal_pct:.1f}% (> 10%)"

    except Exception as e:
        result["pass"] = False
        result["details"]["exception"] = str(e)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — Clustering produces correct object count
# ═══════════════════════════════════════════════════════════════════════════════

def check_clustering(scene_id: str) -> dict:
    """Run clustering and verify object count matches GT."""
    from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
    from src.gaussian.clustering import gaussian_to_objects

    scene_dir = os.path.join(DATA_DIR, scene_id)
    ply_path = os.path.join(scene_dir, "splat.ply")
    gt_path = os.path.join(scene_dir, "ground_truth_relations.json")
    result = {"pass": True, "details": {}}

    try:
        with open(gt_path) as f:
            gt = json.load(f)
        n_gt_objects = len(gt["objects"])
        result["details"]["n_gt_objects"] = n_gt_objects

        cloud = load_gaussian_ply(ply_path)
        filtered = filter_gaussians(cloud, opacity_threshold=0.1)
        pruned = prune_isolated_gaussians(filtered, nb_neighbors=20, std_ratio=2.0)

        target_min = max(2, n_gt_objects - 1)
        target_max = n_gt_objects + 1
        objects, params = gaussian_to_objects(
            pruned, target_min=target_min, target_max=target_max
        )
        n_clusters = len(objects)
        result["details"]["n_clusters"] = n_clusters
        result["details"]["cluster_report"] = f"{n_clusters}/{n_gt_objects}"

        # ±1 is acceptable
        if abs(n_clusters - n_gt_objects) > 1:
            result["pass"] = False
            result["details"]["error"] = (
                f"Cluster count {n_clusters} != GT {n_gt_objects} "
                f"(params: mcs={params.get('min_cluster_size')}, "
                f"method={params.get('cluster_method')})"
            )

    except Exception as e:
        result["pass"] = False
        result["details"]["exception"] = str(e)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — Ground truth validity
# ═══════════════════════════════════════════════════════════════════════════════

def check_gt_validity(scene_id: str) -> dict:
    """Validate ground truth JSON structure and content."""
    scene_dir = os.path.join(DATA_DIR, scene_id)
    gt_path = os.path.join(scene_dir, "ground_truth_relations.json")
    result = {"pass": True, "details": {}, "warnings": []}

    try:
        with open(gt_path) as f:
            gt = json.load(f)

        objects = gt.get("objects", [])
        relations = gt.get("relations", [])
        obj_names = set()

        # Check objects
        for obj in objects:
            name = obj.get("name", "")
            if not name or name == "FILL_IN":
                result["pass"] = False
                result["warnings"].append(f"Object id={obj.get('id')} has no name")
            obj_names.add(name)

            if "centroid" not in obj or obj["centroid"] is None:
                result["pass"] = False
                result["warnings"].append(f"Object '{name}' missing centroid")

        # Check relations non-empty
        if len(relations) == 0:
            result["pass"] = False
            result["warnings"].append("Relations list is empty")

        # Check relation types valid
        invalid_types = set()
        for rel in relations:
            rtype = rel.get("relation", "")
            if rtype not in VALID_GT_RELATIONS:
                invalid_types.add(rtype)
        if invalid_types:
            result["pass"] = False
            result["warnings"].append(f"Invalid relation types: {invalid_types}")

        # Check no duplicate relations
        rel_tuples = [(r["subject"], r["relation"], r["object"]) for r in relations]
        duplicates = len(rel_tuples) - len(set(rel_tuples))
        if duplicates > 0:
            result["warnings"].append(f"{duplicates} duplicate relation(s)")

        # Check object names in relations match objects list
        rel_names = set()
        for rel in relations:
            rel_names.add(rel.get("subject", ""))
            rel_names.add(rel.get("object", ""))
        unknown = rel_names - obj_names
        if unknown:
            result["pass"] = False
            result["warnings"].append(f"Unknown objects in relations: {unknown}")

        # Check inverse relations exist where expected
        rel_set = set(rel_tuples)
        missing_inverses = []
        for subj, rtype, obj in rel_tuples:
            if rtype in INVERSE_PAIRS:
                inv_type = INVERSE_PAIRS[rtype]
                inv_tuple = (obj, inv_type, subj)
                if inv_tuple not in rel_set:
                    missing_inverses.append(f"{obj} {inv_type} {subj}")
        if missing_inverses:
            result["warnings"].append(
                f"Missing {len(missing_inverses)} inverse relation(s): "
                f"{missing_inverses[:3]}{'...' if len(missing_inverses) > 3 else ''}"
            )

        result["details"]["n_objects"] = len(objects)
        result["details"]["n_relations"] = len(relations)
        result["details"]["n_warnings"] = len(result["warnings"])

    except Exception as e:
        result["pass"] = False
        result["warnings"].append(f"Exception: {e}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 5 — Centroid matching quality
# ═══════════════════════════════════════════════════════════════════════════════

def check_centroid_matching(scene_id: str) -> dict:
    """Run clustering, apply Z-flip, Hungarian match against GT centroids."""
    from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
    from src.gaussian.clustering import gaussian_to_objects
    from scipy.optimize import linear_sum_assignment

    scene_dir = os.path.join(DATA_DIR, scene_id)
    ply_path = os.path.join(scene_dir, "splat.ply")
    gt_path = os.path.join(scene_dir, "ground_truth_relations.json")
    result = {"pass": True, "details": {}}

    try:
        with open(gt_path) as f:
            gt = json.load(f)
        gt_objects = gt["objects"]
        n_gt = len(gt_objects)

        cloud = load_gaussian_ply(ply_path)
        filtered = filter_gaussians(cloud, opacity_threshold=0.1)
        pruned = prune_isolated_gaussians(filtered, nb_neighbors=20, std_ratio=2.0)

        target_min = max(2, n_gt - 1)
        target_max = n_gt + 1
        objects, _ = gaussian_to_objects(
            pruned, target_min=target_min, target_max=target_max
        )

        if len(objects) == 0:
            result["pass"] = False
            result["details"]["error"] = "No clusters produced"
            return result

        # Apply Z-flip to cluster centroids (same as run_inference does)
        cluster_centroids = np.array([o.centroid for o in objects])
        cluster_centroids[:, 2] *= -1

        # Apply Z-flip to GT centroids (same as evaluate_scenes.py)
        gt_centroids = []
        for gt_obj in gt_objects:
            c = np.array(gt_obj["centroid"], dtype=float)
            c[2] *= -1  # Z-flip to match inference coordinate system
            gt_centroids.append(c)
        gt_centroids = np.array(gt_centroids)

        # Build cost matrix and run Hungarian matching
        n_clusters = len(objects)
        cost_matrix = np.zeros((n_clusters, n_gt))
        for i in range(n_clusters):
            for j in range(n_gt):
                cost_matrix[i, j] = np.linalg.norm(cluster_centroids[i] - gt_centroids[j])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        distances = []
        matches = []
        for ci, gi in zip(row_ind, col_ind):
            d = cost_matrix[ci, gi]
            distances.append(round(float(d), 3))
            matches.append({
                "cluster_uid": int(objects[ci].uid),
                "gt_name": gt_objects[gi]["name"],
                "distance": round(float(d), 3),
            })

        result["details"]["matches"] = matches
        result["details"]["max_distance"] = round(max(distances), 3) if distances else 0
        result["details"]["mean_distance"] = round(float(np.mean(distances)), 3) if distances else 0

        # Flag if any distance > 2.0
        bad_matches = [m for m in matches if m["distance"] > 2.0]
        if bad_matches:
            result["pass"] = False
            result["details"]["error"] = (
                f"{len(bad_matches)} match(es) with distance > 2.0: "
                f"{[(m['gt_name'], m['distance']) for m in bad_matches]}"
            )

        # Flag if majority > 1.0 (stale centroids)
        stale = [d for d in distances if d > 1.0]
        if len(stale) > len(distances) / 2:
            result["warnings"] = f"GT centroids may be stale: {len(stale)}/{len(distances)} > 1.0"

    except Exception as e:
        result["pass"] = False
        result["details"]["exception"] = str(e)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 6 — Inference runs without errors
# ═══════════════════════════════════════════════════════════════════════════════

def check_inference(scene_id: str) -> dict:
    """Run full inference pipeline and verify output."""
    from src.inference.gaussian_inference import run_inference

    scene_dir = os.path.join(DATA_DIR, scene_id)
    ply_path = os.path.join(scene_dir, "splat.ply")
    gt_path = os.path.join(scene_dir, "ground_truth_relations.json")
    result = {"pass": True, "details": {}}

    try:
        with open(gt_path) as f:
            gt = json.load(f)
        n_gt_objects = len(gt["objects"])

        inf_result = run_inference(
            ply_path,
            labeler="none",
            mode="ensemble",
            n_objects_hint=n_gt_objects,
            scene_dir=scene_dir,
        )

        objects = inf_result.get("objects", [])
        relations = inf_result.get("relations", [])
        result["details"]["n_objects"] = len(objects)
        result["details"]["n_relations"] = len(relations)

        if len(objects) < 2:
            result["pass"] = False
            result["details"]["error_objects"] = f"Only {len(objects)} objects (need >= 2)"

        if len(relations) == 0:
            result["pass"] = False
            result["details"]["error_relations"] = "No relations predicted"

        # Check all predicted relation types are valid
        invalid_rels = set()
        for rel in relations:
            rtype = rel.get("relation", "")
            if rtype not in VALID_INFERENCE_RELATIONS:
                invalid_rels.add(rtype)
        if invalid_rels:
            result["pass"] = False
            result["details"]["error_invalid_rels"] = f"Invalid types: {invalid_rels}"

    except Exception as e:
        result["pass"] = False
        result["details"]["exception"] = str(e)
        result["details"]["traceback"] = traceback.format_exc()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Run all checks and produce report
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_checks():
    """Run all 6 checks for all evaluation scenes."""
    print("=" * 90)
    print("LOGICSPLAT — EVALUATION READINESS CHECK")
    print("=" * 90)
    print(f"Scenes: {', '.join(SCENES)}")
    print()

    full_report = {}

    for scene_id in SCENES:
        print(f"\n{'─' * 90}")
        print(f"  Checking {scene_id}...")
        print(f"{'─' * 90}")

        scene_report = {}

        # Check 1: Files
        print(f"  [1/6] Files...", end=" ", flush=True)
        scene_report["files"] = check_files(scene_id)
        print("✓" if scene_report["files"]["pass"] else "✗")

        # Check 2: Splat quality
        print(f"  [2/6] Splat quality...", end=" ", flush=True)
        scene_report["splat"] = check_splat_quality(scene_id)
        print("✓" if scene_report["splat"]["pass"] else "✗")

        # Check 3: Clustering
        print(f"  [3/6] Clustering...", end=" ", flush=True)
        scene_report["clustering"] = check_clustering(scene_id)
        print("✓" if scene_report["clustering"]["pass"] else "✗")

        # Check 4: GT validity
        print(f"  [4/6] GT validity...", end=" ", flush=True)
        scene_report["gt_validity"] = check_gt_validity(scene_id)
        print("✓" if scene_report["gt_validity"]["pass"] else "✗")

        # Check 5: Centroid matching
        print(f"  [5/6] Centroid matching...", end=" ", flush=True)
        scene_report["matching"] = check_centroid_matching(scene_id)
        print("✓" if scene_report["matching"]["pass"] else "✗")

        # Check 6: Inference
        print(f"  [6/6] Inference...", end=" ", flush=True)
        scene_report["inference"] = check_inference(scene_id)
        print("✓" if scene_report["inference"]["pass"] else "✗")

        # Overall status
        all_pass = all(
            scene_report[k]["pass"]
            for k in ["files", "splat", "clustering", "gt_validity", "matching", "inference"]
        )
        scene_report["status"] = "READY" if all_pass else "NOT READY"
        full_report[scene_id] = scene_report

    # ── Print summary table ───────────────────────────────────────────────────
    print_summary_table(full_report)

    # ── Save report ───────────────────────────────────────────────────────────
    save_report(full_report)

    return full_report


def print_summary_table(report: dict):
    """Print a clean summary table."""
    print("\n")
    print("=" * 90)
    print("READINESS REPORT")
    print("=" * 90)

    header = f"{'Scene':<10} | {'Files':^6} | {'Splat':^6} | {'Clusters':^10} | {'GT Valid':^9} | {'Matching':^10} | {'Inference':^10} | {'Status':<12}"
    print(header)
    print("-" * 90)

    for scene_id, data in report.items():
        files_ok = "✓" if data["files"]["pass"] else "✗"
        splat_ok = "✓" if data["splat"]["pass"] else "✗"

        # Cluster report
        cl = data["clustering"]["details"]
        cluster_str = cl.get("cluster_report", "?/?")
        if not data["clustering"]["pass"]:
            cluster_str = f"✗ {cluster_str}"

        gt_ok = "✓" if data["gt_validity"]["pass"] else "✗"

        # Matching distance
        match_d = data["matching"]["details"]
        mean_dist = match_d.get("mean_distance", "?")
        if not data["matching"]["pass"]:
            match_str = f"✗ {mean_dist}"
        else:
            match_str = f"{mean_dist}"

        inf_ok = "✓" if data["inference"]["pass"] else "✗"
        status = data["status"]

        row = f"{scene_id:<10} | {files_ok:^6} | {splat_ok:^6} | {cluster_str:^10} | {gt_ok:^9} | {match_str:^10} | {inf_ok:^10} | {status:<12}"
        print(row)

    print("=" * 90)

    # Print issues for non-ready scenes
    not_ready = [s for s, d in report.items() if d["status"] != "READY"]
    if not_ready:
        print(f"\n{'─' * 90}")
        print("ISSUES FOUND:")
        print(f"{'─' * 90}")
        for scene_id in not_ready:
            data = report[scene_id]
            print(f"\n  {scene_id}:")
            for check_name in ["files", "splat", "clustering", "gt_validity", "matching", "inference"]:
                check_data = data[check_name]
                if not check_data["pass"]:
                    details = check_data.get("details", {})
                    warnings = check_data.get("warnings", [])
                    errors = [v for k, v in details.items() if "error" in k or "exception" in k]
                    for err in errors:
                        print(f"    [{check_name}] {err}")
                    if isinstance(warnings, list):
                        for w in warnings:
                            print(f"    [{check_name}] WARNING: {w}")
                    elif isinstance(warnings, str):
                        print(f"    [{check_name}] WARNING: {warnings}")
    else:
        print("\n  All scenes READY for evaluation!")

    # Count
    n_ready = sum(1 for d in report.values() if d["status"] == "READY")
    print(f"\n  Summary: {n_ready}/{len(report)} scenes ready")


def save_report(report: dict):
    """Save the full report as JSON."""
    # Convert to JSON-serializable format
    serializable = {}
    for scene_id, data in report.items():
        scene_data = {}
        for check_name, check_data in data.items():
            if check_name == "status":
                scene_data["status"] = check_data
                continue
            scene_data[check_name] = {
                "pass": check_data["pass"],
                "details": check_data.get("details", {}),
            }
            if "warnings" in check_data:
                scene_data[check_name]["warnings"] = check_data["warnings"]
        serializable[scene_id] = scene_data

    with open(REPORT_PATH, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Full report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    run_all_checks()
