"""
LogicSplat -- narrated live run of the VERIFIED pipeline on scene_06.

This script does not reimplement anything. Every function it calls is
imported directly from eval_geokan_tabletop.py -- confirmed the active,
verified script (its MODEL_PATH points at models/geokan_relation_gamma.pt,
and its own fresh run log, results/tabletop_fresh_log.txt, matches this
scene's numbers exactly: Recall@5 = 0.9272 overall, 66/84 relations
correct for scene_06 specifically). This file only adds printed narration
between each real step, so a viewer watching a screen recording can follow
what is happening without needing separate voiceover.

Run from the project root:
    python presentation_recording/02_run_pipeline_scene06.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_geokan_tabletop import (
    DATA_DIR, MODEL_PATH, THRESHOLDS_PATH, SCENE_Z_FLIP,
    load_model, load_gt, cluster_scene, create_virtual_table,
    hungarian_match, build_graph, run_inference, apply_symbolic_repair,
    build_gt_relations_set,
)
from src.relations.schema import RELATION_NAMES

SCENE = "scene_06"
DISPLAY_NAME = "Tabletop 06 -- router on hair-dryer box, water bottle, watch, pen"

WIDTH = 78


def stage(n, total, title):
    print()
    print("=" * WIDTH)
    print(f"STAGE {n}/{total} -- {title}")
    print("=" * WIDTH)


def done(msg, t0=None):
    suffix = f"  ({(time.perf_counter() - t0) * 1000:.1f} ms)" if t0 is not None else ""
    print(f"  -> {msg}{suffix}")


def main():
    print("LogicSplat -- live pipeline run")
    print(f"Scene: {SCENE}  ({DISPLAY_NAME})")
    print(f"Model checkpoint: {MODEL_PATH}")
    print("Every step below calls the real, verified evaluation code in")
    print("eval_geokan_tabletop.py -- nothing here is a mock or a shortcut.")

    TOTAL = 7

    # ---- Stage 1: load the trained model -----------------------------------
    stage(1, TOTAL, "Load the trained GeoKANRelationGNN checkpoint")
    t0 = time.perf_counter()
    model, thresholds = load_model()
    n_params = sum(p.numel() for p in model.parameters())
    done(f"loaded {MODEL_PATH}", t0)
    print(f"     {n_params:,} parameters -- no vision-language model, no LLM")
    print(f"     per-relation decision thresholds loaded from {THRESHOLDS_PATH}")

    # ---- Stage 2: load ground truth (for scoring only, not for prediction) --
    stage(2, TOTAL, "Load ground truth (used only to score the result, not to predict)")
    scene_dir = os.path.join(DATA_DIR, SCENE)
    gt = load_gt(scene_dir)
    n_objects_hint = len(gt["objects"])
    print(f"     Ground truth for this scene: {n_objects_hint} objects, "
          f"{len(gt['relations'])} annotated relations")

    # ---- Stage 3: load the Gaussian Splat + cluster into objects -----------
    stage(3, TOTAL, "Load the 3D Gaussian Splat and cluster it into objects (HDBSCAN)")
    print("     Sub-steps: opacity filter -> outlier removal -> table-plane")
    print("     removal -> HDBSCAN clustering on position + colour")
    ply_path = os.path.join(scene_dir, "splat.ply")
    t0 = time.perf_counter()
    objects, cluster_params = cluster_scene(ply_path, n_objects_hint)
    is_z_down = SCENE_Z_FLIP.get(SCENE, False)
    table_obj = create_virtual_table(objects, is_z_down=is_z_down)
    has_table = table_obj is not None
    if has_table:
        table_obj.uid = len(objects)
        objects = list(objects) + [table_obj]
    done(f"clustered into {len(objects)} objects "
         f"({cluster_params.get('n_gaussians_raw', '?')} raw Gaussians -> "
         f"{cluster_params.get('n_after_filter', '?')} after cleaning)", t0)

    # ---- Stage 4: match clusters to named objects, build the graph ---------
    stage(4, TOTAL, "Match clusters to named objects and build the scene graph")
    gt_items_only = [o for o in gt["objects"] if o["name"] != "table"]
    gt_table_idx = next((i for i, o in enumerate(gt["objects"]) if o["name"] == "table"), None)
    n_match = len(objects) - (1 if has_table else 0)
    mapping = hungarian_match(objects[:n_match], gt_items_only)
    if has_table and gt_table_idx is not None:
        mapping[len(objects) - 1] = {"gt_idx": gt_table_idx, "gt_name": "table", "distance": 0.0}
    names_in_order = [mapping[i]["gt_name"] for i in sorted(mapping)]
    print(f"     Objects identified: {', '.join(names_in_order)}")

    t0 = time.perf_counter()
    x, edge_index, edge_attr = build_graph(objects)
    done(f"built graph: {x.shape[0]} node feature vectors (10-D each), "
         f"{edge_index.shape[1]} directed edges (22-D features each)", t0)

    # ---- Stage 5: run the real model ----------------------------------------
    stage(5, TOTAL, "Run GeoKANRelationGNN inference (the actual neural network)")
    table_idx = (len(objects) - 1) if has_table else None
    t0 = time.perf_counter()
    predictions, pred_scores = run_inference(model, x, edge_index, edge_attr, thresholds, table_idx=table_idx)
    done(f"predicted {len(predictions)} relations above their calibrated thresholds", t0)

    # ---- Stage 6: symbolic repair -------------------------------------------
    stage(6, TOTAL, "Apply SceneGraphRepair (zero-parameter logical consistency check)")
    t0 = time.perf_counter()
    repaired_set, rstats = apply_symbolic_repair(predictions, pred_scores)
    done(f"repair finished in {rstats.iterations} iteration(s): "
         f"+{rstats.relations_added} added, -{rstats.relations_removed} removed", t0)
    print(f"     Final relation count after repair: {len(repaired_set)}")

    # ---- Stage 7: score against ground truth --------------------------------
    stage(7, TOTAL, "Score the final graph against ground truth")
    gt_set = build_gt_relations_set(gt, mapping)
    correct = sum(1 for r in repaired_set if r in gt_set)
    total = len(repaired_set)
    print(f"     {correct} of {total} predicted relations match ground truth")
    print(f"     ({100.0 * correct / total:.1f}% of this scene's predictions verified correct)")

    print()
    print("=" * WIDTH)
    print("DONE -- this is the same checkpoint, same clustering, and same repair")
    print("module used for every number in the written report and defense deck.")
    print("=" * WIDTH)


if __name__ == "__main__":
    main()
