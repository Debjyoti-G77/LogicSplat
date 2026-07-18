"""Diagnose which scenes have 0 precision and why."""
import sys, os, json
sys.path.insert(0, ".")
os.chdir(r"c:\Users\Debjyoti\Desktop\LogicSplat")

# Load the v7 results
with open("data/processed/ensemble_experiment_v7.json") as f:
    data = json.load(f)

ensemble = data["modes"]["ensemble"]

# Check if per-scene data is available
if "scenes" not in ensemble:
    print("No per-scene data in ensemble_experiment_v7.json")
    print("Let me run a quick per-scene diagnostic instead...")
    print()

# Run a quick diagnostic on each scene
from src.inference.gaussian_inference import run_inference
from scripts.evaluate_scenes import (
    evaluate_scene, build_cluster_to_gt_name_map, GT_RELATION_MAP
)

scenes = ["scene_06", "scene_07", "scene_08", "scene_09",
           "scene_10", "scene_11", "scene_12", "scene_13"]

for scene_id in scenes:
    gt_path = f"data/processed/{scene_id}/ground_truth_relations.json"
    if not os.path.exists(gt_path):
        continue
    
    with open(gt_path) as f:
        gt_data = json.load(f)
    
    gt_objects = gt_data.get("objects", [])
    gt_relations = gt_data.get("relations", [])
    
    # Count GT relations by type
    gt_by_type = {}
    for r in gt_relations:
        rel = GT_RELATION_MAP.get(r["relation"], r["relation"])
        gt_by_type[rel] = gt_by_type.get(rel, 0) + 1
    
    # Run inference
    ply_path = f"data/processed/{scene_id}/splat.ply"
    if not os.path.exists(ply_path):
        continue
    
    result = run_inference(
        ply_path, labeler='none', mode='ensemble',
        n_objects_hint=len(gt_objects),
        scene_dir=f"data/processed/{scene_id}"
    )
    
    objects_3d = result["objects"]
    pred_rels = result["relations"]
    
    # Map clusters to GT
    uid_to_name = build_cluster_to_gt_name_map(objects_3d, gt_objects)
    
    # Translate predictions
    pred_triples = set()
    for r in pred_rels:
        subj = uid_to_name.get(r["subject_id"])
        obj = uid_to_name.get(r["object_id"])
        if subj and obj:
            rel = GT_RELATION_MAP.get(r["relation"], r["relation"])
            pred_triples.add((subj, rel, obj))
    
    gt_triples = set()
    for r in gt_relations:
        rel = GT_RELATION_MAP.get(r["relation"], r["relation"])
        gt_triples.add((r["subject"], rel, r["object"]))
    
    # Per-relation analysis for problematic types
    print(f"\n{'='*60}")
    print(f"  {scene_id} — {len(objects_3d)} objects, mapping: {uid_to_name}")
    print(f"{'='*60}")
    
    for rel_type in ["behind", "in_front_of", "adjacent_to", "on_top_of", "higher_than"]:
        gt_rel = {t for t in gt_triples if t[1] == rel_type}
        pred_rel = {t for t in pred_triples if t[1] == rel_type}
        tp = pred_rel & gt_rel
        fp = pred_rel - gt_rel
        fn = gt_rel - pred_rel
        
        if len(gt_rel) == 0 and len(pred_rel) == 0:
            continue
        
        print(f"\n  [{rel_type}] TP={len(tp)} FP={len(fp)} FN={len(fn)}")
        if fp:
            print(f"    FP: {sorted(fp)[:5]}")
        if fn:
            print(f"    FN: {sorted(fn)[:5]}")
