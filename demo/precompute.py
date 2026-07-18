"""
LogicSplat Live -- precompute.py

Runs the REAL, verified pipeline (same model checkpoint, same clustering,
same repair module used to produce the manuscript's numbers) for a fixed
list of scenes and writes demo/data/<scene_id>.json + demo/data/<scene_id>.jpg.

Tabletop scenes reuse eval_geokan_tabletop.py's exact functions.
LERF scenes reuse the exact logic from results/rerun_lerf_eval.py.

No statistic in the output JSON is hardcoded -- every number is computed
here, from the real checkpoint and the real inference run.
"""
import sys
import os
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_DEMO_DIR)
sys.path.insert(0, _DEMO_DIR)
sys.path.insert(0, _PROJECT_ROOT)
import json
import shutil
import time

import numpy as np
import torch

from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
from src.gaussian.clustering import gaussian_to_objects, extract_gaussian_node_features
from scripts.build_3rscan_graphs import extract_3rscan_edge_features
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES
from src.relations.geometry import derive_relations, compute_scene_context
from src.repair.symbolic_repair import SceneGraphRepair

from eval_geokan_tabletop import (
    DATA_DIR as TABLETOP_DATA_DIR, MODEL_PATH, THRESHOLDS_PATH,
    create_virtual_table, load_model, load_gt, cluster_scene, build_graph,
    hungarian_match, run_inference, apply_symbolic_repair, GT_RELATION_MAP,
    SCENE_Z_FLIP,
)
from train_geokan_variants import GeoKANVariantGNN, GeoKANGammaLayer

_REL_NAME_TO_IDX = {v: k for k, v in RELATION_NAMES.items()}

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -- which frame image to use per scene (chosen for a clear, representative view) --
TABLETOP_SCENES = {
    "tabletop_06": {"scene": "scene_06", "frame": "frame_00040.png",
                    "display": "Tabletop 06 — router & desk objects",
                    "photo_override": os.path.join("figures", "fig_input_photo.png")},
    "tabletop_08": {"scene": "scene_08", "frame": "frame_00035.png",
                    "display": "Tabletop 08 — router & perfume"},
    "tabletop_10": {"scene": "scene_10", "frame": "frame_00055.png",
                    "display": "Tabletop 10 — router & cream tub"},
}

# Hand-calibrated pixel boxes per scene (visually verified against the actual
# photo -- see the calibration fallback in the build brief). box_px = [x,y,w,h].
CALIBRATED_BOXES = {
    "tabletop_06": {
        "router": [170, 125, 275, 280],
        "agaro_box": [105, 368, 395, 212],
        "water_bottle": [680, 250, 150, 415],
        "watch": [815, 590, 75, 80],
        "pen": [1410, 695, 135, 190],
        "table": [0, 385, 1545, 558],
    },
    "tabletop_08": {
        "router": [535, 30, 275, 95],
        "agaro_box": [510, 120, 395, 200],
        "perfume": [1255, 390, 290, 465],
        "pen": [630, 450, 260, 130],
        "table": [0, 90, 1920, 990],
    },
    "tabletop_10": {
        "router": [815, 180, 180, 100],
        "agaro_box": [810, 275, 190, 190],
        "cream_tub": [855, 465, 215, 185],
        "watch": [715, 845, 730, 90],
        "table": [0, 280, 1920, 800],
    },
}
LERF_SCENES = {
    "lerf_ramen": {"scene": "ramen", "frame": "frame_00006.jpg", "n_exact": 13,
                   "display": "LERF — ramen (public benchmark)"},
    "lerf_teatime": {"scene": "teatime", "frame": "frame_00025.jpg", "n_exact": 10,
                      "display": "LERF — teatime (public benchmark)"},
}
LERF_DATA_ROOT = "D:/lerf_data/lerf_ovs"


def get_param_count(model):
    return sum(p.numel() for p in model.parameters())


def missed_entry(s, r, d, conf):
    """A ground-truth-true relation the model never predicted (false negative)."""
    return {"subj_name": s, "rel": r, "obj_name": d, "conf": round(float(conf), 4), "gt": "missed"}


def process_tabletop_scene(scene_id, cfg):
    model, thresholds = load_model()
    n_params = get_param_count(model)

    scene = cfg["scene"]
    scene_dir = os.path.join(TABLETOP_DATA_DIR, scene)
    ply_path = os.path.join(scene_dir, "splat.ply")
    gt = load_gt(scene_dir)
    n_objects_hint = len(gt["objects"])

    t0 = time.perf_counter()
    objects, _ = cluster_scene(ply_path, n_objects_hint)
    is_z_down = SCENE_Z_FLIP.get(scene, False)
    table_obj = create_virtual_table(objects, is_z_down=is_z_down)
    has_table = table_obj is not None
    if has_table:
        table_obj.uid = len(objects)
        objects = list(objects) + [table_obj]

    gt_items_only = [o for o in gt["objects"] if o["name"] != "table"]
    gt_table_idx = next((i for i, o in enumerate(gt["objects"]) if o["name"] == "table"), None)
    n_match = len(objects) - (1 if has_table else 0)
    mapping = hungarian_match(objects[:n_match], gt_items_only)
    if has_table and gt_table_idx is not None:
        mapping[len(objects) - 1] = {"gt_idx": gt_table_idx, "gt_name": "table", "distance": 0.0}

    x, edge_index, edge_attr = build_graph(objects)
    table_idx = (len(objects) - 1) if has_table else None

    t_infer0 = time.perf_counter()
    predictions, pred_scores = run_inference(model, x, edge_index, edge_attr, thresholds, table_idx=table_idx)
    t_infer1 = time.perf_counter()

    idx_to_name = {idx: info["gt_name"] for idx, info in mapping.items()}
    score_lookup = {(s, r, d): sc for s, r, d, sc in pred_scores}

    gt_named = set()
    for rel in gt["relations"]:
        rel_idx = GT_RELATION_MAP.get(rel["relation"])
        if rel_idx is None:
            continue
        gt_named.add((rel["subject"], RELATION_NAMES[rel_idx], rel["object"]))

    t_repair0 = time.perf_counter()
    repaired_set, rstats = apply_symbolic_repair(predictions, pred_scores)
    t_repair1 = time.perf_counter()

    def named(triples):
        out = []
        for (s, r, d) in triples:
            rel_name = RELATION_NAMES.get(r)
            if rel_name is None:
                continue
            out.append((idx_to_name.get(s, f"node{s}"), rel_name, idx_to_name.get(d, f"node{d}"),
                        score_lookup.get((s, r, d), 0.0), s, d))
        return out

    pre_named = named(predictions)
    post_named = named(repaired_set)
    pre_set = {(s, r, d) for (s, r, d, _, _, _) in pre_named}
    post_set = {(s, r, d) for (s, r, d, _, _, _) in post_named}

    name_to_idx = {}
    for idx, info in mapping.items():
        name_to_idx.setdefault(info["gt_name"], idx)
    rel_name_to_idx = {v: k for k, v in RELATION_NAMES.items()}

    def _score_for(s, r, d):
        si, di, ri = name_to_idx.get(s), name_to_idx.get(d), rel_name_to_idx.get(r)
        if si is None or di is None or ri is None:
            return 0.0
        return score_lookup.get((si, ri, di), 0.0)

    missed_before = [missed_entry(s, r, d, _score_for(s, r, d)) for (s, r, d) in sorted(gt_named - pre_set)]
    missed_after = [missed_entry(s, r, d, _score_for(s, r, d)) for (s, r, d) in sorted(gt_named - post_set)]

    boxes = CALIBRATED_BOXES.get(scene_id, {})
    objects_out = []
    seen_names = set()
    for idx, info in sorted(mapping.items()):
        name = info["gt_name"]
        if name in seen_names:
            continue
        seen_names.add(name)
        box = boxes.get(name, [0, 0, 0, 0])
        anchor = [box[0] + box[2] // 2, box[1] + box[3] // 2]
        objects_out.append({"id": int(idx), "label": name.replace("_", " "),
                             "box_px": box, "anchor_px": anchor})

    def rel_entry(s, r, d, conf, correct_flag, repair_action=None):
        e = {"subj_name": s, "rel": r, "obj_name": d, "conf": round(float(conf), 4),
             "gt": ("correct" if correct_flag else "wrong")}
        if repair_action is not None:
            e["repair_action"] = repair_action
        return e

    relations_before = [rel_entry(s, r, d, c, (s, r, d) in gt_named)
                         for (s, r, d, c, _, _) in sorted(pre_named)]
    relations_after = []
    for (s, r, d, c, _, _) in sorted(post_named):
        action = "kept" if (s, r, d) in pre_set else "added"
        relations_after.append(rel_entry(s, r, d, c, (s, r, d) in gt_named, action))
    removed_by_repair = [rel_entry(s, r, d, c, (s, r, d) in gt_named)
                          for (s, r, d, c, _, _) in sorted(pre_named)
                          if (s, r, d) not in {(a, b, cc) for (a, b, cc, *_ ) in post_named}]

    correct_after = sum(1 for e in relations_after if e["gt"] == "correct")

    out = {
        "scene_id": scene_id,
        "display_name": cfg["display"],
        "dataset": "tabletop",
        "photo": f"{scene_id}.jpg",
        "objects": objects_out,
        "relations_before": relations_before,
        "relations_after": relations_after,
        "removed_by_repair": removed_by_repair,
        "missed_before": missed_before,
        "missed_after": missed_after,
        "repair_summary": {"added": rstats.relations_added, "removed": rstats.relations_removed,
                            "iterations": rstats.iterations},
        "timings_ms": {"inference": round((t_infer1 - t_infer0) * 1000, 2),
                        "repair": round((t_repair1 - t_repair0) * 1000, 2)},
        "model": {"params": n_params, "name": "GeoKAN-Gamma"},
        "gt_available": True,
        "correct_count": correct_after,
        "total_count": len(relations_after),
    }

    # copy photo (prefer the report's own verified frame if specified)
    src_img = cfg.get("photo_override") or os.path.join(scene_dir, "images", cfg["frame"])
    dst_img = os.path.join(OUT_DIR, f"{scene_id}.jpg")
    if src_img.lower().endswith(".png"):
        from PIL import Image as _Image
        _Image.open(src_img).convert("RGB").save(dst_img, quality=92)
    else:
        shutil.copy(src_img, dst_img)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open(os.path.join(OUT_DIR, f"{scene_id}.json"), "w") as f:
        json.dump(out, f, indent=2, default=_default)
    print(f"[{scene_id}] wrote demo/data/{scene_id}.json + .jpg  "
          f"({len(objects_out)} objects, {len(relations_after)} relations, "
          f"{correct_after}/{len(relations_after)} correct)")


def process_lerf_scene(scene_id, cfg):
    from train_geokan_variants import GeoKANVariantGNN, GeoKANGammaLayer
    from lerf_project import (
        project_scene, load_label, label_boxes, match_to_labels,
    )

    scene = cfg["scene"]
    n_exact = cfg["n_exact"]
    frame_name = cfg["frame"]

    state = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
    hidden_dim = state["node_encoder.0.weight"].shape[0]
    edge_feat_dim = state["conv1.lin_edge.weight"].shape[1]
    model = GeoKANVariantGNN(layer_cls=GeoKANGammaLayer, node_feat_dim=10,
                              edge_feat_dim=edge_feat_dim, hidden_dim=hidden_dim,
                              num_relations=NUM_RELATIONS).to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    n_params = get_param_count(model)
    with open(THRESHOLDS_PATH) as f:
        thresholds = {int(k): v for k, v in json.load(f).items()}

    ply_path = os.path.join(LERF_DATA_ROOT, scene, "splat", "splat.ply")
    cloud = load_gaussian_ply(ply_path)
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)

    t0 = time.perf_counter()
    objects, _ = gaussian_to_objects(cloud, target_min=n_exact, target_max=n_exact + 3, n_exact=n_exact)

    all_mins = np.stack([o.bbox_min for o in objects])
    all_maxs = np.stack([o.bbox_max for o in objects])
    scene_min = all_mins.min(axis=0)
    scene_max = all_maxs.max(axis=0)
    scene_extent = np.maximum(scene_max - scene_min, 1e-6)
    obj_sizes = [np.maximum(o.size, 1e-6) for o in objects]
    obj_diags = [float(np.linalg.norm(s)) for s in obj_sizes]
    scene_mean_diag = float(np.mean(obj_diags))
    obj_volumes = [float(np.prod(s)) for s in obj_sizes]
    scene_median_volume = float(np.median(obj_volumes))
    centroid_zs = np.array([o.centroid[2] for o in objects])
    sorted_z_idx = np.argsort(centroid_zs)
    z_ranks = np.zeros(len(objects))
    if len(objects) > 1:
        for rank, obj_idx in enumerate(sorted_z_idx):
            z_ranks[obj_idx] = rank / (len(objects) - 1)

    x = np.stack([
        extract_gaussian_node_features(o, scene_extent, scene_min,
                                        scene_mean_diag=scene_mean_diag,
                                        scene_median_volume=scene_median_volume,
                                        z_rank=float(z_ranks[i]))
        for i, o in enumerate(objects)
    ])
    src, dst, edge_feats = [], [], []
    for i in range(len(objects)):
        for j in range(len(objects)):
            if i != j:
                src.append(i); dst.append(j)
                edge_feats.append(extract_3rscan_edge_features(objects[i], objects[j], scene_extent))
    x_t = torch.tensor(x, dtype=torch.float32).to(DEVICE)
    ei_t = torch.tensor([src, dst], dtype=torch.long).to(DEVICE)
    ea_t = torch.tensor(np.stack(edge_feats), dtype=torch.float32).to(DEVICE)

    scene_ctx = compute_scene_context(all_mins, all_maxs)
    gt_set = set()
    for i in range(len(objects)):
        for j in range(len(objects)):
            if i == j:
                continue
            rels = derive_relations(objects[i].bbox_min, objects[i].bbox_max,
                                     objects[j].bbox_min, objects[j].bbox_max,
                                     scene_context=scene_ctx)
            for rel in rels:
                rel_idx = int(rel)
                if rel_idx < NUM_RELATIONS:
                    gt_set.add((i, rel_idx, j))

    t_infer0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x_t, ei_t, ea_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    t_infer1 = time.perf_counter()

    n_objects = len(objects)
    pred_with_conf = []
    pred_set_before = set()
    score_lookup = {}
    edge_idx = 0
    for i in range(n_objects):
        for j in range(n_objects):
            if i == j:
                continue
            for r in range(NUM_RELATIONS):
                score = float(probs[edge_idx, r])
                score_lookup[(i, r, j)] = score
                if score >= thresholds.get(r, 0.5):
                    pred_with_conf.append((f"obj_{i}", RELATION_NAMES[r], f"obj_{j}", score))
                    pred_set_before.add((i, r, j))
            edge_idx += 1

    repairer = SceneGraphRepair(max_iterations=10, verbose=False)
    t_repair0 = time.perf_counter()
    repaired, rstats = repairer.repair(pred_with_conf)
    t_repair1 = time.perf_counter()

    repaired_set = set()
    for s, r, o, c in repaired:
        si = int(s.split("_")[1]); oi = int(o.split("_")[1])
        ri = _REL_NAME_TO_IDX.get(r)
        if ri is not None:
            repaired_set.add((si, ri, oi))

    # -- match object indices to LERF-OVS's own semantic labels via 3D->2D projection --
    label = load_label(scene, frame_name)
    boxes_px = label_boxes(label)
    _, projected, _ = project_scene(scene, frame_name, n_exact)
    assignment = match_to_labels(projected, boxes_px)  # idx -> category name
    idx_to_name = {idx: cat for idx, cat in assignment.items()}

    def named(triples_with_score):
        out = []
        for (s, r, d) in triples_with_score:
            if s not in idx_to_name or d not in idx_to_name:
                continue
            rel_name = RELATION_NAMES.get(r)
            if rel_name is None:
                continue
            out.append((idx_to_name[s], rel_name, idx_to_name[d], score_lookup.get((s, r, d), 0.0)))
        return out

    pre_named = named(pred_set_before)
    post_named = named(repaired_set)
    pre_set = {(s, r, d) for (s, r, d, _) in pre_named}
    post_set = {(s, r, d) for (s, r, d, _) in post_named}

    def _missed_named(idx_set):
        out = []
        for (i, r, j) in idx_set:
            if i not in idx_to_name or j not in idx_to_name:
                continue
            rel_name = RELATION_NAMES.get(r)
            if rel_name is None:
                continue
            out.append(missed_entry(idx_to_name[i], rel_name, idx_to_name[j], score_lookup.get((i, r, j), 0.0)))
        return sorted(out, key=lambda e: (e["subj_name"], e["rel"], e["obj_name"]))

    missed_before = _missed_named(gt_set - pred_set_before)
    missed_after = _missed_named(gt_set - repaired_set)

    objects_out = []
    for idx, cat in sorted(assignment.items()):
        x0, y0, w, h = boxes_px[cat]
        objects_out.append({"id": int(idx), "label": cat,
                             "box_px": [round(x0), round(y0), round(w), round(h)],
                             "anchor_px": [round(x0 + w / 2), round(y0 + h / 2)]})

    def rel_entry(s, r, d, conf, correct_flag, repair_action=None):
        e = {"subj_name": s, "rel": r, "obj_name": d, "conf": round(float(conf), 4),
             "gt": ("correct" if correct_flag else "wrong")}
        if repair_action is not None:
            e["repair_action"] = repair_action
        return e

    relations_before = [rel_entry(s, r, d, c, (s, r, d) in gt_set_named(gt_set, idx_to_name, RELATION_NAMES))
                         for (s, r, d, c) in sorted(pre_named)]
    relations_after = []
    for (s, r, d, c) in sorted(post_named):
        action = "kept" if (s, r, d) in {(a, b, cc) for (a, b, cc, *_) in pre_named} else "added"
        relations_after.append(rel_entry(s, r, d, c,
                                (s, r, d) in gt_set_named(gt_set, idx_to_name, RELATION_NAMES), action))
    removed_by_repair = [rel_entry(s, r, d, c, (s, r, d) in gt_set_named(gt_set, idx_to_name, RELATION_NAMES))
                          for (s, r, d, c) in sorted(pre_named) if (s, r, d) not in post_set]

    correct_after = sum(1 for e in relations_after if e["gt"] == "correct")

    out = {
        "scene_id": scene_id,
        "display_name": cfg["display"],
        "dataset": "lerf",
        "photo": f"{scene_id}.jpg",
        "objects": objects_out,
        "relations_before": relations_before,
        "relations_after": relations_after,
        "removed_by_repair": removed_by_repair,
        "missed_before": missed_before,
        "missed_after": missed_after,
        "repair_summary": {"added": rstats.relations_added, "removed": rstats.relations_removed,
                            "iterations": rstats.iterations},
        "timings_ms": {"inference": round((t_infer1 - t_infer0) * 1000, 2),
                        "repair": round((t_repair1 - t_repair0) * 1000, 2)},
        "model": {"params": n_params, "name": "GeoKAN-Gamma"},
        "gt_available": True,
        "correct_count": correct_after,
        "total_count": len(relations_after),
    }

    src_img = os.path.join(LERF_DATA_ROOT, "label", scene, frame_name)
    dst_img = os.path.join(OUT_DIR, f"{scene_id}.jpg")
    shutil.copy(src_img, dst_img)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open(os.path.join(OUT_DIR, f"{scene_id}.json"), "w") as f:
        json.dump(out, f, indent=2, default=_default)
    print(f"[{scene_id}] wrote demo/data/{scene_id}.json + .jpg  "
          f"({len(objects_out)} objects, {len(relations_after)} relations, "
          f"{correct_after}/{len(relations_after)} correct)")


def gt_set_named(gt_set, idx_to_name, relation_names):
    out = set()
    for (i, r, j) in gt_set:
        if i in idx_to_name and j in idx_to_name:
            out.add((idx_to_name[i], relation_names[r], idx_to_name[j]))
    return out


def write_manifest():
    scenes = []
    for fn in sorted(os.listdir(OUT_DIR)):
        if fn.endswith(".json") and fn != "manifest.json":
            with open(os.path.join(OUT_DIR, fn)) as f:
                d = json.load(f)
            scenes.append({
                "scene_id": d["scene_id"],
                "display_name": d["display_name"],
                "dataset": d["dataset"],
                "photo": d["photo"],
            })
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump({"scenes": scenes}, f, indent=2)
    print(f"wrote manifest.json ({len(scenes)} scenes)")


if __name__ == "__main__":
    for scene_id, cfg in TABLETOP_SCENES.items():
        process_tabletop_scene(scene_id, cfg)
    for scene_id, cfg in LERF_SCENES.items():
        process_lerf_scene(scene_id, cfg)
    write_manifest()

