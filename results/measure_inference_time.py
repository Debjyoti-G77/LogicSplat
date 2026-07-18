"""
Measure per-stage wall-clock inference time of the LogicSplat pipeline
(clustering -> feature extraction -> GeoKAN-Gamma forward pass -> symbolic
repair) on one tabletop scene and one LERF scene, to quantify the "no
foundation model, fast inference" claim made in the Introduction/Abstract
against something concrete rather than left implicit.

Usage:
    python results/measure_inference_time.py
"""
import sys
sys.path.insert(0, ".")

import os
import time
import numpy as np
import torch

from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
from src.gaussian.clustering import gaussian_to_objects, extract_gaussian_node_features
from scripts.build_3rscan_graphs import extract_3rscan_edge_features
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES
from src.repair.symbolic_repair import SceneGraphRepair
from train_geokan_variants import GeoKANVariantGNN, GeoKANGammaLayer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/geokan_relation_gamma.pt"
THRESHOLDS_PATH = "models/geokan_relation_geokan-gamma_thresholds.json"

import json
with open(THRESHOLDS_PATH) as f:
    THRESHOLDS = {int(k): v for k, v in json.load(f).items()}

repairer = SceneGraphRepair(max_iterations=10, verbose=False)


def build_model(device):
    state = torch.load(MODEL_PATH, weights_only=False, map_location=device)
    hidden_dim = state["node_encoder.0.weight"].shape[0]
    edge_feat_dim = state["conv1.lin_edge.weight"].shape[1]
    model = GeoKANVariantGNN(
        layer_cls=GeoKANGammaLayer,
        node_feat_dim=10, edge_feat_dim=edge_feat_dim,
        hidden_dim=hidden_dim, num_relations=NUM_RELATIONS,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def time_scene(ply_path, scene_label, n_exact, device):
    print(f"\n{'='*70}\n  {scene_label}  (device={device})\n{'='*70}")
    timings = {}

    t0 = time.perf_counter()
    cloud = load_gaussian_ply(ply_path)
    n_raw = cloud.num_gaussians
    t1 = time.perf_counter()
    timings["load_ply"] = t1 - t0

    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)
    t2 = time.perf_counter()
    timings["filter_clean"] = t2 - t1
    n_clean = cloud.num_gaussians

    objects, params = gaussian_to_objects(
        cloud, target_min=n_exact, target_max=n_exact + 3, n_exact=n_exact,
    )
    t3 = time.perf_counter()
    timings["clustering"] = t3 - t2
    n_objects = len(objects)

    # Feature extraction
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
    for i in range(n_objects):
        for j in range(n_objects):
            if i != j:
                src.append(i)
                dst.append(j)
                edge_feats.append(extract_3rscan_edge_features(objects[i], objects[j], scene_extent))
    t4 = time.perf_counter()
    timings["feature_extraction"] = t4 - t3

    model = build_model(device)
    x_t = torch.tensor(x, dtype=torch.float32).to(device)
    ei_t = torch.tensor([src, dst], dtype=torch.long).to(device)
    ea_t = torch.tensor(np.stack(edge_feats), dtype=torch.float32).to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    t5 = time.perf_counter()
    timings["model_load"] = t5 - t4

    # Warm-up (first CUDA call includes kernel compile/alloc overhead)
    with torch.no_grad():
        _ = model(x_t, ei_t, ea_t)
        if device == "cuda":
            torch.cuda.synchronize()
    t6 = time.perf_counter()
    timings["gnn_inference_first_call"] = t6 - t5

    # Timed repeat (steady-state inference cost)
    n_repeats = 20
    t7 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_repeats):
            logits = model(x_t, ei_t, ea_t)
        if device == "cuda":
            torch.cuda.synchronize()
    t8 = time.perf_counter()
    timings["gnn_inference_steady_state_per_call"] = (t8 - t7) / n_repeats

    probs = torch.sigmoid(logits).cpu().numpy()
    pred_with_conf = []
    edge_idx = 0
    for i in range(n_objects):
        for j in range(n_objects):
            if i == j:
                continue
            for r in range(NUM_RELATIONS):
                score = float(probs[edge_idx, r])
                if score >= THRESHOLDS.get(r, 0.5):
                    pred_with_conf.append((f"obj_{i}", RELATION_NAMES[r], f"obj_{j}", score))
            edge_idx += 1
    t9 = time.perf_counter()
    timings["thresholding"] = t9 - t8

    repaired, stats = repairer.repair(pred_with_conf)
    t10 = time.perf_counter()
    timings["symbolic_repair"] = t10 - t9

    timings["total_end_to_end"] = t10 - t0
    timings["total_excl_model_load"] = timings["total_end_to_end"] - timings["model_load"]

    print(f"  Raw Gaussians: {n_raw:,}  ->  after filter/clean: {n_clean:,}  ->  objects: {n_objects}")
    print(f"  Edges (directed pairs): {len(src)}")
    for k, v in timings.items():
        print(f"    {k:35s} {v*1000:8.2f} ms")

    return timings, n_raw, n_objects


if __name__ == "__main__":
    results = {}

    # Tabletop scene (small scale: ~5 objects)
    tt_ply = "D:/logicsplat_data/processed/scene_06/splat.ply"
    if os.path.exists(tt_ply):
        t, n_raw, n_obj = time_scene(tt_ply, "Tabletop scene_06", n_exact=4, device=DEVICE)
        results["tabletop_scene_06"] = {**t, "n_raw_gaussians": n_raw, "n_objects": n_obj}

    # LERF scene (larger scale: ~13 objects)
    lerf_ply = "D:/lerf_data/lerf_ovs/ramen/splat/splat.ply"
    if os.path.exists(lerf_ply):
        t, n_raw, n_obj = time_scene(lerf_ply, "LERF ramen", n_exact=13, device=DEVICE)
        results["lerf_ramen"] = {**t, "n_raw_gaussians": n_raw, "n_objects": n_obj}

    # Also time GNN inference on CPU for the "no GPU required" comparison point
    if DEVICE == "cuda" and os.path.exists(lerf_ply):
        print("\n\n>>> Repeating LERF ramen GNN-inference timing on CPU for comparison <<<")
        t_cpu, _, _ = time_scene(lerf_ply, "LERF ramen (CPU)", n_exact=13, device="cpu")
        results["lerf_ramen_cpu"] = t_cpu

    with open("results/inference_timing.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved -> results/inference_timing.json")
