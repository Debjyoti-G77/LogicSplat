"""
Generate Z-inverted synthetic tabletop scene graphs for GeoKAN contact-head adaptation.

ROOT CAUSE ANALYSIS:
The 8 tabletop Gaussian splats (scene_06..13) use a Z-DOWN coordinate convention
(camera above table, Z points toward camera = physically higher objects have LOWER Z).
The GeoKAN model was trained on 3RScan where Z is UP. As a result:
  - GT "router on_top_of agaro_box" → router appears BELOW agaro_box in Z
  - contact_score = 0 (a_above_b=False), vz_ratio < 0 → model predicts UNDER/adjacent_to
  - R@5 for on_top_of and under = 0%

FIX: Fine-tune the contact head on synthetic scenes rendered in Z-DOWN convention
but labelled with physical truth (on_top_of still means physically higher object).
The contact head will learn: (vz_ratio<0, contact_score=0, negative vert_gap) → on_top_of.

How Z-inversion works:
  1. Generate physics scene normally (Z-UP: stacked objects have higher Z)
  2. Compute GT labels from physics (on_top_of = physically higher, correct)
  3. Flip ALL Z coordinates: z_flipped = -z_original
  4. Recompute 22-dim edge features and 10-dim node features with flipped coords
  5. Use the ORIGINAL physics labels (on_top_of still = physically above, even with Z-flipped feats)

This creates synthetic data where:
  - on_top_of edges have contact_score=0, vz_ratio<0, negative vert_gap (matches real tabletop)
  - under edges have contact_score>0, vz_ratio>0 (same as normal 3RScan on_top_of — this is OK,
    the model will learn the on_top_of pattern in Z-inverted space)

Output: D:/logicsplat_data/synthetic_zinv/synth_XXX.pt

Usage:
    python scripts/generate_synthetic_zinverted.py
    python scripts/generate_synthetic_zinverted.py --n-scenes 800
"""
import sys
sys.path.insert(0, ".")

import os
import argparse
import numpy as np
import torch
from typing import List, Dict

from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, Relation
from src.relations.geometry import derive_relations, compute_scene_context
from src.graph.definitions import Object3D
from src.gaussian.clustering import extract_gaussian_node_features
from scripts.build_3rscan_graphs import extract_3rscan_edge_features
from scripts.generate_synthetic_tabletop import (
    place_objects_on_table, inject_noise, _dict_to_object3d,
    OBJECT_SIZES, OBJECT_TYPES, P_FLAT, P_STACK, P_CLUSTER,
    TABLE_WIDTH, TABLE_DEPTH, TABLE_Z,
)


DEFAULT_N_SCENES = 800
OUTPUT_DIR = "D:/logicsplat_data/synthetic_zinv"


def compute_gt_relations_physics(objects_normal_z: List[Dict]) -> torch.Tensor:
    """
    Compute GT relations from PHYSICS (Z-UP normal coordinate system).
    These labels are the physical truth regardless of how Z is oriented in the splat.
    """
    n = len(objects_normal_z)
    n_edges = n * (n - 1)
    labels = torch.zeros(n_edges, NUM_RELATIONS, dtype=torch.float32)

    all_mins = np.stack([o["bbox_min"] for o in objects_normal_z])
    all_maxs = np.stack([o["bbox_max"] for o in objects_normal_z])
    scene_ctx = compute_scene_context(all_mins, all_maxs)

    edge_idx = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rels = derive_relations(
                a_min=objects_normal_z[i]["bbox_min"],
                a_max=objects_normal_z[i]["bbox_max"],
                b_min=objects_normal_z[j]["bbox_min"],
                b_max=objects_normal_z[j]["bbox_max"],
                scene_context=scene_ctx,
            )
            for rel in rels:
                labels[edge_idx, int(rel)] = 1.0
            edge_idx += 1

    return labels


def flip_z_objects(objects: List[Dict]) -> List[Dict]:
    """
    Return new object dicts with Z-flipped coordinates.
    z_flipped = -z_original (this makes physically higher objects appear at lower Z,
    matching the tabletop Gaussian splat convention).
    """
    flipped = []
    for obj in objects:
        o = obj.copy()
        # Flip Z of all relevant arrays
        o["bbox_min"] = obj["bbox_min"].copy()
        o["bbox_max"] = obj["bbox_max"].copy()
        o["centroid"] = obj["centroid"].copy()

        # Flip Z
        o["bbox_min"][2] = -obj["bbox_max"][2]
        o["bbox_max"][2] = -obj["bbox_min"][2]
        o["centroid"][2] = -obj["centroid"][2]
        flipped.append(o)
    return flipped


def generate_zinverted_scene(scene_idx: int, rng: np.random.Generator) -> Dict:
    """
    Generate a synthetic tabletop scene in Z-INVERTED coordinate system.

    Physics are simulated in Z-UP, GT labels come from physics.
    Features are extracted from Z-FLIPPED coordinates, matching real tabletop splats.
    """
    n_objects = rng.integers(4, 8)

    # Step 1: Place objects with normal physics (Z-UP)
    clean_objects = place_objects_on_table(rng, n_objects)

    # Step 2: Inject noise
    noisy_objects = inject_noise(clean_objects, rng)
    n = len(noisy_objects)

    # Step 3: Compute GT labels from PHYSICS (Z-UP, correct physical truth)
    edge_label = compute_gt_relations_physics(noisy_objects)

    # Step 4: Flip Z coordinates to simulate Z-DOWN convention
    flipped_objects = flip_z_objects(noisy_objects)

    # Step 5: Compute scene extent from FLIPPED objects
    all_mins_f = np.stack([o["bbox_min"] for o in flipped_objects])
    all_maxs_f = np.stack([o["bbox_max"] for o in flipped_objects])
    scene_min_f = all_mins_f.min(axis=0)
    scene_max_f = all_maxs_f.max(axis=0)
    scene_extent_f = np.maximum(scene_max_f - scene_min_f, 1e-6)

    # Step 6: Convert to Object3D using FLIPPED coordinates
    obj3d_list = [_dict_to_object3d(obj, uid=i) for i, obj in enumerate(flipped_objects)]

    # Compute scene statistics from FLIPPED objects
    obj_sizes = [np.maximum(o.size, 1e-6) for o in obj3d_list]
    obj_diags = [float(np.linalg.norm(s)) for s in obj_sizes]
    scene_mean_diag = float(np.mean(obj_diags)) if obj_diags else 1.0
    obj_volumes = [float(np.prod(s)) for s in obj_sizes]
    scene_median_volume = float(np.median(obj_volumes)) if obj_volumes else 1.0

    # z_rank from FLIPPED coordinates (physically higher objects have LOWER Z → lower rank)
    centroid_zs_f = np.array([o.centroid[2] for o in obj3d_list])
    sorted_z_idx = np.argsort(centroid_zs_f)
    z_ranks = np.zeros(n)
    if n > 1:
        for rank, obj_idx in enumerate(sorted_z_idx):
            z_ranks[obj_idx] = rank / (n - 1)
    else:
        z_ranks[0] = 0.5

    # Step 7: Compute node features from FLIPPED coordinates (10-dim)
    x = np.stack([
        extract_gaussian_node_features(
            obj3d_list[i], scene_extent_f, scene_min_f,
            scene_mean_diag=scene_mean_diag,
            scene_median_volume=scene_median_volume,
            z_rank=float(z_ranks[i]),
        )
        for i in range(n)
    ])

    # Step 8: Compute edge features from FLIPPED coordinates (22-dim)
    edge_feats = []
    src_list, dst_list = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                edge_feats.append(
                    extract_3rscan_edge_features(obj3d_list[i], obj3d_list[j], scene_extent_f)
                )
                src_list.append(i)
                dst_list.append(j)

    scene_id = f"zinv_{scene_idx:03d}"

    return {
        "x":           torch.tensor(x, dtype=torch.float32),
        "edge_index":  torch.tensor(np.array([src_list, dst_list]), dtype=torch.long),
        "edge_attr":   torch.tensor(np.stack(edge_feats), dtype=torch.float32),
        "edge_label":  edge_label,      # PHYSICS labels (on_top_of = physically above)
        "scene_id":    scene_id,
        "z_inverted":  True,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate Z-inverted synthetic tabletop scenes for GeoKAN contact-head adaptation"
    )
    parser.add_argument("--n-scenes", type=int, default=DEFAULT_N_SCENES)
    parser.add_argument("--seed",     type=int, default=123)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"\n{'='*65}")
    print("  LogicSplat - Z-Inverted Synthetic Tabletop Scene Generation")
    print(f"{'='*65}")
    print(f"  Scenes: {args.n_scenes}  Seed: {args.seed}  Out: {args.output_dir}")
    print()

    relation_counts = np.zeros(NUM_RELATIONS, dtype=int)
    total_on_top = 0

    for i in range(1, args.n_scenes + 1):
        scene = generate_zinverted_scene(i, rng)
        torch.save(scene, os.path.join(args.output_dir, f"zinv_{i:03d}.pt"))

        for r in range(NUM_RELATIONS):
            relation_counts[r] += int(scene["edge_label"][:, r].sum())
        total_on_top += int(scene["edge_label"][:, int(Relation.ON_TOP_OF)].sum())

        if i % 100 == 0 or i == args.n_scenes:
            # Quick sanity check: vz_ratio sign for on_top_of edges
            ef = scene["edge_attr"].numpy()
            lbl = scene["edge_label"].numpy()
            ontop_mask = lbl[:, int(Relation.ON_TOP_OF)] > 0.5
            if ontop_mask.any():
                avg_vz = ef[ontop_mask, 19].mean()
                avg_cs = ef[ontop_mask, 12].mean()
            else:
                avg_vz, avg_cs = float('nan'), float('nan')
            print(f"  [{i}/{args.n_scenes}] on_top_of={total_on_top}  "
                  f"vz_ratio_avg={avg_vz:.3f}  contact_score_avg={avg_cs:.3f}  "
                  f"(should be negative/zero for Z-inverted)")

    print("\nPer-relation counts:")
    for r in range(NUM_RELATIONS):
        print(f"  {RELATION_NAMES[r]:20s}: {relation_counts[r]:5d}")
    print(f"\nSaved {args.n_scenes} scenes to {args.output_dir}")


if __name__ == "__main__":
    main()
