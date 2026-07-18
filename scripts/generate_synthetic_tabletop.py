"""
Generate synthetic tabletop scene graphs for GeoKAN fine-tuning.

Each scene:
- 4-7 objects placed on a table surface
- Physics-aware: objects rest on table or stack on each other
- Collision resolution: no interpenetration
- Noise injection: mimic real Gaussian clustering errors
- GT relations derived from known 3D positions
- Features computed using EXACT SAME functions as real inference/training:
    - Edge: extract_3rscan_edge_features (22-dim, matches GeoKAN training)
    - Node: extract_gaussian_node_features (10-dim, scale-invariant)
- Labels: 10-dim schema (no INSIDE/HANGING_FROM)

Output: D:/logicsplat_data/synthetic_v2/synth_XXX.pt (PyG-compatible graph dicts)

Usage:
    python scripts/generate_synthetic_tabletop.py
    python scripts/generate_synthetic_tabletop.py --n-scenes 800
"""
import sys
sys.path.insert(0, ".")

import os
import argparse
import numpy as np
import torch
from typing import List, Tuple, Dict

from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, Relation
from src.relations.geometry import derive_relations, compute_scene_context
from src.graph.definitions import Object3D
from src.gaussian.clustering import extract_gaussian_node_features
from scripts.build_3rscan_graphs import extract_3rscan_edge_features

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_N_SCENES = 800
OUTPUT_DIR = "D:/logicsplat_data/synthetic_v2"

# Table dimensions (meters)
TABLE_WIDTH = 0.80   # X axis
TABLE_DEPTH = 0.60   # Y axis
TABLE_Z = 0.0        # table surface at Z=0

# Object size ranges (meters) — realistic tabletop items
OBJECT_SIZES = {
    "cup":     {"w": (0.06, 0.09), "d": (0.06, 0.09), "h": (0.08, 0.12)},
    "bottle":  {"w": (0.05, 0.08), "d": (0.05, 0.08), "h": (0.15, 0.25)},
    "book":    {"w": (0.12, 0.18), "d": (0.08, 0.14), "h": (0.02, 0.04)},
    "phone":   {"w": (0.06, 0.08), "d": (0.03, 0.04), "h": (0.007, 0.01)},
    "box":     {"w": (0.08, 0.15), "d": (0.06, 0.12), "h": (0.05, 0.12)},
    "plate":   {"w": (0.15, 0.25), "d": (0.15, 0.25), "h": (0.015, 0.03)},
    "bowl":    {"w": (0.10, 0.16), "d": (0.10, 0.16), "h": (0.05, 0.08)},
    "laptop":  {"w": (0.25, 0.35), "d": (0.18, 0.25), "h": (0.015, 0.025)},
    "mug":     {"w": (0.07, 0.09), "d": (0.07, 0.09), "h": (0.08, 0.11)},
    "remote":  {"w": (0.04, 0.05), "d": (0.02, 0.03), "h": (0.15, 0.20)},
}

OBJECT_TYPES = list(OBJECT_SIZES.keys())

# Placement strategy probabilities
# Heavily boosted stacking to generate on_top_of/under training signal — the
# primary gap in the cross-domain failure. More stacking = more contact examples.
P_FLAT = 0.30
P_STACK = 0.50
P_CLUSTER = 0.20

# Noise parameters (mimics real Gaussian clustering errors)
CENTROID_JITTER_STD = 0.02       # ±2cm
BBOX_EXPANSION_RANGE = (0.9, 1.3)
Z_DRIFT_STD = 0.01              # ±1cm
P_OBJECT_MERGE = 0.05           # 5% chance per pair
P_OBJECT_DROP = 0.05            # 5% chance per object


# ── Object generation ─────────────────────────────────────────────────────────

def random_object(rng: np.random.Generator, uid: int) -> Tuple[Object3D, str]:
    """Create a random tabletop object with realistic dimensions."""
    obj_type = rng.choice(OBJECT_TYPES)
    spec = OBJECT_SIZES[obj_type]

    w = rng.uniform(*spec["w"])
    d = rng.uniform(*spec["d"])
    h = rng.uniform(*spec["h"])

    # Random rotation around Z axis (simplified: swap w/d with 50% chance)
    if rng.random() < 0.5:
        w, d = d, w

    return w, d, h, obj_type


def place_objects_on_table(rng: np.random.Generator, n_objects: int) -> List[Dict]:
    """
    Place n_objects on a table with physics-aware strategies.
    Returns list of dicts with 'bbox_min', 'bbox_max', 'centroid', 'label', 'point_count'.
    """
    placed = []

    for i in range(n_objects):
        w, d, h, obj_type = random_object(rng, i)

        # Decide placement strategy
        roll = rng.random()
        if roll < P_FLAT or len(placed) == 0:
            # Flat placement on table
            z_base = TABLE_Z
            x = rng.uniform(w / 2, TABLE_WIDTH - w / 2)
            y = rng.uniform(d / 2, TABLE_DEPTH - d / 2)

        elif roll < P_FLAT + P_STACK and len(placed) > 0:
            # Stack on top of a random existing object
            support_idx = rng.integers(0, len(placed))
            support = placed[support_idx]
            z_base = support["bbox_max"][2]
            # Place centroid within support's XY footprint
            sx_min, sy_min = support["bbox_min"][0], support["bbox_min"][1]
            sx_max, sy_max = support["bbox_max"][0], support["bbox_max"][1]
            x_lo = max(w / 2, sx_min + w / 4)
            x_hi = min(TABLE_WIDTH - w / 2, sx_max - w / 4)
            y_lo = max(d / 2, sy_min + d / 4)
            y_hi = min(TABLE_DEPTH - d / 2, sy_max - d / 4)
            # If support is too small, just center on it
            if x_lo >= x_hi:
                x = (sx_min + sx_max) / 2
            else:
                x = rng.uniform(x_lo, x_hi)
            if y_lo >= y_hi:
                y = (sy_min + sy_max) / 2
            else:
                y = rng.uniform(y_lo, y_hi)

        else:
            # Cluster: place very close to an existing object
            neighbor_idx = rng.integers(0, len(placed))
            neighbor = placed[neighbor_idx]
            z_base = TABLE_Z
            # Place adjacent (within 2cm gap)
            gap = rng.uniform(0.005, 0.02)
            side = rng.integers(0, 4)  # left, right, front, back
            nc = neighbor["centroid"]
            ns = neighbor["bbox_max"] - neighbor["bbox_min"]
            if side == 0:    # left
                x = nc[0] - ns[0] / 2 - w / 2 - gap
                y = nc[1]
            elif side == 1:  # right
                x = nc[0] + ns[0] / 2 + w / 2 + gap
                y = nc[1]
            elif side == 2:  # front
                x = nc[0]
                y = nc[1] - ns[1] / 2 - d / 2 - gap
            else:            # back
                x = nc[0]
                y = nc[1] + ns[1] / 2 + d / 2 + gap

        # Clamp to table bounds
        x = np.clip(x, w / 2, TABLE_WIDTH - w / 2)
        y = np.clip(y, d / 2, TABLE_DEPTH - d / 2)

        bbox_min = np.array([x - w / 2, y - d / 2, z_base], dtype=np.float32)
        bbox_max = np.array([x + w / 2, y + d / 2, z_base + h], dtype=np.float32)
        centroid = (bbox_min + bbox_max) / 2.0

        placed.append({
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "centroid": centroid,
            "label": obj_type,
            "point_count": rng.integers(500, 5000),
        })

    # Collision resolution: push overlapping objects apart in XY
    for iteration in range(10):
        moved = False
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                a, b = placed[i], placed[j]
                # Check XY overlap
                ox = min(a["bbox_max"][0], b["bbox_max"][0]) - max(a["bbox_min"][0], b["bbox_min"][0])
                oy = min(a["bbox_max"][1], b["bbox_max"][1]) - max(a["bbox_min"][1], b["bbox_min"][1])
                # Check Z overlap (only resolve if on same level)
                oz = min(a["bbox_max"][2], b["bbox_max"][2]) - max(a["bbox_min"][2], b["bbox_min"][2])
                if ox > 0 and oy > 0 and oz > 0:
                    # Push apart along the axis with less overlap
                    size_a = a["bbox_max"] - a["bbox_min"]
                    size_b = b["bbox_max"] - b["bbox_min"]
                    if ox < oy:
                        shift = (ox / 2 + 0.005)
                        a["bbox_min"][0] -= shift
                        a["bbox_max"][0] -= shift
                        b["bbox_min"][0] += shift
                        b["bbox_max"][0] += shift
                    else:
                        shift = (oy / 2 + 0.005)
                        a["bbox_min"][1] -= shift
                        a["bbox_max"][1] -= shift
                        b["bbox_min"][1] += shift
                        b["bbox_max"][1] += shift
                    a["centroid"] = (a["bbox_min"] + a["bbox_max"]) / 2.0
                    b["centroid"] = (b["bbox_min"] + b["bbox_max"]) / 2.0
                    moved = True
        if not moved:
            break

    return placed


# ── Noise injection ───────────────────────────────────────────────────────────

def inject_noise(
    objects: List[Dict],
    rng: np.random.Generator,
) -> List[Dict]:
    """
    Apply realistic noise to simulate Gaussian clustering errors.
    Returns a NEW list of (possibly merged/dropped) noisy objects.
    """
    noisy = []

    # Object drop (5% chance per object)
    for obj in objects:
        if rng.random() < P_OBJECT_DROP and len(objects) > 2:
            continue  # drop this object
        noisy.append(obj.copy())

    if len(noisy) < 2:
        noisy = [obj.copy() for obj in objects[:2]]

    # Object merge (5% chance per adjacent pair)
    merged_indices = set()
    for i in range(len(noisy)):
        if i in merged_indices:
            continue
        for j in range(i + 1, len(noisy)):
            if j in merged_indices:
                continue
            # Only merge if close
            dist = np.linalg.norm(noisy[i]["centroid"] - noisy[j]["centroid"])
            if dist < 0.15 and rng.random() < P_OBJECT_MERGE:
                # Merge j into i
                noisy[i]["bbox_min"] = np.minimum(noisy[i]["bbox_min"], noisy[j]["bbox_min"])
                noisy[i]["bbox_max"] = np.maximum(noisy[i]["bbox_max"], noisy[j]["bbox_max"])
                noisy[i]["centroid"] = (noisy[i]["bbox_min"] + noisy[i]["bbox_max"]) / 2.0
                noisy[i]["point_count"] += noisy[j]["point_count"]
                noisy[i]["label"] = noisy[i]["label"] + "+" + noisy[j]["label"]
                merged_indices.add(j)

    noisy = [noisy[i] for i in range(len(noisy)) if i not in merged_indices]

    # Per-object noise
    for obj in noisy:
        # Centroid jitter: ±2cm Gaussian
        jitter = rng.normal(0, CENTROID_JITTER_STD, size=3).astype(np.float32)
        obj["centroid"] = obj["centroid"] + jitter

        # Bbox expansion: multiply size by uniform(0.9, 1.3)
        size = obj["bbox_max"] - obj["bbox_min"]
        scale = rng.uniform(*BBOX_EXPANSION_RANGE, size=3).astype(np.float32)
        new_size = size * scale
        obj["bbox_min"] = obj["centroid"] - new_size / 2
        obj["bbox_max"] = obj["centroid"] + new_size / 2

        # Z drift: ±1cm
        z_drift = rng.normal(0, Z_DRIFT_STD)
        obj["bbox_min"][2] += z_drift
        obj["bbox_max"][2] += z_drift
        obj["centroid"][2] += z_drift

    return noisy


# ── Feature computation ───────────────────────────────────────────────────────

def _dict_to_object3d(obj: Dict, uid: int) -> Object3D:
    """Convert a synthetic object dict to Object3D for feature extraction."""
    o = Object3D(
        uid=uid,
        centroid=obj["centroid"].copy(),
        bbox_min=obj["bbox_min"].copy(),
        bbox_max=obj["bbox_max"].copy(),
        color=np.array([128, 128, 128], dtype=np.uint8),
        point_count=int(obj["point_count"]),
        label=obj.get("label", "object"),
    )
    # Attach dummy opacity/eigenvalue attributes required by extract_gaussian_node_features
    o._mean_opacity = 0.8
    o._eigenvalues = np.array([1.0, 0.5, 0.2])
    o._mean_cov = np.zeros(6, dtype=np.float32)
    return o


# ── GT relation computation ───────────────────────────────────────────────────

def compute_gt_relations(objects: List[Dict]) -> torch.Tensor:
    """
    Compute ground truth multi-hot edge labels using the SAME logic as
    derive_relations() + compute_scene_context() from geometry.py.

    CRITICAL: Uses the NOISY bboxes, not the clean ones.
    """
    n = len(objects)
    n_edges = n * (n - 1)
    labels = torch.zeros(n_edges, NUM_RELATIONS, dtype=torch.float32)

    # Compute scene context (adaptive thresholds)
    all_mins = np.stack([o["bbox_min"] for o in objects])
    all_maxs = np.stack([o["bbox_max"] for o in objects])
    scene_ctx = compute_scene_context(all_mins, all_maxs)

    # Derive relations for all directed pairs
    edge_idx = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rels = derive_relations(
                a_min=objects[i]["bbox_min"],
                a_max=objects[i]["bbox_max"],
                b_min=objects[j]["bbox_min"],
                b_max=objects[j]["bbox_max"],
                scene_context=scene_ctx,
            )
            for rel in rels:
                labels[edge_idx, int(rel)] = 1.0
            edge_idx += 1

    return labels


# ── Scene generation pipeline ─────────────────────────────────────────────────

def generate_scene(scene_idx: int, rng: np.random.Generator,
                   n_objects_fixed: int = None,
                   z_invert: bool = False) -> Dict:
    """
    Generate one synthetic tabletop scene graph.

    z_invert: if True, negate all Z coordinates AFTER computing GT labels.
    This mimics tabletop Gaussian splats where the camera is above, making
    higher physical objects appear at more-negative Z (Y-down convention).
    GT labels are computed from normal-Z geometry and remain correct.

    Returns a dict compatible with PyG Data format.
    """
    n_objects = n_objects_fixed if n_objects_fixed else rng.integers(4, 8)

    # Step 1: Place objects with physics
    clean_objects = place_objects_on_table(rng, n_objects)

    # Step 2: Inject noise (mimics Gaussian clustering errors)
    noisy_objects = inject_noise(clean_objects, rng)
    n = len(noisy_objects)

    # Step 3a: Compute GT labels from CORRECT-Z geometry (BEFORE any Z inversion)
    edge_label = compute_gt_relations(noisy_objects)

    # Step 3b: Optionally invert Z to mimic tabletop Gaussian splat Y-down convention.
    # Physically higher objects get MORE NEGATIVE Z after inversion.
    # GT labels remain correct (derived from normal-Z above).
    if z_invert:
        for obj in noisy_objects:
            obj["centroid"] = obj["centroid"].copy()
            obj["centroid"][2] = -obj["centroid"][2]
            min_z = float(obj["bbox_min"][2])
            max_z = float(obj["bbox_max"][2])
            obj["bbox_min"] = obj["bbox_min"].copy()
            obj["bbox_max"] = obj["bbox_max"].copy()
            obj["bbox_min"][2] = -max_z
            obj["bbox_max"][2] = -min_z

    # Step 3c: Compute scene extent from (possibly Z-inverted) objects
    all_mins = np.stack([o["bbox_min"] for o in noisy_objects])
    all_maxs = np.stack([o["bbox_max"] for o in noisy_objects])
    scene_min = all_mins.min(axis=0)
    scene_max = all_maxs.max(axis=0)
    scene_extent = np.maximum(scene_max - scene_min, 1e-6)

    # Convert to Object3D for feature extraction (uses exact same functions as training)
    obj3d_list = [_dict_to_object3d(obj, uid=i) for i, obj in enumerate(noisy_objects)]

    # Compute scene-level statistics for scale-invariant node features
    obj_sizes = [np.maximum(o.size, 1e-6) for o in obj3d_list]
    obj_diags = [float(np.linalg.norm(s)) for s in obj_sizes]
    scene_mean_diag = float(np.mean(obj_diags)) if obj_diags else 1.0
    obj_volumes = [float(np.prod(s)) for s in obj_sizes]
    scene_median_volume = float(np.median(obj_volumes)) if obj_volumes else 1.0

    centroid_zs = np.array([o.centroid[2] for o in obj3d_list])
    sorted_z_idx = np.argsort(centroid_zs)
    z_ranks = np.zeros(n)
    if n > 1:
        for rank, obj_idx in enumerate(sorted_z_idx):
            z_ranks[obj_idx] = rank / (n - 1)
    else:
        z_ranks[0] = 0.5

    # Step 4: Compute node features (10-dim, scale-invariant — matches training exactly)
    x = np.stack([
        extract_gaussian_node_features(
            obj3d_list[i], scene_extent, scene_min,
            scene_mean_diag=scene_mean_diag,
            scene_median_volume=scene_median_volume,
            z_rank=float(z_ranks[i]),
        )
        for i in range(n)
    ])

    # Step 5: Compute edge features (22-dim — MATCHES extract_3rscan_edge_features exactly)
    edge_feats = []
    src_list, dst_list = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                edge_feats.append(
                    extract_3rscan_edge_features(obj3d_list[i], obj3d_list[j], scene_extent)
                )
                src_list.append(i)
                dst_list.append(j)

    edge_attr = np.stack(edge_feats)
    edge_index = np.array([src_list, dst_list], dtype=np.int64)

    # Step 6: GT labels already computed in step 3a (before Z inversion)

    # Step 7: Build output dict
    scene_id = f"synth_{scene_idx:03d}"
    obj_labels = [o["label"] for o in noisy_objects]

    return {
        "x": torch.tensor(x, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
        "edge_attr": torch.tensor(edge_attr, dtype=torch.float32),
        "edge_label": edge_label,
        "scene_id": scene_id,
        "obj_labels": obj_labels,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic tabletop scene graphs for GNN fine-tuning"
    )
    parser.add_argument("--n-scenes", type=int, default=DEFAULT_N_SCENES,
                        help=f"Number of scenes to generate (default: {DEFAULT_N_SCENES})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--n-objects", type=int, default=None,
                        help="Fixed number of objects per scene (default: random 4-7)")
    parser.add_argument("--z-invert", action="store_true",
                        help="Negate Z coords after GT label generation (mimics Y-down splats)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"\n{'='*60}")
    print("  LogicSplat — Synthetic Tabletop Scene Generation")
    print(f"{'='*60}")
    print(f"  Scenes to generate: {args.n_scenes}")
    print(f"  Output directory:   {args.output_dir}")
    print(f"  Seed:               {args.seed}")

    # Statistics tracking
    total_objects = 0
    total_edges = 0
    total_positive_labels = 0
    relation_counts = np.zeros(NUM_RELATIONS, dtype=int)

    if args.z_invert:
        print(f"  Z-inversion:     ENABLED (GT computed before inversion)")

    for i in range(1, args.n_scenes + 1):
        scene = generate_scene(i, rng, n_objects_fixed=args.n_objects, z_invert=args.z_invert)
        save_path = os.path.join(args.output_dir, f"synth_{i:03d}.pt")
        torch.save(scene, save_path)

        n_obj = scene["x"].shape[0]
        n_edges = scene["edge_index"].shape[1]
        n_pos = int(scene["edge_label"].sum())
        total_objects += n_obj
        total_edges += n_edges
        total_positive_labels += n_pos

        # Count per-relation
        for r in range(NUM_RELATIONS):
            relation_counts[r] += int(scene["edge_label"][:, r].sum())

        if i % 50 == 0 or i == args.n_scenes:
            print(f"  Generated {i}/{args.n_scenes} scenes "
                  f"(last: {n_obj} objs, {n_edges} edges, {n_pos} pos labels)")

    # Summary
    print(f"\n{'─'*60}")
    print(f"  SUMMARY")
    print(f"{'─'*60}")
    print(f"  Total scenes:    {args.n_scenes}")
    print(f"  Total objects:   {total_objects} (avg {total_objects/args.n_scenes:.1f}/scene)")
    print(f"  Total edges:     {total_edges} (avg {total_edges/args.n_scenes:.1f}/scene)")
    print(f"  Total pos labels:{total_positive_labels}")
    print(f"\n  Per-relation counts:")
    for r in range(NUM_RELATIONS):
        print(f"    {RELATION_NAMES[r]:20s}: {relation_counts[r]:5d}")
    print(f"\n  Files saved to: {args.output_dir}/synth_001.pt .. synth_{args.n_scenes:03d}.pt")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
