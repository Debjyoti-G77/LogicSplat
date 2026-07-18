"""
Prepare OCTScenes for manual relation annotation.

For each processed OCTScene that has splat.ply:
    1. Load splat.ply, filter, prune, cluster (same pipeline as inference)
    2. Apply Z-flip (negate Z so +Z = up)
    3. Record per-cluster: centroid, point_count, color, size
    4. Generate empty annotation template JSON
    5. Generate text annotation guide for the user

Output:
    data/octscenes/oct_XX/annotation_template.json
    data/octscenes/oct_XX/annotation_guide.txt

Usage:
    python scripts/prepare_annotation.py
    python scripts/prepare_annotation.py --scenes oct_01 oct_05
"""
import sys
sys.path.insert(0, ".")

import os
import json
import argparse
import numpy as np
from typing import List

from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
from src.gaussian.clustering import gaussian_to_objects
from src.graph.definitions import Object3D


# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "D:/logicsplat_data/octscenes"

RELATION_TYPES = [
    "on_top_of",
    "under",
    "to_the_left_of",
    "to_the_right_of",
    "in_front_of",
    "behind",
    "higher_than",
    "lower_than",
    "adjacent_to",
]


def get_scene_ids() -> list:
    """Find all oct_XX scene directories that have splat.ply."""
    if not os.path.isdir(DATA_DIR):
        return []
    scenes = sorted([
        d for d in os.listdir(DATA_DIR)
        if d.startswith("oct_") and os.path.isdir(os.path.join(DATA_DIR, d))
        and os.path.exists(os.path.join(DATA_DIR, d, "splat.ply"))
    ])
    return scenes


def process_scene(scene_id: str) -> dict:
    """
    Load splat.ply, cluster objects, apply Z-flip, return object info.

    Uses the same pipeline as src/inference/gaussian_inference.py:
        load → filter (opacity > 0.1) → SOR prune → HDBSCAN cluster → Z-flip
    """
    scene_dir = os.path.join(DATA_DIR, scene_id)
    ply_path = os.path.join(scene_dir, "splat.ply")

    print(f"\n  Loading: {ply_path}")
    cloud = load_gaussian_ply(ply_path)
    print(f"    Raw Gaussians: {cloud.num_gaussians:,}")

    # Filter by opacity
    cloud_filtered = filter_gaussians(cloud, opacity_threshold=0.1)
    print(f"    After opacity filter: {cloud_filtered.num_gaussians:,}")

    # Statistical Outlier Removal
    cloud_filtered = prune_isolated_gaussians(cloud_filtered, nb_neighbors=20, std_ratio=2.0)
    print(f"    After SOR prune: {cloud_filtered.num_gaussians:,}")

    # Cluster into objects — auto-tuner picks best clustering by quality (silhouette)
    # target_min/target_max are soft hints, not hard limits
    objects, params = gaussian_to_objects(
        cloud_filtered,
        target_min=5,
        target_max=12,
    )
    print(f"    Clusters found: {len(objects)}")
    print(f"    Params: mcs={params.get('min_cluster_size')}, "
          f"method={params.get('cluster_method')}, "
          f"noise={params.get('noise_fraction', 0):.1%}")

    if len(objects) < 2:
        print(f"    WARNING: Too few objects ({len(objects)}) — skipping")
        return None

    # ── Z-axis flip ───────────────────────────────────────────────────────────
    # Gaussian Splat scenes use Z-down. Negate Z so +Z = up (standard convention).
    for o in objects:
        o.centroid = o.centroid.copy()
        o.centroid[2] *= -1
        o.bbox_min = o.bbox_min.copy()
        o.bbox_min[2] *= -1
        o.bbox_max = o.bbox_max.copy()
        o.bbox_max[2] *= -1
        # After negation, swap min/max Z to keep min < max
        o.bbox_min[2], o.bbox_max[2] = min(o.bbox_min[2], o.bbox_max[2]), max(o.bbox_min[2], o.bbox_max[2])

    return {
        "scene_id": scene_id,
        "objects": objects,
        "params": params,
    }


def generate_annotation_template(scene_id: str, objects: List[Object3D]) -> dict:
    """Generate the empty annotation JSON template."""
    template = {
        "scene_id": scene_id,
        "objects": [],
        "relations": [],
    }

    for obj in objects:
        template["objects"].append({
            "id": obj.uid,
            "name": "FILL_IN",
            "centroid": [round(float(c), 4) for c in obj.centroid],
            "point_count": obj.point_count,
            "color": [int(c) for c in obj.color],
        })

    return template


def generate_annotation_guide(scene_id: str, objects: List[Object3D]) -> str:
    """Generate the text annotation guide for the user."""
    lines = []
    lines.append("-" * 60)
    lines.append(f"Scene: {scene_id}")
    lines.append(f"Preview: preview.png")
    lines.append(f"Objects found: {len(objects)}")
    lines.append("")

    for obj in objects:
        size = obj.size
        cx, cy, cz = obj.centroid
        r, g, b = obj.color
        lines.append(
            f"Obj{obj.uid}: {obj.point_count} pts, "
            f"color=rgb({r},{g},{b}), "
            f"centroid=({cx:.2f},{cy:.2f},{cz:.2f}), "
            f"size=({size[0]:.2f},{size[1]:.2f},{size[2]:.2f})"
        )

    lines.append("")
    lines.append("Please fill in:")
    lines.append("1. Object names (look at preview.png and color to identify)")
    lines.append("2. Relations using ONLY these types:")
    lines.append("   " + ", ".join(RELATION_TYPES))
    lines.append("")
    lines.append("Annotation format (in annotation_template.json):")
    lines.append('  "objects": [{"id": 0, "name": "cup", ...}, ...]')
    lines.append('  "relations": [')
    lines.append('    {"subject": "cup", "relation": "on_top_of", "object": "table"},')
    lines.append('    ...')
    lines.append('  ]')
    lines.append("-" * 60)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare OCTScenes for manual relation annotation"
    )
    parser.add_argument(
        "--scenes", nargs="+", default=None,
        help="Scene IDs to process (default: all with splat.ply)"
    )
    parser.add_argument(
        "--target-min", type=int, default=5,
        help="Minimum number of clusters to find (default: 5)"
    )
    parser.add_argument(
        "--target-max", type=int, default=12,
        help="Maximum number of clusters to find (default: 12)"
    )
    args = parser.parse_args()

    scenes = args.scenes or get_scene_ids()
    if not scenes:
        print("No processed OCTScenes found (need splat.ply).")
        print("Run: python scripts/process_octscenes.py")
        sys.exit(1)

    print("=" * 60)
    print("  OCTScenes — Annotation Preparation")
    print("=" * 60)
    print(f"  Scenes: {len(scenes)}")
    print(f"  Data dir: {os.path.abspath(DATA_DIR)}")

    success_count = 0
    failed = []

    for scene_id in scenes:
        print(f"\n{'─'*40}")
        print(f"  Processing: {scene_id}")
        print(f"{'─'*40}")

        try:
            result = process_scene(scene_id)
            if result is None:
                failed.append(scene_id)
                continue

            objects = result["objects"]
            scene_dir = os.path.join(DATA_DIR, scene_id)

            # Generate annotation template JSON
            template = generate_annotation_template(scene_id, objects)
            template_path = os.path.join(scene_dir, "annotation_template.json")
            with open(template_path, "w") as f:
                json.dump(template, f, indent=2)
            print(f"    Saved: {template_path}")

            # Generate annotation guide
            guide = generate_annotation_guide(scene_id, objects)
            guide_path = os.path.join(scene_dir, "annotation_guide.txt")
            with open(guide_path, "w") as f:
                f.write(guide)
            print(f"    Saved: {guide_path}")

            # Print guide to console
            print(f"\n{guide}")

            success_count += 1

        except Exception as e:
            import traceback
            print(f"    ERROR: {e}")
            traceback.print_exc()
            failed.append(scene_id)
            continue

    # Summary
    print(f"\n{'='*60}")
    print("  Annotation Preparation Complete")
    print(f"{'='*60}")
    print(f"  Successful: {success_count}/{len(scenes)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"\n  Preview images: data/octscenes/oct_XX/preview.png")
    print(f"  Templates:      data/octscenes/oct_XX/annotation_template.json")
    print(f"  Guides:         data/octscenes/oct_XX/annotation_guide.txt")
    print(f"\n  After annotating all scenes, run:")
    print(f"    python scripts/finetune_tabletop.py")


if __name__ == "__main__":
    main()
