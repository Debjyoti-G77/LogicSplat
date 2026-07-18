"""
Auto-annotate OCTScenes with geometry-derived relations (pseudo-labeling).

For each OCTScene with splat.ply:
    1. Load splat.ply, cluster, Z-flip (same pipeline as inference)
    2. Compute scene_context using compute_scene_context()
    3. Run derive_relations() on all object pairs
    4. Save annotation_template.json with relations pre-filled

The "auto": true flag marks machine-generated relations pending human verification.
The user will:
    - Look at preview.png to identify objects (replace "object_0" with real names)
    - Verify each relation (remove wrong ones, add missing ones)
    - Change annotation_status to "verified"

Usage:
    python scripts/auto_annotate_octscenes.py
    python scripts/auto_annotate_octscenes.py --scenes oct_01 oct_05
    python scripts/auto_annotate_octscenes.py --overwrite
"""
import sys
sys.path.insert(0, ".")

import os
import json
import argparse
import numpy as np
from typing import List, Optional

from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
from src.gaussian.clustering import gaussian_to_objects
from src.relations.geometry import compute_scene_context, derive_relations
from src.relations.schema import RELATION_NAMES, Relation
from src.graph.definitions import Object3D


# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "D:/logicsplat_data/octscenes"

# Relations we emit for annotation (tabletop-relevant subset)
ANNOTATION_RELATIONS = {
    "on_top_of",
    "under",
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
    "higher_than",
    "lower_than",
    "adjacent_to",
}


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


def process_scene(scene_id: str, target_min: int = 5, target_max: int = 12) -> Optional[dict]:
    """
    Load splat.ply, cluster objects, Z-flip, derive relations.

    Uses the same pipeline as src/inference/gaussian_inference.py:
        load → filter (opacity > 0.1) → SOR prune → HDBSCAN cluster → Z-flip
    Then runs geometry-based relation derivation on all pairs.
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
        target_min=target_min,
        target_max=target_max,
    )
    print(f"    Clusters found: {len(objects)}")

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

    # ── Compute scene context for adaptive thresholds ─────────────────────────
    all_bbox_mins = np.stack([o.bbox_min for o in objects])
    all_bbox_maxs = np.stack([o.bbox_max for o in objects])
    scene_ctx = compute_scene_context(all_bbox_mins, all_bbox_maxs)
    print(f"    Scene context: z_min={scene_ctx['z_min_threshold']:.4f}, "
          f"dominance={scene_ctx['dominance_ratio']:.2f}")

    # ── Derive relations for all pairs ────────────────────────────────────────
    relations = []
    seen_keys = set()

    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i == j:
                continue
            derived = derive_relations(
                a.bbox_min, a.bbox_max,
                b.bbox_min, b.bbox_max,
                scene_context=scene_ctx,
            )
            for rel in derived:
                rel_name = RELATION_NAMES[int(rel)]
                if rel_name not in ANNOTATION_RELATIONS:
                    continue
                key = (i, rel_name, j)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                relations.append({
                    "subject": f"object_{i}",
                    "relation": rel_name,
                    "object": f"object_{j}",
                    "auto": True,
                })

    print(f"    Auto-derived relations: {len(relations)}")

    return {
        "scene_id": scene_id,
        "objects": objects,
        "relations": relations,
        "params": params,
        "scene_context": scene_ctx,
    }


def generate_auto_annotation(scene_id: str, objects: List[Object3D],
                             relations: list) -> dict:
    """Generate the annotation JSON with pre-filled relations."""
    template = {
        "scene_id": scene_id,
        "objects": [],
        "relations": relations,
        "annotation_status": "auto_generated_needs_verification",
    }

    for obj in objects:
        template["objects"].append({
            "id": obj.uid,
            "name": f"object_{obj.uid}",
            "centroid": [round(float(c), 4) for c in obj.centroid],
            "point_count": obj.point_count,
            "color": [int(c) for c in obj.color],
        })

    return template


def generate_annotation_guide(scene_id: str, objects: List[Object3D],
                              relations: list) -> str:
    """Generate the text annotation guide for the user."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"Scene: {scene_id}")
    lines.append(f"Preview: preview.png")
    lines.append(f"Objects found: {len(objects)}")
    lines.append(f"Auto-derived relations: {len(relations)}")
    lines.append(f"Status: AUTO-GENERATED — NEEDS VERIFICATION")
    lines.append("=" * 60)
    lines.append("")

    lines.append("OBJECTS:")
    lines.append("-" * 40)
    for obj in objects:
        size = obj.size
        cx, cy, cz = obj.centroid
        r, g, b = obj.color
        lines.append(
            f"  object_{obj.uid}: {obj.point_count} pts, "
            f"color=rgb({r},{g},{b}), "
            f"centroid=({cx:.2f},{cy:.2f},{cz:.2f}), "
            f"size=({size[0]:.2f},{size[1]:.2f},{size[2]:.2f})"
        )

    lines.append("")
    lines.append("AUTO-DERIVED RELATIONS:")
    lines.append("-" * 40)
    for rel in relations:
        lines.append(f"  {rel['subject']} --[{rel['relation']}]--> {rel['object']}")

    lines.append("")
    lines.append("INSTRUCTIONS:")
    lines.append("-" * 40)
    lines.append("1. Open preview.png to identify objects by color/position")
    lines.append("2. In annotation_template.json:")
    lines.append("   a. Replace 'object_0', 'object_1', etc. with real names")
    lines.append("      (e.g., 'cup', 'book', 'keyboard')")
    lines.append("   b. Update 'subject' and 'object' fields in relations to match")
    lines.append("   c. Remove incorrect relations (delete the entry)")
    lines.append("   d. Add missing relations")
    lines.append("   e. Change 'annotation_status' to 'verified'")
    lines.append("")
    lines.append("VALID RELATION TYPES:")
    lines.append("  " + ", ".join(sorted(ANNOTATION_RELATIONS)))
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-annotate OCTScenes with geometry-derived relations"
    )
    parser.add_argument(
        "--scenes", nargs="+", default=None,
        help="Scene IDs to process (default: all with splat.ply)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing annotation_template.json files"
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
    print("  OCTScenes — Auto-Annotation (Geometry Rules)")
    print("=" * 60)
    print(f"  Scenes: {len(scenes)}")
    print(f"  Data dir: {os.path.abspath(DATA_DIR)}")
    print(f"  Overwrite: {args.overwrite}")

    success_count = 0
    failed = []
    total_relations = 0

    for scene_id in scenes:
        print(f"\n{'─'*40}")
        print(f"  Processing: {scene_id}")
        print(f"{'─'*40}")

        scene_dir = os.path.join(DATA_DIR, scene_id)
        template_path = os.path.join(scene_dir, "annotation_template.json")

        # Check if already annotated (verified by user)
        if os.path.exists(template_path) and not args.overwrite:
            with open(template_path) as f:
                existing = json.load(f)
            status = existing.get("annotation_status", "")
            if status == "verified":
                print(f"    Already verified — skipping (use --overwrite to regenerate)")
                success_count += 1
                continue

        try:
            result = process_scene(scene_id, target_min=args.target_min, target_max=args.target_max)
            if result is None:
                failed.append(scene_id)
                continue

            objects = result["objects"]
            relations = result["relations"]
            total_relations += len(relations)

            # Generate annotation template JSON
            template = generate_auto_annotation(scene_id, objects, relations)
            with open(template_path, "w") as f:
                json.dump(template, f, indent=2)
            print(f"    Saved: {template_path}")

            # Generate annotation guide
            guide = generate_annotation_guide(scene_id, objects, relations)
            guide_path = os.path.join(scene_dir, "annotation_guide.txt")
            with open(guide_path, "w") as f:
                f.write(guide)
            print(f"    Saved: {guide_path}")

            success_count += 1

        except Exception as e:
            import traceback
            print(f"    ERROR: {e}")
            traceback.print_exc()
            failed.append(scene_id)
            continue

    # Summary
    print(f"\n{'='*60}")
    print("  Auto-Annotation Complete")
    print(f"{'='*60}")
    print(f"  Successful: {success_count}/{len(scenes)}")
    print(f"  Total auto-derived relations: {total_relations}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"\n  Output files:")
    print(f"    data/octscenes/oct_XX/annotation_template.json")
    print(f"    data/octscenes/oct_XX/annotation_guide.txt")
    print(f"\n  Next steps:")
    print(f"    1. Open each preview.png to identify objects")
    print(f"    2. Edit annotation_template.json:")
    print(f"       - Replace object_N names with real names")
    print(f"       - Verify/correct auto-derived relations")
    print(f"       - Set annotation_status to 'verified'")
    print(f"    3. After all scenes verified, run:")
    print(f"       python scripts/finetune_tabletop.py")


if __name__ == "__main__":
    main()
