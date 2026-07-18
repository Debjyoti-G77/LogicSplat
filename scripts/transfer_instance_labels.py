"""
Transfer instance segmentation labels from 3RScan meshes to Gaussian splat centers.

For each scene with both a mesh segmentation PLY and a Gaussian splat PLY:
  1. Load per-vertex instance labels from the annotated mesh
  2. Load Gaussian splat centers from the trained 3DGS point cloud
  3. Build a KD-tree from mesh vertices, query nearest vertex for each Gaussian
  4. Assign the nearest vertex's instance ID to each Gaussian
  5. Mark Gaussians with distance > 0.05m as unlabeled (-1)
  6. Save labels + distances to instance_labels.npz

Resume support: skips scenes where instance_labels.npz already exists.

Usage:
    python scripts/transfer_instance_labels.py
    python scripts/transfer_instance_labels.py --max-scenes 10
    python scripts/transfer_instance_labels.py --force  # re-process all
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from plyfile import PlyData
from scipy.spatial import cKDTree

from src.gaussian.loader import load_gaussian_ply


# === Configuration ===
MESHES_DIR = Path(r"D:\3rscan_meshes")
SPLATS_DIR = Path(r"D:\3rscan_splats")
RELATIONSHIPS_JSON = Path(__file__).parent.parent / "data" / "3DSSG" / "relationships.json"
DISTANCE_THRESHOLD = 0.05  # meters — Gaussians farther than this are unlabeled


def get_ssg_scene_ids() -> list:
    """Extract unique scene IDs from relationships.json (sorted)."""
    with open(RELATIONSHIPS_JSON, "r") as f:
        data = json.load(f)
    return sorted(set(s["scan"] for s in data["scans"]))


def load_instance_mesh(ply_path: str) -> tuple:
    """
    Load per-vertex XYZ + instance label from annotated mesh PLY.

    Returns:
        positions: (N, 3) float32
        instance_ids: (N,) int32
    """
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    names = [p.name for p in v.properties]

    positions = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    # Find instance label field — 3RScan uses 'objectId' or 'label'
    instance_field = None
    for field in ("objectId", "object_id", "instance", "label"):
        if field in names:
            instance_field = field
            break

    if instance_field is None:
        raise ValueError(
            f"No instance label field found in {ply_path}. "
            f"Available fields: {names}"
        )

    instance_ids = np.array(v[instance_field], dtype=np.int32)
    return positions, instance_ids


def transfer_labels_for_scene(
    scene_id: str,
    meshes_dir: Path,
    splats_dir: Path,
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> dict:
    """
    Transfer instance labels from mesh to Gaussian splat for one scene.

    Returns:
        dict with keys: labels, distances, n_instances, coverage, n_gaussians
        or None if files are missing / processing fails.
    """
    # Paths
    mesh_ply = meshes_dir / scene_id / "labels.instances.annotated.v2.ply"
    splat_ply = splats_dir / scene_id / "ckpts" / "point_cloud_30000.ply"

    if not mesh_ply.exists():
        return None
    if not splat_ply.exists():
        return None

    # Load mesh with instance labels
    mesh_positions, mesh_instance_ids = load_instance_mesh(str(mesh_ply))

    # Load Gaussian splat
    cloud = load_gaussian_ply(str(splat_ply))
    gaussian_xyz = cloud.xyz  # (N, 3)

    # Build KD-tree from mesh vertices
    tree = cKDTree(mesh_positions)

    # Query nearest mesh vertex for each Gaussian center
    distances, indices = tree.query(gaussian_xyz, k=1)

    # Assign instance labels from nearest mesh vertex
    labels = mesh_instance_ids[indices]

    # Mark Gaussians too far from any mesh vertex as unlabeled
    labels[distances > distance_threshold] = -1

    # Compute stats
    n_labeled = int((labels != -1).sum())
    n_total = len(labels)
    coverage = n_labeled / max(n_total, 1)
    unique_instances = set(labels[labels != -1].tolist())
    n_instances = len(unique_instances)

    return {
        "labels": labels.astype(np.int32),
        "distances": distances.astype(np.float32),
        "n_instances": n_instances,
        "coverage": coverage,
        "n_gaussians": n_total,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Transfer instance labels from 3RScan meshes to Gaussian splats"
    )
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Limit number of scenes to process")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if .npz already exists")
    parser.add_argument("--meshes-dir", type=str, default=None,
                        help="Override meshes directory")
    parser.add_argument("--splats-dir", type=str, default=None,
                        help="Override splats directory")
    parser.add_argument("--threshold", type=float, default=DISTANCE_THRESHOLD,
                        help=f"Distance threshold in meters (default: {DISTANCE_THRESHOLD})")
    args = parser.parse_args()

    meshes_dir = Path(args.meshes_dir) if args.meshes_dir else MESHES_DIR
    splats_dir = Path(args.splats_dir) if args.splats_dir else SPLATS_DIR

    print("=" * 60)
    print("Instance Label Transfer: 3RScan Mesh → Gaussian Splat")
    print("=" * 60)
    print(f"Meshes dir:  {meshes_dir}")
    print(f"Splats dir:  {splats_dir}")
    print(f"Threshold:   {args.threshold:.3f} m")
    print()

    # Get scene list
    print("[1/3] Loading scene IDs from relationships.json...")
    scene_ids = get_ssg_scene_ids()
    print(f"  Found {len(scene_ids)} scenes in 3DSSG annotations")

    if args.max_scenes:
        scene_ids = scene_ids[:args.max_scenes]
        print(f"  Limited to {args.max_scenes} scenes")

    # Process scenes
    print(f"\n[2/3] Transferring instance labels...")
    start_time = time.time()

    n_processed = 0
    n_skipped_exists = 0
    n_skipped_missing = 0
    n_failed = 0
    coverages = []
    instance_counts = []

    for i, scene_id in enumerate(scene_ids):
        output_path = splats_dir / scene_id / "instance_labels.npz"

        # Resume support
        if output_path.exists() and not args.force:
            n_skipped_exists += 1
            continue

        # Process
        result = transfer_labels_for_scene(
            scene_id, meshes_dir, splats_dir, args.threshold
        )

        if result is None:
            n_skipped_missing += 1
            continue

        # Save
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                str(output_path),
                labels=result["labels"],
                distances=result["distances"],
                n_instances=np.array(result["n_instances"]),
                coverage=np.array(result["coverage"]),
            )
            n_processed += 1
            coverages.append(result["coverage"])
            instance_counts.append(result["n_instances"])

            if n_processed % 20 == 0 or n_processed == 1:
                elapsed = time.time() - start_time
                rate = n_processed / elapsed if elapsed > 0 else 0
                print(
                    f"  [{i+1}/{len(scene_ids)}] Processed {n_processed} scenes "
                    f"({rate:.1f} scenes/s) | "
                    f"Last: {result['n_instances']} instances, "
                    f"{result['coverage']*100:.1f}% coverage"
                )
        except Exception as e:
            n_failed += 1
            print(f"  ERROR saving {scene_id}: {e}")

    elapsed = time.time() - start_time

    # Report
    print(f"\n[3/3] Summary")
    print("=" * 60)
    print(f"Total scenes in 3DSSG:     {len(scene_ids)}")
    print(f"Processed (new):           {n_processed}")
    print(f"Skipped (already exists):  {n_skipped_exists}")
    print(f"Skipped (missing files):   {n_skipped_missing}")
    print(f"Failed:                    {n_failed}")
    print(f"Time:                      {elapsed:.1f}s")

    if coverages:
        print(f"\nCoverage stats (newly processed):")
        print(f"  Mean:   {np.mean(coverages)*100:.1f}%")
        print(f"  Median: {np.median(coverages)*100:.1f}%")
        print(f"  Min:    {np.min(coverages)*100:.1f}%")
        print(f"  Max:    {np.max(coverages)*100:.1f}%")
        print(f"  >90%:   {sum(1 for c in coverages if c > 0.9)} scenes")
        print(f"  >50%:   {sum(1 for c in coverages if c > 0.5)} scenes")

    if instance_counts:
        print(f"\nInstance count stats:")
        print(f"  Mean:   {np.mean(instance_counts):.1f}")
        print(f"  Median: {np.median(instance_counts):.0f}")
        print(f"  Min:    {np.min(instance_counts)}")
        print(f"  Max:    {np.max(instance_counts)}")


if __name__ == "__main__":
    main()
