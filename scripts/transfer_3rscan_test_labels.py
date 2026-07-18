"""
Transfer instance labels from 3RScan meshes to Gaussian splats for the RIO10 test scenes.

Same nearest-neighbor approach as transfer_instance_labels.py but adapted for the
test scene directory structure (D:\\3rscan_test\\{scene_id}\\).

For each test scene:
  1. Load per-vertex instance labels from labels.instances.annotated.v2.ply
  2. Load Gaussian splat centers from the exported splat PLY
  3. Build KD-tree from mesh vertices, query nearest vertex for each Gaussian
  4. Assign instance labels, mark distant Gaussians as unlabeled (-1)
  5. Save to D:\\3rscan_test\\{scene_id}\\instance_labels.npz

Usage:
    python scripts/transfer_3rscan_test_labels.py
    python scripts/transfer_3rscan_test_labels.py --max-scenes 5
    python scripts/transfer_3rscan_test_labels.py --force
    python scripts/transfer_3rscan_test_labels.py --threshold 0.08  # More lenient
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
INPUT_DIR = Path(r"D:\3rscan_test")
RIO10_TEST_SCENES = Path(__file__).parent.parent / "data" / "3DSSG" / "rio10_test_scenes.txt"
DISTANCE_THRESHOLD = 0.05  # meters


def load_test_scene_ids() -> list:
    """Load the 46 RIO10 test scene IDs (handles UTF-16 BOM)."""
    if not RIO10_TEST_SCENES.exists():
        print(f"ERROR: {RIO10_TEST_SCENES} not found!")
        sys.exit(1)
    raw = RIO10_TEST_SCENES.read_bytes()
    if raw[:2] == b'\xff\xfe':
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig")
    return [line.strip() for line in text.splitlines() if line.strip()]


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

    # Find instance label field
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
    scene_dir: Path,
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> dict:
    """
    Transfer instance labels from mesh to Gaussian splat for one test scene.

    Directory structure:
        scene_dir/labels.instances.annotated.v2.ply  (mesh with instance labels)
        scene_dir/splat/point_cloud.ply              (exported Gaussian splat)

    Returns:
        dict with keys: labels, distances, n_instances, coverage, n_gaussians
        or None if files are missing.
    """
    mesh_ply = scene_dir / "labels.instances.annotated.v2.ply"
    splat_ply = scene_dir / "splat" / "point_cloud.ply"

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
        description="Transfer instance labels to Gaussian splats for RIO10 test scenes"
    )
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Limit number of scenes")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if .npz already exists")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Override input directory")
    parser.add_argument("--threshold", type=float, default=DISTANCE_THRESHOLD,
                        help=f"Distance threshold in meters (default: {DISTANCE_THRESHOLD})")
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else INPUT_DIR

    print("=" * 70)
    print("Instance Label Transfer: 3RScan Mesh → Gaussian Splat (Test Scenes)")
    print("=" * 70)
    print(f"  Input dir:   {input_dir}")
    print(f"  Threshold:   {args.threshold:.3f} m")
    print()

    # Get scene list
    scene_ids = load_test_scene_ids()
    if args.max_scenes:
        scene_ids = scene_ids[:args.max_scenes]

    print(f"[1/3] Processing {len(scene_ids)} test scenes...")
    print()

    # Process scenes
    n_processed = 0
    n_skipped = 0
    n_missing = 0
    n_failed = 0
    coverages = []
    instance_counts = []
    start_time = time.time()

    for i, scene_id in enumerate(scene_ids):
        scene_dir = input_dir / scene_id
        output_path = scene_dir / "instance_labels.npz"

        # Resume support — validate existing file integrity
        if output_path.exists() and not args.force:
            try:
                # Verify the npz is not corrupt
                test_npz = np.load(str(output_path))
                _ = test_npz["labels"]
                _ = test_npz["coverage"]
                test_npz.close()
                n_skipped += 1
                continue
            except Exception:
                # Corrupt file from interrupted write — delete and redo
                output_path.unlink(missing_ok=True)

        if not scene_dir.exists():
            n_missing += 1
            continue

        # Process
        result = transfer_labels_for_scene(scene_dir, args.threshold)

        if result is None:
            n_missing += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(scene_ids)}] {scene_id}: missing files")
            continue

        # Save atomically — write to temp file then rename
        # NOTE: np.savez_compressed appends .npz if not already present,
        # so we use a .tmp base name and it creates .tmp.npz
        try:
            tmp_base = str(output_path) + ".tmp"  # will become instance_labels.npz.tmp.npz
            tmp_actual = tmp_base + ".npz"  # what numpy actually creates
            # Clean up any leftover temp from previous interrupted run
            if os.path.exists(tmp_actual):
                os.unlink(tmp_actual)
            np.savez_compressed(
                tmp_base,
                labels=result["labels"],
                distances=result["distances"],
                n_instances=np.array(result["n_instances"]),
                coverage=np.array(result["coverage"]),
            )
            # Atomic rename
            if output_path.exists():
                output_path.unlink()
            Path(tmp_actual).rename(output_path)
            n_processed += 1
            coverages.append(result["coverage"])
            instance_counts.append(result["n_instances"])

            print(
                f"  [{i+1}/{len(scene_ids)}] {scene_id}: "
                f"{result['n_instances']} instances, "
                f"{result['coverage']*100:.1f}% coverage"
            )
        except Exception as e:
            n_failed += 1
            print(f"  [{i+1}/{len(scene_ids)}] {scene_id}: ERROR - {e}")

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'=' * 70}")
    print("[3/3] SUMMARY")
    print("=" * 70)
    print(f"  Total scenes:    {len(scene_ids)}")
    print(f"  Processed (new): {n_processed}")
    print(f"  Skipped (exist): {n_skipped}")
    print(f"  Missing files:   {n_missing}")
    print(f"  Failed:          {n_failed}")
    print(f"  Time:            {elapsed:.1f}s")

    if coverages:
        print(f"\n  Coverage stats:")
        print(f"    Mean:   {np.mean(coverages)*100:.1f}%")
        print(f"    Median: {np.median(coverages)*100:.1f}%")
        print(f"    Min:    {np.min(coverages)*100:.1f}%")
        print(f"    Max:    {np.max(coverages)*100:.1f}%")

    if instance_counts:
        print(f"\n  Instance count stats:")
        print(f"    Mean:   {np.mean(instance_counts):.1f}")
        print(f"    Median: {np.median(instance_counts):.0f}")
        print(f"    Min:    {np.min(instance_counts)}")
        print(f"    Max:    {np.max(instance_counts)}")

    print()
    print("Next step: Build test graphs")
    print("  python scripts/build_3rscan_test_graphs.py")


if __name__ == "__main__":
    main()
