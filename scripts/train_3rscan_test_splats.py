"""
Train Gaussian splats via NerfStudio (splatfacto) for the 46 RIO10 test scenes.

For each scene with a valid ns_format/ directory:
  1. Run ns-train splatfacto with 30k iterations
  2. Export the Gaussian splat PLY

This is the most time-consuming step (~15-30 min per scene × 46 = 11-23 hours).
Designed to run overnight with resume support.

Output:
    D:\\3rscan_test\\{scene_id}\\splat\\
        point_cloud.ply   (exported Gaussian splat)
    D:\\3rscan_test\\{scene_id}\\ns_output\\
        splatfacto/       (NerfStudio training output + checkpoints)

Usage:
    python scripts/train_3rscan_test_splats.py
    python scripts/train_3rscan_test_splats.py --max-scenes 5
    python scripts/train_3rscan_test_splats.py --scene-id <uuid>
    python scripts/train_3rscan_test_splats.py --max-iterations 15000  # Faster training
    python scripts/train_3rscan_test_splats.py --export-only  # Just export from existing checkpoints
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# === Configuration ===
INPUT_DIR = Path(r"D:\3rscan_test")
RIO10_TEST_SCENES = Path(__file__).parent.parent / "data" / "3DSSG" / "rio10_test_scenes.txt"

DEFAULT_MAX_ITERATIONS = 30000
DEFAULT_METHOD = "splatfacto"


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


def find_config_path(ns_output_dir: Path) -> Optional[Path]:
    """Find the NerfStudio config.yml from training output."""
    # NerfStudio saves config at various nesting levels depending on version:
    # output_dir/splatfacto/<timestamp>/config.yml
    # output_dir/ns_format/splatfacto/<timestamp>/config.yml
    # output_dir/<any>/<any>/config.yml
    config_candidates = list(ns_output_dir.rglob("config.yml"))
    if config_candidates:
        # Return the most recent one
        return sorted(config_candidates, key=lambda p: p.stat().st_mtime)[-1]
    return None


def is_scene_trained(scene_dir: Path) -> bool:
    """Check if a scene has already been trained (has valid exported PLY)."""
    splat_dir = scene_dir / "splat"
    ply_path = splat_dir / "point_cloud.ply"
    if not ply_path.exists() or ply_path.stat().st_size < 1000:
        return False
    # Validate PLY magic bytes
    try:
        with open(str(ply_path), "rb") as f:
            header = f.read(4)
        return header == b"ply\n" or header == b"ply\r"
    except OSError:
        return False


def is_scene_training_complete(scene_dir: Path) -> bool:
    """Check if NerfStudio training completed (config exists)."""
    ns_output_dir = scene_dir / "ns_output"
    config_path = find_config_path(ns_output_dir)
    return config_path is not None


def train_scene(
    scene_dir: Path,
    scene_id: str,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    method: str = DEFAULT_METHOD,
) -> dict:
    """
    Train a Gaussian splat for one scene using NerfStudio.

    Returns dict with success, time_seconds, error.
    """
    ns_data_dir = scene_dir / "ns_format"
    ns_output_dir = scene_dir / "ns_output"
    transforms_path = ns_data_dir / "transforms.json"

    if not transforms_path.exists():
        return {"success": False, "error": "No transforms.json found"}

    # Build ns-train command
    cmd = [
        "ns-train", method,
        "--data", str(ns_data_dir),
        "--output-dir", str(ns_output_dir),
        "--max-num-iterations", str(max_iterations),
        "--viewer.quit-on-train-completion", "True",
        "--pipeline.model.num-downscales", "0",
    ]

    print(f"    Running: {' '.join(cmd[:6])}...")
    start_time = time.time()

    try:
        # Set UTF-8 encoding to prevent Windows console encoding errors with rich/nerfstudio
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONLEGACYWINDOWSSTDIO"] = "1"
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout — training takes ~15 min, kill if hung after
            env=env,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.time() - start_time

        if result.returncode != 0:
            # Save error log
            error_log = scene_dir / "train_error.log"
            with open(str(error_log), "w", encoding="utf-8", errors="replace") as f:
                f.write(f"COMMAND: {' '.join(cmd)}\n\n")
                f.write(f"STDOUT:\n{result.stdout[-2000:]}\n\n")
                f.write(f"STDERR:\n{result.stderr[-2000:]}\n")
            return {
                "success": False,
                "error": f"ns-train failed (exit code {result.returncode})",
                "time_seconds": elapsed,
            }

        return {"success": True, "time_seconds": elapsed}

    except subprocess.TimeoutExpired:
        # Timeout likely means training finished but viewer hung
        # Check if checkpoint was saved
        config_path = find_config_path(ns_output_dir)
        if config_path is not None:
            return {"success": True, "time_seconds": 1800, "note": "killed after timeout (viewer hang)"}
        return {"success": False, "error": "Training timed out (>30 min) with no checkpoint"}
    except FileNotFoundError:
        return {"success": False, "error": "ns-train not found. Install NerfStudio: pip install nerfstudio"}


def export_splat(scene_dir: Path, scene_id: str) -> dict:
    """
    Export Gaussian splat PLY from trained NerfStudio model.

    Returns dict with success, error.
    """
    ns_output_dir = scene_dir / "ns_output"
    splat_dir = scene_dir / "splat"
    splat_dir.mkdir(parents=True, exist_ok=True)

    # Find config path
    config_path = find_config_path(ns_output_dir)
    if config_path is None:
        return {"success": False, "error": "No config.yml found (training incomplete?)"}

    # Build ns-export command
    cmd = [
        "ns-export", "gaussian-splat",
        "--load-config", str(config_path),
        "--output-dir", str(splat_dir),
    ]

    print(f"    Exporting: {scene_id}...")

    try:
        # Set UTF-8 encoding to prevent Windows cp1252 UnicodeEncodeError
        # (NerfStudio outputs Unicode characters like ✅ that cp1252 can't handle)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout for export
            env=env,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            # Use errors="replace" to safely handle any remaining Unicode issues in stderr
            error_msg = result.stderr[-500:] if result.stderr else "unknown error"
            return {
                "success": False,
                "error": f"ns-export failed: {error_msg}",
            }

        # Verify output
        ply_path = splat_dir / "point_cloud.ply"
        if not ply_path.exists():
            # NerfStudio might use a different filename
            ply_files = list(splat_dir.glob("*.ply"))
            if ply_files:
                # Rename to standard name
                ply_files[0].rename(ply_path)
            else:
                return {"success": False, "error": "No PLY file in export output"}

        return {"success": True, "size_bytes": ply_path.stat().st_size}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Export timed out"}
    except FileNotFoundError:
        return {"success": False, "error": "ns-export not found"}


def main():
    parser = argparse.ArgumentParser(
        description="Train Gaussian splats via NerfStudio for RIO10 test scenes"
    )
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Limit number of scenes")
    parser.add_argument("--scene-id", type=str, default=None,
                        help="Train a single scene")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
                        help=f"Training iterations (default: {DEFAULT_MAX_ITERATIONS})")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD,
                        help=f"NerfStudio method (default: {DEFAULT_METHOD})")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Override input directory")
    parser.add_argument("--export-only", action="store_true",
                        help="Only export PLY from existing checkpoints")
    parser.add_argument("--force", action="store_true",
                        help="Re-train even if splat already exists")
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else INPUT_DIR

    print("=" * 70)
    print("NerfStudio Gaussian Splat Training (RIO10 Test Scenes)")
    print("=" * 70)
    print(f"  Input dir:       {input_dir}")
    print(f"  Method:          {args.method}")
    print(f"  Max iterations:  {args.max_iterations}")
    print(f"  Export only:     {args.export_only}")
    print()

    # Get scene list
    if args.scene_id:
        scene_ids = [args.scene_id]
    else:
        scene_ids = load_test_scene_ids()
        if args.max_scenes:
            scene_ids = scene_ids[:args.max_scenes]

    # Filter to scenes with ns_format data
    valid_scenes = []
    for scene_id in scene_ids:
        scene_dir = input_dir / scene_id
        transforms_path = scene_dir / "ns_format" / "transforms.json"
        if transforms_path.exists():
            valid_scenes.append(scene_id)

    print(f"  Scenes with NerfStudio data: {len(valid_scenes)} / {len(scene_ids)}")

    if not valid_scenes:
        print("\n  No scenes ready for training!")
        print("  Run convert_3rscan_to_nerfstudio.py first.")
        return

    # Estimate time
    if not args.export_only:
        est_hours = len(valid_scenes) * 20 / 60  # ~20 min per scene
        print(f"  Estimated training time: {est_hours:.1f} hours")
    print()

    # Process scenes
    n_trained = 0
    n_exported = 0
    n_skipped = 0
    n_failed = 0
    total_train_time = 0
    start_time = time.time()

    for i, scene_id in enumerate(valid_scenes):
        scene_dir = input_dir / scene_id

        print(f"\n[{i+1}/{len(valid_scenes)}] Scene: {scene_id}")

        # Check if already done
        if is_scene_trained(scene_dir) and not args.force:
            print(f"    Already trained and exported, skipping")
            n_skipped += 1
            continue

        # Train (unless export-only)
        if not args.export_only:
            if is_scene_training_complete(scene_dir) and not args.force:
                print(f"    Training already complete, skipping to export")
            else:
                # Clean up incomplete training output from previous interrupted run
                ns_output_dir = scene_dir / "ns_output"
                if ns_output_dir.exists() and not find_config_path(ns_output_dir):
                    import shutil
                    print(f"    Cleaning up incomplete training output from previous run...")
                    shutil.rmtree(str(ns_output_dir), ignore_errors=True)

                result = train_scene(
                    scene_dir, scene_id,
                    max_iterations=args.max_iterations,
                    method=args.method,
                )
                if not result["success"]:
                    print(f"    TRAIN FAILED: {result['error']}")
                    n_failed += 1
                    continue
                else:
                    train_time = result.get("time_seconds", 0)
                    total_train_time += train_time
                    n_trained += 1
                    print(f"    Training complete ({train_time/60:.1f} min)")

        # Export PLY from trained checkpoint
        export_result = export_splat(scene_dir, scene_id)
        if export_result["success"]:
            size_mb = export_result.get("size_bytes", 0) / 1024 / 1024
            print(f"    Export OK ({size_mb:.1f} MB)")
            n_exported += 1
        else:
            print(f"    EXPORT FAILED: {export_result['error']}")
            n_failed += 1

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total scenes:     {len(valid_scenes)}")
    print(f"  Trained (new):    {n_trained}")
    print(f"  Exported:         {n_exported}")
    print(f"  Skipped:          {n_skipped}")
    print(f"  Failed:           {n_failed}")
    print(f"  Total time:       {elapsed/3600:.1f} hours")
    if n_trained > 0:
        print(f"  Avg train time:   {total_train_time/n_trained/60:.1f} min/scene")
    print()
    print("Next step: Transfer instance labels")
    print("  python scripts/transfer_3rscan_test_labels.py")

    # Save progress report
    report = {
        "total_scenes": len(valid_scenes),
        "trained": n_trained,
        "exported": n_exported,
        "skipped": n_skipped,
        "failed": n_failed,
        "total_time_hours": round(elapsed / 3600, 2),
        "scenes_status": {},
    }
    for scene_id in valid_scenes:
        scene_dir = input_dir / scene_id
        report["scenes_status"][scene_id] = {
            "trained": is_scene_training_complete(scene_dir),
            "exported": is_scene_trained(scene_dir),
        }
    report_path = input_dir / "training_report.json"
    with open(str(report_path), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
