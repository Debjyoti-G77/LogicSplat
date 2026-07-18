"""
Process OCTScenes through the nerfstudio Gaussian Splatting pipeline.

For each of the 20 downloaded OCTScenes:
    1. ns-process-data images  →  ns_data/ (COLMAP + transforms.json)
    2. ns-train splatfacto     →  outputs/octscenes/oct_XX/
    3. Export splat.ply        →  data/octscenes/oct_XX/splat.ply

Uses 15k iterations (training data, not evaluation).
Processes one scene at a time. Logs errors, continues on failure.

Usage:
    python scripts/process_octscenes.py
    python scripts/process_octscenes.py --scenes oct_01 oct_02
    python scripts/process_octscenes.py --iterations 10000
    python scripts/process_octscenes.py --force

Requirements:
    - nerfstudio installed (ns-process-data, ns-train, ns-export on PATH)
    - COLMAP available (bin/colmap.exe or on PATH)
    - CUDA GPU available
"""
import sys
sys.path.insert(0, ".")

import os
import glob
import shutil
import subprocess
import argparse
import logging
from datetime import datetime


# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = "D:/logicsplat_data/octscenes"
OUTPUTS_DIR = "outputs/octscenes"
LOG_DIR     = "logs/octscenes"
MAX_ITERS   = 15000  # training data — fewer iterations than eval scenes


def get_scene_ids(data_dir: str = DATA_DIR) -> list:
    """Find all oct_XX scene directories."""
    if not os.path.isdir(data_dir):
        return []
    scenes = sorted([
        d for d in os.listdir(data_dir)
        if d.startswith("oct_") and os.path.isdir(os.path.join(data_dir, d))
    ])
    return scenes


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(scene_id: str) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"{scene_id}_{ts}.log")

    logger = logging.getLogger(scene_id)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    if hasattr(ch.stream, 'reconfigure'):
        try:
            ch.stream.reconfigure(encoding='utf-8')
        except Exception:
            pass

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file: {log_file}")
    return logger


# ── Subprocess helper ─────────────────────────────────────────────────────────
def run_cmd(cmd: list, desc: str, logger: logging.Logger) -> bool:
    """Run a command, return True on success."""
    logger.info(f"[START] {desc}")
    logger.debug(f"  cmd: {' '.join(str(c) for c in cmd)}")
    try:
        result = subprocess.run(cmd, stdout=None, stderr=None)
        if result.returncode != 0:
            logger.error(f"[FAIL]  {desc}  (exit code {result.returncode})")
            return False
        logger.info(f"[OK]    {desc}")
        return True
    except FileNotFoundError as e:
        logger.error(f"[FAIL]  {desc}  — executable not found: {e}")
        return False
    except Exception as e:
        logger.error(f"[FAIL]  {desc}  — unexpected error: {e}")
        return False


# ── Pipeline steps ────────────────────────────────────────────────────────────
def step_ns_process_data(scene_id: str, images_dir: str,
                         ns_data_dir: str, logger: logging.Logger) -> bool:
    """
    Run ns-process-data images to run COLMAP and generate transforms.json.
    OCTScenes provides pre-rendered images (not video), so we use 'images' mode.
    """
    transforms_path = os.path.join(ns_data_dir, "transforms.json")
    if os.path.exists(transforms_path):
        import json
        with open(transforms_path) as f:
            t = json.load(f)
        n_frames = len(t.get("frames", []))
        if n_frames >= 10:
            logger.info(f"  transforms.json already exists ({n_frames} frames) — skipping")
            return True

    os.makedirs(ns_data_dir, exist_ok=True)

    # Use local COLMAP binary
    colmap_cmd = os.path.join(os.path.abspath("bin"), "colmap.exe")
    if not os.path.exists(colmap_cmd):
        # Fallback to PATH
        colmap_cmd = "colmap"

    cmd = [
        "ns-process-data", "images",
        "--data", images_dir,
        "--output-dir", ns_data_dir,
        "--colmap-cmd", colmap_cmd,
        "--matching-method", "sequential",
        "--no-gpu",
    ]
    success = run_cmd(cmd, "ns-process-data images (sequential)", logger)

    if not success:
        # Retry with exhaustive matching
        logger.warning("  Sequential matching failed — retrying with exhaustive")
        if os.path.exists(ns_data_dir):
            shutil.rmtree(ns_data_dir, ignore_errors=True)
        os.makedirs(ns_data_dir, exist_ok=True)
        cmd[-3] = "exhaustive"  # replace 'sequential' with 'exhaustive'
        success = run_cmd(cmd, "ns-process-data images (exhaustive)", logger)

    return success


def step_ns_train(scene_id: str, ns_data_dir: str,
                  output_dir: str, iterations: int,
                  logger: logging.Logger) -> bool:
    """Train splatfacto. Skip if already trained."""
    existing_configs = glob.glob(
        os.path.join(output_dir, "**", "splatfacto", "*", "config.yml"),
        recursive=True,
    )
    if existing_configs:
        logger.info(f"  Training output already exists — skipping ns-train")
        return True

    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ns-train", "splatfacto",
        "--data", ns_data_dir,
        "--output-dir", output_dir,
        "--max-num-iterations", str(iterations),
        "--viewer.quit-on-train-completion", "True",
    ]
    return run_cmd(cmd, f"ns-train splatfacto ({iterations} iters)", logger)


def step_export_splat(scene_id: str, output_dir: str,
                      dest_ply: str, logger: logging.Logger) -> bool:
    """Export splat.ply from trained model."""
    if os.path.exists(dest_ply):
        logger.info(f"  splat.ply already exists — skipping export")
        return True

    # Find config.yml
    config_files = glob.glob(
        os.path.join(output_dir, "**", "splatfacto", "*", "config.yml"),
        recursive=True,
    )
    if not config_files:
        logger.error("  No config.yml found — cannot export")
        return False

    config_path = sorted(config_files)[-1]
    scene_dir = os.path.dirname(dest_ply)
    logger.info(f"  Using config: {config_path}")

    # Try ns-export
    export_ok = run_cmd(
        ["ns-export", "gaussian-splat",
         "--load-config", config_path,
         "--output-dir", scene_dir],
        "ns-export gaussian-splat",
        logger,
    )

    if export_ok and os.path.exists(dest_ply):
        return True

    # Fallback: find point_cloud.ply
    logger.warning("  ns-export did not produce splat.ply — trying fallback")
    patterns = [
        os.path.join(output_dir, "**", "point_cloud.ply"),
        os.path.join(output_dir, "**", "splat.ply"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            import re
            matches.sort(
                key=lambda p: int(m.group(1)) if (m := re.search(r"iteration_(\d+)", p)) else 0,
                reverse=True,
            )
            shutil.copy2(matches[0], dest_ply)
            logger.info(f"  Copied {matches[0]} -> {dest_ply}")
            return True

    logger.error("  Could not locate any .ply output")
    return False


# ── Main scene processor ──────────────────────────────────────────────────────
def process_scene(scene_id: str, iterations: int = MAX_ITERS,
                  force: bool = False) -> bool:
    """Full pipeline for one OCTScene. Returns True on success."""
    logger = setup_logging(scene_id)

    print(f"\n{'='*60}")
    print(f"  {scene_id.upper()}")
    print(f"{'='*60}")

    # Paths
    scene_dir   = os.path.join(DATA_DIR, scene_id)
    images_dir  = os.path.join(scene_dir, "images")
    ns_data_dir = os.path.join(scene_dir, "ns_data")
    output_dir  = os.path.join(OUTPUTS_DIR, scene_id)
    dest_ply    = os.path.join(scene_dir, "splat.ply")

    # Already done?
    if os.path.exists(dest_ply) and not force:
        logger.info(f"SKIP — splat.ply already exists at {dest_ply}")
        print(f"  Already processed — skipping (use --force to reprocess)")
        return True

    # Check images exist
    if not os.path.isdir(images_dir):
        logger.error(f"Images directory not found: {images_dir}")
        print(f"  ERROR: No images at {images_dir}")
        print(f"  Run: python scripts/download_octscenes.py")
        return False

    n_images = len([f for f in os.listdir(images_dir) if f.endswith('.png')])
    if n_images < 5:
        logger.error(f"Too few images ({n_images}) in {images_dir}")
        print(f"  ERROR: Only {n_images} images found — need at least 5")
        return False

    logger.info(f"  Found {n_images} images in {images_dir}")

    # Step 1: ns-process-data images
    logger.info("-- Step 1: ns-process-data images --")
    if not step_ns_process_data(scene_id, images_dir, ns_data_dir, logger):
        err = "ns-process-data failed"
        logger.error(err)
        with open(os.path.join(scene_dir, "processing_error.txt"), "w") as f:
            f.write(f"{err}\n")
        return False

    # Verify transforms.json
    transforms_path = os.path.join(ns_data_dir, "transforms.json")
    if not os.path.exists(transforms_path):
        err = f"transforms.json not found after ns-process-data"
        logger.error(err)
        with open(os.path.join(scene_dir, "processing_error.txt"), "w") as f:
            f.write(f"{err}\n")
        return False

    # Step 2: ns-train splatfacto
    logger.info("-- Step 2: ns-train splatfacto --")
    if not step_ns_train(scene_id, ns_data_dir, output_dir, iterations, logger):
        err = "ns-train failed"
        logger.error(err)
        with open(os.path.join(scene_dir, "processing_error.txt"), "w") as f:
            f.write(f"{err}\n")
        return False

    # Step 3: Export splat.ply
    logger.info("-- Step 3: Export splat.ply --")
    if not step_export_splat(scene_id, output_dir, dest_ply, logger):
        err = "splat.ply export failed"
        logger.error(err)
        with open(os.path.join(scene_dir, "processing_error.txt"), "w") as f:
            f.write(f"{err}\n")
        return False

    if not os.path.exists(dest_ply):
        err = f"splat.ply not found at expected path: {dest_ply}"
        logger.error(err)
        return False

    size_mb = os.path.getsize(dest_ply) / (1024 * 1024)
    logger.info(f"[DONE] {scene_id} — splat.ply ({size_mb:.1f} MB)")
    print(f"\n  [OK] {scene_id} done — splat.ply ({size_mb:.1f} MB)")

    # Clear error log on success
    error_log = os.path.join(scene_dir, "processing_error.txt")
    if os.path.exists(error_log):
        os.remove(error_log)

    return True


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Process OCTScenes through nerfstudio Gaussian Splatting pipeline"
    )
    parser.add_argument(
        "--scenes", nargs="+", default=None,
        help="Scene IDs to process (default: all oct_XX in data/octscenes/)"
    )
    parser.add_argument(
        "--iterations", type=int, default=MAX_ITERS,
        help=f"Splatfacto training iterations (default: {MAX_ITERS})"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process scenes even if splat.ply already exists"
    )
    args = parser.parse_args()

    scenes = args.scenes or get_scene_ids()
    if not scenes:
        print("No OCTScenes found. Run: python scripts/download_octscenes.py")
        sys.exit(1)

    print(f"\nLogicSplat — OCTScenes Processing")
    print(f"Scenes     : {', '.join(scenes)}")
    print(f"Iterations : {args.iterations}")
    print(f"Force      : {args.force}")
    print(f"Data dir   : {os.path.abspath(DATA_DIR)}")
    print(f"Output dir : {os.path.abspath(OUTPUTS_DIR)}")

    results = {}
    failed = []

    for idx, scene_id in enumerate(scenes, 1):
        print(f"\nProcessing {scene_id} ({idx}/{len(scenes)})...")
        try:
            success = process_scene(scene_id, args.iterations, args.force)
            results[scene_id] = "OK" if success else "FAIL"
            if not success:
                failed.append(scene_id)
        except KeyboardInterrupt:
            print(f"\n\nInterrupted during {scene_id}. Stopping.")
            results[scene_id] = "interrupted"
            break
        except Exception as e:
            import traceback
            print(f"\n  ERROR in {scene_id}: {e}")
            traceback.print_exc()
            results[scene_id] = f"ERROR: {e}"
            failed.append(scene_id)
            continue

    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for scene, status in results.items():
        marker = "OK" if status == "OK" else "FAIL"
        print(f"  [{marker:4s}]  {scene}")

    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        print(f"Check logs at: {os.path.abspath(LOG_DIR)}")
        sys.exit(1)
    else:
        ok_count = sum(1 for s in results.values() if s == "OK")
        print(f"\nAll {ok_count} scene(s) processed successfully.")
        print(f"\nNext step: python scripts/prepare_annotation.py")


if __name__ == "__main__":
    main()
