"""
Process new scenes (06–13) through the full Gaussian Splatting pipeline.

Pipeline per scene:
    1. Extract frames from .mov video (ffmpeg)
    2. ns-process-data video  →  ns_data/ (COLMAP + transforms.json)
    3. ns-train splatfacto    →  outputs/<scene_id>/
    4. Copy splat.ply         →  data/processed/<scene_id>/splat.ply

Usage:
    python scripts/process_new_scenes.py
    python scripts/process_new_scenes.py --scenes scene_06 scene_07
    python scripts/process_new_scenes.py --scenes scene_06 --iterations 15000
    python scripts/process_new_scenes.py --force   # re-process even if splat.ply exists

Requirements:
    - ffmpeg on PATH
    - nerfstudio installed (ns-process-data, ns-train on PATH)
    - RTX 5060 / CUDA available
"""

import subprocess
import os
import sys
import shutil
import glob
import argparse
import logging
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR      = "D:/logicsplat_data/raw"
DATA_DIR     = "D:/logicsplat_data/processed"
OUTPUTS_DIR  = "outputs"
LOG_DIR      = "logs"

NEW_SCENES   = [f"scene_{i:02d}" for i in range(6, 14)]   # scene_06 … scene_13
MAX_ITERS    = 30000


# ── Logging setup ─────────────────────────────────────────────────────────────
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
    # Force UTF-8 on Windows to handle unicode characters in log messages
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
    """Run a command, stream output, return True on success."""
    logger.info(f"[START] {desc}")
    logger.debug(f"  cmd: {' '.join(str(c) for c in cmd)}")
    try:
        result = subprocess.run(
            cmd,
            stdout=None,   # inherit — lets nerfstudio print its progress bars
            stderr=None,
        )
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


# ── Step helpers ──────────────────────────────────────────────────────────────
def step_extract_frames(scene_id: str, video_path: str,
                        images_dir: str, logger: logging.Logger) -> bool:
    """
    Extract frames from video using ffmpeg.
    Targets ~2 fps to keep frame count manageable for COLMAP.
    Skipped if images already exist.
    """
    if os.path.isdir(images_dir) and len(os.listdir(images_dir)) > 10:
        logger.info(f"  Frames already extracted ({len(os.listdir(images_dir))} files) — skipping")
        return True

    os.makedirs(images_dir, exist_ok=True)
    # 2 fps gives ~120 frames for a 60-second video — good for COLMAP
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", "fps=2",
        "-q:v", "2",
        os.path.join(images_dir, "frame_%05d.png"),
    ]
    return run_cmd(cmd, "ffmpeg frame extraction", logger)


def step_ns_process_data(scene_id: str, video_path: str,
                         ns_data_dir: str, logger: logging.Logger,
                         matching_method: str = "sequential") -> bool:
    """
    Run ns-process-data video to run COLMAP and generate transforms.json.
    This is the recommended nerfstudio path for video input.

    COLMAP 4.x renamed --SiftExtraction.use_gpu → --FeatureExtraction.use_gpu.
    We patch nerfstudio's colmap_utils.py at startup (see patch below), and
    point --colmap-cmd at the local bin/colmap so PATH resolution is reliable.

    matching_method: 'sequential' (fast, default) or 'exhaustive' (slower, more robust)
    """
    transforms_path = os.path.join(ns_data_dir, "transforms.json")
    if os.path.exists(transforms_path):
        # Check if the existing transforms.json has enough frames
        try:
            import json as _json
            with open(transforms_path) as f:
                t = _json.load(f)
            n_frames = len(t.get("frames", []))
            if n_frames >= 10:
                logger.info(f"  transforms.json already exists ({n_frames} frames) — skipping ns-process-data")
                return True
            else:
                logger.warning(f"  transforms.json exists but only has {n_frames} frames — checking for better COLMAP model")
                # COLMAP sometimes produces model 0 with few cameras and model 1 with the full
                # reconstruction. Check for a better model before rerunning from scratch.
                colmap_sparse = os.path.join(ns_data_dir, "colmap", "sparse")
                best_model = None
                best_count = n_frames
                for model_id in ["1", "2", "3"]:
                    images_bin = os.path.join(colmap_sparse, model_id, "images.bin")
                    if os.path.exists(images_bin):
                        try:
                            import struct as _struct
                            with open(images_bin, "rb") as bf:
                                count = _struct.unpack("<Q", bf.read(8))[0]
                            logger.info(f"  COLMAP model {model_id}: {count} cameras")
                            if count > best_count:
                                best_count = count
                                best_model = model_id
                        except Exception:
                            pass
                if best_model:
                    logger.info(f"  Using COLMAP model {best_model} ({best_count} cameras) — regenerating transforms.json")
                    colmap_cmd_path = os.path.join(os.path.abspath("bin"), "colmap.exe")
                    cmd = [
                        "ns-process-data", "video",
                        "--data", video_path,
                        "--output-dir", ns_data_dir,
                        "--colmap-cmd", colmap_cmd_path,
                        "--skip-colmap",
                        "--colmap-model-path", os.path.join("colmap", "sparse", best_model),
                        "--no-gpu",
                    ]
                    return run_cmd(cmd, f"ns-process-data (reuse COLMAP model {best_model})", logger)
                else:
                    logger.warning(f"  No better COLMAP model found — rerunning with exhaustive matching")
                    import shutil as _shutil
                    _shutil.rmtree(ns_data_dir, ignore_errors=True)
                    os.makedirs(ns_data_dir, exist_ok=True)
                    matching_method = "exhaustive"
        except Exception:
            pass

    os.makedirs(ns_data_dir, exist_ok=True)

    # Use the local COLMAP binary (bin/colmap.exe on Windows) to avoid PATH issues.
    colmap_cmd = os.path.join(os.path.abspath("bin"), "colmap.exe")

    logger.info(f"  Using matching method: {matching_method}")
    cmd = [
        "ns-process-data", "video",
        "--data", video_path,
        "--output-dir", ns_data_dir,
        "--colmap-cmd", colmap_cmd,
        "--matching-method", matching_method,
        "--no-gpu",   # passes --FeatureExtraction.use_gpu 0 (patched for COLMAP 4.x)
    ]
    return run_cmd(cmd, f"ns-process-data video ({matching_method} matching)", logger)


def step_ns_train(scene_id: str, ns_data_dir: str,
                  output_dir: str, iterations: int,
                  logger: logging.Logger) -> bool:
    """
    Train Gaussian Splatting with splatfacto.
    Skipped if a checkpoint already exists in the output dir.

    NOTE: nerfstudio appends the last component of --data to --output-dir,
    so with --data .../ns_data and --output-dir outputs/scene_06, the actual
    output lands at outputs/scene_06/ns_data/splatfacto/<timestamp>/.
    We search recursively to handle this.
    """
    # Check if training already completed (config.yml present anywhere under output_dir)
    existing_configs = glob.glob(
        os.path.join(output_dir, "**", "splatfacto", "*", "config.yml"), recursive=True
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
    return run_cmd(cmd, f"ns-train splatfacto ({iterations} iterations)", logger)


def find_splat_ply(output_dir: str, logger: logging.Logger) -> str | None:
    """
    Locate the exported splat.ply inside the nerfstudio output tree.
    nerfstudio saves it at:
      outputs/<scene_id>/splatfacto/<timestamp>/point_cloud/iteration_<N>/point_cloud.ply
    We also check for a direct splat.ply export.
    """
    # Pattern 1: point_cloud.ply from ns-export or direct training output
    patterns = [
        os.path.join(output_dir, "splatfacto", "*", "point_cloud",
                     f"iteration_{MAX_ITERS}", "point_cloud.ply"),
        os.path.join(output_dir, "splatfacto", "*", "point_cloud",
                     "iteration_*", "point_cloud.ply"),
        os.path.join(output_dir, "splatfacto", "*", "splat.ply"),
        os.path.join(output_dir, "**", "point_cloud.ply"),
        os.path.join(output_dir, "**", "splat.ply"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            # prefer the highest iteration number
            matches.sort(key=lambda p: (
                int(m.group(1)) if (m := __import__("re").search(r"iteration_(\d+)", p)) else 0
            ), reverse=True)
            logger.info(f"  Found splat at: {matches[0]}")
            return matches[0]
    return None


def step_export_splat(scene_id: str, output_dir: str,
                      dest_ply: str, logger: logging.Logger) -> bool:
    """
    Try ns-export first; if that fails, fall back to copying point_cloud.ply.

    nerfstudio saves config.yml at:
      outputs/<scene_id>/ns_data/splatfacto/<timestamp>/config.yml
    (the 'ns_data' subdirectory is appended by nerfstudio from the --data path)
    We search recursively to find it regardless of nesting.
    """
    # Find config.yml anywhere under output_dir (handles nerfstudio's nested output)
    config_files = glob.glob(
        os.path.join(output_dir, "**", "splatfacto", "*", "config.yml"), recursive=True
    )
    if not config_files:
        logger.error("  No config.yml found — cannot export")
        return False

    # Use the most recent config
    config_path = sorted(config_files)[-1]
    scene_dir   = os.path.dirname(dest_ply)
    logger.info(f"  Using config: {config_path}")

    # Try ns-export gaussian-splat
    export_ok = run_cmd(
        ["ns-export", "gaussian-splat",
         "--load-config", config_path,
         "--output-dir", scene_dir],
        "ns-export gaussian-splat",
        logger,
    )

    if export_ok:
        # ns-export writes splat.ply directly into scene_dir
        exported = os.path.join(scene_dir, "splat.ply")
        if os.path.exists(exported):
            logger.info(f"  splat.ply exported to {exported}")
            return True

    # Fallback: copy point_cloud.ply
    logger.warning("  ns-export did not produce splat.ply — trying point_cloud.ply fallback")
    src = find_splat_ply(output_dir, logger)
    if src and os.path.exists(src):
        shutil.copy2(src, dest_ply)
        logger.info(f"  Copied {src} -> {dest_ply}")
        return True

    logger.error("  Could not locate any .ply output")
    return False


# ── Main scene processor ──────────────────────────────────────────────────────
def process_scene(scene_id: str, iterations: int = MAX_ITERS,
                  force: bool = False) -> bool:
    """
    Full pipeline for one scene. Returns True on success.
    """
    logger = setup_logging(scene_id)

    print(f"\n{'='*60}")
    print(f"  {scene_id.upper()}")
    print(f"{'='*60}")

    # ── Paths ──────────────────────────────────────────────────────────────────
    video_path  = os.path.join(RAW_DIR, f"{scene_id}.mov")
    scene_dir   = os.path.join(DATA_DIR, scene_id)
    images_dir  = os.path.join(scene_dir, "images")
    ns_data_dir = os.path.join(scene_dir, "ns_data")
    output_dir  = os.path.join(OUTPUTS_DIR, scene_id)
    dest_ply    = os.path.join(scene_dir, "splat.ply")

    # ── Already done? ──────────────────────────────────────────────────────────
    if os.path.exists(dest_ply) and not force:
        logger.info(f"SKIP — splat.ply already exists at {dest_ply}")
        print(f"  ✓ Already processed — skipping (use --force to reprocess)")
        return True

    # ── Check raw video exists ─────────────────────────────────────────────────
    if not os.path.exists(video_path):
        logger.error(f"Raw video not found: {video_path}")
        print(f"  ✗ Raw video not found: {video_path}")
        return False

    os.makedirs(scene_dir, exist_ok=True)

    # ── Step 1: Extract frames ─────────────────────────────────────────────────
    logger.info("-- Step 1: Extract frames --")
    if not step_extract_frames(scene_id, video_path, images_dir, logger):
        err = "Frame extraction failed"
        logger.error(err)
        with open(os.path.join(scene_dir, "processing_error.txt"), "w") as f:
            f.write(f"{err}\n")
        return False

    # ── Step 2: ns-process-data ────────────────────────────────────────────────
    logger.info("-- Step 2: ns-process-data video --")
    if not step_ns_process_data(scene_id, video_path, ns_data_dir, logger):
        err = "ns-process-data failed"
        logger.error(err)
        with open(os.path.join(scene_dir, "processing_error.txt"), "w") as f:
            f.write(f"{err}\n")
        return False

    # Verify transforms.json was created
    transforms_path = os.path.join(ns_data_dir, "transforms.json")
    if not os.path.exists(transforms_path):
        err = f"transforms.json not found after ns-process-data: {transforms_path}"
        logger.error(err)
        with open(os.path.join(scene_dir, "processing_error.txt"), "w") as f:
            f.write(f"{err}\n")
        return False

    # ── Step 3: ns-train splatfacto ───────────────────────────────────────────
    logger.info("-- Step 3: ns-train splatfacto --")
    if not step_ns_train(scene_id, ns_data_dir, output_dir, iterations, logger):
        err = "ns-train failed"
        logger.error(err)
        with open(os.path.join(scene_dir, "processing_error.txt"), "w") as f:
            f.write(f"{err}\n")
        return False

    # ── Step 4: Export splat.ply ──────────────────────────────────────────────
    logger.info("-- Step 4: Export splat.ply --")
    if not step_export_splat(scene_id, output_dir, dest_ply, logger):
        err = "splat.ply export failed"
        logger.error(err)
        with open(os.path.join(scene_dir, "processing_error.txt"), "w") as f:
            f.write(f"{err}\n")
        return False

    if not os.path.exists(dest_ply):
        err = f"splat.ply not found at expected path: {dest_ply}"
        logger.error(err)
        with open(os.path.join(scene_dir, "processing_error.txt"), "w") as f:
            f.write(f"{err}\n")
        return False

    size_mb = os.path.getsize(dest_ply) / (1024 * 1024)
    logger.info(f"[DONE] {scene_id} complete - splat.ply ({size_mb:.1f} MB) at {dest_ply}")
    print(f"\n  [OK] {scene_id} done - splat.ply ({size_mb:.1f} MB)")
    return True


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Process new scenes (06–13) through the Gaussian Splatting pipeline"
    )
    parser.add_argument(
        "--scenes", nargs="+", default=NEW_SCENES,
        help="Scene IDs to process (default: scene_06 through scene_13)"
    )
    parser.add_argument(
        "--iterations", type=int, default=MAX_ITERS,
        help=f"Gaussian Splatting training iterations (default: {MAX_ITERS})"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process scenes even if splat.ply already exists"
    )
    args = parser.parse_args()

    print(f"\nLogicSplat — New Scene Processing")
    print(f"Scenes   : {', '.join(args.scenes)}")
    print(f"Iters    : {args.iterations}")
    print(f"Force    : {args.force}")
    print(f"Raw dir  : {os.path.abspath(RAW_DIR)}")
    print(f"Out dir  : {os.path.abspath(DATA_DIR)}")

    # Check raw videos exist before starting
    missing = [s for s in args.scenes
               if not os.path.exists(os.path.join(RAW_DIR, f"{s}.mov"))]
    if missing:
        print(f"\nWARNING: Raw videos not found for: {', '.join(missing)}")
        print(f"  Expected at: {RAW_DIR}/<scene_id>.mov")

    results: dict[str, str] = {}
    failed: list[str] = []

    for idx, scene_id in enumerate(args.scenes, 1):
        print(f"\nProcessing {scene_id} ({idx}/{len(args.scenes)})...")
        scene_dir = os.path.join(DATA_DIR, scene_id)
        os.makedirs(scene_dir, exist_ok=True)
        error_log = os.path.join(scene_dir, "processing_error.txt")

        try:
            success = process_scene(scene_id, args.iterations, args.force)
            results[scene_id] = "✓" if success else "✗"
            if not success:
                failed.append(scene_id)
                # Write error marker so we know it failed
                with open(error_log, "w") as f:
                    f.write(f"Processing failed for {scene_id}.\n"
                            f"Check logs/ for details.\n")
            else:
                # Clear any previous error log on success
                if os.path.exists(error_log):
                    os.remove(error_log)
        except KeyboardInterrupt:
            print(f"\n\nInterrupted during {scene_id}. Stopping.")
            results[scene_id] = "interrupted"
            break
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            print(f"\n  ✗ Unexpected error in {scene_id}: {e}")
            results[scene_id] = f"✗ error: {e}"
            failed.append(scene_id)
            with open(error_log, "w") as f:
                f.write(f"Unexpected error processing {scene_id}:\n{err_msg}\n")
            continue

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*40}")
    print("Summary")
    print(f"{'='*40}")
    for scene, status in results.items():
        print(f"  {status}  {scene}")

    if failed:
        print(f"\nFailed scenes: {', '.join(failed)}")
        print(f"Check logs/ for details.")
        sys.exit(1)
    else:
        processed = [s for s, r in results.items() if r == "✓"]
        print(f"\nAll {len(processed)} scene(s) processed successfully.")


if __name__ == "__main__":
    main()
