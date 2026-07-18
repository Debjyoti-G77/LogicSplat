"""
Extract 3RScan sequences and convert to NerfStudio format.

For each downloaded test scene:
  1. Extract sequence.zip → get frame-XXXXXX.color.jpg and frame-XXXXXX.pose.txt
  2. Convert 3RScan pose format (4x4 camera-to-world matrices) to NerfStudio transforms.json
  3. Camera intrinsics: fx=fy=577.87, cx=319.5, cy=239.5 for 640x480

Output structure:
    D:\\3rscan_test\\{scene_id}\\ns_format\\
        images/
            frame-000000.color.jpg
            frame-000001.color.jpg
            ...
        transforms.json

Usage:
    python scripts/convert_3rscan_to_nerfstudio.py
    python scripts/convert_3rscan_to_nerfstudio.py --max-scenes 5
    python scripts/convert_3rscan_to_nerfstudio.py --scene-id <uuid>
    python scripts/convert_3rscan_to_nerfstudio.py --skip-every 5  # Use every 5th frame
"""

import argparse
import json
import os
import sys
import time
import zipfile
import shutil
import numpy as np
from pathlib import Path
from typing import List, Optional


# === Configuration ===
INPUT_DIR = Path(r"D:\3rscan_test")
RIO10_TEST_SCENES = Path(__file__).parent.parent / "data" / "3DSSG" / "rio10_test_scenes.txt"

# Standard 3RScan camera intrinsics (640x480)
INTRINSICS_640x480 = {
    "fx": 577.870849,
    "fy": 577.870849,
    "cx": 319.5,
    "cy": 239.5,
    "w": 640,
    "h": 480,
}

# For 960x540 (some 3RScan scenes use this)
INTRINSICS_960x540 = {
    "fx": 866.806274,
    "fy": 866.806274,
    "cx": 479.5,
    "cy": 269.5,
    "w": 960,
    "h": 540,
}

# Frame subsampling — use every Nth frame to reduce training time
DEFAULT_SKIP_EVERY = 3  # Use every 3rd frame (reduces ~1500 frames to ~500)


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


def extract_sequence(scene_dir: Path) -> bool:
    """
    Extract sequence.zip into the scene directory.
    Returns True if extraction succeeded.
    Validates zip integrity before extracting.
    """
    zip_path = scene_dir / "sequence.zip"
    sequence_dir = scene_dir / "sequence"

    if not zip_path.exists():
        return False

    # Skip if already extracted (check for at least one frame)
    if sequence_dir.exists():
        color_files = list(sequence_dir.glob("frame-*.color.jpg"))
        if len(color_files) > 10:
            return True

    # Validate zip integrity before extracting
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            bad_file = zf.testzip()
            if bad_file is not None:
                print(f"    WARNING: Corrupt zip entry: {bad_file}")
                return False
            zf.extractall(str(scene_dir))
        return True
    except (zipfile.BadZipFile, OSError) as e:
        # Clean up partial extraction
        if sequence_dir.exists():
            import shutil
            shutil.rmtree(str(sequence_dir), ignore_errors=True)
        print(f"    ERROR extracting {zip_path}: {e}")
        return False


def load_pose(pose_path: str) -> Optional[np.ndarray]:
    """
    Load a 4x4 camera-to-world matrix from a 3RScan pose file.
    Returns None if the pose is invalid (all zeros or contains inf/nan).
    """
    try:
        pose = np.loadtxt(pose_path).reshape(4, 4)
        # Check for invalid poses
        if np.all(pose == 0) or np.any(np.isinf(pose)) or np.any(np.isnan(pose)):
            return None
        # Check determinant (should be ~1 for valid rotation)
        det = np.linalg.det(pose[:3, :3])
        if abs(det) < 0.5 or abs(det) > 2.0:
            return None
        return pose
    except (ValueError, OSError):
        return None


def detect_intrinsics(image_path: str) -> dict:
    """
    Detect camera intrinsics based on image resolution.
    Falls back to 640x480 intrinsics if PIL is not available.
    """
    try:
        from PIL import Image
        img = Image.open(image_path)
        w, h = img.size
        img.close()

        if w == 960 and h == 540:
            return INTRINSICS_960x540
        elif w == 640 and h == 480:
            return INTRINSICS_640x480
        else:
            # Scale intrinsics proportionally from 640x480
            scale_x = w / 640.0
            scale_y = h / 480.0
            return {
                "fx": 577.870849 * scale_x,
                "fy": 577.870849 * scale_y,
                "cx": (w - 1) / 2.0,
                "cy": (h - 1) / 2.0,
                "w": w,
                "h": h,
            }
    except ImportError:
        return INTRINSICS_640x480


def convert_scene_to_nerfstudio(
    scene_dir: Path,
    skip_every: int = DEFAULT_SKIP_EVERY,
) -> Optional[dict]:
    """
    Convert a 3RScan scene to NerfStudio format.

    Args:
        scene_dir: Path to scene directory (contains sequence/ subfolder)
        skip_every: Use every Nth frame

    Returns:
        dict with stats or None if conversion fails.
    """
    # Look for frames in scene_dir directly (3RScan extracts flat)
    # or in a "sequence" subdirectory (some versions)
    sequence_dir = scene_dir / "sequence"
    if not sequence_dir.exists():
        # Frames extracted directly into scene_dir
        sequence_dir = scene_dir
    ns_dir = scene_dir / "ns_format"
    images_dir = ns_dir / "images"

    # Check if already converted — validate JSON integrity
    transforms_path = ns_dir / "transforms.json"
    if transforms_path.exists():
        try:
            with open(transforms_path, "r") as f:
                existing = json.load(f)
            if len(existing.get("frames", [])) > 10:
                return {"n_frames": len(existing["frames"]), "skipped": True}
        except (json.JSONDecodeError, OSError):
            # Corrupt transforms.json from interrupted write — delete and redo
            transforms_path.unlink(missing_ok=True)

    # Check that we have frames
    if not list(sequence_dir.glob("frame-*.color.jpg")):
        return None

    # Clean up any leftover .tmp file from a previous interrupted run
    tmp_path = ns_dir / "transforms.json.tmp"
    if tmp_path.exists():
        tmp_path.unlink()

    # Find all color frames
    color_files = sorted(sequence_dir.glob("frame-*.color.jpg"))
    if not color_files:
        # Some scenes have frames in a subdirectory
        color_files = sorted(sequence_dir.glob("**/frame-*.color.jpg"))
    if not color_files:
        return None

    # Detect intrinsics from first frame
    intrinsics = detect_intrinsics(str(color_files[0]))

    # Subsample frames
    selected_frames = color_files[::skip_every]

    # Build NerfStudio transforms.json
    frames = []
    valid_count = 0

    for color_path in selected_frames:
        # Derive pose path from color path
        # frame-XXXXXX.color.jpg → frame-XXXXXX.pose.txt
        frame_name = color_path.stem.replace(".color", "")
        pose_path = color_path.parent / f"{frame_name}.pose.txt"

        if not pose_path.exists():
            continue

        # Load and validate pose
        c2w = load_pose(str(pose_path))
        if c2w is None:
            continue

        # NerfStudio expects camera-to-world as a 4x4 matrix (row-major in JSON)
        # 3RScan already provides camera-to-world, so we just need to convert
        # the coordinate system.
        #
        # 3RScan uses: X-right, Y-down, Z-forward (OpenCV convention)
        # NerfStudio expects: X-right, Y-up, Z-backward (OpenGL convention)
        #
        # Apply conversion: negate Y and Z rows of the rotation
        transform_matrix = c2w.copy()
        transform_matrix[1, :] *= -1  # Flip Y
        transform_matrix[2, :] *= -1  # Flip Z

        # Relative image path
        rel_image_path = f"images/{color_path.name}"

        frames.append({
            "file_path": rel_image_path,
            "transform_matrix": transform_matrix.tolist(),
        })
        valid_count += 1

    if valid_count < 10:
        return None

    # Create output directory structure
    images_dir.mkdir(parents=True, exist_ok=True)

    # Copy/symlink selected images to ns_format/images/
    for color_path in selected_frames:
        frame_name = color_path.stem.replace(".color", "")
        pose_path = color_path.parent / f"{frame_name}.pose.txt"
        if not pose_path.exists():
            continue
        c2w = load_pose(str(pose_path))
        if c2w is None:
            continue

        dst = images_dir / color_path.name
        if not dst.exists():
            # Use hard link to save disk space, fall back to copy
            try:
                os.link(str(color_path), str(dst))
            except OSError:
                shutil.copy2(str(color_path), str(dst))

    # Build transforms.json — write atomically to prevent corruption on Ctrl+C
    transforms = {
        "fl_x": intrinsics["fx"],
        "fl_y": intrinsics["fy"],
        "cx": intrinsics["cx"],
        "cy": intrinsics["cy"],
        "w": intrinsics["w"],
        "h": intrinsics["h"],
        "camera_model": "OPENCV",
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "frames": frames,
    }

    tmp_path = transforms_path.with_suffix(".json.tmp")
    with open(str(tmp_path), "w") as f:
        json.dump(transforms, f, indent=2)
    # Atomic rename — if this is interrupted, only the .tmp file is left (ignored on resume)
    if transforms_path.exists():
        transforms_path.unlink()
    tmp_path.rename(transforms_path)

    return {
        "n_frames": valid_count,
        "n_total_frames": len(color_files),
        "skip_every": skip_every,
        "intrinsics": intrinsics,
        "skipped": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract 3RScan sequences and convert to NerfStudio format"
    )
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Limit number of scenes")
    parser.add_argument("--scene-id", type=str, default=None,
                        help="Process a single scene")
    parser.add_argument("--skip-every", type=int, default=DEFAULT_SKIP_EVERY,
                        help=f"Use every Nth frame (default: {DEFAULT_SKIP_EVERY})")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Override input directory")
    parser.add_argument("--force", action="store_true",
                        help="Re-convert even if transforms.json exists")
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else INPUT_DIR

    print("=" * 70)
    print("3RScan → NerfStudio Format Converter")
    print("=" * 70)
    print(f"  Input dir:   {input_dir}")
    print(f"  Skip every:  {args.skip_every} frames")
    print()

    # Get scene list
    if args.scene_id:
        scene_ids = [args.scene_id]
    else:
        scene_ids = load_test_scene_ids()
        if args.max_scenes:
            scene_ids = scene_ids[:args.max_scenes]

    print(f"[1/3] Processing {len(scene_ids)} scenes...")
    print()

    # Process each scene
    n_extracted = 0
    n_converted = 0
    n_skipped = 0
    n_failed = 0
    start_time = time.time()

    for i, scene_id in enumerate(scene_ids):
        scene_dir = input_dir / scene_id

        if not scene_dir.exists():
            print(f"  [{i+1}/{len(scene_ids)}] {scene_id}: directory not found, skipping")
            n_failed += 1
            continue

        # Step 1: Extract sequence.zip
        print(f"  [{i+1}/{len(scene_ids)}] {scene_id}:", end=" ")

        if not extract_sequence(scene_dir):
            print("extraction failed")
            n_failed += 1
            continue
        n_extracted += 1

        # Step 2: Convert to NerfStudio format
        result = convert_scene_to_nerfstudio(scene_dir, skip_every=args.skip_every)

        if result is None:
            print("conversion failed (too few valid frames)")
            n_failed += 1
            continue

        if result.get("skipped") and not args.force:
            print(f"already converted ({result['n_frames']} frames)")
            n_skipped += 1
        else:
            n_converted += 1
            print(
                f"OK — {result['n_frames']} frames "
                f"(from {result.get('n_total_frames', '?')} total, "
                f"skip={result.get('skip_every', '?')})"
            )

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total scenes:    {len(scene_ids)}")
    print(f"  Extracted:       {n_extracted}")
    print(f"  Converted (new): {n_converted}")
    print(f"  Already done:    {n_skipped}")
    print(f"  Failed:          {n_failed}")
    print(f"  Time:            {elapsed:.1f}s")
    print()
    print("Next step: Train Gaussian splats with NerfStudio")
    print("  python scripts/train_3rscan_test_splats.py")


if __name__ == "__main__":
    main()
