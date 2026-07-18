"""
Download all 30 parts of OCTScenes 640x480, concatenate, extract 20 scenes.

The dataset is stored as a split gzip tar archive across 30 files.
All parts must be concatenated before extraction.

Pipeline:
  1. Download all 30 parts to D:/octscenes_download/ (~118 GB)
  2. Concatenate into one tar.gz (~118 GB)
  3. Extract all PNGs (~120 GB)
  4. Pick 20 complete scenes (60 frames each)
  5. Organize into data/octscenes/oct_01/ through oct_20/
  6. Cleanup D: drive to reclaim space

Peak disk usage on D: ~240 GB (parts + extracted).
Final output: ~600 MB in data/octscenes/

Usage:
    python scripts/download_octscenes_full.py
    python scripts/download_octscenes_full.py --step download
    python scripts/download_octscenes_full.py --step concat
    python scripts/download_octscenes_full.py --step extract
    python scripts/download_octscenes_full.py --step organize
    python scripts/download_octscenes_full.py --step cleanup
"""
import sys
sys.path.insert(0, ".")

import os
import glob
import json
import shutil
import random
import tarfile
import argparse
import time
from pathlib import Path
from collections import defaultdict

from huggingface_hub import hf_hub_download


# ── Config ────────────────────────────────────────────────────────────────────
REPO_ID = "Yinxuan/OCTScenes"
DOWNLOAD_DIR = "D:/octscenes_download"
PARTS_DIR = "D:/octscenes_download/640x480"
COMBINED_FILE = "D:/octscenes_download/image_640x480_combined.tar.gz"
EXTRACT_DIR = "D:/octscenes_extracted"
OUTPUT_DIR = "D:/logicsplat_data/octscenes"
NUM_SCENES = 20
FRAMES_PER_SCENE = 60
NUM_PARTS = 30


def format_size(size_bytes):
    """Format bytes to human-readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def format_time(seconds):
    """Format seconds to human-readable."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m"


# ── Step 1: Download ─────────────────────────────────────────────────────────
def step_download():
    """Download all 30 tar parts from HuggingFace."""
    print("=" * 60, flush=True)
    print("  Step 1: Download all 30 parts", flush=True)
    print("=" * 60, flush=True)
    
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    downloaded = 0
    skipped = 0
    start_time = time.time()
    
    for i in range(NUM_PARTS):
        filename = f"640x480/image_640x480_{i:02d}"
        local_path = os.path.join(DOWNLOAD_DIR, filename)
        
        # Check if already downloaded (4 GB expected for most parts)
        if os.path.exists(local_path):
            size = os.path.getsize(local_path)
            if size > 1_000_000_000:  # > 1 GB means likely complete
                print(f"  [{i+1:2d}/30] SKIP {filename} ({format_size(size)})", flush=True)
                skipped += 1
                continue
        
        print(f"  [{i+1:2d}/30] Downloading {filename}...", flush=True)
        t0 = time.time()
        
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=DOWNLOAD_DIR,
            )
            elapsed = time.time() - t0
            size = os.path.getsize(path)
            speed = size / elapsed / (1024**2) if elapsed > 0 else 0
            print(f"         Done: {format_size(size)} in {format_time(elapsed)} ({speed:.0f} MB/s)", flush=True)
            downloaded += 1
        except Exception as e:
            print(f"         ERROR: {e}", flush=True)
            print(f"         Retrying once...", flush=True)
            try:
                time.sleep(5)
                path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=DOWNLOAD_DIR,
                )
                downloaded += 1
                print(f"         Retry succeeded.", flush=True)
            except Exception as e2:
                print(f"         FAILED: {e2}", flush=True)
                print(f"         Run script again to resume.", flush=True)
                return False
    
    total_time = time.time() - start_time
    print(f"\n  Download complete: {downloaded} new, {skipped} skipped", flush=True)
    print(f"  Total time: {format_time(total_time)}", flush=True)
    
    # Verify all parts exist
    missing = []
    for i in range(NUM_PARTS):
        path = os.path.join(PARTS_DIR, f"image_640x480_{i:02d}")
        if not os.path.exists(path):
            missing.append(i)
    
    if missing:
        print(f"\n  WARNING: Missing parts: {missing}", flush=True)
        return False
    
    total_size = sum(
        os.path.getsize(os.path.join(PARTS_DIR, f"image_640x480_{i:02d}"))
        for i in range(NUM_PARTS)
    )
    print(f"  Total downloaded: {format_size(total_size)}", flush=True)
    return True


# ── Step 2: Concatenate ──────────────────────────────────────────────────────
def step_concat():
    """Concatenate all 30 parts into one tar.gz file."""
    print("\n" + "=" * 60, flush=True)
    print("  Step 2: Concatenate parts", flush=True)
    print("=" * 60, flush=True)
    
    if os.path.exists(COMBINED_FILE):
        size = os.path.getsize(COMBINED_FILE)
        print(f"  Combined file already exists: {format_size(size)}", flush=True)
        print(f"  Delete it to re-concatenate.", flush=True)
        return True
    
    parts = sorted(glob.glob(os.path.join(PARTS_DIR, "image_640x480_*")))
    if len(parts) < NUM_PARTS:
        print(f"  ERROR: Only {len(parts)}/{NUM_PARTS} parts found.", flush=True)
        return False
    
    print(f"  Concatenating {len(parts)} parts -> {COMBINED_FILE}", flush=True)
    total_size = sum(os.path.getsize(p) for p in parts)
    print(f"  Total input size: {format_size(total_size)}", flush=True)
    
    start_time = time.time()
    bytes_written = 0
    
    with open(COMBINED_FILE, 'wb') as outfile:
        for idx, part in enumerate(parts):
            part_size = os.path.getsize(part)
            print(f"  [{idx+1:2d}/30] Appending {os.path.basename(part)} ({format_size(part_size)})...", flush=True)
            with open(part, 'rb') as infile:
                while True:
                    chunk = infile.read(64 * 1024 * 1024)  # 64 MB chunks
                    if not chunk:
                        break
                    outfile.write(chunk)
                    bytes_written += len(chunk)
    
    elapsed = time.time() - start_time
    final_size = os.path.getsize(COMBINED_FILE)
    print(f"\n  Concatenation complete: {format_size(final_size)} in {format_time(elapsed)}", flush=True)
    return True


# ── Step 3: Extract ──────────────────────────────────────────────────────────
def step_extract():
    """Extract the combined tar.gz to get all PNG files."""
    print("\n" + "=" * 60, flush=True)
    print("  Step 3: Extract tar.gz", flush=True)
    print("=" * 60, flush=True)
    
    # Check if already extracted
    if os.path.isdir(EXTRACT_DIR):
        existing = glob.glob(os.path.join(EXTRACT_DIR, "**", "*.png"), recursive=True)
        if len(existing) > 100000:
            print(f"  Already extracted: {len(existing)} PNG files found", flush=True)
            return True
    
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    
    if not os.path.exists(COMBINED_FILE):
        print(f"  ERROR: Combined file not found: {COMBINED_FILE}", flush=True)
        print(f"  Run --step concat first.", flush=True)
        return False
    
    print(f"  Extracting: {COMBINED_FILE}", flush=True)
    print(f"  Destination: {EXTRACT_DIR}", flush=True)
    print(f"  This will take a while (decompressing ~118 GB)...", flush=True)
    
    start_time = time.time()
    count = 0
    
    try:
        with tarfile.open(COMBINED_FILE, "r:gz") as tar:
            for member in tar:
                tar.extract(member, EXTRACT_DIR)
                count += 1
                if count % 10000 == 0:
                    elapsed = time.time() - start_time
                    print(f"    Extracted {count:,} files ({format_time(elapsed)})...", flush=True)
    except Exception as e:
        print(f"\n  ERROR during extraction: {e}", flush=True)
        print(f"  Extracted {count} files before error.", flush=True)
        if count > 200000:
            print(f"  This may be enough — continuing with what we have.", flush=True)
        else:
            return False
    
    elapsed = time.time() - start_time
    print(f"\n  Extraction complete: {count:,} files in {format_time(elapsed)}", flush=True)
    return True


# ── Step 4: Organize ─────────────────────────────────────────────────────────
def step_organize():
    """Find 20 complete scenes and organize into data/octscenes/."""
    print("\n" + "=" * 60, flush=True)
    print("  Step 4: Organize 20 scenes", flush=True)
    print("=" * 60, flush=True)
    
    # Find all PNG files
    print("  Scanning extracted files...", flush=True)
    scenes = defaultdict(list)
    
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for fname in files:
            if fname.endswith('.png'):
                parts = fname.replace('.png', '').split('_')
                if len(parts) == 2:
                    scene_id, frame_id = parts
                    scenes[scene_id].append(os.path.join(root, fname))
    
    print(f"  Total scenes found: {len(scenes)}", flush=True)
    
    # Find complete scenes (60 frames)
    complete_scenes = {
        sid: sorted(frames) for sid, frames in scenes.items()
        if len(frames) == FRAMES_PER_SCENE
    }
    print(f"  Complete scenes (60 frames): {len(complete_scenes)}", flush=True)
    
    if len(complete_scenes) < NUM_SCENES:
        # If not enough complete scenes, take the ones with most frames
        print(f"  WARNING: Not enough complete scenes. Taking best available.", flush=True)
        sorted_scenes = sorted(scenes.items(), key=lambda x: len(x[1]), reverse=True)
        selected_ids = [sid for sid, _ in sorted_scenes[:NUM_SCENES]]
    else:
        # Pick 20 diverse scenes (evenly spaced through the sorted list)
        all_complete = sorted(complete_scenes.keys())
        random.seed(42)
        selected_ids = random.sample(all_complete, min(NUM_SCENES, len(all_complete)))
        selected_ids.sort()
    
    print(f"  Selected {len(selected_ids)} scenes: {selected_ids[:5]}...", flush=True)
    
    # Organize into output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}
    
    for idx, scene_id in enumerate(selected_ids, 1):
        oct_id = f"oct_{idx:02d}"
        oct_dir = os.path.join(OUTPUT_DIR, oct_id)
        images_dir = os.path.join(oct_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Get source frames
        if scene_id in complete_scenes:
            source_frames = complete_scenes[scene_id]
        else:
            source_frames = sorted(scenes[scene_id])
        
        # Copy with nerfstudio naming
        for frame_idx, src_path in enumerate(source_frames, 1):
            dst_path = os.path.join(images_dir, f"frame_{frame_idx:05d}.png")
            shutil.copy2(src_path, dst_path)
        
        # Save preview (middle frame)
        mid_frame = len(source_frames) // 2
        preview_src = source_frames[mid_frame] if source_frames else None
        if preview_src:
            shutil.copy2(preview_src, os.path.join(oct_dir, "preview.png"))
        
        results[oct_id] = {
            "source_scene_id": scene_id,
            "frames": len(source_frames),
        }
        print(f"  {oct_id} <- scene {scene_id} ({len(source_frames)} frames)", flush=True)
    
    # Save metadata
    meta_path = os.path.join(OUTPUT_DIR, "download_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "dataset": REPO_ID,
            "resolution": "640x480",
            "num_scenes": len(results),
            "scenes": results,
        }, f, indent=2)
    
    print(f"\n  Organized {len(results)} scenes to {os.path.abspath(OUTPUT_DIR)}", flush=True)
    return True


# ── Step 5: Cleanup ──────────────────────────────────────────────────────────
def step_cleanup():
    """Delete downloaded and extracted files to reclaim disk space."""
    print("\n" + "=" * 60, flush=True)
    print("  Step 5: Cleanup", flush=True)
    print("=" * 60, flush=True)
    
    freed = 0
    
    # Delete extracted files
    if os.path.isdir(EXTRACT_DIR):
        size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, dn, filenames in os.walk(EXTRACT_DIR)
            for f in filenames
        )
        print(f"  Removing extracted files: {format_size(size)}", flush=True)
        shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
        freed += size
    
    # Delete combined tar
    if os.path.exists(COMBINED_FILE):
        size = os.path.getsize(COMBINED_FILE)
        print(f"  Removing combined tar: {format_size(size)}", flush=True)
        os.remove(COMBINED_FILE)
        freed += size
    
    # Delete downloaded parts
    if os.path.isdir(DOWNLOAD_DIR):
        size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, dn, filenames in os.walk(DOWNLOAD_DIR)
            for f in filenames
        )
        print(f"  Removing download dir: {format_size(size)}", flush=True)
        shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)
        freed += size
    
    print(f"\n  Total space reclaimed: {format_size(freed)}", flush=True)
    return True


# ── Step 6: Verify ───────────────────────────────────────────────────────────
def step_verify():
    """Verify the final output."""
    print("\n" + "=" * 60, flush=True)
    print("  Verification", flush=True)
    print("=" * 60, flush=True)
    
    if not os.path.isdir(OUTPUT_DIR):
        print("  ERROR: Output directory not found!", flush=True)
        return False
    
    scenes = sorted([
        d for d in os.listdir(OUTPUT_DIR)
        if d.startswith("oct_") and os.path.isdir(os.path.join(OUTPUT_DIR, d))
    ])
    
    total_frames = 0
    total_size = 0
    issues = []
    
    for scene in scenes:
        scene_dir = os.path.join(OUTPUT_DIR, scene)
        images_dir = os.path.join(scene_dir, "images")
        
        if not os.path.isdir(images_dir):
            issues.append(f"{scene}: no images/ directory")
            continue
        
        frames = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        total_frames += len(frames)
        
        scene_size = sum(
            os.path.getsize(os.path.join(images_dir, f)) for f in frames
        )
        total_size += scene_size
        
        has_preview = os.path.exists(os.path.join(scene_dir, "preview.png"))
        
        if len(frames) != FRAMES_PER_SCENE:
            issues.append(f"{scene}: {len(frames)} frames (expected {FRAMES_PER_SCENE})")
        if not has_preview:
            issues.append(f"{scene}: missing preview.png")
    
    print(f"  Scenes: {len(scenes)}", flush=True)
    print(f"  Total frames: {total_frames}", flush=True)
    print(f"  Total size: {format_size(total_size)}", flush=True)
    
    if issues:
        print(f"\n  Issues ({len(issues)}):", flush=True)
        for issue in issues[:10]:
            print(f"    - {issue}", flush=True)
        return False
    else:
        print(f"\n  All {len(scenes)} scenes verified OK!", flush=True)
        print(f"\n  20 OCTScenes ready at {os.path.abspath(OUTPUT_DIR)}", flush=True)
        print(f"  Each scene: {FRAMES_PER_SCENE} frames at 640x480", flush=True)
        print(f"  Total disk used: {format_size(total_size)}", flush=True)
        print(f"\n  Next: python scripts/process_octscenes.py", flush=True)
        return True


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download OCTScenes 640x480 — full pipeline"
    )
    parser.add_argument(
        "--step", choices=["download", "concat", "extract", "organize", "cleanup", "verify", "all"],
        default="all",
        help="Run a specific step (default: all)"
    )
    args = parser.parse_args()
    
    print("=" * 60, flush=True)
    print("  OCTScenes 640x480 — Full Download Pipeline", flush=True)
    print("=" * 60, flush=True)
    print(f"  Target: {NUM_SCENES} scenes × {FRAMES_PER_SCENE} frames", flush=True)
    print(f"  Download dir: {DOWNLOAD_DIR}", flush=True)
    print(f"  Output dir: {os.path.abspath(OUTPUT_DIR)}", flush=True)
    print(f"  Disk needed: ~240 GB peak, ~600 MB final", flush=True)
    
    if args.step == "all":
        # Run full pipeline
        if not step_download():
            print("\n  ABORTED at download step.", flush=True)
            return
        if not step_concat():
            print("\n  ABORTED at concat step.", flush=True)
            return
        if not step_extract():
            print("\n  ABORTED at extract step.", flush=True)
            return
        if not step_organize():
            print("\n  ABORTED at organize step.", flush=True)
            return
        step_cleanup()
        step_verify()
    elif args.step == "download":
        step_download()
    elif args.step == "concat":
        step_concat()
    elif args.step == "extract":
        step_extract()
    elif args.step == "organize":
        step_organize()
    elif args.step == "cleanup":
        step_cleanup()
    elif args.step == "verify":
        step_verify()


if __name__ == "__main__":
    main()
