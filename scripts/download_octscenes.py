"""
Download 20 diverse scenes from OCTScenes (Hugging Face).

OCTScenes: 5,000 tabletop scenes, 15 everyday objects, 60 frames per scene (360°).
Dataset: https://huggingface.co/datasets/Yinxuan/OCTScenes

Selection criteria:
  - 3-5 objects per scene (similar to our test scenes)
  - Diverse object types
  - Clear spatial arrangements

Output:
  data/octscenes/oct_01/images/frame_00001.png ... frame_00060.png
  data/octscenes/oct_01/preview.png  (frame 30 — middle frame)
  ...
  data/octscenes/oct_20/images/...

Usage:
    pip install datasets
    python scripts/download_octscenes.py
    python scripts/download_octscenes.py --num-scenes 5   # download fewer for testing
    python scripts/download_octscenes.py --explore        # just explore dataset structure
"""
import sys
sys.path.insert(0, ".")

import os
import json
import argparse
import shutil
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' package not installed.")
    print("  Install with: pip install datasets")
    sys.exit(1)

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("ERROR: Pillow or numpy not installed.")
    print("  Install with: pip install Pillow numpy")
    sys.exit(1)

from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "D:/logicsplat_data/octscenes"
DATASET_NAME = "Yinxuan/OCTScenes"
FRAMES_PER_SCENE = 60
PREVIEW_FRAME = 30  # middle frame for annotation preview
NUM_SCENES = 20


def explore_dataset():
    """Explore the OCTScenes dataset structure without downloading everything."""
    print("=" * 60)
    print("  OCTScenes Dataset Explorer")
    print("=" * 60)

    print("\nLoading dataset in streaming mode...")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    print("\nDataset info:")
    print(f"  Name: {DATASET_NAME}")
    print(f"  Features: {ds.features}")

    print("\nSampling first 5 entries to understand structure...")
    for i, sample in enumerate(ds):
        if i >= 5:
            break
        print(f"\n--- Sample {i} ---")
        print(f"  Keys: {list(sample.keys())}")
        for key, value in sample.items():
            if isinstance(value, (str, int, float, bool)):
                print(f"  {key}: {value}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with keys {list(value.keys())}")
            elif isinstance(value, list):
                print(f"  {key}: list of length {len(value)}")
                if len(value) > 0:
                    print(f"    first element type: {type(value[0])}")
            elif hasattr(value, 'size'):
                # PIL Image
                print(f"  {key}: Image {value.size} mode={value.mode}")
            else:
                print(f"  {key}: {type(value)}")

    print("\n\nDone exploring. Use this info to adjust the download script.")


def select_diverse_scenes(ds, num_scenes: int = NUM_SCENES, max_scan: int = 200):
    """
    Scan through the dataset and select diverse scenes.

    Strategy:
    - Scan up to max_scan scenes
    - Pick scenes that are spread out (every N-th scene) to get diversity
    - Prefer scenes with clear images (not too dark/bright)
    """
    print(f"\nScanning up to {max_scan} scenes to select {num_scenes} diverse ones...")

    candidates = []
    for i, sample in enumerate(tqdm(ds, total=max_scan, desc="Scanning")):
        if i >= max_scan:
            break
        candidates.append((i, sample))

    if len(candidates) < num_scenes:
        print(f"  Only found {len(candidates)} scenes, using all of them")
        selected_indices = list(range(len(candidates)))
    else:
        # Evenly space selections across the scanned range for diversity
        step = len(candidates) // num_scenes
        selected_indices = [i * step for i in range(num_scenes)]

    print(f"  Selected {len(selected_indices)} scenes from {len(candidates)} candidates")
    return [(candidates[i][0], candidates[i][1]) for i in selected_indices]


def save_scene(scene_data: dict, scene_idx: int, scene_id: str, output_dir: str):
    """
    Save a single scene's frames to disk.

    Handles multiple possible dataset formats:
    1. scene_data has 'images' key with list of PIL Images
    2. scene_data has numbered frame keys like 'frame_0', 'frame_1', ...
    3. scene_data has a single 'image' key (one frame per row)
    """
    scene_dir = os.path.join(output_dir, scene_id)
    images_dir = os.path.join(scene_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    frames_saved = 0

    # Strategy 1: 'images' key with list
    if 'images' in scene_data and isinstance(scene_data['images'], list):
        for frame_idx, img in enumerate(scene_data['images']):
            if hasattr(img, 'save'):  # PIL Image
                frame_path = os.path.join(images_dir, f"frame_{frame_idx + 1:05d}.png")
                img.save(frame_path)
                frames_saved += 1
                # Save preview (middle frame)
                if frame_idx + 1 == PREVIEW_FRAME:
                    img.save(os.path.join(scene_dir, "preview.png"))

    # Strategy 2: 'image' key (single image per row — dataset is flattened)
    elif 'image' in scene_data and hasattr(scene_data['image'], 'save'):
        frame_path = os.path.join(images_dir, f"frame_{1:05d}.png")
        scene_data['image'].save(frame_path)
        frames_saved = 1
        scene_data['image'].save(os.path.join(scene_dir, "preview.png"))

    # Strategy 3: numbered frame keys
    else:
        for key in sorted(scene_data.keys()):
            if 'frame' in key.lower() or 'image' in key.lower() or 'img' in key.lower():
                val = scene_data[key]
                if hasattr(val, 'save'):
                    frames_saved += 1
                    frame_path = os.path.join(images_dir, f"frame_{frames_saved:05d}.png")
                    val.save(frame_path)
                    if frames_saved == PREVIEW_FRAME:
                        val.save(os.path.join(scene_dir, "preview.png"))

    # If no preview was saved (fewer than PREVIEW_FRAME frames), use last frame
    preview_path = os.path.join(scene_dir, "preview.png")
    if not os.path.exists(preview_path) and frames_saved > 0:
        # Copy the middle available frame
        mid = max(1, frames_saved // 2)
        mid_path = os.path.join(images_dir, f"frame_{mid:05d}.png")
        if os.path.exists(mid_path):
            shutil.copy2(mid_path, preview_path)

    return frames_saved


def download_scenes(num_scenes: int = NUM_SCENES, max_scan: int = 200):
    """Download and save selected OCTScenes."""
    print("=" * 60)
    print("  OCTScenes Downloader")
    print("=" * 60)
    print(f"  Target: {num_scenes} scenes")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if already downloaded
    existing = [d for d in os.listdir(OUTPUT_DIR)
                if d.startswith("oct_") and os.path.isdir(os.path.join(OUTPUT_DIR, d))]
    if len(existing) >= num_scenes:
        print(f"\n  Already have {len(existing)} scenes downloaded. Use --force to re-download.")
        return

    print("\nLoading dataset (streaming)...")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    # Select diverse scenes
    selected = select_diverse_scenes(ds, num_scenes=num_scenes, max_scan=max_scan)

    # Save each scene
    results = {}
    for local_idx, (global_idx, scene_data) in enumerate(
        tqdm(selected, desc="Saving scenes")
    ):
        scene_id = f"oct_{local_idx + 1:02d}"
        frames = save_scene(scene_data, global_idx, scene_id, OUTPUT_DIR)
        results[scene_id] = {
            "global_index": global_idx,
            "frames_saved": frames,
        }
        print(f"  {scene_id}: {frames} frames saved (from dataset index {global_idx})")

    # Save metadata
    meta_path = os.path.join(OUTPUT_DIR, "download_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "dataset": DATASET_NAME,
            "num_scenes": len(results),
            "scenes": results,
        }, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("  Download Complete")
    print(f"{'='*60}")
    total_frames = sum(r["frames_saved"] for r in results.values())
    print(f"  Scenes: {len(results)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Metadata: {meta_path}")
    print(f"\nNext step: python scripts/process_octscenes.py")


def main():
    parser = argparse.ArgumentParser(
        description="Download OCTScenes tabletop dataset from Hugging Face"
    )
    parser.add_argument(
        "--explore", action="store_true",
        help="Just explore dataset structure without downloading"
    )
    parser.add_argument(
        "--num-scenes", type=int, default=NUM_SCENES,
        help=f"Number of scenes to download (default: {NUM_SCENES})"
    )
    parser.add_argument(
        "--max-scan", type=int, default=200,
        help="Max scenes to scan for diversity selection (default: 200)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if scenes already exist"
    )
    args = parser.parse_args()

    if args.explore:
        explore_dataset()
    else:
        if args.force:
            # Clear existing
            if os.path.exists(OUTPUT_DIR):
                shutil.rmtree(OUTPUT_DIR)
        download_scenes(num_scenes=args.num_scenes, max_scan=args.max_scan)


if __name__ == "__main__":
    main()
