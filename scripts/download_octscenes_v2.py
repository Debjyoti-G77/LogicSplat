"""
Download 20 OCTScenes (640x480) from HuggingFace split tar archives.

Strategy:
  - The 640x480 images are stored as split tar parts (image_640x480_00 through _29)
  - Total: ~118 GB for all 5,000 scenes
  - We only need 20 scenes × 60 frames = 1,200 images
  - Download the FIRST tar part (~4 GB), try to extract scenes from it
  - If it's a split archive, we need to concatenate parts first
  - Pick 20 scenes with 60 frames each, organize into data/octscenes/

Usage:
    python scripts/download_octscenes_v2.py
    python scripts/download_octscenes_v2.py --download-only
    python scripts/download_octscenes_v2.py --extract-only
"""
import sys
sys.path.insert(0, ".")

import os
import shutil
import tarfile
import argparse
from pathlib import Path
from collections import defaultdict

try:
    from huggingface_hub import hf_hub_download, HfApi
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────────────────────
REPO_ID = "Yinxuan/OCTScenes"
DOWNLOAD_DIR = "D:/octscenes_download"
EXTRACT_DIR = "D:/octscenes_extracted"
OUTPUT_DIR = "D:/logicsplat_data/octscenes"
NUM_SCENES = 20
FRAMES_PER_SCENE = 60
RESOLUTION = "640x480"


def step1_explore():
    """List files in the 640x480 directory to understand structure."""
    print("=" * 60)
    print("  Step 1: Exploring OCTScenes repo structure")
    print("=" * 60)
    
    api = HfApi()
    files = list(api.list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo="640x480"))
    
    image_files = [f for f in files if "image_640x480" in f.path]
    print(f"\n  Image tar parts: {len(image_files)}")
    total_size = sum(f.size for f in image_files) / (1024**3)
    print(f"  Total image size: {total_size:.1f} GB")
    print(f"  First part: {image_files[0].path} ({image_files[0].size / (1024**3):.2f} GB)")
    
    return image_files


def step2_download_first_part():
    """Download just the first tar part to inspect its contents."""
    print("\n" + "=" * 60)
    print("  Step 2: Downloading first tar part")
    print("=" * 60)
    
    filename = "640x480/image_640x480_00"
    dest = os.path.join(DOWNLOAD_DIR, "image_640x480_00")
    
    if os.path.exists(dest):
        size_gb = os.path.getsize(dest) / (1024**3)
        print(f"  Already downloaded: {dest} ({size_gb:.2f} GB)")
        return dest
    
    print(f"  Downloading: {filename} (~4 GB)")
    print(f"  Destination: {DOWNLOAD_DIR}")
    
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=DOWNLOAD_DIR,
    )
    
    print(f"  Downloaded to: {path}")
    return path


def step3_inspect_tar(tar_path: str):
    """
    Try to open the tar file and inspect its contents.
    Split tar archives may not be valid on their own.
    """
    print("\n" + "=" * 60)
    print("  Step 3: Inspecting tar contents")
    print("=" * 60)
    
    # Try opening as a regular tar
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            members = tar.getnames()[:100]
            print(f"  Successfully opened as tar archive!")
            print(f"  First 20 entries:")
            for m in members[:20]:
                print(f"    {m}")
            print(f"  Total members (sampled): {len(members)}+")
            return "tar", members
    except tarfile.TarError as e:
        print(f"  Not a valid standalone tar: {e}")
    
    # Try as gzip tar
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getnames()[:100]
            print(f"  Successfully opened as tar.gz archive!")
            print(f"  First 20 entries:")
            for m in members[:20]:
                print(f"    {m}")
            return "tar.gz", members
    except Exception as e:
        print(f"  Not a valid tar.gz: {e}")
    
    # It's a split archive — need to check if it's a raw binary split
    print(f"\n  This appears to be a SPLIT archive (binary split of a tar.gz).")
    print(f"  Checking file header...")
    
    with open(tar_path, "rb") as f:
        header = f.read(512)
    
    # Check for gzip magic number
    if header[:2] == b'\x1f\x8b':
        print(f"  File starts with gzip magic number — it's a gzip-compressed split")
        return "split_gz", []
    
    # Check for tar magic
    if b'ustar' in header[257:265]:
        print(f"  File has tar magic at offset 257 — it's an uncompressed tar split")
        return "split_tar", []
    
    print(f"  Header bytes: {header[:32].hex()}")
    print(f"  This is likely a binary split of a tar.gz file.")
    print(f"  Need to concatenate all parts before extraction.")
    return "split_unknown", []


def step4_extract_from_tar(tar_path: str, tar_type: str, max_scenes: int = 30):
    """
    Extract images from the tar, grouping by scene.
    Only extract enough scenes to get our 20.
    """
    print("\n" + "=" * 60)
    print("  Step 4: Extracting scenes from tar")
    print("=" * 60)
    
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    
    if tar_type in ("tar", "tar.gz"):
        mode = "r:*" if tar_type == "tar" else "r:gz"
        scenes = defaultdict(list)
        
        with tarfile.open(tar_path, mode) as tar:
            print("  Scanning tar members...")
            members = tar.getmembers()
            print(f"  Total members: {len(members)}")
            
            # Group by scene_id
            for member in members:
                if member.isfile() and member.name.endswith('.png'):
                    basename = os.path.basename(member.name)
                    # Expected format: XXXXX_YY.png (scene_frame)
                    parts = basename.replace('.png', '').split('_')
                    if len(parts) >= 2:
                        scene_id = parts[0]
                        scenes[scene_id].append(member)
            
            print(f"  Found {len(scenes)} unique scenes in this tar part")
            
            # Find scenes with exactly 60 frames
            complete_scenes = {
                sid: members for sid, members in scenes.items()
                if len(members) == FRAMES_PER_SCENE
            }
            print(f"  Scenes with {FRAMES_PER_SCENE} frames: {len(complete_scenes)}")
            
            # Pick first N complete scenes
            selected = sorted(complete_scenes.keys())[:max_scenes]
            print(f"  Selecting {len(selected)} scenes: {selected[:5]}...")
            
            # Extract selected scenes
            for scene_id in tqdm(selected, desc="Extracting"):
                scene_extract_dir = os.path.join(EXTRACT_DIR, scene_id)
                os.makedirs(scene_extract_dir, exist_ok=True)
                for member in scenes[scene_id]:
                    tar.extract(member, EXTRACT_DIR)
            
            return selected
    else:
        print(f"  Cannot extract split archive directly.")
        print(f"  Need to download and concatenate multiple parts.")
        return []


def step5_organize_scenes(scene_ids: list):
    """
    Organize extracted scenes into data/octscenes/oct_XX/images/ format.
    """
    print("\n" + "=" * 60)
    print("  Step 5: Organizing into LogicSplat format")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find extracted images
    selected = scene_ids[:NUM_SCENES]
    results = {}
    
    for idx, scene_id in enumerate(selected, 1):
        oct_id = f"oct_{idx:02d}"
        oct_dir = os.path.join(OUTPUT_DIR, oct_id)
        images_dir = os.path.join(oct_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Find source images for this scene
        # They could be in EXTRACT_DIR directly or in a subdirectory
        source_images = []
        for root, dirs, files in os.walk(EXTRACT_DIR):
            for f in files:
                if f.startswith(scene_id) and f.endswith('.png'):
                    source_images.append(os.path.join(root, f))
        
        if not source_images:
            print(f"  WARNING: No images found for scene {scene_id}")
            continue
        
        # Sort by frame number and copy with nerfstudio naming
        source_images.sort()
        for frame_idx, src_path in enumerate(source_images, 1):
            dst_path = os.path.join(images_dir, f"frame_{frame_idx:05d}.png")
            shutil.copy2(src_path, dst_path)
        
        # Save preview (middle frame)
        preview_frame = os.path.join(images_dir, f"frame_{30:05d}.png")
        preview_dst = os.path.join(oct_dir, "preview.png")
        if os.path.exists(preview_frame):
            shutil.copy2(preview_frame, preview_dst)
        elif source_images:
            # Use middle image as preview
            mid = len(source_images) // 2
            shutil.copy2(source_images[mid], preview_dst)
        
        results[oct_id] = {
            "source_scene_id": scene_id,
            "frames": len(source_images),
        }
        print(f"  {oct_id} <- scene {scene_id} ({len(source_images)} frames)")
    
    # Save metadata
    import json
    meta_path = os.path.join(OUTPUT_DIR, "download_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "dataset": REPO_ID,
            "resolution": RESOLUTION,
            "num_scenes": len(results),
            "scenes": results,
        }, f, indent=2)
    
    return results


def step6_cleanup():
    """Delete extracted tar contents to save space."""
    print("\n" + "=" * 60)
    print("  Step 6: Cleanup")
    print("=" * 60)
    
    if os.path.exists(EXTRACT_DIR):
        size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, dn, filenames in os.walk(EXTRACT_DIR)
            for f in filenames
        ) / (1024**2)
        print(f"  Removing extracted files: {EXTRACT_DIR} ({size:.0f} MB)")
        shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
    
    # Optionally remove downloaded tar
    tar_path = os.path.join(DOWNLOAD_DIR, "640x480", "image_640x480_00")
    if os.path.exists(tar_path):
        size = os.path.getsize(tar_path) / (1024**3)
        print(f"  Removing downloaded tar: {tar_path} ({size:.2f} GB)")
        os.remove(tar_path)


def step7_verify():
    """Verify the final output."""
    print("\n" + "=" * 60)
    print("  Step 7: Verification")
    print("=" * 60)
    
    if not os.path.isdir(OUTPUT_DIR):
        print("  ERROR: Output directory not found!")
        return False
    
    scenes = sorted([
        d for d in os.listdir(OUTPUT_DIR)
        if d.startswith("oct_") and os.path.isdir(os.path.join(OUTPUT_DIR, d))
    ])
    
    print(f"  Scenes found: {len(scenes)}")
    
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
            os.path.getsize(os.path.join(images_dir, f))
            for f in frames
        )
        total_size += scene_size
        
        has_preview = os.path.exists(os.path.join(scene_dir, "preview.png"))
        
        if len(frames) != FRAMES_PER_SCENE:
            issues.append(f"{scene}: {len(frames)} frames (expected {FRAMES_PER_SCENE})")
        if not has_preview:
            issues.append(f"{scene}: missing preview.png")
    
    print(f"  Total frames: {total_frames}")
    print(f"  Total size: {total_size / (1024**2):.0f} MB")
    
    if issues:
        print(f"\n  Issues ({len(issues)}):")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"\n  All {len(scenes)} scenes verified OK!")
    
    print(f"\n  Summary:")
    print(f"    Location: {os.path.abspath(OUTPUT_DIR)}")
    print(f"    Scenes: {len(scenes)}")
    print(f"    Frames per scene: ~{total_frames // max(len(scenes), 1)}")
    print(f"    Resolution: 640x480")
    print(f"    Total disk: {total_size / (1024**2):.0f} MB")
    
    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(description="Download OCTScenes 640x480")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--skip-cleanup", action="store_true")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  OCTScenes 640x480 Downloader (v2)")
    print("=" * 60)
    print(f"  Target: {NUM_SCENES} scenes × {FRAMES_PER_SCENE} frames")
    print(f"  Resolution: {RESOLUTION}")
    print(f"  Download dir: {DOWNLOAD_DIR}")
    print(f"  Output dir: {os.path.abspath(OUTPUT_DIR)}")
    
    # Check if already done
    existing = [
        d for d in os.listdir(OUTPUT_DIR) if d.startswith("oct_")
    ] if os.path.isdir(OUTPUT_DIR) else []
    if len(existing) >= NUM_SCENES:
        print(f"\n  Already have {len(existing)} scenes. Running verification...")
        step7_verify()
        return
    
    # Step 1: Explore
    step1_explore()
    
    # Step 2: Download first tar part
    tar_path = step2_download_first_part()
    
    if args.download_only:
        print("\n  Download complete. Run again without --download-only to extract.")
        return
    
    # Step 3: Inspect tar format
    tar_type, sample_members = step3_inspect_tar(tar_path)
    
    # Step 4: Extract scenes
    if tar_type in ("tar", "tar.gz"):
        scene_ids = step4_extract_from_tar(tar_path, tar_type)
    elif tar_type in ("split_gz", "split_tar", "split_unknown"):
        # Need to handle split archive
        print("\n  Split archive detected. Attempting partial extraction...")
        # Try treating as raw tar anyway (sometimes split files are valid tars)
        try:
            scene_ids = step4_extract_from_tar(tar_path, "tar")
        except Exception as e:
            print(f"  Failed: {e}")
            print(f"\n  The archive is split and cannot be extracted from a single part.")
            print(f"  Options:")
            print(f"    1. Download ALL parts and concatenate (118 GB)")
            print(f"    2. Use 256x256 resolution instead (smaller download)")
            print(f"    3. Use a different dataset")
            return
    else:
        print(f"  Unknown tar type: {tar_type}")
        return
    
    if not scene_ids:
        print("  No scenes extracted. Aborting.")
        return
    
    # Step 5: Organize
    results = step5_organize_scenes(scene_ids)
    
    # Step 6: Cleanup
    if not args.skip_cleanup:
        step6_cleanup()
    
    # Step 7: Verify
    step7_verify()
    
    print(f"\n  Next steps:")
    print(f"    python scripts/process_octscenes.py")
    print(f"    python scripts/prepare_annotation.py")


if __name__ == "__main__":
    main()
