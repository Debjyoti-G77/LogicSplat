"""
Download 3RScan Gaussian Splats from HuggingFace for benchmarking against ReLaGS.

Dataset: GaussianWorld/3rscan_mcmc_3dgs
Only downloads scenes that overlap with our 3DSSG relationship annotations.

Usage:
    python scripts/download_3rscan_splats.py
    python scripts/download_3rscan_splats.py --all          # Download all scenes (not just overlap)
    python scripts/download_3rscan_splats.py --max-scenes 10  # Download only first N scenes (for testing)
    python scripts/download_3rscan_splats.py --ply-only     # Download only .ply files (skip cfg/stats/tb)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, hf_hub_download


# === Configuration ===
REPO_ID = "GaussianWorld/3rscan_mcmc_3dgs"
REPO_TYPE = "dataset"
OUTPUT_DIR = Path(r"D:\3rscan_splats")
RELATIONSHIPS_JSON = Path(__file__).parent.parent / "data" / "3DSSG" / "relationships.json"
MAX_WORKERS = 4  # Parallel download threads


def get_ssg_scene_ids() -> set:
    """Extract unique scene IDs from relationships.json."""
    with open(RELATIONSHIPS_JSON, "r") as f:
        data = json.load(f)
    return set(s["scan"] for s in data["scans"])


def get_hf_files_by_scene(api: HfApi) -> dict:
    """Get all files from HuggingFace dataset, grouped by scene ID."""
    files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    scenes = {}
    for f in files:
        parts = f.split("/")
        if len(parts) > 1 and "-" in parts[0] and len(parts[0]) > 30:
            scene_id = parts[0]
            if scene_id not in scenes:
                scenes[scene_id] = []
            scenes[scene_id].append(f)
    return scenes


def download_file(repo_id: str, filename: str, output_dir: Path) -> tuple:
    """Download a single file from HuggingFace. Returns (filename, success, size_bytes)."""
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=REPO_TYPE,
            local_dir=str(output_dir),
        )
        size = os.path.getsize(local_path)
        return (filename, True, size)
    except Exception as e:
        return (filename, False, str(e))


def main():
    parser = argparse.ArgumentParser(description="Download 3RScan Gaussian Splats from HuggingFace")
    parser.add_argument("--all", action="store_true", help="Download all scenes, not just overlap with 3DSSG")
    parser.add_argument("--max-scenes", type=int, default=None, help="Limit number of scenes to download")
    parser.add_argument("--ply-only", action="store_true", help="Download only .ply files (skip cfg/stats/tb)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Number of parallel download threads")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("3RScan Gaussian Splat Downloader")
    print("=" * 60)
    print(f"Repository: {REPO_ID}")
    print(f"Output dir: {output_dir}")
    print()

    # Step 1: Get scene IDs from 3DSSG
    print("[1/4] Loading scene IDs from relationships.json...")
    ssg_scenes = get_ssg_scene_ids()
    print(f"  Found {len(ssg_scenes)} scenes in 3DSSG annotations")

    # Step 2: Get file listing from HuggingFace
    print("[2/4] Listing files on HuggingFace...")
    api = HfApi()
    hf_scenes = get_hf_files_by_scene(api)
    print(f"  Found {len(hf_scenes)} scenes in HuggingFace dataset")

    # Step 3: Determine which scenes to download
    if args.all:
        target_scenes = set(hf_scenes.keys())
        print(f"[3/4] Downloading ALL {len(target_scenes)} scenes from HuggingFace")
    else:
        target_scenes = ssg_scenes & set(hf_scenes.keys())
        print(f"[3/4] Overlap: {len(target_scenes)} scenes have both annotations and splats")
        print(f"  ({len(ssg_scenes - set(hf_scenes.keys()))} scenes in 3DSSG lack splats)")

    if args.max_scenes:
        target_scenes = set(sorted(target_scenes)[: args.max_scenes])
        print(f"  Limited to {len(target_scenes)} scenes (--max-scenes)")

    # Build file list
    files_to_download = []
    for scene_id in sorted(target_scenes):
        for f in hf_scenes[scene_id]:
            if args.ply_only and not f.endswith(".ply"):
                continue
            # Skip if already downloaded
            local_path = output_dir / f
            if local_path.exists():
                continue
            files_to_download.append(f)

    print(f"\n[4/4] Downloading {len(files_to_download)} files...")
    if not files_to_download:
        print("  All files already downloaded!")
        _print_summary(output_dir, target_scenes, ssg_scenes)
        return

    # Download with progress
    downloaded = 0
    failed = 0
    total_bytes = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_file, REPO_ID, f, output_dir): f
            for f in files_to_download
        }
        for future in as_completed(futures):
            filename, success, result = future.result()
            if success:
                downloaded += 1
                total_bytes += result
                if downloaded % 10 == 0 or downloaded == len(files_to_download):
                    elapsed = time.time() - start_time
                    speed = total_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0
                    print(
                        f"  [{downloaded}/{len(files_to_download)}] "
                        f"{total_bytes / 1024 / 1024:.1f} MB downloaded "
                        f"({speed:.1f} MB/s)"
                    )
            else:
                failed += 1
                print(f"  FAILED: {filename} - {result}")

    elapsed = time.time() - start_time
    print(f"\nDownload complete in {elapsed:.1f}s")
    print(f"  Downloaded: {downloaded} files ({total_bytes / 1024 / 1024 / 1024:.2f} GB)")
    if failed:
        print(f"  Failed: {failed} files")

    _print_summary(output_dir, target_scenes, ssg_scenes)


def _print_summary(output_dir: Path, target_scenes: set, ssg_scenes: set):
    """Print summary of downloaded data."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Count actual downloaded scenes
    downloaded_scenes = []
    total_size = 0
    for d in output_dir.iterdir():
        if d.is_dir() and "-" in d.name and len(d.name) > 30:
            ply_path = d / "ckpts" / "point_cloud_30000.ply"
            if ply_path.exists():
                downloaded_scenes.append(d.name)
                total_size += ply_path.stat().st_size

    print(f"Scenes downloaded: {len(downloaded_scenes)}")
    print(f"Total PLY size: {total_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"File format: point_cloud_30000.ply (3DGS MCMC, 30k iterations)")
    print(f"Path pattern: {output_dir}/{{scene_id}}/ckpts/point_cloud_30000.ply")

    # Check overlap with 3DSSG
    overlap = set(downloaded_scenes) & ssg_scenes
    print(f"\nScenes matching 3DSSG annotations: {len(overlap)} / {len(downloaded_scenes)}")

    # Save manifest
    manifest_path = output_dir / "download_manifest.json"
    manifest = {
        "repo_id": REPO_ID,
        "total_scenes_downloaded": len(downloaded_scenes),
        "scenes_with_3dssg_annotations": len(overlap),
        "total_ply_size_gb": round(total_size / 1024 / 1024 / 1024, 2),
        "scene_ids": sorted(downloaded_scenes),
        "file_format": "point_cloud_30000.ply (3DGS MCMC trained for 30k iterations)",
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
