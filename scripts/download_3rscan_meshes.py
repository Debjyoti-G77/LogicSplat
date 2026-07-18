"""
Download 3RScan mesh + instance segmentation files for scenes we have splats for.

Downloads per scene:
  - labels.instances.annotated.v2.ply  (per-vertex instance labels — for NN label transfer)
  - semseg.v2.json                     (per-face semantic segmentation with object groupings)
  - mesh.refined.v2.obj                (the mesh itself — vertex positions)

Only downloads for scenes that have BOTH a Gaussian splat AND 3DSSG annotations.

Usage:
    python scripts/download_3rscan_meshes.py
    python scripts/download_3rscan_meshes.py --workers 8
    python scripts/download_3rscan_meshes.py --max-scenes 10   # Test with a few scenes first
    python scripts/download_3rscan_meshes.py --dry-run          # Show what would be downloaded
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# === Configuration ===
BASE_URL = "http://campar.in.tum.de/public_datasets/3RScan/Dataset"
FILES_TO_DOWNLOAD = [
    "labels.instances.annotated.v2.ply",
    "semseg.v2.json",
    "mesh.refined.v2.obj",
]
OUTPUT_DIR = Path(r"D:\3rscan_meshes")
SPLATS_DIR = Path(r"D:\3rscan_splats")
MANIFEST_PATH = SPLATS_DIR / "download_manifest.json"
RELATIONSHIPS_JSON = Path(__file__).parent.parent / "data" / "3DSSG" / "relationships.json"

MAX_WORKERS = 4
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds between retries
TIMEOUT = 120  # seconds per download


def get_splat_scene_ids() -> set:
    """Get scene IDs from the download manifest, or scan the splats directory."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r") as f:
            manifest = json.load(f)
        scene_ids = set(manifest.get("scene_ids", []))
        if scene_ids:
            return scene_ids

    # Fallback: scan directory for scenes with point_cloud_30000.ply
    scene_ids = set()
    if SPLATS_DIR.exists():
        for d in SPLATS_DIR.iterdir():
            if d.is_dir() and "-" in d.name and len(d.name) > 30:
                ply_path = d / "ckpts" / "point_cloud_30000.ply"
                if ply_path.exists():
                    scene_ids.add(d.name)
    return scene_ids


def get_ssg_scene_ids() -> set:
    """Extract unique scene IDs from 3DSSG relationships.json."""
    if not RELATIONSHIPS_JSON.exists():
        print(f"  WARNING: {RELATIONSHIPS_JSON} not found, skipping 3DSSG filter")
        return None
    with open(RELATIONSHIPS_JSON, "r") as f:
        data = json.load(f)
    return set(s["scan"] for s in data["scans"])


def download_file(scene_id: str, filename: str, output_dir: Path) -> dict:
    """
    Download a single file with retry logic.
    Returns dict with scene_id, filename, success, size_bytes or error.
    """
    url = f"{BASE_URL}/{scene_id}/{filename}"
    local_path = output_dir / scene_id / filename

    # Skip if already downloaded
    if local_path.exists() and local_path.stat().st_size > 0:
        return {
            "scene_id": scene_id,
            "filename": filename,
            "success": True,
            "skipped": True,
            "size_bytes": local_path.stat().st_size,
        }

    # Create directory
    local_path.parent.mkdir(parents=True, exist_ok=True)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            urllib.request.urlretrieve(url, str(local_path))
            size = local_path.stat().st_size
            if size == 0:
                local_path.unlink()
                raise ValueError("Downloaded file is empty (0 bytes)")
            return {
                "scene_id": scene_id,
                "filename": filename,
                "success": True,
                "skipped": False,
                "size_bytes": size,
            }
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as e:
            last_error = str(e)
            # Clean up partial download
            if local_path.exists():
                try:
                    local_path.unlink()
                except OSError:
                    pass
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)  # Exponential-ish backoff

    return {
        "scene_id": scene_id,
        "filename": filename,
        "success": False,
        "skipped": False,
        "error": last_error,
    }


def main():
    parser = argparse.ArgumentParser(description="Download 3RScan mesh + instance segmentation files")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel download threads (default: 4)")
    parser.add_argument("--max-scenes", type=int, default=None, help="Limit number of scenes (for testing)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without downloading")
    parser.add_argument("--skip-ssg-filter", action="store_true", help="Download for all splat scenes, not just those with 3DSSG annotations")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("3RScan Mesh + Instance Segmentation Downloader")
    print("=" * 70)
    print(f"  Source:     {BASE_URL}")
    print(f"  Output:     {output_dir}")
    print(f"  Files/scene: {', '.join(FILES_TO_DOWNLOAD)}")
    print(f"  Workers:    {args.workers}")
    print()

    # Step 1: Get scene IDs from splats
    print("[1/4] Loading scene IDs from splats...")
    splat_scenes = get_splat_scene_ids()
    if not splat_scenes:
        print("  ERROR: No splat scenes found! Check D:\\3rscan_splats or download_manifest.json")
        sys.exit(1)
    print(f"  Found {len(splat_scenes)} scenes with Gaussian splats")

    # Step 2: Cross-reference with 3DSSG annotations
    print("[2/4] Cross-referencing with 3DSSG annotations...")
    if args.skip_ssg_filter:
        target_scenes = splat_scenes
        print(f"  Skipping 3DSSG filter (--skip-ssg-filter)")
    else:
        ssg_scenes = get_ssg_scene_ids()
        if ssg_scenes is None:
            target_scenes = splat_scenes
        else:
            target_scenes = splat_scenes & ssg_scenes
            print(f"  3DSSG annotations: {len(ssg_scenes)} scenes")
            print(f"  Overlap (splat + 3DSSG): {len(target_scenes)} scenes")
            skipped = splat_scenes - ssg_scenes
            if skipped:
                print(f"  Skipping {len(skipped)} splat scenes without 3DSSG annotations")

    if args.max_scenes:
        target_scenes = set(sorted(target_scenes)[:args.max_scenes])
        print(f"  Limited to {len(target_scenes)} scenes (--max-scenes)")

    # Step 3: Build download list (skip already-downloaded files)
    print(f"\n[3/4] Building download list for {len(target_scenes)} scenes...")
    download_tasks = []
    skipped_count = 0
    for scene_id in sorted(target_scenes):
        for filename in FILES_TO_DOWNLOAD:
            local_path = output_dir / scene_id / filename
            if local_path.exists() and local_path.stat().st_size > 0:
                skipped_count += 1
            else:
                download_tasks.append((scene_id, filename))

    total_files = len(target_scenes) * len(FILES_TO_DOWNLOAD)
    print(f"  Total files needed: {total_files}")
    print(f"  Already downloaded: {skipped_count}")
    print(f"  Files to download:  {len(download_tasks)}")
    print(f"  Estimated size:     ~{len(download_tasks) * 10:.0f} MB (rough estimate)")

    if args.dry_run:
        print("\n[DRY RUN] Would download:")
        for scene_id, filename in download_tasks[:20]:
            print(f"  {BASE_URL}/{scene_id}/{filename}")
        if len(download_tasks) > 20:
            print(f"  ... and {len(download_tasks) - 20} more files")
        return

    if not download_tasks:
        print("\n  All files already downloaded!")
        _print_verification(output_dir, target_scenes)
        return

    # Step 4: Download
    print(f"\n[4/4] Downloading {len(download_tasks)} files with {args.workers} workers...")
    print("  (Ctrl+C to interrupt — resume support means you can restart anytime)")
    print()

    downloaded = 0
    failed_files = []
    total_bytes = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_file, scene_id, filename, output_dir): (scene_id, filename)
            for scene_id, filename in download_tasks
        }

        try:
            for future in as_completed(futures):
                result = future.result()
                if result["success"]:
                    if not result["skipped"]:
                        downloaded += 1
                        total_bytes += result["size_bytes"]

                    # Progress report every 10 files or at milestones
                    if downloaded % 10 == 0 and downloaded > 0:
                        elapsed = time.time() - start_time
                        speed = total_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0
                        pct = downloaded / len(download_tasks) * 100
                        eta_s = (len(download_tasks) - downloaded) / (downloaded / elapsed) if downloaded > 0 else 0
                        eta_m = eta_s / 60
                        print(
                            f"  [{downloaded}/{len(download_tasks)}] ({pct:.1f}%) "
                            f"{total_bytes / 1024 / 1024:.0f} MB @ {speed:.1f} MB/s "
                            f"ETA: {eta_m:.0f} min"
                        )
                else:
                    failed_files.append(result)
                    print(
                        f"  FAILED: {result['scene_id']}/{result['filename']} "
                        f"- {result['error']}"
                    )
        except KeyboardInterrupt:
            print("\n\n  Interrupted! Cancelling remaining downloads...")
            executor.shutdown(wait=False, cancel_futures=True)
            print("  You can resume by running this script again (already-downloaded files will be skipped).")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Download complete in {elapsed / 60:.1f} minutes")
    print(f"  Downloaded: {downloaded} files ({total_bytes / 1024 / 1024 / 1024:.2f} GB)")
    if failed_files:
        print(f"  Failed:     {len(failed_files)} files")

    # Save failure log
    if failed_files:
        fail_log_path = output_dir / "download_failures.json"
        with open(fail_log_path, "w") as f:
            json.dump(failed_files, f, indent=2)
        print(f"  Failure log: {fail_log_path}")

    _print_verification(output_dir, target_scenes)


def _print_verification(output_dir: Path, target_scenes: set):
    """Verify downloads and print summary."""
    print(f"\n{'=' * 70}")
    print("VERIFICATION")
    print("=" * 70)

    complete_scenes = []
    partial_scenes = []
    missing_scenes = []
    total_size = 0
    file_counts = {f: 0 for f in FILES_TO_DOWNLOAD}

    for scene_id in sorted(target_scenes):
        scene_dir = output_dir / scene_id
        if not scene_dir.exists():
            missing_scenes.append(scene_id)
            continue

        scene_files = []
        for filename in FILES_TO_DOWNLOAD:
            fpath = scene_dir / filename
            if fpath.exists() and fpath.stat().st_size > 0:
                scene_files.append(filename)
                file_counts[filename] += 1
                total_size += fpath.stat().st_size

        if len(scene_files) == len(FILES_TO_DOWNLOAD):
            complete_scenes.append(scene_id)
        elif len(scene_files) > 0:
            partial_scenes.append((scene_id, scene_files))
        else:
            missing_scenes.append(scene_id)

    print(f"\n  Scenes with ALL 3 files:  {len(complete_scenes)} / {len(target_scenes)}")
    print(f"  Scenes with partial data: {len(partial_scenes)}")
    print(f"  Scenes with no data:      {len(missing_scenes)}")
    print(f"\n  File breakdown:")
    for filename, count in file_counts.items():
        print(f"    {filename}: {count} scenes")
    print(f"\n  Total download size: {total_size / 1024 / 1024 / 1024:.2f} GB")

    if partial_scenes:
        print(f"\n  Partial scenes (missing some files):")
        for scene_id, files in partial_scenes[:10]:
            missing = set(FILES_TO_DOWNLOAD) - set(files)
            print(f"    {scene_id}: missing {', '.join(missing)}")
        if len(partial_scenes) > 10:
            print(f"    ... and {len(partial_scenes) - 10} more")

    # Save verification report
    report = {
        "total_target_scenes": len(target_scenes),
        "complete_scenes": len(complete_scenes),
        "partial_scenes": len(partial_scenes),
        "missing_scenes": len(missing_scenes),
        "total_size_gb": round(total_size / 1024 / 1024 / 1024, 2),
        "file_counts": file_counts,
        "complete_scene_ids": complete_scenes,
        "partial_scene_details": [
            {"scene_id": s, "has_files": f} for s, f in partial_scenes
        ],
        "missing_scene_ids": missing_scenes,
    }
    report_path = output_dir / "download_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
