"""
Download 3RScan RGB-D sequences + mesh files for the 46 RIO10 test scenes.

Downloads per scene:
  - sequence.zip          (RGB-D frames + camera poses)
  - labels.instances.annotated.v2.ply  (instance labels)
  - semseg.v2.json        (semantic segmentation)
  - mesh.refined.v2.obj   (mesh)

Usage:
    python scripts/download_3rscan_test_sequences.py
    python scripts/download_3rscan_test_sequences.py --workers 4
    python scripts/download_3rscan_test_sequences.py --max-scenes 5   # Test with a few scenes
    python scripts/download_3rscan_test_sequences.py --dry-run
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
    "sequence.zip",
    "labels.instances.annotated.v2.ply",
    "semseg.v2.json",
    "mesh.refined.v2.obj",
]
OUTPUT_DIR = Path(r"D:\3rscan_test")
RIO10_TEST_SCENES = Path(__file__).parent.parent / "data" / "3DSSG" / "rio10_test_scenes.txt"

MAX_WORKERS = 4
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds between retries
TIMEOUT = 600  # seconds per download (sequence.zip can be large)
CHUNK_SIZE = 8192  # bytes for streaming download


def load_test_scene_ids() -> list:
    """Load the 46 RIO10 test scene IDs (handles UTF-16 BOM)."""
    if not RIO10_TEST_SCENES.exists():
        print(f"ERROR: {RIO10_TEST_SCENES} not found!")
        sys.exit(1)
    # File is UTF-16 LE encoded — detect and handle
    raw = RIO10_TEST_SCENES.read_bytes()
    if raw[:2] == b'\xff\xfe':
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig")
    scene_ids = [line.strip() for line in text.splitlines() if line.strip()]
    return scene_ids


def download_file_with_resume(scene_id: str, filename: str, output_dir: Path) -> dict:
    """
    Download a single file with retry logic and resume support.
    Returns dict with scene_id, filename, success, size_bytes or error.
    """
    url = f"{BASE_URL}/{scene_id}/{filename}"
    local_path = output_dir / scene_id / filename
    partial_path = local_path.with_suffix(local_path.suffix + ".partial")

    # Skip if already downloaded and valid
    if local_path.exists() and local_path.stat().st_size > 0:
        # Extra validation for zip files — check magic bytes
        if filename.endswith(".zip"):
            try:
                with open(str(local_path), "rb") as f:
                    magic = f.read(4)
                if magic != b'PK\x03\x04' and magic != b'PK\x05\x06':
                    # Corrupt zip — delete and re-download
                    local_path.unlink()
                else:
                    return {
                        "scene_id": scene_id,
                        "filename": filename,
                        "success": True,
                        "skipped": True,
                        "size_bytes": local_path.stat().st_size,
                    }
            except OSError:
                pass
        else:
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
            # Check if partial download exists for resume
            existing_size = 0
            if partial_path.exists():
                existing_size = partial_path.stat().st_size

            # Build request with Range header for resume
            req = urllib.request.Request(url)
            if existing_size > 0:
                req.add_header("Range", f"bytes={existing_size}-")

            response = urllib.request.urlopen(req, timeout=TIMEOUT)

            # Check if server supports range requests
            if existing_size > 0 and response.status == 206:
                # Partial content — append
                mode = "ab"
            else:
                # Full download — overwrite
                mode = "wb"
                existing_size = 0

            # Get total size if available
            content_length = response.headers.get("Content-Length")
            total_size = int(content_length) + existing_size if content_length else None

            # Stream download
            with open(str(partial_path), mode) as f:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)

            # Verify download
            final_size = partial_path.stat().st_size
            if final_size == 0:
                partial_path.unlink()
                raise ValueError("Downloaded file is empty (0 bytes)")

            # Rename partial to final
            if local_path.exists():
                local_path.unlink()
            partial_path.rename(local_path)

            return {
                "scene_id": scene_id,
                "filename": filename,
                "success": True,
                "skipped": False,
                "size_bytes": final_size,
            }

        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as e:
            last_error = str(e)
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
    parser = argparse.ArgumentParser(
        description="Download 3RScan sequences + meshes for RIO10 test scenes"
    )
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help="Parallel download threads (default: 4)")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Limit number of scenes (for testing)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")
    parser.add_argument("--sequences-only", action="store_true",
                        help="Download only sequence.zip (skip mesh files)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = ["sequence.zip"] if args.sequences_only else FILES_TO_DOWNLOAD

    print("=" * 70)
    print("3RScan Test Sequence Downloader (RIO10 Test Scenes)")
    print("=" * 70)
    print(f"  Source:      {BASE_URL}")
    print(f"  Output:      {output_dir}")
    print(f"  Files/scene: {', '.join(files_to_download)}")
    print(f"  Workers:     {args.workers}")
    print()

    # Step 1: Load test scene IDs
    print("[1/3] Loading RIO10 test scene IDs...")
    scene_ids = load_test_scene_ids()
    print(f"  Found {len(scene_ids)} test scenes")

    if args.max_scenes:
        scene_ids = scene_ids[:args.max_scenes]
        print(f"  Limited to {args.max_scenes} scenes")

    # Step 2: Build download list (validate existing files)
    print(f"\n[2/3] Building download list for {len(scene_ids)} scenes...")
    download_tasks = []
    skipped_count = 0
    cleaned_count = 0
    for scene_id in scene_ids:
        for filename in files_to_download:
            local_path = output_dir / scene_id / filename
            partial_path = local_path.with_suffix(local_path.suffix + ".partial")

            if local_path.exists() and local_path.stat().st_size > 0:
                # Validate zip files
                if filename.endswith(".zip"):
                    try:
                        with open(str(local_path), "rb") as f:
                            magic = f.read(4)
                        if magic != b'PK\x03\x04' and magic != b'PK\x05\x06':
                            local_path.unlink()
                            cleaned_count += 1
                            download_tasks.append((scene_id, filename))
                            continue
                    except OSError:
                        pass
                skipped_count += 1
            else:
                # Clean up any orphaned partial files that are 0 bytes
                if partial_path.exists() and partial_path.stat().st_size == 0:
                    partial_path.unlink()
                download_tasks.append((scene_id, filename))

    total_files = len(scene_ids) * len(files_to_download)
    print(f"  Total files needed: {total_files}")
    print(f"  Already downloaded: {skipped_count}")
    if cleaned_count > 0:
        print(f"  Corrupt (cleaned): {cleaned_count}")
    print(f"  Files to download:  {len(download_tasks)}")
    print(f"  Estimated size:     ~46 GB (sequences) + ~1 GB (meshes)")

    if args.dry_run:
        print("\n[DRY RUN] Would download:")
        for scene_id, filename in download_tasks[:20]:
            print(f"  {BASE_URL}/{scene_id}/{filename}")
        if len(download_tasks) > 20:
            print(f"  ... and {len(download_tasks) - 20} more files")
        return

    if not download_tasks:
        print("\n  All files already downloaded!")
        _print_verification(output_dir, scene_ids, files_to_download)
        return

    # Step 3: Download
    print(f"\n[3/3] Downloading {len(download_tasks)} files with {args.workers} workers...")
    print("  (Ctrl+C to interrupt — resume support means you can restart anytime)")
    print()

    downloaded = 0
    failed_files = []
    total_bytes = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_file_with_resume, scene_id, filename, output_dir): (scene_id, filename)
            for scene_id, filename in download_tasks
        }

        try:
            for future in as_completed(futures):
                result = future.result()
                if result["success"]:
                    if not result["skipped"]:
                        downloaded += 1
                        total_bytes += result["size_bytes"]

                    if downloaded % 5 == 0 and downloaded > 0:
                        elapsed = time.time() - start_time
                        speed = total_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0
                        pct = downloaded / len(download_tasks) * 100
                        eta_s = (len(download_tasks) - downloaded) / (downloaded / elapsed) if downloaded > 0 else 0
                        eta_m = eta_s / 60
                        print(
                            f"  [{downloaded}/{len(download_tasks)}] ({pct:.1f}%) "
                            f"{total_bytes / 1024 / 1024 / 1024:.2f} GB @ {speed:.1f} MB/s "
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
            print("  You can resume by running this script again.")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Download complete in {elapsed / 60:.1f} minutes")
    print(f"  Downloaded: {downloaded} files ({total_bytes / 1024 / 1024 / 1024:.2f} GB)")
    if failed_files:
        print(f"  Failed:     {len(failed_files)} files")
        fail_log_path = output_dir / "download_failures.json"
        with open(fail_log_path, "w") as f:
            json.dump(failed_files, f, indent=2)
        print(f"  Failure log: {fail_log_path}")

    _print_verification(output_dir, scene_ids, files_to_download)


def _print_verification(output_dir: Path, scene_ids: list, files_to_download: list):
    """Verify downloads and print summary."""
    print(f"\n{'=' * 70}")
    print("VERIFICATION")
    print("=" * 70)

    complete_scenes = []
    partial_scenes = []
    missing_scenes = []
    total_size = 0

    for scene_id in scene_ids:
        scene_dir = output_dir / scene_id
        if not scene_dir.exists():
            missing_scenes.append(scene_id)
            continue

        scene_files = []
        for filename in files_to_download:
            fpath = scene_dir / filename
            if fpath.exists() and fpath.stat().st_size > 0:
                scene_files.append(filename)
                total_size += fpath.stat().st_size

        if len(scene_files) == len(files_to_download):
            complete_scenes.append(scene_id)
        elif len(scene_files) > 0:
            partial_scenes.append((scene_id, scene_files))
        else:
            missing_scenes.append(scene_id)

    print(f"\n  Complete scenes: {len(complete_scenes)} / {len(scene_ids)}")
    print(f"  Partial scenes:  {len(partial_scenes)}")
    print(f"  Missing scenes:  {len(missing_scenes)}")
    print(f"  Total size:      {total_size / 1024 / 1024 / 1024:.2f} GB")

    # Save report
    report = {
        "total_scenes": len(scene_ids),
        "complete": len(complete_scenes),
        "partial": len(partial_scenes),
        "missing": len(missing_scenes),
        "total_size_gb": round(total_size / 1024 / 1024 / 1024, 2),
    }
    report_path = output_dir / "download_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
