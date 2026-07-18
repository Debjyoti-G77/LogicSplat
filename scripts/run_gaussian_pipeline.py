"""
Run full Gaussian Splatting pipeline for multiple scenes.

For each scene:
1. COLMAP feature extraction
2. COLMAP sequential matching
3. COLMAP sparse reconstruction
4. ns-process-data (generate transforms.json)
5. ns-train splatfacto
6. ns-export gaussian-splat → splat.ply

Usage:
    python scripts/run_gaussian_pipeline.py --scenes scene_02 scene_03 scene_04 scene_05
"""
import subprocess
import os
import sys
import argparse

COLMAP_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "colmap.exe")
DATA_DIR = "D:/logicsplat_data/processed"
OUTPUTS_DIR = "outputs"


def run(cmd, desc=""):
    print(f"\n>>> {desc}")
    print(f"    {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: {desc} failed with code {result.returncode}")
        return False
    return True


def process_scene(scene_id: str, iterations: int = 10000):
    scene_path = os.path.join(DATA_DIR, scene_id)
    images_path = os.path.join(scene_path, "images")
    ns_data_path = os.path.join(scene_path, "ns_data")
    colmap_db = os.path.join(ns_data_path, "colmap", "database.db")
    colmap_sparse = os.path.join(ns_data_path, "colmap", "sparse")
    output_dir = os.path.join(OUTPUTS_DIR, f"{scene_id}_splat")
    splat_out = scene_path

    if not os.path.exists(images_path):
        print(f"SKIP {scene_id}: no images folder")
        return False

    # check if already done
    if os.path.exists(os.path.join(scene_path, "splat.ply")):
        print(f"SKIP {scene_id}: splat.ply already exists")
        return True

    os.makedirs(colmap_sparse, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing {scene_id}")
    print(f"{'='*60}")

    # step 1: copy images via ns-process-data (skip colmap)
    run(["ns-process-data", "images",
         "--data", images_path,
         "--output-dir", ns_data_path,
         "--skip-colmap"],
        "ns-process-data (copy images)")

    # step 2: COLMAP feature extraction
    if not run([COLMAP_BIN, "feature_extractor",
                "--database_path", colmap_db,
                "--image_path", os.path.join(ns_data_path, "images"),
                "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", "OPENCV"],
               "COLMAP feature extraction"):
        return False

    # step 3: COLMAP sequential matching
    if not run([COLMAP_BIN, "sequential_matcher",
                "--database_path", colmap_db,
                "--SequentialMatching.overlap", "10"],
               "COLMAP sequential matching"):
        return False

    # step 4: COLMAP sparse reconstruction
    if not run([COLMAP_BIN, "mapper",
                "--database_path", colmap_db,
                "--image_path", os.path.join(ns_data_path, "images"),
                "--output_path", colmap_sparse],
               "COLMAP sparse reconstruction"):
        return False

    # step 5: generate transforms.json
    if not run(["ns-process-data", "images",
                "--data", images_path,
                "--output-dir", ns_data_path,
                "--skip-colmap"],
               "Generate transforms.json"):
        return False

    # check transforms.json was created
    transforms_path = os.path.join(ns_data_path, "transforms.json")
    if not os.path.exists(transforms_path):
        print(f"ERROR: transforms.json not created for {scene_id}")
        return False

    # step 6: Gaussian Splatting training
    if not run(["ns-train", "splatfacto",
                "--data", ns_data_path,
                "--output-dir", output_dir,
                "--max-num-iterations", str(iterations)],
               "Gaussian Splatting training"):
        return False

    # find config.yml
    config_path = None
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f == "config.yml":
                config_path = os.path.join(root, f)
                break
        if config_path:
            break

    if not config_path:
        print(f"ERROR: config.yml not found for {scene_id}")
        return False

    # step 7: export splat.ply
    if not run(["ns-export", "gaussian-splat",
                "--load-config", config_path,
                "--output-dir", splat_out],
               "Export splat.ply"):
        return False

    print(f"\n✓ {scene_id} complete: {os.path.join(splat_out, 'splat.ply')}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+",
                        default=["scene_02", "scene_03", "scene_04", "scene_05"])
    parser.add_argument("--iterations", type=int, default=10000)
    args = parser.parse_args()

    results = {}
    for scene in args.scenes:
        success = process_scene(scene, args.iterations)
        results[scene] = "✓" if success else "✗"

    print("\n" + "="*40)
    print("Summary:")
    for scene, status in results.items():
        print(f"  {status} {scene}")
