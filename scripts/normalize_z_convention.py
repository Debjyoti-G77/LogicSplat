"""
Normalize Z-convention in tabletop Gaussian splat PLY files.

For 3DGS PLY files where Z is DOWN (physically higher = more negative Z),
apply a Z-flip so that all scenes use Z-UP convention matching 3RScan training data.

Z-flip transform:
  - Positions:  z <- -z
  - Normals:    nz <- -nz
  - Quaternions (w,x,y,z): negate x and y:  rot_1 <- -rot_1, rot_2 <- -rot_2
  - Scales:     unchanged
  - SH colors:  unchanged (geometry-only fix; color appearance unaffected for clustering)

Also updates the GT centroid Z values in ground_truth_relations.json.

Usage:
    python scripts/normalize_z_convention.py --scenes scene_06 scene_07
    python scripts/normalize_z_convention.py --scenes scene_06 scene_07 --dry-run
"""
import sys
sys.path.insert(0, ".")

import os
import json
import shutil
import argparse
import numpy as np
from plyfile import PlyData, PlyElement


DATA_DIR = "D:/logicsplat_data/processed"


def flip_z_ply(src_path: str, dst_path: str) -> dict:
    """Load PLY, apply Z-flip, save to dst_path. Returns stats."""
    import os

    # Read PLY — use backup if src==dst (backup already created before this call)
    read_path = src_path
    bak_path = src_path + ".zdown_backup"
    if src_path == dst_path and os.path.exists(bak_path):
        read_path = bak_path  # read from backup so src file is not locked

    ply = PlyData.read(read_path)
    v = ply["vertex"]
    props = [p.name for p in v.properties]
    dtypes = [(p.name, p.val_dtype) for p in v.properties]
    n = len(v)
    is_text = ply.text

    # Eager copy into plain numpy arrays, then release plyfile object
    data = {name: np.array(v[name], copy=True) for name in props}
    z_before = (float(data["z"].min()), float(data["z"].max()))
    del ply, v  # release file handle

    # 1. Negate Z positions
    data["z"] = -data["z"]

    # 2. Negate Z normals
    if "nz" in data:
        data["nz"] = -data["nz"]

    # 3. Fix quaternion for Z-flip: (w, x, y, z) -> (w, -x, -y, z)
    data["rot_1"] = -data["rot_1"]
    data["rot_2"] = -data["rot_2"]

    # Rebuild structured array
    arr = np.empty(n, dtype=dtypes)
    for name in props:
        arr[name] = data[name]

    new_el = PlyElement.describe(arr, "vertex")

    # Write to temp then rename (same drive)
    tmp_path = dst_path + ".tmp_zflip"
    PlyData([new_el], text=is_text).write(tmp_path)
    if os.path.exists(dst_path):
        os.replace(dst_path, dst_path + ".old_zflip")  # soft-rename original
    os.rename(tmp_path, dst_path)
    if os.path.exists(dst_path + ".old_zflip"):
        os.remove(dst_path + ".old_zflip")

    return {
        "n_gaussians": n,
        "z_range_before": z_before,
        "z_range_after":  (float(data["z"].min()), float(data["z"].max())),
    }


def flip_z_gt(src_path: str, dst_path: str):
    """Negate Z component of all object centroids in ground_truth_relations.json."""
    with open(src_path) as f:
        gt = json.load(f)

    for obj in gt["objects"]:
        c = obj["centroid"]
        obj["centroid"] = [c[0], c[1], -c[2]]

    with open(dst_path, "w") as f:
        json.dump(gt, f, indent=2)


def verify_z_up(scene_dir: str) -> bool:
    """
    Quick sanity check: after fix, all higher_than GT pairs should have
    subject_z > object_z (Z-UP convention).
    """
    gt_path = os.path.join(scene_dir, "ground_truth_relations.json")
    with open(gt_path) as f:
        gt = json.load(f)

    name_to_z = {o["name"]: o["centroid"][2] for o in gt["objects"]}

    ht_rels = [r for r in gt["relations"] if r["relation"] == "higher_than"]
    ok, fail = 0, 0
    for r in ht_rels:
        sz = name_to_z.get(r["subject"])
        oz = name_to_z.get(r["object"])
        if sz is None or oz is None:
            continue
        if sz > oz:
            ok += 1
        else:
            fail += 1

    return fail == 0, ok, fail


def main():
    parser = argparse.ArgumentParser(description="Normalize tabletop scene Z-convention to Z-UP")
    parser.add_argument("--scenes", nargs="+", required=True,
                        help="Scene names to normalize, e.g. scene_06 scene_07")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without modifying files")
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    for scene_name in args.scenes:
        scene_dir = os.path.join(args.data_dir, scene_name)
        ply_path = os.path.join(scene_dir, "splat.ply")
        gt_path = os.path.join(scene_dir, "ground_truth_relations.json")

        if not os.path.exists(ply_path):
            print(f"SKIP {scene_name}: splat.ply not found")
            continue

        print(f"\n{'='*55}")
        print(f"  {scene_name}")
        print(f"{'='*55}")

        # Verify it IS Z-DOWN before flipping
        with open(gt_path) as f:
            gt = json.load(f)
        name_to_z = {o["name"]: o["centroid"][2] for o in gt["objects"]}
        ht_rels = [r for r in gt["relations"] if r["relation"] == "higher_than"]
        inverted = sum(1 for r in ht_rels
                       if name_to_z.get(r["subject"], 0) < name_to_z.get(r["object"], 0))
        normal   = sum(1 for r in ht_rels
                       if name_to_z.get(r["subject"], 0) > name_to_z.get(r["object"], 0))
        convention = "Z-DOWN" if inverted > normal else "Z-UP"
        print(f"  Current convention: {convention}  (inverted={inverted}, normal={normal})")

        if convention == "Z-UP":
            print(f"  Already Z-UP — skipping")
            continue

        if args.dry_run:
            print(f"  [DRY RUN] Would apply Z-flip to splat.ply and ground_truth_relations.json")
            continue

        # Backup originals
        bak_ply = ply_path + ".zdown_backup"
        bak_gt  = gt_path  + ".zdown_backup"
        if not os.path.exists(bak_ply):
            shutil.copy2(ply_path, bak_ply)
            print(f"  Backed up splat.ply -> splat.ply.zdown_backup")
        if not os.path.exists(bak_gt):
            shutil.copy2(gt_path, bak_gt)
            print(f"  Backed up ground_truth_relations.json -> .zdown_backup")

        # Apply Z-flip to PLY
        stats = flip_z_ply(ply_path, ply_path)
        print(f"  PLY Z-flip done: {stats['n_gaussians']:,} Gaussians")
        print(f"    Z before: [{stats['z_range_before'][0]:.3f}, {stats['z_range_before'][1]:.3f}]")
        print(f"    Z after:  [{stats['z_range_after'][0]:.3f}, {stats['z_range_after'][1]:.3f}]")

        # Apply Z-flip to GT
        flip_z_gt(gt_path, gt_path)
        print(f"  GT Z-flip done")

        # Verify
        is_up, ok, fail = verify_z_up(scene_dir)
        print(f"  Verification: {ok} higher_than pairs correct, {fail} wrong  -> {'OK' if is_up else 'FAIL'}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
