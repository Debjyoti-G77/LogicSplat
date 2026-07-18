"""Diagnostic: bbox sizes per scene after new preprocessing pipeline."""
import sys, warnings
sys.path.insert(0, ".")
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians, remove_table_background
from src.gaussian.clustering import gaussian_to_objects

DATA_DIR = "D:/logicsplat_data/processed"

# scenes 1-5 (adapt), 6-13 (test) — check both
SCENE_IDS = list(range(1, 14))
GT_COUNTS = {
    # scenes 1-5: no GT, use typical 4-5
    1: 5, 2: 4, 3: 3, 4: 5, 5: 5,
    # scenes 6-13: known GT
    6: 5, 7: 4, 8: 4, 9: 4, 10: 4, 11: 4, 12: 4, 13: 4,
}

print(f"{'Scene':<10} {'N_in':>7} {'N_bg_rm':>8} {'N_objs':>6} {'diags'}")
print("-" * 70)

for sid in SCENE_IDS:
    import os
    ply = os.path.join(DATA_DIR, f"scene_{sid:02d}", "splat.ply")
    if not os.path.exists(ply):
        print(f"scene_{sid:02d}   MISSING")
        continue

    n_hint = GT_COUNTS.get(sid, 4)

    cloud = load_gaussian_ply(ply)
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)
    n_before = cloud.num_gaussians

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cloud_clean = remove_table_background(cloud)
        bg_msg = ""
        for warning in w:
            if "remove_table_background" in str(warning.message):
                bg_msg = str(warning.message).split("Gaussians")[0].split("kept")[1].strip()

    n_after = cloud_clean.num_gaussians

    objects, _ = gaussian_to_objects(
        cloud_clean,
        target_min=n_hint,
        target_max=n_hint * 4,
        n_exact=n_hint,
    )

    if objects:
        diags = [round(float(np.linalg.norm(o.bbox_max - o.bbox_min)), 2) for o in objects]
        diag_str = str(diags)
    else:
        diag_str = "NO OBJECTS"

    print(f"scene_{sid:02d}   {n_before:>7,} {n_after:>8,}  {len(objects):>3}/{n_hint}  {diag_str}")
