"""
Compute tabletop edge-feature normalization statistics v6.

Uses VIRTUAL TABLE (no RANSAC) and correct per-scene Z-flip:
  - Scenes 6-13: hardcoded Z-convention, virtual table, correct delta_z signs
  - Scenes 1-5:  no table (too uncertain to determine Z-convention reliably)

Virtual table is position-computed the same way as create_virtual_table() in
eval_geokan_tabletop.py so normalization stats match inference-time features.

Saves: models/tabletop_feat_mean_v6.npy, models/tabletop_feat_std_v6.npy
"""
import sys, os, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np

from src.gaussian.loader import (
    load_gaussian_ply, filter_gaussians,
    prune_isolated_gaussians, remove_table_background,
)
from src.gaussian.clustering import gaussian_to_objects
from scripts.build_3rscan_graphs import extract_3rscan_edge_features
from src.graph.definitions import Object3D

DATA_DIR  = "D:/logicsplat_data/processed"
OUT_DIR   = "models"
N_HINT    = 4   # number of item clusters per scene

# Scenes 1-5: items only (no table, Z-convention uncertain)
SCENES_ITEMS_ONLY = [f"scene_{i:02d}" for i in range(1, 6)]

# Scenes 6-13: with virtual table, correct Z-convention
SCENES_WITH_TABLE = [f"scene_{i:02d}" for i in range(6, 14)]

SCENE_Z_FLIP = {
    "scene_06": False,
    "scene_07": False,
    "scene_08": False,
    "scene_09": True,
    "scene_10": False,
    "scene_11": False,
    "scene_12": True,
    "scene_13": True,
}


def apply_z_flip(objects):
    for o in objects:
        o.centroid = o.centroid.copy(); o.centroid[2] = -o.centroid[2]
        old_min_z, old_max_z = o.bbox_min[2], o.bbox_max[2]
        o.bbox_min = o.bbox_min.copy(); o.bbox_max = o.bbox_max.copy()
        o.bbox_min[2] = -old_max_z; o.bbox_max[2] = -old_min_z


def create_virtual_table(objects, is_z_down, table_height_scale=5.0):
    items       = [o for o in objects if o.label != "table"]
    if not items:
        return None
    centroid_zs = [float(o.centroid[2]) for o in items]
    sizes_z     = [abs(float(o.bbox_max[2]) - float(o.bbox_min[2])) for o in items]
    median_sz   = float(np.median(sizes_z)) if sizes_z else 1.0
    offset_z    = median_sz * 0.3
    table_ht    = table_height_scale * median_sz

    if is_z_down:
        surface_z  = max(centroid_zs) + offset_z
        bbox_min_z = surface_z
        bbox_max_z = surface_z + table_ht
    else:
        surface_z  = min(centroid_zs) - offset_z
        bbox_max_z = surface_z
        bbox_min_z = surface_z - table_ht

    centroid_z = (bbox_min_z + bbox_max_z) / 2
    all_xmin = float(min(o.bbox_min[0] for o in items))
    all_xmax = float(max(o.bbox_max[0] for o in items))
    all_ymin = float(min(o.bbox_min[1] for o in items))
    all_ymax = float(max(o.bbox_max[1] for o in items))
    mx = (all_xmax - all_xmin) * 0.15
    my = (all_ymax - all_ymin) * 0.15
    centroid = np.array([(all_xmin+all_xmax)/2, (all_ymin+all_ymax)/2, centroid_z], np.float32)
    bbox_min = np.array([all_xmin-mx, all_ymin-my, bbox_min_z], np.float32)
    bbox_max = np.array([all_xmax+mx, all_ymax+my, bbox_max_z], np.float32)
    return Object3D(
        uid=len(items), centroid=centroid, bbox_min=bbox_min, bbox_max=bbox_max,
        color=np.array([200, 200, 200], np.float32),
        point_count=sum(o.point_count for o in items), label="table",
    )


def build_edge_features(objects):
    n = len(objects)
    if n < 2:
        return None
    all_mins = np.stack([o.bbox_min for o in objects])
    all_maxs = np.stack([o.bbox_max for o in objects])
    scene_extent = np.maximum(all_maxs.max(0) - all_mins.min(0), 1e-6)
    feats = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            feats.append(extract_3rscan_edge_features(objects[i], objects[j], scene_extent))
    return np.stack(feats, axis=0)


def load_and_cluster(ply_path, n_items):
    cloud = load_gaussian_ply(ply_path)
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)
    cloud = remove_table_background(cloud)
    objects, _ = gaussian_to_objects(cloud, target_min=n_items,
                                     target_max=n_items + 3, n_exact=n_items)
    return list(objects)


print("Computing tabletop edge feature statistics v6 (virtual table, correct Z-flip)")
print("=" * 75)

all_feats = []

# ── Scenes 1-5: items only (no table) ────────────────────────────────────────
print("\n[Scenes 1-5: items only, no table]")
for scene_name in SCENES_ITEMS_ONLY:
    ply_path = os.path.join(DATA_DIR, scene_name, "splat.ply")
    if not os.path.exists(ply_path):
        print(f"  {scene_name}: MISSING - skipped")
        continue
    try:
        objects = load_and_cluster(ply_path, N_HINT)
    except Exception as e:
        print(f"  {scene_name}: clustering failed: {e}")
        continue
    if len(objects) < 2:
        print(f"  {scene_name}: < 2 objects, skipped")
        continue
    feats = build_edge_features(objects)
    if feats is None:
        continue
    all_feats.append(feats)
    print(f"  {scene_name}: {len(objects)} objects, {len(feats)} edges")

# ── Scenes 6-13: with virtual table, correct Z-flip ──────────────────────────
print("\n[Scenes 6-13: with virtual table, correct Z-flip]")
for scene_name in SCENES_WITH_TABLE:
    ply_path = os.path.join(DATA_DIR, scene_name, "splat.ply")
    if not os.path.exists(ply_path):
        print(f"  {scene_name}: MISSING - skipped")
        continue
    is_z_down = SCENE_Z_FLIP.get(scene_name, False)
    try:
        objects = load_and_cluster(ply_path, N_HINT)
    except Exception as e:
        print(f"  {scene_name}: clustering failed: {e}")
        continue

    # Create virtual table in original Z
    table_obj = create_virtual_table(objects, is_z_down=is_z_down)
    if table_obj is not None:
        table_obj.uid = len(objects)
        objects = objects + [table_obj]

    # Apply Z-flip after creating virtual table (matching eval pipeline)
    if is_z_down:
        apply_z_flip(objects)

    if len(objects) < 2:
        continue
    feats = build_edge_features(objects)
    if feats is None:
        continue
    all_feats.append(feats)
    tbl = "yes" if table_obj is not None else "no"
    flip = " flip" if is_z_down else ""
    print(f"  {scene_name}: {len(objects)} objs (table={tbl}{flip}), {len(feats)} edges")

if not all_feats:
    print("ERROR: no features collected")
    sys.exit(1)

all_feats_np = np.concatenate(all_feats, axis=0)
print(f"\nTotal edges: {len(all_feats_np)}, feat_dim={all_feats_np.shape[1]}")

tt_mean = all_feats_np.mean(0).astype(np.float32)
tt_std  = all_feats_np.std(0).clip(min=1e-6).astype(np.float32)

out_mean = os.path.join(OUT_DIR, "tabletop_feat_mean_v6.npy")
out_std  = os.path.join(OUT_DIR, "tabletop_feat_std_v6.npy")
np.save(out_mean, tt_mean)
np.save(out_std,  tt_std)
print(f"Saved: {out_mean}")
print(f"Saved: {out_std}")

# Compare v2 vs v6
v2m = np.load(os.path.join(OUT_DIR, "tabletop_feat_mean_v2.npy"))
v2s = np.load(os.path.join(OUT_DIR, "tabletop_feat_std_v2.npy"))
rsm = np.load(os.path.join(OUT_DIR, "rscan_feat_mean.npy")).astype(np.float32)
rss = np.load(os.path.join(OUT_DIR, "rscan_feat_std.npy")).astype(np.float32)

print(f"\n{'dim':>4}  {'v2_mean':>9}  {'v6_mean':>9}  {'v2_std':>9}  {'v6_std':>9}  {'rscan_std':>9}")
print("-" * 65)
for i in range(len(tt_mean)):
    d = abs(tt_mean[i] - v2m[i])
    m = " *" if d > 0.05 else ""
    print(f"  {i:>2}  {v2m[i]:>9.4f}  {tt_mean[i]:>9.4f}  "
          f"{v2s[i]:>9.4f}  {tt_std[i]:>9.4f}  {rss[i]:>9.4f}{m}")
print("\nDone.")
