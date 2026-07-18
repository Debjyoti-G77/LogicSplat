"""
Compute tabletop edge-feature normalization statistics v5.

Includes the TABLE node in every scene graph:
  - RANSAC extracts the table as a real cluster
  - Z-flip correction applied per scene (table = physically lowest)
  - Table bbox extended to TABLE_STANDARD_HEIGHT=0.72m (match 3RScan furniture)
  - Statistics computed over ALL directed edges (object-object AND object-table)

Uses all 13 scenes; no GT labels required.
Saves: models/tabletop_feat_mean_v5.npy, models/tabletop_feat_std_v5.npy
"""
import sys, os, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np

from src.gaussian.loader import (
    load_gaussian_ply, filter_gaussians,
    prune_isolated_gaussians, GaussianCloud,
)
from src.gaussian.clustering import gaussian_to_objects
from scripts.build_3rscan_graphs import extract_3rscan_edge_features
from src.graph.definitions import Object3D

DATA_DIR = "D:/logicsplat_data/processed"
SCENES   = [f"scene_{i:02d}" for i in range(1, 14)]
OUT_DIR  = "models"
TABLE_STANDARD_HEIGHT = 0.72
N_HINT = 4


# ── re-use helpers from eval script ──────────────────────────────────────────

def _filter_cloud_bool(cloud, mask):
    return GaussianCloud(
        xyz=cloud.xyz[mask], rgb=cloud.rgb[mask], opacity=cloud.opacity[mask],
        scales=cloud.scales[mask], rotations=cloud.rotations[mask],
        covariance=cloud.covariance[mask],
    )


def _make_table_obj(table_cloud, uid=9999):
    xyz = table_cloud.xyz
    return Object3D(
        uid=uid,
        centroid=xyz.mean(axis=0),
        bbox_min=xyz.min(axis=0),
        bbox_max=xyz.max(axis=0),
        color=table_cloud.rgb.mean(axis=0).astype(np.float32)
              if len(table_cloud.rgb) > 0 else np.array([200, 200, 200], dtype=np.float32),
        point_count=len(xyz),
        label="table",
    )


def extract_table_object(cloud):
    try:
        import open3d as o3d
    except ImportError:
        return None, cloud

    xyz = cloud.xyz
    N = len(xyz)
    if N < 100:
        return None, cloud

    from src.gaussian.loader import remove_table_background
    bbox_diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
    ransac_dist = max(bbox_diag * 0.01, 1e-4)

    def ransac_classify(xyz_arr):
        n = len(xyz_arr)
        n_s = min(n, 50_000)
        rng = np.random.default_rng(42)
        idx = rng.choice(n, n_s, replace=False) if n > n_s else np.arange(n)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_arr[idx].astype(np.float64))
        plane_eq, _ = pcd.segment_plane(
            distance_threshold=float(ransac_dist),
            ransac_n=3, num_iterations=1000,
        )
        a, b, c, d = [float(v) for v in plane_eq]
        nmag = float(np.sqrt(a*a + b*b + c*c))
        if nmag < 1e-6:
            return plane_eq, np.zeros(n), np.zeros(n, dtype=bool), 0.0
        signed = (xyz_arr @ np.array([a, b, c], dtype=np.float32)) / nmag + d / nmag
        inl = np.abs(signed) < ransac_dist
        return plane_eq, signed, inl, float(inl.mean())

    def above_plane_cloud(cloud_in, plane_eq, sdists,
                          above_quantile=0.97, margin_frac=0.05):
        pos_h = sdists[sdists > 0]
        neg_h = -sdists[sdists < 0]
        pq = float(np.quantile(pos_h, 0.95)) if len(pos_h) >= 10 else np.inf
        nq = float(np.quantile(neg_h, 0.95)) if len(neg_h) >= 10 else np.inf
        heights = sdists if pq <= nq else -sdists
        positive = heights[heights > 0]
        if len(positive) < 50:
            return cloud_in
        cutoff = float(np.quantile(positive, above_quantile))
        margin  = cutoff * margin_frac
        mask = (heights >= -margin) & (heights <= cutoff)
        return _filter_cloud_bool(cloud_in, mask) if mask.sum() >= 50 else cloud_in

    try:
        plane_eq1, sd1, inl1, frac1 = ransac_classify(xyz)
    except Exception:
        return None, remove_table_background(cloud)

    if frac1 > 0.75:
        non_floor = _filter_cloud_bool(cloud, ~inl1)
        if len(non_floor.xyz) < 200:
            return None, remove_table_background(cloud)
        try:
            plane_eq2, sd2, inl2, frac2 = ransac_classify(non_floor.xyz)
        except Exception:
            return None, remove_table_background(cloud)
        if 0.05 <= frac2 <= 0.65:
            return _make_table_obj(_filter_cloud_bool(non_floor, inl2)), \
                   above_plane_cloud(non_floor, plane_eq2, sd2)
        return None, remove_table_background(cloud)
    elif frac1 < 0.05:
        return None, remove_table_background(cloud)
    else:
        return _make_table_obj(_filter_cloud_bool(cloud, inl1)), \
               above_plane_cloud(cloud, plane_eq1, sd1)


def detect_z_flip_from_table(objects):
    table_z = None
    item_zs = []
    for o in objects:
        if o.label == "table":
            table_z = float(o.centroid[2])
        else:
            item_zs.append(float(o.centroid[2]))
    if table_z is None or not item_zs:
        return False
    return sum(1 for z in item_zs if z < table_z) > len(item_zs) / 2


def apply_z_flip(objects):
    for o in objects:
        o.centroid = o.centroid.copy()
        o.centroid[2] = -o.centroid[2]
        old_min_z, old_max_z = o.bbox_min[2], o.bbox_max[2]
        o.bbox_min = o.bbox_min.copy(); o.bbox_max = o.bbox_max.copy()
        o.bbox_min[2] = -old_max_z; o.bbox_max[2] = -old_min_z


def extend_table_height(objects, table_height=TABLE_STANDARD_HEIGHT):
    for o in objects:
        if o.label == "table":
            surf_z = float(o.bbox_max[2])
            o.bbox_min = o.bbox_min.copy(); o.bbox_min[2] = surf_z - table_height
            o.centroid = o.centroid.copy(); o.centroid[2] = surf_z - table_height / 2
            break


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


# ── Main ─────────────────────────────────────────────────────────────────────

print("Computing tabletop edge feature statistics v5 (table-inclusive, Z-flip, height-ext)")
print("=" * 75)

all_feats = []

for scene_name in SCENES:
    ply_path = os.path.join(DATA_DIR, scene_name, "splat.ply")
    if not os.path.exists(ply_path):
        print(f"  {scene_name}: MISSING — skipped")
        continue

    cloud = load_gaussian_ply(ply_path)
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)

    table_obj, obj_cloud = extract_table_object(cloud)
    has_table = table_obj is not None
    n_items = N_HINT

    objects, _ = gaussian_to_objects(
        obj_cloud,
        target_min=n_items, target_max=n_items + 3, n_exact=n_items,
    )

    if has_table:
        table_obj.uid = len(objects)
        objects = list(objects) + [table_obj]

    if len(objects) < 2:
        print(f"  {scene_name}: only {len(objects)} objects, skipped")
        continue

    # Z-flip + table height extension (same as eval pipeline)
    if has_table:
        if detect_z_flip_from_table(objects):
            apply_z_flip(objects)
        extend_table_height(objects)

    feats = build_edge_features(objects)
    if feats is None:
        continue

    all_feats.append(feats)
    tbl_str = f"+ table {'(flip)' if has_table and detect_z_flip_from_table(objects) else ''}" if has_table else "(no table)"
    print(f"  {scene_name}: {len(objects)} objects {tbl_str}, {len(feats)} edges, dim={feats.shape[1]}")

if not all_feats:
    print("ERROR: no features collected"); sys.exit(1)

all_feats_np = np.concatenate(all_feats, axis=0)
print(f"\nTotal edges: {len(all_feats_np)}, feat_dim={all_feats_np.shape[1]}")

tt_mean = all_feats_np.mean(0).astype(np.float32)
tt_std  = all_feats_np.std(0).clip(min=1e-6).astype(np.float32)

out_mean = os.path.join(OUT_DIR, "tabletop_feat_mean_v5.npy")
out_std  = os.path.join(OUT_DIR, "tabletop_feat_std_v5.npy")
np.save(out_mean, tt_mean)
np.save(out_std,  tt_std)
print(f"Saved: {out_mean}")
print(f"Saved: {out_std}")

# Compare with v2
v2m = np.load(os.path.join(OUT_DIR, "tabletop_feat_mean_v2.npy"))
v2s = np.load(os.path.join(OUT_DIR, "tabletop_feat_std_v2.npy"))
rsm = np.load(os.path.join(OUT_DIR, "rscan_feat_mean.npy")).astype(np.float32)
rss = np.load(os.path.join(OUT_DIR, "rscan_feat_std.npy")).astype(np.float32)
print(f"\n{'dim':>4}  {'v2_mean':>9}  {'v5_mean':>9}  {'v2_std':>9}  {'v5_std':>9}  {'rscan_std':>9}")
print("-" * 65)
for i in range(len(tt_mean)):
    delta = abs(tt_mean[i] - v2m[i])
    m = " *" if delta > 0.05 else ""
    print(f"  {i:>2}  {v2m[i]:>9.4f}  {tt_mean[i]:>9.4f}  {v2s[i]:>9.4f}  {tt_std[i]:>9.4f}  {rss[i]:>9.4f}{m}")
print("\nDone.")
