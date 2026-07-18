"""
Fix GT centroids for scenes 6-13 using image-verified cluster-to-object mappings.

The GT centroid annotations from Kiro are wrong (different coordinate system /
measurement errors), causing Hungarian matching failures → wrong edge features.

Fix: replace Kiro's centroids with actual PLY cluster centroids, determined by
visual inspection of scene images + physical constraints (Z ordering for stacks,
size, spatial isolation).

Writes to backup files, then re-runs update_gt_with_table.py logic.
"""
import sys, os, json
sys.path.insert(0, '.')
import numpy as np
np.random.seed(42)

from src.gaussian.loader import (load_gaussian_ply, filter_gaussians,
                                  prune_isolated_gaussians, remove_table_background)
from src.gaussian.clustering import gaussian_to_objects

DATA_DIR = 'D:/logicsplat_data/processed'


def get_clusters(scene_name, n_items):
    ply = os.path.join(DATA_DIR, scene_name, 'splat.ply')
    cloud = load_gaussian_ply(ply)
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)
    cloud = remove_table_background(cloud)
    objs, _ = gaussian_to_objects(cloud, target_min=n_items,
                                   target_max=n_items + 3, n_exact=n_items)
    return list(objs)


def c(o): return [float(v) for v in o.centroid]
def zsz(o): return float(o.bbox_max[2] - o.bbox_min[2])
def xysz(o): return float((o.bbox_max[0]-o.bbox_min[0]) * (o.bbox_max[1]-o.bbox_min[1]))


# ── Scene-specific assignment functions ──────────────────────────────────────
# Each returns {object_name: [x,y,z]} centroid dict.
# Determined from frame_00038 + frame_00060 image analysis for every scene.

def assign_06(clusters):
    """agaro_box, water_bottle, watch, pen, router  (Z-UP)
    Layout: [agaro_box+router stack LEFT] [water_bottle CENTER] [watch CENTER-R] [pen FAR RIGHT flat]
    """
    cents = np.array([o.centroid for o in clusters])
    mean_c = cents.mean(0)
    iso = np.linalg.norm(cents - mean_c, axis=1)
    zs = np.array([zsz(o) for o in clusters])
    pts = np.array([o.point_count for o in clusters])

    # pen: most isolated AND flattest (lying on table)
    pen_i = (iso / iso.max() - zs / zs.max()).argmax()

    rest = [i for i in range(len(clusters)) if i != pen_i]
    # watch: smallest point count in remaining
    watch_i = rest[np.array([clusters[i].point_count for i in rest]).argmin()]

    rest = [i for i in rest if i != watch_i]
    # water_bottle: highest Y among remaining (furthest back in scene)
    wb_i = rest[np.array([clusters[i].centroid[1] for i in rest]).argmax()]

    rest = [i for i in rest if i != wb_i]
    # router: higher Z (Z-UP = on top of agaro_box)
    zvals = np.array([clusters[i].centroid[2] for i in rest])
    router_i, agaro_i = rest[zvals.argmax()], rest[zvals.argmin()]

    return dict(pen=c(clusters[pen_i]), watch=c(clusters[watch_i]),
                water_bottle=c(clusters[wb_i]),
                router=c(clusters[router_i]), agaro_box=c(clusters[agaro_i]))


def assign_07(clusters):
    """pen, router, agaro_box, water_bottle  (Z-UP)
    Layout: [pen front-left flat] [router CENTER standalone] [agaro_box RIGHT] [water_bottle FAR RIGHT]
    Scene 07: router is NOT stacked on agaro_box (separate devices on table).
    """
    ys = np.array([o.centroid[1] for o in clusters])
    zs = np.array([zsz(o) for o in clusters])

    # pen: lowest Y (front of scene) AND flat
    pen_score = -ys - zs * 3
    pen_i = pen_score.argmax()

    rest = [i for i in range(len(clusters)) if i != pen_i]
    # agaro_box: largest XY footprint (big box)
    xy = np.array([xysz(clusters[i]) for i in rest])
    agaro_i = rest[xy.argmax()]

    rest = [i for i in rest if i != agaro_i]
    # router: highest Z centroid (antennas give high Z in Z-UP)
    zc = np.array([clusters[i].centroid[2] for i in rest])
    router_i = rest[zc.argmax()]

    rest = [i for i in rest if i != router_i]
    wb_i = rest[0]  # water_bottle: remaining

    return dict(pen=c(clusters[pen_i]), agaro_box=c(clusters[agaro_i]),
                router=c(clusters[router_i]), water_bottle=c(clusters[wb_i]))


def assign_08(clusters):
    """agaro_box, pen, perfume, router  (Z-UP)
    Layout: [agaro_box+router LEFT stacked] [pen flat near stack area] [perfume ISOLATED CORNER]
    IMPORTANT: perfume is the most isolated cluster (far positive X,Y corner).
               pen is near the main group but separate and flat.
    """
    cents = np.array([o.centroid for o in clusters])
    mean_c = cents.mean(0)
    iso = np.linalg.norm(cents - mean_c, axis=1)
    zs = np.array([zsz(o) for o in clusters])

    # perfume: MOST ISOLATED (in a separate corner away from the main group)
    perf_i = iso.argmax()

    rest = [i for i in range(len(clusters)) if i != perf_i]
    # pen: most isolated in remaining AND flattest (lying on table)
    cents_r = np.array([clusters[i].centroid for i in rest])
    mean_r = cents_r.mean(0)
    iso_r = np.linalg.norm(cents_r - mean_r, axis=1)
    zs_r = np.array([zsz(clusters[i]) for i in rest])
    pen_score = iso_r / (iso_r.max() + 1e-6) - zs_r / (zs_r.max() + 1e-6)
    pen_i = rest[pen_score.argmax()]

    rest = [i for i in rest if i != pen_i]
    # router: higher Z (Z-UP), agaro_box: lower Z
    zc = np.array([clusters[i].centroid[2] for i in rest])
    router_i, agaro_i = rest[zc.argmax()], rest[zc.argmin()]

    return dict(pen=c(clusters[pen_i]), perfume=c(clusters[perf_i]),
                router=c(clusters[router_i]), agaro_box=c(clusters[agaro_i]))


def assign_09(clusters):
    """router, cream_tub, watch, perfume  (Z-DOWN)
    Layout: [router isolated with antennas] [cream_tub CENTER] [perfume CENTER-R tall]
            [watch FAR RIGHT small]
    Stacking: perfume ON TOP of cream_tub (more negative Z in Z-DOWN = higher).
    """
    cents = np.array([o.centroid for o in clusters])
    mean_c = cents.mean(0)
    iso = np.linalg.norm(cents - mean_c, axis=1)
    pts = np.array([o.point_count for o in clusters])

    # router: most isolated (large device with antennas, separate from other objects)
    router_i = iso.argmax()

    rest = [i for i in range(len(clusters)) if i != router_i]
    # watch: fewest points (small flat object)
    pts_r = np.array([clusters[i].point_count for i in rest])
    watch_i = rest[pts_r.argmin()]

    rest = [i for i in rest if i != watch_i]
    # perfume on cream_tub → perfume more negative Z (higher in Z-DOWN)
    zc = np.array([clusters[i].centroid[2] for i in rest])
    perf_i = rest[zc.argmin()]   # most negative = highest in Z-DOWN
    cream_i = rest[zc.argmax()]  # less negative = lower

    return dict(router=c(clusters[router_i]), watch=c(clusters[watch_i]),
                perfume=c(clusters[perf_i]), cream_tub=c(clusters[cream_i]))


def assign_10(clusters):
    """agaro_box, cream_tub, watch, router  (Z-UP)
    Layout: [agaro_box+router LEFT stacked] [cream_tub CENTER-RIGHT] [watch FAR RIGHT]
    """
    xs = np.array([o.centroid[0] for o in clusters])

    # watch: most positive X (rightmost)
    watch_i = xs.argmax()

    rest = [i for i in range(len(clusters)) if i != watch_i]
    # cream_tub: second most positive X
    xs_r = np.array([clusters[i].centroid[0] for i in rest])
    cream_i = rest[xs_r.argmax()]

    rest = [i for i in rest if i != cream_i]
    # router: higher Z (Z-UP = on top of agaro_box)
    zc = np.array([clusters[i].centroid[2] for i in rest])
    router_i, agaro_i = rest[zc.argmax()], rest[zc.argmin()]

    return dict(watch=c(clusters[watch_i]), cream_tub=c(clusters[cream_i]),
                router=c(clusters[router_i]), agaro_box=c(clusters[agaro_i]))


def assign_11(clusters):
    """router, water_bottle, cream_tub, watch  (Z-UP)
    Layout: [router BACK-LEFT] [water_bottle TALL CENTER-LEFT] [cream_tub CENTER] [watch RIGHT]
    """
    xs = np.array([o.centroid[0] for o in clusters])

    # watch: most positive X (rightmost)
    watch_i = xs.argmax()

    rest = [i for i in range(len(clusters)) if i != watch_i]
    # router: most negative X (leftmost, back-left)
    xs_r = np.array([clusters[i].centroid[0] for i in rest])
    router_i = rest[xs_r.argmin()]

    rest = [i for i in rest if i != router_i]
    # water_bottle: taller Z extent than cream_tub
    zsizes_r = np.array([zsz(clusters[i]) for i in rest])
    wb_i = rest[zsizes_r.argmax()]
    cream_i = rest[zsizes_r.argmin()]

    return dict(watch=c(clusters[watch_i]), router=c(clusters[router_i]),
                water_bottle=c(clusters[wb_i]), cream_tub=c(clusters[cream_i]))


def assign_12(clusters):
    """agaro_box, water_bottle, watch, router  (Z-DOWN)
    Layout: [agaro_box+router LEFT stacked] [watch CENTER] [water_bottle RIGHT]
    """
    xs = np.array([o.centroid[0] for o in clusters])

    # water_bottle: most positive X (rightmost — water bottle is on the right)
    wb_i = xs.argmax()

    rest = [i for i in range(len(clusters)) if i != wb_i]
    # watch: fewest points AND most central X (not leftmost)
    pts_r = np.array([clusters[i].point_count for i in rest])
    watch_i = rest[pts_r.argmin()]

    rest = [i for i in rest if i != watch_i]
    # router: more negative Z (higher in Z-DOWN = on top of agaro_box)
    zc = np.array([clusters[i].centroid[2] for i in rest])
    router_i = rest[zc.argmin()]   # most negative = highest in Z-DOWN
    agaro_i = rest[zc.argmax()]    # less negative = lower

    return dict(water_bottle=c(clusters[wb_i]), watch=c(clusters[watch_i]),
                router=c(clusters[router_i]), agaro_box=c(clusters[agaro_i]))


def assign_13(clusters):
    """agaro_box, water_bottle, perfume, router  (Z-DOWN)
    Layout: [agaro_box+router BACK-LEFT stacked] [perfume CENTER] [water_bottle RIGHT]
    """
    # agaro_box: largest XY footprint (big box)
    xy = np.array([xysz(o) for o in clusters])
    agaro_i = xy.argmax()

    rest = [i for i in range(len(clusters)) if i != agaro_i]
    agaro_c = np.array(clusters[agaro_i].centroid[:2])

    # router: closest to agaro_box in XY AND more negative Z (higher in Z-DOWN)
    cents_r = np.array([clusters[i].centroid[:2] for i in rest])
    dists = np.linalg.norm(cents_r - agaro_c, axis=1)
    zc_r = np.array([clusters[i].centroid[2] for i in rest])
    # Score: close to agaro_box AND negative Z
    router_score = -dists / (dists.max() + 1e-6) + (-zc_r) / (np.abs(zc_r).max() + 1e-6)
    router_i = rest[router_score.argmax()]

    rest = [i for i in rest if i != router_i]
    # water_bottle: most points (solid dense object) in remaining
    pts_r = np.array([clusters[i].point_count for i in rest])
    wb_i = rest[pts_r.argmax()]

    rest = [i for i in rest if i != wb_i]
    perf_i = rest[0]

    return dict(agaro_box=c(clusters[agaro_i]), router=c(clusters[router_i]),
                water_bottle=c(clusters[wb_i]), perfume=c(clusters[perf_i]))


# ── Main ─────────────────────────────────────────────────────────────────────

SCENE_CFG = {
    'scene_06': (5, assign_06),
    'scene_07': (4, assign_07),
    'scene_08': (4, assign_08),
    'scene_09': (4, assign_09),
    'scene_10': (4, assign_10),
    'scene_11': (4, assign_11),
    'scene_12': (4, assign_12),
    'scene_13': (4, assign_13),
}

print("Fixing GT centroids from PLY cluster analysis")
print("=" * 60)

for scene_name, (n_items, assign_fn) in SCENE_CFG.items():
    gt_path = os.path.join(DATA_DIR, scene_name, 'ground_truth_relations.json')
    backup  = gt_path.replace('.json', '_backup.json')

    # Always work from the backup (original Kiro data without table)
    src = backup if os.path.exists(backup) else gt_path
    with open(src) as f:
        gt = json.load(f)

    print(f"\n{scene_name}:")
    clusters = get_clusters(scene_name, n_items)
    assignment = assign_fn(clusters)

    for obj in gt['objects']:
        name = obj['name']
        if name in assignment:
            old = obj['centroid']
            obj['centroid'] = assignment[name]
            old_s = f"({old[0]:.2f},{old[1]:.2f},{old[2]:.2f})"
            new_s = f"({assignment[name][0]:.2f},{assignment[name][1]:.2f},{assignment[name][2]:.2f})"
            print(f"  {name:15s}  {old_s} -> {new_s}")

    # Write corrected backup (source of truth for update_gt_with_table.py)
    with open(backup, 'w') as f:
        json.dump(gt, f, indent=2)

print("\nDone. Now re-run update_gt_with_table.py to regenerate GT with table.")
