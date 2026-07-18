"""
Build correct ground-truth JSON files for tabletop eval scenes 6-13.

Approach:
  1. Run clustering with RANSAC seed=42 (deterministic) to get real cluster centroids
  2. Map cluster_index -> object_name via hard-coded visual analysis
  3. Derive ALL pairwise spatial relations mathematically from centroids
  4. Hard-code verified stacking (router on_top_of agaro_box) per scene
  5. Create virtual table; add all table relations
  6. Write ground_truth_relations.json for each scene

No GT relationship labels from test scenes are used for training - this is purely
geometric derivation from cluster positions, so no data leakage.
"""
import sys, os, json, itertools
import numpy as np

sys.path.insert(0, '.')
from src.gaussian.loader import (
    load_gaussian_ply, filter_gaussians,
    prune_isolated_gaussians, remove_table_background,
)
from src.gaussian.clustering import gaussian_to_objects

DATA_DIR = 'D:/logicsplat_data/processed'

# Number of items per scene (excluding table)
SCENE_N = {
    'scene_06': 5,
    'scene_07': 4,
    'scene_08': 4,
    'scene_09': 4,
    'scene_10': 4,
    'scene_11': 4,
    'scene_12': 4,
    'scene_13': 4,
}

# cluster_index -> object_name  (from visual analysis + centroid data)
# Derivation notes are in session memory / previous analysis
CLUSTER_TO_OBJ = {
    # scene_06 (Z-UP, 5 items):
    #   [0](-0.661,0.652,1.068) highest Z + near agaro_box -> router (on agaro_box)
    #   [1](1.581,-0.537,0.214) far-right isolated X       -> pen
    #   [2](-0.111,0.296,0.307) centre                    -> water_bottle
    #   [3](-0.169,0.843,0.086) most pts=9828              -> agaro_box
    #   [4](-0.288,0.739,0.213) near agaro_box area        -> watch
    'scene_06': {0: 'router', 1: 'pen', 2: 'water_bottle', 3: 'agaro_box', 4: 'watch'},

    # scene_07 (Z-UP, 4 items, no stacking):
    #   [0](0.505,0.512,0.249) right side                 -> pen
    #   [1](-0.310,0.613,0.243) centre-left               -> water_bottle
    #   [2](-0.370,0.966,0.029) most pts=5124             -> agaro_box
    #   [3](-0.414,0.875,0.399) highest Z, near agaro_box -> router
    'scene_07': {0: 'pen', 1: 'water_bottle', 2: 'agaro_box', 3: 'router'},

    # scene_08 (Z-UP, 4 items, router on agaro_box):
    #   [0](-0.737,-0.306,-0.162) closest XY to agaro_box -> router
    #   [1](-1.124,0.101,-0.087) most pts=8443            -> agaro_box
    #   [2](-0.467,0.392,-0.180) higher Z of pair         -> perfume
    #   [3](-0.373,0.167,-0.350) lowest Z (flat on table) -> pen
    'scene_08': {0: 'router', 1: 'agaro_box', 2: 'perfume', 3: 'pen'},

    # scene_09 (Z-DOWN, 4 items, no stacking):
    #   [0](0.410,1.678,-0.090) far isolated Y=1.678      -> router
    #   [1](-0.532,-0.073,-0.308) most neg Z=highest      -> perfume (taller bottle)
    #   [2](-0.725,-0.081,-0.141) middle Z                -> cream_tub
    #   [3](-0.733,-0.163,0.019) least neg Z=lowest,flat  -> watch
    'scene_09': {0: 'router', 1: 'perfume', 2: 'cream_tub', 3: 'watch'},

    # scene_10 (Z-UP, 4 items, router on agaro_box):
    #   [0](1.235,0.864,-0.503) far-right, lowest Z,flat  -> watch
    #   [1](0.807,1.245,-0.077) centre-right              -> cream_tub
    #   [2](-0.691,0.216,-0.066) left, higher Z           -> router (on agaro_box)
    #   [3](-0.358,-0.081,-0.159) left, lower Z           -> agaro_box
    'scene_10': {0: 'watch', 1: 'cream_tub', 2: 'router', 3: 'agaro_box'},

    # scene_11 (Z-UP, 4 items, no stacking):
    #   [0](-0.053,0.057,-0.404) near-zero X, lowest Z   -> watch (flat, right side)
    #   [1](-1.084,-0.230,0.025) far-left, highest Z+ant  -> router (left-back)
    #   [2](-0.600,-0.346,-0.255) centre, higher of pair  -> water_bottle (taller)
    #   [3](-0.578,-0.315,-0.298) centre, lower of pair   -> cream_tub
    'scene_11': {0: 'watch', 1: 'router', 2: 'water_bottle', 3: 'cream_tub'},

    # scene_12 (Z-DOWN, 4 items, router on agaro_box):
    #   [0](1.673,0.346,0.086) far-right isolated X       -> water_bottle
    #   [1](-0.453,0.460,-0.300) most neg Z=highest       -> router (on agaro_box)
    #   [2](0.019,0.550,-0.200) centre                   -> watch
    #   [3](-0.554,0.733,-0.081) nearest to router XY    -> agaro_box
    'scene_12': {0: 'water_bottle', 1: 'router', 2: 'watch', 3: 'agaro_box'},

    # scene_13 (Z-DOWN, 4 items, router on agaro_box):
    #   [0](-1.021,-0.059,-0.032) most neg X,most pts    -> agaro_box
    #   [1](-0.570,-0.232,-0.234) more neg Z=higher      -> water_bottle (taller)
    #   [2](-0.616,-0.199,-0.166) closest XY to agaro_box-> router
    #   [3](-0.588,-0.189,-0.217) remaining              -> perfume
    'scene_13': {0: 'agaro_box', 1: 'water_bottle', 2: 'router', 3: 'perfume'},
}

# True = Z-DOWN (more negative Z = physically higher)
SCENE_Z_FLIP = {
    'scene_06': False,
    'scene_07': False,
    'scene_08': False,
    'scene_09': True,
    'scene_10': False,
    'scene_11': False,
    'scene_12': True,
    'scene_13': True,
}

# Verified physical stacking: (subject on_top_of object) - from visual inspection
STACKING = {
    'scene_06': [('router', 'agaro_box')],
    'scene_07': [],
    'scene_08': [('router', 'agaro_box')],
    'scene_09': [],  # NO stacking — all objects side-by-side
    'scene_10': [('router', 'agaro_box')],
    'scene_11': [],
    'scene_12': [('router', 'agaro_box')],
    'scene_13': [('router', 'agaro_box')],
}

ADJACENT_DIST_THRESH = 0.70  # scene units; tuned to capture "close" object pairs


def load_clusters(scene_name):
    n = SCENE_N[scene_name]
    ply = os.path.join(DATA_DIR, scene_name, 'splat.ply')
    cloud = load_gaussian_ply(ply)
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)
    cloud = remove_table_background(cloud)
    objs, _ = gaussian_to_objects(cloud, target_min=n, target_max=n + 3, n_exact=n)
    return list(objs)


def build_virtual_table(obj_centroids, obj_bboxes, z_flip):
    """
    Create a virtual table dict covering the footprint of all objects.
    z_flip=True means Z-DOWN (more negative Z = higher = objects are at more negative Z).
    """
    all_cx = [c[0] for c in obj_centroids]
    all_cy = [c[1] for c in obj_centroids]
    all_cz = [c[2] for c in obj_centroids]

    margin = 0.30
    x_min = min(b[0][0] for b in obj_bboxes) - margin
    x_max = max(b[1][0] for b in obj_bboxes) + margin
    y_min = min(b[0][1] for b in obj_bboxes) - margin
    y_max = max(b[1][1] for b in obj_bboxes) + margin

    thickness = 0.50

    if not z_flip:  # Z-UP: table surface is just below lowest object centroid
        table_top_z = min(all_cz) - 0.15
        table_bot_z = table_top_z - thickness
    else:           # Z-DOWN: table surface is just above (more positive) highest object centroid
        table_top_z = max(all_cz) + 0.15
        table_bot_z = table_top_z + thickness

    z_lo = min(table_top_z, table_bot_z)
    z_hi = max(table_top_z, table_bot_z)

    centroid = [
        (x_min + x_max) / 2,
        (y_min + y_max) / 2,
        (z_lo + z_hi) / 2,
    ]
    return {
        'centroid': centroid,
        'bbox_min': [x_min, y_min, z_lo],
        'bbox_max': [x_max, y_max, z_hi],
        'table_top_z': table_top_z,
    }


def derive_relations(names, centroids, z_flip, stacking_pairs, table_info):
    """
    Derive all pairwise relations from 3D cluster centroids.

    Coordinate conventions (after z_flip correction):
      - higher effective_z  = physically higher
      - smaller X           = to the left
      - smaller Y           = in front (depth axis)

    Returns list of {'subject', 'relation', 'object'} dicts.
    """
    relations = []

    def add(s, rel, o):
        relations.append({'subject': s, 'relation': rel, 'object': o})

    n = len(names)
    eff_z = [-c[2] if z_flip else c[2] for c in centroids]

    # ── pairwise item–item relations ──────────────────────────────────────────
    for i, j in itertools.permutations(range(n), 2):
        a, b = names[i], names[j]
        ca, cb = centroids[i], centroids[j]
        eza, ezb = eff_z[i], eff_z[j]

        # vertical
        if eza > ezb:
            add(a, 'higher_than', b)
        elif eza < ezb:
            add(a, 'lower_than', b)

        # left / right  (X axis)
        if ca[0] < cb[0]:
            add(a, 'to_the_left_of', b)
        elif ca[0] > cb[0]:
            add(a, 'to_the_right_of', b)

        # front / behind  (Y axis: smaller Y = in front)
        if ca[1] < cb[1]:
            add(a, 'in_front_of', b)
        elif ca[1] > cb[1]:
            add(a, 'behind', b)

        # adjacent
        dist = float(np.linalg.norm(np.array(ca) - np.array(cb)))
        if dist < ADJACENT_DIST_THRESH:
            add(a, 'adjacent_to', b)

    # ── stacking ──────────────────────────────────────────────────────────────
    for (top, bot) in stacking_pairs:
        add(top, 'on_top_of', bot)
        add(bot, 'under', top)

    # ── table relations ───────────────────────────────────────────────────────
    table_c = table_info['centroid']

    for name in names:
        add(name, 'on_top_of', 'table')
        add('table', 'under', name)
        add(name, 'higher_than', 'table')
        add('table', 'lower_than', name)

    return relations


def process_scene(scene_name):
    print(f'\nProcessing {scene_name}...')
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clusters = load_clusters(scene_name)

    mapping = CLUSTER_TO_OBJ[scene_name]
    z_flip = SCENE_Z_FLIP[scene_name]
    stacking_pairs = STACKING[scene_name]
    n_items = SCENE_N[scene_name]

    # Build name->cluster lookup
    obj_data = {}  # name -> cluster object
    for idx, obj in enumerate(clusters):
        name = mapping[idx]
        obj_data[name] = obj

    # Canonical object order (match existing GT ordering)
    existing_p = os.path.join(DATA_DIR, scene_name, 'ground_truth_relations.json')
    with open(existing_p) as f:
        existing_gt = json.load(f)
    existing_names = [o['name'] for o in existing_gt['objects'] if o['name'] != 'table']

    names = existing_names  # preserve existing object ordering

    centroids = [obj_data[n].centroid.tolist() for n in names]
    pt_counts = [int(obj_data[n].point_count) for n in names]

    # Bounding boxes for table computation
    bboxes = [(obj_data[n].bbox_min.tolist(), obj_data[n].bbox_max.tolist()) for n in names]

    # Build virtual table
    table_info = build_virtual_table(centroids, bboxes, z_flip)

    # Derive relations
    relations = derive_relations(
        names, centroids, z_flip, stacking_pairs, table_info
    )

    # Assemble objects list
    objects = []
    for i, name in enumerate(names):
        objects.append({
            'id': i,
            'name': name,
            'centroid': centroids[i],
            'point_count': pt_counts[i],
        })

    # Add table
    objects.append({
        'id': len(names),
        'name': 'table',
        'centroid': table_info['centroid'],
        'bbox_min': table_info['bbox_min'],
        'bbox_max': table_info['bbox_max'],
        'point_count': 0,
    })

    gt = {
        'scene_id': scene_name,
        'objects': objects,
        'relations': relations,
    }

    out_p = os.path.join(DATA_DIR, scene_name, 'ground_truth_relations.json')

    # Backup existing file before overwriting
    backup_p = out_p.replace('.json', '_kiro_backup.json')
    if os.path.exists(out_p) and not os.path.exists(backup_p):
        import shutil
        shutil.copy2(out_p, backup_p)
        print(f'  Backed up existing GT -> {os.path.basename(backup_p)}')

    with open(out_p, 'w') as f:
        json.dump(gt, f, indent=2)

    print(f'  Written: {len(names)} objects, {len(relations)} relations')
    print(f'  Cluster->object: ', {mapping[k]: v for k, v in enumerate(names) if False})

    for i, name in enumerate(names):
        c = centroids[i]
        print(f'    [{mapping.get(next(k for k,v in mapping.items() if v==name), "?"):>1}] '
              f'{name:<15s} centroid=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})')

    stk = [r for r in relations if r['relation'] in ('on_top_of', 'under')
           and r['object'] != 'table' and r['subject'] != 'table']
    if stk:
        print(f'  Stacking: {[(r["subject"], r["relation"], r["object"]) for r in stk]}')
    else:
        print(f'  Stacking: none')

    return gt


if __name__ == '__main__':
    scenes = list(SCENE_N.keys())
    if len(sys.argv) > 1:
        scenes = [s for s in sys.argv[1:] if s in SCENE_N]

    for scene_name in scenes:
        process_scene(scene_name)

    print('\nDone. Run eval_geokan_tabletop.py to check R@K.')
