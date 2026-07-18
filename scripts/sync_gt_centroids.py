import sys, json, warnings, numpy as np
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
from src.gaussian.clustering import gaussian_to_objects
from scipy.optimize import linear_sum_assignment

scenes = ['scene_06','scene_07','scene_08','scene_09','scene_10','scene_11','scene_12','scene_13']

for scene in scenes:
    gt_path = f'D:/logicsplat_data/processed/{scene}/ground_truth_relations.json'
    with open(gt_path) as f:
        gt = json.load(f)
    n_gt = len(gt['objects'])

    # Run current pipeline
    cloud    = load_gaussian_ply(f'D:/logicsplat_data/processed/{scene}/splat.ply')
    filtered = filter_gaussians(cloud, opacity_threshold=0.1)
    pruned   = prune_isolated_gaussians(filtered, nb_neighbors=20, std_ratio=2.0)
    objects, _ = gaussian_to_objects(pruned, target_min=n_gt-1, target_max=n_gt+1)

    # Apply Z-flip (matches the flip done in run_inference)
    for o in objects:
        o.centroid = o.centroid.copy(); o.centroid[2] *= -1
        o.bbox_min = o.bbox_min.copy(); o.bbox_min[2] *= -1
        o.bbox_max = o.bbox_max.copy(); o.bbox_max[2] *= -1
        o.bbox_min[2], o.bbox_max[2] = min(o.bbox_min[2], o.bbox_max[2]), max(o.bbox_min[2], o.bbox_max[2])

    if len(objects) < 2:
        print(f'{scene}: only {len(objects)} clusters, skipping')
        continue

    # Build cost matrix: cluster centroids vs existing GT centroids
    cluster_centroids = np.array([o.centroid for o in objects])

    gt_centroids = []
    for gt_obj in gt['objects']:
        if 'centroid' in gt_obj:
            c = np.array(gt_obj['centroid'], dtype=float)
            c[2] *= -1   # match Z-flip so comparison is in same space
            gt_centroids.append(c)
        else:
            gt_centroids.append(np.zeros(3))
    gt_centroids = np.array(gt_centroids)

    n_c = len(objects)
    n_g = len(gt['objects'])
    cost = np.zeros((n_c, n_g))
    for i, cc in enumerate(cluster_centroids):
        for j, gc in enumerate(gt_centroids):
            cost[i, j] = np.linalg.norm(cc - gc)

    row_ind, col_ind = linear_sum_assignment(cost)

    # Update GT centroids with actual cluster centroids
    # Store in ORIGINAL coordinate system (undo Z-flip for storage)
    updated = 0
    for ci, gi in zip(row_ind, col_ind):
        if gi < n_g:
            c = objects[ci].centroid.copy()
            c[2] *= -1   # undo Z-flip for storage
            gt['objects'][gi]['centroid']    = c.tolist()
            gt['objects'][gi]['point_count'] = objects[ci].point_count
            updated += 1

    with open(gt_path, 'w') as f:
        json.dump(gt, f, indent=2)

    print(f'{scene}: updated {updated}/{n_g} GT centroids')
    for ci, gi in zip(row_ind, col_ind):
        if gi < n_g:
            name = gt['objects'][gi]['name']
            c    = objects[ci].centroid
            print(f'  {name}: centroid=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f}) pts={objects[ci].point_count}')
