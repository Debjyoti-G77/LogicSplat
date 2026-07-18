"""Deep diagnostic — why does scene_06 work but others fail?"""
import sys, warnings, json, os, numpy as np
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
from src.gaussian.clustering import gaussian_to_objects

scenes = ['scene_06','scene_07','scene_08','scene_09','scene_10','scene_11','scene_12','scene_13']

for scene in scenes:
    print(f'\n{"="*60}')
    print(f'SCENE: {scene}')

    with open(f'D:/logicsplat_data/processed/{scene}/ground_truth_relations.json') as f:
        gt = json.load(f)
    n_gt_obj = len(gt['objects'])
    print(f'GT: {n_gt_obj} objects: {[o["name"] for o in gt["objects"]]}')

    cloud = load_gaussian_ply(f'D:/logicsplat_data/processed/{scene}/splat.ply')
    filtered = filter_gaussians(cloud, opacity_threshold=0.1)
    sor_filtered = prune_isolated_gaussians(filtered, nb_neighbors=20, std_ratio=2.0)

    print(f'Gaussians: {cloud.num_gaussians:,} raw → {filtered.num_gaussians:,} after opacity filter → {sor_filtered.num_gaussians:,} after SOR')
    n_sor_removed = filtered.num_gaussians - sor_filtered.num_gaussians
    if n_sor_removed > 0:
        print(f'  SOR removed: {n_sor_removed:,} isolated Gaussians ({100*n_sor_removed/filtered.num_gaussians:.1f}%)')

    objects, params = gaussian_to_objects(
        sor_filtered, target_min=n_gt_obj-1, target_max=n_gt_obj+1
    )

    print(f'Clusters: {len(objects)} found (target {n_gt_obj})')
    print(f'Params: sat={params.get("sat_threshold"):.1f} mcs={params.get("min_cluster_size")} noise={params.get("noise_fraction"):.2f}')

    for o in objects:
        print(f'  Obj{o.uid}: pts={o.point_count:6d} z={o.centroid[2]:8.2f} '
              f'xy=({o.centroid[0]:.2f},{o.centroid[1]:.2f}) '
              f'size=({o.size[0]:.2f},{o.size[1]:.2f},{o.size[2]:.2f}) '
              f'rgb=({o.color[0]},{o.color[1]},{o.color[2]})')

    z_vals = [o.centroid[2] for o in objects]
    pts_vals = [o.point_count for o in objects]
    xy_vals = [(o.centroid[0], o.centroid[1]) for o in objects]
    if len(z_vals) > 1:
        print(f'Z stats: min={min(z_vals):.2f} max={max(z_vals):.2f} std={np.std(z_vals):.2f}')
        print(f'XY range: x=[{min(x for x,y in xy_vals):.2f},{max(x for x,y in xy_vals):.2f}] '
              f'y=[{min(y for x,y in xy_vals):.2f},{max(y for x,y in xy_vals):.2f}]')
        print(f'Points: min={min(pts_vals)} max={max(pts_vals)} ratio={max(pts_vals)/max(min(pts_vals),1):.1f}x')

    tf_path = f'D:/logicsplat_data/processed/{scene}/ns_data/transforms.json'
    if os.path.exists(tf_path):
        with open(tf_path) as f:
            tf = json.load(f)
        print(f'Frames registered: {len(tf.get("frames", []))}')
        print(f'Intrinsics: w={tf.get("w")} h={tf.get("h")} fl_x={tf.get("fl_x","?")}')
    else:
        print('No transforms.json')

    # Check if GT has centroids
    has_centroids = any('centroid' in o for o in gt['objects'])
    print(f'GT has centroids: {has_centroids}')
