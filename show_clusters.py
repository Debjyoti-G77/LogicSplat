"""Print cluster centroids for each eval scene so we can hard-code assignments."""
import sys, os
sys.path.insert(0, '.')
import numpy as np

from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians, remove_table_background
from src.gaussian.clustering import gaussian_to_objects

DATA_DIR = 'D:/logicsplat_data/processed'

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

for scene_name, n_items in SCENE_N.items():
    ply = os.path.join(DATA_DIR, scene_name, 'splat.ply')
    cloud = load_gaussian_ply(ply)
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)
    cloud = remove_table_background(cloud)
    objs, _ = gaussian_to_objects(cloud, target_min=n_items, target_max=n_items+3, n_exact=n_items)
    objs = list(objs)

    print(f'\n{scene_name} ({n_items} objects):')
    for i, o in enumerate(objs):
        c = o.centroid
        sz = o.bbox_max - o.bbox_min
        print(f'  [{i}] centroid=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})  '
              f'size=({sz[0]:.2f},{sz[1]:.2f},{sz[2]:.2f})  pts={o.point_count}')
