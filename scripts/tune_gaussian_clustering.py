
port sys
sys.path.insert(0, '.')
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from src.gaussian.loader import load_gaussian_ply, filter_gaussians
from sklearn.cluster import HDBSCAN

cloud = load_gaussian_ply('D:/logicsplat_data/processed/scene_01/splat.ply')
cf = filter_gaussians(cloud, opacity_threshold=0.1)

rgb = cf.rgb.astype(float)
saturation = rgb.max(axis=1) - rgb.min(axis=1)
print(f'Saturation: mean={saturation.mean():.1f} std={saturation.std():.1f}')

for sat_thresh in [15, 20, 30, 40]:
    mask = saturation > sat_thresh
    pts = cf.xyz[mask]
    print(f'\nsat>{sat_thresh}: {mask.sum()} pts ({mask.sum()/len(cf.xyz)*100:.0f}%)')
    for mcs in [500, 1000, 2000]:
        labels = HDBSCAN(min_cluster_size=mcs, min_samples=3, cluster_selection_method='eom', copy=False).fit_predict(pts)
        n = len(set(labels)) - (1 if -1 in labels else 0)
        noise = (labels==-1).sum()
        print(f'  mcs={mcs}: {n} clusters, {noise} noise')
