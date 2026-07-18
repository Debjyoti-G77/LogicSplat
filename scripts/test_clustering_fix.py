"""Quick smoke test: verify scenes 09 and 10 now find 4 clusters."""
import sys
sys.path.insert(0, ".")
import warnings
warnings.filterwarnings("ignore")

from src.gaussian.loader import load_gaussian_ply, filter_gaussians
from src.gaussian.clustering import gaussian_to_objects

for scene, expected in [("scene_09", 4), ("scene_10", 4)]:
    cloud = load_gaussian_ply(f"D:/logicsplat_data/processed/{scene}/splat.ply")
    filtered = filter_gaussians(cloud, opacity_threshold=0.1)

    # Default (no hint) — should still work for other scenes
    objects_default, params_default = gaussian_to_objects(filtered)
    print(f"{scene} default: {len(objects_default)} clusters  method={params_default['cluster_method']}  mcs={params_default['min_cluster_size']}")

    # With hint=4
    objects_hint, params_hint = gaussian_to_objects(filtered, target_min=3, target_max=5)
    status = "✓" if len(objects_hint) == expected else "✗"
    print(f"{scene} hint=4:  {len(objects_hint)} clusters  method={params_hint['cluster_method']}  mcs={params_hint['min_cluster_size']}  {status}")
    print()

print("Done.")
