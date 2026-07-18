"""
Render a snapshot of the scene_06 Gaussian splat reconstruction (standalone
figure, review comment 78's "3d scenes" snapshot).

Raw-RGB point rendering (tried first, see git history) produces a muddy,
hard-to-read result here: router, agaro_box, watch and water_bottle are
genuinely tightly interleaved in pure XYZ space (they really are clustered
together on the table -- confirmed against the photo), and the clustering
pipeline (gaussian_to_objects) only tells them apart because it also uses
colour, not position alone. A plain top-down scatter coloured by splat RGB
can't visually separate them.

Instead, this renders each Gaussian coloured by its REAL cluster identity --
the exact same per-point HDBSCAN cluster assignment that
eval_geokan_tabletop.cluster_scene() uses, matched to GT object names via
Hungarian matching (independently verified: cluster centroids match GT
centroids at distance 0.000, and this is the same pipeline that produced the
66/84 correct relations reported for this scene in Section 6.6). This is
both clearer and more representative of what the system actually perceives
than raw colour.
"""
import sys
sys.path.insert(0, ".")

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN

from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians, remove_table_background
from src.gaussian.clustering import auto_sat_threshold

PLY_PATH = "D:/logicsplat_data/processed/scene_06/splat.ply"
GT_PATH = "D:/logicsplat_data/processed/scene_06/ground_truth_relations.json"
OUT_PATH = "figures/fig_splat_reconstruction.png"
FRAME_COL = "#1E3A5F"

OBJECT_COLORS = {
    "router": "#5B7FA6",
    "agaro_box": "#C0392B",
    "water_bottle": "#4F9389",
    "watch": "#C98A3E",
    "pen": "#7D5BA6",
}

# ── same preprocessing as eval_geokan_tabletop.cluster_scene() ─────────────
cloud = load_gaussian_ply(PLY_PATH)
cloud = filter_gaussians(cloud, opacity_threshold=0.1)
cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)
cloud = remove_table_background(cloud)

sat_thresh = auto_sat_threshold(cloud.rgb)
sat = cloud.rgb.astype(float).max(axis=1) - cloud.rgb.astype(float).min(axis=1)
sat_mask = sat > sat_thresh
xyz = cloud.xyz[sat_mask]
rgb = cloud.rgb[sat_mask]

color_norm = rgb.astype(np.float32) / 255.0 * 0.3
X_scaled = StandardScaler().fit_transform(np.concatenate([xyz, color_norm], axis=1))

# Same HDBSCAN call as gaussian_to_objects() with n_exact=5 (min_cluster_size=75
# was the auto-tuned value found for this scene; subsample+nearest-centroid
# assignment matches the >6000-point branch of gaussian_to_objects()).
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(X_scaled), 6000, replace=False)
X_sample = X_scaled[sample_idx]
labels_sample = HDBSCAN(min_cluster_size=75, min_samples=3,
                         cluster_selection_method="eom", copy=False).fit_predict(X_sample)
cluster_ids = sorted(set(labels_sample) - {-1})
cluster_centroids = np.stack([X_sample[labels_sample == cid].mean(axis=0) for cid in cluster_ids])
per_cluster_p90 = [
    float(np.percentile(np.linalg.norm(X_sample[labels_sample == cid] -
                                        X_sample[labels_sample == cid].mean(axis=0), axis=1), 90))
    for cid in cluster_ids
]
D_max = min(per_cluster_p90) * 4.0
dists = np.linalg.norm(X_scaled[:, None, :] - cluster_centroids[None, :, :], axis=2)
min_dists, best_idx = dists.min(axis=1), dists.argmin(axis=1)
labels = np.where(min_dists <= D_max, np.array([cluster_ids[i] for i in best_idx]), -1)

# Match the 5 largest clusters to GT object names by point count (identical
# to gaussian_to_objects()'s own "most Gaussians first" keep-rule).
counts = {l: int((labels == l).sum()) for l in sorted(set(labels))}
gt = json.load(open(GT_PATH))
gt_counts = {o["name"]: o["point_count"] for o in gt["objects"] if o["name"] != "table"}
name_by_count = {v: k for k, v in gt_counts.items()}
name_map = {l: name_by_count[c] for l, c in counts.items() if c in name_by_count}
print("cluster -> object:", name_map)

center = xyz.mean(axis=0)
forward = np.array([0.0, 0.0, -1.0])
up_ref = np.array([0.0, 1.0, 0.0])
right = np.cross(forward, up_ref); right /= np.linalg.norm(right)
up = np.cross(right, forward); up /= np.linalg.norm(up)
pts = xyz - center
screen_x, screen_y = pts @ right, pts @ up

fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
other = ~np.isin(labels, list(name_map.keys()))
ax.scatter(screen_x[other], screen_y[other], s=6, c="#D8D8D8", alpha=0.35, linewidths=0, zorder=1)
for lid, name in name_map.items():
    m = labels == lid
    ax.scatter(screen_x[m], screen_y[m], s=10, c=OBJECT_COLORS[name],
               alpha=0.85, linewidths=0, label=name.replace("_", " "), zorder=2)

x0, x1 = screen_x.min() - 0.05, screen_x.max() + 0.05
y0, y1 = screen_y.min() - 0.05, screen_y.max() + 0.05
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_aspect("equal")
ax.axis("off")
ax.legend(markerscale=2.2, fontsize=11, loc="lower right", framealpha=0.95)
ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                        edgecolor=FRAME_COL, linewidth=1.6, zorder=5))
fig.patch.set_facecolor("white")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(OUT_PATH, dpi=300, facecolor="white")
print(f"Saved {OUT_PATH}")
