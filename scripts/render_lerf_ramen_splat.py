"""
Render a snapshot of the LERF 'ramen' Gaussian splat reconstruction, coloured
by real cluster identity -- the LERF counterpart to render_scene06_splat.py.

Reimplements gaussian_to_objects()'s internal steps inline (that function
only returns aggregated Object3D instances, not the per-Gaussian cluster
label array needed for a scatter plot) with identical parameters to
demo/precompute.py's process_lerf_scene() call: target_min=n_exact,
target_max=n_exact+3, n_exact=13, same rng seed=42. Object naming reuses the
real projection + label-matching path (demo/lerf_project.py) so the legend
matches the same names shown in the LogicSplat Live demo and slide 16.
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "demo")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN

from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
from src.gaussian.clustering import auto_sat_threshold, auto_min_cluster_size, gaussian_to_objects
from lerf_project import project_scene, load_label, label_boxes, match_to_labels

SCENE, FRAME, N_EXACT = "ramen", "frame_00006.jpg", 13
LERF_DATA_ROOT = "D:/lerf_data/lerf_ovs"
PLY_PATH = f"{LERF_DATA_ROOT}/{SCENE}/splat/splat.ply"
OUT_PATH = "figures/fig_splat_reconstruction_lerf_ramen.png"
FRAME_COL = "#1E3A5F"

OBJECT_COLORS = {
    "bowl":            "#5B7FA6",
    "wavy noodles":    "#C98A3E",
    "egg":             "#E8C547",
    "kamaboko":        "#C0392B",
    "nori":            "#3E6B4F",
    "onion segments":  "#7DAA5C",
    "plate":           "#B25FA0",
    "chopsticks":      "#7D5BA6",
    "sake cup":        "#4F9389",
    "napkin":          "#2F6F8F",
}

cloud = load_gaussian_ply(PLY_PATH)
cloud = filter_gaussians(cloud, opacity_threshold=0.1)
cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)

target_min, target_max = N_EXACT, N_EXACT + 3
target_min = max(N_EXACT, target_min)
target_max = max(N_EXACT * 4, target_max)

sat_thresh = auto_sat_threshold(cloud.rgb)
sat = cloud.rgb.astype(float).max(axis=1) - cloud.rgb.astype(float).min(axis=1)
sat_mask = sat > sat_thresh
if sat_mask.sum() < 0.05 * cloud.num_gaussians:
    sat_thresh *= 0.5
    sat_mask = sat > sat_thresh
if sat_mask.sum() < 0.05 * cloud.num_gaussians:
    sat_mask = np.ones(cloud.num_gaussians, dtype=bool)

xyz = cloud.xyz[sat_mask]
rgb = cloud.rgb[sat_mask]
color_norm = rgb.astype(np.float32) / 255.0 * 0.3
X_scaled = StandardScaler().fit_transform(np.concatenate([xyz, color_norm], axis=1))

mcs, method = auto_min_cluster_size(X_scaled, target_min=target_min, target_max=target_max,
                                     methods=("eom", "leaf"))

_MAX_CLUSTER_PTS = 6000
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(X_scaled), _MAX_CLUSTER_PTS, replace=False)
X_sample = X_scaled[sample_idx]
labels_sample = HDBSCAN(min_cluster_size=mcs, min_samples=3,
                         cluster_selection_method=method, copy=False).fit_predict(X_sample)
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

# n_exact trim: keep the N_EXACT clusters with the most Gaussians, renumber 0..N_EXACT-1
counts = {l: int((labels == l).sum()) for l in sorted(set(labels) - {-1})}
kept = sorted(counts, key=counts.get, reverse=True)[:N_EXACT]
kept_sorted = sorted(kept)  # gaussian_to_objects renumbers in ascending original-id order
remap = {old: new for new, old in enumerate(kept_sorted)}
final_labels = np.array([remap.get(l, -1) for l in labels])

# Real object list + real name assignment (identical to precompute.py's process_lerf_scene)
objects, _ = gaussian_to_objects(cloud, target_min=N_EXACT, target_max=N_EXACT + 3, n_exact=N_EXACT)
label = load_label(SCENE, FRAME)
boxes_px = label_boxes(label)
_, projected, _ = project_scene(SCENE, FRAME, N_EXACT)
assignment = match_to_labels(projected, boxes_px)
idx_to_name = {idx: cat for idx, cat in assignment.items()}
print("cluster -> object:", idx_to_name)

object_mask = final_labels != -1
center = xyz[object_mask].mean(axis=0)
forward = np.array([0.0, 0.0, -1.0])
up_ref = np.array([0.0, 1.0, 0.0])
right = np.cross(forward, up_ref); right /= np.linalg.norm(right)
up = np.cross(right, forward); up /= np.linalg.norm(up)
pts = xyz - center
screen_x, screen_y = pts @ right, pts @ up

object_mask = final_labels != -1
for lid, name in sorted(idx_to_name.items()):
    print(name, "points:", int((final_labels == lid).sum()))

fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
other = final_labels == -1
ax.scatter(screen_x[other], screen_y[other], s=6, c="#D8D8D8", alpha=0.3, linewidths=0, zorder=1)
for lid, name in sorted(idx_to_name.items()):
    m = final_labels == lid
    if not m.any():
        continue
    ax.scatter(screen_x[m], screen_y[m], s=14, c=OBJECT_COLORS.get(name, "#333333"),
               alpha=0.9, linewidths=0, label=name, zorder=2)

pad = 0.06
x0, x1 = screen_x[object_mask].min() - pad, screen_x[object_mask].max() + pad
y0, y1 = screen_y[object_mask].min() - pad, screen_y[object_mask].max() + pad
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_aspect("equal")
ax.axis("off")
ax.legend(markerscale=2.4, fontsize=8, loc="lower right", framealpha=0.95, ncol=2)
ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                        edgecolor=FRAME_COL, linewidth=1.6, zorder=5))
fig.patch.set_facecolor("white")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(OUT_PATH, dpi=300, facecolor="white")
print(f"Saved {OUT_PATH}")
