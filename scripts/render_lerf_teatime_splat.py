"""
Render a snapshot of the LERF 'teatime' Gaussian splat reconstruction, coloured
by real cluster identity -- teatime counterpart to render_lerf_ramen_splat.py.

Identical pipeline parameters to demo/precompute.py's process_lerf_scene() for
lerf_teatime: frame_00025.jpg, n_exact=10, rng seed=42. Object naming reuses the
real projection + label-matching path (demo/lerf_project.py) so the legend
matches the names shown in the LogicSplat Live demo and demo/data/lerf_teatime.json.
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

SCENE, FRAME, N_EXACT = "teatime", "frame_00025.jpg", 10
LERF_DATA_ROOT = "D:/lerf_data/lerf_ovs"
PLY_PATH = f"{LERF_DATA_ROOT}/{SCENE}/splat/splat.ply"
OUT_PATH = "figures/fig_splat_reconstruction_lerf_teatime.png"
FRAME_COL = "#1E3A5F"

OBJECT_COLORS = {
    "stuffed bear":    "#8E5B3E",
    "hooves":          "#7D5BA6",
    "coffee":          "#C98A3E",
    "coffee mug":      "#C0392B",
    "tea in a glass":  "#4F9389",
    "plate":           "#B25FA0",
    "three cookies":   "#E8C547",
    "bag of cookies":  "#3E6B4F",
    "paper napkin":    "#5B7FA6",
    "sheep":           "#7DAA5C",
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

# ---- top-down (plan) view, oriented to match the photo's left-right ----
# The photo's COLMAP camera supplies the orientation: world-up from the camera's
# up vector, plan axes = camera right / camera forward projected to horizontal.
# Left-of in the photo stays left-of in the plan view.
from lerf_project import (read_images_binary, qvec2rotmat,
                          load_dataparser_inverse, find_dataparser_transform)
import os
images = read_images_binary(os.path.join(LERF_DATA_ROOT, SCENE, "sparse/0/images.bin"))
to_colmap = load_dataparser_inverse(find_dataparser_transform(SCENE))
img_entry = next(img for img in images.values() if img.name == FRAME)
Rcam = qvec2rotmat(img_entry.qvec)

M = np.array([to_colmap(p) for p in xyz])           # splatfacto -> COLMAP space
up_w = -Rcam.T @ np.array([0.0, 1.0, 0.0])          # camera up (y is down in cam)
up_w /= np.linalg.norm(up_w)
right_w = Rcam.T @ np.array([1.0, 0.0, 0.0])
right_w -= (right_w @ up_w) * up_w
right_w /= np.linalg.norm(right_w)
fwd_w = Rcam.T @ np.array([0.0, 0.0, 1.0])
fwd_w -= (fwd_w @ up_w) * up_w
fwd_w /= np.linalg.norm(fwd_w)
screen_x = M @ right_w
screen_y = M @ fwd_w                                # away from camera plots upward

# Display each cluster's dense core: expanded-assignment points within their own
# cluster's 90th-percentile radius (the same per-cluster statistic the pipeline
# uses for its expansion cap). Keeps filled silhouettes, trims the far tail of
# the assignment that otherwise smears the table surface into one cluster.
p90_arr = np.array(per_cluster_p90)
core_mask = (labels != -1) & (min_dists <= p90_arr[best_idx])
core_final = np.array([remap.get(l, -1) for l in np.where(core_mask, labels, -1)])

# Per cluster, draw the points CLOSEST to the cluster centroid (in the same
# scaled feature space the pipeline clusters in) -- the tight core each object
# is aggregated from -- rather than a random sample, which drags in the
# assignment's far tail (large background surfaces).
_MAX_DRAW = 2200
draw_sets = {}
for lid in sorted(idx_to_name):
    m_idx = np.flatnonzero(core_final == lid)
    if len(m_idx) == 0:
        continue
    order = np.argsort(min_dists[m_idx])
    sel = m_idx[order[:_MAX_DRAW]]
    # trim spatial outliers (remote islands) for display: keep points within
    # 3.5x the median absolute deviation of the cluster's plan-view footprint
    sx, sy = screen_x[sel], screen_y[sel]
    med = np.array([np.median(sx), np.median(sy)])
    r = np.hypot(sx - med[0], sy - med[1])
    mad = np.median(r) + 1e-9
    draw_sets[lid] = sel[r <= 3.5 * mad]

drawn_all = np.concatenate(list(draw_sets.values()))
x0, x1 = np.percentile(screen_x[drawn_all], [1, 99])
y0, y1 = np.percentile(screen_y[drawn_all], [1, 99])
pad_x, pad_y = 0.14 * (x1 - x0), 0.14 * (y1 - y0)
x0, x1 = x0 - pad_x, x1 + pad_x
y0, y1 = y0 - pad_y, y1 + pad_y

fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
bg = (core_final == -1) & (screen_x >= x0) & (screen_x <= x1) & \
     (screen_y >= y0) & (screen_y <= y1)
step = max(1, bg.sum() // 25000)
bg_idx = np.flatnonzero(bg)[::step]
ax.scatter(screen_x[bg_idx], screen_y[bg_idx], s=6, c="#D8D8D8", alpha=0.3,
           linewidths=0, zorder=1)
for lid, name in sorted(idx_to_name.items()):
    if lid not in draw_sets:
        continue
    m_idx = draw_sets[lid]
    print(name, "points drawn:", len(m_idx))
    ax.scatter(screen_x[m_idx], screen_y[m_idx], s=10,
               c=OBJECT_COLORS.get(name, "#333333"),
               alpha=0.85, linewidths=0, label=name, zorder=2)

ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_aspect("equal")
ax.axis("off")
ax.legend(markerscale=2.0, fontsize=8, loc="lower right", framealpha=0.95, ncol=2)
ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                        edgecolor=FRAME_COL, linewidth=1.6, zorder=5))
fig.patch.set_facecolor("white")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(OUT_PATH, dpi=300, facecolor="white")
print(f"Saved {OUT_PATH}")
