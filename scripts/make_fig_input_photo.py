"""
Generate the standalone "sample input image" figure (review comment 78):
a single raw smartphone-video frame from scene_06, the same scene used for
the qualitative example in Fig. 5. The photo is the real captured input,
cropped only to trim the excess empty tabletop on the right/bottom that the
raw 16:9 frame happens to include -- all 5 objects and their surrounding
context stay untouched, with a thin frame for visual consistency with the
other figures.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

PHOTO_PATH = "D:/logicsplat_data/processed/scene_06/images/frame_00038.png"
OUT_PATH = "figures/fig_input_photo.png"
FRAME_COL = "#1E3A5F"

img_full = mpimg.imread(PHOTO_PATH)
h0, w0 = img_full.shape[0], img_full.shape[1]

# Crop out excess empty tabletop/wall: trim ~12% off the left (blank wall
# before the router), ~7% off the right (beyond the pen), and ~12% off the
# bottom (beyond the pen/table edge); keep full top (wall context above
# the router/box).
img = img_full[0:int(h0 * 0.88), int(w0 * 0.12):int(w0 * 0.93)]
h, w = img.shape[0], img.shape[1]

FIG_W_IN = 5.15
FIG_H_IN = FIG_W_IN * (h / w)
fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=300)
ax.imshow(img, extent=(0, w, 0, h))
ax.set_xlim(0, w)
ax.set_ylim(0, h)
ax.axis("off")
ax.add_patch(Rectangle((0, 0), w, h, fill=False, edgecolor=FRAME_COL,
                        linewidth=1.6, zorder=5))

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(OUT_PATH, dpi=300, facecolor="white")
print(f"Saved {OUT_PATH}")
