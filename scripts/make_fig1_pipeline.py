"""
Generate Figure 1 (LogicSplat pipeline) as a polished, code-based diagram.

Layout: two-row zigzag (S-curve), right-aligned so GeoKANRelationGNN sits
directly under HDBSCAN Clustering (clean vertical connector, no diagonal).
Both rows use identical box-to-box spacing -- the previous version stretched
row 2's 3 boxes across the same width as row 1's 4 boxes, producing uneven
gaps. Figure canvas width is set to match the LaTeX \\textwidth the image
will be scaled to, so font sizes in the code are the literal final printed
point size (the original version sized fonts for a 12.5in canvas that then
got shrunk to ~5.15in by \\includegraphics, making text far smaller than
intended).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

# ── Palette: restrained neutral, one navy accent for the learned component ──
COL_NEUTRAL      = "#FAFAFB"
COL_NEUTRAL_EDGE = "#94A0B4"
COL_ACCENT       = "#1E3A5F"
COL_ACCENT_EDGE  = "#142847"
COL_OUTPUT       = "#FFFFFF"
COL_OUTPUT_EDGE  = "#1E3A5F"
TEXT_DARK = "#1F2430"
TEXT_SUB  = "#454C59"
TEXT_ON_ACCENT = "#FFFFFF"
TEXT_ON_ACCENT_SUB = "#C9D4E3"
ARROW_COL = "#3A3F4B"

# Canvas width = LaTeX \textwidth (~5.15in for sn-jnl single column); height
# follows from the 2:1 layout aspect ratio. dpi is high since the physical
# size is now small -- this keeps pixel density (and crispness) up.
FIG_W_IN = 5.15
FIG_H_IN = FIG_W_IN / 2.35
fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=600)
# Tighter data-coordinate range than before: box width in *inches* depends on
# BOX_W / (xlim range) * FIG_W_IN. Shrinking the figure's physical size means
# this ratio must go up (smaller xlim range for the same BOX_W) or boxes
# shrink in absolute inches even though BOX_W in data-units is unchanged --
# that was the bug in the previous render.
ax.set_xlim(0, 104)
ax.set_ylim(5, 44)
ax.axis("off")

BOX_W, BOX_H = 23.7, 15.5
SPACING = 25.5  # identical center-to-center spacing used in BOTH rows

def draw_box(cx, cy, title, subtitle, face, edge, title_color=TEXT_DARK,
             sub_color=TEXT_SUB, linewidth=1.0):
    shadow = FancyBboxPatch(
        (cx - BOX_W / 2 + 0.4, cy - BOX_H / 2 - 0.4), BOX_W, BOX_H,
        boxstyle="round,pad=0,rounding_size=1.8",
        linewidth=0, facecolor="#1F2430", alpha=0.10, zorder=1,
    )
    ax.add_patch(shadow)
    box = FancyBboxPatch(
        (cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0,rounding_size=1.8",
        linewidth=linewidth, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(box)
    ax.text(cx, cy + 3.2, title, ha="center", va="center",
             fontsize=8.0, fontweight="bold", color=title_color, zorder=3,
             linespacing=1.3)
    ax.text(cx, cy - 2.6, subtitle, ha="center", va="center",
             fontsize=6.0, fontweight="semibold", color=sub_color, zorder=3,
             linespacing=1.4)

def draw_arrow(p1, p2, curve=0.0):
    arrow = FancyArrowPatch(
        p1, p2, connectionstyle=f"arc3,rad={curve}",
        arrowstyle="-|>,head_length=4.5,head_width=3.0",
        linewidth=1.3, color=ARROW_COL, zorder=1.5,
    )
    ax.add_patch(arrow)

# ── Row 1 (top): reconstruction + preprocessing ──────────────────────────
ROW1_Y = 35.5
xs1 = [13 + i * SPACING for i in range(4)]   # 13, 38.5, 64, 89.5
draw_box(xs1[0], ROW1_Y, "Smartphone\nVideo", "RGB frames,\nhandheld capture",
         COL_NEUTRAL, COL_NEUTRAL_EDGE)
draw_box(xs1[1], ROW1_Y, "3D Gaussian\nSplat", "COLMAP + NerfStudio\nsplatfacto, 30k iters",
         COL_NEUTRAL, COL_NEUTRAL_EDGE)
draw_box(xs1[2], ROW1_Y, "Gaussian\nCleaning", "opacity > 0.1, SOR,\nRANSAC plane removal",
         COL_NEUTRAL, COL_NEUTRAL_EDGE)
draw_box(xs1[3], ROW1_Y, "HDBSCAN\nClustering", "auto-tuned min cluster\nsize, per scene",
         COL_NEUTRAL, COL_NEUTRAL_EDGE)

for i in range(3):
    draw_arrow((xs1[i] + BOX_W / 2, ROW1_Y), (xs1[i + 1] - BOX_W / 2, ROW1_Y))

# ── Row 2 (bottom): right-aligned under row 1, identical spacing, 4 boxes ──
# (clustering and feature extraction are split into two real stages here,
# rather than leaving an empty 4th slot under "Smartphone Video")
ROW2_Y = 14.5
xs2 = [xs1[3] - i * SPACING for i in range(4)]   # 89.5, 64, 38.5, 13
draw_box(xs2[0], ROW2_Y, "Feature\nExtraction", "10-dim node /\n22-dim edge features",
         COL_NEUTRAL, COL_NEUTRAL_EDGE)
draw_box(xs2[1], ROW2_Y, "GeoKAN\nRelationGNN", "10 relation logits,\nper-relation thresholds",
         COL_NEUTRAL, COL_NEUTRAL_EDGE)
draw_box(xs2[2], ROW2_Y, "SceneGraph\nRepair", "fixed-point logical\nconsistency",
         COL_NEUTRAL, COL_NEUTRAL_EDGE)
draw_box(xs2[3], ROW2_Y, "Scene Graph", "10 relation types,\ndirected pairs",
         COL_NEUTRAL, COL_NEUTRAL_EDGE)

for i in range(3):
    draw_arrow((xs2[i] - BOX_W / 2, ROW2_Y), (xs2[i + 1] + BOX_W / 2, ROW2_Y))

# clean vertical connector -- xs1[3] == xs2[0], so this is a straight drop
draw_arrow((xs1[3], ROW1_Y - BOX_H / 2), (xs2[0], ROW2_Y + BOX_H / 2))

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
plt.savefig("figures/fig1_pipeline.png", dpi=600, facecolor="white")
print("Saved figures/fig1_pipeline.png")
