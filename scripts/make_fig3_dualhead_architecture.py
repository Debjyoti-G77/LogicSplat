"""
Generate Figure 3 (GeoKANRelationGNN dual-head architecture), v2.

v1 had two real problems flagged after review: text too small relative to
the available box area, and a long, mostly-empty vertical drop between the
relation-logit boxes and "Concatenate" that wasted space for no reason.
Both fixed here: font sizes bumped up throughout, and every gap recomputed
to a tight, consistent 3-5 unit clearance instead of an arbitrary large one.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"

COL_NEUTRAL      = "#FAFAFB"
COL_NEUTRAL_EDGE = "#94A0B4"
COL_OUTPUT_EDGE  = "#1E3A5F"
TEXT_DARK = "#1F2430"
TEXT_SUB  = "#454C59"
ARROW_COL = "#3A3F4B"

COL_CONTACT      = ("#EAF4F2", "#4F9389", "#2B5E54")
COL_DIRECTIONAL  = ("#FBF1E1", "#C98A3E", "#8A5A1E")

FIG_W_IN = 5.15
FIG_H_IN = FIG_W_IN / 0.592
fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=600)
ax.set_xlim(0, 100)
ax.set_ylim(-16, 153)
ax.axis("off")

def draw_box(cx, cy, w, h, title, subtitle, face, edge, title_color=TEXT_DARK,
             sub_color=TEXT_SUB, linewidth=1.0, title_fs=10.5, sub_fs=8.0):
    shadow = FancyBboxPatch(
        (cx - w / 2 + 0.4, cy - h / 2 - 0.4), w, h,
        boxstyle="round,pad=0,rounding_size=1.5",
        linewidth=0, facecolor="#1F2430", alpha=0.10, zorder=1,
    )
    ax.add_patch(shadow)
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=1.5",
        linewidth=linewidth, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(box)
    if title:
        ax.text(cx, cy + h * 0.23, title, ha="center", va="center",
                 fontsize=title_fs, fontweight="bold", color=title_color,
                 zorder=3, linespacing=1.3)
        ax.text(cx, cy - h * 0.16, subtitle, ha="center", va="center",
                 fontsize=sub_fs, fontweight="semibold", color=sub_color,
                 zorder=3, linespacing=1.5)
    else:
        # no title -- centre the subtitle on the box instead of leaving the
        # upper portion empty (the offset above is only correct when a
        # title occupies that space)
        ax.text(cx, cy, subtitle, ha="center", va="center",
                 fontsize=sub_fs, fontweight="semibold", color=sub_color,
                 zorder=3, linespacing=1.6)

def draw_arrow(p1, p2, curve=0.0):
    arrow = FancyArrowPatch(
        p1, p2, connectionstyle=f"arc3,rad={curve}",
        arrowstyle="-|>,head_length=4.0,head_width=2.6",
        linewidth=1.2, color=ARROW_COL, zorder=1.5,
    )
    ax.add_patch(arrow)

TW = 62
GAP = 4.5

# ── Shared trunk (computed top-down with a fixed, tight gap) ─────────────
H1, H2, H3, H4 = 16, 16, 20, 16
y1 = 142
y2 = y1 - H1 / 2 - GAP - H2 / 2
y3 = y2 - H2 / 2 - GAP - H3 / 2
y4 = y3 - H3 / 2 - GAP - H4 / 2

draw_box(50, y1, TW, H1, "Node Features", "10-dim, per object instance",
          COL_NEUTRAL, COL_NEUTRAL_EDGE)
draw_arrow((50, y1 - H1 / 2), (50, y2 + H2 / 2))

draw_box(50, y2, TW, H2, "Node Encoder", "Linear + LayerNorm + GELU $\\to$ 128-dim",
          COL_NEUTRAL, COL_NEUTRAL_EDGE)
draw_arrow((50, y2 - H2 / 2), (50, y3 + H3 / 2))

draw_box(50, y3, TW, H3, "GATv2Conv $\\times$ 2",
          "4 heads averaged, residual,\nedge features (22-dim) shape attention",
          COL_NEUTRAL, COL_NEUTRAL_EDGE)
draw_arrow((50, y3 - H3 / 2), (50, y4 + H4 / 2))

draw_box(50, y4, TW, H4, "Pair Construction",
          "$\\mathbf{[h_i,\\ h_j,\\ h_i{-}h_j,\\ h_i{\\odot}h_j]}$ $\\to$ 64-dim",
          COL_NEUTRAL, COL_NEUTRAL_EDGE)

# ── Fork: contact head / directional head ────────────────────────────────
FX_L, FX_R = 25, 75
FW = 44
H5, H6 = 18, 20
GAP_FORK = 18  # extra room: this transition carries a label + bus + arrow
y5 = y4 - H4 / 2 - GAP_FORK - H5 / 2
y6 = y5 - H5 / 2 - GAP - H6 / 2

# Manifold split (mirrors the merge pattern used lower down): one drop from
# the trunk to a bus line, then a clean vertical arrow into each head box.
# Replaces two diagonal lines that visually read as a single curved "frown"
# crossing right through the column labels.
split_label_y = y4 - H4 / 2 - 3
split_bus_y = split_label_y - 5

ax.plot([50, 50], [y4 - H4 / 2, split_bus_y], color=ARROW_COL, linewidth=1.2,
         zorder=1.5, solid_capstyle="round")
ax.plot([FX_L, FX_R], [split_bus_y, split_bus_y], color=ARROW_COL, linewidth=1.2,
         zorder=1.5, solid_capstyle="round")
draw_arrow((FX_L, split_bus_y), (FX_L, y5 + H5 / 2))
draw_arrow((FX_R, split_bus_y), (FX_R, y5 + H5 / 2))

ax.text(FX_L, split_label_y, "Contact Head", ha="center", va="center",
         fontsize=9.2, fontweight="bold", color=COL_CONTACT[2], zorder=3)
ax.text(FX_R, split_label_y, "Directional Head", ha="center", va="center",
         fontsize=9.2, fontweight="bold", color=COL_DIRECTIONAL[2], zorder=3)

draw_box(FX_L, y5, FW, H5, "", "86-dim input\n(64 pair + full 22 edge dims)\n"
          "2$\\times$ GeoKAN layers (Sec. 5.5)",
          COL_CONTACT[0], COL_CONTACT[1], sub_color=TEXT_SUB, sub_fs=8.0)
draw_box(FX_R, y5, FW, H5, "", "74-dim input\n(64 pair + first 10 edge dims)\n"
          "2$\\times$ GeoKAN layers (Sec. 5.5)",
          COL_DIRECTIONAL[0], COL_DIRECTIONAL[1], sub_color=TEXT_SUB, sub_fs=8.0)

draw_arrow((FX_L, y5 - H5 / 2), (FX_L, y6 + H6 / 2))
draw_arrow((FX_R, y5 - H5 / 2), (FX_R, y6 + H6 / 2))

draw_box(FX_L, y6, FW, H6, "Linear $\\to$ 4 logits",
          "on_top_of, under,\nattached_to, adjacent_to",
          COL_CONTACT[0], COL_CONTACT[1], title_color=COL_CONTACT[2],
          title_fs=9.6, sub_color=TEXT_DARK, sub_fs=8.0)
draw_box(FX_R, y6, FW, H6, "Linear $\\to$ 6 logits",
          "left_of, right_of, in_front_of,\nbehind, higher_than, lower_than",
          COL_DIRECTIONAL[0], COL_DIRECTIONAL[1], title_color=COL_DIRECTIONAL[2],
          title_fs=9.6, sub_color=TEXT_DARK, sub_fs=8.0)

# ── Merge: tight manifold drop straight into Concatenate ─────────────────
BUS_Y = y6 - H6 / 2 - 3
ax.plot([FX_L, FX_L], [y6 - H6 / 2, BUS_Y], color=ARROW_COL, linewidth=1.2,
         zorder=1.5, solid_capstyle="round")
ax.plot([FX_R, FX_R], [y6 - H6 / 2, BUS_Y], color=ARROW_COL, linewidth=1.2,
         zorder=1.5, solid_capstyle="round")
ax.plot([FX_L, FX_R], [BUS_Y, BUS_Y], color=ARROW_COL, linewidth=1.2,
         zorder=1.5, solid_capstyle="round")

H7 = 13
y7 = BUS_Y - 3 - H7 / 2
draw_arrow((50, BUS_Y), (50, y7 + H7 / 2))

draw_box(50, y7, 66, H7, "Concatenate", "10-dim relation logit vector (output)",
          COL_NEUTRAL, COL_OUTPUT_EDGE, linewidth=1.8, title_fs=10.5, sub_fs=8.0)

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
plt.savefig("figures/fig3_dualhead_architecture.png", dpi=600, facecolor="white")
print("Saved figures/fig3_dualhead_architecture.png")
