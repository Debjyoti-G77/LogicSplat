"""
Generate Figure 4: the controlled architecture ablation (Section 5.10/6.1).

Two panels sharing the same input/output wrapper (same skip connection with
u, same Concat+Linear output mixing layer, per the manuscript text) -- only
the internal layer differs:
  (a) GeoKAN-Gamma: the metric-warp + fixed radial-basis layer from Fig. 2.
  (b) MLP baseline: a single linear layer to the same output width the basis
      expansion would have produced, then GELU + dropout.
Visual language (box style, arrows, palette) matches Fig. 2 for consistency.
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

PANEL_COLORS = {
    "geokan": ("#EFF3F8", "#5B7FA6", "#2E4D70"),
    "mlp":    ("#F8F3EC", "#B07B3E", "#7A4F1E"),
}

FIG_W_IN = 5.15
FIG_H_IN = FIG_W_IN / 1.16
fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=600)
ax.set_xlim(0, 100)
ax.set_ylim(-4, 71.5)
ax.axis("off")

def draw_box(cx, cy, w, h, title, subtitle, face, edge, title_color=TEXT_DARK,
             sub_color=TEXT_SUB, linewidth=1.0, title_fs=7.6, sub_fs=6.0):
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
                 fontsize=sub_fs, fontweight="semibold", color=sub_color, zorder=3,
                 linespacing=1.6)
    else:
        ax.text(cx, cy, subtitle, ha="center", va="center",
                 fontsize=sub_fs, fontweight="semibold", color=sub_color, zorder=3,
                 linespacing=1.6)

def draw_arrow(p1, p2, curve=0.0):
    arrow = FancyArrowPatch(
        p1, p2, connectionstyle=f"arc3,rad={curve}",
        arrowstyle="-|>,head_length=4.0,head_width=2.6",
        linewidth=1.1, color=ARROW_COL, zorder=1.5,
    )
    ax.add_patch(arrow)

# ── Shared input ──────────────────────────────────────────────────────────
draw_box(50, 65, 36, 11, "Input $u$", "batch-normalised feature vector",
          COL_NEUTRAL, COL_NEUTRAL_EDGE, title_fs=8.3, sub_fs=6.3)

PANEL_X = [27, 73]
PANEL_W = 38
LABELS  = ["(a) GeoKAN-Gamma", "(b) MLP baseline"]
PARAMS  = ["1,018,414 total parameters", "1,570,666 total parameters"]
KEYS    = ["geokan", "mlp"]
ROW_TOP    = 40
ROW_BOTTOM = 22

for px in PANEL_X:
    draw_arrow((50, 65 - 5.5), (px, 54))

for px, label, params, key in zip(PANEL_X, LABELS, PARAMS, KEYS):
    fill, edge, accent_text = PANEL_COLORS[key]
    ax.text(px, 52.5, label, ha="center", va="center", fontsize=7.6,
             fontweight="bold", color=accent_text, zorder=3)
    ax.text(px, 49.4, params, ha="center", va="center", fontsize=5.8,
             style="italic", color=TEXT_SUB, zorder=3)

# ── (a) GeoKAN-Gamma: metric warp, then fixed radial basis (from Fig. 2) ──
draw_box(PANEL_X[0], ROW_TOP, PANEL_W, 14, "Metric $g$",
          "$\\mathbf{g=\\mathrm{softplus}(\\gamma)}$\nfixed scalar per dim.\n$\\mathbf{z=u\\cdot\\sqrt{g}}$",
          *PANEL_COLORS["geokan"][:2], title_color=PANEL_COLORS["geokan"][2])
draw_box(PANEL_X[0], ROW_BOTTOM, PANEL_W, 14, "Basis Expansion",
          "$\\mathbf{\\phi_k=e^{-\\gamma_{rbf}(z-c_k)^2}}$\nradial basis, 12 fixed centres",
          *PANEL_COLORS["geokan"][:2], title_color=PANEL_COLORS["geokan"][2])

# ── (b) MLP baseline: direct learned expansion, no metric step ───────────
draw_box(PANEL_X[1], ROW_TOP, PANEL_W, 14, "Linear",
          "maps $u$ directly to the basis\nexpansion's output width",
          *PANEL_COLORS["mlp"][:2], title_color=PANEL_COLORS["mlp"][2])
draw_box(PANEL_X[1], ROW_BOTTOM, PANEL_W, 14, "GELU + Dropout",
          "no metric-warp step\nexpansion learned, unconstrained",
          *PANEL_COLORS["mlp"][:2], title_color=PANEL_COLORS["mlp"][2])

for px in PANEL_X:
    draw_arrow((px, ROW_TOP - 7.5), (px, ROW_BOTTOM + 7.5))

# Manifold merge into shared output, same pattern as Fig. 2/3.
BUS_Y = 13
for px in PANEL_X:
    ax.plot([px, px], [ROW_BOTTOM - 7, BUS_Y], color=ARROW_COL,
             linewidth=1.1, zorder=1.5, solid_capstyle="round")
ax.plot([PANEL_X[0], PANEL_X[1]], [BUS_Y, BUS_Y], color=ARROW_COL,
         linewidth=1.1, zorder=1.5, solid_capstyle="round")
draw_arrow((50, BUS_Y), (50, 10.2))

# ── Shared output ─────────────────────────────────────────────────────────
draw_box(50, 4, 56, 13, "Concat + Linear",
          "skip connection with $u$  →  layer output\n(identical wrapper for both)",
          COL_NEUTRAL, COL_OUTPUT_EDGE, linewidth=1.8, title_fs=8.3, sub_fs=6.1)

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
plt.savefig("figures/fig4_mlp_ablation.png", dpi=600, facecolor="white")
print("Saved figures/fig4_mlp_ablation.png")
