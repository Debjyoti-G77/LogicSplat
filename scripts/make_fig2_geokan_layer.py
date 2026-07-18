"""
Generate Figure 2 (GeoKAN layer internal computation), v4.

Back to the three-panel layout (preferred over the vertical-fork v3), with
the two real problems from v2 fixed without changing the overall design:
  1. The "12 fixed centres" row was identical across all three columns and
     redundant with the body text (Section 5.5 already states this) --
     deleted entirely rather than shown three times.
  2. RBF/Wavelet's metric formula and Gamma/RBF's basis formula are
     genuinely identical pairs -- each is now written out in full exactly
     once (in its first occurrence, left to right) and the duplicate column
     references back ("same as ...") instead of repeating the formula.
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
    "gamma":   ("#EFF3F8", "#5B7FA6", "#2E4D70"),
    "rbf":     ("#EFF7F3", "#5B9A7B", "#2C6147"),
    "wavelet": ("#F8F0F4", "#A65B85", "#732B52"),
}

FIG_W_IN = 5.15
FIG_H_IN = FIG_W_IN / 1.225
fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=600)
ax.set_xlim(0, 100)
ax.set_ylim(0, 71.5)
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
    ax.text(cx, cy + h * 0.23, title, ha="center", va="center",
             fontsize=title_fs, fontweight="bold", color=title_color,
             zorder=3, linespacing=1.3)
    ax.text(cx, cy - h * 0.16, subtitle, ha="center", va="center",
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

PANEL_X = [16.5, 50, 83.5]
PANEL_W = 29
LABELS  = ["(a) Gamma", "(b) RBF", "(c) Wavelet"]
KEYS    = ["gamma", "rbf", "wavelet"]
ROW_METRIC = 42
ROW_BASIS  = 22

for px in PANEL_X:
    draw_arrow((50, 65 - 5.5), (px, 51.5))

for px, label, key in zip(PANEL_X, LABELS, KEYS):
    fill, edge, accent_text = PANEL_COLORS[key]
    ax.text(px, 50, label, ha="center", va="center", fontsize=7.4,
             fontweight="bold", color=accent_text, zorder=3)

# ── Metric row: full formula written once (Gamma, then RBF); Wavelet refs back
draw_box(PANEL_X[0], ROW_METRIC, PANEL_W, 14, "Metric $g$",
          "$\\mathbf{g=\\mathrm{softplus}(\\gamma)}$\nfixed scalar per dim.\n$\\mathbf{z=u\\cdot\\sqrt{g}}$",
          *PANEL_COLORS["gamma"][:2], title_color=PANEL_COLORS["gamma"][2])
draw_box(PANEL_X[1], ROW_METRIC, PANEL_W, 14, "Metric $g$",
          "$\\mathbf{g=\\mathrm{softplus}}$\n$\\mathbf{(\\mathrm{MetricNet}(u))}$\n$\\mathbf{z=u\\cdot\\sqrt{g}}$",
          *PANEL_COLORS["rbf"][:2], title_color=PANEL_COLORS["rbf"][2])
draw_box(PANEL_X[2], ROW_METRIC, PANEL_W, 14, "Metric $g$",
          "(same as RBF)\ninput-conditioned\n$\\mathbf{z=u\\cdot\\sqrt{g}}$",
          *PANEL_COLORS["wavelet"][:2], title_color=PANEL_COLORS["wavelet"][2])

for px in PANEL_X:
    draw_arrow((px, ROW_METRIC - 7.5), (px, ROW_BASIS + 7.5))

# ── Basis row: full formula written once (Gamma, then refs); Wavelet distinct
draw_box(PANEL_X[0], ROW_BASIS, PANEL_W, 14, "Basis Expansion",
          "$\\mathbf{\\phi_k=e^{-\\gamma_{rbf}(z-c_k)^2}}$\nradial basis",
          *PANEL_COLORS["gamma"][:2], title_color=PANEL_COLORS["gamma"][2])
draw_box(PANEL_X[1], ROW_BASIS, PANEL_W, 14, "Basis Expansion",
          "(same as Gamma)\nradial basis",
          *PANEL_COLORS["rbf"][:2], title_color=PANEL_COLORS["rbf"][2])
draw_box(PANEL_X[2], ROW_BASIS, PANEL_W, 14, "Basis Expansion",
          "$\\mathbf{\\psi_k=(1-t^2)e^{-t^2/2}}$\nMexican hat",
          *PANEL_COLORS["wavelet"][:2], title_color=PANEL_COLORS["wavelet"][2])

# Manifold merge: each column drops straight down to a shared horizontal
# bus line, which then drops once into the output box -- avoids three
# shallow diagonal lines converging at a point (weak, hard-to-see arrowheads).
BUS_Y = 13
for px in PANEL_X:
    ax.plot([px, px], [ROW_BASIS - 7, BUS_Y], color=ARROW_COL,
             linewidth=1.1, zorder=1.5, solid_capstyle="round")
ax.plot([PANEL_X[0], PANEL_X[2]], [BUS_Y, BUS_Y], color=ARROW_COL,
         linewidth=1.1, zorder=1.5, solid_capstyle="round")
draw_arrow((50, BUS_Y), (50, 11.3))

# ── Shared output ─────────────────────────────────────────────────────────
draw_box(50, 6, 52, 11, "Concat + Linear",
          "skip connection with $u$  →  layer output",
          COL_NEUTRAL, COL_OUTPUT_EDGE, linewidth=1.8, title_fs=8.3, sub_fs=6.3)

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
plt.savefig("figures/fig2_geokan_layer.png", dpi=600, facecolor="white")
print("Saved figures/fig2_geokan_layer.png")
