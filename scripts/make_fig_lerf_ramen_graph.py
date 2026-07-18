"""
Generate the LERF-ramen counterpart to figures/fig5_qualitative_example.png:
predicted (post-repair) scene graph for lerf_ramen, restricted to six objects
chosen to show one representative, verified-correct example of each relation
category (contact, vertical, directional, adjacent, attached), drawn from the
real predictions in demo/data/lerf_ramen.json (relations_after, gt=="correct").

The model predicts relations for all 90 ordered object pairs (10 objects),
not just the 5 edges drawn here -- this figure shows one representative
example per relation TYPE that occurs in this scene, exactly as
scripts/make_fig5_qualitative_example.py does for the tabletop scene. Each
edge is drawn once in a single direction; the inverse is logically implied
by the repair step's inverse-completeness constraint.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

COL_NODE      = "#FAFAFB"
COL_NODE_EDGE = "#1E3A5F"
TEXT_DARK = "#1F2430"
TEXT_SUB  = "#454C59"

COL_CONTACT     = "#4F9389"  # on_top_of / under
COL_VERTICAL    = "#C98A3E"  # higher_than / lower_than
COL_DIRECTIONAL = "#5B7FA6"  # left_of / right_of / in_front_of / behind
COL_ADJACENT    = "#A65B85"  # adjacent_to
COL_ATTACHED    = "#8E6C3E"  # attached_to

FIG_W_IN = 5.15
FIG_H_IN = FIG_W_IN / 0.98
fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=600)
ax.set_xlim(0, 100)
ax.set_ylim(0, 102)
ax.axis("off")

NODES = {
    "bowl":            (35, 55),
    "chopsticks":      (65, 70),
    "plate":           (90, 50),
    "onion_segments":  (90, 14),
    "sake_cup":        (13, 25),
    "egg":             (13, 84),
}
NODE_R = 10.5

def draw_node(name, label):
    cx, cy = NODES[name]
    shadow = FancyBboxPatch(
        (cx - NODE_R + 0.4, cy - NODE_R - 0.4), NODE_R * 2, NODE_R * 2,
        boxstyle="round,pad=0,rounding_size=6", linewidth=0,
        facecolor="#1F2430", alpha=0.10, zorder=1,
    )
    ax.add_patch(shadow)
    box = FancyBboxPatch(
        (cx - NODE_R, cy - NODE_R), NODE_R * 2, NODE_R * 2,
        boxstyle="round,pad=0,rounding_size=6", linewidth=1.4,
        edgecolor=COL_NODE_EDGE, facecolor=COL_NODE, zorder=2,
    )
    ax.add_patch(box)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=8.6,
             fontweight="bold", color=TEXT_DARK, zorder=3, linespacing=1.2)

for name, label in [("bowl", "Bowl"), ("chopsticks", "Chopsticks"), ("plate", "Plate"),
                     ("onion_segments", "Onion\nSegments"), ("sake_cup", "Sake\nCup"),
                     ("egg", "Egg")]:
    draw_node(name, label)

# One directed edge per connected pair -- real post-repair predictions,
# all verified correct against ground truth (demo/data/lerf_ramen.json).
EDGES = [
    ("bowl", "chopsticks", "left_of", COL_DIRECTIONAL),
    ("bowl", "sake_cup", "adjacent_to", COL_ADJACENT),
    ("bowl", "egg", "lower_than", COL_VERTICAL),
    ("onion_segments", "plate", "on_top_of", COL_CONTACT),
    ("plate", "chopsticks", "attached_to", COL_ATTACHED),
]

CENTROID_X = sum(x for x, y in NODES.values()) / len(NODES)
CENTROID_Y = sum(y for x, y in NODES.values()) / len(NODES)

def edge_endpoint(src, dst):
    x1, y1 = NODES[src]
    x2, y2 = NODES[dst]
    dx, dy = x2 - x1, y2 - y1
    dist = (dx ** 2 + dy ** 2) ** 0.5
    ux, uy = dx / dist, dy / dist
    return (x1 + ux * NODE_R, y1 + uy * NODE_R), (x2 - ux * NODE_R, y2 - uy * NODE_R)

for src, dst, rel, color in EDGES:
    p1, p2 = edge_endpoint(src, dst)
    arrow = FancyArrowPatch(
        p1, p2, arrowstyle="-|>,head_length=4.0,head_width=2.6",
        linewidth=1.4, color=color, zorder=1.5,
    )
    ax.add_patch(arrow)
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    norm = (dx ** 2 + dy ** 2) ** 0.5
    perp_x, perp_y = -dy / norm, dx / norm
    off = 4.0
    cand_a = (mx + perp_x * off, my + perp_y * off)
    cand_b = (mx - perp_x * off, my - perp_y * off)
    da = (cand_a[0] - CENTROID_X) ** 2 + (cand_a[1] - CENTROID_Y) ** 2
    db = (cand_b[0] - CENTROID_X) ** 2 + (cand_b[1] - CENTROID_Y) ** 2
    lx, ly = cand_a if da > db else cand_b
    ax.text(lx, ly, rel.replace("_", " "), ha="center", va="center", fontsize=6.8,
             color=color, fontweight="semibold", zorder=4,
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                       edgecolor="none", alpha=0.92))

ax.text(50, 4.5, "Predicted scene graph (post-repair), LERF -- ramen\n"
         "one verified-correct example of each relation category, zero-shot, no fine-tuning",
         ha="center", va="center", fontsize=6.6, color=TEXT_SUB,
         style="italic", linespacing=1.6)

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
plt.savefig("figures/fig_qualitative_example_lerf_ramen.png", dpi=600, facecolor="white")
print("Saved figures/fig_qualitative_example_lerf_ramen.png")
