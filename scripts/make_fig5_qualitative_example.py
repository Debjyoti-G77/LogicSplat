"""
Generate Figure 5: predicted (post-repair) scene graph for scene_06, restricted
to the five physical objects (table excluded). Drawn from the real extracted
predictions in scripts/_scene06_predictions.json (see
extract_scene06_predictions.py).

The model predicts relations for all 10 object pairs (a fully-connected
graph), not just the 5 drawn here. This figure shows one representative,
verified-correct example of each relation TYPE that occurs in the scene
(on_top_of/under, higher_than/lower_than, left_of/right_of, in_front_of/behind,
adjacent_to -- 5 underlying types after collapsing each inverse pair into a
single edge), not every pair -- e.g. pen/watch also has a real left_of/right_of
prediction, but that type is already represented by the router/pen edge, so
it is intentionally omitted here to keep the diagram legible. Each shown edge
is drawn once in a single direction; the inverse is logically implied by the
repair step's inverse-completeness constraint, so drawing both directions
would just double every line and label with no added information.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

COL_NODE      = "#FAFAFB"
COL_NODE_EDGE = "#1E3A5F"
TEXT_DARK = "#1F2430"
TEXT_SUB  = "#454C59"

# Edge color by relation category (consistent, restrained palette)
COL_CONTACT = "#4F9389"     # on_top_of / under
COL_VERTICAL = "#C98A3E"    # higher_than / lower_than
COL_DIRECTIONAL = "#5B7FA6" # left_of / right_of / in_front_of / behind
COL_ADJACENT = "#A65B85"    # adjacent_to

FIG_W_IN = 5.15
FIG_H_IN = FIG_W_IN / 0.98
fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=600)
ax.set_xlim(0, 100)
ax.set_ylim(0, 102)
ax.axis("off")

# Node positions chosen so the 5 edges below don't cross each other --
# router fans out to agaro_box/watch/pen at three distinct angles,
# water_bottle sits right next to agaro_box (its only edge in this subset),
# and watch-agaro_box closes the loop without crossing the router fan.
NODES = {
    "router":       (48, 88),
    "agaro_box":    (76, 58),
    "water_bottle": (86, 24),
    "watch":        (48, 18),
    "pen":          (13, 44),
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

for name, label in [("router", "Router"), ("agaro_box", "Hair-dryer\nBox"),
                     ("watch", "Watch"), ("pen", "Pen"),
                     ("water_bottle", "Water\nBottle")]:
    draw_node(name, label)

# One directed edge per connected object pair (real post-repair predictions,
# all verified correct against GT) -- inverse-direction duplicates dropped.
EDGES = [
    ("router", "agaro_box", "on_top_of", COL_CONTACT),
    ("router", "watch", "higher_than", COL_VERTICAL),
    ("router", "pen", "left_of", COL_DIRECTIONAL),
    ("water_bottle", "agaro_box", "in_front_of", COL_DIRECTIONAL),
    ("watch", "agaro_box", "adjacent_to", COL_ADJACENT),
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
    # push the label to whichever perpendicular side sits farther from the
    # layout centroid, so it lands in open space rather than toward the
    # other nodes/edges clustered near the middle of the diagram
    off = 4.0
    cand_a = (mx + perp_x * off, my + perp_y * off)
    cand_b = (mx - perp_x * off, my - perp_y * off)
    da = (cand_a[0] - CENTROID_X) ** 2 + (cand_a[1] - CENTROID_Y) ** 2
    db = (cand_b[0] - CENTROID_X) ** 2 + (cand_b[1] - CENTROID_Y) ** 2
    lx, ly = cand_a if da > db else cand_b
    # rel strings are the raw snake_case schema identifiers (on_top_of, ...) --
    # display as natural-language text, not code, on the rendered figure
    ax.text(lx, ly, rel.replace("_", " "), ha="center", va="center", fontsize=6.8,
             color=color, fontweight="semibold", zorder=4,
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                       edgecolor="none", alpha=0.92))

ax.text(50, 4.5, "Predicted scene graph (post-repair), scene_06\n"
         "one verified-correct example of each relation type that occurs in this scene",
         ha="center", va="center", fontsize=6.6, color=TEXT_SUB,
         style="italic", linespacing=1.6)

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
plt.savefig("figures/fig5_qualitative_example.png", dpi=600, facecolor="white")
print("Saved figures/fig5_qualitative_example.png")
