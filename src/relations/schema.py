"""
LogicSplat Relation Schema.

12 geometrically-derivable spatial relations, organized into 4 groups.
All relations are derived purely from 3D bounding boxes — no human annotation needed.

REMOVED from earlier version:
  - SURROUNDING (6): zero training samples in all datasets
  - OCCLUDES (13):   zero training samples in all datasets
  Both hurt macro F1 by adding classes the model never learns.
"""
from enum import IntEnum


class Relation(IntEnum):
    # ── Physical (how objects physically interact) ────────────────────────────
    ON_TOP_OF    = 0   # A's bottom rests on B's top surface
    UNDER        = 1   # A is below B (inverse of on_top_of)
    INSIDE       = 2   # A is spatially contained within B
    ATTACHED_TO  = 3   # A is fixed/connected to B (wall-mounted, built-in)
    HANGING_FROM = 4   # A is suspended from B above it

    # ── Proximity ─────────────────────────────────────────────────────────────
    ADJACENT_TO  = 5   # A and B are close, same surface level, not touching

    # ── Directional (relative position in space) ──────────────────────────────
    LEFT_OF      = 6   # A is to the left of B (X axis)
    RIGHT_OF     = 7   # A is to the right of B (X axis)
    IN_FRONT_OF  = 8   # A is closer to camera than B (Y axis)
    BEHIND       = 9   # A is further from camera than B

    # ── Comparative (relative size/height) ───────────────────────────────────
    HIGHER_THAN  = 10  # A's centroid is significantly above B's centroid
    LOWER_THAN   = 11  # A's centroid is significantly below B's centroid


NUM_RELATIONS = len(Relation)

RELATION_NAMES = {r.value: r.name.lower() for r in Relation}

RELATION_DESCRIPTIONS = {
    Relation.ON_TOP_OF:   "is resting on top of",
    Relation.UNDER:       "is underneath",
    Relation.INSIDE:      "is inside",
    Relation.ATTACHED_TO: "is attached to",
    Relation.HANGING_FROM:"is hanging from",
    Relation.ADJACENT_TO: "is adjacent to",
    Relation.LEFT_OF:     "is to the left of",
    Relation.RIGHT_OF:    "is to the right of",
    Relation.IN_FRONT_OF: "is in front of",
    Relation.BEHIND:      "is behind",
    Relation.HIGHER_THAN: "is higher than",
    Relation.LOWER_THAN:  "is lower than",
}

# Map from 3DSSG relation names to our schema
# Only spatial/physical relations — appearance relations are dropped
DSSG_TO_SCHEMA = {
    "standing on":     Relation.ON_TOP_OF,
    "supported by":    Relation.ON_TOP_OF,
    "lying on":        Relation.ON_TOP_OF,
    "hanging on":      Relation.HANGING_FROM,
    "inside":          Relation.INSIDE,
    "standing in":     Relation.INSIDE,
    "lying in":        Relation.INSIDE,
    "hanging in":      Relation.INSIDE,
    "attached to":     Relation.ATTACHED_TO,
    "connected to":    Relation.ATTACHED_TO,
    "build in":        Relation.ATTACHED_TO,
    "part of":         Relation.ATTACHED_TO,
    "close by":        Relation.ADJACENT_TO,
    "left":            Relation.LEFT_OF,
    "right":           Relation.RIGHT_OF,
    "front":           Relation.IN_FRONT_OF,
    "behind":          Relation.BEHIND,
    "higher than":     Relation.HIGHER_THAN,
    "lower than":      Relation.LOWER_THAN,
}
