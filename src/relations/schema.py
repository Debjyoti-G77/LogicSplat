"""
LogicSplat Relation Schema.

10 spatial relations (reduced from 12).

REMOVED from earlier version:
  - SURROUNDING: zero training samples
  - OCCLUDES:    zero training samples
  - INSIDE (was 2):       only 262 positives / 0.038% rate — unlearnable
  - HANGING_FROM (was 4): only 974 positives / 0.14% rate, absent in tabletop scenes
  Removing these raises Macro F1 by eliminating classes that drag the mean down
  without contributing to scene understanding in robotics contexts.

LEGACY CACHE COMPATIBILITY:
  Old 12-dim label files (pre-v3) map to the new 10-dim layout via:
    OLD_LABEL_COLS = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11]
  i.e. drop old columns 2 (INSIDE) and 4 (HANGING_FROM).
"""
from enum import IntEnum


class Relation(IntEnum):
    # Physical
    ON_TOP_OF    = 0   # A's bottom rests on B's top surface
    UNDER        = 1   # A is below B (inverse of on_top_of)
    ATTACHED_TO  = 2   # A is fixed/connected to B (wall-mounted, built-in)

    # Proximity
    ADJACENT_TO  = 3   # A and B are close, same surface level, not touching

    # Directional
    LEFT_OF      = 4   # A is to the left of B (X axis)
    RIGHT_OF     = 5   # A is to the right of B (X axis)
    IN_FRONT_OF  = 6   # A is closer to camera than B (Y axis)
    BEHIND       = 7   # A is further from camera than B

    # Comparative height
    HIGHER_THAN  = 8   # A's centroid is significantly above B's centroid
    LOWER_THAN   = 9   # A's centroid is significantly below B's centroid


NUM_RELATIONS = len(Relation)  # 10

RELATION_NAMES = {r.value: r.name.lower() for r in Relation}

RELATION_DESCRIPTIONS = {
    Relation.ON_TOP_OF:   "is resting on top of",
    Relation.UNDER:       "is underneath",
    Relation.ATTACHED_TO: "is attached to",
    Relation.ADJACENT_TO: "is adjacent to",
    Relation.LEFT_OF:     "is to the left of",
    Relation.RIGHT_OF:    "is to the right of",
    Relation.IN_FRONT_OF: "is in front of",
    Relation.BEHIND:      "is behind",
    Relation.HIGHER_THAN: "is higher than",
    Relation.LOWER_THAN:  "is lower than",
}

# Columns to keep when converting legacy 12-dim labels to 10-dim.
# Drop old col 2 (INSIDE) and old col 4 (HANGING_FROM).
LEGACY_12_TO_10_COLS = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11]

# Map from 3DSSG relation names to our schema (inside/hanging_from removed)
DSSG_TO_SCHEMA = {
    "standing on":  Relation.ON_TOP_OF,
    "supported by": Relation.ON_TOP_OF,
    "lying on":     Relation.ON_TOP_OF,
    "attached to":  Relation.ATTACHED_TO,
    "connected to": Relation.ATTACHED_TO,
    "build in":     Relation.ATTACHED_TO,
    "part of":      Relation.ATTACHED_TO,
    "close by":     Relation.ADJACENT_TO,
    "left":         Relation.LEFT_OF,
    "right":        Relation.RIGHT_OF,
    "front":        Relation.IN_FRONT_OF,
    "behind":       Relation.BEHIND,
    "higher than":  Relation.HIGHER_THAN,
    "lower than":   Relation.LOWER_THAN,
}
