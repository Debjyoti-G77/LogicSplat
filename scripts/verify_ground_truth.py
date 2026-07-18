"""
Verify ground truth relation files are complete and consistent.

For each scene, given the object positions and heights, compute ALL relations
that MUST exist and check them against the JSON file.
Reports missing relations and any suspicious ones.
"""
import json
import os
import sys
from itertools import permutations

# ── Scene definitions (from your verbal descriptions) ─────────────────────────
# position: (left_right, depth, height)
#   left_right: 0=far-left, 1=center-left, 2=center, 3=center-right, 4=far-right
#   depth:      0=back(near wall), 1=middle, 2=front(near camera)
#   height:     actual height value (higher number = taller)
# adjacent: list of pairs that are touching

SCENES = {
    "scene_06": {
        "objects": {
            "router":       {"lr": 0, "depth": 0, "height": 5},  # on box, NW
            "agaro_box":    {"lr": 0, "depth": 0, "height": 3},  # NW corner
            "water_bottle": {"lr": 2, "depth": 1, "height": 4},  # center
            "watch":        {"lr": 2, "depth": 1, "height": 1},  # center, adjacent to bottle
            "pen":          {"lr": 4, "depth": 2, "height": 0},  # SE corner, flat
        },
        "on_top_of": [("router", "agaro_box")],
        "adjacent":  [("water_bottle", "watch")],
    },
    "scene_07": {
        "objects": {
            "pen":          {"lr": 0, "depth": 1, "height": 0},  # far-left, flat
            "router":       {"lr": 1, "depth": 1, "height": 2},  # center-left
            "agaro_box":    {"lr": 2, "depth": 1, "height": 1},  # center-right (router > box confirmed)
            "water_bottle": {"lr": 4, "depth": 1, "height": 4},  # far-right, tallest
        },
        "on_top_of": [],
        "adjacent":  [],
    },
    "scene_08": {
        "objects": {
            "router":    {"lr": 0, "depth": 0, "height": 5},  # on box, NW
            "agaro_box": {"lr": 0, "depth": 0, "height": 3},  # NW corner
            "pen":       {"lr": 2, "depth": 1, "height": 0},  # center, flat
            "perfume":   {"lr": 3, "depth": 1, "height": 2},  # center-right
        },
        "on_top_of": [("router", "agaro_box")],
        "adjacent":  [],
    },
    "scene_09": {
        "objects": {
            "router":    {"lr": 0, "depth": 1, "height": 4},  # far-left, tallest
            "cream_tub": {"lr": 1, "depth": 1, "height": 2},  # center-left
            "perfume":   {"lr": 2, "depth": 1, "height": 3},  # center
            "watch":     {"lr": 4, "depth": 1, "height": 0},  # far-right, flat
        },
        "on_top_of": [],
        "adjacent":  [("router", "cream_tub"), ("cream_tub", "perfume"), ("router", "perfume")],
    },
    "scene_10": {
        "objects": {
            "router":    {"lr": 0, "depth": 1, "height": 5},  # on box, left
            "agaro_box": {"lr": 0, "depth": 1, "height": 3},  # left
            "cream_tub": {"lr": 2, "depth": 1, "height": 2},  # center
            "watch":     {"lr": 4, "depth": 1, "height": 0},  # right, flat
        },
        "on_top_of": [("router", "agaro_box")],
        "adjacent":  [],
    },
    "scene_11": {
        "objects": {
            "router":       {"lr": 0, "depth": 1, "height": 3},  # far-left
            "water_bottle": {"lr": 1, "depth": 1, "height": 4},  # center-left, tallest
            "cream_tub":    {"lr": 3, "depth": 1, "height": 2},  # center-right
            "watch":        {"lr": 4, "depth": 1, "height": 0},  # far-right, flat
        },
        "on_top_of": [],
        "adjacent":  [],
    },
    "scene_12": {
        "objects": {
            "router":       {"lr": 0, "depth": 0, "height": 5},  # on box, NW
            "agaro_box":    {"lr": 0, "depth": 0, "height": 3},  # NW, back
            "watch":        {"lr": 3, "depth": 0, "height": 0},  # back-right, same depth as box
            "water_bottle": {"lr": 2, "depth": 1, "height": 4},  # center, in front
        },
        "on_top_of": [("router", "agaro_box")],
        "adjacent":  [],
    },
    "scene_13": {
        "objects": {
            "router":       {"lr": 0, "depth": 1, "height": 5},  # on box, left
            "agaro_box":    {"lr": 0, "depth": 1, "height": 3},  # left
            "water_bottle": {"lr": 2, "depth": 1, "height": 4},  # center
            "perfume":      {"lr": 4, "depth": 1, "height": 3},  # right (same height as box)
        },
        "on_top_of": [("router", "agaro_box")],
        "adjacent":  [],
    },
}

HEIGHT_THRESHOLD = 0  # any height difference counts


def compute_expected_relations(scene_def):
    """Compute all relations that must exist given positions."""
    objs = scene_def["objects"]
    on_top_of = scene_def["on_top_of"]
    adjacent = scene_def["adjacent"]
    expected = set()

    names = list(objs.keys())

    for a, b in permutations(names, 2):
        oa, ob = objs[a], objs[b]

        # on_top_of / under
        if (a, b) in on_top_of:
            expected.add((a, "on_top_of", b))
            expected.add((b, "under", a))

        # adjacent_to (bidirectional)
        if (a, b) in adjacent or (b, a) in adjacent:
            expected.add((a, "adjacent_to", b))

        # left_right
        if oa["lr"] < ob["lr"]:
            expected.add((a, "to_the_left_of", b))
        elif oa["lr"] > ob["lr"]:
            expected.add((a, "to_the_right_of", b))

        # depth (in_front_of / behind)
        if oa["depth"] > ob["depth"]:
            expected.add((a, "in_front_of", b))
        elif oa["depth"] < ob["depth"]:
            expected.add((a, "behind", b))

        # height — include both higher_than AND lower_than
        if oa["height"] > ob["height"]:
            expected.add((a, "higher_than", b))
            expected.add((b, "lower_than", a))
        elif oa["height"] < ob["height"]:
            expected.add((a, "lower_than", b))
            expected.add((b, "higher_than", a))

    return expected


def load_json_relations(scene_id):
    path = f"D:/logicsplat_data/processed/{scene_id}/ground_truth_relations.json"
    with open(path) as f:
        data = json.load(f)
    actual = set()
    for r in data["relations"]:
        actual.add((r["subject"], r["relation"], r["object"]))
    return actual


print("=" * 70)
print("Ground Truth Completeness Verification")
print("=" * 70)

all_ok = True

for scene_id, scene_def in SCENES.items():
    expected = compute_expected_relations(scene_def)
    actual   = load_json_relations(scene_id)

    missing  = expected - actual
    extra    = actual - expected

    missing_filtered = missing  # check everything now including lower_than

    print(f"\n{scene_id}:")
    print(f"  Expected: {len(expected)} | In JSON: {len(actual)} | Missing: {len(missing_filtered)}")

    if missing_filtered:
        all_ok = False
        print(f"  MISSING RELATIONS:")
        for subj, rel, obj in sorted(missing_filtered):
            print(f"    {subj} → {rel} → {obj}")
    else:
        print(f"  OK — all expected relations present")

    # Check for suspicious extras (in JSON but not geometrically expected)
    suspicious = {r for r in extra
                  if r[1] not in ("lower_than",)  # lower_than is fine as explicit
                  and r[1] != "under"}  # under is fine
    if suspicious:
        print(f"  SUSPICIOUS (in JSON but not expected from positions):")
        for subj, rel, obj in sorted(suspicious):
            print(f"    {subj} → {rel} → {obj}")

print()
if all_ok:
    print("ALL SCENES: Complete and consistent.")
else:
    print("ISSUES FOUND — see above.")
