"""
Validate ground truth relations for scenes 06-13.
Checks:
  1. JSON loads and has required keys
  2. All relation subjects/objects reference valid object names
  3. Symmetric relations have their inverse (adjacent_to, etc.)
  4. No duplicate relations
  5. Relation types are from the expected vocabulary
"""
import os, json

SCENES = ["scene_%02d" % i for i in range(6, 14)]

# Relations that must have an explicit inverse
SYMMETRIC = {"adjacent_to", "next_to"}

# Relations and their expected inverses
INVERSES = {
    "on_top_of":      "under",
    "under":          "on_top_of",
    "in_front_of":    "behind",
    "behind":         "in_front_of",
    "to_the_left_of": "to_the_right_of",
    "to_the_right_of":"to_the_left_of",
}

VALID_RELATIONS = {
    "on_top_of", "under", "higher_than", "lower_than",
    "in_front_of", "behind", "to_the_left_of", "to_the_right_of",
    "adjacent_to", "next_to", "supported_by", "supports",
}

all_ok = True

for s in SCENES:
    path = "D:/logicsplat_data/processed/%s/ground_truth_relations.json" % s
    issues = []

    if not os.path.exists(path):
        print("[MISSING] %s — no ground_truth_relations.json" % s)
        all_ok = False
        continue

    try:
        d = json.load(open(path))
    except Exception as e:
        print("[ERROR]   %s — JSON parse error: %s" % (s, e))
        all_ok = False
        continue

    # Check required keys
    if "objects" not in d:
        issues.append("missing 'objects' key")
    if "relations" not in d and "relationships" not in d:
        issues.append("missing 'relations'/'relationships' key")

    obj_names = {o["name"] for o in d.get("objects", [])}
    rels_key  = "relations" if "relations" in d else "relationships"
    rels      = d.get(rels_key, [])

    # Build relation set for lookup
    rel_set = set()
    for r in rels:
        subj = r.get("subject", r.get("object_a", ""))
        obj  = r.get("object",  r.get("object_b", ""))
        rel  = r.get("relation", r.get("predicate", r.get("type", "")))
        rel_set.add((subj, rel, obj))

        # Check objects exist
        if subj not in obj_names:
            issues.append("unknown subject '%s' in relation" % subj)
        if obj not in obj_names:
            issues.append("unknown object '%s' in relation" % obj)

        # Check relation vocabulary
        if rel not in VALID_RELATIONS:
            issues.append("unknown relation type '%s'" % rel)

    # Check duplicates
    if len(rel_set) != len(rels):
        issues.append("duplicate relations found (%d unique vs %d total)" % (len(rel_set), len(rels)))

    # Check inverses exist
    for (subj, rel, obj) in rel_set:
        if rel in INVERSES:
            inv = INVERSES[rel]
            if (obj, inv, subj) not in rel_set:
                issues.append("missing inverse: '%s %s %s' needs '%s %s %s'" % (
                    obj, inv, subj, obj, inv, subj))

        if rel in SYMMETRIC:
            if (obj, rel, subj) not in rel_set:
                issues.append("missing symmetric: '%s %s %s'" % (obj, rel, subj))

    n_obj = len(obj_names)
    n_rel = len(rels)

    if issues:
        all_ok = False
        print("[ISSUES]  %s (%d objects, %d relations):" % (s, n_obj, n_rel))
        for iss in issues:
            print("           - %s" % iss)
    else:
        print("[OK]      %s — %d objects, %d relations, all checks passed" % (s, n_obj, n_rel))

print()
print("Overall: %s" % ("ALL GOOD" if all_ok else "ISSUES FOUND — see above"))
