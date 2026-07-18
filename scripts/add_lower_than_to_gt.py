"""
Add lower_than as the inverse of every higher_than relation in all GT files.
Also adds under as inverse of on_top_of where missing.
"""
import json, os, glob

gt_files = glob.glob("D:/logicsplat_data/processed/scene_*/ground_truth_relations.json")

for path in sorted(gt_files):
    with open(path) as f:
        data = json.load(f)

    existing = {(r["subject"], r["relation"], r["object"]) for r in data["relations"]}
    to_add = []

    for r in data["relations"]:
        subj, rel, obj = r["subject"], r["relation"], r["object"]

        # higher_than → add lower_than inverse
        if rel == "higher_than":
            inv = (obj, "lower_than", subj)
            if inv not in existing:
                to_add.append({"subject": obj, "relation": "lower_than", "object": subj})
                existing.add(inv)

        # on_top_of → ensure under exists
        if rel == "on_top_of":
            inv = (obj, "under", subj)
            if inv not in existing:
                to_add.append({"subject": obj, "relation": "under", "object": subj})
                existing.add(inv)

    if to_add:
        data["relations"].extend(to_add)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        scene = os.path.basename(os.path.dirname(path))
        print(f"{scene}: added {len(to_add)} inverse relations "
              f"(total: {len(data['relations'])})")
    else:
        scene = os.path.basename(os.path.dirname(path))
        print(f"{scene}: already complete")
