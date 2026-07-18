import json, os
for i in range(6, 14):
    s = f"scene_{i:02d}"
    p = os.path.join("data/processed", s, "ground_truth_relations.json")
    if os.path.exists(p):
        d = json.load(open(p))
        print(f"{s}: {len(d['objects'])} objects, {len(d['relations'])} relations")
    else:
        print(f"{s}: NO GT FILE")
