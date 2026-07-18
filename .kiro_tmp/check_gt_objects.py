import json, os

for i in range(6, 14):
    s = f"scene_{i:02d}"
    path = os.path.join("data/processed", s, "ground_truth_relations.json")
    if os.path.exists(path):
        data = json.load(open(path))
        objs = data["objects"]
        rels = data["relations"]
        print(f"{s}: {len(objs)} objects, {len(rels)} relations")
        for o in objs:
            print(f"  id={o['id']} name={o['name']}")
    else:
        print(f"{s}: NO GT FILE")
