import os, json
DATA_DIR = 'D:/logicsplat_data/processed'
for i in range(6, 14):
    scene = f'scene_{i:02d}'
    # Use backup if exists, else main file
    p = os.path.join(DATA_DIR, scene, 'ground_truth_relations_backup.json')
    if not os.path.exists(p):
        p = os.path.join(DATA_DIR, scene, 'ground_truth_relations.json')
    with open(p) as f:
        gt = json.load(f)
    items = [o['name'] for o in gt['objects'] if o['name'] != 'table']
    rels = [(r['subject'], r['relation'], r['object']) for r in gt['relations']]
    print(f'\n=== {scene} === objects: {items}')
    for s, r, o in rels:
        print(f'  {s} {r} {o}')
