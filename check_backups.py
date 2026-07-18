import os, json
DATA_DIR = 'D:/logicsplat_data/processed'
for i in range(6, 14):
    scene = f'scene_{i:02d}'
    backup = os.path.join(DATA_DIR, scene, 'ground_truth_relations_backup.json')
    if os.path.exists(backup):
        with open(backup) as f:
            gt = json.load(f)
        items = [o for o in gt['objects'] if o['name'] != 'table']
        print(f'{scene}:')
        for o in items:
            c = o['centroid']
            print(f"  {o['name']:15s}: [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]")
