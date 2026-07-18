"""Restore ground_truth_relations_backup.json from the original .zdown_backup files."""
import os, json, shutil

DATA_DIR = 'D:/logicsplat_data/processed'

for i in range(6, 14):
    scene = f'scene_{i:02d}'
    scene_dir = os.path.join(DATA_DIR, scene)
    zdown = os.path.join(scene_dir, 'ground_truth_relations.json.zdown_backup')
    backup = os.path.join(scene_dir, 'ground_truth_relations_backup.json')

    if os.path.exists(zdown):
        shutil.copy2(zdown, backup)
        with open(backup) as f:
            gt = json.load(f)
        items = [o for o in gt['objects'] if o['name'] != 'table']
        print(f'{scene}: restored {len(items)} objects from zdown_backup')
        for o in items:
            c = o['centroid']
            print(f"  {o['name']:15s}: [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]")
    else:
        print(f'{scene}: NO zdown_backup found!')
