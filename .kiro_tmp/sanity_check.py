"""Sanity check on scene_06 after geometry fixes."""
import sys
sys.path.insert(0, ".")

from src.inference.gaussian_inference import run_inference

result = run_inference(
    'data/processed/scene_06/splat.ply',
    labeler='none',
    mode='ensemble',
    n_objects_hint=5,
    scene_dir='data/processed/scene_06'
)

rels = [(r['relation'], r['subject_id'], r['object_id']) for r in result['relations']]
print(f'\nPredicted {len(rels)} relations')
for r in sorted(rels):
    print(f'  {r}')

# Summary by type
from collections import Counter
counts = Counter(r[0] for r in rels)
print(f'\nRelation counts:')
for rel_type, count in sorted(counts.items()):
    print(f'  {rel_type}: {count}')
