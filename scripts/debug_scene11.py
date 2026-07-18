"""Debug scene_11 — why does it score F1=0.000?

DIAGNOSIS:
  - The original script used mode='hybrid' (GNN), which never predicts
    left_of / right_of / higher_than / lower_than.
  - Fixed: now runs mode='ensemble' so geometry rules fire.
  - Also adds debug prints showing derive_relations() output per pair,
    GEOMETRIC_RELATIONS set, and whether the geometry loop executes.
"""
import sys, warnings, json, numpy as np
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

# ── Monkey-patch derive_relations to add debug output ─────────────────────────
import src.relations.geometry as _geo_module
_orig_derive = _geo_module.derive_relations

def _debug_derive(a_min, a_max, b_min, b_max, **kwargs):
    result = _orig_derive(a_min, a_max, b_min, b_max, **kwargs)
    from src.relations.schema import RELATION_NAMES
    names = [RELATION_NAMES[int(r)] for r in result]
    a_c = (a_min + a_max) / 2
    b_c = (b_min + b_max) / 2
    print(f"    [derive_relations] A_z={a_c[2]:.3f} B_z={b_c[2]:.3f} "
          f"A_x={a_c[0]:.3f} B_x={b_c[0]:.3f} → {names}")
    return result

_geo_module.derive_relations = _debug_derive

from src.inference.gaussian_inference import run_inference
from src.relations.schema import RELATION_NAMES

# ── Run in ENSEMBLE mode (geometry rules for directional/vertical) ─────────────
print("=" * 60)
print("Running mode=ensemble (geometry rules for left_of/right_of/higher_than/lower_than)")
print("=" * 60)

GEOMETRIC_RELATIONS = {
    "higher_than", "lower_than",
    "left_of", "right_of",
    "in_front_of", "behind",
    "on_top_of", "under",
}
print(f"\nGEOMETRIC_RELATIONS = {sorted(GEOMETRIC_RELATIONS)}")
print(f"Geometry loop WILL execute for these relation types.\n")

result = run_inference(
    'D:/logicsplat_data/processed/scene_11/splat.ply',
    labeler='none',
    n_objects_hint=4,
    scene_dir='D:/logicsplat_data/processed/scene_11',
    mode='ensemble',   # FIX: was missing — defaulted to 'hybrid' (GNN only)
)

print('=== PREDICTED OBJECTS ===')
for o in result['objects']:
    print(f'  Obj{o.uid}: z={o.centroid[2]:.3f} xy=({o.centroid[0]:.2f},{o.centroid[1]:.2f})')

print()
print('=== PREDICTED RELATIONS ===')
for r in result['relations']:
    print(f'  Obj{r["subject_id"]} --[{r["relation"]}]--> Obj{r["object_id"]}')

print()
with open('D:/logicsplat_data/processed/scene_11/ground_truth_relations.json') as f:
    gt = json.load(f)

print('=== GT OBJECTS ===')
for o in gt['objects']:
    c = o.get('centroid', [0,0,0])
    print(f'  {o["name"]}: centroid=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})')

print()
print('=== GT RELATIONS (first 10) ===')
for r in gt['relations'][:10]:
    print(f'  {r["subject"]} --[{r["relation"]}]--> {r["object"]}')

print()
print('=== CLUSTER-TO-GT MATCHING ===')
# Show what the Hungarian matching would produce
from scipy.optimize import linear_sum_assignment
cluster_centroids = np.array([o.centroid for o in result['objects']])
gt_centroids = []
for o in gt['objects']:
    if 'centroid' in o:
        c = np.array(o['centroid'])
        c[2] *= -1  # Z-flip to match inference
        gt_centroids.append(c)
    else:
        gt_centroids.append(np.zeros(3))
gt_centroids = np.array(gt_centroids)

cost = np.zeros((len(result['objects']), len(gt['objects'])))
for i, cc in enumerate(cluster_centroids):
    for j, gc in enumerate(gt_centroids):
        cost[i, j] = np.linalg.norm(cc - gc)

row_ind, col_ind = linear_sum_assignment(cost)
for ci, gi in zip(row_ind, col_ind):
    name = gt['objects'][gi]['name']
    dist = cost[ci, gi]
    print(f'  Obj{ci} -> {name} (dist={dist:.3f})')
