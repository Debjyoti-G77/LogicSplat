import json
with open('data/processed/eval_hybrid.json') as f:
    d = json.load(f)

agg = d['aggregate']
print('=== HYBRID MODE RESULTS ===')
m = agg['micro']
print(f"Micro: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  (TP={m['tp']} FP={m['fp']} FN={m['fn']})")
ma = agg['macro']
print(f"Macro: P={ma['precision']:.3f}  R={ma['recall']:.3f}  F1={ma['f1']:.3f}")
print()
print(f"{'Scene':<12} {'GT':>5} {'Pred':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'F1':>6}")
print('-'*60)
for s in d['scenes']:
    print(f"{s['scene_id']:<12} {s['n_gt_triples']:>5} {s['n_pred_triples']:>5} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} {s['precision']:>6.3f} {s['recall']:>6.3f} {s['f1']:>6.3f}")
print()
print(f"{'Relation':<18} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'F1':>6}")
print('-'*55)
for rel, stats in agg['per_relation'].items():
    print(f"{rel:<18} {stats['tp']:>4} {stats['fp']:>4} {stats['fn']:>4} {stats['precision']:>6.3f} {stats['recall']:>6.3f} {stats['f1']:>6.3f}")
