"""Quick sanity test for the symbolic repair module."""
import sys
sys.path.insert(0, ".")

from src.repair.symbolic_repair import SceneGraphRepair, compute_metrics

# Test case: contradictory predictions
predictions = [
    # Valid: router is on top of box
    ("router", "on_top_of", "box", 0.9),
    ("box", "under", "router", 0.85),
    # Contradiction: router is also under box (conflicts with on_top_of)
    ("router", "under", "box", 0.3),
    # Valid: router is higher than box
    ("router", "higher_than", "box", 0.8),
    # Contradiction: router is also lower than box
    ("router", "lower_than", "box", 0.2),
    # Asymmetry violation: both directions of higher_than
    ("bottle", "higher_than", "pen", 0.7),
    ("pen", "higher_than", "bottle", 0.4),
    # Symmetric: adjacent_to
    ("bottle", "adjacent_to", "watch", 0.6),
    # Directional
    ("router", "left_of", "bottle", 0.75),
    ("bottle", "right_of", "router", 0.7),
]

repairer = SceneGraphRepair(verbose=True)
repaired, stats = repairer.repair(predictions)

print(f"\n{'='*60}")
print(f"Input:  {stats.initial_count} relations")
print(f"Output: {stats.final_count} relations")
print(f"Removed: {stats.relations_removed}")
print(f"Added:   {stats.relations_added}")
print(f"Iterations: {stats.iterations}")
print(f"Violations by type: {stats.violations_by_type}")

print(f"\n{'='*60}")
print("Repaired relations:")
for s, r, o, c in sorted(repaired, key=lambda x: (-x[3], x[0], x[1])):
    print(f"  {s:>10} {r:<15} {o:<10} conf={c:.3f}")

# Verify no contradictions remain
print(f"\n{'='*60}")
print("Checking consistency...")
rel_set = {(s, r, o) for s, r, o, _ in repaired}

# Check: no mutual exclusions in same direction
from src.repair.symbolic_repair import MUTUAL_EXCLUSIONS, ASYMMETRIC_RELATIONS
issues = 0
for s, r, o, _ in repaired:
    for r1, r2 in MUTUAL_EXCLUSIONS:
        if r == r1 and (s, r2, o) in rel_set:
            print(f"  VIOLATION: {s} {r} {o} conflicts with {s} {r2} {o}")
            issues += 1
    if r in ASYMMETRIC_RELATIONS and (o, r, s) in rel_set:
        print(f"  VIOLATION: asymmetry {s} {r} {o} and {o} {r} {s}")
        issues += 1

if issues == 0:
    print("  All constraints satisfied!")
else:
    print(f"  {issues} violations remain!")

# Test metrics
gt = {("router", "on_top_of", "box"), ("box", "under", "router"),
      ("router", "higher_than", "box"), ("bottle", "higher_than", "pen")}
pred_before = {(s, r, o) for s, r, o, _ in predictions}
pred_after = {(s, r, o) for s, r, o, _ in repaired}

m_before = compute_metrics(pred_before, gt)
m_after = compute_metrics(pred_after, gt)
print(f"\nMetrics vs GT:")
print(f"  Before repair: P={m_before['precision']:.3f} R={m_before['recall']:.3f} F1={m_before['f1']:.3f}")
print(f"  After repair:  P={m_after['precision']:.3f} R={m_after['recall']:.3f} F1={m_after['f1']:.3f}")
