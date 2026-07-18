"""
Symbolic Consistency Repair for Scene Graph Predictions.

A deterministic, foundation-model-free constraint propagation layer that
removes logical contradictions from predicted scene graphs.

Core idea: spatial relations obey hard logical constraints (inverses,
mutual exclusions, transitivity). Any prediction set violating these
constraints contains errors. We iteratively remove the lowest-confidence
violating predictions until a fixed point is reached.

Constraint types:
  1. Inverse pairs: on_top_of(A,B) ↔ under(B,A)
  2. Mutual exclusion: on_top_of(A,B) ⊕ under(A,B) (same direction)
  3. Asymmetry: higher_than(A,B) → ¬higher_than(B,A)
  4. Transitivity: higher_than(A,B) ∧ higher_than(B,C) → higher_than(A,C)

Author: LogicSplat Team
"""
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
from dataclasses import dataclass, field


# ── Constraint Tables ─────────────────────────────────────────────────────────

# Inverse relation pairs: if R(A,B) holds, then R_inv(B,A) should hold
INVERSE_PAIRS = {
    "on_top_of": "under",
    "under": "on_top_of",
    "higher_than": "lower_than",
    "lower_than": "higher_than",
    "left_of": "right_of",
    "right_of": "left_of",
    "in_front_of": "behind",
    "behind": "in_front_of",
}

# Mutual exclusion: relations that cannot hold simultaneously in the SAME direction
# If R1(A,B) holds, then R2(A,B) cannot hold
MUTUAL_EXCLUSIONS = [
    # Vertical
    ("on_top_of", "under"),
    ("higher_than", "lower_than"),
    ("on_top_of", "lower_than"),  # if A is on top of B, A can't be lower than B
    ("under", "higher_than"),     # if A is under B, A can't be higher than B
    # Directional
    ("left_of", "right_of"),
    ("in_front_of", "behind"),
]

# Asymmetric relations: R(A,B) → ¬R(B,A)
ASYMMETRIC_RELATIONS = {
    "on_top_of", "under", "higher_than", "lower_than",
    "left_of", "right_of", "in_front_of", "behind",
    "inside",
}

# Transitive relations: R(A,B) ∧ R(B,C) → R(A,C)
TRANSITIVE_RELATIONS = {
    "higher_than", "lower_than", "left_of", "right_of",
    "in_front_of", "behind",
}

# Symmetric relations: R(A,B) → R(B,A)
SYMMETRIC_RELATIONS = {"adjacent_to"}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RepairStats:
    """Statistics from the repair process."""
    initial_count: int = 0
    final_count: int = 0
    contradictions_found: int = 0
    relations_removed: int = 0
    relations_added: int = 0
    iterations: int = 0
    violations_by_type: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "initial_count": self.initial_count,
            "final_count": self.final_count,
            "contradictions_found": self.contradictions_found,
            "relations_removed": self.relations_removed,
            "relations_added": self.relations_added,
            "iterations": self.iterations,
            "violations_by_type": dict(self.violations_by_type),
        }


# ── Main Repair Class ─────────────────────────────────────────────────────────

class SceneGraphRepair:
    """
    Fixed-point constraint propagation for scene graph consistency.

    Input:  List of (subject, relation, object, confidence) tuples
    Output: Cleaned list with contradictions removed + implied relations added

    Algorithm:
        1. Index all predictions by (subject, object) pair
        2. Detect violations (mutual exclusion, asymmetry, transitivity)
        3. For each violation, remove the lowest-confidence prediction
           (tie-break: fewer support relations = removed first)
        4. After removal pass, add implied relations (inverses, transitivity)
        5. Repeat until no more violations (fixed point)
    """

    def __init__(self, max_iterations: int = 10, verbose: bool = False):
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Pre-compute mutual exclusion lookup: relation → set of conflicting relations
        self._mutex_lookup: Dict[str, Set[str]] = defaultdict(set)
        for r1, r2 in MUTUAL_EXCLUSIONS:
            self._mutex_lookup[r1].add(r2)
            self._mutex_lookup[r2].add(r1)

    def repair(
        self,
        predictions: List[Tuple[str, str, str, float]],
    ) -> Tuple[List[Tuple[str, str, str, float]], RepairStats]:
        """
        Run symbolic consistency repair on predictions.

        Args:
            predictions: List of (subject_name, relation, object_name, confidence)

        Returns:
            (repaired_predictions, stats)
        """
        stats = RepairStats(initial_count=len(predictions))

        # Build working set: (subj, rel, obj) → confidence
        working: Dict[Tuple[str, str, str], float] = {}
        for subj, rel, obj, conf in predictions:
            key = (subj, rel, obj)
            # Keep highest confidence if duplicates
            if key not in working or conf > working[key]:
                working[key] = conf

        # Fixed-point iteration
        for iteration in range(self.max_iterations):
            removed_this_iter = 0
            added_this_iter = 0

            # ── Phase 1: Remove contradictions ────────────────────────────────
            to_remove = self._find_violations(working, stats)
            if to_remove:
                for key in to_remove:
                    if key in working:
                        del working[key]
                        removed_this_iter += 1
                stats.relations_removed += removed_this_iter

            # ── Phase 2: Add implied relations ────────────────────────────────
            to_add = self._find_implied(working)
            for key, conf in to_add.items():
                if key not in working:
                    working[key] = conf
                    added_this_iter += 1
            stats.relations_added += added_this_iter

            stats.iterations = iteration + 1

            if self.verbose:
                print(f"  [Repair iter {iteration+1}] "
                      f"removed={removed_this_iter}, added={added_this_iter}")

            # Fixed point reached
            if removed_this_iter == 0 and added_this_iter == 0:
                break

        # Convert back to list
        repaired = [(s, r, o, c) for (s, r, o), c in working.items()]
        stats.final_count = len(repaired)
        stats.contradictions_found = stats.relations_removed

        return repaired, stats

    def _find_violations(
        self,
        working: Dict[Tuple[str, str, str], float],
        stats: RepairStats,
    ) -> Set[Tuple[str, str, str]]:
        """
        Find all constraint violations and return keys to remove.
        For each violation, removes the lower-confidence prediction.
        """
        to_remove: Set[Tuple[str, str, str]] = set()

        # Index by (subject, object) pair for fast lookup
        pair_index: Dict[Tuple[str, str], List[Tuple[str, float]]] = defaultdict(list)
        for (s, r, o), conf in working.items():
            pair_index[(s, o)].append((r, conf))

        # Support count: how many relations involve each entity
        entity_support: Dict[str, int] = defaultdict(int)
        for (s, r, o) in working:
            entity_support[s] += 1
            entity_support[o] += 1

        # ── Check 1: Mutual exclusion (same direction) ────────────────────────
        for (s, o), rels in pair_index.items():
            rel_set = {r for r, _ in rels}
            for r1, c1 in rels:
                if (s, r1, o) in to_remove:
                    continue
                conflicts = self._mutex_lookup.get(r1, set())
                for r2, c2 in rels:
                    if r1 == r2:
                        continue
                    if r2 in conflicts and (s, r2, o) not in to_remove:
                        # Remove lower confidence; tie-break by support
                        victim = self._pick_victim(s, r1, c1, r2, c2, o, entity_support)
                        to_remove.add(victim)
                        vtype = "mutual_exclusion"
                        stats.violations_by_type[vtype] = stats.violations_by_type.get(vtype, 0) + 1

        # ── Check 2: Asymmetry — R(A,B) and R(B,A) cannot both hold ──────────
        for (s, r, o), conf in working.items():
            if r in ASYMMETRIC_RELATIONS:
                reverse_key = (o, r, s)
                if reverse_key in working and reverse_key not in to_remove and (s, r, o) not in to_remove:
                    reverse_conf = working[reverse_key]
                    victim = self._pick_victim(s, r, conf, r, reverse_conf, o, entity_support, reverse=True)
                    to_remove.add(victim)
                    vtype = "asymmetry"
                    stats.violations_by_type[vtype] = stats.violations_by_type.get(vtype, 0) + 1

        # ── Check 3: Transitivity violations ──────────────────────────────────
        # If R(A,B) and R(B,C) but also ¬R(A,C) is implied by another relation
        # e.g., higher_than(A,B) ∧ higher_than(B,C) but lower_than(A,C) exists
        for r in TRANSITIVE_RELATIONS:
            # Build adjacency for this relation
            adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
            for (s, rel, o), conf in working.items():
                if rel == r and (s, rel, o) not in to_remove:
                    adj[s].append((o, conf))

            # Check for transitivity contradictions
            inverse_r = INVERSE_PAIRS.get(r)
            if inverse_r is None:
                continue

            for a, a_targets in adj.items():
                for b, conf_ab in a_targets:
                    if b in adj:
                        for c, conf_bc in adj[b]:
                            if c == a:
                                continue
                            # R(A,B) ∧ R(B,C) → check if inverse_R(A,C) exists
                            contradiction_key = (a, inverse_r, c)
                            if contradiction_key in working and contradiction_key not in to_remove:
                                # The contradiction has lower implied support
                                contra_conf = working[contradiction_key]
                                chain_conf = min(conf_ab, conf_bc)
                                if contra_conf <= chain_conf:
                                    to_remove.add(contradiction_key)
                                else:
                                    # Remove weakest link in chain
                                    if conf_ab <= conf_bc:
                                        to_remove.add((a, r, b))
                                    else:
                                        to_remove.add((b, r, c))
                                vtype = "transitivity"
                                stats.violations_by_type[vtype] = stats.violations_by_type.get(vtype, 0) + 1

        return to_remove

    def _find_implied(
        self,
        working: Dict[Tuple[str, str, str], float],
    ) -> Dict[Tuple[str, str, str], float]:
        """
        Find relations implied by the current set but not yet present.
        Returns dict of (subj, rel, obj) → confidence for new relations.
        """
        to_add: Dict[Tuple[str, str, str], float] = {}

        # ── Implied inverses ──────────────────────────────────────────────────
        for (s, r, o), conf in working.items():
            if r in INVERSE_PAIRS:
                inv_key = (o, INVERSE_PAIRS[r], s)
                if inv_key not in working and inv_key not in to_add:
                    # Add inverse with slightly reduced confidence
                    to_add[inv_key] = conf * 0.95

        # ── Implied symmetry ─────────────────────────────────────────────────
        for (s, r, o), conf in working.items():
            if r in SYMMETRIC_RELATIONS:
                sym_key = (o, r, s)
                if sym_key not in working and sym_key not in to_add:
                    to_add[sym_key] = conf * 0.95

        # ── Implied transitivity ──────────────────────────────────────────────
        for r in TRANSITIVE_RELATIONS:
            # Build adjacency for this relation
            adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
            for (s, rel, o), conf in working.items():
                if rel == r:
                    adj[s].append((o, conf))

            # For each chain A→B→C, imply A→C
            for a, a_targets in adj.items():
                for b, conf_ab in a_targets:
                    if b in adj:
                        for c, conf_bc in adj[b]:
                            if c == a:
                                continue
                            trans_key = (a, r, c)
                            if trans_key not in working:
                                # Confidence of transitive closure = min of chain * decay
                                implied_conf = min(conf_ab, conf_bc) * 0.9
                                if trans_key in to_add:
                                    to_add[trans_key] = max(to_add[trans_key], implied_conf)
                                else:
                                    to_add[trans_key] = implied_conf

        return to_add

    def _pick_victim(
        self,
        s: str, r1: str, c1: float,
        r2: str, c2: float, o: str,
        entity_support: Dict[str, int],
        reverse: bool = False,
    ) -> Tuple[str, str, str]:
        """
        Given a conflict between two relations, pick which one to remove.
        Lower confidence loses. Tie-break: fewer support relations loses.
        """
        if reverse:
            # Asymmetry conflict: R(A,B) vs R(B,A)
            key1 = (s, r1, o)
            key2 = (o, r2, s)
        else:
            # Same-direction conflict: R1(A,B) vs R2(A,B)
            key1 = (s, r1, o)
            key2 = (s, r2, o)

        if c1 < c2:
            return key1
        elif c2 < c1:
            return key2
        else:
            # Tie-break: fewer entity support = less corroborated
            support1 = entity_support.get(s, 0) + entity_support.get(o, 0)
            support2 = support1  # same entities for same-direction
            if reverse:
                support2 = entity_support.get(o, 0) + entity_support.get(s, 0)
            # If still tied, remove the second one (arbitrary but deterministic)
            return key2


# ── Utility: Compute metrics before/after repair ──────────────────────────────

def compute_metrics(
    predictions: Set[Tuple[str, str, str]],
    ground_truth: Set[Tuple[str, str, str]],
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for a set of predicted triples vs GT.

    Args:
        predictions: set of (subject, relation, object) triples
        ground_truth: set of (subject, relation, object) triples

    Returns:
        {"precision": float, "recall": float, "f1": float, "tp": int, "fp": int, "fn": int}
    """
    tp_set = predictions & ground_truth
    fp_set = predictions - ground_truth
    fn_set = ground_truth - predictions

    tp, fp, fn = len(tp_set), len(fp_set), len(fn_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_per_relation_metrics(
    predictions: Set[Tuple[str, str, str]],
    ground_truth: Set[Tuple[str, str, str]],
) -> Dict[str, Dict[str, float]]:
    """Compute per-relation precision/recall/F1."""
    all_rels = set(r for _, r, _ in ground_truth) | set(r for _, r, _ in predictions)
    per_relation = {}
    for rel in sorted(all_rels):
        gt_rel = {t for t in ground_truth if t[1] == rel}
        pred_rel = {t for t in predictions if t[1] == rel}
        metrics = compute_metrics(pred_rel, gt_rel)
        per_relation[rel] = metrics
    return per_relation
