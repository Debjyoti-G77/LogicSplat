"""
Add cluster centroids to ground_truth_relations.json files.

Uses the GT relation constraints (higher_than, on_top_of, left_of, etc.)
to find the cluster assignment that best satisfies the known relations.

Strategy:
  1. Run clustering to get cluster centroids
  2. For each possible assignment of clusters → GT objects:
     - Score it by how many GT relations are satisfied
  3. Pick the highest-scoring assignment
  4. Write centroids into GT JSON

For scenes where cluster count != GT object count, we use the
best partial assignment (only map clusters that have a GT object).

Usage:
    python scripts/add_gt_centroids.py
    python scripts/add_gt_centroids.py --scene scene_06
"""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

import os
import json
import argparse
import numpy as np
from itertools import permutations

from src.gaussian.loader import load_gaussian_ply, filter_gaussians
from src.gaussian.clustering import gaussian_to_objects

DATA_DIR = 'D:/logicsplat_data/processed'
SCENES = [f'scene_{i:02d}' for i in range(6, 14)]

# GT relation name → schema name
GT_MAP = {
    'to_the_left_of': 'left_of',
    'to_the_right_of': 'right_of',
    'on_top_of': 'on_top_of',
    'under': 'under',
    'higher_than': 'higher_than',
    'lower_than': 'lower_than',
    'in_front_of': 'in_front_of',
    'behind': 'behind',
    'adjacent_to': 'adjacent_to',
}


def check_relation(rel: str, a_centroid: np.ndarray, b_centroid: np.ndarray) -> bool:
    """
    Check if the geometric relation holds between two centroids.
    Returns True if the relation is geometrically satisfied.

    Coordinate system: x=left/right, y=front/back, z=up/down (inverted: more negative = higher)
    """
    dx = a_centroid[0] - b_centroid[0]
    dy = a_centroid[1] - b_centroid[1]
    dz = a_centroid[2] - b_centroid[2]  # negative dz = a is higher

    if rel == 'higher_than':
        return dz < -0.05   # a is higher (more negative z)
    elif rel == 'lower_than':
        return dz > 0.05
    elif rel == 'on_top_of':
        return dz < -0.05   # a is above b
    elif rel == 'under':
        return dz > 0.05    # a is below b
    elif rel == 'left_of':
        return dx < -0.05
    elif rel == 'right_of':
        return dx > 0.05
    elif rel == 'in_front_of':
        return dy < -0.05   # smaller y = closer to camera
    elif rel == 'behind':
        return dy > 0.05
    elif rel == 'adjacent_to':
        dist = np.linalg.norm(a_centroid[:2] - b_centroid[:2])
        return dist < 1.0   # within 1 unit
    return False


def score_assignment(assignment: dict, gt_relations: list, centroids: dict) -> int:
    """
    Score an assignment {gt_name: cluster_uid} by counting satisfied GT relations.
    centroids: {cluster_uid: np.ndarray}
    """
    score = 0
    for r in gt_relations:
        subj = r['subject']
        obj = r['object']
        rel = GT_MAP.get(r['relation'], r['relation'])

        if subj not in assignment or obj not in assignment:
            continue

        subj_uid = assignment[subj]
        obj_uid = assignment[obj]

        if subj_uid not in centroids or obj_uid not in centroids:
            continue

        if check_relation(rel, centroids[subj_uid], centroids[obj_uid]):
            score += 1

    return score


def find_best_assignment(clusters, gt_objects: list, gt_relations: list) -> dict:
    """
    Find the assignment {gt_name: cluster_uid} that maximises satisfied relations.

    For small N (≤8 objects), tries all permutations.
    For larger N, uses a greedy approach.
    """
    n_c = len(clusters)
    n_g = len(gt_objects)
    gt_names = [o['name'] for o in gt_objects]
    cluster_uids = [o.uid for o in clusters]
    centroids = {o.uid: o.centroid for o in clusters}

    if n_c == 0 or n_g == 0:
        return {}

    # If more clusters than GT objects, pick the n_g most-populated clusters
    if n_c > n_g:
        sorted_by_pts = sorted(clusters, key=lambda o: -o.point_count)
        clusters_to_use = sorted_by_pts[:n_g]
        cluster_uids = [o.uid for o in clusters_to_use]
    elif n_c < n_g:
        # Fewer clusters than GT objects — can only assign n_c objects
        gt_names = gt_names[:n_c]

    best_score = -1
    best_assignment = {}

    if len(cluster_uids) <= 8:
        # Try all permutations
        for perm in permutations(cluster_uids, len(gt_names)):
            assignment = {gt_names[i]: perm[i] for i in range(len(gt_names))}
            score = score_assignment(assignment, gt_relations, centroids)
            if score > best_score:
                best_score = score
                best_assignment = assignment
    else:
        # Greedy: assign by point count rank
        sorted_clusters = sorted(clusters, key=lambda o: -o.point_count)
        best_assignment = {gt_names[i]: sorted_clusters[i].uid
                          for i in range(len(gt_names))}
        best_score = score_assignment(best_assignment, gt_relations, centroids)

    return best_assignment, best_score


def add_centroids_to_scene(scene_id: str, dry_run: bool = False):
    gt_path = os.path.join(DATA_DIR, scene_id, 'ground_truth_relations.json')
    if not os.path.exists(gt_path):
        print(f"  [{scene_id}] No GT file — skipping")
        return

    with open(gt_path) as f:
        gt_data = json.load(f)

    gt_objects = gt_data.get('objects', [])
    gt_relations = gt_data.get('relations', [])

    print(f"\n{'='*60}")
    print(f"  {scene_id}  ({len(gt_objects)} GT objects, {len(gt_relations)} GT relations)")
    print(f"{'='*60}", flush=True)

    ply = os.path.join(DATA_DIR, scene_id, 'splat.ply')
    cloud = load_gaussian_ply(ply)
    cloud_f = filter_gaussians(cloud, opacity_threshold=0.1)
    clusters, params = gaussian_to_objects(cloud_f)

    print(f"  Clusters found: {len(clusters)}")
    for o in sorted(clusters, key=lambda x: x.uid):
        print(f"    uid={o.uid} pts={o.point_count:5d} "
              f"centroid=[{o.centroid[0]:6.3f}, {o.centroid[1]:6.3f}, {o.centroid[2]:6.3f}]")

    result = find_best_assignment(clusters, gt_objects, gt_relations)
    if isinstance(result, tuple):
        best_assignment, best_score = result
    else:
        best_assignment, best_score = result, 0

    n_relations = len(gt_relations)
    print(f"\n  Best assignment (score={best_score}/{n_relations} relations satisfied):")
    centroids = {o.uid: o.centroid for o in clusters}
    for gt_name, uid in best_assignment.items():
        c = centroids[uid]
        print(f"    {gt_name:20s} ← Cluster {uid} "
              f"[{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]")

    if dry_run:
        print(f"\n  [DRY RUN] Would write centroids to {gt_path}")
        return

    # Write centroids into GT objects
    for gt_obj in gt_objects:
        name = gt_obj['name']
        if name in best_assignment:
            uid = best_assignment[name]
            gt_obj['centroid'] = centroids[uid].tolist()

    with open(gt_path, 'w') as f:
        json.dump(gt_data, f, indent=2)
    print(f"\n  Saved to {gt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default=None)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    scenes = [args.scene] if args.scene else SCENES
    for scene in scenes:
        add_centroids_to_scene(scene, dry_run=args.dry_run)

    print("\nAll done.")


if __name__ == '__main__':
    main()
