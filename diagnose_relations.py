"""
Diagnose why in_front_of/behind/lower_than/right_of have low R@3.
For each weak relation type, show model scores and what beats them.
"""
import sys, os, json, warnings
sys.path.insert(0, '.')

import numpy as np
import torch
from collections import defaultdict

from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians, remove_table_background
from src.gaussian.clustering import gaussian_to_objects, extract_gaussian_node_features
from scripts.build_3rscan_graphs import extract_3rscan_edge_features
from geokan_gamma_relation import GeoKANGammaRelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, Relation
from scipy.optimize import linear_sum_assignment

# Reuse helpers from eval script
sys.path.insert(0, '.')
import eval_geokan_tabletop as E

DATA_DIR = "D:/logicsplat_data/processed"
SCENES = [f"scene_{i:02d}" for i in range(6, 14)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCENE_N = {f'scene_{i:02d}': (5 if i==6 else 4) for i in range(6,14)}

WEAK_RELS = {
    'in_front_of': int(Relation.IN_FRONT_OF),
    'behind': int(Relation.BEHIND),
    'lower_than': int(Relation.LOWER_THAN),
    'right_of': int(Relation.RIGHT_OF),
}


def main():
    model, thresholds = E.load_model()

    # Per-relation: list of (score_of_gt_rel, rank_of_gt_rel, top3_relations)
    rel_data = defaultdict(list)

    for scene_name in SCENES:
        n = SCENE_N[scene_name]
        ply = os.path.join(DATA_DIR, scene_name, 'splat.ply')
        gt_path = os.path.join(DATA_DIR, scene_name, 'ground_truth_relations.json')

        with open(gt_path) as f:
            gt = json.load(f)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cloud = load_gaussian_ply(ply)
            cloud = filter_gaussians(cloud, opacity_threshold=0.1)
            cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)
            cloud = remove_table_background(cloud)
            objs_raw, _ = gaussian_to_objects(cloud, target_min=n, target_max=n+3, n_exact=n)
            objects = list(objs_raw)

        gt_items = [o for o in gt['objects'] if o['name'] != 'table']
        mapping = E.hungarian_match(objects, gt_items)

        z_flip_needed = E.SCENE_Z_FLIP.get(scene_name, False)
        if z_flip_needed:
            E.apply_z_flip(objects)

        table_obj = E.create_virtual_table(objects, is_z_down=False)
        all_objects = objects + [table_obj]
        table_idx = len(objects)

        x, edge_index, edge_attr = E.build_graph(all_objects)
        _, pred_scores = E.run_inference(model, x, edge_index, edge_attr, thresholds, table_idx=table_idx)

        gt_set = E.build_gt_relations_set(gt, {**mapping, table_idx: {'gt_name': 'table'}})

        # Group pred scores by (src, dst)
        pair_scores = defaultdict(list)
        for src, rel_idx, dst, score in pred_scores:
            pair_scores[(src, dst)].append((score, rel_idx))
        for key in pair_scores:
            pair_scores[key].sort(reverse=True)

        # For each GT triple, compute rank of GT relation
        name_to_pred = {info['gt_name']: pi for pi, info in mapping.items()}
        name_to_pred['table'] = table_idx

        for rel in gt['relations']:
            rel_name = rel['relation']
            if rel_name not in E.GT_RELATION_MAP:
                continue
            rel_idx = E.GT_RELATION_MAP[rel_name]
            src_idx = name_to_pred.get(rel['subject'])
            dst_idx = name_to_pred.get(rel['object'])
            if src_idx is None or dst_idx is None:
                continue

            ranked = pair_scores.get((src_idx, dst_idx), [])
            rank = None
            gt_score = 0.0
            for r, (sc, ri) in enumerate(ranked):
                if ri == rel_idx:
                    rank = r + 1  # 1-indexed
                    gt_score = sc
                    break

            top3_names = [RELATION_NAMES.get(ri, str(ri)) for _, ri in ranked[:3]]
            if rank is not None:
                rel_data[rel_name].append((gt_score, rank, top3_names, scene_name, rel['subject'], rel['object']))

    # Report
    print("\n" + "="*70)
    print("RELATION DIAGNOSIS: Why is R@3 low for these relations?")
    print("="*70)

    for rel_name, rel_idx in WEAK_RELS.items():
        items = rel_data[rel_name]
        if not items:
            continue

        ranks = [r for _, r, _, _, _, _ in items]
        scores = [s for s, _, _, _, _, _ in items]
        top3_counts = defaultdict(int)
        for _, rank, top3, _, _, _ in items:
            for r in top3:
                top3_counts[r] += 1

        in_top1 = sum(1 for r in ranks if r == 1)
        in_top3 = sum(1 for r in ranks if r <= 3)
        in_top5 = sum(1 for r in ranks if r <= 5)
        total = len(items)

        print(f"\n{'-'*60}")
        print(f"  {rel_name.upper()}  (total GT instances: {total})")
        print(f"  R@1={in_top1/total:.3f}  R@3={in_top3/total:.3f}  R@5={in_top5/total:.3f}")
        print(f"  Mean score={np.mean(scores):.3f}  Mean rank={np.mean(ranks):.1f}")
        print(f"  Rank distribution: " + ", ".join(
            f"rank{r}:{sum(1 for x in ranks if x==r)}"
            for r in range(1, 8)
        ))
        print(f"  Relations in top-3 (freq >2):")
        for r, cnt in sorted(top3_counts.items(), key=lambda x: -x[1]):
            if cnt > 2:
                print(f"    {r}: appears {cnt}/{total*3} times")

        # Show worst cases (rank >= 5)
        bad = [(s,r,t,sc,sb,ob) for s,r,t,sc,sb,ob in items if r >= 5]
        if bad:
            print(f"  Sample bad cases (rank>=5):")
            for s, r, t, sc, sb, ob in bad[:4]:
                print(f"    {sc}: {sb} {rel_name} {ob} -> rank={r}, score={s:.3f}, top3={t}")

        # Check for consistent confusion with the opposite relation
        opposite = {'in_front_of': 'behind', 'behind': 'in_front_of',
                    'lower_than': 'higher_than', 'right_of': 'left_of'}
        opp = opposite.get(rel_name, '')
        if opp:
            opp_in_top3 = sum(1 for _, _, top3, _, _, _ in items if opp in top3)
            print(f"  Opposite '{opp}' appears in top-3: {opp_in_top3}/{total} times")


if __name__ == '__main__':
    main()
