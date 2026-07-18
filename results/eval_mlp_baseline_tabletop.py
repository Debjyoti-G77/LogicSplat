"""
Evaluate the matched-budget MLP-head baseline (trained in train_mlp_baseline.py)
on the 8 custom tabletop scenes, reusing the exact same clustering, virtual-table,
Z-flip, feature-normalization, and symbolic-repair logic already verified for
GeoKAN-Gamma in eval_geokan_tabletop.py. Only the model is swapped.

Produces the same headline numbers as Table 9 (Section 6.2): aggregate Micro F1,
precision, recall, R@3, R@5, before and after symbolic repair, using the MLP
model's own tuned thresholds (trained on 3RScan val, same role as "3RScan
thresholds" for GeoKAN-Gamma).

Usage:
    python results/eval_mlp_baseline_tabletop.py
"""
import sys
sys.path.insert(0, ".")

import os
import json
import torch
import torch.nn as nn

import eval_geokan_tabletop as tt
from train_geokan_variants import GeoKANVariantGNN
from src.relations.schema import NUM_RELATIONS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/mlp_baseline_relation.pt"
RESULTS_JSON = "results/mlp_baseline_results.json"


class MLPCompareLayer(nn.Module):
    """Same definition as in train_mlp_baseline.py -- needed to load the checkpoint."""

    def __init__(self, in_dim: int, out_dim: int, n_bases: int = 12,
                 metric_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        expand_dim = in_dim * n_bases
        self.expand = nn.Sequential(
            nn.Linear(in_dim, expand_dim), nn.GELU(), nn.Dropout(dropout),
        )
        mix_in_dim = expand_dim + in_dim
        self.linear_mix = nn.Sequential(
            nn.Linear(mix_in_dim, out_dim), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, u):
        u_normed = self.bn(u)
        expanded = self.expand(u_normed)
        return self.linear_mix(torch.cat([expanded, u_normed], dim=-1))

    def metric_regularization(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)


with open(RESULTS_JSON) as f:
    mlp_saved = json.load(f)
thresholds = {int(k): v for k, v in mlp_saved["thresholds"].items()}

state = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
hidden_dim = state["node_encoder.0.weight"].shape[0]
edge_feat_dim = state["conv1.lin_edge.weight"].shape[1]
model = GeoKANVariantGNN(
    layer_cls=MLPCompareLayer, node_feat_dim=10, edge_feat_dim=edge_feat_dim,
    hidden_dim=hidden_dim, num_relations=NUM_RELATIONS,
).to(DEVICE)
model.load_state_dict(state, strict=True)
model.eval()
model.hidden_dim = hidden_dim
print(f"MLP-baseline model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# tt.run_inference / tt.normalize_edge_features use the module's own DEVICE constant
tt.DEVICE = DEVICE


def main():
    print("=" * 70)
    print("MLP-BASELINE Cross-Domain Evaluation: Tabletop Scenes 06-13")
    print("=" * 70)

    all_pred_before, all_pred_after, all_gt = set(), set(), set()
    all_scores = []
    offset = 0

    for scene_name in tt.SCENES:
        scene_dir = os.path.join(tt.DATA_DIR, scene_name)
        ply_path = os.path.join(scene_dir, "splat.ply")
        if not os.path.exists(ply_path):
            print(f"  SKIP {scene_name}: splat.ply not found")
            continue

        gt = tt.load_gt(scene_dir)
        n_objects_hint = len(gt["objects"])

        objects, _ = tt.cluster_scene(ply_path, n_objects_hint)
        if len(objects) < 2:
            print(f"  SKIP {scene_name}: too few clusters")
            continue

        is_z_down = tt.SCENE_Z_FLIP.get(scene_name, False)
        table_obj = tt.create_virtual_table(objects, is_z_down=is_z_down)
        has_table = table_obj is not None
        if has_table:
            table_obj.uid = len(objects)
            objects = list(objects) + [table_obj]

        gt_items_only = [o for o in gt["objects"] if o["name"] != "table"]
        gt_table_idx = next((i for i, o in enumerate(gt["objects"]) if o["name"] == "table"), None)
        n_match = len(objects) - (1 if has_table else 0)
        mapping = tt.hungarian_match(objects[:n_match], gt_items_only)
        if has_table and gt_table_idx is not None:
            mapping[len(objects) - 1] = {"gt_idx": gt_table_idx, "gt_name": "table", "distance": 0.0}

        if is_z_down:
            tt.apply_z_flip(objects)

        x, edge_index, edge_attr = tt.build_graph(objects)
        table_idx = (len(objects) - 1) if has_table else None
        pred_before, pred_scores = tt.run_inference(model, x, edge_index, edge_attr, thresholds, table_idx=table_idx)

        gt_set = tt.build_gt_relations_set(gt, mapping)
        pred_after, stats = tt.apply_symbolic_repair(pred_before, pred_scores)

        m_before = tt.compute_f1(pred_before, gt_set)
        m_after = tt.compute_f1(pred_after, gt_set)
        rk = tt.compute_recall_at_k(pred_scores, gt_set)
        print(f"  {scene_name}: GT={len(gt_set)}  F1 before={m_before['f1']:.3f}  "
              f"after={m_after['f1']:.3f}  R@3={rk[3]:.3f}  R@5={rk[5]:.3f}  "
              f"repair(removed={stats.relations_removed}, added={stats.relations_added})")

        # Offset indices for global aggregation (avoid cross-scene index collisions)
        def shift(s, off):
            return {(a + off, r, b + off) for (a, r, b) in s}
        def shift_scores(scores, off):
            return [(a + off, r, b + off, sc) for (a, r, b, sc) in scores]

        all_pred_before |= shift(pred_before, offset)
        all_pred_after |= shift(pred_after, offset)
        all_gt |= shift(gt_set, offset)
        all_scores.extend(shift_scores(pred_scores, offset))
        offset += len(objects) + 1000  # generous gap to avoid collisions

    print(f"\n{'=' * 70}")
    print("AGGREGATE RESULTS (8 tabletop scenes) -- MLP-baseline")
    print(f"{'=' * 70}")
    metrics_before = tt.compute_f1(all_pred_before, all_gt)
    metrics_after = tt.compute_f1(all_pred_after, all_gt)
    rk = tt.compute_recall_at_k(all_scores, all_gt)

    print(f"  Total GT triples: {len(all_gt)}")
    print(f"  Before repair: Micro F1={metrics_before['f1']:.4f}  "
          f"P={metrics_before['precision']:.4f}  R={metrics_before['recall']:.4f}  "
          f"R@3={rk[3]:.4f}  R@5={rk[5]:.4f}")
    print(f"  After repair:  Micro F1={metrics_after['f1']:.4f}  "
          f"P={metrics_after['precision']:.4f}  R={metrics_after['recall']:.4f}  "
          f"R@3={rk[3]:.4f}  R@5={rk[5]:.4f}")

    with open("results/mlp_baseline_tabletop_results.json", "w") as f:
        json.dump({
            "total_gt": len(all_gt),
            "before": metrics_before, "after": metrics_after,
            "r_at_3": rk[3], "r_at_5": rk[5],
        }, f, indent=2)
    print("\nSaved -> results/mlp_baseline_tabletop_results.json")


if __name__ == "__main__":
    main()
