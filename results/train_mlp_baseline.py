"""
Controlled ablation: GeoKAN head vs. a standard MLP head, with everything
else held identical — same 3RScan/3DSSG cache, same RIO10-excluded
480/85 scene split (seed=42), same augmentation, same loss (ASL + inverse-
consistency, metric-reg term is simply ~0 for the MLP since it has no
metric), same optimizer/schedule, same threshold-tuning protocol, same
evaluation (Macro F1, R@3, R@5).

MLPCompareLayer is a drop-in replacement for GeoKANGammaLayer / GeoKANWaveletLayer
(identical __init__/forward signature), so it plugs into the existing
GeoKANVariantGNN/GeoKANVariantHead wrapper unmodified. It replaces the
learnable-metric + fixed-basis-function expansion with a single learned
Linear(in_dim, in_dim*n_bases) of the same output width, keeping the same
skip connection and final linear_mix shape -- this gives the MLP MORE
learnable parameters in the expansion step than GeoKAN's metric (in_dim^2*n_bases
vs roughly 128*in_dim), erring in the MLP's favor on capacity rather than
against it.

Usage:
    python results/train_mlp_baseline.py
"""
import sys
sys.path.insert(0, ".")

import os
import json
import torch
import torch.nn as nn

from train_geokan_variants import (
    CONFIG, load_rio10_test_scene_ids, load_all_graphs, split_by_scene,
    augment_training, compute_class_weights, PyGGraphDataset,
    train_variant_full, evaluate, compute_recall_at_k, tune_thresholds,
    GeoKANVariantGNN,
)
from torch_geometric.loader import DataLoader
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES


class MLPCompareLayer(nn.Module):
    """Drop-in MLP replacement for a GeoKAN layer (same interface, no metric/basis)."""

    def __init__(self, in_dim: int, out_dim: int, n_bases: int = 12,
                 metric_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        expand_dim = in_dim * n_bases
        self.expand = nn.Sequential(
            nn.Linear(in_dim, expand_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        mix_in_dim = expand_dim + in_dim
        self.linear_mix = nn.Sequential(
            nn.Linear(mix_in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        u_normed = self.bn(u)
        expanded = self.expand(u_normed)
        features = torch.cat([expanded, u_normed], dim=-1)
        return self.linear_mix(features)

    def metric_regularization(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)


def main():
    device = CONFIG["device"] if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("CONTROLLED ABLATION: GeoKAN head vs. matched-budget MLP head")
    print("=" * 70)

    rio10_ids = load_rio10_test_scene_ids()
    graphs = load_all_graphs(CONFIG["cache_dir"], exclude_scene_ids=rio10_ids)
    train_graphs, val_graphs = split_by_scene(graphs, CONFIG["train_frac"])
    print(f"  Split: {len(train_graphs)} train / {len(val_graphs)} val (seed=42, matches GeoKAN runs)")

    train_graphs_aug = augment_training(
        train_graphs, CONFIG["augment_factor"], CONFIG["augment_jitter"]
    )
    pos_weight = compute_class_weights(train_graphs, cap=CONFIG["pos_weight_cap"]).to(device)

    train_dataset = PyGGraphDataset(train_graphs_aug)
    val_dataset = PyGGraphDataset(val_graphs)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    mlp_model = GeoKANVariantGNN(
        layer_cls=MLPCompareLayer,
        node_feat_dim=CONFIG["node_feat_dim"],
        edge_feat_dim=CONFIG["edge_feat_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_relations=NUM_RELATIONS,
        dropout=CONFIG["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in mlp_model.parameters())
    print(f"  MLP-head model: {n_params:,} parameters "
          f"(GeoKAN-Gamma=1,018,414 / RBF=1,071,918 / Wavelet=1,071,914 for reference)")

    mlp_save = os.path.join(CONFIG["save_dir"], "mlp_baseline_relation.pt")
    # Same default lr/warmup as RBF and Wavelet (no special-casing, unlike Gamma's
    # documented lower lr/longer warmup) -- the fairest, least-biased choice.
    _, mlp_model = train_variant_full(
        "MLP-baseline", mlp_model, train_loader, val_loader,
        pos_weight, device, mlp_save,
    )

    print(f"\n{'='*70}")
    print("FINAL EVALUATION (val set, tuned thresholds + constraints)")
    print(f"{'='*70}")

    thresholds = tune_thresholds(mlp_model, val_loader, device)
    macro_f1, per_rel, all_probs, all_labels = evaluate(
        mlp_model, val_loader, device, thresholds=thresholds, use_constraints=True
    )
    r_at_3 = compute_recall_at_k(all_probs, all_labels, 3)
    r_at_5 = compute_recall_at_k(all_probs, all_labels, 5)

    print(f"\n  MLP-baseline Results:")
    print(f"    Macro F1:  {macro_f1:.4f}")
    print(f"    R@3:       {r_at_3:.4f}")
    print(f"    R@5:       {r_at_5:.4f}")
    print(f"    Params:    {n_params:,}")
    print(f"    Per-relation F1:")
    for rel_name, f1_val in per_rel.items():
        print(f"      {rel_name:20s}  {f1_val:.3f}")

    result = {
        "macro_f1": macro_f1, "r_at_3": r_at_3, "r_at_5": r_at_5,
        "params": n_params, "per_relation_f1": per_rel,
        "thresholds": {str(k): v for k, v in thresholds.items()},
    }
    with open("results/mlp_baseline_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved -> results/mlp_baseline_results.json")


if __name__ == "__main__":
    main()
