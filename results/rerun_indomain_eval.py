"""
Fresh re-evaluation of GeoKAN-Gamma / RBF / Wavelet on the 3DSSG validation
split (85 scenes), using the exact same data split as training (seed=42,
train_frac=0.85, RIO10 excluded).

Reproduces report Tables 4.2 and 4.3 (per-relation F1, Macro F1, R@3, R@5).

Usage:
    python results/rerun_indomain_eval.py
"""
import sys
sys.path.insert(0, ".")

import json
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from train_geokan_variants import (
    CONFIG, load_rio10_test_scene_ids, load_all_graphs, split_by_scene,
    PyGGraphDataset, evaluate, compute_recall_at_k,
    GeoKANVariantGNN, GeoKANGammaLayer, GeoKANWaveletLayer,
)
from geokan_relation import GeoKANRelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# IMPORTANT: Gamma and Wavelet MUST use the INLINE layer classes from
# train_geokan_variants.py (the script that actually trained these checkpoints),
# NOT the standalone geokan_gamma_relation.py / geokan_wavelet_relation.py files.
# Verified via strict state_dict load: the standalone Wavelet class is missing
# `sigma` keys present in the real checkpoint's architecture and FAILS to load;
# the inline GeoKANVariantGNN(layer_cls=...) wrapper succeeds for both.
VARIANTS = {
    "GeoKAN-Gamma": {
        "ckpt": "models/geokan_relation_gamma.pt",
        "thresholds": "models/geokan_relation_geokan-gamma_thresholds.json",
        "build": lambda **kw: GeoKANVariantGNN(layer_cls=GeoKANGammaLayer, **kw),
    },
    "GeoKAN-RBF": {
        "ckpt": "models/geokan_relation_v4.pt",
        "thresholds": "models/geokan_relation_geokan-rbf_v4_thresholds.json",
        "build": lambda **kw: GeoKANRelationGNN(**kw),
    },
    "GeoKAN-Wavelet": {
        "ckpt": "models/geokan_relation_wavelet.pt",
        "thresholds": "models/geokan_relation_geokan-wavelet_thresholds.json",
        "build": lambda **kw: GeoKANVariantGNN(layer_cls=GeoKANWaveletLayer, **kw),
    },
}


def load_variant(name, info):
    state = torch.load(info["ckpt"], weights_only=False, map_location=DEVICE)
    hidden_dim = state["node_encoder.0.weight"].shape[0]
    edge_feat_dim = state["conv1.lin_edge.weight"].shape[1]
    model = info["build"](
        node_feat_dim=10, edge_feat_dim=edge_feat_dim,
        hidden_dim=hidden_dim, num_relations=NUM_RELATIONS,
    ).to(DEVICE)
    # strict=True deliberately — fail loudly if architecture doesn't match,
    # rather than silently loading a wrong/partial model.
    model.load_state_dict(state, strict=True)
    model.eval()
    with open(info["thresholds"]) as f:
        thresholds = {int(k): v for k, v in json.load(f).items()}
    return model, thresholds


def main():
    print("=" * 70)
    print("FRESH IN-DOMAIN RE-EVALUATION (3DSSG validation, 85 scenes)")
    print("=" * 70)

    rio10_ids = load_rio10_test_scene_ids()
    graphs = load_all_graphs(CONFIG["cache_dir"], exclude_scene_ids=rio10_ids)
    print(f"Loaded {len(graphs)} graphs total")

    train_graphs, val_graphs = split_by_scene(graphs, CONFIG["train_frac"])
    print(f"Split: {len(train_graphs)} train / {len(val_graphs)} val (seed=42, frac=0.85)")

    val_dataset = PyGGraphDataset(val_graphs)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    total_triples = sum(int((g["edge_label"] >= 0.5).sum()) for g in val_graphs)
    print(f"Total GT positive triples in val set: {total_triples}\n")

    results = {}
    for name, info in VARIANTS.items():
        print(f"--- {name} ---")
        model, thresholds = load_variant(name, info)
        macro_f1, per_rel_f1, all_probs, all_labels = evaluate(
            model, val_loader, DEVICE, thresholds=thresholds
        )
        r3 = compute_recall_at_k(all_probs, all_labels, k=3)
        r5 = compute_recall_at_k(all_probs, all_labels, k=5)
        n_params = sum(p.numel() for p in model.parameters())

        print(f"  Macro F1 = {macro_f1:.4f}")
        print(f"  R@3 = {r3:.4f}  R@5 = {r5:.4f}")
        print(f"  Params = {n_params:,}")
        for rel_name, f1 in per_rel_f1.items():
            print(f"    {rel_name:<15s} F1={f1:.4f}")
        print()

        results[name] = {
            "macro_f1": macro_f1,
            "r_at_3": r3,
            "r_at_5": r5,
            "params": n_params,
            "per_relation_f1": per_rel_f1,
        }

    with open("results/indomain_results_fresh.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved -> results/indomain_results_fresh.json")


if __name__ == "__main__":
    main()
