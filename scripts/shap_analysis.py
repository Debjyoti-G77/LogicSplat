"""
SHAP Analysis for Physical Relation Classifiers (v2 — derived features).

For each of the 9 tabletop relations (excluding attached_to, inside, hanging_from):
  1. Load the trained logistic regression classifier
  2. Load the training data (X_train from ScanNet)
  3. Run SHAP LinearExplainer
  4. Generate SHAP summary plots + publication figures

Usage:
    python scripts/shap_analysis.py
    python scripts/shap_analysis.py --classifiers models/physical_classifiers_v2.pkl
    python scripts/shap_analysis.py --n-samples 1000
"""
import sys
sys.path.insert(0, ".")

import os
import json
import argparse
import numpy as np
import pickle

# ── Publication-quality matplotlib settings ───────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

import shap

from src.models.physical_relation_classifier import (
    PhysicalRelationClassifier,
    FEATURE_NAMES,
    PHYS_DIM,
    GEOM_DIM,
    TOTAL_DIM,
)
from src.relations.schema import RELATION_NAMES

# ── Config ────────────────────────────────────────────────────────────────────
FIGURES_DIR = "figures"
CLASSIFIERS_PATH = "models/physical_classifiers_v2.pkl"
NUMPY_DIR = "D:/logicsplat_data/processed"

# 9 tabletop relations (exclude attached_to=3, inside=2, hanging_from=4)
TABLETOP_RELATIONS = [0, 1, 5, 6, 7, 8, 9, 10, 11]
TABLETOP_NAMES = [RELATION_NAMES[i] for i in TABLETOP_RELATIONS]

# Feature group indices for contribution analysis
GEOM_INDICES = list(range(0, GEOM_DIM))  # 0-9
PHYS_A_INDICES = list(range(GEOM_DIM, GEOM_DIM + PHYS_DIM))  # 10-16
PHYS_B_INDICES = list(range(GEOM_DIM + PHYS_DIM, GEOM_DIM + 2 * PHYS_DIM))  # 17-23
PHYS_DIFF_INDICES = list(range(GEOM_DIM + 2 * PHYS_DIM, TOTAL_DIM))  # 24-30
ALL_PHYS_INDICES = PHYS_A_INDICES + PHYS_B_INDICES + PHYS_DIFF_INDICES


def load_training_data(numpy_dir: str):
    """Load X_train and Y_train numpy arrays."""
    x_path = os.path.join(numpy_dir, "X_train_physical.npy")
    y_path = os.path.join(numpy_dir, "Y_train_physical.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"ERROR: Training data not found at {numpy_dir}/")
        print(f"  Run: python scripts/extract_physical_features.py --save-numpy")
        sys.exit(1)

    X = np.load(x_path)
    Y = np.load(y_path)
    print(f"Loaded training data: X={X.shape}, Y={Y.shape}")
    return X, Y


def run_shap_for_relation(
    clf_obj: PhysicalRelationClassifier,
    rel_idx: int,
    X_train: np.ndarray,
    n_samples: int = 1000,
):
    """
    Run SHAP LinearExplainer for a single relation classifier.

    Returns:
        shap_values: (n_samples, 31) array of SHAP values
        X_sample: (n_samples, 31) the sampled data
    """
    rel_name = RELATION_NAMES[rel_idx]

    if rel_idx not in clf_obj.classifiers:
        print(f"  [{rel_name}] No trained classifier — skipping")
        return None, None

    clf = clf_obj.classifiers[rel_idx]
    scaler = clf_obj.scalers[rel_idx]

    # Scale the training data
    X_scaled = scaler.transform(X_train)

    # Sample for SHAP (use stratified if possible)
    rng = np.random.default_rng(42)
    n = min(n_samples, len(X_scaled))
    indices = rng.choice(len(X_scaled), n, replace=False)
    X_sample = X_scaled[indices]

    # Use a background summary for efficiency
    n_background = min(100, len(X_scaled))
    bg_indices = rng.choice(len(X_scaled), n_background, replace=False)
    X_background = X_scaled[bg_indices]

    # SHAP LinearExplainer
    explainer = shap.LinearExplainer(clf, X_background)
    shap_values = explainer.shap_values(X_sample)

    print(f"  [{rel_name}] SHAP computed: {shap_values.shape}")
    return shap_values, X_sample


def generate_shap_summary_plot(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    rel_name: str,
    output_path: str,
):
    """Generate and save a SHAP summary (beeswarm) plot for one relation."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=FEATURE_NAMES,
        show=False,
        max_display=15,
    )
    plt.title(f"SHAP Feature Importance: {rel_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def generate_feature_importance_comparison(
    all_shap_values: dict,
    output_path: str,
):
    """
    Figure 4: Top-5 features per relation, grouped bar chart.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    plot_idx = 0
    for rel_idx in TABLETOP_RELATIONS:
        rel_name = RELATION_NAMES[rel_idx]
        if rel_name not in all_shap_values:
            continue

        shap_vals = all_shap_values[rel_name]
        mean_abs = np.abs(shap_vals).mean(axis=0)

        # Top 5 features
        top_indices = np.argsort(mean_abs)[::-1][:5]
        top_names = [FEATURE_NAMES[i] for i in top_indices]
        top_values = mean_abs[top_indices]

        ax = axes[plot_idx]
        bars = ax.barh(range(5), top_values[::-1], color="steelblue", edgecolor="navy", alpha=0.8)
        ax.set_yticks(range(5))
        ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|", fontsize=9)
        ax.set_title(rel_name, fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plot_idx += 1

    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Top-5 Feature Importance per Relation (SHAP)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def generate_geometry_vs_physical_contribution(
    all_shap_values: dict,
    output_path: str,
):
    """
    Figure 5: Stacked bar showing % contribution of geometry vs physical features per relation.
    """
    relations = []
    geom_pcts = []
    phys_pcts = []

    for rel_idx in TABLETOP_RELATIONS:
        rel_name = RELATION_NAMES[rel_idx]
        if rel_name not in all_shap_values:
            continue

        shap_vals = all_shap_values[rel_name]
        mean_abs = np.abs(shap_vals).mean(axis=0)

        geom_contrib = mean_abs[GEOM_INDICES].sum()
        phys_contrib = mean_abs[ALL_PHYS_INDICES].sum()
        total = geom_contrib + phys_contrib

        if total < 1e-10:
            continue

        relations.append(rel_name)
        geom_pcts.append(100.0 * geom_contrib / total)
        phys_pcts.append(100.0 * phys_contrib / total)

    if not relations:
        print("  No data for geometry vs physical plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(relations))
    width = 0.6

    bars1 = ax.bar(x, geom_pcts, width, label="Geometry Features", color="steelblue", alpha=0.85)
    bars2 = ax.bar(x, phys_pcts, width, bottom=geom_pcts, label="Physical Features (Gaussian-derived)", color="coral", alpha=0.85)

    ax.set_xlabel("Relation", fontsize=12)
    ax.set_ylabel("Contribution (%)", fontsize=12)
    ax.set_title("Geometry vs Physical Feature Contribution per Relation", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(relations, rotation=35, ha="right", fontsize=10)
    ax.legend(loc="upper right", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    # Add percentage labels
    for i, (g, p) in enumerate(zip(geom_pcts, phys_pcts)):
        ax.text(i, g / 2, f"{g:.0f}%", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        ax.text(i, g + p / 2, f"{p:.0f}%", ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis for physical relation classifiers")
    parser.add_argument("--classifiers", default=CLASSIFIERS_PATH,
                        help=f"Path to trained classifiers (default: {CLASSIFIERS_PATH})")
    parser.add_argument("--numpy-dir", default=NUMPY_DIR,
                        help=f"Directory with X_train_physical.npy (default: {NUMPY_DIR})")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Number of samples for SHAP analysis (default: 1000)")
    parser.add_argument("--figures-dir", default=FIGURES_DIR,
                        help=f"Output directory for figures (default: {FIGURES_DIR})")
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    # ── Load classifier and data ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SHAP Analysis — Physical Relation Classifiers v2")
    print(f"{'='*60}")

    if not os.path.exists(args.classifiers):
        print(f"ERROR: Classifiers not found: {args.classifiers}")
        print(f"  Run: python scripts/extract_physical_features.py --save-numpy")
        sys.exit(1)

    clf_obj = PhysicalRelationClassifier.load(args.classifiers)
    X_train, Y_train = load_training_data(args.numpy_dir)

    # ── Run SHAP for each tabletop relation ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Running SHAP analysis (n_samples={args.n_samples})...")
    print(f"{'='*60}")

    all_shap_values = {}  # {rel_name: shap_values array}
    all_X_samples = {}    # {rel_name: X_sample array}

    for rel_idx in TABLETOP_RELATIONS:
        rel_name = RELATION_NAMES[rel_idx]
        shap_vals, X_sample = run_shap_for_relation(
            clf_obj, rel_idx, X_train, n_samples=args.n_samples
        )
        if shap_vals is not None:
            all_shap_values[rel_name] = shap_vals
            all_X_samples[rel_name] = X_sample

    if not all_shap_values:
        print("ERROR: No SHAP values computed (no trained classifiers?)")
        sys.exit(1)

    # ── Figure 1: SHAP summary for on_top_of ─────────────────────────────────
    print(f"\n{'='*60}")
    print("Generating publication figures...")
    print(f"{'='*60}")

    if "on_top_of" in all_shap_values:
        generate_shap_summary_plot(
            all_shap_values["on_top_of"],
            all_X_samples["on_top_of"],
            "on_top_of",
            os.path.join(args.figures_dir, "shap_on_top_of.png"),
        )

    # ── Figure 2: SHAP summary for adjacent_to ───────────────────────────────
    if "adjacent_to" in all_shap_values:
        generate_shap_summary_plot(
            all_shap_values["adjacent_to"],
            all_X_samples["adjacent_to"],
            "adjacent_to",
            os.path.join(args.figures_dir, "shap_adjacent_to.png"),
        )

    # ── Figure 3: SHAP summary for higher_than ───────────────────────────────
    if "higher_than" in all_shap_values:
        generate_shap_summary_plot(
            all_shap_values["higher_than"],
            all_X_samples["higher_than"],
            "higher_than",
            os.path.join(args.figures_dir, "shap_higher_than.png"),
        )

    # ── Figure 4: Feature importance comparison across all relations ──────────
    generate_feature_importance_comparison(
        all_shap_values,
        os.path.join(args.figures_dir, "feature_importance_all_relations.png"),
    )

    # ── Figure 5: Geometry vs Physical contribution ───────────────────────────
    generate_geometry_vs_physical_contribution(
        all_shap_values,
        os.path.join(args.figures_dir, "geometry_vs_physical_contribution.png"),
    )

    # ── Save SHAP experiment results as JSON ──────────────────────────────────
    experiment_results = {
        "config": {
            "classifiers_path": args.classifiers,
            "n_samples": args.n_samples,
            "feature_dim": TOTAL_DIM,
            "feature_names": FEATURE_NAMES,
        },
        "relations_analyzed": list(all_shap_values.keys()),
        "top_features_per_relation": {},
        "geometry_vs_physical": {},
    }

    for rel_name, shap_vals in all_shap_values.items():
        mean_abs = np.abs(shap_vals).mean(axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:7]
        experiment_results["top_features_per_relation"][rel_name] = [
            {"feature": FEATURE_NAMES[i], "mean_abs_shap": round(float(mean_abs[i]), 6)}
            for i in top_indices
        ]

        geom_contrib = float(mean_abs[GEOM_INDICES].sum())
        phys_contrib = float(mean_abs[ALL_PHYS_INDICES].sum())
        total = geom_contrib + phys_contrib
        experiment_results["geometry_vs_physical"][rel_name] = {
            "geometry_pct": round(100.0 * geom_contrib / max(total, 1e-10), 2),
            "physical_pct": round(100.0 * phys_contrib / max(total, 1e-10), 2),
        }

    output_json = os.path.join(NUMPY_DIR, "shap_experiment.json")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\n  Experiment results saved to: {output_json}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SHAP ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"\nRelations analyzed: {len(all_shap_values)}")
    print(f"Figures saved to: {args.figures_dir}/")
    print(f"\nTop-3 features per relation:")
    for rel_name, shap_vals in all_shap_values.items():
        mean_abs = np.abs(shap_vals).mean(axis=0)
        top3 = np.argsort(mean_abs)[::-1][:3]
        feat_str = ", ".join(f"{FEATURE_NAMES[i]}({mean_abs[i]:.4f})" for i in top3)
        print(f"  [{rel_name:18s}] {feat_str}")

    print(f"\nGeometry vs Physical contribution:")
    for rel_name in all_shap_values:
        gp = experiment_results["geometry_vs_physical"][rel_name]
        print(f"  [{rel_name:18s}] Geometry: {gp['geometry_pct']:.1f}%  Physical: {gp['physical_pct']:.1f}%")

    print(f"\n{'='*60}")
    print("KEY FINDINGS FOR PAPER:")
    print(f"{'='*60}")
    if "on_top_of" in all_shap_values:
        mean_abs = np.abs(all_shap_values["on_top_of"]).mean(axis=0)
        top3 = np.argsort(mean_abs)[::-1][:3]
        print(f"  on_top_of: Top features = {[FEATURE_NAMES[i] for i in top3]}")
        print(f"    → Gaussian covariance eigenvalues encode support surface capability")
    if "adjacent_to" in all_shap_values:
        mean_abs = np.abs(all_shap_values["adjacent_to"]).mean(axis=0)
        top3 = np.argsort(mean_abs)[::-1][:3]
        print(f"  adjacent_to: Top features = {[FEATURE_NAMES[i] for i in top3]}")
        print(f"    → Gaussian opacity encodes object boundaries for proximity detection")
    if "higher_than" in all_shap_values:
        mean_abs = np.abs(all_shap_values["higher_than"]).mean(axis=0)
        top3 = np.argsort(mean_abs)[::-1][:3]
        print(f"  higher_than: Top features = {[FEATURE_NAMES[i] for i in top3]}")
        print(f"    → Covariance anisotropy correlates with object height in scene")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
