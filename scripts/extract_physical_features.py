"""
Extract physical features from ScanNet v4 cache and train physical classifiers (v2).

v2: Uses 7-dim derived physical features (planarity, elongation, sphericity,
    anisotropy, support_likelihood, opacity, density) instead of raw eigenvalue ratios.
    Total feature vector: 31-dim (10 geometry + 7 phys_A + 7 phys_B + 7 phys_diff).

Tasks 3 + 4: Extract training data from ScanNet cache, train 12 binary
logistic regression classifiers (one per relation), save to models/.

Usage:
    python scripts/extract_physical_features.py
    python scripts/extract_physical_features.py --cache-dir data/scannet_cache
    python scripts/extract_physical_features.py --sample-frac 0.5 --seed 42
"""
import sys
sys.path.insert(0, ".")

import os
import glob
import time
import argparse
import numpy as np
import torch

from src.models.physical_relation_classifier import (
    PhysicalRelationClassifier,
    extract_physical_features_from_node,
    build_full_feature_vector,
    TOTAL_DIM,
)
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES

CACHE_DIR    = "D:/logicsplat_data/scannet_cache"
OUTPUT_PATH  = "models/physical_classifiers_v2.pkl"
NUMPY_DIR    = "D:/logicsplat_data/processed"


def load_cache_file(path: str):
    """Load a single v4 cache file. Returns dict or None on error."""
    try:
        data = torch.load(path, weights_only=False, map_location="cpu")
        return data
    except Exception as e:
        print(f"  WARN: failed to load {path}: {e}")
        return None


def extract_features_from_graph(data: dict):
    """
    Extract (X, Y) from a single graph cache dict.

    Returns:
        X: (E, 31) float32 feature matrix
        Y: (E, 12) float32 label matrix
    """
    x          = data["x"]           # (N, 10) node features
    edge_index = data["edge_index"]  # (2, E)
    edge_attr  = data["edge_attr"]   # (E, 10) geometry features
    edge_label = data["edge_label"]  # (E, 12) multi-hot labels

    if x is None or edge_index is None or edge_attr is None or edge_label is None:
        return None, None

    # Convert to numpy
    x_np    = x.numpy().astype(np.float32)          # (N, 10)
    ea_np   = edge_attr.numpy().astype(np.float32)  # (E, 10)
    el_np   = edge_label.numpy().astype(np.float32) # (E, 12)
    src_arr = edge_index[0].numpy()                  # (E,)
    dst_arr = edge_index[1].numpy()                  # (E,)

    E = len(src_arr)
    if E == 0:
        return None, None

    X = np.zeros((E, TOTAL_DIM), dtype=np.float32)
    Y = el_np  # (E, 12)

    for e in range(E):
        i = int(src_arr[e])
        j = int(dst_arr[e])

        phys_i = extract_physical_features_from_node(x_np[i])
        phys_j = extract_physical_features_from_node(x_np[j])
        edge_f = ea_np[e]  # (10,)

        X[e] = build_full_feature_vector(edge_f, phys_i, phys_j)

    return X, Y


def main():
    parser = argparse.ArgumentParser(
        description="Extract physical features from ScanNet cache and train classifiers"
    )
    parser.add_argument("--cache-dir", default=CACHE_DIR,
                        help=f"ScanNet cache directory (default: {CACHE_DIR})")
    parser.add_argument("--output", default=OUTPUT_PATH,
                        help=f"Output path for trained classifiers (default: {OUTPUT_PATH})")
    parser.add_argument("--sample-frac", type=float, default=0.5,
                        help="Fraction of cache files to use (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--save-numpy", action="store_true",
                        help="Also save X_train.npy and Y_train.npy to data/processed/")
    args = parser.parse_args()

    # ── Discover cache files ──────────────────────────────────────────────────
    pattern = os.path.join(args.cache_dir, "*_v4_axisalign.pt")
    all_files = sorted(glob.glob(pattern))

    if not all_files:
        print(f"ERROR: No v4_axisalign.pt files found in {args.cache_dir}")
        sys.exit(1)

    print(f"Found {len(all_files)} cache files in {args.cache_dir}")

    # ── Sample 50% ───────────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    n_sample = max(1, int(len(all_files) * args.sample_frac))
    selected = rng.choice(len(all_files), n_sample, replace=False)
    selected_files = [all_files[i] for i in sorted(selected)]

    print(f"Using {len(selected_files)} files ({args.sample_frac*100:.0f}% sample, seed={args.seed})")

    # ── Extract features ──────────────────────────────────────────────────────
    X_list = []
    Y_list = []
    t0 = time.time()
    n_edges_total = 0
    n_files_ok = 0

    for idx, fpath in enumerate(selected_files):
        if idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx:4d}/{len(selected_files)}] edges so far: {n_edges_total:,}  "
                  f"({elapsed:.0f}s)")

        data = load_cache_file(fpath)
        if data is None:
            continue

        X, Y = extract_features_from_graph(data)
        if X is None:
            continue

        X_list.append(X)
        Y_list.append(Y)
        n_edges_total += len(X)
        n_files_ok += 1

    elapsed = time.time() - t0
    print(f"\nExtraction complete: {n_files_ok} files, {n_edges_total:,} edges in {elapsed:.1f}s")

    if not X_list:
        print("ERROR: No valid data extracted")
        sys.exit(1)

    X_train = np.concatenate(X_list, axis=0)  # (N_total, 31)
    Y_train = np.concatenate(Y_list, axis=0)  # (N_total, 12)

    print(f"\nTraining data shape: X={X_train.shape}  Y={Y_train.shape}")
    print(f"Label distribution per relation:")
    for rel_idx in range(NUM_RELATIONS):
        n_pos = int(Y_train[:, rel_idx].sum())
        pct = 100.0 * n_pos / len(Y_train)
        print(f"  [{RELATION_NAMES[rel_idx]:18s}] pos={n_pos:7d} ({pct:.2f}%)")

    # ── Optionally save numpy arrays ──────────────────────────────────────────
    if args.save_numpy:
        np.save(os.path.join(NUMPY_DIR, "X_train_physical.npy"), X_train)
        np.save(os.path.join(NUMPY_DIR, "Y_train_physical.npy"), Y_train)
        print(f"\nSaved numpy arrays to {NUMPY_DIR}/")

    # ── Train classifiers ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training 12 binary logistic regression classifiers...")
    print(f"{'='*60}")

    clf = PhysicalRelationClassifier(C=1.0, max_iter=1000)
    t_train = time.time()
    metrics = clf.fit(X_train, Y_train, verbose=True)
    train_elapsed = time.time() - t_train

    print(f"\nTotal training time: {train_elapsed:.1f}s")

    # ── Feature importance ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Top-5 feature importance per relation:")
    print(f"{'='*60}")
    importance = clf.feature_importance(top_k=5)
    for rel_name, feats in importance.items():
        feat_str = ", ".join(f"{n}({c:+.3f})" for n, c in feats)
        print(f"  [{rel_name:18s}] {feat_str}")

    # ── Save classifiers ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    clf.save(args.output)

    print(f"\nDone. Classifiers saved to: {args.output}")
    print(f"Run evaluation with:")
    print(f"  python scripts/evaluate_scenes.py --mode physical")

    return clf, metrics


if __name__ == "__main__":
    main()
