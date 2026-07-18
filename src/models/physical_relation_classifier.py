"""
Physical Relation Classifier — Task 2 (v2: derived features)

Per-relation binary logistic regression classifiers that combine:
  - 10-dim geometry edge features
  - 7-dim derived physical features of subject node A
  - 7-dim derived physical features of object node B
  - 7-dim physical difference features (A - B)
Total: 31-dim input vector

Derived physical features (7-dim per node):
  [0] planarity          — how flat (table, watch, book): (λ₂ - λ₃) / λ₁
  [1] elongation         — how stick-like (pen, bottle): (λ₁ - λ₂) / λ₁
  [2] sphericity         — how ball-like (apple, ball): λ₃ / λ₁
  [3] anisotropy         — shape irregularity: (λ₁ - λ₃) / λ₁
  [4] support_likelihood — can support other objects: planarity * (1 - height_ratio)
  [5] opacity            — how solid/transparent (0-1)
  [6] density_norm       — compactness (points per volume, normalized)

Feature names (31-dim):
  Geometry (10): delta_x, delta_y, delta_z, xy_dist, dist_3d,
                 bbox_overlap, vol_ratio, h_ratio, vert_gap, size_ratio
  A physics (7): A_planarity, A_elongation, A_sphericity, A_anisotropy,
                 A_support, A_opacity, A_density
  B physics (7): B_planarity, B_elongation, B_sphericity, B_anisotropy,
                 B_support, B_opacity, B_density
  Diff      (7): diff_planarity, diff_elongation, diff_sphericity, diff_anisotropy,
                 diff_support, diff_opacity, diff_density
"""
import numpy as np
import pickle
import os
from typing import Dict, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.relations.schema import NUM_RELATIONS, RELATION_NAMES

FEATURE_NAMES = [
    # geometry (10)
    "delta_x", "delta_y", "delta_z", "xy_dist", "dist_3d",
    "bbox_overlap", "vol_ratio", "h_ratio", "vert_gap", "size_ratio",
    # A physical (7)
    "A_planarity", "A_elongation", "A_sphericity", "A_anisotropy",
    "A_support", "A_opacity", "A_density",
    # B physical (7)
    "B_planarity", "B_elongation", "B_sphericity", "B_anisotropy",
    "B_support", "B_opacity", "B_density",
    # difference (7)
    "diff_planarity", "diff_elongation", "diff_sphericity", "diff_anisotropy",
    "diff_support", "diff_opacity", "diff_density",
]

PHYS_DIM = 7
GEOM_DIM = 10
TOTAL_DIM = GEOM_DIM + 2 * PHYS_DIM + PHYS_DIM  # 31


def extract_physical_features_from_node(node_feat: np.ndarray) -> np.ndarray:
    """
    Extract 7-dim derived physical features from a ScanNet v4 node feature vector (10-dim).

    ScanNet approximation (meshes are solid, no real Gaussians):
      We approximate eigenvalue-derived shape features from bbox dimensions.
      Treat size_x, size_y, size_z as pseudo-eigenvalues (sorted descending).

    Node feature layout (v4_axisalign):
      [0-2]  centroid xyz (normalized)
      [3-5]  bbox size xyz (normalized)
      [6]    volume (normalized)
      [7]    height_ratio
      [8]    point_density
      [9]    z_relative
    """
    size_x = float(node_feat[3])
    size_y = float(node_feat[4])
    size_z = float(node_feat[5])

    # Sort dimensions descending to mimic eigenvalues [λ₁, λ₂, λ₃]
    dims = np.sort([size_x, size_y, size_z])[::-1]
    lam0, lam1, lam2 = float(dims[0]), float(dims[1]), float(dims[2])
    lam0_safe = max(lam0, 1e-8)

    # 1. Planarity: (λ₂ - λ₃) / λ₁
    planarity = (lam1 - lam2) / lam0_safe

    # 2. Elongation: (λ₁ - λ₂) / λ₁
    elongation = (lam0 - lam1) / lam0_safe

    # 3. Sphericity: λ₃ / λ₁
    sphericity = lam2 / lam0_safe

    # 4. Anisotropy: (λ₁ - λ₃) / λ₁
    anisotropy = (lam0 - lam2) / lam0_safe

    # 5. Support likelihood: planarity * (1 - height_ratio)
    height_ratio = float(node_feat[7])
    support_likelihood = planarity * (1.0 - min(height_ratio, 1.0))

    # 6. Opacity: ScanNet meshes are solid
    opacity = 1.0

    # 7. Density (normalized): already stored as point_density
    density = float(node_feat[8])
    density_norm = min(density / 1000.0, 1.0) if density > 1.0 else density

    return np.array([planarity, elongation, sphericity, anisotropy,
                     support_likelihood, opacity, density_norm],
                    dtype=np.float32)


def extract_physical_features_from_gaussian(obj) -> np.ndarray:
    """
    Compute 7-dim physically meaningful derived features from Gaussian properties.

    These features encode what the Gaussian representation tells us about
    the physical nature of the object — its shape, solidity, and structure.

    Uses actual Gaussian properties: _mean_opacity, _eigenvalues, point_count,
    volume, size. Falls back gracefully if attributes are missing.
    """
    # eigenvalues → shape descriptors
    eigenvalues = getattr(obj, "_eigenvalues", None)
    if eigenvalues is None or len(eigenvalues) < 3 or not np.all(np.isfinite(eigenvalues)):
        eigenvalues = np.array([1.0, 1.0, 1.0])
    lam = np.sort(np.abs(eigenvalues))[::-1]  # descending: [λ₁, λ₂, λ₃]
    lam0_safe = max(float(lam[0]), 1e-8)

    # 1. Planarity: how flat is the object? (table, watch, book)
    #    High when λ₁ ≈ λ₂ >> λ₃ (two large, one small)
    planarity = float(lam[1] - lam[2]) / lam0_safe

    # 2. Elongation: how stick-like? (pen, bottle)
    #    High when λ₁ >> λ₂ ≈ λ₃
    elongation = float(lam[0] - lam[1]) / lam0_safe

    # 3. Sphericity: how ball-like? (apple, ball)
    #    High when λ₁ ≈ λ₂ ≈ λ₃
    sphericity = float(lam[2]) / lam0_safe

    # 4. Anisotropy: overall shape irregularity
    #    0 = perfect sphere, 1 = maximally anisotropic
    anisotropy = float(lam[0] - lam[2]) / lam0_safe

    # 5. Support surface likelihood: could this object support another?
    #    High planarity + large XY extent + low height = good support surface
    size = np.maximum(obj.size, 1e-6)
    height_ratio = float(size[2] / max(size[0], size[1]))
    support_likelihood = planarity * (1.0 - min(height_ratio, 1.0))

    # 6. Opacity (solidity): how solid/transparent
    opacity = getattr(obj, "_mean_opacity", None)
    if opacity is None or not np.isfinite(opacity):
        opacity = 0.5
    opacity = float(opacity)

    # 7. Density: points per unit volume (compactness)
    volume = max(float(obj.volume), 1e-6)
    density = float(obj.point_count) / volume
    density_norm = min(density / 1000.0, 1.0)  # normalize to [0,1]

    return np.array([planarity, elongation, sphericity, anisotropy,
                     support_likelihood, opacity, density_norm],
                    dtype=np.float32)


def build_full_feature_vector(
    edge_feat: np.ndarray,
    phys_a: np.ndarray,
    phys_b: np.ndarray,
) -> np.ndarray:
    """
    Concatenate geometry + physical A + physical B + physical diff → 31-dim.
    """
    phys_diff = phys_a - phys_b
    return np.concatenate([edge_feat, phys_a, phys_b, phys_diff]).astype(np.float32)


class PhysicalRelationClassifier:
    """
    12 per-relation binary logistic regression classifiers.

    Each classifier takes a 31-dim feature vector and predicts whether
    a given spatial relation holds for a directed edge A→B.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.max_iter = max_iter
        self.classifiers: Dict[int, LogisticRegression] = {}
        self.scalers: Dict[int, StandardScaler] = {}
        self.feature_names = FEATURE_NAMES
        self.num_relations = NUM_RELATIONS

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = True) -> dict:
        """
        Train one binary classifier per relation.

        Args:
            X: (N, 31) feature matrix
            Y: (N, 12) multi-hot label matrix
            verbose: print per-relation train F1

        Returns:
            dict with per-relation train metrics
        """
        import time
        from sklearn.metrics import f1_score

        metrics = {}
        t0 = time.time()

        for rel_idx in range(self.num_relations):
            rel_name = RELATION_NAMES[rel_idx]
            y = Y[:, rel_idx].astype(int)

            n_pos = y.sum()
            n_neg = (y == 0).sum()

            if n_pos < 2:
                if verbose:
                    print(f"  [{rel_name:18s}] SKIP — only {n_pos} positive samples")
                continue

            # Scale features per-relation (different relations may have
            # different feature importance scales)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            clf = LogisticRegression(
                class_weight="balanced",
                C=self.C,
                max_iter=self.max_iter,
                solver="lbfgs",
                random_state=42,
            )
            clf.fit(X_scaled, y)

            y_pred = clf.predict(X_scaled)
            train_f1 = f1_score(y, y_pred, zero_division=0)

            self.classifiers[rel_idx] = clf
            self.scalers[rel_idx] = scaler
            metrics[rel_name] = {
                "train_f1": round(float(train_f1), 4),
                "n_pos": int(n_pos),
                "n_neg": int(n_neg),
            }

            if verbose:
                print(f"  [{rel_name:18s}] train_F1={train_f1:.3f}  "
                      f"pos={n_pos:6d}  neg={n_neg:6d}")

        elapsed = time.time() - t0
        if verbose:
            print(f"\n  Training complete in {elapsed:.1f}s")

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability for each relation.

        Args:
            X: (N, 31) feature matrix

        Returns:
            (N, 12) probability matrix — 0.0 for untrained relations
        """
        N = len(X)
        probs = np.zeros((N, self.num_relations), dtype=np.float32)

        for rel_idx, clf in self.classifiers.items():
            scaler = self.scalers[rel_idx]
            X_scaled = scaler.transform(X)
            probs[:, rel_idx] = clf.predict_proba(X_scaled)[:, 1]

        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.35) -> np.ndarray:
        """
        Predict binary labels for each relation.

        Args:
            X: (N, 31) feature matrix
            threshold: probability threshold

        Returns:
            (N, 12) binary prediction matrix
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importance(self, top_k: int = 5) -> dict:
        """
        Return top-k most important features per relation based on
        absolute logistic regression coefficients.

        Returns:
            {rel_name: [(feature_name, coef), ...]}
        """
        importance = {}
        for rel_idx, clf in self.classifiers.items():
            rel_name = RELATION_NAMES[rel_idx]
            coefs = clf.coef_[0]  # (31,)
            ranked = sorted(
                zip(self.feature_names, coefs),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            importance[rel_name] = [(name, round(float(c), 4))
                                    for name, c in ranked[:top_k]]
        return importance

    def save(self, path: str):
        """Save classifiers and scalers to a pickle file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "classifiers": self.classifiers,
                "scalers": self.scalers,
                "C": self.C,
                "max_iter": self.max_iter,
                "feature_names": self.feature_names,
            }, f)
        print(f"Saved {len(self.classifiers)} classifiers to {path}")

    @classmethod
    def load(cls, path: str) -> "PhysicalRelationClassifier":
        """Load classifiers from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(C=data["C"], max_iter=data["max_iter"])
        obj.classifiers = data["classifiers"]
        obj.scalers = data["scalers"]
        obj.feature_names = data.get("feature_names", FEATURE_NAMES)
        print(f"Loaded {len(obj.classifiers)} classifiers from {path}")
        return obj
