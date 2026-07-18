"""
End-to-end inference: Gaussian Splat → Scene Graph

Pipeline:
    splat.ply
      → filter_gaussians()          remove low-opacity background
      → gaussian_to_objects()       HDBSCAN clustering → Object3D list
      → semantic labeling           YOLO or Grounding DINO (optional)
      → extract features            10-dim node + 10-dim edge (matches ScanNet)
      → RelationGNN                 predict 14 relations
      → SceneGraph                  structured output
      → natural language            human-readable description

Usage:
    python src/inference/gaussian_inference.py --scene scene_01
    python src/inference/gaussian_inference.py --scene scene_01 --labeler yolo
    python src/inference/gaussian_inference.py --scene scene_01 --labeler dino
    python src/inference/gaussian_inference.py --ply path/to/splat.ply
"""
import sys
sys.path.insert(0, ".")

import os
import json
import argparse
import numpy as np
import torch
from typing import List, Optional, Tuple

from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
from src.gaussian.clustering import (
    gaussian_to_objects,
    extract_gaussian_node_features,
    extract_gaussian_edge_features,
)
from scripts.build_3rscan_graphs import extract_3rscan_edge_features
from src.models.relation_gnn import RelationGNN
from geokan_relation import GeoKANRelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES, RELATION_DESCRIPTIONS, Relation
from src.graph.definitions import Object3D

DATA_DIR      = "D:/logicsplat_data/processed"
MODEL_PATH    = "models/relation_gnn_v7_dualhead.pt"
FALLBACK_MODEL = "models/relation_gnn_gat_scannet_geometry_multilabel_v3_axisalign.pt"
THRESHOLDS_PATH = "models/relation_gnn_v7_dualhead_thresholds.json"


def find_model() -> str:
    """Find the best available trained model."""
    for path in [MODEL_PATH, FALLBACK_MODEL,
                 "models/relation_gnn_gat_scannet_geometry_multilabel_v2.pt"]:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No trained model found. Run: python src/training/train_scannet.py"
    )


def load_tuned_thresholds() -> Optional[dict]:
    """Load per-relation tuned thresholds if available."""
    if os.path.exists(THRESHOLDS_PATH):
        with open(THRESHOLDS_PATH) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    return None


def load_labels(scene_dir: str, objects: List[Object3D]) -> List[Object3D]:
    """
    Load cached semantic labels from yolo_labels.json or dino_labels.json.
    Prefers dino_labels.json (higher quality) over yolo_labels.json.
    Does nothing if neither file exists.
    """
    for fname in ("dino_labels.json", "yolo_labels.json"):
        labels_path = os.path.join(scene_dir, fname)
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                cached = json.load(f)
            source = "Grounding DINO" if "dino" in fname else "YOLO"
            print(f"  Loaded {source} labels from {labels_path}")
            for obj in objects:
                lbl = cached.get(str(obj.uid))
                if lbl and lbl != "object":
                    obj.label = lbl
            return objects
    print("  No cached labels found — objects will show as 'object'")
    print("  Run: python src/labeling/grounding_dino_labeler.py --scene <scene>")
    return objects


def run_labeling(
    objects: List[Object3D],
    scene_dir: str,
    labeler: str = "auto",
    n_frames: int = 30,
    force: bool = False,
) -> List[Object3D]:
    """
    Run semantic labeling on objects.

    Args:
        labeler: "auto"  — use cached labels if available, else skip
                 "yolo"  — run YOLO labeling
                 "dino"  — run Grounding DINO labeling (best quality)
                 "none"  — skip labeling entirely
        force:   re-run even if cached labels exist
    """
    if labeler == "none":
        return objects

    transforms_path = os.path.join(scene_dir, "ns_data", "transforms.json")
    images_dir      = os.path.join(scene_dir, "ns_data", "images")

    if not os.path.exists(transforms_path) or not os.path.isdir(images_dir):
        print("  Labeling skipped — transforms.json or images/ not found")
        return load_labels(scene_dir, objects)

    if labeler == "auto":
        # just load whatever is cached
        return load_labels(scene_dir, objects)

    if labeler == "dino":
        # remove old cache so it re-runs
        if force:
            for f in ("dino_labels.json", "yolo_labels.json"):
                p = os.path.join(scene_dir, f)
                if os.path.exists(p):
                    os.remove(p)
        dino_cache = os.path.join(scene_dir, "dino_labels.json")
        if os.path.exists(dino_cache) and not force:
            return load_labels(scene_dir, objects)
        try:
            from src.labeling.grounding_dino_labeler import label_objects_with_dino
            return label_objects_with_dino(
                objects, transforms_path, images_dir,
                n_frames=n_frames, scene_dir=scene_dir,
            )
        except ImportError as e:
            print(f"  Grounding DINO not available ({e}), falling back to YOLO")
            labeler = "yolo"

    if labeler == "yolo":
        if force:
            p = os.path.join(scene_dir, "yolo_labels.json")
            if os.path.exists(p):
                os.remove(p)
        from src.labeling.yolo_labeler import label_objects_with_yolo
        return label_objects_with_yolo(
            objects, transforms_path, images_dir,
            n_frames=n_frames, scene_dir=scene_dir,
        )

    return objects


def build_graph(objects: List[Object3D]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build node features, edge index, and edge features for GNN inference.
    Uses the same 10/10 feature layout as the ScanNet-trained model.
    """
    if len(objects) < 2:
        raise ValueError("Need at least 2 objects for relation inference")

    # scene extent for normalization
    all_mins = np.stack([o.bbox_min for o in objects])
    all_maxs = np.stack([o.bbox_max for o in objects])
    scene_min = all_mins.min(axis=0)
    scene_max = all_maxs.max(axis=0)
    scene_extent = np.maximum(scene_max - scene_min, 1e-6)

    # node features
    x = np.stack([
        extract_gaussian_node_features(o, scene_extent, scene_min)
        for o in objects
    ])

    # all directed pairs
    src, dst, edge_feats = [], [], []
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i == j:
                continue
            src.append(i)
            dst.append(j)
            edge_feats.append(extract_3rscan_edge_features(a, b, scene_extent))

    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor([src, dst], dtype=torch.long),
        torch.tensor(np.stack(edge_feats), dtype=torch.float32),
    )


def _filter_by_point_count(objects: List[Object3D]) -> List[Object3D]:
    """
    Remove floater clusters whose point count is below 5% of the median.

    Using the median (not max) makes this robust when one cluster dominates.
    All thresholds are computed from the scene's own data — no hardcoded values.
    """
    if len(objects) < 2:
        return objects
    pts = np.array([o.point_count for o in objects], dtype=float)
    median_pts = np.median(pts)
    threshold = median_pts * 0.05
    filtered = [o for o in objects if o.point_count >= threshold]
    if len(filtered) >= 2:
        removed = len(objects) - len(filtered)
        if removed > 0:
            print(f"  Point-count filter: removed {removed} floater(s) "
                  f"(threshold={threshold:.0f} pts, median={median_pts:.0f})")
            for i, o in enumerate(filtered):
                o.uid = i
        return filtered
    return objects


def _filter_by_xy_range(objects: List[Object3D]) -> List[Object3D]:
    """
    Remove clusters whose XY centroid is a spatial outlier relative to the
    rest of the scene.

    Uses 4× MAD of XY distances from the median centroid.  Falls back to
    3× std when MAD is near zero (all objects tightly clustered).
    All thresholds are computed from the scene's own data — no hardcoded values.
    """
    if len(objects) < 2:
        return objects
    xy = np.array([[o.centroid[0], o.centroid[1]] for o in objects])
    median_xy = np.median(xy, axis=0)
    dists = np.linalg.norm(xy - median_xy, axis=1)
    mad_xy = np.median(np.abs(dists - np.median(dists)))
    # If MAD is near zero (all objects close together), use std instead
    spread = max(mad_xy * 4.0, np.std(dists) * 3.0, 0.5)
    filtered = [o for o, d in zip(objects, dists) if d <= spread]
    if len(filtered) >= 2:
        removed = len(objects) - len(filtered)
        if removed > 0:
            print(f"  XY-range filter: removed {removed} coordinate-explosion cluster(s) "
                  f"(threshold={spread:.2f} units)")
            for i, o in enumerate(filtered):
                o.uid = i
        return filtered
    return objects


def _filter_by_z_range(objects: List[Object3D]) -> List[Object3D]:
    """
    Remove clusters with extreme Z coordinates (background floaters).

    Uses a robust 3-MAD rule on raw (pre-flip) Z.  All thresholds are
    computed from the scene's own data — no hardcoded values.
    """
    if len(objects) < 2:
        return objects
    centroids_z = np.array([o.centroid[2] for o in objects])
    median_z = float(np.median(centroids_z))
    mad_z    = float(np.median(np.abs(centroids_z - median_z)))
    if mad_z <= 1e-6:
        return objects
    keep = [o for o in objects if abs(o.centroid[2] - median_z) <= 3.0 * mad_z]
    n_removed = len(objects) - len(keep)
    if n_removed > 0 and len(keep) >= 2:
        print(f"  [Z-filter] Removed {n_removed} outlier cluster(s) "
              f"(median_z={median_z:.2f}, mad_z={mad_z:.2f})")
        for new_uid, o in enumerate(keep):
            o.uid = new_uid
        return keep
    elif n_removed > 0:
        print(f"  [Z-filter] Would remove {n_removed} outlier(s) but "
              f"only {len(keep)} objects would remain — skipping filter")
    return objects


def run_inference(
    ply_path: str,
    model_path: str = None,
    confidence_threshold: float = 0.25,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    labeler: str = "auto",
    n_label_frames: int = 30,
    scene_dir: Optional[str] = None,
    n_objects_hint: Optional[int] = None,
    mode: str = "hybrid",
) -> dict:
    """
    Full pipeline: splat.ply → scene graph.

    Args:
        ply_path:             path to splat.ply
        model_path:           path to trained .pt model (auto-detected if None)
        confidence_threshold: minimum softmax confidence to include a relation
        device:               "cpu" or "cuda"
        labeler:              "auto" | "yolo" | "dino" | "none"
                              auto = load cached labels if available
                              yolo = run YOLO (80 COCO classes)
                              dino = run Grounding DINO (open-vocabulary, best)
                              none = skip labeling
        n_label_frames:       frames to sample for labeling
        scene_dir:            scene root dir (auto-detected from ply_path if None)
        n_objects_hint:       expected number of objects (from GT) — used to
                              guide HDBSCAN clustering when auto-tuning fails
        mode:                 inference mode for ablation study
                              "hybrid"   — GNN + geometry validation (default)
                              "geometry" — geometry rules only, GNN bypassed
                              "gnn"      — GNN sigmoid only, no geometry validation
                              "ensemble" — geometry rules for directional/vertical,
                                           physical classifiers for adjacent_to

    Returns:
        {
          'objects':   List[Object3D],
          'relations': List[dict],
          'params':    dict,
        }
    """
    if model_path is None and mode not in ("geometry", "ensemble"):
        model_path = find_model()

    # infer scene_dir from ply_path if not given
    if scene_dir is None:
        scene_dir = os.path.dirname(ply_path)

    # ── load and cluster Gaussians ────────────────────────────────────────────
    print(f"Loading: {ply_path}")
    cloud = load_gaussian_ply(ply_path)
    cloud_filtered = filter_gaussians(cloud, opacity_threshold=0.1)
    print(f"  Gaussians: {cloud.num_gaussians:,} raw → {cloud_filtered.num_gaussians:,} after opacity filter")
    cloud_filtered = prune_isolated_gaussians(cloud_filtered, nb_neighbors=20, std_ratio=2.0)
    print(f"  Gaussians: {cloud_filtered.num_gaussians:,} after SOR pruning")

    # If we know the expected object count, pass the tight target range directly
    # to gaussian_to_objects so the auto-tuner (which now tries both eom and
    # leaf methods across a wider mcs range) finds the right cluster count.
    if n_objects_hint is not None:
        target_min = max(2, n_objects_hint - 1)
        target_max = n_objects_hint + 1
        objects, params = gaussian_to_objects(
            cloud_filtered,
            target_min=target_min,
            target_max=target_max,
        )
        # If still under-segmented, retry with progressively higher color weights.
        # Higher color_weight = color drives clustering more than position,
        # which helps separate same-position different-color objects (scenes 08/13).
        if len(objects) < n_objects_hint - 1:
            for color_w in [0.5, 0.7, 1.0, 1.5]:
                objects2, params2 = gaussian_to_objects(
                    cloud_filtered,
                    target_min=target_min,
                    target_max=target_max,
                    color_weight=color_w,
                )
                print(f"  Retry color_weight={color_w}: {len(objects2)} clusters")
                if len(objects2) >= n_objects_hint - 1:
                    objects, params = objects2, params2
                    break
    else:
        objects, params = gaussian_to_objects(cloud_filtered)

    print(f"Objects found: {len(objects)} | params: {params}")

    # Post-clustering filters removed — floaters are now pruned at the Gaussian
    # level by prune_isolated_gaussians() (SOR) before clustering.
    # _filter_by_point_count, _filter_by_xy_range, _filter_by_z_range are kept
    # below for debugging but are no longer called in the main pipeline.

    # ── Z-axis normalisation ──────────────────────────────────────────────────
    # Gaussian Splat scenes use a coordinate system where Z increases downward
    # (more negative Z = physically higher). All geometry rules in
    # src/relations/geometry.py assume +Z = up (standard right-hand convention).
    # Negate Z on every Object3D so the rules fire correctly.
    for o in objects:
        o.centroid  = o.centroid.copy();  o.centroid[2]  *= -1
        o.bbox_min  = o.bbox_min.copy();  o.bbox_min[2]  *= -1
        o.bbox_max  = o.bbox_max.copy();  o.bbox_max[2]  *= -1
        # After negation bbox_min[2] > bbox_max[2] — swap to keep min < max
        o.bbox_min[2], o.bbox_max[2] = min(o.bbox_min[2], o.bbox_max[2]), max(o.bbox_min[2], o.bbox_max[2])

    # ── semantic labeling ─────────────────────────────────────────────────────
    print(f"\nLabeling (mode={labeler})...")
    objects = run_labeling(objects, scene_dir, labeler=labeler,
                           n_frames=n_label_frames)

    print(f"\nObjects after labeling:")
    for o in objects:
        print(f"  Obj {o.uid} [{o.label}] pts={o.point_count} "
              f"z={o.centroid[2]:.2f} color=rgb({o.color[0]},{o.color[1]},{o.color[2]})")

    if len(objects) < 2:
        print("Not enough objects for relation inference.")
        return {"objects": objects, "relations": [], "params": params}

    # ── build graph features ──────────────────────────────────────────────────
    x, edge_index, edge_attr = build_graph(objects)

    # ── GEOMETRY-ONLY MODE ────────────────────────────────────────────────────
    # Bypass the GNN entirely — derive all relations from 3D geometry rules.
    # This is the baseline for the ablation study (--mode geometry).
    if mode == "geometry":
        from src.relations.geometry import derive_relations, compute_scene_context

        GEOMETRIC_RELATIONS = {
            "higher_than", "lower_than",
            "left_of", "right_of",
            "in_front_of", "behind",
            "on_top_of", "under",
        }

        # Compute scene-adaptive thresholds once for all pairs
        all_mins = np.stack([o.bbox_min for o in objects])
        all_maxs = np.stack([o.bbox_max for o in objects])
        scene_ctx = compute_scene_context(all_mins, all_maxs)
        print(f"  [geometry] Scene context: z_min={scene_ctx['z_min_threshold']:.4f} "
              f"z_max={scene_ctx['z_max_threshold']:.4f} "
              f"xy_foot_min={scene_ctx['xy_footprint_min']:.4f} "
              f"dominance={scene_ctx['dominance_ratio']:.2f}")

        seen_keys: set = set()
        relations: list = []

        for i, a in enumerate(objects):
            for j, b in enumerate(objects):
                if i == j:
                    continue
                derived = derive_relations(a.bbox_min, a.bbox_max,
                                           b.bbox_min, b.bbox_max,
                                           scene_context=scene_ctx)
                derived_names = {RELATION_NAMES[int(r)] for r in derived}

                for rel in derived:
                    rel_name = RELATION_NAMES[int(rel)]
                    if rel_name not in GEOMETRIC_RELATIONS:
                        continue
                    key = (i, rel_name, j)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    relations.append({
                        "subject_id":    i,
                        "subject_label": objects[i].label,
                        "relation":      rel_name,
                        "object_id":     j,
                        "object_label":  objects[j].label,
                        "confidence":    1.0,
                    })

                # Emit higher_than alongside on_top_of (GT annotates both)
                if "on_top_of" in derived_names:
                    key = (i, "higher_than", j)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        relations.append({
                            "subject_id":    i,
                            "subject_label": objects[i].label,
                            "relation":      "higher_than",
                            "object_id":     j,
                            "object_label":  objects[j].label,
                            "confidence":    1.0,
                        })

                # Emit lower_than alongside under
                if "under" in derived_names:
                    key = (i, "lower_than", j)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        relations.append({
                            "subject_id":    i,
                            "subject_label": objects[i].label,
                            "relation":      "lower_than",
                            "object_id":     j,
                            "object_label":  objects[j].label,
                            "confidence":    1.0,
                        })

        # adjacent_to — relative gap criterion (scale-invariant)
        on_top_of_pairs = {(r["subject_id"], r["object_id"])
                           for r in relations if r["relation"] == "on_top_of"}
        n_obj = len(objects)
        centroids_xy = np.array([o.centroid[:2] for o in objects])
        all_dists = []
        for i in range(n_obj):
            for j in range(i + 1, n_obj):
                if (i, j) not in on_top_of_pairs and (j, i) not in on_top_of_pairs:
                    d = float(np.linalg.norm(centroids_xy[i] - centroids_xy[j]))
                    all_dists.append((d, i, j))
        if all_dists:
            all_dists.sort()
            threshold = all_dists[0][0] * 1.2
            for d, i, j in all_dists:
                if d > threshold:
                    break
                for (s, o) in [(i, j), (j, i)]:
                    key = (s, "adjacent_to", o)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    relations.append({
                        "subject_id":    s,
                        "subject_label": objects[s].label,
                        "relation":      "adjacent_to",
                        "object_id":     o,
                        "object_label":  objects[o].label,
                        "confidence":    1.0,
                    })

        return {"objects": objects, "relations": relations, "params": params}

    # ── ENSEMBLE MODE ─────────────────────────────────────────────────────────
    # Routes each relation type to the best predictor:
    #   - Geometry rules  → left_of, right_of, in_front_of, behind,
    #                        higher_than, lower_than, on_top_of, under
    #   - Physical classifiers (threshold=0.45) → adjacent_to only
    #   - Excluded (ScanNet-specific, never valid for tabletop):
    #       attached_to, inside, hanging_from
    if mode == "ensemble":
        from src.relations.geometry import derive_relations, compute_scene_context
        from src.models.physical_relation_classifier import (
            PhysicalRelationClassifier,
            extract_physical_features_from_gaussian,
            build_full_feature_vector,
        )
        from src.gaussian.clustering import extract_gaussian_edge_features

        ENSEMBLE_GEOMETRY_RELATIONS = {
            "higher_than", "lower_than",
            "left_of", "right_of",
            "in_front_of", "behind",
            "on_top_of", "under",
        }

        # Compute scene-adaptive thresholds once for all pairs
        all_mins = np.stack([o.bbox_min for o in objects])
        all_maxs = np.stack([o.bbox_max for o in objects])
        scene_ctx = compute_scene_context(all_mins, all_maxs)
        print(f"  [ensemble] Scene context: z_min={scene_ctx['z_min_threshold']:.4f} "
              f"z_max={scene_ctx['z_max_threshold']:.4f} "
              f"xy_foot_min={scene_ctx['xy_footprint_min']:.4f} "
              f"dominance={scene_ctx['dominance_ratio']:.2f}")

        seen_keys: set = set()
        relations: list = []

        # ── Step 1: Geometry rules for directional + vertical relations ────────
        for i, a in enumerate(objects):
            for j, b in enumerate(objects):
                if i == j:
                    continue
                derived = derive_relations(a.bbox_min, a.bbox_max,
                                           b.bbox_min, b.bbox_max,
                                           scene_context=scene_ctx)
                derived_names = {RELATION_NAMES[int(r)] for r in derived}

                for rel in derived:
                    rel_name = RELATION_NAMES[int(rel)]
                    if rel_name not in ENSEMBLE_GEOMETRY_RELATIONS:
                        continue
                    key = (i, rel_name, j)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    relations.append({
                        "subject_id":    i,
                        "subject_label": objects[i].label,
                        "relation":      rel_name,
                        "object_id":     j,
                        "object_label":  objects[j].label,
                        "confidence":    1.0,
                    })

                # Emit higher_than alongside on_top_of (GT annotates both)
                if "on_top_of" in derived_names:
                    key = (i, "higher_than", j)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        relations.append({
                            "subject_id":    i,
                            "subject_label": objects[i].label,
                            "relation":      "higher_than",
                            "object_id":     j,
                            "object_label":  objects[j].label,
                            "confidence":    1.0,
                        })

                # Emit lower_than alongside under
                if "under" in derived_names:
                    key = (i, "lower_than", j)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        relations.append({
                            "subject_id":    i,
                            "subject_label": objects[i].label,
                            "relation":      "lower_than",
                            "object_id":     j,
                            "object_label":  objects[j].label,
                            "confidence":    1.0,
                        })

        # ── Step 2: Physical classifiers for adjacent_to ──────────────────────
        classifiers_path = "models/physical_classifiers_v2.pkl"
        if not os.path.exists(classifiers_path):
            # Fall back to v1 if v2 not available
            classifiers_path = "models/physical_classifiers.pkl"
        if os.path.exists(classifiers_path):
            clf = PhysicalRelationClassifier.load(classifiers_path)

            # Scene extent for edge feature normalization
            all_mins = np.stack([o.bbox_min for o in objects])
            all_maxs = np.stack([o.bbox_max for o in objects])
            scene_extent = np.maximum(all_maxs.max(axis=0) - all_mins.min(axis=0), 1e-6)

            n = len(objects)
            pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

            from src.models.physical_relation_classifier import TOTAL_DIM as _PHYS_TOTAL_DIM
            from src.models.physical_relation_classifier import GEOM_DIM as _PHYS_GEOM_DIM
            X = np.zeros((len(pairs), _PHYS_TOTAL_DIM), dtype=np.float32)
            for k, (i, j) in enumerate(pairs):
                a, b = objects[i], objects[j]
                edge_f = extract_gaussian_edge_features(a, b, scene_extent)[:_PHYS_GEOM_DIM]
                phys_a = extract_physical_features_from_gaussian(a)
                phys_b = extract_physical_features_from_gaussian(b)
                X[k] = build_full_feature_vector(edge_f, phys_a, phys_b)

            probs = clf.predict_proba(X)  # (E, 12)

            # Find adjacent_to relation index
            adj_idx = None
            for ri, rn in RELATION_NAMES.items():
                if rn == "adjacent_to":
                    adj_idx = ri
                    break

            if adj_idx is not None:
                # ── Dynamic per-scene threshold for adjacent_to ───────────────
                # Fully adaptive: uses the score distribution itself to find
                # the natural separation between "adjacent" and "not adjacent".
                #
                # Algorithm:
                #   1. Collect all adjacent_to scores across all directed pairs
                #   2. If max(scores) < 0.3 → no signal, skip all predictions
                #   3. Otherwise use percentile-based threshold:
                #      - Compute the 90th percentile of scores
                #      - Threshold = max(p90, median + 2*MAD)
                #      - This adapts to both uniform and skewed distributions
                #   4. Floor: never go below 50% of max score (ensures selectivity)
                adj_scores = np.array([float(probs[k, adj_idx]) for k in range(len(pairs))])
                max_score = float(adj_scores.max()) if len(adj_scores) > 0 else 0.0

                if max_score < 0.3:
                    print(f"  [ensemble] adjacent_to: max score {max_score:.3f} < 0.3 "
                          f"— no signal, skipping all predictions")
                    adj_threshold = 1.0  # effectively disable
                else:
                    median_score = float(np.median(adj_scores))
                    mad_score = float(np.median(np.abs(adj_scores - median_score)))
                    p90 = float(np.percentile(adj_scores, 90))

                    # Robust threshold: whichever is higher of p90 or median+2*MAD
                    # This ensures we only fire for clear outliers in the score dist.
                    # Floor at 50% of max score to maintain selectivity.
                    adj_threshold = max(p90, median_score + 2.0 * mad_score, max_score * 0.5)
                    # Cap at 95% of max to always allow the strongest signal through
                    adj_threshold = min(adj_threshold, max_score * 0.95)

                    print(f"  [ensemble] adjacent_to: max={max_score:.3f} "
                          f"median={median_score:.3f} MAD={mad_score:.3f} p90={p90:.3f} "
                          f"→ dynamic threshold={adj_threshold:.3f}")

                for k, (i, j) in enumerate(pairs):
                    conf = float(probs[k, adj_idx])
                    if conf < adj_threshold:
                        continue
                    key = (i, "adjacent_to", j)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    relations.append({
                        "subject_id":    i,
                        "subject_label": objects[i].label,
                        "relation":      "adjacent_to",
                        "object_id":     j,
                        "object_label":  objects[j].label,
                        "confidence":    round(conf, 3),
                    })
        else:
            print(f"  [ensemble] WARNING: classifiers not found at {classifiers_path} "
                  f"— adjacent_to will be skipped")

        return {"objects": objects, "relations": relations, "params": params}

    # ── load model (GNN modes: "hybrid" and "gnn") ────────────────────────────
    node_feat_dim = x.shape[1]   # 10
    edge_feat_dim = edge_attr.shape[1]  # 17

    # Infer hidden_dim from the checkpoint so we don't hardcode it.
    # node_encoder.0.weight has shape (hidden_dim, node_feat_dim).
    state = torch.load(model_path, weights_only=False, map_location=device)
    hidden_dim = state["node_encoder.0.weight"].shape[0]  # 128 (v1) or 256 (v2+)

    # Select model class based on model path
    if "geokan" in os.path.basename(model_path).lower():
        model = GeoKANRelationGNN(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            num_relations=NUM_RELATIONS,
        ).to(device)
    else:
        model = RelationGNN(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            num_relations=NUM_RELATIONS,
        ).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"Model: {model_path}  (hidden_dim={hidden_dim}, edge_feat_dim={edge_feat_dim})")

    # ── Load per-relation tuned thresholds ────────────────────────────────────
    tuned_thresholds = load_tuned_thresholds()
    if tuned_thresholds:
        print(f"  Using tuned per-relation thresholds from {THRESHOLDS_PATH}")

    # ── inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(x.to(device), edge_index.to(device), edge_attr.to(device))
        probs_sigmoid = torch.sigmoid(logits).cpu()

    # ── GNN + GEOMETRY VALIDATION ─────────────────────────────────────────────
    # Architecture: GNN proposes, geometry confirms.
    #
    # mode="hybrid" — GNN sigmoid threshold + geometry_validates() gate
    # mode="gnn"    — GNN sigmoid threshold only, no geometry filtering
    #
    # Per-relation thresholds from validation tuning (if available),
    # otherwise fall back to 0.35 for recall on rare classes.

    default_threshold = 0.35

    seen_keys: set = set()
    relations: list = []

    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()

    # Apply hierarchical constraints on the full prediction matrix
    # before iterating over individual predictions
    n_edges = probs_sigmoid.shape[0]
    if tuned_thresholds:
        preds_matrix = torch.zeros(n_edges, NUM_RELATIONS)
        for rel_idx in range(NUM_RELATIONS):
            thresh = tuned_thresholds.get(rel_idx, default_threshold)
            preds_matrix[:, rel_idx] = (probs_sigmoid[:, rel_idx] >= thresh).float()
    else:
        preds_matrix = (probs_sigmoid >= default_threshold).float()

    # Apply hierarchical constraints (import from training module)
    from src.training.train_scannet import apply_hierarchical_constraints
    preds_matrix = apply_hierarchical_constraints(preds_matrix, probs_sigmoid)

    for edge_idx, (src, dst) in enumerate(zip(src_list, dst_list)):
        for rel_idx in range(NUM_RELATIONS):
            rel_name = RELATION_NAMES[rel_idx]

            # Use constraint-adjusted predictions
            if preds_matrix[edge_idx, rel_idx] < 1.0:
                continue

            conf = float(probs_sigmoid[edge_idx, rel_idx])

            # Geometry validation gate (hybrid mode only)
            if mode == "hybrid" and not geometry_validates(objects[src], objects[dst], rel_name):
                continue

            key = (src, rel_name, dst)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            relations.append({
                "subject_id":    src,
                "subject_label": objects[src].label,
                "relation":      rel_name,
                "object_id":     dst,
                "object_label":  objects[dst].label,
                "confidence":    round(conf, 3),
            })

    return {"objects": objects, "relations": relations, "params": params}


def geometry_validates(obj_a: "Object3D", obj_b: "Object3D", relation: str) -> bool:
    """
    Returns True if the predicted relation is geometrically plausible
    given the 3D bounding boxes of the two objects.

    Uses LOOSE thresholds — geometry is a filter, not a predictor.
    A prediction that passes but is wrong is a GNN error.
    A prediction that fails is a geometric impossibility.
    """
    a_min, a_max = obj_a.bbox_min, obj_a.bbox_max
    b_min, b_max = obj_b.bbox_min, obj_b.bbox_max

    a_center = (a_min + a_max) / 2
    b_center = (b_min + b_max) / 2
    delta    = a_center - b_center

    a_size   = np.maximum(a_max - a_min, 1e-6)
    b_size   = np.maximum(b_max - b_min, 1e-6)
    avg_size   = (np.linalg.norm(a_size) + np.linalg.norm(b_size)) / 2
    avg_height = (a_size[2] + b_size[2]) / 2

    if relation == "left_of":
        # A must be to the left of B (negative X delta)
        return delta[0] < 0

    if relation == "right_of":
        return delta[0] > 0

    if relation == "higher_than":
        # A centroid must be above B centroid by at least 10% of 3D distance
        dist_3d = float(np.linalg.norm(delta))
        return delta[2] > dist_3d * 0.1 and delta[2] > 0.05

    if relation == "lower_than":
        dist_3d = float(np.linalg.norm(delta))
        return delta[2] < -(dist_3d * 0.1) and delta[2] < -0.05

    if relation == "in_front_of":
        # A must be in front of B (negative Y delta in our convention)
        return delta[1] < 0

    if relation == "behind":
        return delta[1] > 0

    if relation == "on_top_of":
        # Centroid-based: A above B by 5cm–1.5 units, A's XY within B's footprint
        a_size   = np.maximum(a_max - a_min, 1e-6)
        b_size   = np.maximum(b_max - b_min, 1e-6)
        centroid_z_diff  = float(a_center[2] - b_center[2])
        centroid_xy_dist = float(np.linalg.norm(a_center[:2] - b_center[:2]))
        a_xy_size = float(np.linalg.norm(a_size[:2]))
        b_xy_size = float(np.linalg.norm(b_size[:2]))
        return (centroid_z_diff > 0.05 and
                centroid_z_diff < 1.5 and
                centroid_xy_dist < b_xy_size * 0.8 and
                a_xy_size < b_xy_size * 2.0)

    if relation == "under":
        a_size   = np.maximum(a_max - a_min, 1e-6)
        b_size   = np.maximum(b_max - b_min, 1e-6)
        centroid_z_diff_inv = float(b_center[2] - a_center[2])
        centroid_xy_dist    = float(np.linalg.norm(a_center[:2] - b_center[:2]))
        a_xy_size = float(np.linalg.norm(a_size[:2]))
        b_xy_size = float(np.linalg.norm(b_size[:2]))
        return (centroid_z_diff_inv > 0.05 and
                centroid_z_diff_inv < 1.5 and
                centroid_xy_dist < a_xy_size * 0.8 and
                b_xy_size < a_xy_size * 2.0)

    if relation == "adjacent_to":
        # Objects must be at similar height and within reasonable XY distance.
        # Use 1.5× avg_size (same as geometry.py's proximity_factor) to avoid
        # firing for every pair on dense tabletop scenes.
        z_diff  = abs(float(delta[2]))
        dist_xy = float(np.linalg.norm(delta[:2]))
        return (z_diff  < avg_height * 1.5 and
                dist_xy < avg_size   * 1.5)

    if relation == "inside":
        # A must be smaller than B AND spatially contained within B's bbox
        # (with generous 20% tolerance to handle bbox imprecision)
        a_vol = float(np.prod(a_size))
        b_vol = float(np.prod(b_size))
        tolerance = b_size * 0.2
        contained = (np.all(a_min >= b_min - tolerance) and
                     np.all(a_max <= b_max + tolerance))
        return a_vol < b_vol * 0.8 and contained

    if relation == "hanging_from":
        # A must be above B
        return a_center[2] > b_center[2]

    if relation == "attached_to":
        # Objects must be physically touching or nearly touching (very small gap).
        # "attached_to" in ScanNet means wall-mounted / built-in — requires
        # actual contact, not just proximity.  Use a tight gap threshold:
        # bboxes must be within 5% of avg_size of each other.
        from src.relations.geometry import _min_bbox_gap
        gap = _min_bbox_gap(a_min, a_max, b_min, b_max)
        return gap < avg_size * 0.05

    # Unknown relation — don't filter
    return True


def print_scene_graph(result: dict):
    """Pretty print the predicted scene graph."""
    objects = result["objects"]
    relations = result["relations"]

    print(f"\n{'='*60}")
    print("PREDICTED SCENE GRAPH")
    print(f"{'='*60}")
    print(f"\nObjects ({len(objects)}):")
    for o in objects:
        print(f"  [{o.uid}] {o.label} — {o.point_count} pts, z={o.centroid[2]:.2f}")

    print(f"\nRelations ({len(relations)}):")
    for r in sorted(relations, key=lambda x: -x["confidence"]):
        desc = RELATION_DESCRIPTIONS.get(Relation[r["relation"].upper()], r["relation"])
        subj = f"Obj_{r['subject_id']}({r['subject_label']})"
        obj  = f"Obj_{r['object_id']}({r['object_label']})"
        print(f"  {subj} {desc} {obj}  (conf={r['confidence']:.2f})")

    print(f"\nNatural Language:")
    seen = set()
    for r in sorted(relations, key=lambda x: -x["confidence"]):
        key = (r["subject_id"], r["relation"], r["object_id"])
        if key in seen:
            continue
        seen.add(key)
        desc = RELATION_DESCRIPTIONS.get(Relation[r["relation"].upper()], r["relation"])
        subj = r["subject_label"] if r["subject_label"] != "object" else f"Object_{r['subject_id']}"
        obj  = r["object_label"]  if r["object_label"]  != "object" else f"Object_{r['object_id']}"
        print(f"  • The {subj} {desc} the {obj}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scene_01",
                        help="Scene name (looks for data/processed/<scene>/splat.ply)")
    parser.add_argument("--ply", default=None,
                        help="Direct path to splat.ply (overrides --scene)")
    parser.add_argument("--model", default=None,
                        help="Path to trained model .pt file")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Confidence threshold for relation predictions")
    parser.add_argument("--labeler", default="auto",
                        choices=["auto", "yolo", "dino", "none"],
                        help="Semantic labeler: auto=load cache, yolo, dino, none")
    parser.add_argument("--n-frames", type=int, default=30,
                        help="Frames to sample for labeling")
    parser.add_argument("--force-label", action="store_true",
                        help="Re-run labeling even if cached labels exist")
    parser.add_argument("--mode", default="hybrid",
                        choices=["hybrid", "geometry", "gnn", "ensemble"],
                        help="Inference mode: hybrid (default), geometry, gnn, ensemble")
    args = parser.parse_args()

    ply_path  = args.ply or os.path.join(DATA_DIR, args.scene, "splat.ply")
    scene_dir = args.ply and os.path.dirname(args.ply) or os.path.join(DATA_DIR, args.scene)

    if not os.path.exists(ply_path):
        print(f"splat.ply not found: {ply_path}")
        sys.exit(1)

    result = run_inference(
        ply_path,
        model_path=args.model,
        confidence_threshold=args.threshold,
        labeler=args.labeler,
        n_label_frames=args.n_frames,
        scene_dir=scene_dir,
        mode=args.mode,
    )
    print_scene_graph(result)
