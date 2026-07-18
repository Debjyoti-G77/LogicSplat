"""
3RScan Gaussian Splat Loader for GNN Training.

Loads 3RScan Gaussian splat .ply files, clusters them into objects,
matches clusters to 3DSSG annotations, and builds graph tensors
for training the v7 dual-head RelationGNN.

Pipeline:
    1. Load .ply from D:/3rscan_splats/{scene_id}/ckpts/point_cloud_30000.ply
    2. Filter + prune Gaussians
    3. Cluster into objects via HDBSCAN (using 3DSSG object count as hint)
    4. Match clusters to 3DSSG object IDs by spatial ordering
    5. Extract 10-dim node features + 17-dim edge features
    6. Build multi-hot edge labels from 3DSSG relationships
    7. Cache processed graphs to D:/logicsplat_data/3rscan_cache/

Author: LogicSplat Team
"""
import os
import json
import warnings
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset

from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
from src.gaussian.clustering import (
    gaussian_to_objects,
    extract_gaussian_node_features,
    extract_gaussian_edge_features,
)
from src.relations.schema import DSSG_TO_SCHEMA, NUM_RELATIONS, Relation


# ── Cache version ─────────────────────────────────────────────────────────────
CACHE_VERSION = "v1_3rscan_splat"


# ── 3DSSG annotation loader ──────────────────────────────────────────────────

def load_3dssg_annotations(
    objects_path: str = "data/3DSSG/objects.json",
    relationships_path: str = "data/3DSSG/relationships.json",
) -> Tuple[Dict[str, List[dict]], Dict[str, List[list]]]:
    """
    Load 3DSSG objects and relationships indexed by scan_id.

    Returns:
        objects_by_scan: {scan_id: [obj_dict, ...]}
        rels_by_scan:    {scan_id: [[subj_id, obj_id, rel_type_id, rel_name], ...]}
    """
    with open(objects_path, "r") as f:
        objects_data = json.load(f)
    with open(relationships_path, "r") as f:
        rels_data = json.load(f)

    objects_by_scan = {}
    for scan_entry in objects_data["scans"]:
        scan_id = scan_entry["scan"]
        objects_by_scan[scan_id] = scan_entry["objects"]

    rels_by_scan = {}
    for scan_entry in rels_data["scans"]:
        scan_id = scan_entry["scan"]
        rels_by_scan[scan_id] = scan_entry["relationships"]

    return objects_by_scan, rels_by_scan


# ── Cluster-to-annotation matching ───────────────────────────────────────────

def match_clusters_to_annotations(
    objects_list,  # List[Object3D] from clustering
    gt_objects: List[dict],  # 3DSSG objects for this scene
) -> Optional[Dict[int, str]]:
    """
    Match HDBSCAN clusters to 3DSSG object IDs using canonical spatial ordering.

    Strategy: Sort both clusters and GT objects by a canonical order
    (Z centroid descending, then X, then Y) and match 1:1.

    Only succeeds if cluster count == GT object count.

    Args:
        objects_list: clustered Object3D instances (from gaussian_to_objects)
        gt_objects:   3DSSG object dicts with 'id' and 'label' fields

    Returns:
        {cluster_uid: gt_object_id} mapping, or None if counts don't match
    """
    n_clusters = len(objects_list)
    n_gt = len(gt_objects)

    if n_clusters != n_gt:
        return None

    # Sort clusters by canonical spatial order: Z desc, then X asc, then Y asc
    sorted_clusters = sorted(
        objects_list,
        key=lambda o: (-o.centroid[2], o.centroid[0], o.centroid[1])
    )

    # Sort GT objects by their ID (integer order — this is the canonical GT order)
    sorted_gt = sorted(gt_objects, key=lambda o: int(o["id"]))

    mapping = {}
    for cluster, gt_obj in zip(sorted_clusters, sorted_gt):
        mapping[cluster.uid] = gt_obj["id"]

    return mapping


# ── Scene processing ──────────────────────────────────────────────────────────

def process_scene(
    scene_id: str,
    splats_dir: str,
    gt_objects: List[dict],
    gt_relationships: List[list],
) -> Optional[Dict]:
    """
    Process a single 3RScan scene: load splat → cluster → extract features → build graph.

    Args:
        scene_id:         3RScan scan UUID
        splats_dir:       root directory containing splat files
        gt_objects:       3DSSG objects for this scene
        gt_relationships: 3DSSG relationships for this scene

    Returns:
        Graph dict {"x", "edge_index", "edge_attr", "edge_label", "scene_id"}
        or None if processing fails.
    """
    # ── 1. Load Gaussian splat ────────────────────────────────────────────────
    ply_path = os.path.join(splats_dir, scene_id, "ckpts", "point_cloud_30000.ply")
    if not os.path.exists(ply_path):
        return None

    try:
        cloud = load_gaussian_ply(ply_path)
    except Exception as e:
        warnings.warn(f"Failed to load {scene_id}: {e}", RuntimeWarning)
        return None

    if cloud.num_gaussians < 100:
        return None

    # ── 2. Filter and prune ───────────────────────────────────────────────────
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    if cloud.num_gaussians < 50:
        return None

    cloud = prune_isolated_gaussians(cloud)
    if cloud.num_gaussians < 50:
        return None

    # ── 3. Z-flip (same as run_inference) ─────────────────────────────────────
    # 3RScan scenes may have inverted Z axis; flip if most mass is above median
    z_median = np.median(cloud.xyz[:, 2])
    if np.mean(cloud.xyz[:, 2] > z_median) > 0.6:
        cloud.xyz[:, 2] = -cloud.xyz[:, 2]

    # ── 4. Cluster into objects ───────────────────────────────────────────────
    n_gt_objects = len(gt_objects)
    objects, params = gaussian_to_objects(
        cloud,
        target_min=max(2, n_gt_objects - 1),
        target_max=n_gt_objects + 1,
    )

    if len(objects) == 0:
        return None

    # ── 5. Match clusters to GT objects ───────────────────────────────────────
    # Only use scenes where cluster count exactly matches GT object count
    cluster_mapping = match_clusters_to_annotations(objects, gt_objects)
    if cluster_mapping is None:
        return None

    # Build reverse mapping: gt_object_id → cluster_uid
    gt_id_to_cluster_uid = {gt_id: uid for uid, gt_id in cluster_mapping.items()}

    # ── 6. Extract node features ──────────────────────────────────────────────
    all_xyz = np.concatenate([cloud.xyz], axis=0)
    scene_min = all_xyz.min(axis=0)
    scene_max = all_xyz.max(axis=0)
    scene_extent = scene_max - scene_min

    node_features = []
    for obj in objects:
        feat = extract_gaussian_node_features(obj, scene_extent, scene_min)
        node_features.append(feat)
    x = np.stack(node_features)  # (N, 10)

    # ── 7. Build ALL directed edges + extract edge features ───────────────────
    n_objects = len(objects)
    if n_objects < 2:
        return None

    src_list, dst_list, edge_feats = [], [], []
    for i in range(n_objects):
        for j in range(n_objects):
            if i == j:
                continue
            feat = extract_gaussian_edge_features(objects[i], objects[j], scene_extent)
            src_list.append(i)
            dst_list.append(j)
            edge_feats.append(feat)

    edge_index = np.array([src_list, dst_list])  # (2, E)
    edge_attr = np.stack(edge_feats)  # (E, 17)

    # ── 8. Build multi-hot edge labels from 3DSSG annotations ─────────────────
    # Create mapping from (cluster_idx_i, cluster_idx_j) → edge position
    edge_to_pos = {}
    for pos, (i, j) in enumerate(zip(src_list, dst_list)):
        edge_to_pos[(i, j)] = pos

    n_edges = len(src_list)
    edge_label = np.zeros((n_edges, NUM_RELATIONS), dtype=np.float32)

    # Map GT object IDs to cluster indices
    gt_id_to_cluster_idx = {}
    for obj in objects:
        gt_id = cluster_mapping.get(obj.uid)
        if gt_id is not None:
            gt_id_to_cluster_idx[str(gt_id)] = obj.uid

    # Object uid → index in objects list
    uid_to_idx = {obj.uid: idx for idx, obj in enumerate(objects)}

    for rel in gt_relationships:
        # Format: [subject_id, object_id, rel_type_id, "relation_name"]
        if len(rel) < 4:
            continue

        subj_gt_id = str(rel[0])
        obj_gt_id = str(rel[1])
        rel_name = rel[3] if isinstance(rel[3], str) else str(rel[3])

        # Map relation name to our schema
        if rel_name not in DSSG_TO_SCHEMA:
            continue

        relation_idx = int(DSSG_TO_SCHEMA[rel_name])

        # Map GT object IDs to cluster indices
        subj_cluster_uid = gt_id_to_cluster_idx.get(subj_gt_id)
        obj_cluster_uid = gt_id_to_cluster_idx.get(obj_gt_id)

        if subj_cluster_uid is None or obj_cluster_uid is None:
            continue

        subj_idx = uid_to_idx.get(subj_cluster_uid)
        obj_idx = uid_to_idx.get(obj_cluster_uid)

        if subj_idx is None or obj_idx is None:
            continue

        edge_pos = edge_to_pos.get((subj_idx, obj_idx))
        if edge_pos is not None:
            edge_label[edge_pos, relation_idx] = 1.0

    # ── 9. Validate and return ────────────────────────────────────────────────
    # Check that we have at least some positive labels
    if edge_label.sum() == 0:
        return None

    return {
        "x": torch.tensor(x, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
        "edge_attr": torch.tensor(edge_attr, dtype=torch.float32),
        "edge_label": torch.tensor(edge_label, dtype=torch.float32),
        "scene_id": scene_id,
    }


# ── Dataset class ─────────────────────────────────────────────────────────────

class Dataset3RScanSplat(Dataset):
    """
    PyTorch Dataset for 3RScan Gaussian splat scene graphs.

    Loads from cache if available, otherwise processes raw splats.
    """

    def __init__(
        self,
        splats_dir: str = "D:/3rscan_splats",
        cache_dir: str = "D:/logicsplat_data/3rscan_cache",
        objects_path: str = "data/3DSSG/objects.json",
        relationships_path: str = "data/3DSSG/relationships.json",
        max_scenes: Optional[int] = None,
        verbose: bool = True,
    ):
        self.splats_dir = splats_dir
        self.cache_dir = cache_dir
        self.graphs: List[Dict] = []

        os.makedirs(cache_dir, exist_ok=True)

        # Check cache first
        cache_files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
        if cache_files:
            if verbose:
                print(f"Loading {len(cache_files)} cached graphs from {cache_dir}...")
            self._load_cache(cache_files, max_scenes, verbose)
            return

        # Build from scratch
        if verbose:
            print("No cache found. Processing 3RScan splats...")

        objects_by_scan, rels_by_scan = load_3dssg_annotations(
            objects_path, relationships_path
        )

        # Find scenes that have both splats and annotations
        available_scenes = self._find_available_scenes(
            splats_dir, objects_by_scan, rels_by_scan
        )
        if verbose:
            print(f"Found {len(available_scenes)} scenes with splats + annotations")

        if max_scenes:
            available_scenes = available_scenes[:max_scenes]

        self._process_scenes(available_scenes, objects_by_scan, rels_by_scan, verbose)

        if verbose:
            print(f"\nSuccessfully processed {len(self.graphs)} scenes "
                  f"(cached to {cache_dir})")

    def _find_available_scenes(
        self,
        splats_dir: str,
        objects_by_scan: Dict,
        rels_by_scan: Dict,
    ) -> List[str]:
        """Find scene IDs that have both splat files and 3DSSG annotations."""
        available = []

        # Get all scene dirs in splats_dir
        if not os.path.isdir(splats_dir):
            print(f"Splats directory not found: {splats_dir}")
            return []

        for scene_id in os.listdir(splats_dir):
            ply_path = os.path.join(
                splats_dir, scene_id, "ckpts", "point_cloud_30000.ply"
            )
            if not os.path.exists(ply_path):
                continue
            if scene_id not in objects_by_scan:
                continue
            if scene_id not in rels_by_scan:
                continue
            # Must have at least 2 objects and some relationships
            if len(objects_by_scan[scene_id]) < 2:
                continue
            if len(rels_by_scan[scene_id]) < 1:
                continue
            available.append(scene_id)

        return sorted(available)

    def _process_scenes(
        self,
        scene_ids: List[str],
        objects_by_scan: Dict,
        rels_by_scan: Dict,
        verbose: bool,
    ):
        """Process all scenes, caching successful ones."""
        try:
            from tqdm import tqdm
            iterator = tqdm(scene_ids, desc="Processing scenes", unit="scene")
        except ImportError:
            iterator = scene_ids

        n_success = 0
        n_failed = 0
        n_mismatch = 0

        for scene_id in iterator:
            gt_objects = objects_by_scan[scene_id]
            gt_rels = rels_by_scan[scene_id]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                graph = process_scene(
                    scene_id=scene_id,
                    splats_dir=self.splats_dir,
                    gt_objects=gt_objects,
                    gt_relationships=gt_rels,
                )

            if graph is None:
                n_failed += 1
                continue

            # Validate dimensions
            if graph["x"].shape[1] != 10:
                n_failed += 1
                continue
            if graph["edge_attr"].shape[1] != 17:
                n_failed += 1
                continue

            # Cache
            cache_path = os.path.join(
                self.cache_dir, f"{scene_id}_{CACHE_VERSION}.pt"
            )
            torch.save(graph, cache_path)
            self.graphs.append(graph)
            n_success += 1

        if verbose:
            print(f"\nProcessing complete:")
            print(f"  Success: {n_success}")
            print(f"  Failed (load/cluster/no-labels): {n_failed}")

    def _load_cache(
        self,
        cache_files: List[str],
        max_scenes: Optional[int],
        verbose: bool,
    ):
        """Load graphs from cache directory."""
        try:
            from tqdm import tqdm
            iterator = tqdm(cache_files, desc="Loading cache", unit="file")
        except ImportError:
            iterator = cache_files

        for fname in iterator:
            if max_scenes and len(self.graphs) >= max_scenes:
                break
            path = os.path.join(self.cache_dir, fname)
            try:
                g = torch.load(path, weights_only=False)
                # Validate
                if g["x"].shape[1] != 10:
                    continue
                if g["edge_attr"].shape[1] != 17:
                    continue
                if g["edge_label"].dim() != 2 or g["edge_label"].shape[1] != NUM_RELATIONS:
                    continue
                self.graphs.append(g)
            except Exception as e:
                if verbose:
                    print(f"  Skipping corrupt cache: {fname}: {e}")

        if verbose:
            print(f"Loaded {len(self.graphs)} valid graphs from cache")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Dict:
        return self.graphs[idx]
