"""
3DSSG Dataset Loader.

Loads objects and spatial relations from 3DSSG annotations.
Maps 3DSSG relation names to our LogicSplat relation schema.
Produces (node_features, edge_index, edge_labels) for GNN training.

Note: Until 3RScan point clouds arrive, node features are semantic only.
Geometry features plug in as a drop-in upgrade when 3RScan arrives.
"""
import json
import os
import numpy as np
from typing import List, Dict, Optional
from torch.utils.data import Dataset
import torch
from src.relations.schema import DSSG_TO_SCHEMA, NUM_RELATIONS, Relation


# ── object feature encoding ───────────────────────────────────────────────────

def _load_class_index(classes_path: str) -> Dict[str, int]:
    idx = {}
    with open(classes_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                idx[parts[1].lower()] = int(parts[0])
    return idx


def encode_object(obj: dict, class_index: Dict[str, int], num_classes: int) -> np.ndarray:
    """
    Encode a 3DSSG object into a fixed-size semantic feature vector.
    Geometry features (centroid, bbox) will be appended when 3RScan arrives.

    Features:
        [0] class_id normalized
        [1] has_color attribute
        [2] has_shape attribute
        [3] has_state attribute
        [4] affordance: placing items on
        [5] affordance: sitting on
        [6] affordance: hanging on
        [7] affordance: storing items
    """
    label = obj.get("label", "").lower()
    class_id = class_index.get(label, 0)
    class_norm = class_id / max(num_classes, 1)

    attrs = obj.get("attributes", {})
    has_color = 1.0 if attrs.get("color") else 0.0
    has_shape = 1.0 if attrs.get("shape") else 0.0
    has_state = 1.0 if attrs.get("state") else 0.0

    affordances = [a.lower() for a in obj.get("affordances", [])]
    aff_place = 1.0 if any("placing" in a for a in affordances) else 0.0
    aff_sit   = 1.0 if any("sitting" in a for a in affordances) else 0.0
    aff_hang  = 1.0 if any("hanging" in a for a in affordances) else 0.0
    aff_store = 1.0 if any("storing" in a for a in affordances) else 0.0

    return np.array([
        class_norm, has_color, has_shape, has_state,
        aff_place, aff_sit, aff_hang, aff_store,
    ], dtype=np.float32)


NODE_FEATURE_DIM = 8  # increases when geometry is added
EDGE_FEATURE_DIM = 4  # semantic edge features (geometry added when 3RScan arrives)


def encode_edge(obj_a: dict, obj_b: dict, class_index: Dict[str, int], num_classes: int) -> np.ndarray:
    """
    Semantic edge features between object pair (A, B).
    Geometry features (delta_z, xy_dist etc.) added when 3RScan arrives.

    Features:
        [0] same class (0/1)
        [1] a_class_norm
        [2] b_class_norm
        [3] shared affordance count (normalized)
    """
    label_a = obj_a.get("label", "").lower()
    label_b = obj_b.get("label", "").lower()
    same_class = 1.0 if label_a == label_b else 0.0
    class_a = class_index.get(label_a, 0) / max(num_classes, 1)
    class_b = class_index.get(label_b, 0) / max(num_classes, 1)

    aff_a = set(obj_a.get("affordances", []))
    aff_b = set(obj_b.get("affordances", []))
    shared = len(aff_a & aff_b) / max(len(aff_a | aff_b), 1)

    return np.array([same_class, class_a, class_b, shared], dtype=np.float32)


# ── scene graph builder ───────────────────────────────────────────────────────

def build_scene_graph(
    objects: List[dict],
    relationships: List[list],
    class_index: Dict[str, int],
    num_classes: int,
) -> Optional[Dict]:
    """
    Convert a 3DSSG scene into tensors for GNN training.
    Only keeps spatial/physical relations from our schema.
    Appearance relations (same color, same shape etc.) are discarded.
    """
    id_to_idx = {obj["id"]: i for i, obj in enumerate(objects)}

    x = np.stack([
        encode_object(obj, class_index, num_classes)
        for obj in objects
    ])

    src, dst, labels, edge_feats = [], [], [], []
    for rel in relationships:
        subj_id = str(rel[0])
        obj_id  = str(rel[1])
        rel_name = rel[3]

        if rel_name not in DSSG_TO_SCHEMA:
            continue
        if subj_id not in id_to_idx or obj_id not in id_to_idx:
            continue

        i = id_to_idx[subj_id]
        j = id_to_idx[obj_id]
        src.append(i)
        dst.append(j)
        labels.append(int(DSSG_TO_SCHEMA[rel_name]))
        edge_feats.append(encode_edge(objects[i], objects[j], class_index, num_classes))

    if len(src) == 0:
        return None

    return {
        "x":          torch.tensor(x, dtype=torch.float32),
        "edge_index": torch.tensor([src, dst], dtype=torch.long),
        "edge_attr":  torch.tensor(np.stack(edge_feats), dtype=torch.float32),
        "edge_label": torch.tensor(labels, dtype=torch.long),
        "obj_labels": [obj["label"] for obj in objects],
    }


# ── dataset class ─────────────────────────────────────────────────────────────

class SceneGraphDataset3DSSG(Dataset):
    """
    PyTorch Dataset wrapping 3DSSG scenes as scene graphs.
    Each item is one scene with node features and relation labels.
    """

    def __init__(self, data_dir: str = "data/3DSSG"):
        obj_path = os.path.join(data_dir, "objects.json")
        rel_path = os.path.join(data_dir, "relationships.json")
        cls_path = os.path.join(data_dir, "classes.txt")

        with open(obj_path) as f:
            obj_data = json.load(f)
        with open(rel_path) as f:
            rel_data = json.load(f)

        class_index = _load_class_index(cls_path)
        num_classes = len(class_index)

        obj_map = {s["scan"]: s["objects"] for s in obj_data["scans"]}
        rel_map = {s["scan"]: s["relationships"] for s in rel_data["scans"]}

        self.graphs = []
        for scan_id in obj_map:
            if scan_id not in rel_map:
                continue
            graph = build_scene_graph(
                obj_map[scan_id],
                rel_map[scan_id],
                class_index,
                num_classes,
            )
            if graph is not None:
                graph["scan_id"] = scan_id
                self.graphs.append(graph)

        print(f"Loaded {len(self.graphs)} scenes")
        self._print_stats()

    def _print_stats(self):
        from collections import Counter
        from src.relations.schema import RELATION_NAMES
        all_labels = []
        for g in self.graphs:
            all_labels.extend(g["edge_label"].tolist())
        counts = Counter(all_labels)
        print("Relation distribution:")
        for idx, count in sorted(counts.items()):
            print(f"  {RELATION_NAMES[idx]:20s} {count:6d}")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
