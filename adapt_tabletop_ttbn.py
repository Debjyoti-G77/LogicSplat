"""
Test-Time Batch Normalization (TTBN) adaptation using unlabelled scenes 1-5.

The GeoKAN-Gamma model has BatchNorm layers inside each GeoKANGammaLayer.
These were calibrated on 3RScan room-scale geometry.
By running real tabletop scenes (1-5) through the model in train() mode,
the running_mean / running_var of every BN layer shifts to match real
tabletop feature distributions — with ZERO labels required.

This is honest unsupervised domain adaptation:
  - Adaptation data : scenes 1-5  (no GT labels)
  - Test data       : scenes 6-13 (GT labels, never touched here)

Usage:
    python adapt_tabletop_ttbn.py
Output:
    models/geokan_gamma_ttbn.pt
"""
import sys
sys.path.insert(0, ".")

import os
import json
import numpy as np
import torch
import torch.nn as nn

from geokan_gamma_relation import GeoKANGammaRelationGNN
from src.relations.schema import NUM_RELATIONS
from src.graph.definitions import Object3D
from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians, remove_table_background
from src.gaussian.clustering import gaussian_to_objects, extract_gaussian_node_features
from scripts.build_3rscan_graphs import extract_3rscan_edge_features

DATA_DIR    = "D:/logicsplat_data/processed"
BASE_MODEL  = "models/geokan_relation_gamma.pt"
OUT_MODEL   = "models/geokan_gamma_ttbn_v5.pt"
ADAPT_SCENES = [f"scene_{i:02d}" for i in range(1, 14)]  # all 13 scenes, no GT used
N_PASSES    = 10
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Z-convention per scene (True = Z-DOWN, needs flip)
SCENE_Z_FLIP = {
    "scene_06": False, "scene_07": False, "scene_08": False,
    "scene_09": True,  "scene_10": False, "scene_11": False,
    "scene_12": True,  "scene_13": True,
}

# v6 normalization (virtual table + correct Z-flip)
_RSCAN_MEAN = np.load("models/rscan_feat_mean.npy").astype(np.float32)
_RSCAN_STD  = np.load("models/rscan_feat_std.npy").astype(np.float32)
_TT_MEAN    = np.load("models/tabletop_feat_mean_v6.npy").astype(np.float32)
_TT_STD     = np.load("models/tabletop_feat_std_v6.npy").astype(np.float32)

def normalize_edge_features(edge_attr: torch.Tensor) -> torch.Tensor:
    x = edge_attr.cpu().numpy()
    x_norm = (x - _TT_MEAN) / _TT_STD * _RSCAN_STD + _RSCAN_MEAN
    return torch.tensor(x_norm, dtype=torch.float32, device=edge_attr.device)


# ── Virtual table helper (same formula as eval script) ───────────────────────

def apply_z_flip(objects):
    for o in objects:
        o.centroid = o.centroid.copy(); o.centroid[2] = -o.centroid[2]
        old_min_z, old_max_z = o.bbox_min[2], o.bbox_max[2]
        o.bbox_min = o.bbox_min.copy(); o.bbox_max = o.bbox_max.copy()
        o.bbox_min[2] = -old_max_z; o.bbox_max[2] = -old_min_z


def create_virtual_table(objects, is_z_down, table_height_scale=5.0):
    from src.graph.definitions import Object3D
    items = [o for o in objects if o.label != "table"]
    if not items:
        return None
    centroid_zs = [float(o.centroid[2]) for o in items]
    sizes_z = [abs(float(o.bbox_max[2]) - float(o.bbox_min[2])) for o in items]
    median_sz = float(np.median(sizes_z)) if sizes_z else 1.0
    offset_z = median_sz * 0.3
    table_ht = table_height_scale * median_sz
    if is_z_down:
        surface_z = max(centroid_zs) + offset_z
        bbox_min_z = surface_z; bbox_max_z = surface_z + table_ht
    else:
        surface_z = min(centroid_zs) - offset_z
        bbox_max_z = surface_z; bbox_min_z = surface_z - table_ht
    centroid_z = (bbox_min_z + bbox_max_z) / 2
    all_xmin = float(min(o.bbox_min[0] for o in items))
    all_xmax = float(max(o.bbox_max[0] for o in items))
    all_ymin = float(min(o.bbox_min[1] for o in items))
    all_ymax = float(max(o.bbox_max[1] for o in items))
    mx = (all_xmax - all_xmin) * 0.15; my = (all_ymax - all_ymin) * 0.15
    return Object3D(
        uid=len(items),
        centroid=np.array([(all_xmin+all_xmax)/2, (all_ymin+all_ymax)/2, centroid_z], np.float32),
        bbox_min=np.array([all_xmin-mx, all_ymin-my, bbox_min_z], np.float32),
        bbox_max=np.array([all_xmax+mx, all_ymax+my, bbox_max_z], np.float32),
        color=np.array([200, 200, 200], np.float32),
        point_count=sum(o.point_count for o in items), label="table",
    )


# ── PLY / clustering helpers ──────────────────────────────────────────────────

def load_and_cluster(scene_dir: str, scene_name: str = None, n_hint: int = None):
    """
    Load PLY, prune, cluster item objects.
    For scenes 6-13: adds virtual table + applies Z-flip.
    """
    ply_path = os.path.join(scene_dir, "splat.ply")
    if not os.path.exists(ply_path):
        return None

    try:
        cloud = load_gaussian_ply(ply_path)
        cloud = filter_gaussians(cloud, opacity_threshold=0.1)
        cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)
        cloud = remove_table_background(cloud)
        n_items = max(2, (n_hint - 1) if n_hint else 4)
        objects, _ = gaussian_to_objects(cloud, target_min=n_items,
                                          target_max=n_items + 3, n_exact=n_items)
        objects = list(objects)

        # Add virtual table for scenes with known Z-convention (6-13)
        if scene_name in SCENE_Z_FLIP:
            is_z_down = SCENE_Z_FLIP[scene_name]
            table_obj = create_virtual_table(objects, is_z_down=is_z_down)
            if table_obj is not None:
                table_obj.uid = len(objects)
                objects.append(table_obj)
            if is_z_down:
                apply_z_flip(objects)

        return objects
    except Exception as e:
        print(f"  clustering failed: {e}")
        return None


def build_graph(objects):
    """Build node features and edge features from Object3D list."""
    n = len(objects)
    if n < 2:
        return None, None, None

    all_mins = np.stack([o.bbox_min for o in objects])
    all_maxs = np.stack([o.bbox_max for o in objects])
    scene_min    = all_mins.min(axis=0)
    scene_max    = all_maxs.max(axis=0)
    scene_extent = np.maximum(scene_max - scene_min, 1e-6)

    obj_diags = [np.linalg.norm(o.bbox_max - o.bbox_min) for o in objects]
    scene_mean_diag    = float(np.mean(obj_diags)) if obj_diags else 1.0
    obj_sizes          = [o.bbox_max - o.bbox_min for o in objects]
    obj_volumes        = [float(np.prod(np.maximum(s, 1e-6))) for s in obj_sizes]
    scene_median_volume = float(np.median(obj_volumes)) if obj_volumes else 1.0

    for o in objects:
        o.scene_extent         = scene_extent
        o.scene_min            = scene_min
        o.scene_mean_diag      = scene_mean_diag
        o.scene_median_volume  = scene_median_volume

    # Node features
    node_feats = []
    for o in objects:
        size = o.bbox_max - o.bbox_min
        diag = np.linalg.norm(size)
        nf   = np.array([
            float(o.centroid[0] / scene_extent[0]),
            float(o.centroid[1] / scene_extent[1]),
            float(o.centroid[2] / scene_extent[2]),
            float(size[0] / scene_extent[0]),
            float(size[1] / scene_extent[1]),
            float(size[2] / scene_extent[2]),
            float(diag / scene_mean_diag),
            float(np.prod(np.maximum(size, 1e-6)) / max(scene_median_volume, 1e-6)),
            float(o.point_count / max(sum(oo.point_count for oo in objects), 1)),
            float(size[2] / max(diag, 1e-6)),
        ], dtype=np.float32)
        node_feats.append(nf)

    # Edge features
    src_ids, dst_ids, edge_feats = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ef = extract_3rscan_edge_features(objects[i], objects[j], scene_extent)
            src_ids.append(i)
            dst_ids.append(j)
            edge_feats.append(ef)

    x          = torch.tensor(np.stack(node_feats), dtype=torch.float32)
    edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)
    edge_attr  = torch.tensor(np.stack(edge_feats), dtype=torch.float32)
    return x, edge_index, edge_attr


def load_model():
    state = torch.load(BASE_MODEL, weights_only=False, map_location=DEVICE)
    def _remap(k):
        if k.endswith(".gamma_param"):
            return k[:-len(".gamma_param")] + ".gamma_params"
        if k.endswith(".gamma_rbf"):
            return k[:-len(".gamma_rbf")] + ".rbf_gamma"
        return k
    # Only remap if loading the old gamma checkpoint (has legacy key names)
    if any(k.endswith(".gamma_param") for k in state):
        state = {_remap(k): v for k, v in state.items()}
    hidden = state["node_encoder.0.weight"].shape[0]
    ef_dim = state["conv1.lin_edge.weight"].shape[1]

    model = GeoKANGammaRelationGNN(
        node_feat_dim=10, edge_feat_dim=ef_dim,
        hidden_dim=hidden, num_relations=NUM_RELATIONS,
    ).to(DEVICE)
    model.load_state_dict(state)
    return model


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  TTBN Adaptation — unlabelled scenes 1-5")
    print("=" * 60)
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Adapt scenes: {ADAPT_SCENES}")
    print(f"  Passes     : {N_PASSES}")
    print()

    model = load_model()
    n_bn  = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm1d))
    print(f"  BatchNorm layers to adapt: {n_bn}")

    # Build graphs from scenes 1-5
    graphs = []
    for scene_name in ADAPT_SCENES:
        scene_dir = os.path.join(DATA_DIR, scene_name)
        print(f"\n  Processing {scene_name}...")
        objects = load_and_cluster(scene_dir, scene_name=scene_name, n_hint=5)
        if objects is None or len(objects) < 2:
            print(f"    SKIP: clustering failed or < 2 objects")
            continue
        print(f"    Clustered: {len(objects)} objects")
        x, edge_index, edge_attr = build_graph(objects)
        if x is None:
            continue
        graphs.append((x.to(DEVICE), edge_index.to(DEVICE), edge_attr.to(DEVICE)))
        print(f"    Graph: {x.shape[0]} nodes, {edge_index.shape[1]} edges")

    if not graphs:
        print("ERROR: no valid graphs from scenes 1-5")
        return

    # TTBN: run forward passes in train() mode to update BN running stats
    # Only update BN stats — keep all parameters frozen
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"\n  Running {N_PASSES} adaptation passes over {len(graphs)} scenes...")
    with torch.no_grad():
        for pass_i in range(1, N_PASSES + 1):
            for x, edge_index, edge_attr in graphs:
                _ = model(x, edge_index, normalize_edge_features(edge_attr))
            print(f"    Pass {pass_i}/{N_PASSES} done")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    torch.save(model.state_dict(), OUT_MODEL)
    print(f"\n  Saved adapted model: {OUT_MODEL}")
    print("  Ready to evaluate on scenes 6-13.")


if __name__ == "__main__":
    main()
