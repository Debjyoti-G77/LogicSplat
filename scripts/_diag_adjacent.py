"""
Print GNN confidence scores for adjacent_to on scene_06.
Uses fixed mcs=861.
"""
import sys
sys.path.insert(0, ".")
import json
import numpy as np
import torch
from src.gaussian.loader import load_gaussian_ply, filter_gaussians
from src.gaussian.clustering import gaussian_to_objects
from src.inference.gaussian_inference import build_graph, find_model
from src.models.relation_gnn import RelationGNN
from src.relations.schema import NUM_RELATIONS, RELATION_NAMES

ply     = "D:/logicsplat_data/processed/scene_06/splat.ply"
gt_path = "D:/logicsplat_data/processed/scene_06/ground_truth_relations.json"

with open(gt_path) as f:
    gt = json.load(f)

cloud   = load_gaussian_ply(ply)
cf      = filter_gaussians(cloud, opacity_threshold=0.1)
objects, _ = gaussian_to_objects(cf, min_cluster_size=861)

# Z-flip
for o in objects:
    o.centroid = o.centroid.copy(); o.centroid[2] *= -1
    o.bbox_min = o.bbox_min.copy(); o.bbox_min[2] *= -1
    o.bbox_max = o.bbox_max.copy(); o.bbox_max[2] *= -1
    o.bbox_min[2], o.bbox_max[2] = min(o.bbox_min[2], o.bbox_max[2]), max(o.bbox_min[2], o.bbox_max[2])

# GT centroid matching
gt_objects = gt["objects"]
gt_cents = np.array([o["centroid"] for o in gt_objects], dtype=float)
gt_cents[:, 2] *= -1
cluster_cents = np.array([o.centroid for o in objects])
assigned = set()
uid_to_name = {}
for gi, gt_obj in enumerate(gt_objects):
    dists = np.linalg.norm(cluster_cents - gt_cents[gi], axis=1)
    for used in assigned:
        dists[used] = np.inf
    best = int(np.argmin(dists))
    uid_to_name[best] = gt_obj["name"]
    assigned.add(best)

# Build graph and run model
x, edge_index, edge_attr = build_graph(objects)
model_path = find_model()
model = RelationGNN(node_feat_dim=10, edge_feat_dim=14, hidden_dim=128, num_relations=NUM_RELATIONS)
model.load_state_dict(torch.load(model_path, weights_only=False, map_location="cpu"))
model.eval()

with torch.no_grad():
    logits = model(x, edge_index, edge_attr)
    probs = torch.sigmoid(logits).cpu().numpy()

adj_idx = RELATION_NAMES.index("adjacent_to")
gt_adjacent = {(r["subject"], r["object"]) for r in gt["relations"] if r["relation"] == "adjacent_to"}

print(f"adjacent_to confidence for all pairs (GT adjacent pairs marked *):")
print(f"{'Subject':<14} {'Object':<14} {'conf':>8}  GT")
print("-" * 45)
src_nodes = edge_index[0].tolist()
dst_nodes = edge_index[1].tolist()
rows = []
for idx, (s, d) in enumerate(zip(src_nodes, dst_nodes)):
    conf = float(probs[idx][adj_idx])
    sname = uid_to_name.get(s, f"obj{s}")
    dname = uid_to_name.get(d, f"obj{d}")
    is_gt = (sname, dname) in gt_adjacent
    rows.append((conf, sname, dname, is_gt))

rows.sort(reverse=True)
for conf, sname, dname, is_gt in rows:
    marker = " *" if is_gt else ""
    print(f"{sname:<14} {dname:<14} {conf:>8.4f}{marker}")
