"""
RCA profiler — benchmarks old vs new PLY reader, and cache performance.
Run: python scripts/profile_scene_load.py
"""
import sys, time, os
sys.path.insert(0, ".")

import numpy as np
import json

SCENE_DIR = "D:/scannet/scans/scene0000_00"
SCENE_ID  = "scene0000_00"
SCANNET_DIR = "D:/scannet/scans"
CACHE_DIR   = "D:/logicsplat_data/scannet_cache"

print(f"=== PLY Reader Benchmark: {SCENE_ID} ===\n")

mesh_path   = f"{SCENE_DIR}/{SCENE_ID}_vh_clean_2.ply"
labels_path = f"{SCENE_DIR}/{SCENE_ID}_vh_clean_2.labels.ply"

# ── OLD: plyfile ──────────────────────────────────────────────────────────────
from plyfile import PlyData

t0 = time.perf_counter()
ply = PlyData.read(mesh_path)
v = ply["vertex"]
pts_old = np.stack([v["x"], v["y"], v["z"]], axis=1).astype("float32")
t1 = time.perf_counter()
print(f"plyfile  mesh read : {t1-t0:.3f}s  ({len(pts_old):,} vertices)")

t0 = time.perf_counter()
ply2 = PlyData.read(labels_path)
t1 = time.perf_counter()
print(f"plyfile  labels read: {t1-t0:.3f}s")

# ── NEW: open3d ───────────────────────────────────────────────────────────────
import open3d as o3d

t0 = time.perf_counter()
mesh = o3d.io.read_triangle_mesh(mesh_path)
pts_new = np.asarray(mesh.vertices, dtype=np.float32)
t1 = time.perf_counter()
print(f"\nopen3d   mesh read : {t1-t0:.3f}s  ({len(pts_new):,} vertices)")
print(f"Vertex match: {np.allclose(pts_old, pts_new, atol=1e-4)}")

# ── Full scene load + cache write ─────────────────────────────────────────────
print(f"\n=== Full Scene Load (first run — no cache) ===\n")
from src.dataset.loader_scannet import load_scannet_scene, build_scannet_scene_graph_geometric
import torch

cache_path = os.path.join(CACHE_DIR, f"{SCENE_ID}_v1.pt")
if os.path.exists(cache_path):
    os.remove(cache_path)
    print("(removed existing cache to force fresh load)")

os.makedirs(CACHE_DIR, exist_ok=True)

t0 = time.perf_counter()
scene_data = load_scannet_scene(SCENE_ID, SCANNET_DIR)
t1 = time.perf_counter()
graph = build_scannet_scene_graph_geometric(scene_data)
t2 = time.perf_counter()
torch.save(graph, cache_path)
t3 = time.perf_counter()

print(f"load_scannet_scene()     : {t1-t0:.3f}s")
print(f"build_scene_graph()      : {t2-t1:.3f}s")
print(f"torch.save() to cache    : {t3-t2:.3f}s")
print(f"TOTAL first run          : {t3-t0:.3f}s")
print(f"Projected 1468 scenes    : {(t3-t0)*1468/60:.1f} min")

# ── Cache read ────────────────────────────────────────────────────────────────
print(f"\n=== Cache Read (subsequent runs) ===\n")
t0 = time.perf_counter()
graph2 = torch.load(cache_path, weights_only=False)
t1 = time.perf_counter()
print(f"torch.load() from cache  : {t1-t0:.4f}s")
print(f"Projected 1468 scenes    : {(t1-t0)*1468:.1f}s  ({(t1-t0)*1468/60:.2f} min)")
print(f"\nCache file size: {os.path.getsize(cache_path)/1024:.1f} KB")
print(f"Estimated total cache size: {os.path.getsize(cache_path)*1468/1024/1024:.0f} MB")
