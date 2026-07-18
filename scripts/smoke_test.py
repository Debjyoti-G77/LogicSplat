"""Smoke test — verifies all fixed components work correctly."""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import torch

passed = 0
failed = 0

def check(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✓ {name}")
        passed += 1
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        failed += 1

print("=" * 50)
print("Smoke Test — LogicSplat")
print("=" * 50)

# ── 1. Gaussian loader (vectorized covariance) ────────────────────────────────
print("\n[1] Gaussian Loader")

def test_loader():
    from src.gaussian.loader import load_gaussian_ply, filter_gaussians
    cloud = load_gaussian_ply("D:/logicsplat_data/processed/scene_01/splat.ply")
    assert cloud.num_gaussians > 0
    assert cloud.covariance.shape == (cloud.num_gaussians, 6)
    assert np.isfinite(cloud.covariance).all(), "NaN/Inf in covariance"
    filtered = filter_gaussians(cloud)
    assert filtered.num_gaussians < cloud.num_gaussians

check("load_gaussian_ply + vectorized covariance", test_loader)

# ── 2. Clustering ─────────────────────────────────────────────────────────────
print("\n[2] Clustering")

def test_clustering():
    from src.gaussian.loader import load_gaussian_ply, filter_gaussians
    from src.gaussian.clustering import gaussian_to_objects, extract_gaussian_node_features, extract_gaussian_edge_features
    cloud = load_gaussian_ply("D:/logicsplat_data/processed/scene_01/splat.ply")
    filtered = filter_gaussians(cloud)
    objects, params = gaussian_to_objects(filtered)
    assert len(objects) >= 2
    # feature dims
    scene_extent = np.array([1.0, 1.0, 1.0])
    scene_min = np.zeros(3)
    nf = extract_gaussian_node_features(objects[0], scene_extent, scene_min)
    ef = extract_gaussian_edge_features(objects[0], objects[1], scene_extent)
    assert nf.shape == (10,), f"node feat dim={nf.shape}"
    assert ef.shape == (10,), f"edge feat dim={ef.shape}"

check("gaussian_to_objects + feature extraction (10/10 dims)", test_clustering)

# ── 3. Geometry rules ─────────────────────────────────────────────────────────
print("\n[3] Geometry Rules")

def test_on_top_of():
    from src.relations.geometry import derive_relations
    from src.relations.schema import Relation
    # A on top of B: A at z=0.45-0.95, B at z=0-0.5
    # vertical_gap = a_min[2] - b_max[2] = 0.45 - 0.5 = -0.05 (slight overlap, valid)
    # avg_height = (0.5 + 0.5) / 2 = 0.5, threshold = 0.5 * 0.4 = 0.2
    # gap (-0.05) is in (-0.25, 0.2] ✓, XY footprints overlap ✓
    rels = derive_relations(
        np.array([0.1, 0.1, 0.45]), np.array([0.9, 0.9, 0.95]),
        np.array([0.0, 0.0, 0.00]), np.array([1.0, 1.0, 0.50])
    )
    assert Relation.ON_TOP_OF in rels, f"Expected ON_TOP_OF, got {[r.name for r in rels]}"

def test_left_of():
    from src.relations.geometry import derive_relations
    from src.relations.schema import Relation
    # A to the left of B: A at x=-1, B at x=1
    rels = derive_relations(
        np.array([-1.5, 0.0, 0.0]), np.array([-0.5, 1.0, 1.0]),
        np.array([0.5,  0.0, 0.0]), np.array([1.5,  1.0, 1.0])
    )
    assert Relation.LEFT_OF in rels, f"Expected LEFT_OF, got {rels}"

def test_inside():
    from src.relations.geometry import derive_relations
    from src.relations.schema import Relation
    # A inside B: A is small box inside large box
    rels = derive_relations(
        np.array([0.2, 0.2, 0.2]), np.array([0.4, 0.4, 0.4]),  # small A
        np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])   # large B
    )
    assert Relation.INSIDE in rels, f"Expected INSIDE, got {rels}"

check("ON_TOP_OF detection", test_on_top_of)
check("LEFT_OF detection", test_left_of)
check("INSIDE detection", test_inside)

# ── 4. Augmentation ───────────────────────────────────────────────────────────
print("\n[4] Augmentation")

def test_augmentation_shapes():
    from src.training.augmentation import augment_graph, AUGMENTATIONS
    g = {
        "x": torch.rand(5, 10),
        "edge_attr": torch.rand(20, 10),
        "edge_label": torch.randint(0, 12, (20,)),
        "scene_id": "test",
        "edge_index": torch.zeros(2, 20, dtype=torch.long),
        "obj_labels": [],
    }
    for name, _ in AUGMENTATIONS:
        aug = augment_graph(g, name)
        assert aug["x"].shape == g["x"].shape
        assert aug["edge_attr"].shape == g["edge_attr"].shape
        assert aug["edge_label"].shape == g["edge_label"].shape

def test_flip_x_remapping():
    from src.training.augmentation import augment_graph
    from src.relations.schema import Relation
    labels = torch.tensor([int(Relation.LEFT_OF), int(Relation.RIGHT_OF),
                           int(Relation.ON_TOP_OF)])
    g = {
        "x": torch.rand(3, 10),
        "edge_attr": torch.rand(3, 10),
        "edge_label": labels,
        "scene_id": "test",
        "edge_index": torch.zeros(2, 3, dtype=torch.long),
        "obj_labels": [],
    }
    aug = augment_graph(g, "flip_x")
    assert aug["edge_label"][0] == int(Relation.RIGHT_OF), "flip_x: left→right"
    assert aug["edge_label"][1] == int(Relation.LEFT_OF),  "flip_x: right→left"
    assert aug["edge_label"][2] == int(Relation.ON_TOP_OF), "flip_x: on_top_of unchanged"

check("All augmentation shapes preserved", test_augmentation_shapes)
check("flip_x label remapping correct", test_flip_x_remapping)

# ── 5. Definitions ────────────────────────────────────────────────────────────
print("\n[5] Data Structures")

def test_definitions():
    from src.graph.definitions import Object3D, RelationEdge, SceneGraph
    o1 = Object3D(0, np.zeros(3), np.zeros(3), np.ones(3), np.array([128,128,128]), 100, "router")
    o2 = Object3D(1, np.array([1,0,0]), np.array([0.5,0,0]), np.array([1.5,1,1]), np.array([200,100,50]), 50, "box")
    r = RelationEdge(0, "on_top_of", 1, 0.95)
    sg = SceneGraph("test", [o1, o2], [r])
    text = r.to_text([o1, o2])
    assert "router" in text and "box" in text, f"to_text missing labels: {text}"
    assert "on top of" in text, f"to_text missing relation: {text}"
    assert sg.get_object(0) is o1
    assert sg.get_object(99) is None

check("Object3D, RelationEdge, SceneGraph", test_definitions)

# ── 6. Inference pipeline ─────────────────────────────────────────────────────
print("\n[6] Inference Pipeline")

def test_inference():
    from src.inference.gaussian_inference import run_inference
    result = run_inference(
        "D:/logicsplat_data/processed/scene_01/splat.ply",
        labeler="none",
        confidence_threshold=0.55,
    )
    assert "objects" in result
    assert "relations" in result
    assert len(result["objects"]) >= 2
    assert len(result["relations"]) > 0

check("run_inference end-to-end", test_inference)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 50)
print(f"Results: {passed} passed, {failed} failed")
print("=" * 50)
if failed > 0:
    sys.exit(1)
