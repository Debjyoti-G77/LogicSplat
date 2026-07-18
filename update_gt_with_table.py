"""
Update GT JSONs for scenes 6-13:
  1. Add 'table' as an explicit object (virtual position from item centroids)
  2. Add 'X on_top_of table' + 'table under X' for ALL objects in the scene
     (every object in the scene is above the table, whether directly resting
      on it or stacked on another object)
  3. Add 'X higher_than table' + 'table lower_than X' for all objects
  4. Fix missing relation in scene_09: perfume on_top_of cream_tub

Z-conventions (from stacking pair analysis):
  Z-UP  (no flip): scene_06, 07, 08, 10, 11
  Z-DOWN (flip):   scene_09, 12, 13

Virtual table position formula:
  Z-UP  : surface_z = min(item_centroid_z) - 0.3 * z_range * 0.25
  Z-DOWN: surface_z = max(item_centroid_z) + 0.3 * z_range * 0.25
  Height: 5 * (z_range * 0.25)  (adaptive to scene scale)
"""
import sys, os, json, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np

DATA_DIR = "D:/logicsplat_data/processed"

# Z-convention per scene (True = Z-DOWN, needs flip)
SCENE_Z_FLIP = {
    "scene_06": False,
    "scene_07": False,
    "scene_08": False,
    "scene_09": True,
    "scene_10": False,
    "scene_11": False,
    "scene_12": True,
    "scene_13": True,
}

# ALL objects per scene (every object is on_top_of table in the broad sense)
ON_TABLE = {
    "scene_06": ["agaro_box", "water_bottle", "watch", "pen", "router"],
    "scene_07": ["pen", "router", "agaro_box", "water_bottle"],
    "scene_08": ["agaro_box", "pen", "perfume", "router"],
    "scene_09": ["router", "cream_tub", "watch", "perfume"],
    "scene_10": ["agaro_box", "cream_tub", "watch", "router"],
    "scene_11": ["router", "water_bottle", "cream_tub", "watch"],
    "scene_12": ["agaro_box", "water_bottle", "watch", "router"],
    "scene_13": ["agaro_box", "water_bottle", "perfume", "router"],
}

# Missing stacking relations
MISSING_STACKING = {
    "scene_09": [("perfume", "on_top_of", "cream_tub"),
                 ("cream_tub", "under", "perfume")],
}


def compute_virtual_table_position(gt_objects, is_z_down, table_height_scale=5.0):
    """
    Compute virtual table centroid + bbox from GT object centroids.

    Uses same formula as create_virtual_table() in eval script so that
    the predicted virtual table and GT table have matching Z coordinates,
    enabling correct Hungarian matching.
    """
    centroids   = np.array([o["centroid"] for o in gt_objects])
    centroid_zs = centroids[:, 2]

    z_range    = float(centroid_zs.max() - centroid_zs.min())
    median_sz  = z_range * 0.25 if z_range > 1e-6 else 0.1
    offset_z   = median_sz * 0.3
    table_ht   = table_height_scale * median_sz

    if is_z_down:
        surface_z  = float(centroid_zs.max()) + offset_z
        bbox_min_z = surface_z
        bbox_max_z = surface_z + table_ht
    else:
        surface_z  = float(centroid_zs.min()) - offset_z
        bbox_max_z = surface_z
        bbox_min_z = surface_z - table_ht

    centroid_z = (bbox_min_z + bbox_max_z) / 2

    xy_center = centroids[:, :2].mean(axis=0)
    margin    = 0.15
    centroid  = [float(xy_center[0]), float(xy_center[1]), float(centroid_z)]
    bbox_min  = [float(centroids[:, 0].min() - margin),
                 float(centroids[:, 1].min() - margin),
                 float(bbox_min_z)]
    bbox_max  = [float(centroids[:, 0].max() + margin),
                 float(centroids[:, 1].max() + margin),
                 float(bbox_max_z)]

    return centroid, bbox_min, bbox_max


for scene_name, on_table_objects in ON_TABLE.items():
    gt_path    = os.path.join(DATA_DIR, scene_name, "ground_truth_relations.json")
    backup_path = gt_path.replace(".json", "_backup.json")

    # Load original backup (to start fresh each run)
    if os.path.exists(backup_path):
        with open(backup_path) as f:
            gt = json.load(f)
        print(f"{scene_name}: loaded from backup")
    else:
        with open(gt_path) as f:
            gt = json.load(f)
        with open(backup_path, "w") as f:
            json.dump(gt, f, indent=2)
        print(f"{scene_name}: created backup, loaded original")

    is_z_down = SCENE_Z_FLIP.get(scene_name, False)

    # Compute virtual table position from original GT object centroids
    gt_items = [o for o in gt["objects"] if o["name"] != "table"]
    centroid, bbox_min, bbox_max = compute_virtual_table_position(gt_items, is_z_down)

    # Add table as last object
    table_id    = len(gt_items)
    table_point = int(np.prod(np.array(bbox_max) - np.array(bbox_min)) * 10000)
    gt["objects"] = gt_items + [{
        "id": table_id,
        "name": "table",
        "description": "table surface supporting all objects",
        "centroid": centroid,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "point_count": max(table_point, 100),
    }]

    # Collect existing relations to avoid duplicates
    existing = {(r["subject"], r["relation"], r["object"]) for r in gt["relations"]}
    new_rels  = []

    def add_rel(subj, rel, obj):
        triple = (subj, rel, obj)
        if triple not in existing:
            new_rels.append({"subject": subj, "relation": rel, "object": obj})
            existing.add(triple)

    # Fix missing stacking relations
    for subj, rel, obj in MISSING_STACKING.get(scene_name, []):
        add_rel(subj, rel, obj)

    # All objects are on_top_of / higher_than table
    obj_names = [o["name"] for o in gt_items]
    for obj_name in on_table_objects:
        if obj_name not in obj_names:
            print(f"  WARNING: {obj_name} not in {scene_name} GT, skipping")
            continue
        add_rel(obj_name, "on_top_of", "table")
        add_rel("table",   "under",     obj_name)
        add_rel(obj_name, "higher_than", "table")
        add_rel("table",  "lower_than",  obj_name)

    gt["relations"].extend(new_rels)

    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)

    print(f"  table centroid Z = {centroid[2]:.3f}  "
          f"({'Z-DOWN' if is_z_down else 'Z-UP'}), "
          f"+{len(new_rels)} new rels, total {len(gt['relations'])}")

print("\nDone. GT files updated.")
