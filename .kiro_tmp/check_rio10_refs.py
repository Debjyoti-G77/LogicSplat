"""
Check if RIO10 scenes are references or rescans in 3RScan.json,
and whether their reference scans have annotations in relationships.json.
"""
import json

# Load RIO10 test scenes
try:
    with open(r"data/3DSSG/rio10_test_scenes.txt", encoding="utf-8-sig") as f:
        rio10_scenes = set(line.strip() for line in f if line.strip())
except UnicodeDecodeError:
    with open(r"data/3DSSG/rio10_test_scenes.txt", encoding="utf-16") as f:
        rio10_scenes = set(line.strip() for line in f if line.strip())

# Load 3RScan.json
with open(r"data/3DSSG/3RScan.json") as f:
    rscan_data = json.load(f)

# Load relationships.json
with open(r"data/3DSSG/relationships.json") as f:
    rel_data = json.load(f)
rel_scan_ids = set(s.get("scan") for s in rel_data["scans"] if "scan" in s)

# Build maps
reference_entries = {}  # ref_id -> entry
rescan_to_ref = {}  # rescan_id -> ref_id

for entry in rscan_data:
    ref_id = entry["reference"]
    reference_entries[ref_id] = entry
    for scan in entry.get("scans", []):
        rescan_id = scan.get("reference", "")
        if rescan_id:
            rescan_to_ref[rescan_id] = ref_id

# Classify RIO10 scenes
rio10_as_reference = []
rio10_as_rescan = []
rio10_not_found = []

for scene_id in sorted(rio10_scenes):
    if scene_id in reference_entries:
        rio10_as_reference.append(scene_id)
    elif scene_id in rescan_to_ref:
        rio10_as_rescan.append(scene_id)
    else:
        rio10_not_found.append(scene_id)

print(f"=== RIO10 Scene Classification ===")
print(f"As REFERENCE scans: {len(rio10_as_reference)}")
print(f"As RESCANS: {len(rio10_as_rescan)}")
print(f"NOT FOUND: {len(rio10_not_found)}")

# For reference scans - check if they have annotations
print(f"\n=== RIO10 scenes that are REFERENCE scans ===")
ref_with_annotations = 0
ref_without_annotations = 0
for scene_id in rio10_as_reference:
    entry = reference_entries[scene_id]
    has_annotation = scene_id in rel_scan_ids
    entry_type = entry.get("type", "unknown")
    if has_annotation:
        ref_with_annotations += 1
    else:
        ref_without_annotations += 1
    print(f"  {scene_id} type={entry_type} annotated={has_annotation}")

# For rescans - check if their reference has annotations
print(f"\n=== RIO10 scenes that are RESCANS ===")
rescan_ref_annotated = 0
rescan_ref_not_annotated = 0
for scene_id in rio10_as_rescan:
    ref_id = rescan_to_ref[scene_id]
    ref_entry = reference_entries.get(ref_id, {})
    ref_type = ref_entry.get("type", "unknown")
    ref_has_annotation = ref_id in rel_scan_ids
    rescan_has_annotation = scene_id in rel_scan_ids
    if ref_has_annotation:
        rescan_ref_annotated += 1
    else:
        rescan_ref_not_annotated += 1
    print(f"  {scene_id} -> ref={ref_id} ref_type={ref_type} ref_annotated={ref_has_annotation} self_annotated={rescan_has_annotation}")

print(f"\n=== Summary ===")
print(f"RIO10 reference scans with annotations: {ref_with_annotations}/{len(rio10_as_reference)}")
print(f"RIO10 reference scans WITHOUT annotations: {ref_without_annotations}/{len(rio10_as_reference)}")
print(f"RIO10 rescans whose reference HAS annotations: {rescan_ref_annotated}/{len(rio10_as_rescan)}")
print(f"RIO10 rescans whose reference LACKS annotations: {rescan_ref_not_annotated}/{len(rio10_as_rescan)}")

# Check what type the RIO10 reference entries are
print(f"\n=== Types of RIO10 reference entries ===")
types = {}
for scene_id in rio10_as_reference:
    t = reference_entries[scene_id].get("type", "unknown")
    types[t] = types.get(t, 0) + 1
print(types)

# Check what type the parent references of RIO10 rescans are
print(f"\n=== Types of parent references for RIO10 rescans ===")
types2 = {}
for scene_id in rio10_as_rescan:
    ref_id = rescan_to_ref[scene_id]
    ref_entry = reference_entries.get(ref_id, {})
    t = ref_entry.get("type", "unknown")
    types2[t] = types2.get(t, 0) + 1
print(types2)
