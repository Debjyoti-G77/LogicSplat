"""Check if the 46 RIO10 test scenes appear in 3RScan.json as references or rescans."""
import json

# Load RIO10 test scenes
with open(r"data/3DSSG/rio10_test_scenes.txt") as f:
    rio10_scenes = set(line.strip() for line in f if line.strip())

print(f"Total RIO10 test scenes: {len(rio10_scenes)}")

# Load 3RScan.json
with open(r"data/3DSSG/3RScan.json") as f:
    rscan_data = json.load(f)

print(f"Total 3RScan entries: {len(rscan_data)}")

# Check each RIO10 scene
found_as_reference = []
found_as_rescan = []
not_found = []

for scene_id in sorted(rio10_scenes):
    found = False
    # Check if it's a reference scan
    for entry in rscan_data:
        if entry["reference"] == scene_id:
            found_as_reference.append((scene_id, entry.get("type", "unknown")))
            found = True
            break
        # Check if it's in the scans list (rescan)
        for scan in entry.get("scans", []):
            if scan.get("reference") == scene_id:
                found_as_rescan.append((scene_id, entry["reference"], entry.get("type", "unknown")))
                found = True
                break
        if found:
            break
    if not found:
        not_found.append(scene_id)

print(f"\n--- Results ---")
print(f"Found as REFERENCE scan: {len(found_as_reference)}")
for sid, stype in found_as_reference:
    print(f"  {sid} (type: {stype})")

print(f"\nFound as RESCAN: {len(found_as_rescan)}")
for sid, ref_id, stype in found_as_rescan:
    print(f"  {sid} -> reference: {ref_id} (type: {stype})")

print(f"\nNOT FOUND in 3RScan.json: {len(not_found)}")
for sid in not_found:
    print(f"  {sid}")

# Now check relationships.json
print("\n\n--- Checking relationships.json ---")
with open(r"data/3DSSG/relationships.json") as f:
    rel_data = json.load(f)

rel_scans = set(rel_data.get("scans", {}).keys()) if isinstance(rel_data.get("scans"), dict) else set()
if isinstance(rel_data.get("scans"), list):
    rel_scans = set(s.get("scan") for s in rel_data["scans"] if "scan" in s)

print(f"Total scenes in relationships.json: {len(rel_scans)}")

# Check overlap
overlap = rio10_scenes & rel_scans
print(f"RIO10 scenes in relationships.json: {len(overlap)}")

# Check if reference scans of RIO10 scenes are in relationships.json
if found_as_rescan:
    print("\n--- Checking if REFERENCE scans of RIO10 rescans are in relationships.json ---")
    ref_ids = set(ref_id for _, ref_id, _ in found_as_rescan)
    ref_overlap = ref_ids & rel_scans
    print(f"Reference scans found in relationships.json: {len(ref_overlap)}/{len(ref_ids)}")
    for rid in sorted(ref_overlap):
        print(f"  {rid}")

if found_as_reference:
    print("\n--- Checking if RIO10 reference scans are in relationships.json ---")
    ref_ids = set(sid for sid, _ in found_as_reference)
    ref_overlap = ref_ids & rel_scans
    print(f"RIO10 reference scans found in relationships.json: {len(ref_overlap)}/{len(ref_ids)}")
    for rid in sorted(ref_overlap):
        print(f"  {rid}")
