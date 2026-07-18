"""
Check what's in objects.json for the RIO10 test scenes.
The key question: does objects.json contain RELATIONSHIP annotations too,
or just object labels? And can we use it for evaluation?
"""
import json

# Load the full objects.json
with open(".kiro_tmp/3DSSG_extracted/3DSSG/objects.json") as f:
    obj_data = json.load(f)

# Check structure
print(f"Top-level keys: {list(obj_data.keys())}")
if "scans" in obj_data:
    scans = obj_data["scans"]
    print(f"Number of scans: {len(scans)}")
    # Look at first scan structure
    if scans:
        print(f"\nFirst scan keys: {list(scans[0].keys())}")
        print(f"First scan ID: {scans[0].get('scan')}")
        if 'objects' in scans[0]:
            print(f"Number of objects in first scan: {len(scans[0]['objects'])}")
            if scans[0]['objects']:
                print(f"First object keys: {list(scans[0]['objects'][0].keys())}")
                print(f"First object sample: {scans[0]['objects'][0]}")

# Now find a RIO10 scene
try:
    with open(r"data/3DSSG/rio10_test_scenes.txt", encoding="utf-16") as f:
        rio10_scenes = set(line.strip() for line in f if line.strip())
except:
    with open(r"data/3DSSG/rio10_test_scenes.txt", encoding="utf-8-sig") as f:
        rio10_scenes = set(line.strip() for line in f if line.strip())

rio10_example = None
for scan in scans:
    if scan.get("scan") in rio10_scenes:
        rio10_example = scan
        break

if rio10_example:
    print(f"\n=== RIO10 example scene: {rio10_example['scan']} ===")
    print(f"Keys: {list(rio10_example.keys())}")
    print(f"Number of objects: {len(rio10_example.get('objects', []))}")
    if rio10_example.get('objects'):
        print(f"First object: {rio10_example['objects'][0]}")
        print(f"Object keys: {list(rio10_example['objects'][0].keys())}")

# Now check relationships.json structure
with open(".kiro_tmp/3DSSG_extracted/3DSSG/relationships.json") as f:
    rel_data = json.load(f)

print(f"\n=== relationships.json ===")
print(f"Top-level keys: {list(rel_data.keys())}")
if "scans" in rel_data:
    rel_scans = rel_data["scans"]
    print(f"Number of scans: {len(rel_scans)}")
    if rel_scans:
        print(f"First scan keys: {list(rel_scans[0].keys())}")
        print(f"First scan ID: {rel_scans[0].get('scan')}")
        if 'relationships' in rel_scans[0]:
            print(f"Number of relationships in first scan: {len(rel_scans[0]['relationships'])}")
            if rel_scans[0]['relationships']:
                print(f"First relationship: {rel_scans[0]['relationships'][0]}")

# Check: does the relationships.json contain the same scans as objects.json minus RIO10?
rel_scan_ids = set(s.get("scan") for s in rel_scans)
obj_scan_ids = set(s.get("scan") for s in scans)
print(f"\nScans in objects.json: {len(obj_scan_ids)}")
print(f"Scans in relationships.json: {len(rel_scan_ids)}")
print(f"Scans in objects but NOT in relationships: {len(obj_scan_ids - rel_scan_ids)}")
print(f"Of those, how many are RIO10: {len((obj_scan_ids - rel_scan_ids) & rio10_scenes)}")
print(f"Non-RIO10 scans missing from relationships: {len((obj_scan_ids - rel_scan_ids) - rio10_scenes)}")
