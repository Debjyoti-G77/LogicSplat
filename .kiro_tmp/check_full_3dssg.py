"""
Download the full 3DSSG relationships.json and check if it has the RIO10 test scenes.
Also check for objects.json in the same path.
"""
import urllib.request
import json

# First check more URLs
urls_to_check = [
    "https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG/objects.json",
    "https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG/classes.txt",
    "https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG/relationships.txt",
]

print("=== Checking additional URLs ===")
for url in urls_to_check:
    try:
        req = urllib.request.Request(url, method='HEAD')
        resp = urllib.request.urlopen(req, timeout=10)
        size = resp.headers.get('Content-Length', 'unknown')
        print(f"  EXISTS ({size} bytes): {url}")
    except urllib.error.HTTPError as e:
        print(f"  {e.code}: {url}")
    except Exception as e:
        print(f"  ERROR ({type(e).__name__}): {url}")

# Download the full 3DSSG relationships.json
print("\n=== Downloading 3DSSG/relationships.json ===")
url = "https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG/relationships.json"
print(f"Downloading from {url}...")
urllib.request.urlretrieve(url, ".kiro_tmp/relationships_full.json")
print("Downloaded!")

# Load and analyze
with open(".kiro_tmp/relationships_full.json") as f:
    full_rel = json.load(f)

if isinstance(full_rel.get("scans"), list):
    full_scan_ids = set(s.get("scan") for s in full_rel["scans"] if "scan" in s)
else:
    full_scan_ids = set(full_rel.get("scans", {}).keys())

print(f"Total scenes in full 3DSSG: {len(full_scan_ids)}")

# Load RIO10
try:
    with open(r"data/3DSSG/rio10_test_scenes.txt", encoding="utf-8-sig") as f:
        rio10_scenes = set(line.strip() for line in f if line.strip())
except UnicodeDecodeError:
    with open(r"data/3DSSG/rio10_test_scenes.txt", encoding="utf-16") as f:
        rio10_scenes = set(line.strip() for line in f if line.strip())

# Check overlap
overlap = rio10_scenes & full_scan_ids
print(f"RIO10 scenes in full 3DSSG: {len(overlap)}/{len(rio10_scenes)}")
if overlap:
    print("FOUND! These RIO10 scenes have annotations:")
    for s in sorted(overlap):
        print(f"  {s}")
else:
    print("Still NO overlap!")
    
# Compare with our subset
with open(r"data/3DSSG/relationships.json") as f:
    subset_rel = json.load(f)
subset_scan_ids = set(s.get("scan") for s in subset_rel["scans"] if "scan" in s)

print(f"\nSubset scenes: {len(subset_scan_ids)}")
print(f"Full scenes: {len(full_scan_ids)}")
print(f"Overlap between subset and full: {len(subset_scan_ids & full_scan_ids)}")
print(f"In subset but not full: {len(subset_scan_ids - full_scan_ids)}")
print(f"In full but not subset: {len(full_scan_ids - subset_scan_ids)}")

# Show some sample IDs from the full that aren't in subset
diff = full_scan_ids - subset_scan_ids
if diff:
    print(f"\nSample IDs in full but not subset: {sorted(diff)[:5]}")
