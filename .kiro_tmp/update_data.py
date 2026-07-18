"""
Update local data files with the full versions from the 3DSSG/3RScan servers.
"""
import urllib.request
import shutil
import os

# 1. Copy the full objects.json (already extracted)
src = ".kiro_tmp/3DSSG_extracted/3DSSG/objects.json"
dst = "data/3DSSG/objects.json"
if os.path.exists(dst):
    # Backup existing
    shutil.copy2(dst, dst + ".bak")
    print(f"Backed up existing {dst}")
shutil.copy2(src, dst)
print(f"Copied full objects.json (1482 scans) to {dst}")

# 2. Download the full 3RScan.json (3.1MB with all 1482 scans)
url = "https://www.campar.in.tum.de/public_datasets/3RScan/3RScan.json"
dst2 = "data/3DSSG/3RScan.json"
shutil.copy2(dst2, dst2 + ".bak")
print(f"Backed up existing {dst2}")
print(f"Downloading full 3RScan.json from {url}...")
urllib.request.urlretrieve(url, dst2)
print(f"Downloaded full 3RScan.json to {dst2}")

# Verify
import json
with open(dst) as f:
    obj = json.load(f)
print(f"\nobjects.json: {len(obj['scans'])} scans")

with open(dst2) as f:
    rscan = json.load(f)
print(f"3RScan.json: {len(rscan)} entries")

# Count total scans in 3RScan
total = len(rscan)
total_with_rescans = sum(1 + len(e.get('scans', [])) for e in rscan)
print(f"3RScan.json: {total} locations, {total_with_rescans} total scans")
