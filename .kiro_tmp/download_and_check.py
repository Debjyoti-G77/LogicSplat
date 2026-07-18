"""
Download the 3DSSG.zip and objects.json to check if they contain RIO10 test scene annotations.
"""
import urllib.request
import json
import zipfile
import os

# Download 3DSSG.zip
print("=== Downloading 3DSSG.zip (2.8MB) ===")
urllib.request.urlretrieve(
    "https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG.zip",
    ".kiro_tmp/3DSSG.zip"
)
print("Downloaded!")

# Extract and list contents
print("\n=== Contents of 3DSSG.zip ===")
with zipfile.ZipFile(".kiro_tmp/3DSSG.zip", 'r') as z:
    for name in z.namelist():
        info = z.getinfo(name)
        print(f"  {name} ({info.file_size} bytes)")
    
    # Extract all
    z.extractall(".kiro_tmp/3DSSG_extracted")

# Check what files were extracted
print("\n=== Extracted files ===")
for root, dirs, files in os.walk(".kiro_tmp/3DSSG_extracted"):
    for f in files:
        path = os.path.join(root, f)
        size = os.path.getsize(path)
        print(f"  {path} ({size} bytes)")

# Now check if the extracted relationships.json has RIO10 scenes
# Look for any JSON files
for root, dirs, files in os.walk(".kiro_tmp/3DSSG_extracted"):
    for f in files:
        if f.endswith('.json'):
            path = os.path.join(root, f)
            print(f"\n=== Checking {path} ===")
            with open(path) as fp:
                data = json.load(fp)
            if isinstance(data, dict) and "scans" in data:
                scans = data["scans"]
                if isinstance(scans, list):
                    scan_ids = set(s.get("scan") for s in scans if "scan" in s)
                    print(f"  Contains {len(scan_ids)} scans")
                    # Check RIO10 overlap
                    try:
                        with open(r"data/3DSSG/rio10_test_scenes.txt", encoding="utf-16") as rf:
                            rio10 = set(line.strip() for line in rf if line.strip())
                    except:
                        with open(r"data/3DSSG/rio10_test_scenes.txt", encoding="utf-8-sig") as rf:
                            rio10 = set(line.strip() for line in rf if line.strip())
                    overlap = rio10 & scan_ids
                    print(f"  RIO10 overlap: {len(overlap)}/{len(rio10)}")
                    if overlap:
                        print(f"  FOUND RIO10 scenes!")
                        for s in sorted(overlap)[:5]:
                            print(f"    {s}")
