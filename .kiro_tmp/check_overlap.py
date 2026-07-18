"""Check overlap between 3DSSG scenes and HuggingFace dataset scenes"""
import json
from huggingface_hub import HfApi

# Get scene IDs from relationships.json
with open(r'c:\Users\Debjyoti\Desktop\LogicSplat\data\3DSSG\relationships.json') as f:
    data = json.load(f)
ssg_scenes = set(s['scan'] for s in data['scans'])
print(f"Scenes in relationships.json: {len(ssg_scenes)}")

# Get scene IDs from HuggingFace dataset
api = HfApi()
files = api.list_repo_files("GaussianWorld/3rscan_mcmc_3dgs", repo_type="dataset")

# Extract unique scene IDs (top-level folders)
hf_scenes = set()
for f in files:
    parts = f.split('/')
    if len(parts) > 1 and '-' in parts[0] and len(parts[0]) > 30:  # UUID-like
        hf_scenes.add(parts[0])

print(f"Scenes in HuggingFace dataset: {len(hf_scenes)}")

# Overlap
overlap = ssg_scenes & hf_scenes
print(f"Scenes in BOTH (overlap): {len(overlap)}")
print(f"Scenes only in 3DSSG: {len(ssg_scenes - hf_scenes)}")
print(f"Scenes only in HF: {len(hf_scenes - ssg_scenes)}")

# Count PLY files to download
ply_files = [f for f in files if f.endswith('.ply') and f.split('/')[0] in overlap]
print(f"\nPLY files to download (overlap scenes): {len(ply_files)}")
print(f"Sample PLY paths:")
for p in ply_files[:5]:
    print(f"  {p}")
