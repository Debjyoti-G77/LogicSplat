"""Check the HuggingFace dataset structure for GaussianWorld/3rscan_mcmc_3dgs"""
from huggingface_hub import HfApi

api = HfApi()

# List files in the dataset to understand structure and total size
try:
    files = api.list_repo_files("GaussianWorld/3rscan_mcmc_3dgs", repo_type="dataset")
    print(f"Total files: {len(files)}")
    print("\n--- First 50 files ---")
    for f in files[:50]:
        print(f)
    print("\n--- Last 20 files ---")
    for f in files[-20:]:
        print(f)
    
    # Check file extensions
    extensions = set()
    for f in files:
        if '.' in f.split('/')[-1]:
            ext = '.' + f.split('/')[-1].rsplit('.', 1)[1]
            extensions.add(ext)
    print(f"\n--- File extensions found: {extensions}")
    
    # Check if scene IDs appear in paths
    sample_scenes = [
        "f62fd5fd-9a3f-2f44-883a-1e5cf819608e",
        "6a36053f-fa53-2915-9579-3938283bc154",
        "02b33df9-be2b-2d54-9062-1253be3ce186"
    ]
    print("\n--- Checking if sample scene IDs appear in file paths ---")
    for scene_id in sample_scenes:
        matches = [f for f in files if scene_id in f]
        if matches:
            print(f"  {scene_id}: {len(matches)} files")
            for m in matches[:5]:
                print(f"    {m}")
        else:
            print(f"  {scene_id}: NOT FOUND")
            
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying as a model repo instead...")
    try:
        files = api.list_repo_files("GaussianWorld/3rscan_mcmc_3dgs", repo_type="model")
        print(f"Total files (model): {len(files)}")
        for f in files[:30]:
            print(f)
    except Exception as e2:
        print(f"Model repo error: {e2}")
