"""Clean up partial nerfstudio outputs from failed runs, preserving ground truth JSONs."""
import os
import shutil

scenes = [f"scene_{i:02d}" for i in range(6, 14)]
base = "D:/logicsplat_data/processed"

for scene in scenes:
    scene_dir = os.path.join(base, scene)
    
    # Delete partial ns_data (failed COLMAP)
    ns_data = os.path.join(scene_dir, "ns_data")
    if os.path.isdir(ns_data):
        shutil.rmtree(ns_data)
        print(f"Deleted {scene}/ns_data")
    
    # Delete partial images extracted by ffmpeg
    images = os.path.join(scene_dir, "images")
    if os.path.isdir(images):
        shutil.rmtree(images)
        print(f"Deleted {scene}/images")
    
    # Delete error marker
    err = os.path.join(scene_dir, "processing_error.txt")
    if os.path.exists(err):
        os.remove(err)
        print(f"Deleted {scene}/processing_error.txt")

# Delete partial outputs folder
outputs = "outputs"
if os.path.isdir(outputs):
    for name in os.listdir(outputs):
        if name.startswith("scene_0") and int(name.split("_")[1]) >= 6:
            shutil.rmtree(os.path.join(outputs, name))
            print(f"Deleted outputs/{name}")

print("\nDone. Ground truth JSONs preserved.")
print("\nCurrent state of scenes 06-13:")
for scene in scenes:
    scene_dir = os.path.join(base, scene)
    contents = os.listdir(scene_dir) if os.path.isdir(scene_dir) else []
    print(f"  {scene}: {contents}")
