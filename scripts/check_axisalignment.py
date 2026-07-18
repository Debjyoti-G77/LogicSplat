"""Check if ScanNet scenes have axisAlignment metadata files."""
import os

scene_dir = "D:/scannet/scans/scene0000_00"
files = os.listdir(scene_dir)
print("Files in scene0000_00:")
for f in sorted(files):
    print(f"  {f}")

# Check specifically for .txt metadata file
txt_files = [f for f in files if f.endswith(".txt")]
print(f"\n.txt files: {txt_files}")

# Read the .txt file if it exists
for f in txt_files:
    path = os.path.join(scene_dir, f)
    with open(path) as fh:
        content = fh.read()
    if "axisAlignment" in content:
        print(f"\n'{f}' contains axisAlignment!")
        # Print the relevant line
        for line in content.splitlines():
            if "axisAlignment" in line:
                print(f"  {line}")
    else:
        print(f"\n'{f}' does NOT contain axisAlignment")
        print(f"  First 200 chars: {content[:200]}")
