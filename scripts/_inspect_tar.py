"""Inspect the downloaded tar file to determine its format."""
import tarfile
import os
import sys
from collections import defaultdict

tar_path = "D:/octscenes_download/640x480/image_640x480_00"
print(f"File: {tar_path}")
print(f"Size: {os.path.getsize(tar_path) / 1024**3:.2f} GB")

# Check header bytes
with open(tar_path, "rb") as f:
    header = f.read(512)

print(f"\nFirst 16 bytes (hex): {header[:16].hex()}")
print(f"Gzip magic? {header[:2] == b'\\x1f\\x8b'}")
print(f"Tar magic at 257? {header[257:262]}")

# Try opening as different formats
for mode_name, mode in [("tar (uncompressed)", "r:"), ("tar.gz", "r:gz"), ("tar.bz2", "r:bz2"), ("tar (auto)", "r:*")]:
    try:
        print(f"\nTrying mode: {mode_name}...")
        tf = tarfile.open(tar_path, mode)
        # Just get first few members
        count = 0
        scenes = defaultdict(int)
        for member in tf:
            count += 1
            if count <= 10:
                print(f"  {member.name}  ({member.size} bytes)")
            # Parse scene_id from filename
            basename = os.path.basename(member.name)
            if basename.endswith('.png'):
                parts = basename.replace('.png', '').split('_')
                if len(parts) >= 2:
                    scene_id = '_'.join(parts[:-1])  # everything except last part (frame)
                    scenes[scene_id] += 1
            if count >= 5000:  # Don't scan everything
                break
        
        print(f"\n  Total members scanned: {count}")
        print(f"  Unique scenes found: {len(scenes)}")
        # Show first 10 scenes with frame counts
        sorted_scenes = sorted(scenes.items())
        print(f"  First 10 scenes:")
        for sid, fcount in sorted_scenes[:10]:
            print(f"    {sid}: {fcount} frames")
        if len(sorted_scenes) > 10:
            print(f"    ... and {len(sorted_scenes) - 10} more")
        
        tf.close()
        print(f"\n  SUCCESS with mode: {mode_name}")
        break
    except Exception as e:
        print(f"  Failed: {e}")
