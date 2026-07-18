import json, os

with open("D:/logicsplat_data/processed/scene_01/ns_data/transforms.json") as f:
    t = json.load(f)

frames = t["frames"]
images_dir = "D:/logicsplat_data/processed/scene_01/ns_data/images"
sorted_files = sorted(os.listdir(images_dir))

print("First 10 frames in transforms.json:")
for i, f in enumerate(frames[:10]):
    print(f"  transforms[{i}]: {f['file_path']}")

print()
print("First 10 files in images/ dir (sorted):")
for i, f in enumerate(sorted_files[:10]):
    print(f"  sorted[{i}]: {f}")

print()
print("Order comparison (first 10):")
mismatches = 0
for i in range(min(10, len(frames))):
    tf = os.path.basename(frames[i]["file_path"])
    sf = sorted_files[i]
    match = "MATCH" if tf == sf else "MISMATCH"
    if tf != sf:
        mismatches += 1
    print(f"  [{i}] transforms={tf}  sorted={sf}  -> {match}")

print(f"\nMismatches in first 10: {mismatches}")

# Build a lookup: filename -> index in sorted list
fname_to_sorted_idx = {f: i for i, f in enumerate(sorted_files)}

# Check what sorted index corresponds to transforms frame 100
f100 = os.path.basename(frames[100]["file_path"])
print(f"\ntransforms[100] = {f100}")
print(f"  sorted index of {f100} = {fname_to_sorted_idx.get(f100, 'NOT FOUND')}")
