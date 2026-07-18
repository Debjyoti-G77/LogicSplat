"""Scan the entire first tar part to find complete scenes (60 frames)."""
import tarfile
import os
import sys
from collections import defaultdict

tar_path = "D:/octscenes_download/640x480/image_640x480_00"
print(f"Scanning entire tar: {tar_path}", flush=True)
print(f"Size: {os.path.getsize(tar_path) / 1024**3:.2f} GB", flush=True)
print("This may take a few minutes...\n", flush=True)

scenes = defaultdict(list)
total_count = 0

with tarfile.open(tar_path, "r:gz") as tf:
    for member in tf:
        if not member.isfile():
            continue
        total_count += 1
        basename = os.path.basename(member.name)
        if basename.endswith('.png'):
            parts = basename.replace('.png', '').split('_')
            if len(parts) == 2:
                scene_id, frame_id = parts
                scenes[scene_id].append(frame_id)
        
        if total_count % 10000 == 0:
            print(f"  Scanned {total_count} files, {len(scenes)} scenes so far...", flush=True)

print(f"\nDone scanning.")
print(f"  Total files: {total_count}")
print(f"  Unique scenes: {len(scenes)}")

# Find scenes with most frames
frame_counts = {sid: len(frames) for sid, frames in scenes.items()}
by_count = defaultdict(list)
for sid, count in frame_counts.items():
    by_count[count].append(sid)

print(f"\n  Frame count distribution:")
for count in sorted(by_count.keys(), reverse=True)[:15]:
    scene_list = sorted(by_count[count])
    preview = ', '.join(scene_list[:5])
    if len(scene_list) > 5:
        preview += f", ... (+{len(scene_list)-5} more)"
    print(f"    {count} frames: {len(scene_list)} scenes  [{preview}]")

# Check if any have 60 frames (complete)
complete = sorted(by_count.get(60, []))
print(f"\n  Complete scenes (60 frames): {len(complete)}")
if complete:
    print(f"    First 20: {complete[:20]}")

# If not enough complete scenes, what's the max?
max_frames = max(frame_counts.values())
print(f"\n  Max frames in any scene: {max_frames}")

# How many parts would we need for 20 complete scenes?
# Each part has ~total_count/30 files = ~10000 files
# 5000 scenes × 60 frames = 300,000 total files / 30 parts = 10,000 per part
# So each scene's 60 frames are spread across ~2 parts on average
print(f"\n  Files per tar part (approx): {total_count}")
print(f"  Total dataset: 300,000 files across 30 parts")
print(f"  Average frames per scene in this part: {total_count / max(len(scenes),1):.1f}")
