"""Scan tar - write results to file to avoid output issues."""
import tarfile
import os
import sys
import json
from collections import defaultdict

tar_path = "D:/octscenes_download/640x480/image_640x480_00"
output_file = "D:/octscenes_download/scan_results.json"

print(f"Scanning: {tar_path}", flush=True)
print(f"Results will be saved to: {output_file}", flush=True)

scenes = defaultdict(list)
total_count = 0
errors = []

try:
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
            
            if total_count % 5000 == 0:
                sys.stdout.write(f"\r  Scanned {total_count} files, {len(scenes)} scenes...")
                sys.stdout.flush()
except Exception as e:
    errors.append(str(e))
    print(f"\nError during scan: {e}", flush=True)

print(f"\n\nDone. Total files: {total_count}, Scenes: {len(scenes)}", flush=True)

# Compute stats
frame_counts = {sid: len(frames) for sid, frames in scenes.items()}
distribution = defaultdict(int)
for count in frame_counts.values():
    distribution[count] += 1

complete_scenes = sorted([sid for sid, count in frame_counts.items() if count == 60])
max_frames = max(frame_counts.values()) if frame_counts else 0

# Save results
results = {
    "total_files": total_count,
    "unique_scenes": len(scenes),
    "max_frames_in_scene": max_frames,
    "complete_scenes_60": complete_scenes,
    "num_complete": len(complete_scenes),
    "distribution_top": {str(k): v for k, v in sorted(distribution.items(), reverse=True)[:20]},
    "errors": errors,
}

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_file}", flush=True)
print(f"Complete scenes (60 frames): {len(complete_scenes)}", flush=True)
print(f"Max frames in any scene: {max_frames}", flush=True)
print(f"Distribution (top):", flush=True)
for count, num_scenes in sorted(distribution.items(), reverse=True)[:10]:
    print(f"  {count} frames: {num_scenes} scenes", flush=True)
