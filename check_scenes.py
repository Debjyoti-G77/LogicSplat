import os, json, struct

scenes_new = ["scene_%02d" % i for i in range(6, 14)]

def count_images_bin(path):
    try:
        with open(path, "rb") as f:
            return struct.unpack("<Q", f.read(8))[0]
    except:
        return 0

print("%-12s %-10s %-8s %-8s %-8s" % ("Scene", "splat.ply", "MB", "frames", "status"))
print("-" * 52)

for s in scenes_new:
    base = "D:/logicsplat_data/processed/" + s
    ply  = base + "/splat.ply"
    tf   = base + "/ns_data/transforms.json"

    ply_ok = os.path.exists(ply)
    ply_mb = ("%.1f" % (os.path.getsize(ply) / 1024 / 1024)) if ply_ok else "-"

    frames = 0
    if os.path.exists(tf):
        try:
            frames = len(json.load(open(tf)).get("frames", []))
        except:
            frames = -1

    # Determine status
    if not ply_ok:
        status = "NO PLY"
    elif ply_ok and frames < 10:
        status = "BAD (retrain needed)"
    elif ply_ok and float(ply_mb) < 5:
        status = "TOO SMALL"
    else:
        status = "OK"

    print("%-12s %-10s %-8s %-8s %s" % (s, "YES" if ply_ok else "NO", ply_mb, frames, status))
