"""
Project each LERF scene's clustered 3D object centroids into the exact
camera frame that carries LERF-OVS's own polygon labels, using the
reconstruction's own COLMAP camera pose (no OpenGL/nerfstudio convention
ambiguity -- these are the raw COLMAP poses the labels were drawn on).
Matches projected points to the nearest labeled polygon to recover real
object names + pixel-accurate boxes, verified by checking it visually.
"""
import sys
sys.path.insert(0, ".")
import os
import json
import numpy as np

from nerfstudio.process_data.colmap_utils import read_cameras_binary, read_images_binary
from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
from src.gaussian.clustering import gaussian_to_objects

DATA_ROOT = "D:/lerf_data/lerf_ovs"


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])


def load_label(scene, frame_name):
    path = os.path.join(DATA_ROOT, "label", scene, frame_name.replace(".jpg", ".json"))
    with open(path) as f:
        return json.load(f)


def label_boxes(label):
    """category -> (x,y,w,h) axis-aligned box from polygon points, plus centroid."""
    out = {}
    for obj in label["objects"]:
        pts = np.array(obj["segmentation"])
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        cat = obj["category"]
        out.setdefault(cat, []).append((x0, y0, x1 - x0, y1 - y0))
    # if a category appears multiple times, keep the largest instance
    final = {}
    for cat, boxes in out.items():
        final[cat] = max(boxes, key=lambda b: b[2] * b[3])
    return final


def load_dataparser_inverse(transform_path):
    """nerfstudio applies: X_ns = scale * (R @ X_colmap + t). Return a
    function that maps a splatfacto-space point back to raw COLMAP space:
    X_colmap = R^T @ (X_ns / scale) - R^T @ t."""
    with open(transform_path) as f:
        d = json.load(f)
    M = np.array(d["transform"])  # 3x4
    R = M[:, :3]
    t = M[:, 3]
    scale = d["scale"]

    def inverse(x_ns):
        return R.T @ (np.array(x_ns) / scale) - R.T @ t

    return inverse


def find_dataparser_transform(scene):
    base = os.path.join(DATA_ROOT, scene, "ns_output")
    for root, _, files in os.walk(base):
        if "dataparser_transforms.json" in files:
            return os.path.join(root, "dataparser_transforms.json")
    return None


def project_scene(scene, frame_name, n_exact):
    cameras = read_cameras_binary(os.path.join(DATA_ROOT, scene, "sparse/0/cameras.bin"))
    images = read_images_binary(os.path.join(DATA_ROOT, scene, "sparse/0/images.bin"))
    transform_path = find_dataparser_transform(scene)
    to_colmap = load_dataparser_inverse(transform_path) if transform_path else (lambda x: x)

    img_entry = None
    for img in images.values():
        if img.name == frame_name:
            img_entry = img
            break
    if img_entry is None:
        raise RuntimeError(f"frame {frame_name} not found in images.bin for {scene}")

    cam = cameras[img_entry.camera_id]
    fx, fy, cx, cy = (cam.params[0], cam.params[0], cam.params[1], cam.params[2]) if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL") \
        else (cam.params[0], cam.params[1], cam.params[2], cam.params[3])

    R = qvec2rotmat(img_entry.qvec)
    t = img_entry.tvec

    ply_path = os.path.join(DATA_ROOT, scene, "splat", "splat.ply")
    cloud = load_gaussian_ply(ply_path)
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)
    objects, _ = gaussian_to_objects(cloud, target_min=n_exact, target_max=n_exact + 3, n_exact=n_exact)

    projected = []
    for i, o in enumerate(objects):
        X = to_colmap(np.array(o.centroid))
        Xc = R @ X + t
        if Xc[2] <= 0:
            projected.append((i, None))
            continue
        u = fx * Xc[0] / Xc[2] + cx
        v = fy * Xc[1] / Xc[2] + cy
        projected.append((i, (float(u), float(v))))

    return objects, projected, (cam.width, cam.height)


def match_to_labels(projected, boxes):
    """Greedy nearest-centroid assignment, one label per object index."""
    label_centroids = {cat: (x + w/2, y + h/2) for cat, (x, y, w, h) in boxes.items()}
    used = set()
    assignment = {}
    # sort by distance globally for a reasonable greedy match
    candidates = []
    for idx, pt in projected:
        if pt is None:
            continue
        for cat, (cx_, cy_) in label_centroids.items():
            d = np.hypot(pt[0] - cx_, pt[1] - cy_)
            candidates.append((d, idx, cat))
    candidates.sort(key=lambda c: c[0])
    assigned_idx = set()
    for d, idx, cat in candidates:
        if idx in assigned_idx or cat in used:
            continue
        assignment[idx] = cat
        assigned_idx.add(idx)
        used.add(cat)
    return assignment


if __name__ == "__main__":
    import sys as _s
    scene = _s.argv[1] if len(_s.argv) > 1 else "ramen"
    frame = _s.argv[2] if len(_s.argv) > 2 else "frame_00006.jpg"
    n_exact = {"ramen": 13, "teatime": 10, "waldo_kitchen": 10, "figurines": 10}[scene]

    label = load_label(scene, frame)
    boxes = label_boxes(label)
    print(f"labeled categories in {frame}: {list(boxes.keys())}")

    objects, projected, (w, h) = project_scene(scene, frame, n_exact)
    print(f"image size per COLMAP: {w}x{h}, label declares: {label['info']['width']}x{label['info']['height']}")

    assignment = match_to_labels(projected, boxes)
    print(f"\nmatched {len(assignment)} / {len(objects)} objects:")
    for idx, cat in sorted(assignment.items()):
        pt = dict(projected)[idx]
        print(f"  obj_{idx} -> {cat}  (projected at {pt[0]:.0f},{pt[1]:.0f}, label box {boxes[cat]})")
