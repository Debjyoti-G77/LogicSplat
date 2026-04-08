"""COLMAP binary file loader."""
import struct
import os
import numpy as np


def read_points3D(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse COLMAP points3D.bin into xyz and rgb arrays.
    Returns:
        points: (N, 3) float64
        colors: (N, 3) uint8
    """
    points, colors = [], []
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            f.read(8)                                    # point id
            xyz = struct.unpack("<ddd", f.read(24))
            rgb = struct.unpack("BBB", f.read(3))
            f.read(8)                                    # reprojection error
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)                        # track entries
            points.append(xyz)
            colors.append(rgb)
    return np.array(points, dtype=np.float64), np.array(colors, dtype=np.uint8)


def load_scene_points(scene_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load points and colors from a scene directory.
    Expects: <scene_path>/colmap/sparse/0/points3D.bin
    """
    bin_path = os.path.join(scene_path, "colmap", "sparse", "0", "points3D.bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"points3D.bin not found at: {bin_path}")
    return read_points3D(bin_path)
