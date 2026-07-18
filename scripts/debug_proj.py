import sys, json, numpy as np, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
from src.gaussian.loader import load_gaussian_ply, filter_gaussians
from src.gaussian.clustering import gaussian_to_objects

cloud = load_gaussian_ply('D:/logicsplat_data/processed/scene_01/splat.ply')
cf = filter_gaussians(cloud, opacity_threshold=0.1)
objects, _ = gaussian_to_objects(cf)

t = json.load(open('D:/logicsplat_data/processed/scene_01/ns_data/transforms.json'))
w, h = t['w'], t['h']
fx, fy = t['fl_x'], t['fl_y']
cx, cy = t['cx'], t['cy']

# try multiple frames
for frame_idx in [50, 100, 150, 200]:
    frame = t['frames'][frame_idx]
    T = np.array(frame['transform_matrix'])
    w2c = np.linalg.inv(T)

    print(f'\nFrame {frame_idx}: {frame["file_path"]}')
    for obj in objects[:3]:
        p = np.array([*obj.centroid, 1.0])
        p_cam = w2c @ p
        print(f'  Obj {obj.uid} cam_xyz={p_cam[:3].round(3)}')

        # try all sign combinations for z
        for sx, sy, sz in [(1,1,1),(1,-1,1),(1,1,-1),(1,-1,-1)]:
            x = sx * p_cam[0]
            y = sy * p_cam[1]
            z = sz * p_cam[2]
            if z > 0:
                u = fx * x / z + cx
                v = fy * y / z + cy
                if 0 <= u < w and 0 <= v < h:
                    print(f'    signs=({sx},{sy},{sz}) -> u={u:.0f} v={v:.0f} IN FRAME')
