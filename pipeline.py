import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

# from pathlib import Path
import os
from PIL import Image as PILImage
import open3d as o3d
from tqdm import tqdm

from dense_recon.utils.read_write_model import read_model
from dense_recon.utils.sparse_depth import pts3d_to_sparse_depth
from dense_recon.estimator import LeastSquaresEstimator

import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, "Thirdparty/Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2


# assume that you finished SfM and have a colmap model
def pipeline(model_path: str, images_path: str, interval=5, max_depth=8.0, min_depth=0.1, out_path="out.ply", save=False, visualize=False):
    # read the colmap model
    cameras, images, points3D = read_model(model_path)

    # sample images
    im_list = list(images.values())
    im_list = [im_list[i] for i in range(0, len(im_list), interval)]
    # load depth model
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'Thirdparty/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()

    partial_3ds = []

    for img in tqdm(im_list):
        rgb = cv2.imread(os.path.join(images_path, img.name))
        # depth prediction
        depth_pred = model.infer_image(rgb)
        # get sparse depth
        cam = cameras[img.camera_id]
        sparse_depth = pts3d_to_sparse_depth(cam, img, points3D)
        sparse_depth_valid = (sparse_depth < max_depth) * (sparse_depth > min_depth)
        sparse_depth_valid = sparse_depth_valid.astype(bool)
        sparse_depth[~sparse_depth_valid] = np.inf
        sparse_depth = 1.0 / sparse_depth
        # global alignment
        GlobalAlignment = LeastSquaresEstimator(
            estimate = depth_pred,
            target = sparse_depth,
            valid = sparse_depth_valid,
        )
        GlobalAlignment.compute_scale_and_shift()
        GlobalAlignment.apply_scale_and_shift()
        GlobalAlignment.clamp_min_max(clamp_min=min_depth, clamp_max=max_depth)
        int_depth = GlobalAlignment.output.astype(np.float32)
        ga_depth = 1.0 / int_depth
        # backprojection
        fx, fy, cx, cy = cam.params
        u, v = np.meshgrid(np.arange(cam.width), np.arange(cam.height))
        x = (u - cx) * ga_depth / fx
        y = (v - cy) * ga_depth / fy
        pts3d = np.stack((x, y, ga_depth), axis=-1)
        o3d_color = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) / 255.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts3d.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(o3d_color.reshape(-1, 3))
        # cam to world
        R = img.qvec2rotmat()
        t = img.tvec
        t_c2w = -R.T @ t
        R_c2w = R.T
        pcd.rotate(R_c2w, center=(0, 0, 0))
        pcd.translate(t_c2w)
        partial_3ds.append(pcd)

    # merge partial point clouds
    pcd = partial_3ds[0]
    for i in range(1, len(partial_3ds)):
        pcd += partial_3ds[i]

    if save:
        o3d.io.write_point_cloud(out_path, pcd)
    if visualize:
        # overlap sparse points - only for visualization
        sparse = []
        for pt3d in points3D.values():
            sparse.append(pt3d.xyz)
        sparse_pcd = o3d.geometry.PointCloud()
        sparse_pcd.points = o3d.utility.Vector3dVector(sparse)
        sparse_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        pcd += sparse_pcd
        o3d.visualization.draw_geometries([pcd])
    
    return pcd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the colmap model.")
    parser.add_argument("--images_path", type=str, required=True, help="Path to the images.")
    parser.add_argument("--interval", type=int, default=5, help="Sampling interval of the images.")
    parser.add_argument("--max_depth", type=float, default=8.0, help="Maximum depth.")
    parser.add_argument("--min_depth", type=float, default=0.1, help="Minimum depth.")
    parser.add_argument("--out_path", type=str, default="out.ply", help="Path to save the point cloud.")
    parser.add_argument("--save", action="store_true", help="Save the point cloud.") # 값이 있으면 True, 없으면 False
    parser.add_argument("--visualize", action="store_true", help="Visualize the point cloud.")
    args = parser.parse_args()
    pipeline(args.model_path, args.images_path, args.interval, args.max_depth, args.min_depth, args.out_path, args.save, args.visualize)
    # model_path = "/home/keunmo/workspace/depth_to_3d/dataset2/replica_capture/apartment_1_slam1/sparse/hloc_sfm_loftr/sfm"
    # images_path = "/home/keunmo/workspace/depth_to_3d/dataset2/replica_capture/apartment_1_slam1"
    # pipeline(model_path, images_path, save=True, visualize=True)