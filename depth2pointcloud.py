from promptda.utils.io_wrapper import load_depth

from PIL import Image
import numpy as np
import open3d as o3d
import os
import json
DEVICE = 'cuda'
import ptvsd
import cv2
if 0:
    ptvsd.enable_attach(address=('0.0.0.0', 5691))


# image_path = "assets/yuanjin/40753679_6791.914.jpg"

# prompt_depth_path = "assets/yuanjin/40753679_6791.914.png"
image_path = "assets/gaosu/1736503211099999810.jpg"
# prompt_depth_path = "assets/senseauto/pts_depth.png"
# output_path = "assets/senseauto/pts_depth.ply"

# prompt_depth_path = "/nas/users/yuanweizhong/PromptDA/results/global_depth_0.png"
# output_path = "results/vis_ply/vggt_global_depth.ply"

prompt_depth_path = "assets/gaosu/pts_depth.npz"
output_path = "results/vis_ply/lwlr_depth.ply"
intrinsic_path = 'assets/gaosu/center_camera_fov30-intrinsic.json'

# resize_image(image_path)

# image,scale = load_image(image_path)
# image = image.to(DEVICE)
image = cv2.imread(image_path)
prompt_depth = load_depth(prompt_depth_path).to(DEVICE) # 192x256, ARKit LiDAR depth in meters
#resize prompt_depth 到 image 的尺寸

width = image.shape[1]  
height = image.shape[0]
# Generate mesh grid and calculate point cloud coordinates

intrinsic = json.load(open(intrinsic_path))
intrinsic_np = np.array(intrinsic['value0']['param']['cam_K_new']['data'])
K=intrinsic_np
h, w = int(intrinsic['value0']['param']['img_dist_h']), int(intrinsic['value0']['param']['img_dist_w'])

x, y = np.meshgrid(np.arange(width), np.arange(height))
K_new = K
cx, cy = K_new[0,2], K_new[1,2]
fx, fy = K_new[0,0], K_new[1,1]
x = (x - cx) / fx
y = (y - cy) / fy
# import ipdb;ipdb.set_trace()
z = prompt_depth[0][0].cpu().numpy()
# import ipdb;ipdb.set_trace()
points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
# import ipdb;ipdb.set_trace()
colors = image[:,:,::-1].reshape(-1, 3)/255.

# Create the point cloud and save it to the output directory
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(output_path, pcd)
