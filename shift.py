from genericpath import exists
import os
from tqdm import tqdm
import cv2
import json
import numpy as np
import open3d as o3d

from logging import root
from promptda.promptda import PromptDA
from promptda.utils.io_wrapper import load_image, load_depth, save_depth

DEVICE = 'cuda'
model = PromptDA.from_pretrained("/iag_ad_01/ad/yuanweizhong/PromptDA/depth-anything/prompt-depth-anything-vitl/model.ckpt").to(DEVICE).eval()


def process_single_file(output_path, image_path, prompt_depth_path):
    """
    处理单个图像和深度文件。
    """
    image, scale = load_image(image_path)
    image = image.to(DEVICE)
    prompt_depth = load_depth(prompt_depth_path).to(DEVICE)  # 192x256, ARKit LiDAR depth in meters

    depth = model.predict(image, prompt_depth)  # HxW, depth in meters

    save_depth(depth, prompt_depth=prompt_depth, image=image)
    prompt_depth_resized = cv2.resize((prompt_depth[0][0] * 1000).cpu().numpy().astype(np.uint16), (image.shape[3], image.shape[2]))
    cv2.imwrite(os.path.join(output_path, f"{os.path.basename(image_path)}_prompt_depth_resized.png"), prompt_depth_resized)
    width = depth.shape[3]
    height = depth.shape[2]

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    cx, cy = 640,400
    fx, fy = 640,640
    x = (x - cx) / fx
    y = (y - cy) / fy
    z = depth.cpu().numpy()
    z_resized = prompt_depth_resized / 1000
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    points_resized = np.stack((np.multiply(x, z_resized), np.multiply(y, z_resized), z_resized), axis=-1).reshape(-1, 3)
    colors = image.cpu().numpy()[0].transpose(1, 2, 0).reshape(-1, 3)

    # Create the point cloud and save it to the output directory
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(output_path, f"{os.path.basename(image_path)}_test1.ply"), pcd)

    pcd_resized = o3d.geometry.PointCloud()
    pcd_resized.points = o3d.utility.Vector3dVector(points_resized)
    pcd_resized.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(output_path, f"{os.path.basename(image_path)}_test1_resized.ply"), pcd_resized)

def main(root_path,output_path):
    os.makedirs(output_path,exist_ok=True)

    image_files = sorted([os.path.join(root_path, f) for f in os.listdir(root_path) if f.endswith(('.jpg', '.jpeg'))])
    depth_files = sorted([os.path.join(root_path, f) for f in os.listdir(root_path) if f.endswith('.png')])

    for image_path, depth_path in tqdm(zip(image_files, depth_files)):
        process_single_file(output_path, image_path, depth_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量处理深度图并生成点云")
    parser.add_argument("--output_path", type=str, required=True, help="指定输出路径",default="./output")
    parser.add_argument("--root_path", type=str, required=True, help="指定根路径")
    args = parser.parse_args()

    main(args.root_path,args.output_path)