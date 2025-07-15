import argparse
import json
import os

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

from promptda.promptda import PromptDA
from promptda.utils.io_wrapper import load_depth, load_image, save_depth

DEVICE = 'cuda'
import sys

import ipdb
import ptvsd

if 0:
    ptvsd.enable_attach(address=('0.0.0.0', 5691))

def resize_image(image_path, new_width=1920):
    """
    调整图像大小，宽度为 new_width，同时保持纵横比。

    Args:
        image_path: 图像文件的路径。
        new_width: 新的宽度（默认为 1920 像素）。
    """
    try:
        img = Image.open(image_path)
        width, height = img.size

        # 计算新的高度以保持纵横比
        new_height = int(height * (new_width / width))

        # 调整图像大小
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        # 保存调整大小后的图像
        resized_img.save(image_path)  # 覆盖原始文件
        print(f"已将 '{image_path}' 的大小调整为 {new_width}x{new_height}")

    except FileNotFoundError:
        print(f"找不到文件: {image_path}")
    except Exception as e:
        print(f"发生错误: {e}")

def process_single_file(output_path, image_path, prompt_depth_path, intrinsic_path):
    """
    处理单个图像和深度文件。
    """
    image, scale = load_image(image_path)
    image = image.to(DEVICE)
    prompt_depth = load_depth(prompt_depth_path).to(DEVICE)  # 192x256, ARKit LiDAR depth in meters

    model = PromptDA.from_pretrained("depth-anything/prompt-depth-anything-vitl/model.ckpt").to(DEVICE).eval()
    depth = model.predict(image, prompt_depth)  # HxW, depth in meters

    save_depth(depth, prompt_depth=prompt_depth, image=image)
    prompt_depth_resized = cv2.resize((prompt_depth[0][0] * 1000).cpu().numpy().astype(np.uint16), (image.shape[3], image.shape[2]))
    cv2.imwrite(os.path.join(output_path, f"{os.path.basename(image_path)}_prompt_depth_resized.png"), prompt_depth_resized)
    width = depth.shape[3]
    height = depth.shape[2]

    intrinsic = json.load(open(intrinsic_path))
    intrinsic_np = np.array(intrinsic['value0']['param']['cam_K_new']['data'])
    K = intrinsic_np
    h, w = int(intrinsic['value0']['param']['img_dist_h']), int(intrinsic['value0']['param']['img_dist_w'])

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    K_new = K * scale
    cx, cy = K_new[0, 2], K_new[1, 2]
    fx, fy = K_new[0, 0], K_new[1, 1]
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

def main(output_path, image_dir, prompt_depth_dir, intrinsic_path):
    """
    批量处理图像和深度文件。
    """
    # 获取所有图像文件和深度文件
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    depth_files = sorted([os.path.join(prompt_depth_dir, f) for f in os.listdir(prompt_depth_dir) if f.endswith('.npz')])

    if len(image_files) != len(depth_files):
        print("图像文件和深度文件数量不匹配，请检查输入目录！")
        return

    # 批量处理
    for image_path, depth_path in zip(image_files, depth_files):
        print(f"正在处理图像: {image_path} 和深度文件: {depth_path}")
        process_single_file(output_path, image_path, depth_path, intrinsic_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量处理深度图并生成点云")
    parser.add_argument("--output_path", type=str, required=True, help="指定输出路径")
    parser.add_argument("--image_dir", type=str, required=True, help="指定输入图像目录")
    parser.add_argument("--prompt_depth_dir", type=str, required=True, help="指定深度图目录")
    parser.add_argument("--intrinsic_path", type=str, required=True, help="指定相机内参路径")
    args = parser.parse_args()

    try:
        main(args.output_path, args.image_dir, args.prompt_depth_dir, args.intrinsic_path)
    except:
        type, value, traceback = sys.exc_info()
        ipdb.post_mortem(traceback)