from promptda.promptda import PromptDA
from promptda.utils.io_wrapper import load_image, load_depth, save_depth

from PIL import Image
import numpy as np
import open3d as o3d
import os
import json
DEVICE = 'cuda'
import ptvsd
import cv2
import sys
import ipdb
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
def main():

    # image_path = "assets/yuanjin/40753679_6791.914.jpg"

    # prompt_depth_path = "assets/yuanjin/40753679_6791.914.png"
    image_path = "assets/gaosu/1736503211099999810.jpg"
    prompt_depth_path = "assets/gaosu/pts_depth.npz"
    intrinsic_path = 'assets/gaosu/center_camera_fov30-intrinsic.json'
    # resize_image(image_path)

    image,scale = load_image(image_path)
    image = image.to(DEVICE)
    prompt_depth = load_depth(prompt_depth_path).to(DEVICE) # 192x256, ARKit LiDAR depth in meters
    #resize prompt_depth 到 image 的尺寸


    model = PromptDA.from_pretrained("depth-anything/prompt-depth-anything-vitl/model.ckpt").to(DEVICE).eval()
    depth = model.predict(image, prompt_depth) # HxW, depth in meters

    save_depth(depth, prompt_depth=prompt_depth, image=image)
    prompt_depth_resized = cv2.resize((prompt_depth[0][0]*1000).cpu().numpy().astype(np.uint16), (image.shape[3], image.shape[2]))
    cv2.imwrite("results/prompt_depth_resized.png", prompt_depth_resized)
    width = depth.shape[3]
    height = depth.shape[2]
    # Generate mesh grid and calculate point cloud coordinates

    
    intrinsic = json.load(open(intrinsic_path))
    intrinsic_np = np.array(intrinsic['value0']['param']['cam_K_new']['data'])
    K=intrinsic_np
    h, w = int(intrinsic['value0']['param']['img_dist_h']), int(intrinsic['value0']['param']['img_dist_w'])

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    # cx, cy = K[0,2], K[1,2]
    # fx, fy = K[0,0], K[1,1]
    # scale_x = 0.8319984375
    # scale_y = 1.10933125
    # scale_x = fx/w 
    # scale_y = fy/h
    # # import ipdb;ipdb.set_trace()
    # focal_length_x=scale_x*width
    # focal_length_y=scale_y*height
    # x = (x - width / 2) / focal_length_x
    # y = (y - height / 2) / focal_length_y
    # import ipdb;ipdb.set_trace()
    K_new = K*scale
    cx, cy = K_new[0,2], K_new[1,2]
    fx, fy = K_new[0,0], K_new[1,1]
    x = (x - cx) / fx
    y = (y - cy) / fy
    # import ipdb;ipdb.set_trace()
    z = depth.cpu().numpy()
    z_resized = prompt_depth_resized/1000
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    points_resized = np.stack((np.multiply(x, z_resized), np.multiply(y, z_resized), z_resized), axis=-1).reshape(-1, 3)
    colors = image.cpu().numpy()[0].transpose(1,2,0).reshape(-1, 3) 

    # Create the point cloud and save it to the output directory
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join("./", "test1" + ".ply"), pcd)

    pcd_resized = o3d.geometry.PointCloud()
    pcd_resized.points = o3d.utility.Vector3dVector(points_resized)
    pcd_resized.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join("./", "test1_resized" + ".ply"), pcd_resized)


if __name__ == "__main__":
    try:
        main()
    except:
        type, value, traceback = sys.exc_info()
        ipdb.post_mortem(traceback)