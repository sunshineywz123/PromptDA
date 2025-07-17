#将tool文件夹与promptda件夹置于同一目录下
import os

root_path = "/iag_ad_01/ad/yuanweizhong/datasets/senseauto/2025_01_10_10_03_03_AutoCollect_pilotGtRawParser"

os.system(f"python ../tool/lidar2image.py --root_path {root_path}")
os.system(f"python ../tool/knn_insert.py --root_path {root_path}")
os.system(f"python ./PromptDA/test.py --output_path ./output --root_path {root_path}")