# time python -m ptvsd --host 0.0.0.0 --port 5691 test.py
# image_path = "assets/gaosu/1736503211099999810.jpg"
# prompt_depth_path = "assets/gaosu/pts_depth.npz"
# intrinsic_path = 'assets/gaosu/center_camera_fov30-intrinsic.json'
# 设置为入参
# image_path,prompt_depth_path,intrinsic_path设置为2025_01_10_10_03_03_AutoCollect_pilotGtRawParser数据下的对应路径
time python test.py --output_path ./output --root_dir /iag_ad_01/ad/yuanweizhong/datasets/senseauto/2025_01_10_10_03_03_AutoCollect_pilotGtRawParser
#设置入参生成路径--output_path $save_path

# depth2pointcloud.py 同样需要改入参